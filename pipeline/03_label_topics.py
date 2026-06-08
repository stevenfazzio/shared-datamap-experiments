"""Stage 03 — hierarchical region labels via Toponymy + Claude (Opus 4.8).

Toponymy names *regions of the map* (place-naming), not individual figures. The 2D UMAP
coords are the substrate the named regions sit on (clusterable_vectors); the 1024D
embeddings carry the semantic content used while clustering (embedding_vectors). Layers
are stored FINEST FIRST (label_layer_0 = finest) — DataMapPlot wants finest first. Figures
in unnamed space come back "Unlabelled"; at ~100 points that fraction is high by design
(sparse density) and is a gap on the map (signal), not a failure.

At this RAW baseline the corpuses are two blobs, so expect coarse regions to be largely
corpus-pure (a Greek region, a Norse region) — the "before" picture for the Phase-2 ablation.

Opus 4.8 note: the stock AsyncAnthropicNamer passes `temperature`, which Opus 4.7/4.8
removed (400, fail-fast). OpusAnthropicNamer below drops it.

Reads:  data/entities.parquet, data/embeddings.npz, data/umap_coords.npz
Output: data/labels.parquet  (id + label_layer_0..k, finest first)
"""

from __future__ import annotations

import os

import nest_asyncio
import numpy as np
import pandas as pd
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MAX_CONCURRENCY,
    ANTHROPIC_MODEL_NAMING,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    EMBEDDINGS_NPZ,
    ENTITIES_PARQUET,
    LABELS_PARQUET,
    TOPONYMY_BASE_MIN_CLUSTER_SIZE,
    TOPONYMY_MIN_CLUSTERS,
    TOPONYMY_MIN_SAMPLES,
    UMAP_COORDS_NPZ,
)
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()

MAX_DOC_CHARS = 2_000


class OpusAnthropicNamer(AsyncAnthropicNamer):
    """AsyncAnthropicNamer for the Opus 4.8 call surface: `temperature` is removed on
    Opus 4.7/4.8 (the stock namer passes it and would 400 fail-fast)."""

    async def _call_single_llm(self, prompt, temperature, max_tokens):
        async with self.semaphore:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt + self.extra_prompting}],
            )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    async def _call_single_llm_with_system(self, system_prompt, user_prompt, temperature, max_tokens):
        async with self.semaphore:
            resp = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt + self.extra_prompting}],
            )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def main():
    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.embedding_wrappers import CohereEmbedder

    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    ed = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    cd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)

    # Align embeddings + coords to entities row order by id.
    epos = {i: k for k, i in enumerate(list(ed["id"]))}
    cpos = {i: k for k, i in enumerate(list(cd["id"]))}
    embeddings = ed["emb"][[epos[i] for i in df["id"]]].astype(np.float32)
    coords = cd["coords"][[cpos[i] for i in df["id"]]].astype(np.float32)

    documents = df["text"].str.slice(0, MAX_DOC_CHARS).tolist()
    print(f"Loaded {len(documents)} figures; embeddings {embeddings.shape}")

    llm = OpusAnthropicNamer(
        api_key=ANTHROPIC_API_KEY,
        model=ANTHROPIC_MODEL_NAMING,
        max_concurrent_requests=ANTHROPIC_MAX_CONCURRENCY,
    )
    embedder = CohereEmbedder(api_key=CO_API_KEY, model=COHERE_EMBED_MODEL)
    clusterer = ToponymyClusterer(
        min_clusters=TOPONYMY_MIN_CLUSTERS,
        base_min_cluster_size=TOPONYMY_BASE_MIN_CLUSTER_SIZE,
        min_samples=TOPONYMY_MIN_SAMPLES,
    )

    topic_model = Toponymy(
        llm_wrapper=llm,
        text_embedding_model=embedder,
        clusterer=clusterer,
        object_description="Greek and Norse mythological figures",
        corpus_description=(
            "figures from Greek and Norse mythology (gods, Titans, jötnar, and "
            "personifications), each described by its Wikipedia lead"
        ),
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    np.random.seed(42)
    topic_model.fit(objects=documents, embedding_vectors=embeddings, clusterable_vectors=coords)

    n_layers = len(topic_model.topic_name_vectors_)
    if n_layers == 0:
        raise ValueError("Toponymy produced 0 cluster layers")
    print(f"Toponymy produced {n_layers} layer(s)")

    out = {"id": df["id"].to_numpy()}
    for i, vec in enumerate(topic_model.topic_name_vectors_):  # finest first
        names = np.asarray(vec, dtype=object)
        out[f"label_layer_{i}"] = names
        named = int(np.sum(names != "Unlabelled"))
        uniq = sorted({n for n in names.tolist() if n != "Unlabelled"})
        print(f"  layer {i}: {len(uniq)} regions, {named}/{len(names)} figures named")
        if uniq:
            print(f"    regions: {uniq[:12]}")

    out_df = pd.DataFrame(out)
    tmp = str(LABELS_PARQUET) + ".tmp"
    out_df.to_parquet(tmp, index=False)
    os.replace(tmp, LABELS_PARQUET)
    print(f"Wrote {LABELS_PARQUET} ({n_layers} layers)")


if __name__ == "__main__":
    main()
