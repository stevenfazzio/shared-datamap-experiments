"""Toponymy region labeling, shared by the baseline (stage 03) and the ablation maps
(stage 07) so both label identically.

Toponymy names *regions of the map* (place-naming), not individual figures: the 2D coords
are the substrate (clusterable_vectors), the high-D embeddings carry semantic content
(embedding_vectors), and figures in unnamed space come back "Unlabelled" (a gap, signal).
Returns per-layer label vectors FINEST FIRST (the order DataMapPlot wants).

Opus 4.8 note: the stock AsyncAnthropicNamer passes `temperature`, which Opus 4.7/4.8
removed (400, fail-fast). OpusAnthropicNamer drops it.
"""

from __future__ import annotations

import nest_asyncio
import numpy as np
from config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MAX_CONCURRENCY,
    ANTHROPIC_MODEL_NAMING,
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    TOPONYMY_BASE_MIN_CLUSTER_SIZE,
    TOPONYMY_MIN_CLUSTERS,
    TOPONYMY_MIN_SAMPLES,
)
from toponymy.llm_wrappers import AsyncAnthropicNamer

nest_asyncio.apply()


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


def label_regions(documents, embeddings, coords, object_description, corpus_description, seed=42, log=True):
    """Fit Toponymy and return per-layer per-document label vectors, FINEST FIRST."""
    from toponymy import Toponymy, ToponymyClusterer
    from toponymy.embedding_wrappers import CohereEmbedder

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
        object_description=object_description,
        corpus_description=corpus_description,
        lowest_detail_level=0.5,
        highest_detail_level=1.0,
    )
    np.random.seed(seed)
    topic_model.fit(objects=documents, embedding_vectors=embeddings, clusterable_vectors=coords)

    layers = [np.asarray(v, dtype=object) for v in topic_model.topic_name_vectors_]  # finest first
    if log:
        for i, names in enumerate(layers):
            named = int(np.sum(names != "Unlabelled"))
            uniq = sorted({n for n in names.tolist() if n != "Unlabelled"})
            print(f"    layer {i}: {len(uniq)} regions, {named}/{len(names)} named -> {uniq[:8]}")
    return layers
