"""Stage 11b — label grounding / routing accuracy (#4): are the region NAMES trustworthy
enough to stand behind the receipts stage 11 quotes?

Toponymy's region names are LLM-generated; this checks they're grounded and discriminative,
independent of the integration method (the namer is fixed — so this CERTIFIES the receipts,
it does NOT rank methods). Two reads per (method, layer), all in the ORIGINAL Cohere semantic
space (the per-method experiments/*/embeddings.npz are centered/LEACE'd/harmonized, so a fresh
label embedding isn't comparable to them — but whether a NAME matches its MEMBERS' content is a
property of the raw space, using each method's labels only for membership):

  - routing_accuracy: embed each region NAME (Cohere, clustering, same space as the docs); for
    each NAMED document, is its own region's label its nearest label among the layer's siblings?
    High vs the 1/n_regions chance line = names specific enough to re-route docs to their region.
  - inter_label_cosine: mean pairwise cosine among sibling label embeddings. Low = distinct names.

Reads:  data/<dataset>/{entities.parquet, embeddings.npz}, experiments/<method>/labels.parquet
Writes: data/<dataset>/label_routing.json
"""

import json
import os

import cohere
import numpy as np
import pandas as pd
from config import (
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    COHERE_INPUT_TYPE,
    COHERE_OUTPUT_DIM,
    DATA_DIR,
    EMBED_BATCH,
    EMBEDDINGS_NPZ,
    ENTITIES_PARQUET,
)
from integrations import METHODS
from metrics import _unit

EXP_DIR = DATA_DIR / "experiments"
LABEL_ROUTING_JSON = DATA_DIR / "label_routing.json"


def _layer_cols(df):
    return sorted((c for c in df.columns if c.startswith("label_layer_")), key=lambda c: int(c.split("_")[-1]))


def _aligned_layer(ldf, col, ids):
    pos = {i: k for k, i in enumerate(ldf["id"].tolist())}
    vec = ldf[col].to_numpy(dtype=object)[[pos[i] for i in ids]]
    return np.array(
        ["Unlabelled" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in vec],
        dtype=object,
    )


def _embed(co, texts):
    """Cohere-embed label strings in the same (clustering) space as the docs, batched."""
    out = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = co.embed(
            texts=texts[i : i + EMBED_BATCH],
            model=COHERE_EMBED_MODEL,
            input_type=COHERE_INPUT_TYPE,
            embedding_types=["float"],
            output_dimension=COHERE_OUTPUT_DIM,
        )
        out.extend(resp.embeddings.float_)
    return _unit(np.asarray(out, dtype=np.float64))


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    ids = df["id"].tolist()
    ed = np.load(EMBEDDINGS_NPZ, allow_pickle=True)  # ORIGINAL Cohere embeddings (raw semantic space)
    epos = {i: k for k, i in enumerate(list(ed["id"]))}
    doc_emb = _unit(ed["emb"][[epos[i] for i in ids]].astype(np.float64))
    co = cohere.ClientV2(api_key=CO_API_KEY)

    print(f"{DATA_DIR.name}: routing in original {doc_emb.shape[1]}-d space")
    results = {"dataset": DATA_DIR.name, "input_type": COHERE_INPUT_TYPE, "methods": {}}

    for method in METHODS:
        lp = EXP_DIR / method / "labels.parquet"
        if not lp.exists():
            continue
        ldf = pd.read_parquet(lp)
        layers = [(c, _aligned_layer(ldf, c, ids)) for c in _layer_cols(ldf)]

        # dedup names across layers → one Cohere embed pass per method
        names = sorted({n for _, rl in layers for n in rl.tolist() if n != "Unlabelled"})
        if not names:
            continue
        name2vec = dict(zip(names, _embed(co, names)))

        rows = []
        for col, rl in layers:
            named = rl != "Unlabelled"
            regions = sorted(set(rl[named].tolist()))
            if len(regions) < 2:
                continue  # routing undefined with <2 labels
            lab_emb = np.stack([name2vec[r] for r in regions])
            ridx = {r: k for k, r in enumerate(regions)}
            pred = (doc_emb[named] @ lab_emb.T).argmax(axis=1)
            true = np.array([ridx[r] for r in rl[named]])
            ll = lab_emb @ lab_emb.T
            iu = np.triu_indices(len(regions), k=1)
            rows.append(
                {
                    "layer": col,
                    "n_regions": len(regions),
                    "n_named": int(named.sum()),
                    "routing_accuracy": float(np.mean(pred == true)),
                    "chance": 1.0 / len(regions),
                    "inter_label_cosine": float(ll[iu].mean()),
                }
            )
            r = rows[-1]
            print(
                f"  {method:8s} {col:14s} route={r['routing_accuracy']:.2f} (chance {r['chance']:.2f}) "
                f"redund={r['inter_label_cosine']:.2f}  R={len(regions)}"
            )
        results["methods"][method] = rows

    if not results["methods"]:
        print("No per-method labels found — run stage 07 first.")
        return
    tmp = str(LABEL_ROUTING_JSON) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, LABEL_ROUTING_JSON)
    print(f"Wrote {LABEL_ROUTING_JSON}")


if __name__ == "__main__":
    main()
