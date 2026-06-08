"""Stage 06 — integration ablation (quantitative core; maps are stage 07).

For each integration method, transform the raw embeddings, re-run the SAME UMAP, and score
with the shared metrics. The "raw" row reproduces the Phase-1 baseline (a consistency
check against stage 05). Aptness is reported under both plain cosine NN and CSLS, since
CSLS is an orthogonal, eval-time fix for the hubness, not an embedding change.

Reads:  data/entities.parquet, data/embeddings.npz
Writes: data/experiments/<method>/{embeddings.npz, umap_coords.npz}, data/ablation.json
"""

import json

import numpy as np
import pandas as pd
import umap
from config import (
    DATA_DIR,
    EMBEDDINGS_NPZ,
    ENTITIES_PARQUET,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
)
from integrations import METHODS, integrate
from metrics import (
    aptness_hit_rate,
    cross_corpus_mixing,
    cross_corpus_nn,
    fully_mixed_baseline,
    linear_recoverability,
)

EXP_DIR = DATA_DIR / "experiments"


def run_umap(emb):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    return reducer.fit_transform(emb).astype(np.float32)


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    pos = {i: k for k, i in enumerate(list(data["id"]))}
    raw = data["emb"][[pos[i] for i in df["id"]]].astype(np.float64)
    corpus = df["corpus"].to_numpy()
    names = df["name"].to_numpy()
    ids = df["id"].to_numpy()

    mixed = fully_mixed_baseline(corpus)
    rows = []
    for method in METHODS:
        emb = integrate(method, raw, corpus)
        coords = run_umap(emb)

        mdir = EXP_DIR / method
        mdir.mkdir(parents=True, exist_ok=True)
        np.savez(mdir / "embeddings.npz", emb=emb.astype(np.float32), id=ids)
        np.savez(mdir / "umap_coords.npz", coords=coords, id=ids)

        mixing = cross_corpus_mixing(emb, corpus)
        acc, chance = linear_recoverability(emb, corpus)
        nn_cos = cross_corpus_nn(emb, corpus, names, "cosine")
        nn_csls = cross_corpus_nn(emb, corpus, names, "csls")
        apt_cos, h_cos, checked = aptness_hit_rate(nn_cos)
        apt_csls, h_csls, _ = aptness_hit_rate(nn_csls)
        rows.append(
            {
                "method": method,
                "mixing": mixing,
                "recoverability": acc,
                "chance": chance,
                "aptness_cosine": apt_cos,
                "aptness_csls": apt_csls,
                "hits_cosine": h_cos,
                "hits_csls": h_csls,
                "checked": checked,
                "nn_cosine": nn_cos,
                "nn_csls": nn_csls,
            }
        )
        print(f"  {method:8s} mixing={mixing:.3f} recov={acc:.3f} apt(cos)={apt_cos:.0%} apt(csls)={apt_csls:.0%}")

    print("\n================= INTEGRATION ABLATION =================")
    print(
        f"fully-mixed mixing ≈ {mixed:.3f}   chance recoverability ≈ {rows[0]['chance']:.3f}   "
        f"aptness over {rows[0]['checked']} known pairs\n"
    )
    hdr = f"{'method':9s} {'mixing':>8s} {'recover':>9s} {'apt:cos':>9s} {'apt:csls':>9s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['method']:9s} {r['mixing']:8.3f} {r['recoverability']:9.3f} "
            f"{r['aptness_cosine']:9.0%} {r['aptness_csls']:9.0%}"
        )
    print("=======================================================")

    out = DATA_DIR / "ablation.json"
    with open(out, "w") as f:
        json.dump({"fully_mixed": mixed, "methods": rows}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
