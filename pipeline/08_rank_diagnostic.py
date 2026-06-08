"""Stage 08 — higher-rank erasure diagnostic.

Sweep INLP rank k (iteratively project out the top-k corpus-discriminative linear
directions) and watch cross-corpus mixing rise and CV corpus-recoverability fall. This
separates how much of the corpus separation is linear-rank-k (erasable by removing more
directions) from how much is irreducible (nonlinear / genuine structure we'd want to keep).

Run with DATASET=marvel_dc. Reads data/<ds>/{entities.parquet, embeddings.npz}.
"""

import numpy as np
import pandas as pd
from config import EMBEDDINGS_NPZ, ENTITIES_PARQUET
from integrations import inlp
from metrics import cross_corpus_mixing, fully_mixed_baseline, linear_recoverability

KS = [0, 1, 2, 4, 8, 16, 32, 64]


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    pos = {i: k for k, i in enumerate(list(data["id"]))}
    raw = data["emb"][[pos[i] for i in df["id"]]].astype(np.float64)
    corpus = df["corpus"].to_numpy()

    print(f"fully-mixed mixing ~ {fully_mixed_baseline(corpus):.3f}\n")
    print("rank k   mixing   cv_recoverability")
    print("-" * 36)
    for k in KS:
        emb = raw if k == 0 else inlp(raw, corpus, k)
        mix = cross_corpus_mixing(emb, corpus)
        acc, chance = linear_recoverability(emb, corpus)
        print("%-7d  %.3f    %.3f" % (k, mix, acc))
    print(f"\n(chance recoverability ~ {chance:.3f})")


if __name__ == "__main__":
    main()
