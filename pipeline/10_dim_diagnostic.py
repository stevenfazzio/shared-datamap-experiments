"""Stage 10 — output-dimension diagnostic (does d<n make recoverability trustworthy?).

embed-v4.0's `output_dimension` is a renormalized prefix (Matryoshka), so truncating the
saved 1024-d embeddings to d' equals a native d'-dim embedding (verified cosine ~1.0). This
sweeps d' and reports mixing + recoverability per integration method — so for a dataset with
n in between (e.g. pantheons_mm: 312 points), we can read recoverability where d<n instead of
the unreliable d=1024 >> n regime. Mixing is the dim-robust check; recoverability is the
reading that becomes trustworthy once d<n.

Run with DATASET=pantheons_mm (n=312 > 256). Reads data/<ds>/{entities.parquet, embeddings.npz}.
"""

import numpy as np
import pandas as pd
from config import EMBEDDINGS_NPZ, ENTITIES_PARQUET
from integrations import METHODS, integrate
from metrics import cross_corpus_mixing, fully_mixed_baseline, linear_recoverability

DIMS = [1024, 512, 256]


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    pos = {i: k for k, i in enumerate(list(data["id"]))}
    raw = data["emb"][[pos[i] for i in df["id"]]].astype(np.float64)
    corpus = df["corpus"].to_numpy()
    n = len(df)

    chance = linear_recoverability(raw, corpus)[1]
    print(f"n={n} points, d_full={raw.shape[1]}")
    print(f"fully-mixed mixing ~ {fully_mixed_baseline(corpus):.3f}  |  chance recov ~ {chance:.3f}")
    for d in DIMS:
        emb_d = raw[:, :d]  # Matryoshka prefix; metrics _unit re-normalizes
        flag = "d<n  (recov trustworthy)" if d < n else "d>=n (recov overfit)"
        print(f"\n--- d={d}   {flag} ---")
        print(f"  {'method':8s} {'mixing':>7s} {'recov':>7s}")
        for m in METHODS:
            e = integrate(m, emb_d, corpus)
            mix = cross_corpus_mixing(e, corpus)
            acc, _ = linear_recoverability(e, corpus)
            print(f"  {m:8s} {mix:7.3f} {acc:7.3f}")


if __name__ == "__main__":
    main()
