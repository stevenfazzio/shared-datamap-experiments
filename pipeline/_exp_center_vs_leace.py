"""SCRATCH dig-in (delete if not promoted): WHY does plain per-corpus centering beat LEACE on
archetype recovery (stage 12 gold-clustering), and is LEACE actually best in the regime it was
designed for (d<n)? Isolates the two ways LEACE differs from centering:

  center      : subtract each corpus mean (pure translation; within-group spread along the
                pantheon-mean directions is PRESERVED, only the offset is removed)
  ortho_proj  : orthogonally project out the rank-(K-1) class-mean subspace, NO whitening
                (isolates PROJECTION-vs-translation: this flattens that subspace for everyone)
  leace       : whitened projection = ortho_proj + whitening (isolates WHITENING)

and sweeps the Matryoshka output dim across the d>>n -> d<n boundary (n=198, so d<=128 is d<n).
Archetype-ARI is mean±std over seeds (k-means is noisy at these modest ARIs); a second table
reports pantheon recoverability so we can confirm every method actually erases the confounder
(an ARI gap only means something if guardedness is equal).

Run: DATASET=pantheons uv run python pipeline/_exp_center_vs_leace.py
"""

import numpy as np
import pandas as pd
from config import DATA_DIR, EMBEDDINGS_NPZ, ENTITIES_PARQUET
from integrations import _center, _leace
from metrics import _unit, linear_recoverability
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

DIMS = [1024, 256, 128, 64]
SEEDS = list(range(10))


def _ortho_proj(emb, corpus):
    """Remove the rank-(K-1) class-mean subspace by ORTHOGONAL projection (no whitening)."""
    labels = sorted(set(corpus.tolist()))
    mu = emb.mean(0, keepdims=True)
    Xc = emb - mu
    M = np.stack([Xc[corpus == c].mean(0) for c in labels])  # (K, d)
    M = M - M.mean(0, keepdims=True)
    _, sv, Wt = np.linalg.svd(M, full_matrices=False)
    r = int((sv > 1e-9 * (sv.max() + 1e-300)).sum())
    B = Wt[:r]  # (r, d) orthonormal basis of the class-mean subspace
    return (Xc - (Xc @ B.T) @ B) + mu


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    ids = df["id"].tolist()
    pantheon = df["corpus"].to_numpy()
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    pos = {i: k for k, i in enumerate(list(data["id"]))}
    raw_full = data["emb"][[pos[i] for i in ids]].astype(np.float64)

    gold_map = pd.read_parquet(DATA_DIR / "gold_archetypes.parquet").set_index("id")["archetype"]
    gold = df["id"].map(gold_map).to_numpy()
    real = gold != "Other"
    k = len(set(gold[real].tolist()))
    n = len(df)

    methods = {"raw": lambda e, c: e, "center": _center, "ortho_proj": _ortho_proj, "leace": _leace}

    def ari(emb):
        accs = [
            adjusted_rand_score(gold[real], KMeans(n_clusters=k, random_state=s, n_init=5).fit(_unit(emb)).labels_[real])
            for s in SEEDS
        ]
        return np.mean(accs), np.std(accs)

    print(f"pantheons  n={n}  k={k}  (d<n when d<={n - 1}; n=198)")
    print(f"\n=== archetype-ARI  (mean±std over {len(SEEDS)} seeds; higher=better recovery of shared archetypes) ===")
    print(f"  {'dim':>5s} | " + "".join(f"{m:>15s}" for m in methods))
    ari_tab = {}
    for d in DIMS:
        ed = _unit(raw_full[:, :d])  # native d-dim Matryoshka prefix (renormalized)
        row = {m: ari(fn(ed, pantheon)) for m, fn in methods.items()}
        ari_tab[d] = row
        flag = "d<n" if d < n else "d>n"
        print(f"  {d:>5d} | " + "".join(f"{mu:>8.3f}±{sd:<5.2f}" for mu, sd in row.values()) + f"  [{flag}]")

    print("\n=== pantheon recoverability (CV acc; ~chance 0.26 = confounder erased) ===")
    print(f"  {'dim':>5s} | " + "".join(f"{m:>15s}" for m in methods))
    for d in DIMS:
        ed = _unit(raw_full[:, :d])
        print(f"  {d:>5d} | " + "".join(f"{linear_recoverability(fn(ed, pantheon), pantheon)[0]:>15.3f}" for fn in methods.values()))


if __name__ == "__main__":
    main()
