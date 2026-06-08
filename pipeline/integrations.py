"""Integration methods that transform the raw co-embedding to dissolve the corpus split.

Each takes raw embeddings (N x d) + corpus labels (N,) and returns transformed embeddings
of the same shape, to be fed to the SAME UMAP + metrics. CSLS is deliberately NOT here —
it's an eval-time nearest-neighbor correction (see metrics.cross_corpus_nn), not an
embedding transform.

  raw:     identity (the two-blob baseline)
  center:  subtract each corpus's mean (crude offset removal)
  leace:   rank-1 least-squares concept erasure of the corpus direction (binary, closed
           form). Computed in the covariance eigen row-space so it's well-defined when
           d >> n (here 1024 >> 102, so the ambient covariance is singular).
  harmony: Harmony batch-integration (iterative cluster-and-correct)
"""

import numpy as np

METHODS = ["raw", "center", "leace", "harmony"]


def integrate(method, emb, corpus):
    emb = np.asarray(emb, dtype=np.float64)
    corpus = np.asarray(corpus)
    if method == "raw":
        return emb
    if method == "center":
        return _center(emb, corpus)
    if method == "leace":
        return _leace_binary(emb, corpus)
    if method == "harmony":
        return _harmony(emb, corpus)
    raise ValueError(f"unknown integration method: {method}")


def _center(emb, corpus):
    out = emb.copy()
    for c in set(corpus.tolist()):
        m = corpus == c
        out[m] -= emb[m].mean(axis=0, keepdims=True)
    return out


def _leace_binary(emb, corpus, tol=1e-9):
    """LEACE for a binary concept: erase the whitened class-mean-difference direction.

    For binary z, the LEACE eraser removes the column space of W·Σ_XZ in whitened
    coordinates, which is exactly the whitened difference of class means. We compute the
    whitening in the eigen row-space of the covariance (eigenvalues > tol), so the singular
    d>>n case is handled and the zero-variance null space is left untouched.
    """
    labels = sorted(set(corpus.tolist()))
    z = (corpus == labels[0]).astype(np.float64)
    mu = emb.mean(axis=0, keepdims=True)
    Xc = emb - mu

    cov = (Xc.T @ Xc) / (len(emb) - 1)
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > tol * vals.max()
    V = vecs[:, keep]  # (d, k) row-space basis
    s = np.sqrt(vals[keep])  # (k,) singular values

    Xw = (Xc @ V) / s  # (n, k) whitened (isotropic within row-space)
    u = Xw[z == 1].mean(0) - Xw[z == 0].mean(0)  # ∝ W·Σ_XZ for binary z
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        return emb.copy()
    u = u / nu
    Xw_e = Xw - (Xw @ u)[:, None] * u[None, :]  # project the corpus direction out

    Xc_e = (Xw_e * s) @ V.T  # un-whiten back to ambient space
    return Xc_e + mu


def _harmony(emb, corpus):
    import harmonypy
    import pandas as pd

    emb = np.asarray(emb)
    meta = pd.DataFrame({"corpus": np.asarray(corpus)})
    ho = harmonypy.run_harmony(emb, meta, vars_use=["corpus"])
    z = np.asarray(ho.Z_corr)
    # harmonypy's Z_corr orientation varies by version; normalize to (n_points, d).
    return z if z.shape[0] == emb.shape[0] else z.T


def inlp(emb, corpus, k):
    """Iterative Nullspace Projection: project out the top-k corpus-discriminative linear
    directions (each step fits a logistic classifier and removes its weight normal). A
    higher-rank generalization of the rank-1 LEACE edit, used by the rank diagnostic to see
    how much of the corpus separation is linear-rank-k (erasable) vs irreducible."""
    from sklearn.linear_model import LogisticRegression

    emb = np.asarray(emb, dtype=np.float64)
    corpus = np.asarray(corpus)
    mu = emb.mean(0, keepdims=True)
    Xc = emb - mu
    y = (corpus == sorted(set(corpus.tolist()))[0]).astype(int)
    for _ in range(k):
        clf = LogisticRegression(max_iter=2000).fit(Xc, y)
        w = clf.coef_[0].astype(np.float64)
        n = np.linalg.norm(w)
        if n < 1e-12:
            break
        w /= n
        Xc = Xc - np.outer(Xc @ w, w)  # project out direction w
    return Xc + mu
