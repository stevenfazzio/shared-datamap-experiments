"""Integration methods that transform the raw co-embedding to dissolve the corpus split.

Each takes raw embeddings (N x d) + corpus labels (N,) and returns transformed embeddings
of the same shape, to be fed to the SAME UMAP + metrics. CSLS is deliberately NOT here —
it's an eval-time nearest-neighbor correction (see metrics.cross_corpus_nn), not an
embedding transform.

  raw:     identity (the two-blob baseline)
  center:  subtract each corpus's mean (crude offset removal)
  leace:   least-squares concept erasure of the corpus signal for K>=2 classes. Erases
           the span of the whitened class-mean deviations (dim <= K-1; rank-1 for K=2),
           computed in the covariance eigen row-space so it's well-defined when d >> n.
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
        return _leace(emb, corpus)
    if method == "harmony":
        return _harmony(emb, corpus)
    raise ValueError(f"unknown integration method: {method}")


def _center(emb, corpus):
    out = emb.copy()
    for c in set(corpus.tolist()):
        m = corpus == c
        out[m] -= emb[m].mean(axis=0, keepdims=True)
    return out


def _leace(emb, corpus, tol=1e-9):
    """LEACE concept erasure of the corpus signal, for K >= 2 classes.

    LEACE removes (in whitened coordinates) the column space of W·Σ_XZ where Z is the
    one-hot class indicator. For one-hot Z that column space is exactly the span of the
    whitened class-mean deviations {μ_c − μ̄}, of dimension <= K−1 (they sum to zero).
    Projecting that whole subspace out kills every linear corpus direction at once; for
    K=2 it reduces to the rank-1 difference-of-means edit. Whitening is done in the
    covariance eigen row-space (eigenvalues > tol) so the singular d >> n case is
    well-defined and the zero-variance null space is left untouched.
    """
    labels = sorted(set(corpus.tolist()))
    mu = emb.mean(axis=0, keepdims=True)
    Xc = emb - mu

    cov = (Xc.T @ Xc) / (len(emb) - 1)
    vals, vecs = np.linalg.eigh(cov)
    keep = vals > tol * vals.max()
    V = vecs[:, keep]  # (d, k) row-space basis
    s = np.sqrt(vals[keep])  # (k,) singular values
    Xw = (Xc @ V) / s  # (n, k) whitened (isotropic within row-space)

    # Whitened class-mean deviations; their row space IS the corpus subspace (dim <= K-1).
    M = np.stack([Xw[corpus == c].mean(0) for c in labels])  # (K, k)
    M = M - M.mean(0, keepdims=True)
    _, sv, Wt = np.linalg.svd(M, full_matrices=False)  # rows of Wt span M's row space
    r = int((sv > tol * (sv.max() + 1e-300)).sum())
    if r == 0:
        return emb.copy()
    B = Wt[:r]  # (r, k) orthonormal basis of the corpus subspace in whitened coords
    Xw_e = Xw - (Xw @ B.T) @ B  # project the whole corpus subspace out

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
    directions. Generalized to K >= 2 classes — each step fits a multinomial logistic
    classifier and removes the single most discriminative direction (the leading right
    singular vector of its K×d coefficient matrix). A higher-rank generalization of the
    LEACE edit, used by the rank diagnostic to see how much of the corpus separation is
    linear-rank-k (erasable, expected <= K−1) vs irreducible (nonlinear / genuine)."""
    from sklearn.linear_model import LogisticRegression

    emb = np.asarray(emb, dtype=np.float64)
    corpus = np.asarray(corpus)
    mu = emb.mean(0, keepdims=True)
    Xc = emb - mu
    for _ in range(k):
        clf = LogisticRegression(max_iter=2000).fit(Xc, corpus)
        C = np.atleast_2d(clf.coef_).astype(np.float64)  # (K or 1, d)
        _, sv, Wt = np.linalg.svd(C, full_matrices=False)
        if sv[0] < 1e-12:
            break
        w = Wt[0] / (np.linalg.norm(Wt[0]) + 1e-12)
        Xc = Xc - np.outer(Xc @ w, w)  # project out the top discriminative direction
    return Xc + mu
