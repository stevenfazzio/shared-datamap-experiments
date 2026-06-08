"""Shared cross-corpus metrics, used by both the baseline eval (stage 05) and the
ablation (stage 06) so every method is scored with identical code.

  - cross_corpus_mixing: mean fraction of each point's k nearest neighbors that are
    cross-corpus. ~0.5 = fully mixed, -> 0 = two blobs.
  - linear_recoverability: 5-fold CV accuracy of predicting corpus from the embedding.
    ~1.0 = separable, -> chance = integrated. CAVEAT: with d >> n (1024 >> 102),
    in-sample linear separation is trivial, so kNN mixing and the map are the more
    trustworthy reads; recoverability is a soft corroborator.
  - cross_corpus_nn: each figure's nearest other-corpus figure, by cosine or CSLS.
    CSLS de-hubs the matching (penalizes attractors like Frigg/Thor).
  - aptness_hit_rate: fraction of KNOWN correspondences whose top-1 match is acceptable.
"""

import numpy as np
from config import KNOWN  # active dataset's analogue set (re-exported for callers)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def _unit(emb):
    emb = np.asarray(emb, dtype=np.float64)
    return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)


def cosine_sim(emb):
    """Cosine similarity matrix with the diagonal set to -inf (self excluded)."""
    u = _unit(emb)
    sim = u @ u.T
    np.fill_diagonal(sim, -np.inf)
    return sim


def cross_corpus_mixing(emb, corpus, k=10):
    sim = cosine_sim(emb)
    n = len(corpus)
    frac = [np.mean(corpus[np.argsort(-sim[i])[:k]] != corpus[i]) for i in range(n)]
    return float(np.mean(frac))


def fully_mixed_baseline(corpus):
    n = len(corpus)
    return float(np.mean([np.sum(corpus != corpus[i]) / (n - 1) for i in range(n)]))


def linear_recoverability(emb, corpus, cv=5):
    labels = sorted(set(corpus.tolist()))
    y = (corpus == labels[0]).astype(int)
    acc = float(cross_val_score(LogisticRegression(max_iter=2000), _unit(emb), y, cv=cv).mean())
    chance = float(max(y.mean(), 1 - y.mean()))
    return acc, chance


def _topk_mean(block, k):
    """Mean of the top-k values in each row of `block`."""
    k = min(k, block.shape[1])
    return np.sort(block, axis=1)[:, -k:].mean(axis=1)


def cross_corpus_nn(emb, corpus, names, method="cosine", csls_k=10):
    """Map each figure -> its nearest figure in the *other* corpus.

    cosine: plain nearest cosine neighbor. csls: 2*cos(i,j) - r_B(i) - r_A(j), where
    r_B(i) is i's mean similarity to its k nearest in the target corpus and r_A(j) is j's
    mean similarity to its k nearest in the source corpus — penalizing hub candidates.
    """
    u = _unit(emb)
    sim = u @ u.T
    labels = sorted(set(corpus.tolist()))
    out = {}
    for a in labels:
        ai = np.where(corpus == a)[0]
        for b in labels:
            if b == a:
                continue
            bj = np.where(corpus == b)[0]
            block = sim[np.ix_(ai, bj)]  # (|A|, |B|) cosine
            if method == "csls":
                r_b = _topk_mean(block, csls_k)  # per source a: mean sim to top-k in B
                r_a = _topk_mean(sim[np.ix_(bj, ai)], csls_k)  # per target b: mean sim to top-k in A
                score = 2 * block - r_b[:, None] - r_a[None, :]
            else:
                score = block
            nn = bj[np.argmax(score, axis=1)]
            for idx, qi in enumerate(ai):
                out[names[qi]] = names[nn[idx]]
    return out


def aptness_hit_rate(cross_nn, known=KNOWN):
    hits = checked = 0
    for g, acceptable in known.items():
        if g in cross_nn:
            checked += 1
            hits += cross_nn[g] in acceptable
    rate = hits / checked if checked else float("nan")
    return rate, hits, checked
