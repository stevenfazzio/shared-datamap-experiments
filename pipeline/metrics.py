"""Shared cross-corpus metrics, used by both the baseline eval (stage 05) and the
ablation (stage 06) so every method is scored with identical code.

  - cross_corpus_mixing: mean fraction of each point's k nearest neighbors that are
    cross-corpus. -> 0 = blobs; fully mixed -> (K-1)/K (0.5 for two corpuses, ~0.75 for four).
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
    corpus = np.asarray(corpus)
    acc = float(cross_val_score(LogisticRegression(max_iter=2000), _unit(emb), corpus, cv=cv).mean())
    _, counts = np.unique(corpus, return_counts=True)
    chance = float(counts.max() / counts.sum())  # majority-class frequency (1/K if balanced)
    return acc, chance


def _topk_mean(block, k):
    """Mean of the top-k values in each row of `block`."""
    k = min(k, block.shape[1])
    return np.sort(block, axis=1)[:, -k:].mean(axis=1)


def cross_corpus_nn(emb, corpus, names, method="cosine", csls_k=10):
    """Map each figure -> its single nearest figure in a DIFFERENT corpus.

    The candidate pool for a source figure is every figure not in its own corpus, so this
    generalizes to K > 2 (binary is the special case of one other corpus). cosine: plain
    nearest cosine neighbor. csls: 2*cos(i,j) - r_O(i) - r_S(j), where r_O(i) is i's mean
    similarity to its top-k in the combined OTHER-corpus pool and r_S(j) is candidate j's
    mean similarity to its top-k within i's own (Source) corpus — penalizing hub candidates.
    """
    u = _unit(emb)
    sim = u @ u.T
    corpus = np.asarray(corpus)
    out = {}
    for i in range(len(corpus)):
        other = np.where(corpus != corpus[i])[0]
        block = sim[i, other]  # (|other|,)
        if method == "csls":
            r_o = _topk_mean(block[None, :], csls_k)[0]  # i's mean sim to its top-k others
            same = np.where(corpus == corpus[i])[0]
            r_s = _topk_mean(sim[np.ix_(other, same)], csls_k)  # each cand's top-k into i's corpus
            score = 2 * block - r_o - r_s
        else:
            score = block
        out[names[i]] = names[other[np.argmax(score)]]
    return out


def aptness_hit_rate(cross_nn, known=KNOWN):
    hits = checked = 0
    for g, acceptable in known.items():
        if g in cross_nn:
            checked += 1
            hits += cross_nn[g] in acceptable
    rate = hits / checked if checked else float("nan")
    return rate, hits, checked


def cross_modal_retrieval(emb, modality, pair_id):
    """Paired cross-modal retrieval (the multimodal readout). For each point, restrict
    candidates to the OTHER modality and check whether its paired point (same pair_id) is
    the nearest; report recall@1 and MRR per direction. Once the modality gap is erased, an
    entity's two representations should retrieve each other."""
    u = _unit(emb)
    sim = u @ u.T
    modality = np.asarray(modality)
    pair_id = np.asarray(pair_id)
    mods = sorted(set(modality.tolist()))
    res = {}
    for src, tgt in [(mods[0], mods[1]), (mods[1], mods[0])]:
        si = np.where(modality == src)[0]
        ti = np.where(modality == tgt)[0]
        tgt_of = {pair_id[j]: j for j in ti}
        r1 = mrr = n = 0
        for i in si:
            j = tgt_of.get(pair_id[i])
            if j is None:
                continue
            n += 1
            rank = int(np.sum(sim[i, ti] > sim[i, j])) + 1  # rank of the true pair (1 = best)
            r1 += rank == 1
            mrr += 1.0 / rank
        res[f"{src}->{tgt}"] = {
            "recall@1": r1 / n if n else float("nan"),
            "mrr": mrr / n if n else float("nan"),
            "n": int(n),
        }
    return res
