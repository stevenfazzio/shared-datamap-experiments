"""Stage 05 — Phase-1 go/no-go evaluation on the RAW embeddings (high-D, no integration).

Three readouts:
  [1] Cross-corpus nearest neighbors — the real signal. Does each Greek figure's nearest
      Norse figure (cosine, 1024D) make archetypal sense? Soft hit-rate vs a hand-set of
      known correspondences, plus the full Greek→Norse list to eyeball.
  [2] Cross-corpus kNN mixing rate — how separated the corpuses are. ~0.49 = fully mixed,
      → 0 = two blobs. Expected low at baseline; the Phase-2 anchor.
  [3] Linear corpus recoverability — 5-fold CV accuracy of predicting corpus from the
      embedding. ~1.0 = trivially separable (the linear direction LEACE would erase).

Reads:  data/entities.parquet, data/embeddings.npz
Output: data/eval_report.json  (+ printed report)
"""

import json

import numpy as np
import pandas as pd
from config import EMBEDDINGS_NPZ, ENTITIES_PARQUET, EVAL_REPORT_JSON
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Lenient semi-ground-truth: for each Greek figure, Norse figures any of which would be an
# apt top-1 cross-corpus analogue (comparative-mythology archetypes). Stretch entries
# included on purpose — a low hit-rate on the clear ones (sun/moon/sea/underworld/war/love)
# would be the real warning sign.
KNOWN = {
    "Zeus": {"Odin", "Thor"},
    "Hera": {"Frigg"},
    "Poseidon": {"Njörðr", "Ægir", "Rán"},
    "Hades": {"Hel"},
    "Aphrodite": {"Freyja", "Frigg"},
    "Ares": {"Týr", "Thor"},
    "Artemis": {"Skaði", "Ullr"},
    "Helios": {"Sól"},
    "Selene": {"Máni", "Sól"},
    "Nyx": {"Nótt"},
    "Eos": {"Dagr", "Sól"},
    "Hermes": {"Hermóðr", "Loki"},
    "Dionysus": {"Kvasir", "Bragi"},
    "Cronus": {"Ymir", "Surtr", "Borr"},
}


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    emb = data["emb"].astype(np.float64)

    # Align embeddings to entities row order by id.
    pos = {i: k for k, i in enumerate(list(data["id"]))}
    emb = emb[[pos[i] for i in df["id"]]]

    corpus = df["corpus"].to_numpy()
    names = df["name"].to_numpy()
    greek_mask = corpus == "Greek"
    norse_mask = corpus == "Norse"
    n = len(df)

    norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)

    def nearest_cross(i):
        cand = np.where(norse_mask if corpus[i] == "Greek" else greek_mask)[0]
        return cand[np.argmax(sim[i, cand])]

    cross_nn = {names[i]: names[nearest_cross(i)] for i in range(n)}

    # [1] aptness hit-rate on known correspondences
    hits = checked = 0
    for g, acceptable in KNOWN.items():
        if g in cross_nn:
            checked += 1
            hits += cross_nn[g] in acceptable
    hit_rate = hits / checked if checked else float("nan")

    # [2] cross-corpus kNN mixing (k=10)
    k = 10
    cross_frac = [np.mean(corpus[np.argsort(-sim[i])[:k]] != corpus[i]) for i in range(n)]
    mixing = float(np.mean(cross_frac))
    expected_mixed = float(np.mean([np.sum(corpus != corpus[i]) / (n - 1) for i in range(n)]))

    # [3] linear corpus recoverability (5-fold CV logistic regression)
    y = greek_mask.astype(int)
    acc = float(cross_val_score(LogisticRegression(max_iter=2000), norm, y, cv=5).mean())
    majority = float(max(y.mean(), 1 - y.mean()))

    print("\n========== PHASE-1 GO/NO-GO (raw baseline, high-D) ==========")
    print(f"\n[1] Cross-corpus aptness on {checked} known correspondences: {hits}/{checked} = {hit_rate:.0%}")
    for g, acceptable in KNOWN.items():
        if g in cross_nn:
            mark = "OK " if cross_nn[g] in acceptable else " . "
            print(f"   {mark} {g:11s} -> {cross_nn[g]:12s} (apt: {', '.join(sorted(acceptable))})")
    print("\n    All Greek -> nearest Norse:")
    for i in np.where(greek_mask)[0]:
        print(f"      {names[i]:12s} -> {names[nearest_cross(i)]}")

    print(f"\n[2] Cross-corpus kNN mixing (k={k}): {mixing:.3f}  (fully-mixed ~ {expected_mixed:.3f}, two-blobs -> 0)")
    print(f"[3] Linear corpus recoverability (5-fold CV acc): {acc:.3f}  (chance ~ {majority:.3f})")
    print("\n=============================================================")

    report = {
        "n": int(n),
        "n_greek": int(greek_mask.sum()),
        "n_norse": int(norse_mask.sum()),
        "aptness_hit_rate": hit_rate,
        "aptness_hits": int(hits),
        "aptness_checked": int(checked),
        "cross_corpus_mixing_k10": mixing,
        "mixing_if_fully_mixed": expected_mixed,
        "linear_recoverability_cv_acc": acc,
        "chance_acc": majority,
        "cross_nn_greek_to_norse": {names[i]: names[nearest_cross(i)] for i in np.where(greek_mask)[0]},
        "cross_nn_norse_to_greek": {names[i]: names[nearest_cross(i)] for i in np.where(norse_mask)[0]},
    }
    with open(EVAL_REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Wrote {EVAL_REPORT_JSON}")


if __name__ == "__main__":
    main()
