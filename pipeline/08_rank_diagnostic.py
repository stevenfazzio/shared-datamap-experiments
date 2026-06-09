"""Stage 08 — higher-rank erasure diagnostic.

Sweep INLP rank k (iteratively project out the top-k corpus-discriminative linear
directions) and watch cross-corpus mixing rise and CV corpus-recoverability fall. This
separates how much of the corpus separation is linear-rank-k (erasable by removing more
directions) from how much is irreducible (nonlinear / genuine structure we'd want to keep).

Two regimes show up in the chart:
  - text corpus identity (e.g. pantheons, K=4): recoverability collapses by k=K-1 AND mixing
    rises toward fully-mixed — the corpus signal is a linear rank-(K-1) offset, erasing it
    interleaves the map.
  - modality (pantheons_mm, K=2): recoverability still collapses, but mixing STAYS LOW (the
    cones don't interleave) — the gap is nonlinear, so no amount of linear erasure merges it.

Reads:  data/<ds>/{entities.parquet, embeddings.npz}
Writes: data/<ds>/rank_diagnostic.{json,html}   (+ printed table)
Run with e.g. DATASET=pantheons.
"""

import json
import os

import numpy as np
import pandas as pd
from config import DATA_DIR, EMBEDDINGS_NPZ, ENTITIES_PARQUET
from integrations import inlp
from metrics import cross_corpus_mixing, fully_mixed_baseline, linear_recoverability

KS = [0, 1, 2, 3, 4, 8, 16, 32, 64]  # K-1 (=3 for 4 corpuses) is the expected linear rank
RANK_JSON = DATA_DIR / "rank_diagnostic.json"
RANK_HTML = DATA_DIR / "rank_diagnostic.html"


def _write_chart(results):
    """Two lines over INLP rank k: recoverability (falls) and mixing (rises iff the signal is
    linear). x is spaced by index so the uneven k grid (…,8,16,32,64) stays legible. Needs plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [chart skipped] plotly not installed — `uv add plotly` to enable rank_diagnostic.html")
        return
    ks = [r["k"] for r in results["ks"]]
    xs = list(range(len(ks)))  # even spacing; tick labels carry the real k
    rank = results["expected_rank"]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[r["recoverability"] for r in results["ks"]],
            mode="lines+markers",
            name="corpus recoverability (CV acc)",
            line=dict(color="#d62728", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=[r["mixing"] for r in results["ks"]],
            mode="lines+markers",
            name="cross-corpus mixing",
            line=dict(color="#1f77b4", width=3),
            marker=dict(size=8),
        )
    )
    fig.add_hline(
        y=results["chance"],
        line=dict(color="#d62728", dash="dot"),
        annotation_text="chance recoverability",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=results["fully_mixed"],
        line=dict(color="#1f77b4", dash="dot"),
        annotation_text="fully mixed",
        annotation_position="top right",
    )
    if rank in ks:
        xi = ks.index(rank)
        fig.add_vline(
            x=xi,
            line=dict(color="gray", dash="dash"),
            annotation_text=f"k = K−1 = {rank}",
            annotation_position="top",
        )
    fig.update_layout(
        title=(
            f"Linear rank of the corpus signal — {results['dataset']}<br>"
            f"<sub>INLP removes the top-k corpus-discriminative linear directions. Recoverability "
            f"collapsing by k=K−1={rank} ⇒ a linear rank-(K−1) signal; mixing rising with it ⇒ that "
            f"erasure interleaves the map (flat mixing ⇒ a nonlinear gap linear erasure can't merge).</sub>"
        ),
        xaxis=dict(
            title="linear directions removed (INLP rank k)",
            tickvals=xs,
            ticktext=[str(k) for k in ks],
        ),
        yaxis=dict(title="value (CV accuracy / mixing fraction)", range=[-0.02, 1.02]),
        legend=dict(title=None, x=0.5, y=0.5),
        template="plotly_white",
        width=880,
        height=560,
    )
    tmp = str(RANK_HTML) + ".tmp"
    fig.write_html(tmp, include_plotlyjs="inline")
    os.replace(tmp, RANK_HTML)
    print(f"Wrote {RANK_HTML}")


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    pos = {i: k for k, i in enumerate(list(data["id"]))}
    raw = data["emb"][[pos[i] for i in df["id"]]].astype(np.float64)
    corpus = df["corpus"].to_numpy()
    K = len(set(corpus.tolist()))
    fully = fully_mixed_baseline(corpus)

    print(f"{DATA_DIR.name}: n={len(df)}  K={K}  expected linear rank K-1={K - 1}")
    print(f"fully-mixed mixing ~ {fully:.3f}\n")
    print("rank k   mixing   cv_recoverability")
    print("-" * 36)
    rows = []
    chance = None
    for k in KS:
        emb = raw if k == 0 else inlp(raw, corpus, k)
        mix = cross_corpus_mixing(emb, corpus)
        acc, chance = linear_recoverability(emb, corpus)
        rows.append({"k": k, "mixing": mix, "recoverability": acc})
        print("%-7d  %.3f    %.3f" % (k, mix, acc))
    print(f"\n(chance recoverability ~ {chance:.3f})")

    results = {
        "dataset": DATA_DIR.name,
        "n": len(df),
        "K": K,
        "expected_rank": K - 1,
        "fully_mixed": fully,
        "chance": chance,
        "ks": rows,
    }
    tmp = str(RANK_JSON) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, RANK_JSON)
    print(f"Wrote {RANK_JSON}")
    _write_chart(results)


if __name__ == "__main__":
    main()
