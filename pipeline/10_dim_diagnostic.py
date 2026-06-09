"""Stage 10 — output-dimension diagnostic (does d<n make recoverability trustworthy?).

embed-v4.0's `output_dimension` is a renormalized prefix (Matryoshka), so truncating the
saved 1024-d embeddings to d' equals a native d'-dim embedding (verified cosine ~1.0). This
sweeps d' and reports mixing + recoverability per integration method — so for a dataset with
n in between (e.g. pantheons_mm: 312 points), we can read recoverability where d<n instead of
the unreliable d=1024 >> n regime. Mixing is the dim-robust check; recoverability is the
reading that becomes trustworthy once d<n.

The payoff (pantheons_mm): at d=256 (<312) raw recoverability stays ~0.99 — the modality
split is a GENUINE linear separability, not a d>>n fitting artifact — while linear erasure
drops it to chance yet mixing stays low (the nonlinear plateau holds at trustworthy d).

Reads:  data/<ds>/{entities.parquet, embeddings.npz}
Writes: data/<ds>/dim_diagnostic.{json,html}   (+ printed table)
Run with DATASET=pantheons_mm (n=312 > 256).
"""

import json
import os

import numpy as np
import pandas as pd
from config import DATA_DIR, EMBEDDINGS_NPZ, ENTITIES_PARQUET
from integrations import METHODS, integrate
from metrics import cross_corpus_mixing, fully_mixed_baseline, linear_recoverability

DIMS = [1024, 512, 256]
DIM_JSON = DATA_DIR / "dim_diagnostic.json"
DIM_HTML = DATA_DIR / "dim_diagnostic.html"
COLORS = {"raw": "#9a9a9a", "center": "#1f77b4", "leace": "#2ca02c", "harmony": "#d62728"}


def _write_chart(results):
    """Recoverability vs output dimension, one line per method. The point: raw stays high even
    where d<n (genuine linear separability), so a low recoverability after erasure is real, not
    a d>>n artifact. Needs plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [chart skipped] plotly not installed — `uv add plotly` to enable dim_diagnostic.html")
        return
    dims = [d["d"] for d in results["dims"]]
    xs = list(range(len(dims)))  # even spacing, large d -> small d left to right
    per_method = {m: [] for m in METHODS}
    for d in results["dims"]:
        for row in d["methods"]:
            per_method[row["method"]].append(row["recoverability"])
    fig = go.Figure()
    for m in METHODS:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=per_method[m],
                mode="lines+markers",
                name=m,
                line=dict(color=COLORS.get(m), width=3),
                marker=dict(size=9),
            )
        )
    fig.add_hline(
        y=results["chance"],
        line=dict(color="gray", dash="dot"),
        annotation_text=f"chance = {results['chance']:.2f}",
        annotation_position="bottom right",
    )
    # mark where d crosses below n (recoverability becomes trustworthy)
    below = [i for i, d in enumerate(dims) if d < results["n"]]
    if below:
        fig.add_vrect(
            x0=below[0] - 0.5,
            x1=xs[-1] + 0.4,
            fillcolor="LightGreen",
            opacity=0.15,
            line_width=0,
            annotation_text="d < n (recoverability trustworthy)",
            annotation_position="top left",
        )
    fig.update_layout(
        title=(
            f"Corpus recoverability vs embedding dimension — {results['dataset']}<br>"
            f"<sub>Matryoshka-truncated embeddings (n={results['n']}). raw staying high where d<n "
            f"⇒ the split is genuinely linearly separable; erasure dropping to chance there ⇒ a real "
            f"removal, not a d≫n fitting artifact.</sub>"
        ),
        xaxis=dict(
            title="output dimension (Matryoshka prefix)",
            tickvals=xs,
            ticktext=[str(d) for d in dims],
        ),
        yaxis=dict(title="corpus recoverability (CV accuracy)", range=[-0.02, 1.02]),
        legend=dict(title="integration method"),
        template="plotly_white",
        width=880,
        height=560,
    )
    tmp = str(DIM_HTML) + ".tmp"
    fig.write_html(tmp, include_plotlyjs="inline")
    os.replace(tmp, DIM_HTML)
    print(f"Wrote {DIM_HTML}")


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    pos = {i: k for k, i in enumerate(list(data["id"]))}
    raw = data["emb"][[pos[i] for i in df["id"]]].astype(np.float64)
    corpus = df["corpus"].to_numpy()
    n = len(df)

    chance = linear_recoverability(raw, corpus)[1]
    fully = fully_mixed_baseline(corpus)
    print(f"n={n} points, d_full={raw.shape[1]}")
    print(f"fully-mixed mixing ~ {fully:.3f}  |  chance recov ~ {chance:.3f}")

    dim_rows = []
    for d in DIMS:
        emb_d = raw[:, :d]  # Matryoshka prefix; metrics _unit re-normalizes
        flag = "d<n  (recov trustworthy)" if d < n else "d>=n (recov overfit)"
        print(f"\n--- d={d}   {flag} ---")
        print(f"  {'method':8s} {'mixing':>7s} {'recov':>7s}")
        methods = []
        for m in METHODS:
            e = integrate(m, emb_d, corpus)
            mix = cross_corpus_mixing(e, corpus)
            acc, _ = linear_recoverability(e, corpus)
            methods.append({"method": m, "mixing": mix, "recoverability": acc})
            print(f"  {m:8s} {mix:7.3f} {acc:7.3f}")
        dim_rows.append({"d": d, "d_lt_n": bool(d < n), "methods": methods})

    results = {
        "dataset": DATA_DIR.name,
        "n": n,
        "chance": chance,
        "fully_mixed": fully,
        "dims": dim_rows,
    }
    tmp = str(DIM_JSON) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, DIM_JSON)
    print(f"\nWrote {DIM_JSON}")
    _write_chart(results)


if __name__ == "__main__":
    main()
