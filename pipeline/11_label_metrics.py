"""Stage 11 — Toponymy-label metrics for the integration ablation.

The maps (stage 07) qualitatively show that after corpus-erasure the named regions become
cross-corpus archetypes. This stage turns that into numbers, using Toponymy's own output —
the named-region hierarchy — as the instrument:

  #1 Per-region cross-corpus mixing + depth curve. For each NAMED region, the Gini-Simpson
     corpus impurity (metrics.per_region_mixing), size-weighted over named points → one
     mixing value per layer, on the SAME (K-1)/K scale as the kNN mixing in stage 06. The
     per-layer curve (finest..coarsest) tests the project's claim: coarse regions mix into
     shared archetypes, fine regions may re-separate into genuine sub-structure. Always read
     against COVERAGE (named fraction) — a method can fake high mixing by naming only its
     blended core and leaving pure fringes Unlabelled. Plus "receipts": the most cross-corpus
     named regions with their member breakdown.

  #3 Structural / coverage table + where the unnamed space lands. Unnamed points are unnamed
     *space*, not noise; per_point_cross_corpus asks whether they sit on the corpus seam
     (higher cross-corpus kNN fraction than named points).

  #2 If a high-D (EVoC) labels file is present (experiments/<method>/labels_highd.parquet),
     compute #1 on it too and report the 2D-vs-high-D delta ("does the UMAP map overstate
     the merge?"). Written by the EVoC labeling pass; absent → silently skipped.

Reads:  data/<dataset>/entities.parquet,
        data/<dataset>/experiments/<method>/{labels.parquet, embeddings.npz}
        (+ optional labels_highd.parquet)
Writes: data/<dataset>/label_metrics.json (atomic) + label_metrics.html (depth-curve chart)
"""

import json
import os

import numpy as np
import pandas as pd
from config import DATA_DIR, ENTITIES_PARQUET
from integrations import METHODS
from metrics import fully_mixed_baseline, per_point_cross_corpus, per_region_mixing

EXP_DIR = DATA_DIR / "experiments"
LABEL_METRICS_JSON = DATA_DIR / "label_metrics.json"
LABEL_METRICS_HTML = DATA_DIR / "label_metrics.html"
KNN_K = 10  # match metrics.cross_corpus_mixing
N_RECEIPTS = 5  # most-mixed named regions to surface per method


def _layer_cols(df):
    return sorted((c for c in df.columns if c.startswith("label_layer_")), key=lambda c: int(c.split("_")[-1]))


def _aligned_labels(path, ids):
    """Load a labels parquet → [(layer_name, region_labels)] aligned to `ids`, finest first.
    Region labels are str; missing/NaN → 'Unlabelled' (unnamed space)."""
    df = pd.read_parquet(path)
    pos = {i: k for k, i in enumerate(df["id"].tolist())}
    order = [pos[i] for i in ids]
    out = []
    for c in _layer_cols(df):
        vec = df[c].to_numpy(dtype=object)[order]
        vec = np.array(
            ["Unlabelled" if (v is None or (isinstance(v, float) and np.isnan(v))) else str(v) for v in vec],
            dtype=object,
        )
        out.append((c, vec))
    return out


def _substrate_metrics(corpus, layers):
    """Per-layer #1 + #3 for one (method, substrate). `layers` = [(name, region_labels)]."""
    rows = []
    for name, rl in layers:
        agg, per_region = per_region_mixing(corpus, rl)
        n_regions = len(per_region)
        imp = np.array([r["impurity"] for r in per_region.values()])
        n_mixed = int(np.sum(imp > 0)) if imp.size else 0
        rows.append(
            {
                "layer": name,
                "mixing": agg,  # size-weighted region impurity
                "coverage": float(np.mean(rl != "Unlabelled")),  # named fraction (1 - %Unlabelled)
                "n_regions": n_regions,
                "n_pure": n_regions - n_mixed,  # single-corpus named regions
                "n_mixed": n_mixed,  # >=2 corpuses present
                "frac_mixed": (n_mixed / n_regions) if n_regions else float("nan"),
                "per_region": per_region,
            }
        )
    return rows


def _summarize(rows):
    """Collapse a substrate's per-layer rows to a few comparable numbers (for the 2D/high-D
    delta and the JSON consumer)."""
    mix = [r["mixing"] for r in rows if not np.isnan(r["mixing"])]
    return {
        "n_layers": len(rows),
        "coarsest_mixing": rows[-1]["mixing"] if rows else float("nan"),
        "coarsest_coverage": rows[-1]["coverage"] if rows else float("nan"),
        "mean_mixing": float(np.mean(mix)) if mix else float("nan"),
    }


def _receipts(corpus, layers, names, top=N_RECEIPTS):
    """Most cross-corpus NAMED regions (member breakdown) at the coarsest layer with >=2
    named regions — the human-readable archetypes."""
    for name, rl in reversed(layers):  # coarsest first
        _, per_region = per_region_mixing(corpus, rl)
        if len(per_region) >= 2:
            ranked = sorted(per_region.items(), key=lambda kv: (-kv[1]["impurity"], -kv[1]["size"]))
            out = []
            for region, info in ranked[:top]:
                members = [str(names[j]) for j in np.where(rl == region)[0]]
                out.append(
                    {
                        "layer": name,
                        "region": region,
                        "impurity": round(info["impurity"], 3),
                        "counts": info["counts"],
                        "members": members[:10],
                    }
                )
            return out
    return []


def _where_unnamed(corpus, emb, layers):
    """#3 spatial: per layer, mean cross-corpus kNN fraction for UNNAMED vs NAMED points.
    unnamed > named ⇒ the unnamed space sits on the corpus seam (signal)."""
    ppc = per_point_cross_corpus(emb, corpus, KNN_K)  # depends only on emb + corpus
    rows = []
    for name, rl in layers:
        unnamed = rl == "Unlabelled"
        rows.append(
            {
                "layer": name,
                "unnamed_xcorpus": float(ppc[unnamed].mean()) if unnamed.any() else float("nan"),
                "named_xcorpus": float(ppc[~unnamed].mean()) if (~unnamed).any() else float("nan"),
                "n_unnamed": int(unnamed.sum()),
            }
        )
    return rows


def _load_emb(mdir, ids):
    d = np.load(mdir / "embeddings.npz", allow_pickle=True)
    pos = {i: k for k, i in enumerate(list(d["id"]))}
    return d["emb"][[pos[i] for i in ids]].astype(np.float64)


def _write_chart(results):
    """Depth-curve HTML — one clean line per method: per-region mixing from Toponymy's finest
    layer to its coarsest (2D substrate only; the high-D #2 numbers stay in the JSON). x is
    normalized fine->coarse so methods with different layer counts align by zoom. Needs plotly."""
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("  [chart skipped] plotly not installed — `uv add plotly` to enable label_metrics.html")
        return
    max_mix = results["max_mixing"]
    is_modality = set(results["corpuses"]) == {"image", "text"}
    group = "image–text" if is_modality else "cross-corpus"
    unit = "modality" if is_modality else "corpus"
    colors = {"raw": "#9a9a9a", "center": "#1f77b4", "leace": "#2ca02c", "harmony": "#d62728"}

    fig = go.Figure()
    for method, entry in results["methods"].items():
        rows = entry["substrates"]["2d"]
        n = len(rows)
        xs = [i / (n - 1) for i in range(n)] if n > 1 else [0.5]  # fine(0) -> coarse(1)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=[r["mixing"] for r in rows],
                mode="lines+markers",
                name=method,
                line=dict(color=colors.get(method), width=3),
                marker=dict(size=9),
                customdata=[r["coverage"] for r in rows],
                hovertemplate=f"{method}<br>mixing %{{y:.2f}} · coverage %{{customdata:.0%}}<extra></extra>",
            )
        )
    fig.add_hline(
        y=max_mix,
        line=dict(color="gray", dash="dot"),
        annotation_text=f"fully mixed = {max_mix:.2f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title=(
            f"{group.capitalize()} mixing of Toponymy regions — {results['dataset']}<br>"
            f"<sub>each line: how mixed the named regions are, finest layer (left) → coarsest "
            f"(right); higher = more mixed, {max_mix:.2f} = fully balanced</sub>"
        ),
        xaxis=dict(
            title="Toponymy region scale",
            tickvals=[0, 1],
            ticktext=["finest layer<br>(many small regions)", "coarsest layer<br>(few big regions)"],
            range=[-0.06, 1.06],
        ),
        yaxis=dict(
            title=f"{group} mixing within a region<br>(0 = single-{unit} · {max_mix:.2f} = balanced)",
            range=[-0.02, max_mix * 1.15],
        ),
        legend=dict(title="integration method"),
        template="plotly_white",
        width=880,
        height=560,
    )
    tmp = str(LABEL_METRICS_HTML) + ".tmp"
    fig.write_html(tmp, include_plotlyjs="inline")  # self-contained, like the maps
    os.replace(tmp, LABEL_METRICS_HTML)
    print(f"Wrote {LABEL_METRICS_HTML}")


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    ids = df["id"].tolist()
    corpus = df["corpus"].to_numpy()
    names = df["name"].to_numpy()
    corpuses = sorted(set(corpus.tolist()))
    K = len(corpuses)
    fully = fully_mixed_baseline(corpus)
    max_mix = (K - 1) / K

    print(
        f"{DATA_DIR.name}: n={len(df)}  corpuses={corpuses}  "
        f"fully-mixed kNN ref={fully:.3f}  region-mixing max (K-1)/K={max_mix:.3f}"
    )

    results = {
        "dataset": DATA_DIR.name,
        "n": len(df),
        "corpuses": corpuses,
        "fully_mixed": fully,
        "max_mixing": max_mix,
        "knn_k": KNN_K,
        "methods": {},
    }

    for method in METHODS:
        mdir = EXP_DIR / method
        lp = mdir / "labels.parquet"
        if not lp.exists():
            print(f"\n  [skip] {method}: no labels.parquet — run stage 07 first")
            continue
        layers2d = _aligned_labels(lp, ids)
        emb = _load_emb(mdir, ids)

        substrates = {"2d": _substrate_metrics(corpus, layers2d)}
        hp = mdir / "labels_highd.parquet"
        if hp.exists():
            substrates["highd"] = _substrate_metrics(corpus, _aligned_labels(hp, ids))

        entry = {
            "substrates": substrates,
            "summary": {s: _summarize(rows) for s, rows in substrates.items()},
            "receipts": _receipts(corpus, layers2d, names),
            "where_unnamed": _where_unnamed(corpus, emb, layers2d),
        }
        if "highd" in substrates:
            entry["summary"]["delta_coarsest"] = (
                entry["summary"]["2d"]["coarsest_mixing"] - entry["summary"]["highd"]["coarsest_mixing"]
            )
            entry["summary"]["delta_mean"] = (
                entry["summary"]["2d"]["mean_mixing"] - entry["summary"]["highd"]["mean_mixing"]
            )
        results["methods"][method] = entry

        print(f"\n  {method}:  (layer: mixing | coverage | regions[mixed])")
        for r in entry["substrates"]["2d"]:
            print(
                f"    {r['layer']:14s} {r['mixing']:.3f} | cover {r['coverage']:.2f} | {r['n_regions']}[{r['n_mixed']}]"
            )
        wu = entry["where_unnamed"][0]  # finest layer (most unnamed)
        if wu["n_unnamed"]:
            print(
                f"    seam check (finest): unnamed xcorpus {wu['unnamed_xcorpus']:.3f} "
                f"vs named {wu['named_xcorpus']:.3f}  (n_unnamed={wu['n_unnamed']})"
            )
        if "highd" in substrates:
            d = entry["summary"]["delta_coarsest"]
            print(
                f"    high-D lens: coarsest mixing 2d={entry['summary']['2d']['coarsest_mixing']:.3f} "
                f"vs highd={entry['summary']['highd']['coarsest_mixing']:.3f}  (Δ={d:+.3f})"
            )
        if entry["receipts"]:
            top = entry["receipts"][0]
            print(f'    top archetype: "{top["region"]}" {top["counts"]} (impurity {top["impurity"]})')

    if not results["methods"]:
        print("\nNo per-method labels found. Run stage 07 (now persists labels.parquet) first.")
        return

    tmp = str(LABEL_METRICS_JSON) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, LABEL_METRICS_JSON)
    print(f"\nWrote {LABEL_METRICS_JSON}")
    _write_chart(results)


if __name__ == "__main__":
    main()
