"""Figure: gold-category clustering crossover, across datasets. Reads each dataset's
gold_clustering.json (stage 12) and draws, per method, two grouped bars — ARI vs the gold
ARCHETYPE (what we want clusters to find, ↑) and ARI vs the CORPUS confounder (↓ after erasure)
— with the shuffle destruction control. The visual: raw clusters by corpus (gray tall, green
short); erasure flips it (gray ~0, green up); shuffle ~0 on both despite mixing the most.

Run from repo root: uv run python pipeline/_fig_gold_crossover.py
Writes: data/gold_clustering_crossover.html
"""

import json
import os
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

DATASETS = ["greek_norse", "pantheons", "marvel_dc"]
ROOT = Path("data")
OUT = ROOT / "gold_clustering_crossover.html"
ARCH_COLOR = "#2ca02c"  # alignment with gold archetype (want high)
CORP_COLOR = "#7f7f7f"  # alignment with corpus confounder (want low)


def main():
    present = [
        (ds, json.loads((ROOT / ds / "gold_clustering.json").read_text()))
        for ds in DATASETS
        if (ROOT / ds / "gold_clustering.json").exists()
    ]
    if not present:
        print("no gold_clustering.json found yet")
        return

    fig = make_subplots(
        rows=1,
        cols=len(present),
        shared_yaxes=True,
        subplot_titles=[f"{ds}  (n={d['n']}, k={d['k']})" for ds, d in present],
        horizontal_spacing=0.04,
    )
    for col, (ds, d) in enumerate(present, start=1):
        methods = [m["method"] for m in d["methods"]]
        arch = [m["ari_archetype"] for m in d["methods"]]
        archerr = [m.get("ari_archetype_std", 0) for m in d["methods"]]
        corp = [m["ari_pantheon"] for m in d["methods"]]
        first = col == 1
        fig.add_trace(
            go.Bar(
                x=methods, y=arch, name="ARI vs archetype  (want ↑)",
                error_y=dict(type="data", array=archerr, thickness=1),
                marker_color=ARCH_COLOR, legendgroup="arch", showlegend=first,
            ),
            row=1, col=col,
        )
        fig.add_trace(
            go.Bar(
                x=methods, y=corp, name="ARI vs corpus (confounder, want ↓)",
                marker_color=CORP_COLOR, legendgroup="corp", showlegend=first,
            ),
            row=1, col=col,
        )
    fig.add_hline(y=0, line=dict(color="black", width=0.8))
    fig.update_layout(
        barmode="group",
        template="plotly_white",
        title=(
            "Erasing the corpus reorganizes clusters from source to shared meaning<br>"
            "<sub>k-means on the (integrated) embeddings vs gold categories — Fan et al.'s yardstick. "
            "Green ↑ = clusters track the ARCHETYPE we want; grey ↓ = clusters track the CORPUS we erase. "
            "raw clusters by corpus; erasure flips it; <b>shuffle</b> (destroyed embeddings) mixes the most yet "
            "recovers nothing — so the gain is real structure, not scrambling. Error bars = ±1 std over seeds.</sub>"
        ),
        legend=dict(orientation="h", y=-0.16, x=0.5, xanchor="center"),
        width=420 * len(present),
        height=560,
        bargap=0.25,
    )
    fig.update_yaxes(title="Adjusted Rand Index", row=1, col=1, range=[-0.05, 0.85])
    tmp = str(OUT) + ".tmp"
    fig.write_html(tmp, include_plotlyjs="inline")
    os.replace(tmp, OUT)
    print(f"Wrote {OUT}  ({len(present)} datasets: {', '.join(ds for ds, _ in present)})")


if __name__ == "__main__":
    main()
