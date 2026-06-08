"""Stage 04 — interactive datamapplot HTML, colored by CORPUS (the one required overlay).

Color = corpus (red Greek / blue Norse) so the map reads like the owner-type panel in the
GitHub map: at this RAW baseline, expect two blobs. Toponymy region names render as text
labels (finest-first, the order DataMapPlot wants). Hover shows the figure; click opens its
Wikipedia page.

Reads:  data/entities.parquet, data/umap_coords.npz, data/labels.parquet
Output: data/map.html
"""

import datamapplot
import numpy as np
import pandas as pd
from config import (
    CORPUS_COLOR_MAPPING,
    ENTITIES_PARQUET,
    LABELS_PARQUET,
    MAP_HTML,
    UMAP_COORDS_NPZ,
)


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    cd = np.load(UMAP_COORDS_NPZ, allow_pickle=True)
    labels = pd.read_parquet(LABELS_PARQUET)

    # Align coords + labels to entities row order by id.
    cpos = {i: k for k, i in enumerate(list(cd["id"]))}
    coords = cd["coords"][[cpos[i] for i in df["id"]]].astype(np.float32)
    labels = labels.set_index("id").reindex(df["id"]).reset_index()

    label_cols = sorted(c for c in labels.columns if c.startswith("label_layer_"))  # finest first
    label_layers = [labels[c].fillna("Unlabelled").astype(str).values for c in label_cols]

    corpus = df["corpus"].astype(str).values
    names = df["name"].astype(str).values

    # Marker size ~ article length (a rough prominence proxy), gently scaled.
    cl = np.sqrt(df["char_len"].clip(lower=1).to_numpy(dtype=float))
    marker_sizes = 4 + 11 * (cl - cl.min()) / (cl.max() - cl.min() + 1e-9)

    marker_colors = np.array([CORPUS_COLOR_MAPPING[c] for c in corpus])

    extra = pd.DataFrame({"name": names, "corpus": corpus, "url": df["url"].astype(str).values})
    hover_template = (
        "<div style='padding:4px 8px;max-width:240px'>"
        "<b>{name}</b><br><span style='color:#888'>{corpus} mythology</span></div>"
    )

    fig = datamapplot.create_interactive_plot(
        coords,
        *label_layers,
        hover_text=names.tolist(),
        hover_text_html_template=hover_template,
        marker_size_array=marker_sizes,
        extra_point_data=extra,
        on_click="window.open(`{url}`)",
        marker_color_array=marker_colors,
        title="Greek × Norse — RAW baseline",
        sub_title="Wikipedia leads, co-embedded with no integration (red = Greek, blue = Norse)",
        enable_search=True,
        search_field="",
        darkmode=False,
        # Inline fonts/JS/CSS so the HTML renders without internet (the browser sandbox
        # can't reach Google Fonts / CDNs; generation happens online).
        offline_mode=True,
    )
    fig.save(str(MAP_HTML))
    print(f"Saved interactive map to {MAP_HTML}")


if __name__ == "__main__":
    main()
