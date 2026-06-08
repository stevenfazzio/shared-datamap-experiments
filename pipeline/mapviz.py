"""datamapplot rendering, shared by the baseline (stage 04) and the ablation maps (07).

Color = corpus (red Greek / blue Norse) via marker_color_array, so corpus is the default
view regardless of the cluster colormap. offline_mode inlines fonts/JS so the HTML renders
without a network. Labels are passed finest-first.
"""

import datamapplot
import numpy as np
import pandas as pd
from config import CORPUS_COLOR_MAPPING

_HOVER_TEMPLATE = (
    "<div style='padding:4px 8px;max-width:240px'><b>{name}</b><br>"
    "<span style='color:#888'>{corpus} mythology</span></div>"
)


def render_map(coords, label_layers, names, corpus, urls, char_len, title, subtitle, out_html):
    marker_colors = np.array([CORPUS_COLOR_MAPPING[c] for c in corpus])
    cl = np.sqrt(np.clip(np.asarray(char_len, dtype=float), 1, None))
    marker_sizes = 4 + 11 * (cl - cl.min()) / (cl.max() - cl.min() + 1e-9)

    extra = pd.DataFrame({"name": names, "corpus": corpus, "url": urls})

    fig = datamapplot.create_interactive_plot(
        coords,
        *label_layers,
        hover_text=list(names),
        hover_text_html_template=_HOVER_TEMPLATE,
        marker_size_array=marker_sizes,
        extra_point_data=extra,
        on_click="window.open(`{url}`)",
        marker_color_array=marker_colors,
        title=title,
        sub_title=subtitle,
        enable_search=True,
        search_field="",
        darkmode=False,
        offline_mode=True,
    )
    fig.save(str(out_html))
