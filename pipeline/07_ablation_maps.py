"""Stage 07 — render a datamapplot per integration method (the visual ablation).

For each integrated embedding from stage 06, label regions with Toponymy and render the
map (color = corpus). The questions: does the map interleave (vs the raw two-blob
baseline), and do the region names become CROSS-CORPUS archetypes (vs corpus-pure)?

Reads:  data/<dataset>/experiments/<method>/{embeddings.npz, umap_coords.npz}, entities.parquet
Writes: data/<dataset>/experiments/<method>/map.html
"""

import numpy as np
import pandas as pd
from config import (
    CORPUS_DESCRIPTION,
    DATA_DIR,
    DATASET_TITLE,
    ENTITIES_PARQUET,
    OBJECT_DESCRIPTION,
)
from mapviz import render_map
from topic_labeling import label_regions

EXP_DIR = DATA_DIR / "experiments"
METHODS_TO_MAP = ["raw", "center", "leace", "harmony"]
MAX_DOC_CHARS = 2_000


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    documents = df["text"].str.slice(0, MAX_DOC_CHARS).tolist()
    corpora = sorted(df["corpus"].unique())

    for method in METHODS_TO_MAP:
        mdir = EXP_DIR / method
        ed = np.load(mdir / "embeddings.npz", allow_pickle=True)
        cd = np.load(mdir / "umap_coords.npz", allow_pickle=True)
        ep = {i: k for k, i in enumerate(list(ed["id"]))}
        cp = {i: k for k, i in enumerate(list(cd["id"]))}
        emb = ed["emb"][[ep[i] for i in df["id"]]].astype(np.float32)
        coords = cd["coords"][[cp[i] for i in df["id"]]].astype(np.float32)

        print(f"{method}:")
        layers = label_regions(documents, emb, coords, OBJECT_DESCRIPTION, CORPUS_DESCRIPTION)
        render_map(
            coords,
            layers,
            df["name"].to_numpy(),
            df["corpus"].to_numpy(),
            df["url"].to_numpy(),
            df["char_len"].to_numpy(),
            f"{DATASET_TITLE} — {method.upper()}",
            f"{' / '.join(corpora)} — corpus signal removed via {method}",
            mdir / "map.html",
        )
        print(f"  wrote {mdir / 'map.html'}")


if __name__ == "__main__":
    main()
