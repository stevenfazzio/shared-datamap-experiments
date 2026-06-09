"""Stage 07b — EVoC high-D region labels for the integration ablation (the high-D lens, #2).

Stage 07 names regions on the 2D UMAP coords (the map substrate). This names regions that
EVoC discovers in the NATIVE 1024-d embedding space (EVoC does its own internal reduction;
ToponymyClusterer would cluster the 2D projection). Comparing the two answers "does the UMAP
map overstate how merged a method is?" — stage 11 reads both labels.parquet (2D) and
labels_highd.parquet (this) and reports the 2D-vs-high-D mixing delta.

Scoped (via DATASET) to where the substrate matters most: pantheons_mm (the nonlinear
modality gap — where the 2D map may merge cones the high-D space keeps apart) and pantheons
(a corpus control, where 2D and high-D should agree). Needs the optional `evoc` package
(uv add evoc). No map is rendered — the metric needs region membership, not layout.

Reads:  data/<dataset>/experiments/<method>/{embeddings.npz, umap_coords.npz}, entities.parquet
Writes: data/<dataset>/experiments/<method>/labels_highd.parquet  (id + label_layer_0..N)
"""

import numpy as np
import pandas as pd
from config import CORPUS_DESCRIPTION, DATA_DIR, ENTITIES_PARQUET, OBJECT_DESCRIPTION
from integrations import METHODS
from topic_labeling import label_regions, save_labels

EXP_DIR = DATA_DIR / "experiments"
MAX_DOC_CHARS = 2_000


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    documents = df["text"].str.slice(0, MAX_DOC_CHARS).tolist()

    for method in METHODS:
        mdir = EXP_DIR / method
        ed = np.load(mdir / "embeddings.npz", allow_pickle=True)
        cd = np.load(mdir / "umap_coords.npz", allow_pickle=True)
        ep = {i: k for k, i in enumerate(list(ed["id"]))}
        cp = {i: k for k, i in enumerate(list(cd["id"]))}
        emb = ed["emb"][[ep[i] for i in df["id"]]].astype(np.float32)
        coords = cd["coords"][[cp[i] for i in df["id"]]].astype(np.float32)

        print(f"{method} (EVoC high-D):")
        layers = label_regions(documents, emb, coords, OBJECT_DESCRIPTION, CORPUS_DESCRIPTION, clusterer="evoc")
        save_labels(mdir / "labels_highd.parquet", df["id"].to_numpy(), layers)
        print(f"  wrote {mdir / 'labels_highd.parquet'}")


if __name__ == "__main__":
    main()
