"""Stage 02 — RAW co-embed both corpuses to 2D with UMAP (baseline; NO integration yet).

This is the Phase-1 baseline the later integration ablation is measured against. Expect
the two corpuses to separate here (corpus is hugely predictive of raw content) — that's
not failure, it's the thing integration has to dissolve.

Reads:  data/embeddings.npz
Output: data/umap_coords.npz  (coords [N x 2] float32, id [N] aligned)
"""

import os

import numpy as np
import umap
from config import (
    EMBEDDINGS_NPZ,
    UMAP_COORDS_NPZ,
    UMAP_METRIC,
    UMAP_MIN_DIST,
    UMAP_N_NEIGHBORS,
    UMAP_RANDOM_STATE,
)


def main():
    data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
    emb = data["emb"]
    ids = data["id"]
    print(f"Loaded embeddings {emb.shape}")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=UMAP_RANDOM_STATE,
    )
    coords = reducer.fit_transform(emb).astype(np.float32)

    tmp = str(UMAP_COORDS_NPZ) + ".tmp.npz"
    np.savez(tmp, coords=coords, id=ids)
    os.replace(tmp, UMAP_COORDS_NPZ)
    print(f"Saved coords {coords.shape} to {UMAP_COORDS_NPZ}")


if __name__ == "__main__":
    main()
