"""Stage 01 — embed entity text with Cohere embed-v4.0 (clustering, 1024D).

input_type="clustering": the only downstream use is grouping/visualization (UMAP +
Toponymy), never a dot-product/search. Embeddings are saved raw (not unit-normed);
UMAP uses cosine, and the Phase-2 integration step will handle its own normalization.

Reads:  data/entities.parquet
Output: data/embeddings.npz  (emb [N x dim] float32, id [N] aligned to entities row order)
"""

import os

import cohere
import numpy as np
import pandas as pd
from config import (
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    COHERE_INPUT_TYPE,
    COHERE_OUTPUT_DIM,
    EMBED_BATCH,
    EMBEDDINGS_NPZ,
    ENTITIES_PARQUET,
)
from tqdm import tqdm


def main():
    df = pd.read_parquet(ENTITIES_PARQUET)
    texts = df["text"].tolist()
    ids = df["id"].to_numpy()
    print(f"Loaded {len(df)} entities")

    co = cohere.ClientV2(api_key=CO_API_KEY)

    all_emb = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="Embedding"):
        batch = texts[i : i + EMBED_BATCH]
        resp = co.embed(
            texts=batch,
            model=COHERE_EMBED_MODEL,
            input_type=COHERE_INPUT_TYPE,
            embedding_types=["float"],
            output_dimension=COHERE_OUTPUT_DIM,
        )
        all_emb.extend(resp.embeddings.float_)

    emb = np.asarray(all_emb, dtype=np.float32)
    assert emb.shape[0] == len(df), f"embedding/row mismatch: {emb.shape[0]} vs {len(df)}"

    # Atomic write (np.savez keeps the .npz suffix exactly when present).
    tmp = str(EMBEDDINGS_NPZ) + ".tmp.npz"
    np.savez(tmp, emb=emb, id=ids)
    os.replace(tmp, EMBEDDINGS_NPZ)
    print(f"Saved embeddings {emb.shape} to {EMBEDDINGS_NPZ}")


if __name__ == "__main__":
    main()
