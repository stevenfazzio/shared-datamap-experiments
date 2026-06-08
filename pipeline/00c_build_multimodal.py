"""Stage 00c — build the multimodal dataset `pantheons_mm` from `pantheons`.

The modality experiment: embed the SAME mythological figures two ways and erase the
{image, text} split (modality is the "corpus"). For each pantheons figure we fetch its
Wikipedia lead image, embed it with embed-v4.0 (input_type="image"), and pair it with the
figure's already-computed text embedding. Output is 2N row-aligned points (a text row and
an image row per figure), so 06/07/08 run unchanged with corpus = modality.

Reads:  data/pantheons/{entities.parquet, embeddings.npz}   (figures + clustering text)
Writes: data/<DATASET>/{entities.parquet (2N rows; +pair_id,+pantheon), embeddings.npz}
Shared under data/pantheons/ (reused across variants): image_uris.json (downloaded
thumbnails) + image_embeddings.npz (image side embedded once).

Run with DATASET=pantheons_mm (text=clustering) or pantheons_mm_sd (text=search_document) —
the text input_type comes from the dataset config; images are embedded once and reused, so
the input_type A/B costs only a text re-embed.
"""

import base64
import json
import os
import time

import cohere
import numpy as np
import pandas as pd
import requests
from config import (
    CO_API_KEY,
    COHERE_EMBED_MODEL,
    COHERE_INPUT_TYPE,
    COHERE_OUTPUT_DIM,
    EMBEDDINGS_NPZ,
    ENTITIES_PARQUET,
    MM_TEXT_INPUT_TYPE,
    PROJECT_ROOT,
    WIKIPEDIA_API,
    WIKIPEDIA_USER_AGENT,
)
from tqdm import tqdm

SRC = PROJECT_ROOT / "data" / "pantheons"
CACHE = SRC / "image_uris.json"  # shared across mm variants (downloaded thumbnails)
IMG_EMB_NPZ = SRC / "image_embeddings.npz"  # shared image embeddings, keyed by figure id
IMG_BATCH = 32
MAX_FIGURES = 250  # effectively all (~170 of 198 figures have a usable lead image)
SUPPORTED = {"image/jpeg", "image/png", "image/webp", "image/gif"}


def _get_json(session, params, max_retries=6, timeout=30):
    """GET the API with exponential backoff on network errors / 429."""
    for attempt in range(max_retries):
        try:
            r = session.get(WIKIPEDIA_API, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            print(f"  query retry ({e})")
            time.sleep(min(2**attempt * 3, 45))


def thumbnail_urls(session, titles, batch=50):
    """{resolved title -> lead-image thumbnail URL} via BATCHED pageimages + continuation.

    Batching (following the picontinue token) keeps this to a handful of requests instead of
    one per title — which is what avoids the throttling that makes per-title pageimages drop
    thumbnails under load (a thumbnail-less response, not an error).
    """
    urls = {}
    for i in range(0, len(titles), batch):
        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "pageimages",
            "piprop": "thumbnail|name",
            "pithumbsize": 384,
            "redirects": 1,
            "titles": "|".join(titles[i : i + batch]),
        }
        cont = {}
        while True:
            data = _get_json(session, {**params, **cont})
            for p in data.get("query", {}).get("pages", []):
                src = p.get("thumbnail", {}).get("source")
                if src:
                    urls[p["title"]] = src
            if "continue" in data:
                cont = data["continue"]
            else:
                break
        time.sleep(0.5)
    return urls


def download_uri(session, url, max_retries=5, timeout=30):
    """Download a thumbnail -> base64 data URI (or None for unsupported/failed), with backoff."""
    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=timeout)
            r.raise_for_status()
            ct = r.headers.get("Content-Type", "").split(";")[0].strip()
            if ct not in SUPPORTED:
                return None
            return f"data:{ct};base64,{base64.b64encode(r.content).decode()}"
        except requests.exceptions.RequestException:
            if attempt == max_retries - 1:
                return None
            time.sleep(min(2**attempt * 3, 30))
    return None


def load_or_fetch_uris(df):
    """{figure id -> data URI} for figures with a usable lead image. Caches successes only,
    so throttle/transient failures retry on the next run instead of sticking as None."""
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    cache = {k: v for k, v in cache.items() if v}  # keep successes; drop any stale None
    if os.environ.get("MM_SKIP_FETCH"):  # variants reuse the primary build's images (no new fetch)
        print(f"MM_SKIP_FETCH set; using {len(cache)} cached images")
        return cache
    todo = df[~df["id"].isin(cache)].reset_index(drop=True)
    if len(todo):
        if cache:
            print(f"reusing {len(cache)} cached images; fetching {len(todo)} more")
        session = requests.Session()
        session.headers.update({"User-Agent": WIKIPEDIA_USER_AGENT})
        urls = thumbnail_urls(session, todo["title"].tolist())
        for k, r in enumerate(tqdm(todo.itertuples(), total=len(todo), desc="lead images")):
            url = urls.get(r.title)
            if url:
                uri = download_uri(session, url)
                if uri:
                    cache[r.id] = uri
                time.sleep(5.0)  # pace only real downloads (upload.wikimedia throttles bursts)
            if k % 25 == 0:
                CACHE.write_text(json.dumps(cache))
        CACHE.write_text(json.dumps(cache))
    return cache


def embed_images(uris):
    co = cohere.ClientV2(api_key=CO_API_KEY)
    out = []
    for i in tqdm(range(0, len(uris), IMG_BATCH), desc="embedding images"):
        resp = co.embed(
            images=uris[i : i + IMG_BATCH],
            model=COHERE_EMBED_MODEL,
            input_type="image",
            embedding_types=["float"],
            output_dimension=COHERE_OUTPUT_DIM,
        )
        out.extend(resp.embeddings.float_)
    return np.asarray(out, dtype=np.float32)


def load_or_embed_images(ids, uris):
    """Image embeddings for `ids`, reusing IMG_EMB_NPZ (shared across variants); embed and
    merge-save the rest. Keeps the image side embedded once, even across input_type variants."""
    cache = {}
    if IMG_EMB_NPZ.exists():
        d = np.load(IMG_EMB_NPZ, allow_pickle=True)
        cache = {i: v for i, v in zip(list(d["id"]), d["emb"])}
    todo = [i for i in ids if i not in cache]
    if todo:
        for i, v in zip(todo, embed_images([uris[i] for i in todo])):
            cache[i] = v
        all_ids = list(cache)
        all_emb = np.asarray([cache[i] for i in all_ids], dtype=np.float32)
        tmp = str(IMG_EMB_NPZ) + ".tmp.npz"
        np.savez(tmp, emb=all_emb, id=np.array(all_ids))
        os.replace(tmp, IMG_EMB_NPZ)
    return np.asarray([cache[i] for i in ids], dtype=np.float32)


def text_embeddings(kept, input_type):
    """Text embeddings for kept figures. clustering -> reuse pantheons/embeddings.npz; any
    other input_type -> re-embed the text fresh (the input_type A/B knob)."""
    if input_type == COHERE_INPUT_TYPE:
        td = np.load(SRC / "embeddings.npz", allow_pickle=True)
        tpos = {i: k for k, i in enumerate(list(td["id"]))}
        return td["emb"][[tpos[i] for i in kept["id"]]].astype(np.float32)
    print(f"re-embedding text with input_type={input_type!r}")
    co = cohere.ClientV2(api_key=CO_API_KEY)
    texts, out = kept["text"].tolist(), []
    for i in tqdm(range(0, len(texts), 96), desc="embedding text"):
        resp = co.embed(
            texts=texts[i : i + 96],
            model=COHERE_EMBED_MODEL,
            input_type=input_type,
            embedding_types=["float"],
            output_dimension=COHERE_OUTPUT_DIM,
        )
        out.extend(resp.embeddings.float_)
    return np.asarray(out, dtype=np.float32)


def main():
    src = pd.read_parquet(SRC / "entities.parquet").reset_index(drop=True)
    src = src.nlargest(MAX_FIGURES, "char_len").reset_index(drop=True)
    print(f"source: {len(src)} pantheons figures")

    uris = load_or_fetch_uris(src)
    kept = src[src["id"].map(lambda i: bool(uris.get(i)))].reset_index(drop=True)
    missing = sorted(set(src["name"]) - set(kept["name"]))
    print(f"images: {len(kept)}/{len(src)} usable" + (f"  | dropped: {missing}" if missing else ""))

    text_emb = text_embeddings(kept, MM_TEXT_INPUT_TYPE)
    img_emb = load_or_embed_images(kept["id"].tolist(), uris)

    # interleave a text row and an image row per figure; image row's `text` = the figure's
    # text so Toponymy has content to name regions with (position still comes from the image
    # embedding); char_len shared so a figure's two markers match.
    rows, embs = [], []
    for k, r in kept.iterrows():
        base = {
            "name": r["name"],
            "title": r["title"],
            "url": r["url"],
            "text": r["text"],
            "char_len": r["char_len"],
            "pair_id": r["id"],
            "pantheon": r["corpus"],
        }
        rows.append({**base, "id": f"{r['id']}__text", "corpus": "text"})
        embs.append(text_emb[k])
        rows.append({**base, "id": f"{r['id']}__image", "corpus": "image"})
        embs.append(img_emb[k])

    out = pd.DataFrame(rows)[["id", "name", "corpus", "title", "url", "text", "char_len", "pair_id", "pantheon"]]
    emb = np.asarray(embs, dtype=np.float32)
    ids = out["id"].to_numpy()
    assert emb.shape[0] == len(out) == 2 * len(kept)

    tmp = str(ENTITIES_PARQUET) + ".tmp"
    out.to_parquet(tmp, index=False)
    assert len(pd.read_parquet(tmp)) == len(out)
    os.replace(tmp, ENTITIES_PARQUET)

    tmp = str(EMBEDDINGS_NPZ) + ".tmp.npz"
    np.savez(tmp, emb=emb, id=ids)
    os.replace(tmp, EMBEDDINGS_NPZ)

    print(f"wrote {len(out)} rows ({len(kept)} figures x 2 modalities) -> {ENTITIES_PARQUET}")
    print(f"wrote embeddings {emb.shape} -> {EMBEDDINGS_NPZ}")


if __name__ == "__main__":
    main()
