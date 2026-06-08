"""Stage 00 — fetch Greek & Norse mythological figures from Wikipedia.

Two corpuses, SAME source (Wikipedia), so the medium/register is controlled and any
separation the map shows is content-driven, not style-driven. Per-entity text = the
figure's NAME prefixed to its plain-text article (lead + start of body), truncated.
We deliberately do NOT template the pantheon ("Greek god of X") into the text — that
would hand-feed the exact corpus axis later experiments try to erase; the corpus signal
should arise from genuine content (names, vocabulary, myths).

Output: data/entities.parquet  (id, name, corpus, title, url, text, char_len)
"""

import os
import time

import pandas as pd
import requests
from config import (
    ENTITIES_PARQUET,
    MAX_TEXT_CHARS,
    WIKIPEDIA_API,
    WIKIPEDIA_USER_AGENT,
)
from tqdm import tqdm

# Curated Wikipedia page titles. redirects=1 resolves common variants; any "missing"
# or empty titles are reported at the end so the lists can be corrected and re-run.
GREEK = [
    "Zeus",
    "Hera",
    "Poseidon",
    "Demeter",
    "Athena",
    "Apollo",
    "Artemis",
    "Ares",
    "Aphrodite",
    "Hephaestus",
    "Hermes",
    "Hestia",
    "Dionysus",
    "Hades",
    "Persephone",
    "Cronus",
    "Rhea (mythology)",
    "Uranus (mythology)",
    "Gaia",
    "Oceanus",
    "Hyperion (Titan)",
    "Theia",
    "Coeus",
    "Phoebe (Titaness)",
    "Crius",
    "Iapetus",
    "Mnemosyne",
    "Themis",
    "Prometheus",
    "Epimetheus",
    "Atlas (mythology)",
    "Helios",
    "Selene",
    "Eos",
    "Nyx",
    "Erebus",
    "Nike (mythology)",
    "Eros",
    "Pan (god)",
    "Nemesis",
    "Iris (mythology)",
    "Hecate",
    "Hypnos",
    "Thanatos",
    "Tyche",
    "Asclepius",
    "Heracles",
    "Eris (mythology)",
    "Charon",
    "Amphitrite",
    "Leto",
    "Triton (mythology)",
]
NORSE = [
    "Odin",
    "Frigg",
    "Thor",
    "Loki",
    "Baldr",
    "Týr",
    "Heimdall",
    "Bragi",
    "Iðunn",
    "Njörðr",
    "Freyr",
    "Freyja",
    "Sif",
    "Hel (being)",
    "Fenrir",
    "Jörmungandr",
    "Sleipnir",
    "Skaði",
    "Forseti",
    "Ullr",
    "Váli",
    "Víðarr",
    "Hœnir",
    "Mímir",
    "Hermóðr",
    "Höðr",
    "Nanna (Norse deity)",
    "Sigyn",
    "Gefjon",
    "Nótt",
    "Dagr",
    "Sól (Germanic mythology)",
    "Máni",
    "Ægir",
    "Rán",
    "Gerðr",
    "Angrboða",
    "Surtr",
    "Ymir",
    "Búri",
    "Borr",
    "Auðumbla",
    "Eir",
    "Fulla",
    "Sjöfn",
    "Vár",
    "Hlín",
    "Kvasir",
    "Móði and Magni",
    "Njörun",
]


def fetch_extracts(titles, max_retries=6, timeout=30, batch_size=20):
    """Return ({resolved_title: lead_plaintext}, [missing/empty titles]), following redirects.

    Intro-only extracts, batched: exintro lets exlimit go up to 20/request, so the whole
    corpus is ~6 requests. (Whole-page extracts are capped to 1/request AND rate-limited
    hard — that's the 429 — so intro+batch is both correct and polite.) Leads of mythology
    articles are rich enough to carry the figure's archetype.
    """
    extracts, missing = {}, []
    session = requests.Session()
    session.headers.update({"User-Agent": WIKIPEDIA_USER_AGENT})

    for i in tqdm(range(0, len(titles), batch_size), desc="Wikipedia"):
        batch = titles[i : i + batch_size]
        params = {
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "extracts",
            "explaintext": 1,
            "exintro": 1,
            "exlimit": "max",
            "redirects": 1,
            "titles": "|".join(batch),
        }
        data = None
        for attempt in range(max_retries):
            try:
                resp = session.get(WIKIPEDIA_API, params=params, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise
                wait = min(2**attempt * 5, 60)
                print(f"  batch @ {i} failed ({e}); retrying in {wait}s")
                time.sleep(wait)

        for page in data.get("query", {}).get("pages", []):
            title = page.get("title", "?")
            if page.get("missing"):
                missing.append(title)
                continue
            extract = (page.get("extract") or "").strip()
            if extract:
                extracts[title] = extract
            else:
                missing.append(f"{title} (empty extract)")
        time.sleep(1.0)  # be polite between batches

    return extracts, missing


def build_rows(extracts, corpus):
    rows = []
    for title, text in extracts.items():
        name = title.split(" (")[0]  # drop disambiguation parenthetical for display
        body = text[:MAX_TEXT_CHARS]
        rows.append(
            {
                "id": title.lower().replace(" ", "_"),
                "name": name,
                "corpus": corpus,
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                "text": f"{name}\n{body}",
                "char_len": len(body),
            }
        )
    return rows


def main():
    greek_ex, greek_missing = fetch_extracts(GREEK)
    norse_ex, norse_missing = fetch_extracts(NORSE)

    rows = build_rows(greek_ex, "Greek") + build_rows(norse_ex, "Norse")
    df = pd.DataFrame(rows).sort_values(["corpus", "name"]).reset_index(drop=True)

    # Atomic write + verify (never clobber expensive artifacts in place).
    tmp = str(ENTITIES_PARQUET) + ".tmp"
    df.to_parquet(tmp, index=False)
    verify = pd.read_parquet(tmp)
    assert len(verify) == len(df), f"row count mismatch: {len(verify)} vs {len(df)}"
    os.replace(tmp, ENTITIES_PARQUET)

    n_greek = int((df["corpus"] == "Greek").sum())
    n_norse = int((df["corpus"] == "Norse").sum())
    print(f"Wrote {len(df)} entities ({n_greek} Greek, {n_norse} Norse) to {ENTITIES_PARQUET}")
    print(f"Median article length: {int(df['char_len'].median())} chars (truncated at {MAX_TEXT_CHARS})")

    all_missing = greek_missing + norse_missing
    if all_missing:
        print(f"\n{len(all_missing)} MISSING / empty (fix titles and re-run):")
        for t in all_missing:
            print(f"  - {t}")


if __name__ == "__main__":
    main()
