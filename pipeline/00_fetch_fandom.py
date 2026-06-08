"""Stage 00 (marvel_dc) — fetch ~1k notable characters each from the Marvel and DC Fandom
wikis, single mainstream continuity per side, with clean prose from the character template.

Roster: members of Category:Earth-616/Characters (Marvel) / Category:New Earth Characters
(DC), ranked by page length (a notability proxy), top TARGET_PER_CORPUS each. Text: the
character template's prose params (Overview / Personality / Origin / History), strip_code'd
and truncated — NOT the infobox (alias/reference noise). Fandom lacks TextExtracts, and the
two wikis differ in register, which is exactly the Phase-3 medium gap to erase.

Run with DATASET=marvel_dc. Output: data/marvel_dc/entities.parquet
"""

import os
import re
import time

import mwparserfromhell as mw
import pandas as pd
import requests
from config import ENTITIES_PARQUET
from tqdm import tqdm

UA = {"User-Agent": "datamap-experiments/0.1 (educational research pipeline)"}
TARGET_PER_CORPUS = int(os.environ.get("MAXCHARS_PER_CORPUS", "1000"))
TEXT_CHARS = 3000
WIKITEXT_BATCH = 20  # full-page wikitext per request (pages are large)
PROSE_PARAMS = ["Overview", "Personality", "Origin", "HistoryText", "History", "Abilities"]

SOURCES = [
    {"corpus": "Marvel", "base": "https://marvel.fandom.com", "category": "Category:Earth-616/Characters"},
    {"corpus": "DC", "base": "https://dc.fandom.com", "category": "Category:New Earth Characters"},
]


def api(base, params, max_retries=5, timeout=60):
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(max_retries):
        try:
            r = requests.get(base + "/api.php", params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise
            wait = min(2**attempt * 4, 60)
            print(f"  request failed ({e}); retry in {wait}s")
            time.sleep(wait)


def roster(base, category):
    """All non-redirect character pages in the category, mapped to page length (bytes)."""
    items, cont, page = {}, {}, 0
    while True:
        data = api(
            base,
            {
                "action": "query",
                "generator": "categorymembers",
                "gcmtitle": category,
                "gcmtype": "page",
                "gcmnamespace": 0,  # main namespace only (drops User:/Talk: junk)
                "gcmlimit": 500,
                "prop": "info",
                **cont,
            },
        )
        for p in data.get("query", {}).get("pages", []):
            if p.get("redirect"):
                continue
            items[p["title"]] = p.get("length", 0)
        page += 1
        if page % 10 == 0:
            print(f"    rostered {len(items)} pages so far ...")
        if "continue" in data:
            cont = data["continue"]
            time.sleep(0.2)
        else:
            break
    return items


def extract_text(wikitext, name):
    """Prose params from the (biggest) character template; strip_code; truncate."""
    tmps = mw.parse(wikitext).filter_templates()
    char_tmps = [t for t in tmps if "character template" in str(t.name).lower()]
    cand = char_tmps or tmps
    if not cand:
        return ""
    big = max(cand, key=lambda t: len(str(t)))
    parts = []
    for pname in PROSE_PARAMS:
        if big.has(pname):
            txt = " ".join(mw.parse(str(big.get(pname).value)).strip_code().split())
            if len(txt) > 30:
                parts.append(txt)
        if sum(len(p) for p in parts) > TEXT_CHARS:
            break
    body = "  ".join(parts)[:TEXT_CHARS]
    return f"{name}\n{body}" if body else ""


def fetch_corpus(src):
    print(f"[{src['corpus']}] rostering {src['category']} ...")
    lengths = roster(src["base"], src["category"])
    titles = [t for t, _ in sorted(lengths.items(), key=lambda kv: -kv[1])[:TARGET_PER_CORPUS]]
    print(f"[{src['corpus']}] {len(lengths)} members; fetching text for top {len(titles)} by length")

    rows = []
    for i in tqdm(range(0, len(titles), WIKITEXT_BATCH), desc=src["corpus"]):
        batch = titles[i : i + WIKITEXT_BATCH]
        data = api(
            src["base"],
            {
                "action": "query",
                "prop": "revisions",
                "rvprop": "content",
                "rvslots": "main",
                "titles": "|".join(batch),
            },
        )
        for p in data.get("query", {}).get("pages", []):
            title = p.get("title", "")
            revs = p.get("revisions")
            if not revs:
                continue
            wt = revs[0]["slots"]["main"]["content"]
            name = re.sub(r"\s*\([^()]*\)\s*$", "", title)  # drop trailing reality parenthetical
            text = extract_text(wt, name)
            if len(text) < 60:
                continue
            rows.append(
                {
                    "id": f"{src['corpus'].lower()}:{title}",
                    "name": name,
                    "corpus": src["corpus"],
                    "title": title,
                    "url": f"{src['base']}/wiki/{title.replace(' ', '_')}",
                    "text": text,
                    "char_len": len(text),
                }
            )
        time.sleep(0.3)
    print(f"[{src['corpus']}] built {len(rows)} character rows")
    return rows


def main():
    rows = []
    for src in SOURCES:
        rows.extend(fetch_corpus(src))
    df = pd.DataFrame(rows).sort_values(["corpus", "name"]).reset_index(drop=True)

    tmp = str(ENTITIES_PARQUET) + ".tmp"
    df.to_parquet(tmp, index=False)
    assert len(pd.read_parquet(tmp)) == len(df), "row count mismatch"
    os.replace(tmp, ENTITIES_PARQUET)
    print(f"\nWrote {len(df)} characters {df['corpus'].value_counts().to_dict()} to {ENTITIES_PARQUET}")
    print(f"Median text length: {int(df['char_len'].median())} chars")


if __name__ == "__main__":
    main()
