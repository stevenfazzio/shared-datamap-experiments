"""Stage 00b (marvel_dc_anon) — derive a name-stripped variant of the marvel_dc entities.

Removes the character's name tokens (and the prepended name line) from the embed text, so
cross-corpus matching is driven by archetype rather than shared name words — the source of
the Stephen Strange->Adam Strange / Nick Fury->Helena "Fury" artifacts. `name` is kept for
display and the aptness check; only `text` changes.

Run with DATASET=marvel_dc_anon.
Reads:  data/marvel_dc/entities.parquet
Output: data/marvel_dc_anon/entities.parquet
"""

import os
import re

import pandas as pd
from config import ENTITIES_PARQUET, PROJECT_ROOT

SOURCE = PROJECT_ROOT / "data" / "marvel_dc" / "entities.parquet"


def strip_names(text, name):
    body = text.split("\n", 1)[-1]  # drop the prepended name line
    tokens = sorted({name, *[t for t in re.split(r"\s+", name) if len(t) >= 3]}, key=len, reverse=True)
    for t in tokens:
        body = re.sub(r"\b" + re.escape(t) + r"\b", " ", body, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", body).strip()


def main():
    df = pd.read_parquet(SOURCE).reset_index(drop=True)
    df["text"] = [strip_names(t, n) for t, n in zip(df["text"], df["name"])]
    df["char_len"] = df["text"].str.len()
    df = df[df["char_len"] >= 40].reset_index(drop=True)

    tmp = str(ENTITIES_PARQUET) + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, ENTITIES_PARQUET)
    print(f"Wrote {len(df)} name-stripped entities {df['corpus'].value_counts().to_dict()} to {ENTITIES_PARQUET}")
    print(f"Median text length: {int(df['char_len'].median())} chars")
    s = df.iloc[0]
    print(f"sample [{s['name']}]: {s['text'][:160]}")


if __name__ == "__main__":
    main()
