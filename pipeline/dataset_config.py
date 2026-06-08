"""Per-dataset settings, selected via the DATASET env var (read in config.py).

Each dataset is one cross-corpus pair. Adding a pair = a new entry here + a fetch stage
that writes data/<dataset>/entities.parquet with the shared schema
(id, name, corpus, title, url, text, char_len). Everything downstream is dataset-agnostic.
"""

# Lenient semi-ground-truth analogue sets for the aptness metric (corpus_a -> {acceptable
# corpus_b matches}, by display name). Stretch entries included on purpose.
GREEK_NORSE_KNOWN = {
    "Zeus": {"Odin", "Thor"},
    "Hera": {"Frigg"},
    "Poseidon": {"Njörðr", "Ægir", "Rán"},
    "Hades": {"Hel"},
    "Aphrodite": {"Freyja", "Frigg"},
    "Ares": {"Týr", "Thor"},
    "Artemis": {"Skaði", "Ullr"},
    "Helios": {"Sól"},
    "Selene": {"Máni", "Sól"},
    "Nyx": {"Nótt"},
    "Eos": {"Dagr", "Sól"},
    "Hermes": {"Hermóðr", "Loki"},
    "Dionysus": {"Kvasir", "Bragi"},
    "Cronus": {"Ymir", "Surtr", "Borr"},
}

# Provisional Marvel -> DC analogues (real-name display forms; refine after the fetch shows
# the actual derived display names).
# Real-name display forms (page titles minus the reality parenthetical), verified present.
MARVEL_DC_KNOWN = {
    "Thanos": {"Darkseid"},
    "Wade Wilson": {"Slade Wilson"},  # Deadpool -> Deathstroke
    "Namor McKenzie": {"Orin", "Arthur Curry"},  # Namor -> Aquaman
    "Clinton Barton": {"Oliver Queen"},  # Hawkeye -> Green Arrow
    "Pietro Maximoff": {"Barry Allen", "Wally West"},  # Quicksilver -> Flash
    "Henry Pym": {"Raymond Palmer"},  # Ant-Man -> Atom
    "Stephen Strange": {"Kent Nelson"},  # Doctor Strange -> Doctor Fate
    "Nicholas Fury": {"Amanda Waller"},  # spymasters
}

DATASETS = {
    "greek_norse": {
        "colors": {"Greek": "#e41a1c", "Norse": "#377eb8"},
        "object_description": "Greek and Norse mythological figures",
        "corpus_description": (
            "figures from Greek and Norse mythology (gods, Titans, jötnar, and "
            "personifications), each described by its Wikipedia lead"
        ),
        "title": "Greek × Norse",
        "known": GREEK_NORSE_KNOWN,
        # Tuned for a small (~100-point) map.
        "toponymy": {"min_clusters": 3, "base_min_cluster_size": 4, "min_samples": 3},
    },
    "marvel_dc": {
        "colors": {"Marvel": "#ec1d24", "DC": "#0476f2"},  # Marvel red, DC blue
        "object_description": "Marvel and DC comic-book characters",
        "corpus_description": (
            "superheroes and villains from Marvel and DC comics, described by their "
            "character-wiki overview, personality, and origin"
        ),
        "title": "Marvel × DC",
        "known": MARVEL_DC_KNOWN,
        # Defaults suited to a ~2000-point map.
        "toponymy": {"min_clusters": 6, "base_min_cluster_size": 20, "min_samples": 5},
    },
}

# Name-stripped variant of marvel_dc (same settings; entities derived by 00b_strip_names.py
# from data/marvel_dc/entities.parquet, with character names removed from the embed text).
DATASETS["marvel_dc_anon"] = {**DATASETS["marvel_dc"], "title": "Marvel × DC (names stripped)"}
