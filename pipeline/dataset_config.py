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

# Cross-pantheon archetype groups (display names) for the N-way aptness ground truth: each
# group is one shared archetype with its members in each tradition. Flattened below into the
# name -> {acceptable cross-pantheon analogues} form the aptness metric consumes. Lenient and
# many-to-many on purpose (a figure may sit in several groups); a low hit-rate on the clear
# archetypes (sun / moon / sea / death / love) would be the real warning sign.
PANTHEONS_ARCHETYPES = {
    "sky-father / king of the gods": {
        "Greek": ["Zeus"],
        "Norse": ["Odin"],
        "Egyptian": ["Ra", "Amun", "Atum"],
        "Hindu": ["Indra", "Dyaus"],
    },
    "thunder / storm / war-champion": {
        "Greek": ["Zeus", "Ares"],
        "Norse": ["Thor", "Týr"],
        "Egyptian": ["Set", "Montu", "Anhur"],
        "Hindu": ["Indra", "Kartikeya"],
    },
    "love / beauty / desire": {
        "Greek": ["Aphrodite", "Eros"],
        "Norse": ["Freyja"],
        "Egyptian": ["Hathor"],
        "Hindu": ["Kamadeva", "Radha", "Lakshmi"],
    },
    "great queen / consort-mother": {
        "Greek": ["Hera"],
        "Norse": ["Frigg"],
        "Egyptian": ["Mut", "Isis"],
        "Hindu": ["Parvati", "Saraswati"],
    },
    "sea & the waters": {
        "Greek": ["Poseidon", "Oceanus", "Amphitrite", "Triton"],
        "Norse": ["Njörðr", "Ægir", "Rán"],
        "Egyptian": ["Sobek", "Hapi"],
        "Hindu": ["Varuna", "Ganga"],
    },
    "death & the underworld": {
        "Greek": ["Hades", "Persephone", "Thanatos", "Charon"],
        "Norse": ["Hel"],
        "Egyptian": ["Osiris", "Anubis", "Nephthys"],
        "Hindu": ["Yama"],
    },
    "the sun": {
        "Greek": ["Helios", "Apollo"],
        "Norse": ["Sól"],
        "Egyptian": ["Ra", "Aten", "Khepri"],
        "Hindu": ["Surya", "Pushan", "Savitr"],
    },
    "the moon": {"Greek": ["Selene"], "Norse": ["Máni"], "Egyptian": ["Khonsu", "Thoth"], "Hindu": ["Chandra"]},
    "dawn": {"Greek": ["Eos"], "Norse": ["Dagr"], "Hindu": ["Ushas"]},
    "night": {"Greek": ["Nyx"], "Norse": ["Nótt"], "Egyptian": ["Nut"], "Hindu": ["Ratri"]},
    "wisdom, craft & writing": {
        "Greek": ["Athena", "Hephaestus", "Hermes"],
        "Norse": ["Bragi", "Kvasir", "Mímir"],
        "Egyptian": ["Thoth", "Ptah"],
        "Hindu": ["Saraswati", "Vishvakarma", "Brahma"],
    },
    "trickster / messenger / liminal guide": {
        "Greek": ["Hermes"],
        "Norse": ["Loki", "Hermóðr"],
        "Egyptian": ["Thoth", "Anubis"],
    },
    "the hunt & wild beasts": {
        "Greek": ["Artemis", "Pan"],
        "Norse": ["Skaði", "Ullr"],
        "Egyptian": ["Neith", "Bastet", "Sekhmet"],
        "Hindu": ["Rudra"],
    },
    "healing & medicine": {
        "Greek": ["Asclepius"],
        "Norse": ["Eir"],
        "Egyptian": ["Serqet", "Heka"],
        "Hindu": ["Dhanvantari", "Ashvins"],
    },
    "fire & the forge": {"Greek": ["Hephaestus"], "Egyptian": ["Ptah"], "Hindu": ["Agni"]},
    "grain, fertility & the cultivated earth": {
        "Greek": ["Demeter", "Dionysus", "Persephone"],
        "Norse": ["Freyr", "Gefjon", "Sif"],
        "Egyptian": ["Osiris", "Renenutet"],
        "Hindu": ["Prithvi"],
    },
    "the earth": {"Greek": ["Gaia"], "Egyptian": ["Geb"], "Hindu": ["Prithvi"]},
    "primordial origin / first beings": {
        "Greek": ["Gaia", "Uranus", "Cronus"],
        "Norse": ["Ymir", "Búri", "Borr", "Auðumbla"],
        "Egyptian": ["Atum"],
        "Hindu": ["Brahma", "Aditi"],
    },
    "wind & air": {"Egyptian": ["Shu"], "Hindu": ["Vayu", "Maruts"]},
    "wealth & fortune": {"Greek": ["Tyche"], "Egyptian": ["Hapi"], "Hindu": ["Kubera", "Lakshmi"]},
}


def _known_from_archetypes(archetypes):
    """Flatten archetype groups into the aptness format: display name -> {acceptable
    analogues from OTHER pantheons}, unioned over every group a figure appears in.
    Same-pantheon members are never acceptable cross-corpus matches."""
    known = {}
    for group in archetypes.values():
        for pantheon, members in group.items():
            others = {m for p, ms in group.items() if p != pantheon for m in ms}
            for m in members:
                known.setdefault(m, set()).update(others)
    return {k: v for k, v in known.items() if v}


PANTHEONS_KNOWN = _known_from_archetypes(PANTHEONS_ARCHETYPES)

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
    "pantheons": {
        "colors": {
            "Greek": "#e41a1c",  # red
            "Norse": "#377eb8",  # blue
            "Egyptian": "#ff7f00",  # gold / orange
            "Hindu": "#4daf4a",  # green
        },
        "object_description": "Greek, Norse, Egyptian, and Hindu mythological figures",
        "corpus_description": (
            "deities and mythological figures from Greek, Norse, Egyptian, and Hindu "
            "traditions (gods, goddesses, Titans, primordial beings), each described by "
            "its Wikipedia lead"
        ),
        "title": "Greek × Norse × Egyptian × Hindu",
        "known": PANTHEONS_KNOWN,
        # ~190-point map: between the greek_norse pilot (100) and marvel_dc (2000).
        "toponymy": {"min_clusters": 4, "base_min_cluster_size": 5, "min_samples": 3},
    },
    "pantheons_mm": {
        # Modality experiment: the same myth figures embedded twice (image vs text); the
        # "corpus" we erase is MODALITY, not pantheon. Built by 00c_build_multimodal.py.
        "colors": {"image": "#ff7f00", "text": "#377eb8"},  # image=orange, text=blue
        "object_description": (
            "mythological figures, each represented both as an artwork (image) and as encyclopedic text"
        ),
        "corpus_description": (
            "the same set of mythological figures embedded two ways — as a depiction in "
            "art (image) and as a Wikipedia description (text)"
        ),
        "title": "Myth figures: image × text",
        "known": {},  # no cross-modal analogue ground truth; validated by cross-modal retrieval
        "text_input_type": "clustering",  # text side embedded with this input_type
        # ~396-point map (up to 198 figures x 2 modalities).
        "toponymy": {"min_clusters": 4, "base_min_cluster_size": 8, "min_samples": 3},
    },
}

# Name-stripped variant of marvel_dc (same settings; entities derived by 00b_strip_names.py
# from data/marvel_dc/entities.parquet, with character names removed from the embed text).
DATASETS["marvel_dc_anon"] = {**DATASETS["marvel_dc"], "title": "Marvel × DC (names stripped)"}

# input_type A/B: same images + figures as pantheons_mm, text re-embedded with search_query
# (the one text input_type that actually differs — in embed-v4.0 clustering == search_document
# == classification, only search_query is distinct, cos ~0.78) to test whether the modality gap
# is an input_type artifact. search_query is also the retrieval-appropriate "query" encoding.
DATASETS["pantheons_mm_sq"] = {
    **DATASETS["pantheons_mm"],
    "title": "Myth figures: image × text (search_query)",
    "text_input_type": "search_query",
}
