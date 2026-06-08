"""Central config. The DATASET env var selects the corpus pair (greek_norse / marvel_dc);
all artifacts live under data/<DATASET>/. Every stage does `from config import ...` (the
stage's own dir is on sys.path when run as `python pipeline/XX.py`). Edit constants here
(or dataset_config.py for per-pair settings) for smoke tests rather than adding CLI args.
"""

import os
from pathlib import Path

from dataset_config import DATASETS
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Active dataset ───────────────────────────────────────────────────────────
DATASET = os.environ.get("DATASET", "greek_norse")
if DATASET not in DATASETS:
    raise ValueError(f"Unknown DATASET={DATASET!r}; choose from {sorted(DATASETS)}")
_DS = DATASETS[DATASET]

DATA_DIR = PROJECT_ROOT / "data" / DATASET
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── API keys (from the shell env; .env may supplement, not override) ─────────
load_dotenv(PROJECT_ROOT / ".env")
CO_API_KEY = os.environ.get("CO_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# ── Row-aligned artifacts (each carries an `id`) ─────────────────────────────
ENTITIES_PARQUET = DATA_DIR / "entities.parquet"
EMBEDDINGS_NPZ = DATA_DIR / "embeddings.npz"
UMAP_COORDS_NPZ = DATA_DIR / "umap_coords.npz"
LABELS_PARQUET = DATA_DIR / "labels.parquet"
MAP_HTML = DATA_DIR / "map.html"
EVAL_REPORT_JSON = DATA_DIR / "eval_report.json"

# ── Wikipedia fetch (greek_norse) ────────────────────────────────────────────
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = "datamap-experiments/0.1 (educational research pipeline)"
MAX_TEXT_CHARS = 2_500

# ── Embedding (Cohere embed-v4.0) ────────────────────────────────────────────
COHERE_EMBED_MODEL = "embed-v4.0"
COHERE_INPUT_TYPE = "clustering"
COHERE_OUTPUT_DIM = 1024
EMBED_BATCH = 96

# ── UMAP ─────────────────────────────────────────────────────────────────────
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.05
UMAP_METRIC = "cosine"
UMAP_RANDOM_STATE = 42

# ── Toponymy (Opus naming; cluster knobs scale with map size, per dataset) ───
ANTHROPIC_MODEL_NAMING = "claude-opus-4-8"
ANTHROPIC_MAX_CONCURRENCY = 12
TOPONYMY_MIN_CLUSTERS = _DS["toponymy"]["min_clusters"]
TOPONYMY_BASE_MIN_CLUSTER_SIZE = _DS["toponymy"]["base_min_cluster_size"]
TOPONYMY_MIN_SAMPLES = _DS["toponymy"]["min_samples"]

# ── Per-dataset display/eval settings ────────────────────────────────────────
CORPUS_COLOR_MAPPING = _DS["colors"]
OBJECT_DESCRIPTION = _DS["object_description"]
CORPUS_DESCRIPTION = _DS["corpus_description"]
DATASET_TITLE = _DS["title"]
KNOWN = _DS["known"]
# Multimodal text-side input_type (pantheons_mm variants); defaults to the standard text type.
MM_TEXT_INPUT_TYPE = _DS.get("text_input_type", COHERE_INPUT_TYPE)
