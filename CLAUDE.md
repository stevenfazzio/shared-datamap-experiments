# datamap-experiments

Testing an idea: build a **combined datamap of two (or more) corpuses by erasing the
corpus-identity signal**, so points position by shared *archetype* rather than by which
corpus they came from. Toponymy region-names then read as the cross-corpus analogies
(e.g. a region "Atlantean Aquatic Rulers" holding both Namor and Aquaman).

Detailed findings + the Fandom-fetch playbook live in **project memory** (loaded each
session). Read the **`toponymy` skill** before reasoning about Toponymy output, and the
**`running-data-pipelines` skill** before running/modifying stages.

## How to run

`uv` for env (`uv sync --extra dev`), `ruff` for lint/format. Stages are run as scripts;
the active corpus pair is chosen by the **`DATASET` env var**, and all artifacts land in
`data/<DATASET>/`. Per-dataset settings (colors, descriptions, KNOWN analogues, Toponymy
knobs) are in `pipeline/dataset_config.py`.

```bash
# Greek × Norse pilot (Wikipedia; checkable). DATASET defaults to greek_norse.
uv run python pipeline/00_fetch_deities.py
uv run python pipeline/01_embed.py && uv run python pipeline/02_reduce_umap.py
uv run python pipeline/03_label_topics.py && uv run python pipeline/04_visualize.py
uv run python pipeline/05_evaluate.py
uv run python pipeline/06_ablation.py && uv run python pipeline/07_ablation_maps.py

# Marvel × DC (Fandom wikis, ~2k chars). Uses the ablation path (06/07 cover raw + 3 methods).
DATASET=marvel_dc uv run python pipeline/00_fetch_fandom.py
DATASET=marvel_dc uv run python pipeline/01_embed.py
DATASET=marvel_dc uv run python pipeline/06_ablation.py
DATASET=marvel_dc uv run python pipeline/07_ablation_maps.py
DATASET=marvel_dc uv run python pipeline/08_rank_diagnostic.py   # INLP rank-sweep

# Marvel × DC, names stripped from text (A/B variant)
DATASET=marvel_dc_anon uv run python pipeline/00b_strip_names.py   # derives from marvel_dc
DATASET=marvel_dc_anon uv run python pipeline/01_embed.py
DATASET=marvel_dc_anon uv run python pipeline/06_ablation.py

# Greek × Norse × Egyptian × Hindu (N-way, Wikipedia). Multiclass LEACE; ablation path.
DATASET=pantheons uv run python pipeline/00_fetch_deities.py
DATASET=pantheons uv run python pipeline/01_embed.py
DATASET=pantheons uv run python pipeline/06_ablation.py
DATASET=pantheons uv run python pipeline/07_ablation_maps.py
DATASET=pantheons uv run python pipeline/08_rank_diagnostic.py   # INLP sweep: signal is rank-(K-1)=3
```

(`make pipeline` runs the greek_norse single-method path only.) **View a map:** serve the
data dir and open via the claude-in-chrome extension (never a `file://` URL):
`python3 -m http.server 8754 --bind 127.0.0.1 --directory data` →
`http://127.0.0.1:8754/<dataset>/experiments/<method>/map.html`. Maps use
datamapplot `offline_mode=True`, so the HTML is self-contained (works without a network).

## Pipeline stages (`pipeline/`)

- **00_fetch_deities.py** — greek_norse + pantheons: curated Wikipedia leads (intro extracts,
  batched); roster set chosen by DATASET via the `ROSTERS` dict (a polite pause between pantheons).
- **00_fetch_fandom.py** — marvel_dc: Fandom character prose; roster by page length (ns0),
  prose from the `{{Character Template}}` params via `mwparserfromhell`.
- **00b_strip_names.py** — marvel_dc_anon: strip character names from marvel_dc text.
- **01_embed.py** — Cohere `embed-v4.0`, `input_type=clustering`, 1024-d → `embeddings.npz`.
- **02_reduce_umap.py** — UMAP → `umap_coords.npz` (n_neighbors=15, min_dist=0.05, cosine, seed 42).
- **03_label_topics.py / topic_labeling.py** — Toponymy + Opus-4.8 namer → `labels.parquet`.
- **04_visualize.py / mapviz.py** — datamapplot interactive HTML, **color = corpus**.
- **05_evaluate.py / metrics.py** — baseline go/no-go (cross-corpus NN aptness, mixing, recoverability).
- **06_ablation.py** — integrate raw/center/leace/harmony → UMAP → metrics → `ablation.json`
  + `experiments/<method>/`. `integrations.py` holds the methods.
- **07_ablation_maps.py** — render a map per integration method.
- **08_rank_diagnostic.py** — INLP rank-k sweep (how much corpus separation is linear vs irreducible).

Artifacts are row-aligned by an `id` column; entities schema: `id, name, corpus, title, url,
text, char_len`. Writes are atomic (tmp + `os.replace`); see `running-data-pipelines`.

## Conventions & gotchas

- **Cohere embed-v4.0 is multimodal** (text + images, shared space) — relevant for the planned
  multimodal experiment; no new embedder needed.
- **DataMapPlot label layers are finest-first** (semantic-github-map's "coarsest first" comment
  is wrong; taskmaster-map is the correct reference).
- **Opus 4.8 namer**: the stock `AsyncAnthropicNamer` passes `temperature`, which Opus 4.7/4.8
  reject (400) — `OpusAnthropicNamer` in topic_labeling.py drops it.
- **Toponymy knobs scale with map size** (per dataset): small maps need a small
  `base_min_cluster_size`. Toponymy names *regions of space*; `Unlabelled` = unnamed region
  (signal), not a failure.
- **Fandom APIs** differ from Wikipedia (no TextExtracts; prose in template params; tag-URL
  escaping `.`→`*d*`/`&`→`*a*`; Mostlinkedpages disabled). See the `fandom-mediawiki-extraction`
  memory.

## Metrics (how to read)

1. **Cross-corpus kNN mixing** (primary) — fraction of each point's neighbors from a *different*
   corpus; ~0 = separate blobs, fully interleaved → (K−1)/K (0.5 for two corpuses, ~0.75 for four).
2. **Linear recoverability** — CV accuracy predicting corpus; only reliable when **d < n**
   (unreliable in the ~100-pt pilot where d=1024 ≫ n).
3. **Aptness vs KNOWN analogues** — top-1 cross-corpus NN, under cosine and CSLS.

## Findings so far (see memory for detail)

Rank-1 **LEACE ≈ per-corpus centering ≥ Harmony**; the corpus difference is a low-rank linear
signal. Same-medium (Greek/Norse) interleaves nearly fully; different-medium + scale (Marvel/DC)
half-merges, and the residual is **genuine franchise structure to keep**, not under-merging.
CSLS is subsumed by integration. Input text dominates: names contaminate matching.
**N-way (Greek/Norse/Egyptian/Hindu) generalizes this exactly**: the corpus signal is **rank-(K−1)**
(INLP recoverability collapses 1.00→0.005 at k=3; closed-form multiclass LEACE — erasing the (K−1)-dim
class-mean subspace — removes precisely it), again **LEACE ≈ centering ≥ Harmony**, regions name as
cross-pantheon archetypes (Hades↔Hel↔Anubis↔Yama), residual is genuine (a Vishnu-avatar pocket; Horus a hub).

## Status & next

Done: greek_norse (pilot), marvel_dc (+ marvel_dc_anon), pantheons (N=4 mythologies; rank-(K−1)
confirmed). **Next:** (1) multimodal — paintings (images) × film/book summaries (text), testing
whether rank-1 erasure closes the CLIP-style modality gap (Cohere embed-v4.0 is already multimodal);
(2) write up + publish via GitHub Pages, possibly share to the TutteInstitute/toponymy discussions.

## Env

Keys from the shell env: `CO_API_KEY` (Cohere), `ANTHROPIC_API_KEY` (Toponymy naming).
`data/` is git-ignored; `uv.lock` is committed. Nothing is committed yet.
