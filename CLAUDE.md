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

# Myth figures image × text (multimodal; the erased "corpus" is MODALITY). Derives from pantheons.
DATASET=pantheons_mm uv run python pipeline/00c_build_multimodal.py   # fetch+embed lead images, stack
DATASET=pantheons_mm uv run python pipeline/06_ablation.py
DATASET=pantheons_mm uv run python pipeline/07_ablation_maps.py
DATASET=pantheons_mm uv run python pipeline/08_rank_diagnostic.py
DATASET=pantheons_mm uv run python pipeline/09_multimodal_eval.py   # cross-modal retrieval
DATASET=pantheons_mm uv run python pipeline/10_dim_diagnostic.py    # d<n recoverability (Matryoshka truncation)
# input_type A/B (reuses pantheons_mm images via MM_SKIP_FETCH): text re-embedded with search_query
MM_SKIP_FETCH=1 DATASET=pantheons_mm_sq uv run python pipeline/00c_build_multimodal.py
DATASET=pantheons_mm_sq uv run python pipeline/06_ablation.py       # then 08, 09 as above

# Toponymy-label metrics (run AFTER 07 for any ablation dataset; 07 now persists per-method labels.parquet).
DATASET=<ds> uv run python pipeline/11_label_metrics.py    # #1 per-region mixing depth curve + #3 seam -> label_metrics.{json,html}
DATASET=<ds> uv run python pipeline/11b_label_routing.py   # #4 label grounding / routing accuracy -> label_routing.json
DATASET=pantheons uv run python pipeline/07b_highd_labels.py  # #2 EVoC high-D labels (needs `evoc`); re-run 11 for the 2D-vs-high-D delta
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
- **00c_build_multimodal.py** — pantheons_mm / pantheons_mm_sq: fetch each figure's Wikipedia lead image
  (batched `pageimages` + continuation; backed-off downloads), embed via embed-v4.0 `input_type=image`,
  stack with text (input_type from the dataset config) → 2N points labeled by modality (+`pair_id`,
  +`pantheon`). Image URIs + embeddings cache under `data/pantheons/`, reused across variants
  (`MM_SKIP_FETCH=1` skips fetching entirely).
- **01_embed.py** — Cohere `embed-v4.0`, `input_type=clustering`, 1024-d → `embeddings.npz`.
- **02_reduce_umap.py** — UMAP → `umap_coords.npz` (n_neighbors=15, min_dist=0.05, cosine, seed 42).
- **03_label_topics.py / topic_labeling.py** — Toponymy + Opus-4.8 namer → `labels.parquet`.
- **04_visualize.py / mapviz.py** — datamapplot interactive HTML, **color = corpus**.
- **05_evaluate.py / metrics.py** — baseline go/no-go (cross-corpus NN aptness, mixing, recoverability).
- **06_ablation.py** — integrate raw/center/leace/harmony → UMAP → metrics → `ablation.json`
  + `experiments/<method>/`. `integrations.py` holds the methods.
- **07_ablation_maps.py** — render a map per integration method; **also persists per-method
  `labels.parquet`** (id + label_layer_0..N, finest first) via `save_labels`, for stages 11/11b.
- **07b_highd_labels.py** — **#2 high-D lens**: EVoC region labels (`labels_highd.parquet`) clustering the
  native 1024-d embeddings (vs 07's 2D-UMAP substrate) via `EVoCClusterer`. No map (membership only); needs
  the optional `evoc` package. Stage 11 reports the 2D-vs-high-D mixing delta when this file is present.
- **08_rank_diagnostic.py** — INLP rank-k sweep (how much corpus separation is linear vs irreducible).
- **09_multimodal_eval.py** — pantheons_mm: cross-modal retrieval (recall@1/MRR, image↔text) per method.
- **10_dim_diagnostic.py** — sweep output dim (Matryoshka prefix truncation of the 1024-d embeddings);
  reports mixing + recoverability per method, so recoverability is readable where d<n (pantheons_mm: 312 pts).
- **11_label_metrics.py** — Toponymy-label metrics from stage-07/07b labels: **#1** per-region cross-corpus
  mixing (Gini-Simpson, `metrics.per_region_mixing`, same (K−1)/K scale as kNN) + purity-vs-depth curve +
  coverage + named-region receipts; **#3** structural table + where unnamed space sits (the corpus seam, via
  `per_point_cross_corpus`). Writes `label_metrics.{json,html}` (plotly depth curve). **#2** 2D-vs-high-D
  delta when `labels_highd.parquet` exists. Always read mixing against coverage.
- **11b_label_routing.py** — **#4** label grounding: routing accuracy (in the ORIGINAL Cohere space, does
  each named doc's own region label win vs siblings?) + inter-label redundancy, per method/layer →
  `label_routing.json`. Certifies the receipts; does not rank methods (label quality is a namer property).

Artifacts are row-aligned by an `id` column; entities schema: `id, name, corpus, title, url,
text, char_len`. Writes are atomic (tmp + `os.replace`); see `running-data-pipelines`.

## Conventions & gotchas

- **Cohere embed-v4.0 is multimodal** (text + images, one shared space): embed images via
  `co.embed(images=[<base64 data URI>], input_type="image", ...)` — same 1024-d space as text.
  Text `input_type` mostly aliases: **clustering == search_document == classification** (cos 1.0); only
  **search_query** differs (cos ~0.97 on long text). `output_dimension` is **Matryoshka** (a renormalized
  prefix), so truncating 1024→256 == native 256-d (cos 1.0) — used for the d<n recoverability diagnostic.
- **Wikipedia lead images** (`prop=pageimages`): batched queries silently cap thumbnails per request —
  page through the `picontinue` token (~4 requests for ~200 titles). And `upload.wikimedia.org` throttles
  bursty downloads — space them (~5s) with backoff; the throttle clears on cooldown, isolated requests are fine.
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
- **EVoC clusterer** (`EVoCClusterer`, the 07b high-D lens) is defined in `toponymy/clustering.py` but
  inside a `try: import evoc` block — so it silently fails to bind until the optional `evoc` package is
  installed (`uv add evoc`); a direct import or `dir()` then shows nothing. **Not** a Toponymy-version
  limitation. `EVoCClusterer.fit` clusters the high-D `embedding_vectors` and ignores `clusterable_vectors`
  (so `label_regions(..., clusterer="evoc")` still passes the 2D coords; they're unused for clustering).

## Metrics (how to read)

1. **Cross-corpus kNN mixing** (primary) — fraction of each point's neighbors from a *different*
   corpus; ~0 = separate blobs, fully interleaved → (K−1)/K (0.5 for two corpuses, ~0.75 for four).
2. **Linear recoverability** — CV accuracy predicting corpus; only reliable when **d < n**
   (unreliable in the ~100-pt pilot where d=1024 ≫ n).
3. **Aptness vs KNOWN analogues** — top-1 cross-corpus NN, under cosine and CSLS.
4. **Per-region mixing** (label layer, stage 11) — Gini-Simpson corpus impurity within each NAMED region,
   size-weighted over named points; same (K−1)/K scale as kNN mixing. Always read against COVERAGE (a method
   can fake high mixing by naming only its blended core and leaving pure fringes Unlabelled). The per-layer
   sequence (finest→coarsest) is the **depth curve**.
5. **Label routing accuracy** (stage 11b) — does each named doc's own region label win in the original Cohere
   space, vs the 1/n_regions chance line? Certifies label grounding; does not rank methods.

## Findings so far (see memory for detail)

Rank-1 **LEACE ≈ per-corpus centering ≥ Harmony**; the corpus difference is a low-rank linear
signal. Same-medium (Greek/Norse) interleaves nearly fully; different-medium + scale (Marvel/DC)
half-merges, and the residual is **genuine franchise structure to keep**, not under-merging.
CSLS is subsumed by integration. Input text dominates: names contaminate matching.
**N-way (Greek/Norse/Egyptian/Hindu) generalizes this exactly**: the corpus signal is **rank-(K−1)**
(INLP recoverability collapses 1.00→0.005 at k=3; closed-form multiclass LEACE — erasing the (K−1)-dim
class-mean subspace — removes precisely it), again **LEACE ≈ centering ≥ Harmony**, regions name as
cross-pantheon archetypes (Hades↔Hel↔Anubis↔Yama), residual is genuine (a Vishnu-avatar pocket; Horus a hub).
**Multimodal (`pantheons_mm`, image × text) BREAKS the pattern**: the modality gap is largely
**nonlinear**, unlike the low-rank-linear corpus signal. Linear erasure (center/LEACE/INLP) kills modality
recoverability by rank ~2 but mixing plateaus at ~half-merged (~0.23 of 0.50) through k=64 — no linear
subspace interleaves the modalities; only Harmony (nonlinear) merges the cones (mixing ~0.42). On the map,
raw shows color-pure per-figure clumps (image vs text split); Harmony intermixes them. **Robust across three
checks**: full **N=156** (stronger, not weaker), **d=256** (Matryoshka-truncated so d<n → recoverability is
trustworthy, and raw stays 0.99: modality is genuinely linearly separable — a real mean-offset, not a d≫n
artifact), and **input_type** (`search_query` text gives the same gap). The gap = a real linear mean-offset
(centering kills recoverability) + nonlinear manifold structure (only Harmony interleaves) — linear
*separability* ≠ linear *interleavability*. Cross-modal **retrieval** is modest (~0.25) with small method
differences (the earlier N=94 "centering wins big" was small-N inflation; search_query helps slightly).

**Toponymy-label metrics (stages 07b/11/11b) measure the integration on Toponymy's named-region output**
(built to tie the analysis to Toponymy itself, for that audience). **#1 per-region mixing**
(`metrics.per_region_mixing` — Gini-Simpson corpus impurity within each named region, same (K−1)/K scale as
kNN mixing, read against coverage) reproduces `LEACE ≈ centering ≥/≈ Harmony` on the named regions with
cross-corpus archetype receipts, and the depth curve re-expresses the multimodal nonlinearity as scale (only
Harmony mixes at the fine layer). The **#2 EVoC high-D lens**, **#3** unnamed-space seam, and **#4** label
routing corroborate rather than add (integrated merges are real in native 1024-d, so the maps don't overstate
them — the one 2D inflation is raw multimodal, Δ+.13; labels grounded). Per-dataset detail in
`label_metrics.{json,html}` + `label_routing.json`.

## Status & next

Done: greek_norse (pilot), marvel_dc (+ marvel_dc_anon), pantheons (N=4; rank-(K−1)), pantheons_mm
(image × text; modality gap is nonlinear — confirmed across N=156 / d=256 / input_type, via
`pantheons_mm_sq` + `10_dim_diagnostic.py`). **Plus Toponymy-label metrics (stages 07b/11/11b): #1 per-region
mixing + depth curve, #2 EVoC high-D delta, #3 seam, #4 routing — all four datasets, all reproducing the
geometric findings on the named regions.** **Next:** write up + publish via GitHub Pages, possibly share to
the TutteInstitute/toponymy discussions (the label metrics tie the analysis to Toponymy for that audience).

## Env

Keys from the shell env: `CO_API_KEY` (Cohere), `ANTHROPIC_API_KEY` (Toponymy naming).
`data/` is git-ignored; `uv.lock` is committed.
