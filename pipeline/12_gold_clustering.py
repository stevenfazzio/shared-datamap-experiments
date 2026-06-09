"""Stage 12 — gold-category clustering (re-grounding in Fan et al.'s evaluation).

Fan et al. ("The Medium Is Not the Message", arXiv:2507.01234) measure deconfounding by
whether k-means clusters of the (erased) embeddings align with GOLD SEMANTIC categories
(Purity / ARI), NOT by how mixed the sources got. This stage ports that test: assign each
item a gold ARCHETYPE from a fixed taxonomy, then for each integration method cluster the
high-D integrated embeddings and score alignment with the gold archetype (what we WANT) vs
the corpus (the confounder we erase).

A `shuffle` destruction baseline (each point given another point's embedding) shows that
mixing + unrecoverability are gameable — destruction passes them — but archetype alignment is
NOT: only real erasure raises ARI/Purity against the gold semantic categories.

Cluster metrics are averaged over several k-means seeds (mean±std): single-seed k-means is
noisy at these modest ARIs and produced spurious method-orderings.

Gold labels are LLM-assigned (Opus) from a PRE-REGISTERED, per-dataset taxonomy, independent
of the embedding pipeline, and cached (gold_archetypes.parquet) with per-batch checkpointing.

Reads:  data/<ds>/{entities.parquet, experiments/<method>/embeddings.npz}
Writes: data/<ds>/{gold_archetypes.parquet (cache), gold_clustering.json}
Run: DATASET=pantheons uv run python pipeline/12_gold_clustering.py
"""

import json
import os

import numpy as np
import pandas as pd
from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL_NAMING, DATA_DIR, ENTITIES_PARQUET
from integrations import METHODS
from metrics import _unit, cross_corpus_mixing, linear_recoverability
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Pre-registered taxonomies (chosen a priori from domain knowledge, NOT by inspecting the
# embeddings/clusters). "Other" allowed for items that fit no major archetype.
DEITY_ARCHETYPES = [
    "Sky / Storm / Thunder", "Sun / Light", "War / Battle", "Love / Beauty / Desire",
    "Fertility / Agriculture / Harvest", "Death / Underworld", "Sea / Water / Rivers",
    "Wisdom / Knowledge / Craft", "Trickery / Messengers / Travel",
    "Creation / Primordial / Supreme", "Mother / Earth", "Fate / Justice / Kingship",
    "Magic / Healing / Medicine",
]
HERO_ARCHETYPES = [
    "Cosmic / Abstract Entity", "Magic / Mystic", "Tech / Genius Inventor",
    "Super-Soldier / Enhanced Human", "Mutant / Genetic Outsider", "Alien / Extraterrestrial",
    "Street-Level Vigilante / Martial Artist", "Speedster", "Aquatic / Atlantean",
    "Warrior / Demigod", "Archer / Marksman", "Antihero / Mercenary / Assassin",
    "Monster / Creature", "Crime Boss / Mastermind",
]
TAXONOMIES = {"marvel_dc": HERO_ARCHETYPES, "marvel_dc_anon": HERO_ARCHETYPES}
ARCHETYPES = TAXONOMIES.get(DATA_DIR.name, DEITY_ARCHETYPES)

GOLD_PARQUET = DATA_DIR / "gold_archetypes.parquet"
OUT_JSON = DATA_DIR / "gold_clustering.json"
BATCH = 50
SEEDS = list(range(10))


def _write_gold(df, done):
    part = df[df["id"].isin(done)][["id", "name", "corpus"]].copy()
    part["archetype"] = part["id"].map(done)
    tmp = str(GOLD_PARQUET) + ".tmp"
    part.to_parquet(tmp, index=False)
    os.replace(tmp, GOLD_PARQUET)


def _assign_gold(df):
    """LLM-assign one archetype per row, cached + checkpointed. Returns array aligned to df."""
    done = {}
    if GOLD_PARQUET.exists():
        c = pd.read_parquet(GOLD_PARQUET)
        done = dict(zip(c["id"].tolist(), c["archetype"].tolist()))
    todo = df[~df["id"].isin(done)].reset_index(drop=True)
    if len(todo):
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        options = "\n".join(f"- {a}" for a in ARCHETYPES)
        print(f"  labeling {len(todo)} items ({len(done)} cached) via {ANTHROPIC_MODEL_NAMING}")
        for start in range(0, len(todo), BATCH):
            chunk = todo.iloc[start : start + BATCH]
            cids = chunk["id"].tolist()
            listing = "\n".join(
                "{}. {} ({}): {}".format(j, r["name"], r["corpus"], str(r["text"])[:200].replace(chr(10), " "))
                for j, (_, r) in enumerate(chunk.iterrows())
            )
            prompt = (
                "Assign each figure its single best-fitting archetype from this fixed list "
                '(use the exact string; if truly none fits, use "Other"):\n'
                f"{options}\n\nFigures (numbered):\n{listing}\n\n"
                'Respond with ONLY a JSON object mapping each number (string) to one archetype string, '
                'e.g. {"0": "War / Battle"}. No prose, no code fences.'
            )
            text = None
            for attempt in range(3):
                try:
                    resp = client.messages.create(
                        model=ANTHROPIC_MODEL_NAMING,
                        max_tokens=3000,
                        messages=[{"role": "user", "content": prompt}],
                    )  # no temperature: Opus 4.8 rejects it
                    text = resp.content[0].text
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"    batch@{start} attempt {attempt + 1} failed: {e}")
            if text is None:
                raise RuntimeError(f"gold labeling failed at batch {start} (progress checkpointed)")
            obj = json.loads(text[text.index("{") : text.rindex("}") + 1])
            for j, a in obj.items():
                done[cids[int(j)]] = a if a in ARCHETYPES else "Other"
            _write_gold(df, done)  # checkpoint after each batch
            print(f"    labeled {min(start + BATCH, len(todo))}/{len(todo)}")
    else:
        print(f"  using cached gold labels ({GOLD_PARQUET.name})")
    return df["id"].map(done).fillna("Other").to_numpy()


def _purity(pred, gold):
    pred, gold = np.asarray(pred), np.asarray(gold)
    return sum(np.unique(gold[pred == c], return_counts=True)[1].max() for c in set(pred.tolist())) / len(gold)


def _load_emb(method, ids):
    d = np.load(DATA_DIR / "experiments" / method / "embeddings.npz", allow_pickle=True)
    pos = {i: k for k, i in enumerate(list(d["id"]))}
    return d["emb"][[pos[i] for i in ids]].astype(np.float64)


def _score(emb, gold, real, pantheon, k):
    """Cluster `emb` over SEEDS; return mean (+std on archetype-ARI) of the alignment metrics."""
    u = _unit(emb)
    arch, panth, pur, nmi = [], [], [], []
    for s in SEEDS:
        pred = KMeans(n_clusters=k, random_state=s, n_init=5).fit(u).labels_
        arch.append(adjusted_rand_score(gold[real], pred[real]))
        panth.append(adjusted_rand_score(pantheon, pred))
        pur.append(_purity(pred[real], gold[real]))
        nmi.append(normalized_mutual_info_score(gold[real], pred[real]))
    return {
        "ari_archetype": float(np.mean(arch)), "ari_archetype_std": float(np.std(arch)),
        "ari_pantheon": float(np.mean(panth)),
        "purity_archetype": float(np.mean(pur)), "nmi_archetype": float(np.mean(nmi)),
    }


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    ids = df["id"].tolist()
    pantheon = df["corpus"].to_numpy()
    gold = _assign_gold(df)

    real = gold != "Other"
    used = sorted(set(gold[real].tolist()))
    k = len(used)
    cov = float(real.mean())
    counts = pd.Series(gold).value_counts()
    print(f"\n{DATA_DIR.name}: n={len(df)}  gold archetypes k={k}  coverage(non-Other)={cov:.2f}  seeds={len(SEEDS)}")
    print("  counts:", counts.to_dict())

    raw = _load_emb("raw", ids)
    rng = np.random.default_rng(42)
    variants = {m: _load_emb(m, ids) for m in METHODS}
    variants["shuffle"] = raw[rng.permutation(len(raw))]  # destruction baseline

    rows = []
    for name, emb in variants.items():
        sc = _score(emb, gold, real, pantheon, k)
        sc.update(method=name, mixing=cross_corpus_mixing(emb, pantheon),
                  recoverability=linear_recoverability(emb, pantheon)[0])
        rows.append(sc)

    print(
        f"\n  {'method':9s} {'mix':>5s} {'recov':>6s} | {'ARI:arch':>13s} {'ARI:panth':>10s} "
        f"{'Pur:arch':>9s} {'NMI:arch':>9s}   (gold=archetype, confounder=corpus)"
    )
    print("  " + "-" * 82)
    for r in rows:
        print(
            f"  {r['method']:9s} {r['mixing']:5.2f} {r['recoverability']:6.2f} | "
            f"{r['ari_archetype']:7.3f}±{r['ari_archetype_std']:<5.2f} {r['ari_pantheon']:10.3f} "
            f"{r['purity_archetype']:9.3f} {r['nmi_archetype']:9.3f}"
        )

    out = {
        "dataset": DATA_DIR.name, "n": len(df), "k": k, "n_seeds": len(SEEDS),
        "coverage_non_other": cov, "archetypes_used": used,
        "archetype_counts": {str(a): int(c) for a, c in counts.items()}, "methods": rows,
    }
    tmp = str(OUT_JSON) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, OUT_JSON)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
