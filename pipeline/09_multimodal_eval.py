"""Stage 09 — multimodal readout (pantheons_mm): cross-modal retrieval per method.

For each integration method, load the integrated embedding stage 06 saved and measure paired
cross-modal retrieval (does an image retrieve its own text, and vice versa?). Combines with
06's mixing/recoverability into one table — the headline modality-gap result: erasing the
{image, text} split should merge the cones (mixing up) while retrieval holds or improves.

Reads:  data/pantheons_mm/{entities.parquet, ablation.json, experiments/<method>/embeddings.npz}
Writes: data/pantheons_mm/multimodal_eval.json   (+ printed table)

Run with DATASET=pantheons_mm.
"""

import json

import numpy as np
import pandas as pd
from config import DATA_DIR, ENTITIES_PARQUET
from integrations import METHODS
from metrics import cross_modal_retrieval

EXP_DIR = DATA_DIR / "experiments"


def main():
    df = pd.read_parquet(ENTITIES_PARQUET).reset_index(drop=True)
    modality = df["corpus"].to_numpy()
    pair_id = df["pair_id"].to_numpy()

    with open(DATA_DIR / "ablation.json") as f:
        abl = {m["method"]: m for m in json.load(f)["methods"]}

    rows = []
    for method in METHODS:
        d = np.load(EXP_DIR / method / "embeddings.npz", allow_pickle=True)
        pos = {i: k for k, i in enumerate(list(d["id"]))}
        emb = d["emb"][[pos[i] for i in df["id"]]].astype(np.float64)
        ret = cross_modal_retrieval(emb, modality, pair_id)
        i2t, t2i = ret["image->text"], ret["text->image"]
        rows.append(
            {
                "method": method,
                "mixing": abl.get(method, {}).get("mixing"),
                "recoverability": abl.get(method, {}).get("recoverability"),
                "i2t_recall@1": i2t["recall@1"],
                "i2t_mrr": i2t["mrr"],
                "t2i_recall@1": t2i["recall@1"],
                "t2i_mrr": t2i["mrr"],
            }
        )

    print("\n============== MULTIMODAL (image x text) ==============")
    print("modality mixing: 0=two cones, ~0.5=merged   |   retrieval: image<->text recall@1 / MRR\n")
    hdr = f"{'method':9s} {'mixing':>7s} {'recover':>8s} {'i2t@1':>7s} {'i2t mrr':>8s} {'t2i@1':>7s} {'t2i mrr':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['method']:9s} {r['mixing']:7.3f} {r['recoverability']:8.3f} "
            f"{r['i2t_recall@1']:7.2f} {r['i2t_mrr']:8.2f} {r['t2i_recall@1']:7.2f} {r['t2i_mrr']:8.2f}"
        )
    print("=" * len(hdr))

    out = DATA_DIR / "multimodal_eval.json"
    with open(out, "w") as f:
        json.dump({"methods": rows}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
