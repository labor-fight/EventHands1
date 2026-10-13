#!/usr/bin/env python3
"""S43: S39-stack RA of arms under the protocol's init-noise draw (seed 0, the recorded evaluation) and seeds 1..10.

    python tools/s43/initseed_table.py --arms dt_dz_l3 s43n_sp --ref dt_dz_l3 --out outputs/s43/reports/initseed.json
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEEDS = (3407, 3408, 3409)
STACK = "af9f41a0_rn0.015-0.03-0.06-0.09-0.172185-0.261517-0.404576-0.5-0.8_fn0.04-0.08-0.16-0.24-0.4-0.56-0.8_tn0.025-0.05-0.1-0.15-0.25-0.35-0.5_w200_clip.json"


def ra(path, part="overall"):
    d = json.loads(Path(path).read_text())["model"]
    v = d["overall"]["mpjpe_ra_mm"] if part == "overall" else d[part]["mpjpe_ra_mm"]
    return v["mean"] if isinstance(v, dict) else (v[0] if isinstance(v, list) else v)


def seed0(run):
    for root in (REPO / "outputs/s43/full" / run, REPO / "outputs/trk6/s39_repro" / run):
        if (root / STACK).exists():
            return root / STACK
    raise FileNotFoundError(run)


def table(arm, part):
    out = {}
    for s in SEEDS:
        run = f"{arm}_s{s}"
        vals = {0: ra(seed0(run), part)}
        for p in glob.glob(str(REPO / "outputs/s43/initseed" / run / "off_init*.json")):
            k = int(Path(p).stem.split("init")[1])
            vals[k] = ra(p, part)
        out[s] = vals
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = {}
    for part in ("overall", "zgz_global", "zgz_local"):
        t = {arm: table(arm, part) for arm in a.arms}
        common = sorted(set.intersection(*[set(t[arm][s]) for arm in a.arms for s in SEEDS]))
        rep[part] = {"init_seeds": common}
        for arm in a.arms:
            m = np.array([[t[arm][s][k] for k in common] for s in SEEDS])           # (3 model seeds, n init seeds)
            rep[part][arm] = {"mean": float(m.mean()), "per_model_seed": m.mean(1).tolist(),
                              "sd_over_init_seeds": m.std(1, ddof=1).tolist() if len(common) > 1 else None,
                              "seed0": m[:, common.index(0)].tolist() if 0 in common else None}
            if arm != a.ref:
                r = np.array([[t[a.ref][s][k] for k in common] for s in SEEDS])
                d = m - r
                rep[part][arm]["paired_vs_ref"] = {"mean": float(d.mean()), "per_model_seed": d.mean(1).tolist(),
                                                   "frac_negative_cells": float((d < 0).mean()), "cells": int(d.size)}
        print(f"== {part}  init seeds {common}")
        for arm in a.arms:
            e = rep[part][arm]
            pv = e.get("paired_vs_ref")
            print(f"{arm:16s} mean {e['mean']:.3f}  per model seed " + " / ".join(f"{x:.3f}" for x in e["per_model_seed"])
                  + (f"  sd(init) " + " / ".join(f"{x:.3f}" for x in e["sd_over_init_seeds"]) if e["sd_over_init_seeds"] else "")
                  + (f"  vs {a.ref}: {pv['mean']:+.3f} (" + " / ".join(f"{x:+.3f}" for x in pv["per_model_seed"])
                     + f"), negative in {pv['frac_negative_cells']:.0%} of {pv['cells']} cells" if pv else ""))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
