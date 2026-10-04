#!/usr/bin/env python3
"""DT2: the fixed filters applied to finished runs, as one markdown table (means over the seeds that have all three files).

Per run: bare tracker = outputs/semkine/<run>/evalx_val_core_last_tf_pert.json; constant filter (0.5, 1.0, 0.5) and the
recommended AdaptiveFilter (spec chosen on dt_dz_l3 by the two-fold zgz sweep, outputs/dt2/filter_sweep/recommended_spec.json)
from outputs/dt2/filter_sweep/<run>/ (dt_dz_l3) or outputs/dt2/filter_apply/<run>/ (every other arm; the spec is applied
unchanged, not re-selected). Columns: RA overall / global / local (mm), root-aligned acc_err (mm/step^2), root-aligned
acceleration ratio, root speed ratio global / local, 10 deg root perturbation retention after one step.

    python tools/dt/dt2_filter_table.py --arms dt_dz_l3 dt_dz_l3w128 --seeds 3407 3408 3409
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SEM = REPO / "outputs/semkine"
TAGS = {"bare": None, "const 0.5/1/0.5": "r0.5_f1.0_t0.5", "adaptive": "afad1f21_rn0.3-0.5-0.8_f0.8_t0.5"}
SEQS = ("zgz_global", "zgz_local")


W = 50                                                   # evidence window of the evaluations to tabulate (--w)


def path(run: str, tag):
    suf = "" if W == 50 else f"_w{W}"
    if tag is None:                                      # evalx tags a non-default window itself, then our --suffix
        return SEM / run / ("evalx_val_core_last_tf_pert.json" if W == 50 else f"evalx_val_core_last_tf_pert_wfixed{W}_w{W}.json")
    for d in (REPO / "outputs/dt2/filter_sweep", REPO / "outputs/dt2/filter_apply", REPO / "outputs/dt2/wscan"):
        if (d / run / f"{tag}{suf}.json").exists():
            return d / run / f"{tag}{suf}.json"
    return None


def metrics(d: dict) -> dict:
    m = d["model"]
    j = m["jitter"]
    k1 = d.get("perturb", {}).get("10", {}).get("retention_k1_k2_k5_k10_k20", [float("nan")])[0]
    return {"RA": m["overall"]["mpjpe_ra_mm"], "RA_g": m["zgz_global"]["mpjpe_ra_mm"][0],
            "RA_l": m["zgz_local"]["mpjpe_ra_mm"][0], "acc_ra": j.get("acc_err_ra_mm", float("nan")),
            "accr_ra": j.get("acc_ratio_ra", float("nan")),
            "rsr_g": m["zgz_global"]["motion"]["root_speed_ratio"], "rsr_l": m["zgz_local"]["motion"]["root_speed_ratio"],
            "k1": k1}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[3407, 3408, 3409])
    ap.add_argument("--w", type=int, default=50, help="evidence window (ms) of the evaluations: 50 = the recorded rows")
    a = ap.parse_args()
    global W
    W = a.w
    print(f"evidence window {W} ms\n")
    print("| arm | seeds | filter | RA | RA g / l | acc_err_ra | acc ratio ra | root speed g / l | k1 (10 deg) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in a.arms:
        runs = [f"{arm}_s{s}" for s in a.seeds]
        runs = [r for r in runs if all(path(r, t) is not None and path(r, t).exists() for t in TAGS.values())]
        if not runs:
            print(f"| {arm} | - | | | | | | | |")
            continue
        seeds = ",".join(r.rsplit("_s", 1)[1] for r in runs)
        for name, tag in TAGS.items():
            ms = [metrics(json.loads(path(r, tag).read_text())) for r in runs]
            M = {k: sum(x[k] for x in ms) / len(ms) for k in ms[0]}
            print(f"| {arm} | {seeds} | {name} | {M['RA']:.3f} | {M['RA_g']:.2f} / {M['RA_l']:.2f} | {M['acc_ra']:.2f} | "
                  f"{M['accr_ra']:.2f} | {M['rsr_g']:.2f} / {M['rsr_l']:.2f} | {M['k1']:.2f} |")


if __name__ == "__main__":
    main()
