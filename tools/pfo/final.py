#!/usr/bin/env python3
"""PFO final rows and comparison (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103 success criteria and budget); CPU only.

    python -m tools.pfo.final row --name N --readings TAG[,TAG...] --cost COSTDIR [--label L]
        -> outputs/semkine/pfo_<N>_main_row.json in tools/report_table.py's format: accuracy = the mean over the given
           engine readings (one per seed, step 6000, E1), cost = the given same-block cost reading (PyTorch default flags;
           Latency column = row form x 1.75 / anchor, FLOPs = executed MACs median, Params incl. the router)
    python -m tools.pfo.final compare --cand N --ref pfo_s43 [--out FILE]
        -> the written success criteria (all five 3-seed means below S43's; >= 2 / 3 seeds each below S43's mean) and the
           budget ratios (params, MACs, row latency, end-to-end p50 / p99; <= 1.5 near, <= 2.0 same order)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.common import OUT, ZGZ, read, require, save  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
NEAR, SAME = 1.5, 2.0


def reading(tag):
    r = read(OUT / "readings" / tag / "e1" / "report.json")
    s = r["per_sequence"]
    blk = {"local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
           "overall": {k: r["overall"][k] for k in KEYS}}
    return blk, r


def row(name, tags, cost, label=None):
    per = {}
    for i, t in enumerate(tags):
        blk, r = reading(t)
        per[str(i)] = {"reading": t, "checkpoint": r["identity"]["checkpoint"], **blk}
    mean = {b: {k: float(np.mean([per[s][b][k] for s in per])) for k in KEYS} for b in ("local", "global", "overall")}
    c = read(OUT / "costs" / cost / "report.json")
    require(c["policy"] == "default", "the table's cost comes from the PyTorch-default-flag reading")
    out = {"arm": name, "label": label, "split": "val_core", "n_seeds": len(tags), "two_seed_mean": mean, "per_seed": per,
           "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": cost,
           "note": "PFO unified harness (tools/pfo/engine.py E1, deterministic flags; tools/pfo/cost.py same block, "
                   "PyTorch default flags, lyq_local 600 packets); 'two_seed_mean' is the N-seed mean"}
    p = SEMKINE / f"pfo_{name}_main_row.json"
    save(p, out)
    print("wrote", p)
    return p


def compare(cand, ref):
    a = read(SEMKINE / f"pfo_{cand}_main_row.json")
    b = read(SEMKINE / f"pfo_{ref}_main_row.json")
    cols = (("local", "mpjpe_ra_mm"), ("global", "mpjpe_ra_mm"), ("local", "mpvpe_ra_mm"), ("global", "mpvpe_ra_mm"),
            ("overall", "mpjpe_ra_mm"))
    acc = []
    for blk, k in cols:
        cm, rm = a["two_seed_mean"][blk][k], b["two_seed_mean"][blk][k]
        seeds_below = sum(a["per_seed"][s][blk][k] < rm for s in a["per_seed"])
        acc.append({"metric": f"{blk}/{k}", "cand": cm, "ref": rm, "diff": cm - rm, "below": cm < rm,
                    "seeds_below": seeds_below, "n_seeds": len(a["per_seed"])})
    ca, cb = read(OUT / "costs" / a["cost_reading"] / "report.json"), read(OUT / "costs" / b["cost_reading"] / "report.json")
    ratios = {"params": ca["params_total"] / cb["params_total"],
              "macs": ca["macs_layers"]["median"] / cb["macs_layers"]["median"],
              "latency_row": ca["latency_row"]["scaled_ms"] / cb["latency_row"]["scaled_ms"],
              "latency_e2e_p50": ca["latency_e2e_ms"]["p50"] / cb["latency_e2e_ms"]["p50"],
              "latency_e2e_p99": ca["latency_e2e_ms"]["p99"] / cb["latency_e2e_ms"]["p99"]}
    budget = {k: ("near" if v <= NEAR else "same_order" if v <= SAME else "over") for k, v in ratios.items()}
    passed = all(x["below"] for x in acc)
    strong = passed and all(x["seeds_below"] * 3 >= 2 * x["n_seeds"] for x in acc)
    within = all(v != "over" for v in budget.values())
    res = {"cand": cand, "ref": ref, "accuracy": acc, "cost_ratios": ratios, "budget": budget,
           "accuracy_pass": passed, "accuracy_strong": strong, "budget_within": within,
           "goal_met": passed and within}
    return res


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("row")
    r.add_argument("--name", required=True)
    r.add_argument("--readings", required=True)
    r.add_argument("--cost", required=True)
    r.add_argument("--label")
    c = sub.add_parser("compare")
    c.add_argument("--cand", required=True)
    c.add_argument("--ref", default="s43")
    c.add_argument("--out")
    a = p.parse_args(argv)
    if a.verb == "row":
        row(a.name, a.readings.split(","), a.cost, a.label)
    else:
        res = compare(a.cand, a.ref)
        print(json.dumps(res, indent=1))
        if a.out:
            save(a.out, res)


if __name__ == "__main__":
    main()
