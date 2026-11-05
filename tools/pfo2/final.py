#!/usr/bin/env python3
"""PFO2 final rows and the comparison with S43 (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103 acceptance and
delivery); CPU only.

    python -m tools.pfo2.final row --name N --readings TAG[,TAG...] --cost COSTTAG [--label L]
        -> outputs/semkine/pfo2_<N>_main_row.json in tools/report_table.py's format: accuracy = the mean over the given
           E1 readings (one per seed, step 6000; a tag "pfo:<tag>" reads PFO's frozen readings), cost = the given
           same-block cost reading (PyTorch default flags), FLOPs = executed Linear / Conv MACs median (+ the PFO2 extra
           functional MACs of tools/pfo2/cost.py), Params incl. the router
    python -m tools.pfo2.final compare --cand N --ref-cost COSTTAG [--out FILE]
        -> the five three-seed means against S43's engine re-computation (12.81 / 8.02 / 10.77 / 6.64 / 10.24, the mean of
           PFO's s43_s340{7,8,9} E1 readings), the positive-gap sum, seeds below S43 per metric, and the cost ratios against
           the S43 reading of the same block (params, MACs, row latency, end-to-end p50 / p99; <= 1.5 near, <= 2.0 same order)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo2.common import OUT, PFO_OUT, S43_SEEDS, ZGZ, now, read, require, save  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
COLS = (("local", "mpjpe_ra_mm"), ("global", "mpjpe_ra_mm"), ("local", "mpvpe_ra_mm"), ("global", "mpvpe_ra_mm"),
        ("overall", "mpjpe_ra_mm"))
NEAR, SAME = 1.5, 2.0


def reading(tag):
    root = PFO_OUT if tag.startswith("pfo:") else OUT
    r = read(root / "readings" / tag.split(":", 1)[-1] / "e1" / "report.json")
    s = r["per_sequence"]
    return {"local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
            "overall": {k: r["overall"][k] for k in KEYS}}, r


def cost(tag):
    c = read(OUT / "costs" / tag / "report.json")
    extra = OUT / "costs" / tag / "pfo2_extra.json"
    c["macs_total_median"] = c["macs_layers"]["median"] + (read(extra)["extra_functional_macs"] if extra.exists() else 0)
    return c


def row(name, tags, cost_tag, label=None):
    per = {}
    for i, t in enumerate(tags):
        blk, r = reading(t)
        per[str(i)] = {"reading": t, "checkpoint": r["identity"]["checkpoint"], **blk}
    mean = {b: {k: float(np.mean([per[s][b][k] for s in per])) for k in KEYS} for b in ("local", "global", "overall")}
    c = cost(cost_tag)
    require(c["policy"] == "default", "the table's cost comes from the PyTorch-default-flag reading")
    out = {"arm": name, "label": label, "split": "val_core", "n_seeds": len(tags), "two_seed_mean": mean, "per_seed": per,
           "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_total_median"],
           "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": cost_tag,
           "written_utc": now(),
           "note": "PFO2: PFO's engine E1 (deterministic flags) and cost harness (same exclusive block, PyTorch default "
                   "flags, lyq_local 600 packets); 'two_seed_mean' is the N-seed mean"}
    p = SEMKINE / f"pfo2_{name}_main_row.json"
    save(p, out)
    print("wrote", p)
    return p


def s43_reference():
    rows = []
    for s in S43_SEEDS:
        blk, _ = reading(f"pfo:s43_s{s}")
        rows.append([blk[b][k] for b, k in COLS])
    return np.mean(rows, 0)


def compare(cand, ref_cost):
    a = read(SEMKINE / f"pfo2_{cand}_main_row.json")
    ref = s43_reference()
    acc = []
    for (blk, k), rm in zip(COLS, ref):
        cm = a["two_seed_mean"][blk][k]
        acc.append({"metric": f"{blk}/{k}", "cand": cm, "s43": float(rm), "diff": cm - float(rm), "below": cm < rm,
                    "seeds_below": int(sum(a["per_seed"][s][blk][k] < rm for s in a["per_seed"])),
                    "n_seeds": len(a["per_seed"])})
    ca, cb = cost(a["cost_reading"]), cost(ref_cost)
    ratios = {"params": ca["params_total"] / cb["params_total"],
              "macs": ca["macs_total_median"] / cb["macs_layers"]["median"],
              "latency_row": ca["latency_row"]["scaled_ms"] / cb["latency_row"]["scaled_ms"],
              "latency_e2e_p50": ca["latency_e2e_ms"]["p50"] / cb["latency_e2e_ms"]["p50"],
              "latency_e2e_p99": ca["latency_e2e_ms"]["p99"] / cb["latency_e2e_ms"]["p99"],
              "latency_e2e_mean": ca["latency_e2e_ms"]["mean"] / cb["latency_e2e_ms"]["mean"]}
    budget = {k: ("near" if v <= NEAR else "same_order" if v <= SAME else "over") for k, v in ratios.items()}
    met = all(x["below"] for x in acc)
    return {"cand": cand, "ref_cost": ref_cost, "s43_engine": ref.tolist(), "accuracy": acc,
            "gap_sum": float(sum(max(0.0, x["diff"]) for x in acc)), "accuracy_met": met,
            "accuracy_strong": met and all(3 * x["seeds_below"] >= 2 * x["n_seeds"] for x in acc),
            "cost_ratios": ratios, "budget": budget, "budget_within_2x": all(v != "over" for v in budget.values())}


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
    c.add_argument("--ref-cost", required=True)
    c.add_argument("--out")
    a = p.parse_args(argv)
    if a.verb == "row":
        row(a.name, a.readings.split(","), a.cost, a.label)
    else:
        res = compare(a.cand, a.ref_cost)
        print(json.dumps(res, indent=1))
        if a.out:
            save(a.out, res)


if __name__ == "__main__":
    main()
