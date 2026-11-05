#!/usr/bin/env python3
"""EVF main rows and the comparison with S43 (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104 deliverables 1 and 4);
CPU only, read-only on the readings and the final cost block.

    python -m tools.evf.final row --name N --readings ROOT:TAG[,ROOT:TAG ...] --cost COSTTAG [--label L]
        -> outputs/semkine/evf_<N>_main_row.json in tools/report_table.py's format: accuracy = the mean over the given E1
           readings (ROOT = evf | pfo2 | pfo: the readings of that study; one per seed), cost = the EVF final block's
           reading COSTTAG (outputs/evf/20261104/costs/, PyTorch default flags), FLOPs = executed Linear / Conv MACs
           median (+ the router for routed networks), Params incl. the router
    python -m tools.evf.final compare --name N --ref-cost COSTTAG
        -> the five means against S43's engine re-computation (12.81 / 8.02 / 10.77 / 6.64 / 10.24), the positive-gap sum
           G, and the cost ratios against the S43 reading of the same block (params, MACs median, row latency, end-to-end
           p50 / p99; <= 1.5 near, <= 2.0 same order), plus the full-load MACs when the reading has them
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.evf.common import OUT, S43_ENGINE, now, read, require, save  # noqa: E402
from tools.evf.mech import ROOTS  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
COLS = (("local", "mpjpe_ra_mm"), ("global", "mpjpe_ra_mm"), ("local", "mpvpe_ra_mm"), ("global", "mpvpe_ra_mm"),
        ("overall", "mpjpe_ra_mm"))
NEAR, SAME = 1.5, 2.0


def reading(item):
    root, tag = item.split(":", 1)
    r = read(ROOTS[root] / "readings" / tag / "e1" / "report.json")
    s = r["per_sequence"]
    return {"local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
            "overall": {k: r["overall"][k] for k in KEYS}}, r


def cost(tag):
    return read(OUT / "costs" / tag / "report.json")


def row(name, items, cost_tag, label=None):
    per = {}
    for i, t in enumerate(items):
        blk, r = reading(t)
        per[str(i)] = {"reading": t, "checkpoint": r["identity"]["checkpoint"], **blk}
    mean = {b: {k: float(np.mean([per[s][b][k] for s in per])) for k in KEYS} for b in ("local", "global", "overall")}
    c = cost(cost_tag)
    require(c["policy"] == "default", "the table's cost comes from the PyTorch-default-flag reading")
    out = {"arm": name, "label": label, "split": "val_core", "n_seeds": len(items), "two_seed_mean": mean,
           "per_seed": per, "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "macs_full_load": c.get("macs_full_load"), "params_total": c["params_total"],
           "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": cost_tag, "written_utc": now(),
           "note": "EVF: PFO's engine E1 (deterministic flags); cost from the EVF final block (GPU 4 exclusive, PyTorch "
                   "default flags, lyq_local 600 packets); 'two_seed_mean' is the N-seed mean"}
    p = SEMKINE / f"evf_{name}_main_row.json"
    save(p, out)
    print("wrote", p)
    return p


def compare(name, ref_cost):
    a = read(SEMKINE / f"evf_{name}_main_row.json")
    five = [a["two_seed_mean"][b][k] for b, k in COLS]
    ca, cb = cost(a["cost_reading"]), cost(ref_cost)
    ratios = {"params": ca["params_total"] / cb["params_total"],
              "macs_median": ca["macs_layers"]["median"] / cb["macs_layers"]["median"],
              "latency_row": ca["latency_row"]["scaled_ms"] / cb["latency_row"]["scaled_ms"],
              "latency_e2e_p50": ca["latency_e2e_ms"]["p50"] / cb["latency_e2e_ms"]["p50"],
              "latency_e2e_p99": ca["latency_e2e_ms"]["p99"] / cb["latency_e2e_ms"]["p99"]}
    if ca.get("macs_full_load"):
        ratios["macs_full_load"] = ca["macs_full_load"] / cb["macs_layers"]["median"]
    budget = {k: ("near" if v <= NEAR else "same_order" if v <= SAME else "over") for k, v in ratios.items()}
    return {"name": name, "five": five, "s43_engine": list(S43_ENGINE),
            "diff": [x - y for x, y in zip(five, S43_ENGINE)], "G": float(sum(max(0.0, x - y) for x, y in zip(five, S43_ENGINE))),
            "all_below_s43": all(x < y for x, y in zip(five, S43_ENGINE)), "n_seeds": a["n_seeds"], "ref_cost": ref_cost,
            "cost_ratios": ratios, "budget": budget}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("row")
    r.add_argument("--name", required=True)
    r.add_argument("--readings", required=True)
    r.add_argument("--cost", required=True)
    r.add_argument("--label")
    c = sub.add_parser("compare")
    c.add_argument("--name", required=True)
    c.add_argument("--ref-cost", required=True)
    c.add_argument("--out")
    a = p.parse_args(argv)
    if a.verb == "row":
        row(a.name, a.readings.split(","), a.cost, a.label)
    else:
        res = compare(a.name, a.ref_cost)
        print(json.dumps(res, indent=1))
        if a.out:
            save(a.out, res)


if __name__ == "__main__":
    main()
