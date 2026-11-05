#!/usr/bin/env python3
"""ACU stage-3 written decisions (docs/EXPERIMENT_RECORDS.md#acu-s3-prereg-20261104); CPU, read-only, each run once.

    python -m tools.acu.decide_s3 trunk      MeshGNN check on the 2x2 winner W (seed 3407): NOMP if W with nomp has E1 RA
                                             <= W with gshare + 0.3 mm (the cheaper trunk is kept), else GSHARE
    python -m tools.acu.decide_s3 final      the final configuration's three-seed means against S43's engine numbers
                                             (TARGET_MET iff all five are lower) and the cost ratios to S43 from the cost
                                             block (params, MACs, row latency, end-to-end p50; <= 1.5 near, <= 2.0 same
                                             order, > 2.0 over), both flag policies
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.acu import arms_s2 as A  # noqa: E402
from tools.acu.common import OUT, S43_ENGINE, now, read, require, save  # noqa: E402
from tools.acu.decide import summary  # noqa: E402

NOMP_TOLERANCE_MM = 0.3


def winner():
    return read(OUT / "decisions" / "s2.json")["decisions"]["winner"]


def trunk():
    out = OUT / "decisions" / "s3_trunk.json"
    require(not out.exists(), "decided once")
    w = winner()
    g = summary(OUT / "readings" / A.run_name(w, 3407))
    n = summary(OUT / "readings" / A.run_name(w, 3407, trunk="nomp"))
    dec = "NOMP" if n["five"][4] <= g["five"][4] + NOMP_TOLERANCE_MM else "GSHARE"
    save(out, {"winner": w, "gshare": g, "nomp": n, "tolerance_mm": NOMP_TOLERANCE_MM, "decision": dec,
               "decided_at_utc": now()}, exclusive=True)
    print(w, dec, g["five"][4], n["five"][4], flush=True)


def chosen():
    d = read(OUT / "decisions" / "s3_trunk.json")
    return d["winner"], ("nomp" if d["decision"] == "NOMP" else "gshare")


def ratio_class(r):
    return "near" if r <= 1.5 else ("same_order" if r <= 2.0 else "over")


def final():
    out = OUT / "decisions" / "s3_final.json"
    require(not out.exists(), "decided once")
    w, t = chosen()
    seeds = {s: summary(OUT / "readings" / A.run_name(w, s, trunk=t)) for s in A.SEEDS}
    five = np.mean([seeds[s]["five"] for s in A.SEEDS], 0).tolist()
    met = all(f < s for f, s in zip(five, S43_ENGINE))
    cost = {}
    for policy in ("default", "det"):
        mine = read(OUT / "costs" / f"s3__{A.run_name(w, 3407, trunk=t)}__{policy}" / "report.json")
        s43 = read(OUT / "costs" / f"ref__s43__exec__{policy}" / "report.json")
        b2 = read(OUT / "costs" / f"ref__b2__fast__{policy}" / "report.json")
        rows = {}
        for key, f in (("params", lambda r: r["params_total"]), ("macs", lambda r: r["macs_layers"]["mean"]),
                       ("latency_row", lambda r: r["latency_row"]["scaled_ms"]), ("latency_e2e_p50", lambda r: r["latency_e2e_ms"]["p50"])):
            rr = f(mine) / f(s43)
            rows[key] = {"acu": f(mine), "s43": f(s43), "b2": f(b2), "ratio_to_s43": rr, "class": ratio_class(rr)}
        cost[policy] = rows
    save(out, {"winner": w, "trunk": t, "per_seed": seeds, "three_seed_five": five, "s43_engine": S43_ENGINE,
               "target": "TARGET_MET" if met else "TARGET_NOT_MET", "cost": cost, "decided_at_utc": now()}, exclusive=True)
    print(w, t, five, "TARGET_MET" if met else "TARGET_NOT_MET", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("trunk", "final"))
    a = p.parse_args(argv)
    trunk() if a.verb == "trunk" else final()


if __name__ == "__main__":
    main()
