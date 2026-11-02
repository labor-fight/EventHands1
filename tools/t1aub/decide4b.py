#!/usr/bin/env python3
"""T1A-UB round-4 amendment 1 (docs/EXPERIMENT_RECORDS.md#t1aub-r4-amend1-20261102): the adoption of training-side
mechanisms by two paths, after tools/t1aub/decide4.py (unchanged) has written summary_r4.json.

    python tools/t1aub/decide4b.py --block B4 --ref-block B1     -> outputs/t1aub/20261102/summary_r4_amend1.json

Path 1 = decide4's HELPS within budget. Path 2 = the XCL single-seed zgz rule on local against candidate B (E1 local at step
6000 better by >= 5 mm, E1 / E2 / E3 same sign, at least two of the three late checkpoints in the same direction) with C <= 0.8,
in-sample not worse by >= 2.5 mm at all three checkpoints, and within budget. Adopted = the union of the two paths.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402
from tools.t1aub import decide as D  # noqa: E402
from tools.t1aub import decide4 as D4  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--block", required=True)
    p.add_argument("--ref-block", required=True)
    a = p.parse_args(argv)
    C.verify_protocol()
    s4 = C.read(C.OUT / "summary_r4.json")
    base = D.readings(D4.BASE, a.ref_block)
    out = {"amendment": "t1aub-r4-amend1-20261102", "paths": {}, "adopted": []}
    for mech, net in D4.MECHANISMS.items():
        m = s4["mechanisms"][mech]
        r = D.readings(net, a.block)
        support, d1 = D.zgz_support(base, r, "local")
        gains = D.diffs(r, base)
        worse_all = all(g <= -D.TH["insample"] for g in gains.values())
        p1 = m["class"] == "HELPS" and m["budget_ok"]
        p2 = bool(support and d1 > 0 and r["fixed_point"]["C"] <= D.TH["C"] and not worse_all and m["budget_ok"])
        out["paths"][mech] = {"network": net, "path1_insample": p1, "path2_zgz": p2, "zgz_e1_local_gain_6000": d1,
                              "class_decide4": m["class"]}
        if p1 or p2:
            out["adopted"].append(mech)
    out["computed_at_utc"] = C.now()
    C.save(C.OUT / "summary_r4_amend1.json", out, exclusive=True)
    print(out["adopted"])


if __name__ == "__main__":
    main()
