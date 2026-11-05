#!/usr/bin/env python3
"""PFO2 cost readings (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103 acceptance): PFO's frozen cost harness
(tools/pfo/cost.py: the first 600 lyq_local packets, parameters incl. the router, executed Linear / Conv MACs on the
actual tensors + the router's dense trunk once, MANO FK apart, row-form and end-to-end latency, memory) with its
output root redirected to outputs/pfo2/20261103/costs/<tag>/; run inside an exclusive block (GPU 4 alone).

    python -m tools.pfo2.cost run --net s43 --impl exec|dense --policy default|det
    python -m tools.pfo2.cost run --net pfo --arm f1 --seed 3407 --policy ...        PFO's F (step 6000)
    python -m tools.pfo2.cost run --net pfo2 --arm A --seed 3407 --policy ...        an accepted PFO2 arm (stage 0 / 1)
Tag final__<name>__<impl>__<policy>. Functional multiply-adds outside Linear / Conv that a PFO2 change adds (the disp
head's LBS Jacobian and least squares) are counted analytically in `extra_functional_macs` (pfo2 arms only).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2.common import OUT, read, require, save  # noqa: E402

V, NJ, NS = 778, 16, 51


def disp_extra_macs():
    """Analytic multiply-adds of one disp step outside Linear layers (semkine/pfo2_trunk.py lbs_jacobian + least_squares)."""
    jac = (V * 3 * 10 + NJ * V * 3 + V * 3 * 135 +          # shape blend, joints, pose blend
           NJ * V * 9 + NJ * V * 3 +                          # per-joint carried vertices x_vj
           NJ * NJ * V * 3 + NJ * NJ * V +                    # subtree sums S, W_k
           NJ * 9 * 3 + NJ * 3 * V * 6)                       # omega, cross products
    ls = (V * 3) * NS * NS + (V * 3) * NS + NS ** 3 // 3 + NS  # J^T J, J^T u, the solve, the scaling
    return int(jac + ls)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--net", required=True, choices=("s43", "pfo", "pfo2"))
    r.add_argument("--arm")
    r.add_argument("--seed", type=int, default=3407)
    r.add_argument("--impl", default="fast", choices=("fast", "exec", "dense"))
    r.add_argument("--policy", required=True, choices=("default", "det"))
    a = p.parse_args(argv)
    from tools.pfo import cost as C
    C.OUT = OUT
    extra = 0
    if a.net == "s43":
        require(a.impl in ("exec", "dense"), "S43 implementations: exec | dense")
        name, sp = "s43_s3407", None
    elif a.net == "pfo":
        from tools.pfo2 import run as R
        name, sp = R.pfo_spec(a.arm, a.seed, 6000)
        R.prepare(sp)
    else:
        from tools.pfo2 import reg1 as G
        if a.arm in G.ARMS1:
            from tools.pfo2 import run_s1 as RS
            name, sp = RS.arm_spec(a.arm, a.seed, 6000)
            RS.prepare(sp)
            if "disp" in G.spec(a.arm)["trunk"].split("+"):
                extra = disp_extra_macs()
        else:
            from tools.pfo2 import run as R
            from tools.pfo2 import run2 as R2
            name, sp = R2.arm_spec(a.arm, a.seed, 6000)
            R.prepare(sp)
    tag = f"final__{name}__{a.impl}__{a.policy}"
    C.run(name, a.impl, str(sp) if sp else None, tag, a.policy)
    rep = OUT / "costs" / tag / "report.json"
    save(OUT / "costs" / tag / "pfo2_extra.json", {"extra_functional_macs": extra, "report": str(rep),
                                                   "macs_layers_median": read(rep)["macs_layers"]["median"]},
         exclusive=True)


if __name__ == "__main__":
    main()
