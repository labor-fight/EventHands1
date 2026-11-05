#!/usr/bin/env python3
"""PFO cost-harness smoke for an arm trained by tools/pfo/train4.py (an engineering check before the final cost block,
not a reading): tools/pfo/cost.py's `run`, unchanged, on an early checkpoint of the arm, with run7's trunk / router
setup. Everything it writes (eval config, spec, cost report) goes under outputs/pfo/20261103/smoke/, so no study
config, spec or cost reading is created; its timings are meaningless (the GPU may be shared).

    python -m tools.pfo.cost_smoke --arm f1 --seed 3407 --step 500 [--policy default|det]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo import arms3 as A3  # noqa: E402
from tools.pfo import cost as C  # noqa: E402
from tools.pfo import run4 as R4  # noqa: E402
from tools.pfo.common import OUT, require, save  # noqa: E402
from tools.pfo.train4 import trunk4_variant  # noqa: E402

SMOKE = OUT / "smoke"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=tuple(A3.ARMS3))
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--step", required=True, type=int)
    p.add_argument("--policy", default="default", choices=("det", "default"))
    a = p.parse_args(argv)
    R4.ARMS2, R4.build_config, R4.spec = A3.ARMS3, A3.build_config, A3.spec
    R4.CONFIGS = SMOKE / "configs"
    C.OUT = SMOKE
    s = A3.spec(a.arm)
    name = R4.run_name(a.arm, a.seed)
    ck = R4.run_dir(a.arm, a.seed) / f"{name}-step={a.step}.ckpt"
    require(ck.exists(), "no checkpoint " + str(ck))
    sp = SMOKE / "specs" / f"{name}@{a.step}.json"
    save(sp, {"kind": "routed", "checkpoint": str(ck), "config": str(R4.eval_config(a.arm, a.seed)), "trunk": s["trunk"]})
    if s.get("router") and not s.get("train_router"):
        from semkine import pfo_router as PR
        PR.register()
    from semkine import pfo_trunk4 as PT
    PT.install(trunk4_variant(s["trunk"]))
    C.run(f"{name}@{a.step}", "fast", str(sp), None, a.policy)


if __name__ == "__main__":
    main()
