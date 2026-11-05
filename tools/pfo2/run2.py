#!/usr/bin/env python3
"""PFO2 readings of tools/pfo2/arms.py arms with race-safe spec / config writing
(docs/EXPERIMENT_RECORDS.md#pfo2-amend2-20261103): tools/pfo2/run.py's `eval` (unchanged otherwise: the accepted
checkpoint, the evaluation config with the protocol's segment-start noise and the deployed router, PFO's frozen engine
with the PFO2 output root), with tools/pfo2/specs.py writing the shared files.

    python -m tools.pfo2.run2 eval --arm A --seed S --task T [--step 6000]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2 import arms as A  # noqa: E402
from tools.pfo2 import run as R  # noqa: E402
from tools.pfo2 import specs as S  # noqa: E402
from tools.pfo2.common import CONFIGS, OUT, read, require, sha  # noqa: E402


def eval_config(arm, seed):
    name = A.run_name(arm, seed)
    return S.config(CONFIGS / "eval" / f"{name}.yaml", R._eval_cfg(A.build_config(arm, seed), name))


def arm_spec(arm, seed, step):
    receipt = read(OUT / "receipts" / f"{A.run_name(arm, seed)}.json")
    require(receipt["status"] == "PASS", "training not accepted")
    ck = R.checkpoint(arm, seed, step)
    require(sha(ck) == receipt["checkpoint_sha256_by_step"][str(step)], "checkpoint changed since acceptance")
    tag = A.run_name(arm, seed) + ("" if step == 6000 else f"@{step}")
    return tag, S.spec(tag, {"kind": "routed", "checkpoint": str(ck), "config": str(eval_config(arm, seed)),
                             "trunk": A.recipe(arm)["trunk"], "module": A.spec(arm).get("module", "trunk4")})


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("eval")
    s.add_argument("--arm", required=True)
    s.add_argument("--seed", required=True, type=int)
    s.add_argument("--task", required=True)
    s.add_argument("--step", type=int, default=6000, choices=R.STEPS)
    a = p.parse_args(argv)
    tag, sp = arm_spec(a.arm, a.seed, a.step)
    R.prepare(sp)
    R.engine_run(tag, a.task, sp)


if __name__ == "__main__":
    main()
