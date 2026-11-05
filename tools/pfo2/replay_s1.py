#!/usr/bin/env python3
"""PFO2 stage-1 implementation check (docs/EXPERIMENT_RECORDS.md#pfo2-s1-prereg-20261103): B2's recipe trained through
semkine/pfo2_trunk.py with no change (the plan's replay arm, tools/pfo2/train_s1.py) must reproduce B2's 40-step
preflight (semkine/pfo_trunk4.py, tools/pfo2/train_det.py) bitwise: model, optimiser and scheduler states at step 40
and both ranks' first-batch data streams (tools/pfo2/replay.py `check`). Receipt receipts/cmp_<arm>_preflight.json.

    python -m tools.pfo2.replay_s1 --arm REPLAY --base B2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2.arms import run_dir  # noqa: E402
from tools.pfo2.replay import check, ckpt  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True)
    p.add_argument("--base", required=True)
    a = p.parse_args(argv)
    bd, qd = run_dir(a.base, 3407, "preflight"), run_dir(a.arm, 3407, "preflight")
    check(f"{a.arm}_preflight", ckpt(bd, bd.name, 40), ckpt(qd, qd.name, 40), bd, qd)


if __name__ == "__main__":
    main()
