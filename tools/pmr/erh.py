#!/usr/bin/env python3
"""PMR stage S3c EvRealHands readings (docs/EXPERIMENT_RECORDS.md#pmr-s3c-prereg-20261109; report only, no reading enters
any rule): TPD's frozen evaluator (tools/tpd/erh_eval.py: the converted right-hand recordings and trials of
#tpd-erh-prereg-20261106, own closed loop on the 50 ms grid scored at the labels, h200) with a PMR rollout arm as the
network (ordinary packet form, deployed router) and the readings written under outputs/pmr/20261109/erh/readings/
(through the evaluator's scratch-root variable TPD_ERH_SMOKE_DIR; the recordings and trials are read from TPD's tree).

    python -m tools.pmr.erh run --net arm --arm A_rol5 --seed S --tier 3d
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr.common import OUT  # noqa: E402

os.environ["TPD_ERH_SMOKE_DIR"] = str(OUT / "erh" / "readings")
os.environ.pop("TPD_ERH_SMOKE_TRIALS", None)

from tools.tpd import erh_eval as E  # noqa: E402

_native = E.load_net


def load_net(net, seed, arm=None, device=None):
    if net != "arm":
        return _native(net, seed, arm, device)
    from tools.pfo import nets as N
    from tools.pmr import run as R
    from tools.pmr import run_rol2 as RR
    tag, sp = RR.arm_spec(arm, seed)
    R.prepare(sp)
    return tag, N.load(tag, device=device, spec=E._spec_obj(sp)), False


E.load_net = load_net

if __name__ == "__main__":
    sys.exit(E.main())
