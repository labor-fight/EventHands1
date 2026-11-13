#!/usr/bin/env python3
"""ABL readings and acceptance (docs/EXPERIMENT_RECORDS.md#abl-prereg-20261111): the frozen tools/pmr/run.py unchanged, with
its registry rebound to the union registry 6, its acceptance replaced by tools/pmr/ablrt.py `accept_base` (the router check
generalised to r_s2d / RDR / none), RDR checkpoints enabled and the engine loader taking arms without a router.

    python -m tools.pmr.run11 read --arm A --seed S --task T
    python -m tools.pmr.run11 accept --arm A --phase preflight|formal [--seed S]
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any6  # noqa: E402,F401
from semkine import rdr_hooks as RH  # noqa: E402
from tools.pmr import ablrt as RT  # noqa: E402
from tools.pmr import arms_any2 as U  # noqa: E402
from tools.pmr import run as R  # noqa: E402

RH.enable()
RT.patch_loader()
R.A = U
R.accept = RT.accept_base

if __name__ == "__main__":
    R.main()
