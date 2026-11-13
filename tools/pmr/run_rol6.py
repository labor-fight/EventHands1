#!/usr/bin/env python3
"""ABL rollout readings and acceptance: tools/pmr/run_rol2.py unchanged, union registry 6, acceptance tools/pmr/ablrt.py
`accept_rol`, RDR enabled, the loader taking arms without a router.

    python -m tools.pmr.run_rol6 read --arm A_rol5|A_rol1 --seed S --task T
    python -m tools.pmr.run_rol6 accept --arm A_rol5|A_rol1 --seed S --phase preflight|formal
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any6  # noqa: E402,F401
from semkine import rdr_hooks as RH  # noqa: E402
from tools.pmr import ablrt as RT  # noqa: E402
from tools.pmr import run_rol2 as R  # noqa: E402

RH.enable()
RT.patch_loader()
R.accept = RT.accept_rol

if __name__ == "__main__":
    R.main()
