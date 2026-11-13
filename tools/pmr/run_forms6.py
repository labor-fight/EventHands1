#!/usr/bin/env python3
"""ABL input-form readings: tools/pmr/run_forms.py unchanged, union registry 6, RDR enabled, the loader taking arms without a
router.

    python -m tools.pmr.run_forms6 read --arm A_rol5|A_rol1 --seed S --form str --task e1
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any6  # noqa: E402,F401
from semkine import rdr_hooks as RH  # noqa: E402
from tools.pmr import ablrt as RT  # noqa: E402
from tools.pmr import run_forms as F  # noqa: E402

RH.enable()
RT.patch_loader()

if __name__ == "__main__":
    F.main()
