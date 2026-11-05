#!/usr/bin/env python3
"""PFO acceptance / readings / costs of any registered arm (tools/pfo/arms3.py): tools/pfo/run4.py's procedure, unchanged,
resolving the arm through the data-driven registry.

    python -m tools.pfo.run5 accept|eval|cost --arm A --seed S [...]   (run4's arguments)
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo import arms3 as A3  # noqa: E402
from tools.pfo import run4 as R4  # noqa: E402

R4.ARMS2, R4.build_config, R4.spec = A3.ARMS3, A3.build_config, A3.spec

if __name__ == "__main__":
    R4.main()
