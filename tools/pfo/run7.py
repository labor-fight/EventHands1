#!/usr/bin/env python3
"""PFO acceptance / readings / costs of arms trained by tools/pfo/train4.py (composable trunk, semkine/pfo_trunk4.py):
tools/pfo/run4.py's procedure (accept, eval through the frozen engine, cost through the frozen harness) with the
data-driven registry (tools/pfo/arms3.py) and the composable trunk.

    python -m tools.pfo.run7 accept|eval|cost --arm A --seed S [--task T] [--step 6000] [--policy det|default]
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo import arms3 as A3  # noqa: E402
from tools.pfo import run4 as R4  # noqa: E402
from tools.pfo.train4 import trunk4_variant  # noqa: E402

R4.ARMS2, R4.build_config, R4.spec = A3.ARMS3, A3.build_config, A3.spec


class _Trunk4:
    """Stands in for semkine.pfo_trunk / pfo_trunk2 inside run4.main: installs the composable trunk."""

    @staticmethod
    def install(trunk):
        from semkine import pfo_trunk4 as PT
        return PT.install(trunk4_variant(trunk))

    @staticmethod
    def installed():
        from semkine import pfo_trunk4 as PT
        return PT.installed()


def main(argv=None):
    import semkine
    import semkine.pfo_trunk  # noqa: F401  (make sure the attributes exist before replacing them)
    import semkine.pfo_trunk2  # noqa: F401
    shim = _Trunk4()
    semkine.pfo_trunk, semkine.pfo_trunk2 = shim, shim
    sys.modules["semkine.pfo_trunk"], sys.modules["semkine.pfo_trunk2"] = shim, shim
    R4.main(argv)


if __name__ == "__main__":
    main()
