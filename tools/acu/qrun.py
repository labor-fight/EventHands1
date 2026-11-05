#!/usr/bin/env python3
"""ACU PR revision (docs/EXPERIMENT_RECORDS.md#acu-pr2-prereg-20261104): run a registered ACU entry point (frozen, unchanged)
under the namespace "q" (tools/acu/ns_q.py: PR2 in place of PR, deterministic window sums, roots outputs/acu/20261104q and
configs/acu_20261104q).

    python -m tools.acu.qrun <module> [args ...]      <module>.main(args), or <module>.main() for an entry without arguments

The trainings use the script wrappers tools/acu/router_pr2.py and tools/acu/train_s2q.py (Lightning's DDP re-runs the
launching script in the second rank).
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu import ns_q  # noqa: E402


def main():
    ns_q.enter()
    from tools.acu.common import require
    require(len(sys.argv) >= 2, "usage: python -m tools.acu.qrun <module> [args ...]")
    target = importlib.import_module(sys.argv[1])
    args = sys.argv[2:]
    if inspect.signature(target.main).parameters:
        target.main(args)
    else:
        require(not args, f"{sys.argv[1]} takes no arguments")
        target.main()


if __name__ == "__main__":
    main()
