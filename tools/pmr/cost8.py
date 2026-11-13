#!/usr/bin/env python3
"""PMR D2 screen cost readings (docs/EXPERIMENT_RECORDS.md#pmr-d2-screen-cost-20261110; descriptive): the EvHandTrack cost
tool's `run` (tools/evhandtrack/cost2.py: PFO's frozen harness, PMR's stage breakdown, the streaming form's hooks)
unchanged, in the streaming form with the reference trunk, for one rollout arm at seed 3407.

    python -m tools.pmr.cost8 run --arm d1r_rol5|d1hubfull_rol5
        output root outputs/pmr/20261109/costs_s9x/<arm>/ (the tool's tag does not carry the arm)

The reference trunk, not semkine/pmr_fast.py: the fast path rebuilds the propagation from D1's four typed maps and would
drop D2's per-part part -> hand maps. D1 R + rol5 and D2 + rol5 are read back to back on one GPU, so their ratio is a
same-block one; the block is not a quiet one (CMP's trainings load the host), so the absolute latencies are not
comparable to the EvHandTrack identity's cost block.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import arms_any4  # noqa: E402,F401
from tools.pmr.common import OUT  # noqa: E402

ARMS = ("d1r_rol5", "d1hubfull_rol5")
COSTS = OUT / "costs_s9x"
TAG = "evhandtrack_s3407__stream__ref__default"


def report(arm):
    return COSTS / arm / "costs" / TAG / "report.json"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", required=True, choices=ARMS)
    a = p.parse_args(argv)
    from tools.evhandtrack import cost2 as EC2
    EC2.run("stream", "default", False, COSTS / a.arm, a.arm)


if __name__ == "__main__":
    main()
