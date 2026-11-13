#!/usr/bin/env python3
"""EvHandTrack streaming-form asynchronous checks (docs/EXPERIMENT_RECORDS.md#evhandtrack-d1-20261110): TPD's frozen
check (tools/tpd/asynccheck_str.py: prefix + order + partition on the streaming form) on EvHandTrack's network (PMR
`pmr_d1r_rol5_s<seed>`, step 1500; --arm d1ghand_rol5 for the ghand + rol5 record), with TPD's trunk preparation
replaced by PMR's (tools/pmr/asynccheck.py's binding). CPU; receipt <out-root>/receipts/async_evhandtrack_s<seed>.json.

    python -m tools.evhandtrack.asynccheck --seed S [--arm d1r_rol5|d1ghand_rol5] [--out-root DIR]
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

OUT = ROOT / "outputs/evhandtrack"
SPECS = ROOT / "outputs/pmr/20261109/specs"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--arm", default="d1r_rol5", choices=("d1r_rol5", "d1ghand_rol5"))
    p.add_argument("--out-root", default=str(OUT))
    a = p.parse_args(argv)
    from tools.pmr import run as R
    from tools.tpd import run_rol as TR
    TR.prepare = R.prepare                                          # the PMR trunk instead of B2's gshare trunk
    from tools.tpd import asynccheck_str as C
    sp = SPECS / f"pmr_{a.arm}_s{a.seed}.json"
    return C.main(["--name", f"evhandtrack_s{a.seed}", "--spec", str(sp), "--out-root", a.out_root])


if __name__ == "__main__":
    main()
