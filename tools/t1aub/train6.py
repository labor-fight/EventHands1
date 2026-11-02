#!/usr/bin/env python3
"""T1A-UB rounds >= 6: the frozen round-1 training wrapper (tools/t1aub/train.py, unchanged) with tools/t1aub/r6.py's config
generator and hooks, so router-location-feature arms (gtfeat + a router in its face_route mode, S45's own association) are
validated against their registered config and installed router-first. Lightning re-executes this file for DDP rank 1
(sys.argv[0]), so both ranks patch before the model is built. Other arms behave exactly as under train.py.

    python tools/t1aub/train6.py --arm A --phase {preflight,formal} --config C
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]


def main():
    from tools.t1aub import r6
    r6.patch_common()
    from tools.t1aub import train as W
    W.main()


if __name__ == "__main__":
    main()
