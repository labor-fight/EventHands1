#!/usr/bin/env python3
"""MV2 wrapper entry for the M5 root-restricted self-prev arm: `tools/mv2/slr_entry.py` with `model.MNISTModel` replaced by
`semkine.mv2_rootself.RootSelfModel` (which is `SLRModel` without `MODEL.ROOTSELF`, and `MNISTModel` without `MODEL.SLR`), so S39, SLR and
ROOTSELF checkpoints all load through it. `s3` runs `tools/mv2/s3_check.py` with the same class swap.

    python tools/mv2/rootself_entry.py train --config configs/mv2/mv2_slr_rootself.yaml --seed 3407 --run-name N --output-dir D
    python tools/mv2/rootself_entry.py filter_eval ... | evalx row ... | s3 --arm-ckpt C --arm-config Y --step 2000 ...
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from mv2 import slr_entry  # noqa: E402

TARGETS = dict(slr_entry.TARGETS, s3=REPO / "tools" / "mv2" / "s3_check.py")


def patch():
    import model as M
    from semkine.mv2_rootself import RootSelfModel
    if M.MNISTModel is not RootSelfModel:
        M.MNISTModel = RootSelfModel
    slr_entry.patch = lambda: RootSelfModel           # s3_check imports `patch` from slr_entry at call time
    return RootSelfModel


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        raise SystemExit(f"usage: rootself_entry.py {{{'|'.join(TARGETS)}}} ...")
    target = TARGETS[sys.argv[1]]
    patch()
    sys.argv = [str(target)] + sys.argv[2:]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
