#!/usr/bin/env python3
"""MV2 wrapper entry: run a frozen entry point with `model.MNISTModel` replaced by `semkine.slr.SLRModel`.

`SLRModel` without `MODEL.SLR` is `MNISTModel` bit for bit, so every existing checkpoint still loads; with it, the
surface-lifted readout is built. The frozen entry points (`semkine/train.py`, `tools/tracking/evalx.py`,
`tools/dt/filter_eval.py`, `tools/make_s36_row.py`) import `MNISTModel` by name at import time, so the class is swapped
in the `model` module before they are imported; none of them is modified (they are in the R2 / MRF / AURA freezes).

    python tools/mv2/slr_entry.py train  --config C [--seed S --run-name N --output-dir D]
    python tools/mv2/slr_entry.py evalx  row --arm A --runs R ... --variant w200cfg
    python tools/mv2/slr_entry.py filter_eval ...
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

TARGETS = {"train": REPO / "semkine" / "train.py", "evalx": REPO / "tools" / "tracking" / "evalx.py",
           "filter_eval": REPO / "tools" / "dt" / "filter_eval.py", "make_s36_row": REPO / "tools" / "make_s36_row.py"}


def patch():
    import model as M
    from semkine.slr import SLRModel
    if M.MNISTModel is not SLRModel:
        M.MNISTModel = SLRModel
    return SLRModel


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in TARGETS:
        raise SystemExit(f"usage: slr_entry.py {{{'|'.join(TARGETS)}}} ...")
    target = TARGETS[sys.argv[1]]
    patch()
    sys.argv = [str(target)] + sys.argv[2:]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
