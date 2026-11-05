"""PFO2 (closed-loop gain on the PFO base F, docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103): constants, locations.

CPU only, imports nothing heavy. Physical GPUs 2-7 only; GPU 0-1 are never used. The PFO study's frozen engine, cost
harness, F / S43 readings and helpers are reused read-only (tools/pfo/*, outputs/pfo/20261103/); every PFO2 product goes
under outputs/pfo2/20261103/, every PFO2 config under configs/pfo2_20261103/.
"""
from __future__ import annotations

from pathlib import Path

from tools.pfo.common import (COUNTS, PAIRS, PHYSICAL_GPUS, PY, ROOT, S43_SEEDS, TRAIN_LOCAL, ZGZ, now, path,  # noqa: F401
                              read, require, save, sha)
from tools.pfo.common import OUT as PFO_OUT

STUDY = "PFO2_20261103"
OUT = ROOT / "outputs/pfo2/20261103"
CONFIGS = ROOT / "configs/pfo2_20261103"
PFO_CONFIGS = ROOT / "configs/pfo_20261103"
SEEDS = (3407, 3408, 3409)
#: S43 engine re-computation (3-seed means, #pfo-amend1-20261103): MPJPE-L, MPJPE-G, MPVPE-L, MPVPE-G, RA
S43_ENGINE = (12.81, 8.02, 10.77, 6.64, 10.24)
COLS = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")


def pfo_run(arm, seed, phase="formal"):
    """A PFO run directory (read-only)."""
    return PFO_OUT / "runs" / (f"pfo_{arm}_s{seed}" + ("_preflight" if phase == "preflight" else ""))


__all__ = ["CONFIGS", "COLS", "COUNTS", "OUT", "PAIRS", "PFO_CONFIGS", "PFO_OUT", "PHYSICAL_GPUS", "PY", "ROOT",
           "S43_ENGINE", "S43_SEEDS", "SEEDS", "STUDY", "TRAIN_LOCAL", "ZGZ", "now", "path", "pfo_run", "read",
           "require", "save", "sha"]
