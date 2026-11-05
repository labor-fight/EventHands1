"""EVF (event points x previous-state virtual frame, docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104): constants,
locations, I/O.

CPU only, imports nothing heavy. Physical GPUs 2-7 only; GPU 0-1 are never used. PFO's frozen readout engine and cost
helpers and PFO2's B2 runs / readings are reused read-only; every EVF product goes under outputs/evf/20261104/, every
EVF config under configs/evf_20261104/.
"""
from __future__ import annotations

from pathlib import Path

from tools.pfo.common import (COUNTS, PAIRS, PHYSICAL_GPUS, PY, ROOT, TRAIN_LOCAL, ZGZ, now, path, read,  # noqa: F401
                              require, save, sha)

STUDY = "EVF_20261104"
OUT = ROOT / "outputs/evf/20261104"
CONFIGS = ROOT / "configs/evf_20261104"
SEEDS = (3407, 3408, 3409)
SCREEN_SEED = 3407
#: S43 engine re-computation (3-seed means, #pfo-amend1-20261103): MPJPE-L, MPJPE-G, MPVPE-L, MPVPE-G, RA
S43_ENGINE = (12.81, 8.02, 10.77, 6.64, 10.24)
COLS = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")
#: B2 = F + gshare (#pfo2-s0-results-20261103), the comparator: PFO2's runs and readings, read-only
PFO2_OUT = ROOT / "outputs/pfo2/20261103"
B2_ARM = "fgs"
B2_CONFIG = ROOT / "configs/pfo2_20261103/pfo2_fgs_s{seed}.yaml"


def b2_run(seed):
    return PFO2_OUT / "runs" / f"pfo2_fgs_s{seed}"


def b2_checkpoint(seed, step):
    return b2_run(seed) / f"pfo2_fgs_s{seed}-step={step}.ckpt"


__all__ = ["B2_ARM", "B2_CONFIG", "COLS", "CONFIGS", "COUNTS", "OUT", "PAIRS", "PFO2_OUT", "PHYSICAL_GPUS", "PY", "ROOT",
           "S43_ENGINE", "SCREEN_SEED", "SEEDS", "STUDY", "TRAIN_LOCAL", "ZGZ", "b2_checkpoint", "b2_run", "now",
           "path", "read", "require", "save", "sha"]
