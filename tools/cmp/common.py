"""CMP (comparison experiments, docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110): constants, locations, I/O.

CPU only, imports nothing heavy. Five methods on one protocol (nine training subjects, zgz, PFO's E1 closed loop):
EventHands, EvHandPose, event encoder + previous-state MLP, event encoder + GRU (all four trained here, three seeds) and
EvHandTrack (the existing identity, outputs/evhandtrack/identity.json, not retrained). Every CMP product goes under
outputs/cmp/20261110/, every CMP config under configs/cmp_20261110/. Trainings run on direct GPUs 4-7 or, through
`gpu-borrow run`, on GPUs 0-3; readings and the cost block only on direct GPUs.
"""
from __future__ import annotations

from tools.pfo.common import COUNTS, PY, ROOT, TRAIN_LOCAL, ZGZ, now, path, read, require, save, sha  # noqa: F401

STUDY = "CMP_20261110"
SECTION = "cmp-prereg-20261110"
OUT = ROOT / "outputs/cmp/20261110"
CONFIGS = ROOT / "configs/cmp_20261110"
SEEDS = (3407, 3408, 3409)
DIRECT_GPUS = (4, 5, 6, 7)
BORROW_GPUS = (0, 1, 2, 3)
#: B2's training config of the seed: the data, augmentation, prior-noise, MANO and loss blocks every CMP arm inherits
B2_CONFIG = ROOT / "configs/pfo2_20261103/pfo2_fgs_s{seed}.yaml"
EVHANDTRACK = ROOT / "outputs/evhandtrack/identity.json"


def run_dir(arm, seed, phase="formal"):
    return OUT / "runs" / (f"cmp_{arm}_s{seed}" + ("" if phase == "formal" else f"_{phase}"))


__all__ = ["B2_CONFIG", "BORROW_GPUS", "CONFIGS", "COUNTS", "DIRECT_GPUS", "EVHANDTRACK", "OUT", "PY", "ROOT", "SEEDS",
           "SECTION", "STUDY", "TRAIN_LOCAL", "ZGZ", "now", "path", "read", "require", "run_dir", "save", "sha"]
