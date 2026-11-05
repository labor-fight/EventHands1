"""ACU (line A correspondence interface x line B update rule x strict causal asynchrony,
docs/EXPERIMENT_RECORDS.md#acu-s1-prereg-20261104): constants, locations, I/O.

CPU only, imports nothing heavy. Physical GPUs 2-7 only; GPU 0-1 are never used. PFO's frozen readout engine and PFO2 /
B2UB runs, specs, receipts and readings are reused read-only; every ACU product goes under outputs/acu/20261104/, every
ACU config under configs/acu_20261104/.
"""
from __future__ import annotations

from tools.pfo.common import (COUNTS, PAIRS, PHYSICAL_GPUS, PY, ROOT, TRAIN_LOCAL, XAG_SOURCE_ROOT, ZGZ, now,  # noqa: F401
                              path, read, require, save, sha)

STUDY = "ACU_20261104"
SECTION = "acu-s1-prereg-20261104"
OUT = ROOT / "outputs/acu/20261104"
CONFIGS = ROOT / "configs/acu_20261104"
SCREEN_SEED = 3407
#: S43 engine re-computation (3-seed means, #pfo-amend1-20261103): MPJPE-L, MPJPE-G, MPVPE-L, MPVPE-G, RA
S43_ENGINE = (12.81, 8.02, 10.77, 6.64, 10.24)
COLS = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")

PFO2_OUT = ROOT / "outputs/pfo2/20261103"
B2UB_OUT = ROOT / "outputs/b2ub/20261104"
#: the bases the stage-1 arms are paired with (same seed, same data stream, one change)
B2_RUN = PFO2_OUT / "runs/pfo2_fgs_s3407"
B2GT_RUN = B2UB_OUT / "runs/b2ub_b2gt_s3407"
B2_READ = PFO2_OUT / "readings/pfo2_fgs_s3407"
B2GT_LABEL_READ = B2UB_OUT / "readings/b2gt_s3407__label"
B2GT_ROUTER_READ = B2UB_OUT / "readings/b2gt_s3407"


def run_dir_base(base, phase="formal"):
    d = {"fgs": B2_RUN, "b2gt": B2GT_RUN}[base]
    return d if phase == "formal" else d.with_name(d.name + f"_{phase}")


__all__ = ["B2GT_LABEL_READ", "B2GT_ROUTER_READ", "B2GT_RUN", "B2UB_OUT", "B2_READ", "B2_RUN", "COLS", "CONFIGS", "COUNTS",
           "OUT", "PAIRS", "PFO2_OUT", "PHYSICAL_GPUS", "PY", "ROOT", "S43_ENGINE", "SCREEN_SEED", "SECTION", "STUDY",
           "TRAIN_LOCAL", "XAG_SOURCE_ROOT", "ZGZ", "now", "path", "read", "require", "run_dir_base", "save", "sha"]
