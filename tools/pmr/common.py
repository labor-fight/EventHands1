"""PMR (redesign of the propagation module P, docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109): constants, locations, I/O.

CPU only, imports nothing heavy. Physical GPUs 2-7 only (pairs (2, 3), (4, 5), (6, 7)); GPU 0-1 never. PFO's frozen engine
and PFO2's B2 run / spec / receipt / readings (seed 3407) are reused read-only; every PMR product goes under
outputs/pmr/20261109/, every PMR config under configs/pmr_20261109/.
"""
from __future__ import annotations

from tools.pfo.common import (COUNTS, PAIRS, PHYSICAL_GPUS, PY, ROOT, TRAIN_LOCAL, ZGZ, now, path, read, require,  # noqa: F401
                              save, sha)

STUDY = "PMR_20261109"
SECTION = "pmr-prereg-20261109"
OUT = ROOT / "outputs/pmr/20261109"
CONFIGS = ROOT / "configs/pmr_20261109"
SEED = 3407
SEEDS = (3407, 3408, 3409)
#: S43 engine re-computation (3-seed means, #pfo-amend1-20261103): MPJPE-L, MPJPE-G, MPVPE-L, MPVPE-G, RA
S43_ENGINE = (12.81, 8.02, 10.77, 6.64, 10.24)
COLS = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")
NOISE_RA = 0.8                       # single-seed noise floor of RA (#pfo2-prereg-20261103)

PFO2_OUT = ROOT / "outputs/pfo2/20261103"
EVF_OUT = ROOT / "outputs/evf/20261104"
#: B2's deployable form measured in EVF's final cost block (GPU 4 exclusive, PyTorch default flags, lyq_local 600 packets)
B2_COST = EVF_OUT / "costs/final__pfo2_fgs_s3407__fast__default/report.json"
B2_ARM = "fgs"
B2_TAG = "pfo2_fgs_s3407"


def b2_run(phase="formal", seed=SEED):
    return PFO2_OUT / "runs" / (f"pfo2_fgs_s{seed}" + ("" if phase == "formal" else f"_{phase}"))


def b2_spec():
    """PFO2's reading spec of B2 seed 3407 at step 6000 (kind routed, the deployed router r_s2d)."""
    return PFO2_OUT / "specs" / f"{B2_TAG}.json"


def check_b2():
    """B2's step-6000 checkpoint must be the one PFO2 accepted."""
    receipt = read(PFO2_OUT / "receipts" / f"{B2_TAG}.json")
    require(receipt["status"] == "PASS", "B2 training not accepted")
    ck = read(b2_spec())["checkpoint"]
    require(sha(ck) == receipt["checkpoint_sha256_by_step"]["6000"], "B2 checkpoint changed since acceptance")
    return ck


__all__ = ["B2_ARM", "B2_COST", "B2_TAG", "COLS", "CONFIGS", "COUNTS", "EVF_OUT", "NOISE_RA", "OUT", "PAIRS",
           "PFO2_OUT", "PHYSICAL_GPUS", "PY", "ROOT", "S43_ENGINE", "SECTION", "SEED", "SEEDS", "STUDY", "TRAIN_LOCAL",
           "ZGZ", "b2_run", "b2_spec", "check_b2", "now", "path", "read", "require", "save", "sha"]
