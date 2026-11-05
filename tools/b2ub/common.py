"""B2UB (layered ideal experiments and the upper bound of B2 = F + gshare, docs/EXPERIMENT_RECORDS.md#b2ub-prereg-20261104):
constants, locations, I/O.

CPU only, imports nothing heavy. Physical GPUs 2-7 only; GPU 0-1 are never used. PFO's frozen readout engine, PFO2's B2
runs / specs / readings / receipts, P0-B's F label reading and EVF's final cost block are reused read-only; every B2UB
product goes under outputs/b2ub/20261104/, every B2UB config under configs/b2ub_20261104/.
"""
from __future__ import annotations

from tools.pfo.common import (COUNTS, PAIRS, PHYSICAL_GPUS, PY, ROOT, TRAIN_LOCAL, XAG_SOURCE_ROOT, ZGZ, now,  # noqa: F401
                              path, read, require, save, sha)

STUDY = "B2UB_20261104"
SECTION = "b2ub-prereg-20261104"
OUT = ROOT / "outputs/b2ub/20261104"
CONFIGS = ROOT / "configs/b2ub_20261104"
SEEDS = (3407, 3408, 3409)
U3_SEED = 3407
#: S43 engine re-computation (3-seed means, #pfo-amend1-20261103): MPJPE-L, MPJPE-G, MPVPE-L, MPVPE-G, RA
S43_ENGINE = (12.81, 8.02, 10.77, 6.64, 10.24)
COLS = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")
#: pfagtdz (S45-SM round 5, label routes in training and test, old recipe, seed 3407), the published main row
PFAGTDZ = (8.77, 5.00, 7.43, 4.08, 6.75)

#: B2 = F + gshare (#pfo2-s0-results-20261103): PFO2's runs, specs, receipts and readings, read-only
PFO2_OUT = ROOT / "outputs/pfo2/20261103"
EVF_OUT = ROOT / "outputs/evf/20261104"
B2_ARM = "fgs"
#: B2's deployable form measured in EVF's final cost block (GPU 4 exclusive, PyTorch default flags, lyq_local 600 packets)
B2_COST = EVF_OUT / "costs/final__pfo2_fgs_s3407__fast__default/report.json"
#: P0-B's plain label-route reading of F (seed 3407), the reference of engine check K2
F_LABEL_E1 = PFO2_OUT / "readings/pfo_f1_s3407__g0m0__label/e1"


def b2_run(seed, phase="formal"):
    return PFO2_OUT / "runs" / (f"pfo2_fgs_s{seed}" + ("" if phase == "formal" else f"_{phase}"))


def b2_spec(seed):
    """PFO2's reading spec of B2 at step 6000 (kind routed, the deployed router r_s2d in the evaluation config)."""
    return PFO2_OUT / "specs" / f"pfo2_fgs_s{seed}.json"


def b2_reading(seed, task):
    return PFO2_OUT / "readings" / f"pfo2_fgs_s{seed}" / task


def f1_spec(seed=3407):
    return PFO2_OUT / "specs" / f"pfo_f1_s{seed}.json"


def check_b2(seed):
    """B2's step-6000 checkpoint must be the one PFO2 accepted."""
    receipt = read(PFO2_OUT / "receipts" / f"pfo2_fgs_s{seed}.json")
    require(receipt["status"] == "PASS", "B2 training not accepted")
    ck = read(b2_spec(seed))["checkpoint"]
    require(sha(ck) == receipt["checkpoint_sha256_by_step"]["6000"], "B2 checkpoint changed since acceptance")
    return ck


__all__ = ["B2_ARM", "B2_COST", "COLS", "CONFIGS", "COUNTS", "EVF_OUT", "F_LABEL_E1", "OUT", "PAIRS", "PFAGTDZ", "PFO2_OUT",
           "PHYSICAL_GPUS", "PY", "ROOT", "S43_ENGINE", "SECTION", "SEEDS", "STUDY", "TRAIN_LOCAL", "U3_SEED",
           "XAG_SOURCE_ROOT", "ZGZ", "b2_reading", "b2_run", "b2_spec", "check_b2", "f1_spec", "now", "path", "read",
           "require", "save", "sha"]
