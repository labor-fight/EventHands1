"""IDP (identity / point ladder on B2 = F + gshare, docs/EXPERIMENT_RECORDS.md#idp-prereg-20261105): constants,
locations, I/O.

CPU only, imports nothing heavy. Physical GPUs 2-7 only; GPU 0-1 are never used. PFO's frozen engine, PFO2's B2 run /
spec / receipt / readings (seed 3407), B2UB's test-time `label_part` reading of B2 and EVF's B2 cost block are reused
read-only; every IDP product goes under outputs/idp/20261105/, every IDP config under configs/idp_20261105/.
"""
from __future__ import annotations

from tools.pfo.common import (COUNTS, PAIRS, PHYSICAL_GPUS, PY, ROOT, TRAIN_LOCAL, XAG_SOURCE_ROOT, ZGZ, now,  # noqa: F401
                              path, read, require, save, sha)

STUDY = "IDP_20261105"
SECTION = "idp-prereg-20261105"
OUT = ROOT / "outputs/idp/20261105"
CONFIGS = ROOT / "configs/idp_20261105"
SEED = 3407
#: S43 engine re-computation (3-seed means, #pfo-amend1-20261103): MPJPE-L, MPJPE-G, MPVPE-L, MPVPE-G, RA
S43_ENGINE = (12.81, 8.02, 10.77, 6.64, 10.24)
COLS = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")
NOISE_RA = 0.8                       # single-seed noise floor of RA (#pfo2-prereg-20261103)

PFO2_OUT = ROOT / "outputs/pfo2/20261103"
B2UB_OUT = ROOT / "outputs/b2ub/20261104"
EVF_OUT = ROOT / "outputs/evf/20261104"
#: B2's deployable form measured in EVF's final cost block (GPU 4 exclusive, PyTorch default flags, lyq_local 600 packets)
B2_COST = EVF_OUT / "costs/final__pfo2_fgs_s3407__fast__default/report.json"
B2_ARM = "fgs"
B2_TAG = "pfo2_fgs_s3407"


def b2_run(phase="formal"):
    return PFO2_OUT / "runs" / (B2_TAG + ("" if phase == "formal" else f"_{phase}"))


def b2_spec():
    """PFO2's reading spec of B2 seed 3407 at step 6000 (kind routed, the deployed router r_s2d)."""
    return PFO2_OUT / "specs" / f"{B2_TAG}.json"


def b2_reading(task):
    return PFO2_OUT / "readings" / B2_TAG / task


def b2ub_reading(route, task):
    return B2UB_OUT / "readings" / f"b2_s{SEED}__{route}" / task


def check_b2():
    """B2's step-6000 checkpoint must be the one PFO2 accepted."""
    receipt = read(PFO2_OUT / "receipts" / f"{B2_TAG}.json")
    require(receipt["status"] == "PASS", "B2 training not accepted")
    ck = read(b2_spec())["checkpoint"]
    require(sha(ck) == receipt["checkpoint_sha256_by_step"]["6000"], "B2 checkpoint changed since acceptance")
    return ck


__all__ = ["B2_ARM", "B2_COST", "B2_TAG", "B2UB_OUT", "COLS", "CONFIGS", "COUNTS", "EVF_OUT", "NOISE_RA", "OUT", "PAIRS",
           "PFO2_OUT", "PHYSICAL_GPUS", "PY", "ROOT", "S43_ENGINE", "SECTION", "SEED", "STUDY", "TRAIN_LOCAL",
           "XAG_SOURCE_ROOT", "ZGZ", "b2_reading", "b2_run", "b2_spec", "b2ub_reading", "check_b2", "now", "path", "read",
           "require", "save", "sha"]
