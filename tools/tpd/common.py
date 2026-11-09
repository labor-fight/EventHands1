"""TPD (B2 three-problem decomposition, docs/EXPERIMENT_RECORDS.md#tpd-prereg-20261105): constants, locations, I/O.

CPU only, imports nothing heavy. Physical GPUs 2-7 only; GPU 0-1 are never used. PFO's frozen engine, PFO2's B2 runs /
specs / receipts / readings (seeds 3407-3409), IDP's L1 and B2UB's readings and EVF's B2 cost block are reused read-only;
every TPD product goes under outputs/tpd/20261105/, every TPD config under configs/tpd_20261105/.
"""
from __future__ import annotations

from tools.pfo.common import (COUNTS, PAIRS, PHYSICAL_GPUS, PY, ROOT, TRAIN_LOCAL, ZGZ, now, path, read, require,  # noqa: F401
                              save, sha)

STUDY = "TPD_20261105"
SECTION = "tpd-prereg-20261105"
OUT = ROOT / "outputs/tpd/20261105"
CONFIGS = ROOT / "configs/tpd_20261105"
SEEDS = (3407, 3408)
#: S43 engine re-computation (3-seed means, #pfo-amend1-20261103): MPJPE-L, MPJPE-G, MPVPE-L, MPVPE-G, RA
S43_ENGINE = (12.81, 8.02, 10.77, 6.64, 10.24)
COLS = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")
NOISE_RA = 0.8                       # single-seed noise floor of RA (#pfo2-prereg-20261103)

PFO2_OUT = ROOT / "outputs/pfo2/20261103"
IDP_OUT = ROOT / "outputs/idp/20261105"
B2UB_OUT = ROOT / "outputs/b2ub/20261104"
EVF_OUT = ROOT / "outputs/evf/20261104"
#: B2's deployable form measured in EVF's final cost block (GPU 4 exclusive, PyTorch default flags, lyq_local 600 packets)
B2_COST = EVF_OUT / "costs/final__pfo2_fgs_s3407__fast__default/report.json"


def b2_tag(seed):
    return f"pfo2_fgs_s{seed}"


def b2_run(seed, phase="formal"):
    return PFO2_OUT / "runs" / (b2_tag(seed) + ("" if phase == "formal" else f"_{phase}"))


def b2_spec(seed):
    """PFO2's reading spec of B2 at step 6000 (kind routed, the deployed router r_s2d)."""
    return PFO2_OUT / "specs" / f"{b2_tag(seed)}.json"


def b2_checkpoint(seed):
    """B2's step-6000 checkpoint, which must be the one PFO2 accepted."""
    receipt = read(PFO2_OUT / "receipts" / f"{b2_tag(seed)}.json")
    require(receipt["status"] == "PASS", "B2 training not accepted")
    ck = read(b2_spec(seed))["checkpoint"]
    require(sha(ck) == receipt["checkpoint_sha256_by_step"]["6000"], "B2 checkpoint changed since acceptance")
    return ck


__all__ = ["B2_COST", "B2UB_OUT", "COLS", "CONFIGS", "COUNTS", "EVF_OUT", "IDP_OUT", "NOISE_RA", "OUT", "PAIRS",
           "PFO2_OUT", "PHYSICAL_GPUS", "PY", "ROOT", "S43_ENGINE", "SECTION", "SEEDS", "STUDY", "TRAIN_LOCAL", "ZGZ",
           "b2_checkpoint", "b2_run", "b2_spec", "b2_tag", "now", "path", "read", "require", "save", "sha"]
