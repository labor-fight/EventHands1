"""S45-SM constants, locations and I/O helpers (CPU only; imports nothing heavy).

The package lives in `tools/s45sm/` (while it is being introduced it sits in a sibling directory; nothing in the package
names its own directory, so the rename is mechanical). The per-round registries `registry_r<k>.json` are read from the
package directory and, while the package is still in that sibling directory, from `tools/s45sm/`.

Everything a reader (summary, table, selection, reading) reads or writes below `outputs/s45sm/20261031/` goes through
`results_dir()`: `$S45SM_RESULTS_DIR` if set (a directory whose children are symbolic links to the real artifacts
except the files being recomputed; used to verify the readers without touching the real outputs), else `OUT`.
Trainers, evaluators and acceptance use `OUT` directly.

Verification in a copy of the repository (the artifacts and configs record absolute paths below the root they were
written under): `$S45SM_ROOT_ALIAS` names that root, and `recorded(path)` renders a path of this tree under it. Unset
(the normal case), `recorded` is `str`.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from tools.s45tr.common import now, read, require, save, sha  # noqa: E402,F401

STUDY = "S45SM_20261031"
OUT = ROOT / "outputs/s45sm/20261031"
CONFIGS = ROOT / "configs/s45sm_20261031"
MANIFESTS = OUT / "manifests"
ROUTERS_DIR = OUT / "routers"
BANKS = OUT / "banks"
IDENTITY_MANIFEST = "p1_core.json"
REGISTRY_DIR = PACKAGE if any(PACKAGE.glob("registry_r*.json")) else ROOT / "tools" / "s45sm"
PHYSICAL_GPUS = (2, 3, 4, 5, 6, 7)
S45_CONFIG = ROOT / "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml"
S45_RUN = ROOT / "outputs/c1sf/20261029/runs/c1sf_so3_s3407"
S45_EVAL = ROOT / "outputs/c1sf/20261029/evaluations/so3"
S45_E1_TRACE = S45_EVAL / "e1_step6000_repeat1_none/states.npz"
S45AT = ROOT / "outputs/s45at/20261031"
S45TR = ROOT / "outputs/s45tr/20261030"
S45CT = ROOT / "outputs/s45ct/20261030"
DZ_RUN = S45AT / "runs/s45at_dz_s3407"
SAMP_RUN = S45AT / "runs/s45at_s45samp_s3407"
T1DDZ_CONFIG = ROOT / "configs/s45at_20261031/s45at_t1ddz_s3407.yaml"
T1DDZ_RUN = S45AT / "runs/s45at_t1ddz_s3407"
T1DDZ_IDENTITY = {"ARM": "t1ddz", "BASE": "t1d", "EXCLUDE_SUBJECT": None, "SAMPLER": "legacy_none"}
HELDOUT = "ylf"
S45_PARAMS = 459059
BUDGET = {"params_total_max": 600000, "latency_scaled_extra_ms_max": 2.0}
TARGET = {"local": 12.82, "global": 6.58, "ra": 9.48}           # S45-AT t1ddz main row (oracle)
STEPS = (5000, 5500, 6000)
NOISE = {"local": 3.5, "global": 2.75, "ra": 3.1}
TH = {"at_target_mm": 1.5, "C": 0.8, "R_high": 0.6, "R_null": 0.15}
VALIDATION = {"batches": 126, "samples": 64094}
SEEDED = 3407
SAME_STEPS = 500
SAMPLERS = ("legacy_none", "dedicated")
SCHEDULES = ("constant", "cosine")


ROOT_ALIAS = os.environ.get("S45SM_ROOT_ALIAS") or None


def recorded(path):
    """str(path) as recorded in artifacts: this tree's root replaced by $S45SM_ROOT_ALIAS when set."""
    text = str(path)
    return text.replace(str(ROOT), ROOT_ALIAS) if ROOT_ALIAS else text


def results_dir():
    """Where readers read and write the derived artifacts (see the module docstring)."""
    override = os.environ.get("S45SM_RESULTS_DIR")
    return Path(override) if override else OUT


def require_one_gpu():
    require(os.environ.get("CUDA_VISIBLE_DEVICES", "") in tuple(map(str, PHYSICAL_GPUS)), "one GPU among 2-7")
