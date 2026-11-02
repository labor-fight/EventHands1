"""S45-RT identity helpers (docs/EXPERIMENT_RECORDS.md#s45rt-prereg-20261031); CPU only, no scoring.

S45-RT = deployable routing for S45: plan 1 reopens S45-TR's registered arm T1b (S45 + routes of the frozen T0,
S45-TR's own files and tools, products under outputs/s45tr/20261030/); plan 2 trains the surface-patch router T0P
(semkine/s45rt_router.py) and S45 + its frozen routes (T1p). Plan-2 identity helpers live in tools/s45rt/plan2.py,
frozen in manifests/plan2.json before T0P's preflight.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.frozen import accepts  # noqa: E402
from tools.s45tr.common import now, read, require, save, sha  # noqa: E402,F401

OUT = ROOT / "outputs/s45rt/20261031"
PROTOCOL = OUT / "protocol.json"
MANIFESTS = OUT / "manifests"
PHYSICAL_GPUS = (2, 3, 4, 5, 6, 7)
PAIRS = ((2, 3), (4, 5), (6, 7))
TAG = "S45RT-PREREG-20261031"
S45TR = ROOT / "outputs/s45tr/20261030"
S45CT = ROOT / "outputs/s45ct/20261030"
S45_RUN = ROOT / "outputs/c1sf/20261029/runs/c1sf_so3_s3407"
S45_CONFIG = ROOT / "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml"
S45_EVAL = ROOT / "outputs/c1sf/20261029/evaluations/so3"
T0_CHECKPOINT = S45TR / "t0/t0_final.ckpt"
T0_CHECKPOINT_SHA256 = "f514f70b65fd82e4f38c26305553578a77a53c8ce23aa6cf6613922bf3240949"
STEPS = (5000, 5500, 6000)
#: plan 1 runs these S45-TR files unchanged (their SHAs are frozen by S45-TR and again here)
PLAN1_FILES = ("configs/s45tr_20261030/s45tr_t1b_s3407.yaml", "configs/s45tr_20261030/s45tr_t1b_s3407_preflight.yaml",
               "semkine/s45tr_t1b.py", "semkine/s45tr_t0.py", "semkine/s45tr_t1a.py", "semkine/xag.py",
               "semkine/xyz_mesh.py", "semkine/train.py", "semkine/dataset.py", "semkine/eval_track.py",
               "model/model.py", "tools/s45tr/common.py", "tools/s45tr/train.py", "tools/s45tr/train_xag.py",
               "tools/s45tr/accept.py", "tools/s45tr/evaluate.py", "tools/s45tr/readouts.py", "tools/s45tr/route_acc.py",
               "tools/xyz_c1_repro/train.py", "tools/xcl/evaluate.py", "tools/make_s36_row.py",
               "configs/c1sf_20261029/r2/c1sf_so3_s3407.yaml")
SOURCES = ("tools/s45rt/__init__.py", "tools/s45rt/common.py", "tools/s45rt/chain.py", "tools/s45rt/readouts.py",
           "tools/s45rt/decide.py")


def verify_protocol():
    p = read(PROTOCOL)
    require(p["study"] == "S45RT_20261031" and p["status"] == "FROZEN", "S45-RT protocol identity")
    for name, digest in p["file_sha256"].items():
        require(accepts(ROOT / name, digest), "frozen S45-RT file changed: " + name)
    require(sha(T0_CHECKPOINT) == T0_CHECKPOINT_SHA256, "T0 checkpoint changed")
    for path in sorted(MANIFESTS.glob("*.json")) if MANIFESTS.is_dir() else ():
        for name, digest in read(path)["file_sha256"].items():
            require(accepts(ROOT / name, digest), f"file frozen in {path.name} changed: {name}")
    return p
