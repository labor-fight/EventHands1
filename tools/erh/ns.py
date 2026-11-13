"""EvRealHands study namespace (docs/EXPERIMENT_RECORDS.md#erh-prereg-20261111).

User 2026-11-11: "/data1/lyq/code/mesh/data1/EvRealHands这个数据也是，按照EventHands，EvHandPose，EventHPE，FastMETRO-Event，
EvHandTrack全跑一遍" -- train and test the five methods on EvRealHands exactly as on the real DAVIS346 data. Every real-study
tool runs unchanged in this namespace: `install()` re-targets tools/real/ns.py (data roots, split manifest, test / dev
sequence lists, output and config directories, study and section names) before any real-study module reads it, so the
CMP arms (tools/real/train.py, run.py, the per-arm canvases n346 / n256, FastMETRO's padding, EvHandPose's crop mean from
this study's trans_means.json) and EvHandTrack (tools/real/evht*.py, xag_build.py) read EvRealHands.

Data (tools/erh/convert.py): /data1/lyq/code/mesh/EventHands/data/erh_n346 (346 x 260, undistorted) and erh_n256
(256 x 192), splits_erh.json = the official EvHandPose pretrain.yml split (train subjects 0 2 4 5 6 8 9, dev subject 7,
test subjects 1 3; both hands, left hands mirrored; the four 2-D-only fast trials excluded).
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/data1/lyq/code/mesh/EventHands/data")
CANVAS = {"n346": {"root": DATA / "erh_n346", "W": 346, "H": 260},
          "n256": {"root": DATA / "erh_n256", "W": 256, "H": 192}}
OUT = ROOT / "outputs/erh/20261111"
CONFIGS = ROOT / "configs/erh_20261111"
STUDY = "ERH_20261111"
SECTION = "erh-prereg-20261111"
SPLITS = "splits_erh.json"
#: data workers per CMP training (the real study: 10): host memory is the binding resource (#real-amend5-run-20261111);
#: a loader setting, the recipes are unchanged
NUM_WORKERS = 6
#: modules that bind tools/real/ns.py values at import and do not install this namespace themselves: they must not be
#: imported before `install()` (tools/real/evht.py and tools/real/xag_build.py call `install()` first thing when the
#: environment has ERH_NAMESPACE=1, which tools/erh/run.py sets and a DDP relaunch inherits)
BINDERS = ("tools.real.train", "tools.real.run", "tools.real.table2", "tools.real.evht_jobs", "tools.cmp.arms")


def splits(canvas):
    return CANVAS[canvas]["root"] / SPLITS


def install():
    import tools.real.ns as R
    if getattr(R, "ERH", False):
        return R
    bound = [m for m in BINDERS if m in sys.modules]
    if R._ENTERED or bound:
        raise RuntimeError(f"tools.real.ns was used before the EvRealHands namespace: entered={R._ENTERED}, {bound}")
    R.CANVAS, R.OUT, R.CONFIGS, R.STUDY = CANVAS, OUT, CONFIGS, STUDY
    R.NUM_WORKERS = NUM_WORKERS
    R.SECTION = R.AMENDMENT = SECTION
    R.splits = splits
    m = json.loads(splits("n346").read_text())
    R.TEST, R.DEV = tuple(m["test"]["trials"]), tuple(m["val"]["trials"])
    R.ERH = True
    return R
