"""The real-data study's namespace (docs/EXPERIMENT_RECORDS.md#real-prereg-20261110, #real-amend1-20261110).

CMP's training / reading code (tools/cmp/, semkine/cmp_*.py) runs unchanged in this namespace: `enter()` moves CMP's
output and config directories to outputs/real/20261110 and configs/real_20261110 and makes tools/cmp/arms.py
`build_config` write each arm on the real data in its input form, recording the protocol in a REAL block. Call it before
anything imports tools.cmp.arms; call `set_frame(cfg)` in every process that builds evaluation event images.

Input rule (user 2026-11-10, amendment 1): every method gets the same undistorted DAVIS346 stream at the native
346 x 260 and applies its own official preprocessing -- we downsample nobody's input:
  eho          EventHands as published: LNES of the whole frame -> canvas n346 (346 x 260)
  epxf -> epx  EvHandPose (online-trained crop): the official 346 x 260 frame, 128 x 128 hand crop -> canvas n346
  epf -> ehpe  EventHPE: the official event frame of `--img_size 256` -> canvas n256 (256 x 192, aspect kept; the
               native stream scaled by 256 / 346); epf = EvHandPose's full-frame FlowNet stage that gives EventHPE its
               flow, on EventHPE's frame. 256 x 192 is divisible by the FlowNet's 64, so no padding (PAD = 0).
  fme          (amendment 2) FastMETRO-Event, CMP's deployable form (the official crop needs the annotated hand box):
               the whole frame -> canvas n346, zero-padded to 352 x 288 for ResNet-34's stride 32 (CMP.FME.PAD).
Per canvas: DATA.HEIGHT / WIDTH and MODEL.RENDER_H / RENDER_W = the canvas, MODEL.RENDER_SCALE = canvas / 640 (the aux K
stays on the 640 x 480 grid), and the domain-randomisation shift (written for the 240-wide frame: B2's 14 px, FastMETRO's
0) scaled with the canvas so that it moves the hand by the same fraction of the image.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = Path("/data1/lyq/code/mesh/EventHands/data")
CANVAS = {"n346": {"root": DATA / "hand_data_real_n346", "W": 346, "H": 260},
          "n256": {"root": DATA / "hand_data_real_n256", "W": 256, "H": 192}}
ARM_CANVAS = {"eho": "n346", "epxf": "n346", "epx": "n346", "epf": "n256", "ehpe": "n256", "fme": "n346"}
FME_PAD = [3, 3, 14, 14]               # left, right, top, bottom: 346 x 260 -> 352 x 288
FME_SAVE_EVERY = 4000
OUT = ROOT / "outputs/real/20261110"
CONFIGS = ROOT / "configs/real_20261110"
STUDY = "REAL_20261110"
SECTION = "real-prereg-20261110"
AMENDMENT = "real-amend1-20261110"
ARMS = ("eho", "epxf", "epx", "epf", "ehpe", "fme")
SEEDS = (3407, 3408, 3409)
TEST = tuple(f"{s}_{m}{l}" for s in ("kqj", "lqh") for m in ("global", "local") for l in ("", "_dark"))
DEV = tuple(f"wh_{m}{l}" for m in ("global", "local") for l in ("", "_dark"))
NUM_WORKERS = 10                       # data workers per training (several trainings share one card and the host)
B2_SHIFT_PX, B2_WIDTH = 14.0, 240
_ENTERED = False


def splits(canvas):
    return CANVAS[canvas]["root"] / "splits_real.json"


def trans_mean_crop():
    return [float(x) for x in json.loads((OUT / "trans_means.json").read_text())["used"]["trans_mean_crop"]]


def set_frame(cfg):
    """The evaluator's event-image size (semkine/eval_track.py module constants) for this config's canvas."""
    from semkine import eval_track as ET
    ET.H, ET.W = int(cfg["DATA"]["HEIGHT"]), int(cfg["DATA"]["WIDTH"])


def enter():
    global _ENTERED
    if _ENTERED:
        return
    for p in (str(ROOT / "model"), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    if "tools.cmp.arms" in sys.modules:
        raise RuntimeError("tools.real.ns.enter() must run before tools.cmp.arms is imported")
    import tools.cmp.common as C
    from tools.cmp.common import sha
    C.OUT, C.CONFIGS, C.STUDY, C.SECTION = OUT, CONFIGS, STUDY, SECTION
    import tools.cmp.arms as A
    native = A.build_config
    tmc = trans_mean_crop()
    split_sha = {c: sha(splits(c)) for c in CANVAS}

    def build_config(arm, seed, phase="formal"):
        if arm not in ARMS:
            raise ValueError("not an arm of the real study: " + arm)
        cfg = native(arm, seed, phase)
        c = ARM_CANVAS[arm]
        W, H = CANVAS[c]["W"], CANVAS[c]["H"]
        cfg["DATA"].update(ROOT=str(CANVAS[c]["root"]), SPLITS_MANIFEST=str(splits(c)), HEIGHT=H, WIDTH=W)
        cfg["MODEL"].update(RENDER_H=H, RENDER_W=W, RENDER_SCALE=W / 640.0)
        dr = cfg["AUG"].get("DOMRAND", {})
        if dr.get("ENABLED"):
            dr["SHIFT_PX"] = float(dr["SHIFT_PX"]) * W / B2_WIDTH
        cfg["TRAIN"]["NUM_WORKERS"] = 4 if phase == "smoke" else NUM_WORKERS
        cfg["REAL"] = {"SECTION": SECTION, "AMENDMENT": AMENDMENT, "CANVAS": c, "SPLITS_SHA256": split_sha[c],
                       "CONVERSION": str(CANVAS[c]["root"] / "conversion.json")}
        if arm in ("epxf", "epx"):
            cfg["REAL"]["TRANS_MEAN_CROP"] = tmc
        if arm == "fme":
            cfg["CMP"]["FME"]["PAD"] = list(FME_PAD)
            if phase == "formal":                      # amendment 2: resumable after a preemption (CMP: 48000)
                cfg["TRAIN"]["SAVE_EVERY_N_STEPS"] = FME_SAVE_EVERY
            if (W + FME_PAD[0] + FME_PAD[1]) % 32 or (H + FME_PAD[2] + FME_PAD[3]) % 32:
                raise ValueError("FastMETRO's padded frame must be divisible by 32")
        if arm in ("epf", "ehpe") and (W % 64 or H % 64):
            raise ValueError("EventHPE's frame must be divisible by the FlowNet's 64 (no padding)")
        return cfg
    A.build_config = build_config
    import semkine.cmp_evhp as EH
    import semkine.cmp_ehpe as EP
    import semkine.cmp_evhpc as EC
    import semkine.cmp_evhpx as EX
    EH.PAD = EP.PAD = (0, 0, 0, 0)                     # the full-frame FlowNet runs only on n256 (256 x 192)
    EC.TRANS_MEAN_CROP = EX.TRANS_MEAN_CROP = tuple(tmc)
    _ENTERED = True


def write_configs(phase="formal"):
    enter()
    import tools.cmp.arms as A
    return A.write_configs(phase, list(ARMS))
