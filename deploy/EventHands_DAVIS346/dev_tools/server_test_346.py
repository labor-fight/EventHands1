#!/usr/bin/env python3
"""(Server side, not shipped.) DAVIS346-resolution integration test: zgz events resampled 640x480 -> 346x260
(x * 346 // 640), intrinsics scaled likewise, run through run_davis346.py's SensorMap path, accuracy vs the official
640x480 run on the same steps."""
import glob, json, subprocess, sys
from pathlib import Path
import numpy as np
PKG = Path("/data1/lyq/code/mesh/EventHands1/deploy/EventHands_DAVIS346")
clip = np.load(PKG / "tests" / "zgz_clip.npz")
K = clip["K"].astype(np.float64)
for name, sel in (("local", clip["t_us"] >= 100_000_000), ("global", clip["t_us"] < 100_000_000)):
    t, x, y, p = clip["t_us"][sel].astype(np.int64), clip["x"][sel], clip["y"][sel], clip["p"][sel]
    x346 = (x.astype(np.int32) * 346 // 640).astype(np.uint16)
    y260 = (y.astype(np.int32) * 260 // 480).astype(np.uint16)
    np.savez(f"/tmp/zgz346_{name}.npz", t_us=t, x=x346, y=y260, p=p, width=346, height=260, name="zgz resampled to 346x260")
K346 = K.copy(); K346[0] *= 346 / 640; K346[1] *= 260 / 480
json.dump({"width": 346, "height": 260, "K": K346.tolist(), "count_scale": 1.0}, open("/tmp/calib_zgz346.json", "w"))
print("K346", np.round(K346, 2).tolist())
