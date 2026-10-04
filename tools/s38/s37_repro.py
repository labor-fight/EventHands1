#!/usr/bin/env python3
"""S38: S37 must stay bitwise reproducible. Re-evaluates rt_s37_s3407 (last step, GPU, protocol loop) and
compares with the recorded 23.557357022200772 mm (docs/S37_ROOT_TRACKING_VERDICT.md section 9)."""
import sys, json
from pathlib import Path
import numpy as np, torch
REPO = Path("/data1/lyq/code/mesh/EventHands1")
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools"), str(REPO / "tools/tracking")]
import evalx as EX
from config import load_config
from model import MNISTModel
from mano_layer import ManoLayer
from semkine.dataset import sequences_for_split
run = REPO / "outputs/semkine/rt_s37_s3407"
cfg = load_config(next(iter(sorted(run.glob("*.yaml")))))
dev = torch.device("cuda")
ck, step, _ = EX.find_ckpt(run, "last")
m = MNISTModel.load_from_checkpoint(str(ck), cfg=cfg, map_location=dev).to(dev).eval()
mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(dev).eval()
root = Path(cfg["DATA"]["ROOT"]); seqs = sequences_for_split(root, "val_core", Path(cfg["DATA"]["SPLITS_MANIFEST"]))
rng = np.random.default_rng(0); tot, n = 0.0, 0
for s, d in seqs:
    r = EX.run_sequence(m, cfg, root, d, s, dev, rng); mm = EX.per_step_metrics(mano, r, dev)
    tot += mm["mpjpe_ra_mm"].sum(); n += len(mm["mpjpe_ra_mm"])
print(json.dumps({"ckpt": str(ck), "RA": float(tot / n), "reference_gpu": 23.557357022200772, "bitwise": float(tot / n) == 23.557357022200772}))
