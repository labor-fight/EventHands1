#!/usr/bin/env python3
"""EvHandPose's dataset-dependent output offsets for the real data (docs/EXPERIMENT_RECORDS.md#real-prereg-20261110).

The official EvHandPose predictor decodes translation as raw x 0.3 + the training-set mean. The CMP ports hard-code the
hand_data51 values: semkine/cmp_evhp.py TRANS_MEAN (full frame; unused by the real study's arms, whose full-frame stage
epf is FlowNet only) and semkine/cmp_evhpc.py TRANS_MEAN_CROP = (-(c_x - 64) z' / f_x, -(c_y - 64) z' / f_y, z'), the
crop-frame translation of a centred hand at the crop-frame depth z' = Z / scale (scale = 128 / (2 x the larger side of the
2D joint extent), event-frame intrinsics). This tool measures z' over the training labels of a data root (every 10th
annotated frame of the base recordings) with the same definitions, on hand_data51 as a check of the definition (the port
uses 0.52) and on the real data, and writes outputs/real/20261110/trans_means.json.

The definition does not reproduce the port's constant (hand_data51 measures 0.58, the port uses 0.52), so the value
used is the port's constant scaled by the measured real / sim ratio: z'_real = 0.52 x z'_real,measured / z'_sim,measured,
with x / y from the same formula and the real event-frame intrinsics ("used").
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

SIM = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
REAL = Path("/data1/lyq/code/mesh/EventHands/data/hand_data_real_n346")   # EvHandPose's canvas (amendment 1)


def measure(root, manifest, render_scale, stride=10):
    import torch
    from mano_layer import ManoLayer
    from semkine.dataset import _read_meta51
    m = json.loads((root / manifest).read_text())["train"]
    mano = ManoLayer(str(ROOT / "assets/mano_right.npz"), add_mean=True)
    zs, Z, cs = [], [], set()
    for seq in m["trials"]:
        if seq.split("_")[-1] in ("v2", "v3", "v4"):
            continue
        base = root / m["legacy_dir"][seq] / seq
        aux = np.load(str(base) + "_aux.npz")
        P = _read_meta51(str(base) + ".meta")
        idx = np.asarray(aux["gt_ms_idx"])[np.asarray(aux["gt_valid"], bool)][::stride]
        idx = idx[np.asarray(aux["valid_ms"])[idx]]
        K = np.asarray(aux["camera_K"], np.float64) * render_scale
        cs.add(tuple(np.round([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], 4)))
        p = torch.tensor(P[idx])
        b = torch.tensor(np.asarray(aux["betas"], np.float32)[None].repeat(len(idx), 0))
        with torch.no_grad():
            _, j = mano(b, p[:, 3:6], p[:, 6:51], p[:, 0:3])
        j = j.numpy().astype(np.float64)
        u = K[0, 0] * j[..., 0] / j[..., 2] + K[0, 2]
        v = K[1, 1] * j[..., 1] / j[..., 2] + K[1, 2]
        side = 2.0 * np.maximum(u.max(1) - u.min(1), v.max(1) - v.min(1))
        scale = 128.0 / np.maximum(side, 1.0)
        zs.append(P[idx, 2] / scale)
        Z.append(P[idx, 2])
    zs, Z = np.concatenate(zs), np.concatenate(Z)
    (fx, fy, cx, cy), = cs
    zp = float(np.mean(zs))
    return {"frames": int(len(zs)), "z_prime_mean": zp, "translation_z_mean": float(np.mean(Z)),
            "event_frame_K": [fx, fy, cx, cy],
            "trans_mean_crop": [-(cx - 64) * zp / fx, -(cy - 64) * zp / fy, zp]}


def main():
    out = {"sim_check": measure(SIM, "splits_semkine.json", 0.375), "real": measure(REAL, "splits_real.json", 346 / 640),
           "port_constant_sim": [-0.13, -0.06, 0.52]}
    zp = 0.52 * out["real"]["z_prime_mean"] / out["sim_check"]["z_prime_mean"]
    fx, fy, cx, cy = out["real"]["event_frame_K"]
    out["used"] = {"rule": "0.52 x real / sim measured z'", "z_prime": zp,
                   "trans_mean_crop": [-(cx - 64) * zp / fx, -(cy - 64) * zp / fy, zp]}
    p = ROOT / "outputs/real/20261110/trans_means.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
