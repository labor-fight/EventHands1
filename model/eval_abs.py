#!/usr/bin/env python3
"""Evaluate absolute-pose baselines with MANO-space MPJPE/MPVPE metrics."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config
from mano_layer import ManoLayer
from model import MNISTModel
from pose_repr import decode_to_mano_inputs

# OpenPose-21 finger groups (indices into joints21)
FINGER_JOINTS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}


def root_align(j):
    """Root-align joints/verts. Accepts (..., J, 3) numpy or torch."""
    return j - j[..., :1, :]


@torch.no_grad()
def evaluate_sequence(
    model,
    mano: ManoLayer,
    root: Path,
    split: str,
    seq: str,
    pose_repr: str,
    window_ms: int,
    device: torch.device,
):
    base = root / split / seq
    events = np.load(str(base) + "_events.npy", mmap_mode="r")
    offsets = np.load(str(base) + "_offsets.npy")
    aux = np.load(str(base) + "_aux.npz", allow_pickle=True)
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    gt_ms = np.asarray(aux["gt_ms_idx"], dtype=np.int64)
    gt_valid = np.asarray(aux["gt_valid"], dtype=bool)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    category = str(aux["category"])

    # Load mesh GT for joints/verts at annotation frames
    # Find original mesh path via summary or reconstruct from hand_data
    # We store enough in meta; load mesh from known src if available.
    # Prefer using MANO-forward of GT params51 at annotation times for fair compare,
    # AND also compare to saved joints if we can locate them.
    import struct

    with open(str(base) + ".meta", "rb") as f:
        ncomps, = struct.unpack("<i", f.read(4))
        dt = np.dtype([("data", np.float64, ncomps), ("m0", "u1"), ("m1", "u1")])
        raw = np.fromfile(f, dtype=dt)
    pos51 = raw["data"].astype(np.float32)

    components = mano.hands_components
    hands_mean = mano.hands_mean

    H, W = 180, 240
    pred_joints_all = []
    gt_joints_all = []
    pred_verts_all = []
    gt_verts_all = []
    finger_err = defaultdict(list)

    model.eval()
    for fi, end in enumerate(gt_ms):
        if not gt_valid[fi]:
            continue
        # Need window history inside a valid run
        start = end - window_ms + 1
        if start < 0:
            continue
        ok = False
        for a, b in runs:
            if a <= start and end < b:
                ok = True
                break
        if not ok:
            continue

        # Build LNES
        img = np.zeros((H, W, 2), np.float32)
        a0 = int(offsets[start])
        a1 = int(offsets[end + 1])
        if a1 > a0:
            ev = events[a0:a1]
            counts = np.diff(offsets[start : end + 2]).astype(np.int64)
            frame_rel = np.repeat(np.arange(window_ms, dtype=np.float32), counts)
            tval = frame_rel / float(window_ms)
            xs = ev[:, 0].astype(np.intp)
            ys = ev[:, 1].astype(np.intp)
            ps = np.clip(ev[:, 2].astype(np.intp), 0, 1)
            img[ys, xs, ps] = tval

        x = torch.from_numpy(img).unsqueeze(0).to(device)
        prev = torch.zeros(1, model.output_dim, device=device)
        pred = model(x, prev)

        dec = decode_to_mano_inputs(pred, pose_repr, components, hands_mean)
        pv, pj = mano(
            betas,
            dec["global_orient"],
            dec["local_full_aa"],
            dec["transl"],
        )

        # GT from params51 at this ms (same MANO, same beta)
        g51 = torch.tensor(pos51[end], dtype=torch.float32, device=device).view(1, -1)
        gt_dec = decode_to_mano_inputs(
            # Build a fake 51D or 12D tensor matching pose_repr from meta51
            _meta51_as_target(g51, pose_repr, components),
            pose_repr,
            components,
            hands_mean,
        )
        gv, gj = mano(
            betas,
            gt_dec["global_orient"],
            gt_dec["local_full_aa"],
            gt_dec["transl"],
        )

        pred_joints_all.append(pj.cpu().numpy()[0])
        gt_joints_all.append(gj.cpu().numpy()[0])
        pred_verts_all.append(pv.cpu().numpy()[0])
        gt_verts_all.append(gv.cpu().numpy()[0])

        pj_ra = root_align(pj).cpu().numpy()[0]
        gj_ra = root_align(gj).cpu().numpy()[0]
        for name, idxs in FINGER_JOINTS.items():
            err = np.linalg.norm(pj_ra[idxs] - gj_ra[idxs], axis=-1) * 1000
            finger_err[name].append(err.mean())

    if not pred_joints_all:
        return None

    pj = np.stack(pred_joints_all)
    gj = np.stack(gt_joints_all)
    pv = np.stack(pred_verts_all)
    gv = np.stack(gt_verts_all)

    mpjpe_abs = float(np.linalg.norm(pj - gj, axis=-1).mean() * 1000)
    mpvpe_abs = float(np.linalg.norm(pv - gv, axis=-1).mean() * 1000)
    mpjpe_ra = float(np.linalg.norm(root_align(pj) - root_align(gj), axis=-1).mean() * 1000)
    mpvpe_ra = float(np.linalg.norm(root_align(pv) - root_align(gv), axis=-1).mean() * 1000)

    return {
        "seq": seq,
        "split": split,
        "category": category,
        "n_frames": int(len(pj)),
        "mpjpe_abs_mm": float(mpjpe_abs),
        "mpvpe_abs_mm": float(mpvpe_abs),
        "mpjpe_ra_mm": float(mpjpe_ra),
        "mpvpe_ra_mm": float(mpvpe_ra),
        "per_finger_mpjpe_ra_mm": {k: float(np.mean(v)) for k, v in finger_err.items()},
    }


def _meta51_as_target(g51, pose_repr, components):
    """g51: (1,51) [t,R,residual] -> network layout tensor."""
    if pose_repr == "mano_full_axis_angle":
        return g51
    t = g51[:, 0:3]
    R = g51[:, 3:6]
    residual = g51[:, 6:51]
    pinv = torch.linalg.pinv(components[:6])
    alpha = residual @ pinv
    return torch.cat([alpha, t, R], dim=-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_flops_latency(model, device, output_dim):
    x = torch.randn(1, 180, 240, 2, device=device)
    prev = torch.zeros(1, output_dim, device=device)
    model.eval()
    # Warmup
    for _ in range(20):
        _ = model(x, prev)
    torch.cuda.synchronize()
    t0 = time.time()
    n = 100
    for _ in range(n):
        _ = model(x, prev)
    torch.cuda.synchronize()
    latency_ms = (time.time() - t0) / n * 1000

    flops = None
    try:
        from thop import profile

        # thop needs channels-first; wrap
        class W(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                return self.m(x, torch.zeros(x.shape[0], output_dim, device=x.device))

        w = W(model)
        flops, _ = profile(w, inputs=(x,), verbose=False)
    except Exception:
        try:
            from fvcore.nn import FlopCountAnalysis

            class W(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, x):
                    return self.m(x, torch.zeros(x.shape[0], output_dim, device=x.device))

            flops = FlopCountAnalysis(W(model), x).total()
        except Exception as e:
            flops = None
            print("FLOPs unavailable:", e)
    return latency_ms, flops


def aggregate(results):
    def mean_key(key, subset=None):
        vals = []
        for r in results:
            if subset and r["category"] != subset:
                continue
            vals.append(r[key] * r["n_frames"])
        frames = sum(r["n_frames"] for r in results if (not subset or r["category"] == subset))
        return float(sum(vals) / max(frames, 1))

    out = {
        "overall": {
            "mpjpe_ra_mm": mean_key("mpjpe_ra_mm"),
            "mpvpe_ra_mm": mean_key("mpvpe_ra_mm"),
            "mpjpe_abs_mm": mean_key("mpjpe_abs_mm"),
            "mpvpe_abs_mm": mean_key("mpvpe_abs_mm"),
            "n_frames": sum(r["n_frames"] for r in results),
        },
        "global": {
            "mpjpe_ra_mm": mean_key("mpjpe_ra_mm", "global"),
            "mpvpe_ra_mm": mean_key("mpvpe_ra_mm", "global"),
            "n_frames": sum(r["n_frames"] for r in results if r["category"] == "global"),
        },
        "local": {
            "mpjpe_ra_mm": mean_key("mpjpe_ra_mm", "local"),
            "mpvpe_ra_mm": mean_key("mpvpe_ra_mm", "local"),
            "n_frames": sum(r["n_frames"] for r in results if r["category"] == "local"),
        },
    }
    # per-finger overall
    fingers = {}
    for name in FINGER_JOINTS:
        num = 0.0
        den = 0
        for r in results:
            num += r["per_finger_mpjpe_ra_mm"][name] * r["n_frames"]
            den += r["n_frames"]
        fingers[name] = float(num / max(den, 1))
    out["per_finger_mpjpe_ra_mm"] = fingers
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--window-ms", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel.load_from_checkpoint(args.ckpt, cfg=cfg, map_location=device)
    model = model.to(device).eval()

    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    # decode_to_mano_inputs already adds mean; mano expects full AA
    window = int(args.window_ms or cfg.get("EVAL", {}).get("WINDOW_MS", 100))
    root = Path(cfg["DATA"]["ROOT"])
    with open(root / "splits.json") as f:
        splits = json.load(f)
    trials = splits[args.split]["trials"]

    results = []
    for seq in tqdm(trials, desc="eval"):
        r = evaluate_sequence(
            model,
            mano,
            root,
            args.split,
            seq,
            cfg["MODEL"]["POSE_REPR"],
            window,
            device,
        )
        if r:
            results.append(r)
            print(
                f"{seq}: mpjpe_ra={r['mpjpe_ra_mm']:.2f} mpvpe_ra={r['mpvpe_ra_mm']:.2f} n={r['n_frames']}"
            )

    summary = aggregate(results)
    params = count_params(model)
    latency_ms, flops = measure_flops_latency(model, device, model.output_dim)
    summary["params"] = int(params)
    summary["flops"] = None if flops is None else int(flops)
    summary["latency_ms"] = float(latency_ms)
    summary["pose_repr"] = cfg["MODEL"]["POSE_REPR"]
    summary["output_dim"] = cfg["MODEL"]["OUTPUT_DIM"]
    summary["checkpoint"] = str(args.ckpt)
    summary["per_sequence"] = results

    out_dir = Path(cfg["EVAL"]["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "metrics.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Markdown table row
    md = out_dir / "metrics.md"
    md.write_text(
        f"| {cfg['MODEL']['POSE_REPR']} | {cfg['MODEL']['OUTPUT_DIM']} | "
        f"{summary['overall']['mpjpe_ra_mm']:.2f} | {summary['overall']['mpvpe_ra_mm']:.2f} | "
        f"{summary['local']['mpjpe_ra_mm']:.2f} | {summary['global']['mpjpe_ra_mm']:.2f} | "
        f"{summary['params']} | {summary['flops']} | {summary['latency_ms']:.3f} |\n"
    )
    print(json.dumps({k: summary[k] for k in summary if k != "per_sequence"}, indent=2))
    print("wrote", out_json)


if __name__ == "__main__":
    main()
