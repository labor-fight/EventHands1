#!/usr/bin/env python3
"""
Render side-by-side videos: left = LNES visualization, right = predicted MANO mesh.

Four videos (table rows): {abs_pca12, abs_full51} x {zgz_local, zgz_global}.

Usage:
  CUDA_VISIBLE_DEVICES=4 python tools/render_videos.py --all
  CUDA_VISIBLE_DEVICES=4 python tools/render_videos.py \
      --config configs/eventhands_abs_pca12.yaml --ckpt <ckpt> --seq zgz_local
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import PolyCollection
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

from config import load_config
from mano_layer import ManoLayer
from model import MNISTModel
from pose_repr import decode_to_mano_inputs

FFMPEG = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/ffmpeg"
H, W = 180, 240


def select_frames(gt_ms, gt_valid, runs, window_ms):
    """Same frame filter as eval_abs.evaluate_sequence."""
    keep = []
    for fi, end in enumerate(gt_ms):
        if not gt_valid[fi]:
            continue
        start = end - window_ms + 1
        if start < 0:
            continue
        if any(a <= start and end < b for a, b in runs):
            keep.append(int(end))
    return keep


def build_lnes(events, offsets, end, window_ms):
    img = np.zeros((H, W, 2), np.float32)
    start = end - window_ms + 1
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
    return img


def lnes_to_rgb(img, scale):
    """pos polarity -> green, neg -> red, brightness = recency."""
    rgb = np.zeros((H, W, 3), np.float32)
    rgb[..., 0] = img[..., 0]
    rgb[..., 1] = img[..., 1]
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rgb = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    return rgb


class MeshRenderer:
    """Painter's-algorithm mesh render with matplotlib Agg (no pyrender/cv2)."""

    def __init__(self, K, faces, scale):
        self.w, self.h = W * scale, H * scale
        s = 240.0 / 640.0 * scale  # 0.375 * scale, same for x and y
        self.fx, self.fy = K[0, 0] * s, K[1, 1] * s
        self.cx, self.cy = K[0, 2] * s, K[1, 2] * s
        self.faces = faces  # (F, 3)
        self.light = np.array([0.35, -0.35, -0.87])
        self.light /= np.linalg.norm(self.light)
        self.base = np.array([0.62, 0.71, 0.92])

        dpi = 100
        self.fig = plt.figure(figsize=(self.w / dpi, self.h / dpi), dpi=dpi)
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, self.w)
        self.ax.set_ylim(self.h, 0)
        self.ax.axis("off")
        self.fig.patch.set_facecolor("#101014")

    def render(self, verts):
        """verts: (778, 3) camera coords (meters, +Z forward) -> (h, w, 3) uint8."""
        tri = verts[self.faces]  # (F, 3, 3)
        z = tri[..., 2].mean(axis=1)
        vis = z > 1e-6
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        shade = 0.30 + 0.70 * np.abs(n @ self.light)
        colors = np.clip(self.base[None] * shade[:, None], 0, 1)

        u = self.fx * tri[..., 0] / tri[..., 2] + self.cx
        v = self.fy * tri[..., 1] / tri[..., 2] + self.cy
        poly = np.stack([u, v], axis=-1)  # (F, 3, 2)

        order = np.argsort(-z[vis])
        for coll in list(self.ax.collections):
            coll.remove()
        pc = PolyCollection(
            poly[vis][order],
            facecolors=colors[vis][order],
            edgecolors="none",
            antialiaseds=False,
        )
        self.ax.add_collection(pc)
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[..., :3]
        return buf


def find_font():
    try:
        path = font_manager.findfont("DejaVu Sans")
        return ImageFont.truetype(path, 15)
    except Exception:
        return ImageFont.load_default()


def render_video(cfg_path, ckpt, seq, split, window_ms, scale, fps, out_path, batch=512):
    cfg = load_config(cfg_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel.load_from_checkpoint(ckpt, cfg=cfg, map_location=device)
    model = model.to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    pose_repr = cfg["MODEL"]["POSE_REPR"]
    run_name = cfg["TRAIN"]["RUN_NAME"]

    root = Path(cfg["DATA"]["ROOT"])
    base = root / split / seq
    events = np.load(str(base) + "_events.npy", mmap_mode="r")
    offsets = np.load(str(base) + "_offsets.npy")
    aux = np.load(str(base) + "_aux.npz", allow_pickle=True)
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    gt_ms = np.asarray(aux["gt_ms_idx"], dtype=np.int64)
    gt_valid = np.asarray(aux["gt_valid"], dtype=bool)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    category = str(aux["category"])
    K = np.asarray(aux["camera_K"], dtype=np.float64)

    frames = select_frames(gt_ms, gt_valid, runs, window_ms)
    if not frames:
        print(f"[{run_name} x {seq}] no frames, skip")
        return

    renderer = MeshRenderer(K, mano.f.cpu().numpy(), scale)
    font = find_font()
    header = 28
    pw, ph = W * scale, H * scale
    tw, th = pw * 2, ph + header

    proc = subprocess.Popen(
        [
            FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{tw}x{th}",
            "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )

    title_left = f"LNES {window_ms}ms"
    title_right = f"MANO pred  {run_name}  {seq} ({category})"

    with torch.no_grad():
        for i0 in tqdm(range(0, len(frames), batch), desc=f"{run_name} x {seq}"):
            chunk = frames[i0 : i0 + batch]
            imgs = np.stack([build_lnes(events, offsets, e, window_ms) for e in chunk])
            x = torch.from_numpy(imgs).to(device)
            prev = torch.zeros(len(chunk), model.output_dim, device=device)
            pred = model(x, prev)
            dec = decode_to_mano_inputs(pred, pose_repr, mano.hands_components, mano.hands_mean)
            verts, _ = mano(
                betas.expand(len(chunk), -1),
                dec["global_orient"],
                dec["local_full_aa"],
                dec["transl"],
            )
            verts = verts.float().cpu().numpy()

            for bi, end in enumerate(chunk):
                left = lnes_to_rgb(imgs[bi], scale)
                right = renderer.render(verts[bi])
                canvas = np.zeros((th, tw, 3), np.uint8)
                canvas[:] = 16
                canvas[header:, :pw] = left
                canvas[header:, pw:] = right
                im = Image.fromarray(canvas)
                d = ImageDraw.Draw(im)
                d.text((6, 6), title_left, fill=(235, 235, 235), font=font)
                d.text((pw + 6, 6), title_right, fill=(235, 235, 235), font=font)
                d.text(
                    (tw - 92, 6), f"t={end / 1000.0:6.2f}s", fill=(180, 180, 180), font=font
                )
                proc.stdin.write(np.asarray(im).tobytes())

    proc.stdin.close()
    proc.wait()
    print(f"wrote {out_path} ({len(frames)} frames)")


def best_ckpt(run_dir):
    pat = re.compile(r"val_loss=([0-9.]+)\.ckpt$")
    best, best_v = None, float("inf")
    for p in Path(run_dir).glob("*.ckpt"):
        m = pat.search(p.name)
        if m and float(m.group(1)) < best_v:
            best, best_v = p, float(m.group(1))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="render all 4 combos")
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--seq")
    ap.add_argument("--split", default="val")
    ap.add_argument("--window-ms", type=int, default=100)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--out-dir", default="outputs/hand_data51/videos")
    args = ap.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        combos = []
        for cfg_path in (
            ROOT / "configs/eventhands_abs_pca12.yaml",
            ROOT / "configs/eventhands_abs_full51.yaml",
        ):
            cfg = load_config(cfg_path)
            ck = best_ckpt(ROOT / cfg["TRAIN"]["OUTPUT_DIR"])
            assert ck is not None, f"no ckpt in {cfg['TRAIN']['OUTPUT_DIR']}"
            for seq in ("zgz_local", "zgz_global"):
                combos.append((cfg_path, ck, seq, cfg["TRAIN"]["RUN_NAME"]))
        for cfg_path, ck, seq, run_name in combos:
            out = out_dir / f"{run_name}_{seq}.mp4"
            render_video(
                cfg_path, str(ck), seq, args.split, args.window_ms, args.scale, args.fps, out
            )
    else:
        assert args.config and args.ckpt and args.seq
        cfg = load_config(args.config)
        out = out_dir / f"{cfg['TRAIN']['RUN_NAME']}_{args.seq}.mp4"
        render_video(
            args.config, args.ckpt, args.seq, args.split, args.window_ms, args.scale,
            args.fps, out,
        )


if __name__ == "__main__":
    main()
