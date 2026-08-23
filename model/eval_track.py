#!/usr/bin/env python3
"""
Recursive tracking evaluation for the delta model (and same-protocol baselines).

Protocol: for each valid run of a val sequence, initialize the state from GT at
the run start plus gaussian noise (emulating RGB-based init), then chain
`prevpos <- previous prediction` with a fixed step (= LNES window). Absolute
models (PREDICT_DELTA=false) run the exact same loop -- they simply ignore
prevpos -- giving a same-protocol baseline.

Usage:
  CUDA_VISIBLE_DEVICES=4 python model/eval_track.py \
      --config configs/eventhands_track_delta51.yaml \
      --ckpt outputs/hand_data51/track_delta51/last.ckpt --step-ms 50

  # aggregate report from multiple metrics json files
  python model/eval_track.py --report out1.json out2.json --report-out report.md
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config
from mano_layer import ManoLayer
from model import MNISTModel
from pose_repr import decode_to_mano_inputs

H, W = 180, 240
DRIFT_BUCKET_MS = 5000
STATIC_THRESH_MM = 0.5  # GT mean-joint displacement per step below this = static


def load_sequence(root: Path, split: str, seq: str):
    base = root / split / seq
    events = np.load(str(base) + "_events.npy", mmap_mode="r")
    offsets = np.load(str(base) + "_offsets.npy")
    aux = np.load(str(base) + "_aux.npz", allow_pickle=True)
    with open(str(base) + ".meta", "rb") as f:
        ncomps, = struct.unpack("<i", f.read(4))
        dt = np.dtype([("data", np.float64, ncomps), ("m0", "u1"), ("m1", "u1")])
        raw = np.fromfile(f, dtype=dt)
    pos51 = raw["data"].astype(np.float32)
    return events, offsets, aux, pos51


def build_lnes(events, offsets, end, window):
    img = np.zeros((H, W, 2), np.float32)
    start = end - window + 1
    a0 = int(offsets[start])
    a1 = int(offsets[end + 1])
    if a1 > a0:
        ev = events[a0:a1]
        counts = np.diff(offsets[start : end + 2]).astype(np.int64)
        frame_rel = np.repeat(np.arange(window, dtype=np.float32), counts)
        tval = frame_rel / float(window)
        img[ev[:, 1].astype(np.intp), ev[:, 0].astype(np.intp),
            np.clip(ev[:, 2].astype(np.intp), 0, 1)] = tval
    return img


def sample_init_noise(cfg, rng, scale):
    track = cfg.get("TRACK", {}) or {}
    sig_t = float(track.get("PREV_NOISE_T", 0.0)) * scale
    sig_r = float(track.get("PREV_NOISE_R", 0.0)) * scale
    sig_pose = float(track.get("PREV_NOISE_POSE", 0.0)) * scale
    noise = np.zeros(51, np.float32)
    noise[0:3] = rng.standard_normal(3) * sig_t
    noise[3:6] = rng.standard_normal(3) * sig_r
    noise[6:51] = rng.standard_normal(45) * sig_pose
    return noise


def root_align(x):
    return x - x[..., :1, :]


@torch.no_grad()
def track_sequence(model, mano, cfg, root, split, seq, step_ms, device, rng, noise_scale):
    events, offsets, aux, pos51 = load_sequence(root, split, seq)
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, camera_K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    category = str(aux["category"])
    pose_repr = cfg["MODEL"]["POSE_REPR"]
    assert pose_repr == "mano_full_axis_angle", "tracking eval expects 51D layout"

    preds, gts, elapsed, run_ids = [], [], [], []
    for run_id, (a, b) in enumerate(runs):
        ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if len(ends) == 0:
            continue
        prev = pos51[a].copy() + sample_init_noise(cfg, rng, noise_scale)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        for end in ends:
            img = build_lnes(events, offsets, int(end), step_ms)
            x = torch.from_numpy(img).unsqueeze(0).to(device)
            pred = model(x, prev_t)
            prev_t = pred
            preds.append(pred.cpu().numpy()[0])
            gts.append(pos51[end])
            elapsed.append(int(end - a))
            run_ids.append(run_id)

    if not preds:
        return None
    preds = np.stack(preds)
    gts = np.stack(gts)
    elapsed = np.asarray(elapsed, dtype=np.int64)
    run_ids = np.asarray(run_ids, dtype=np.int64)

    # Batched MANO forward for preds and GTs
    def mano_fk(params):
        out_j, out_v = [], []
        for i0 in range(0, len(params), 2048):
            chunk = torch.from_numpy(params[i0 : i0 + 2048]).to(device)
            dec = decode_to_mano_inputs(chunk, "mano_full_axis_angle",
                                        mano.hands_components, mano.hands_mean)
            v, j = mano(betas.expand(len(chunk), -1), dec["global_orient"],
                        dec["local_full_aa"], dec["transl"])
            out_j.append(j.cpu().numpy())
            out_v.append(v.cpu().numpy())
        return np.concatenate(out_j), np.concatenate(out_v)

    pj, pv = mano_fk(preds)
    gj, gv = mano_fk(gts)

    err_j_abs = np.linalg.norm(pj - gj, axis=-1).mean(-1) * 1000  # (N,)
    err_v_abs = np.linalg.norm(pv - gv, axis=-1).mean(-1) * 1000
    err_j_ra = np.linalg.norm(root_align(pj) - root_align(gj), axis=-1).mean(-1) * 1000
    err_v_ra = np.linalg.norm(root_align(pv) - root_align(gv), axis=-1).mean(-1) * 1000

    # Drift curve: bucket by elapsed time since init
    buckets = {}
    for bkt in np.unique(elapsed // DRIFT_BUCKET_MS):
        m = (elapsed // DRIFT_BUCKET_MS) == bkt
        buckets[int(bkt * DRIFT_BUCKET_MS)] = {
            "mpjpe_ra_mm": float(err_j_ra[m].mean()),
            "mpjpe_abs_mm": float(err_j_abs[m].mean()),
            "n": int(m.sum()),
        }

    # Jitter: consecutive-step prediction movement (same run), split by GT static
    jitter_all, jitter_static, gt_move = [], [], []
    for r in np.unique(run_ids):
        idx = np.where(run_ids == r)[0]
        if len(idx) < 2:
            continue
        dp = np.linalg.norm(pj[idx[1:]] - pj[idx[:-1]], axis=-1).mean(-1) * 1000
        dg = np.linalg.norm(gj[idx[1:]] - gj[idx[:-1]], axis=-1).mean(-1) * 1000
        jitter_all.append(dp)
        gt_move.append(dg)
        jitter_static.append(dp[dg < STATIC_THRESH_MM])
    jitter_all = np.concatenate(jitter_all) if jitter_all else np.zeros(0)
    jitter_static = np.concatenate(jitter_static) if jitter_static else np.zeros(0)
    gt_move = np.concatenate(gt_move) if gt_move else np.zeros(0)

    return {
        "seq": seq,
        "category": category,
        "n_frames": int(len(preds)),
        "n_runs": int(len(np.unique(run_ids))),
        "mpjpe_ra_mm": float(err_j_ra.mean()),
        "mpvpe_ra_mm": float(err_v_ra.mean()),
        "mpjpe_abs_mm": float(err_j_abs.mean()),
        "mpvpe_abs_mm": float(err_v_abs.mean()),
        "drift": buckets,
        "jitter_all_mm_per_step": float(jitter_all.mean()) if len(jitter_all) else None,
        "jitter_static_mm_per_step": (
            float(jitter_static.mean()) if len(jitter_static) else None
        ),
        "n_static_steps": int(len(jitter_static)),
        "gt_move_mm_per_step": float(gt_move.mean()) if len(gt_move) else None,
    }


def run_eval(args):
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel.load_from_checkpoint(args.ckpt, cfg=cfg, map_location=device)
    model = model.to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    with open(root / "splits.json") as f:
        splits = json.load(f)
    rng = np.random.default_rng(args.seed)

    results = []
    for seq in splits[args.split]["trials"]:
        r = track_sequence(
            model, mano, cfg, root, args.split, seq, args.step_ms, device, rng,
            args.init_noise_scale,
        )
        if r:
            results.append(r)
            print(
                f"{seq}: mpjpe_ra={r['mpjpe_ra_mm']:.2f} mpvpe_ra={r['mpvpe_ra_mm']:.2f} "
                f"jitter_static={r['jitter_static_mm_per_step']} n={r['n_frames']}"
            )

    n = sum(r["n_frames"] for r in results)
    summary = {
        "mode": "track" if bool(cfg["MODEL"].get("PREDICT_DELTA", False)) else "absolute",
        "run_name": cfg["TRAIN"].get("RUN_NAME", "?"),
        "checkpoint": str(args.ckpt),
        "step_ms": args.step_ms,
        "init_noise_scale": args.init_noise_scale,
        "overall": {
            k: float(sum(r[k] * r["n_frames"] for r in results) / max(n, 1))
            for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
        },
        "per_sequence": results,
    }
    summary["overall"]["n_frames"] = n

    out_dir = Path(args.out_dir or cfg["EVAL"]["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"track_metrics_step{args.step_ms}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in summary if k != "per_sequence"}, indent=2))
    print("wrote", out_json)
    return out_json


def run_report(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = []
    for p in args.report:
        d = json.loads(Path(p).read_text())
        entries.append(d)

    md = ["# Tracking vs absolute baselines (recursive protocol)", ""]
    md.append(
        "| model | mode | step(ms) | MPJPE RA | MPVPE RA | MPJPE abs | "
        "jitter static (mm/step) | jitter all |"
    )
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for d in entries:
        js = [r.get("jitter_static_mm_per_step") for r in d["per_sequence"]]
        ja = [r.get("jitter_all_mm_per_step") for r in d["per_sequence"]]
        js = np.mean([v for v in js if v is not None]) if any(v is not None for v in js) else float("nan")
        ja = np.mean([v for v in ja if v is not None]) if any(v is not None for v in ja) else float("nan")
        o = d["overall"]
        md.append(
            f"| {d['run_name']} | {d['mode']} | {d['step_ms']} | {o['mpjpe_ra_mm']:.2f} | "
            f"{o['mpvpe_ra_mm']:.2f} | {o['mpjpe_abs_mm']:.2f} | {js:.3f} | {ja:.3f} |"
        )
    md.append("")

    # Drift curves per sequence, one subplot per sequence, one line per entry
    seqs = sorted({r["seq"] for d in entries for r in d["per_sequence"]})
    fig, axes = plt.subplots(1, len(seqs), figsize=(6 * len(seqs), 4), squeeze=False)
    for si, seq in enumerate(seqs):
        ax = axes[0][si]
        for d in entries:
            r = next((r for r in d["per_sequence"] if r["seq"] == seq), None)
            if not r:
                continue
            xs = sorted(int(k) for k in r["drift"].keys())
            ys = [r["drift"][str(x)]["mpjpe_ra_mm"] for x in xs]
            ax.plot([x / 1000.0 for x in xs], ys, marker="o", label=f"{d['run_name']} ({d['mode']})")
        ax.set_title(f"{seq} drift")
        ax.set_xlabel("time since init (s)")
        ax.set_ylabel("RA-MPJPE (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_md = Path(args.report_out or "outputs/hand_data51/report_track_delta.md")
    out_png = out_md.with_suffix(".png")
    fig.savefig(out_png, dpi=120)
    md.append(f"![drift]({out_png.name})")
    md.append("")
    out_md.write_text("\n".join(md) + "\n")
    print("wrote", out_md, "and", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--ckpt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--init-noise-scale", type=float, default=1.0,
                    help="scale on TRACK.PREV_NOISE_* for the init state (0=GT init)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--report", nargs="*", default=None,
                    help="aggregate metrics json files into a markdown report")
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args()

    if args.report:
        run_report(args)
    else:
        assert args.config and args.ckpt, "--config and --ckpt required for eval"
        run_eval(args)


if __name__ == "__main__":
    main()
