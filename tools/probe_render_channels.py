#!/usr/bin/env python3
"""Zero-training probes on the rendered input channels.

The input is LNES(2) + whatever `MODEL.RENDER_CHANNELS` asks the rasterizer for.
Every face is multiplied by the silhouette, so all of them share the same support
and the only thing a face can add over `sil` is its *value* inside the mask.

Two probes, neither of which trains anything:

  --mode stats   per-channel dynamic range inside the mask (from GT poses) and, if
                 a checkpoint is given, the per-input-channel conv1 weight norms.
                 Rasterization does not depend on the weights, so the range part
                 runs on an untrained model and can gate a run before it starts.
  --mode zero    closed-loop RA with some render channels forced to zero, using the
                 exact protocol of model/eval_track.py.

`--mode zero --zero none` reproduces a serial `model/eval_track.py` run bit-for-bit;
it is the control that proves the wrapper is inert. It can differ from a *recorded*
number by ~0.01 mm when the historical evaluation packed several rollouts per GPU
and contention selected different kernels.

Zeroing reads *dependence*, not *necessity*: a zeroed face is an off-distribution
input, so a large jump only says the trained net leans on that channel, never that
the information was needed. The unmasked faces are worse off here than `sil`/`inv`,
whose zero value is in-distribution outside the mask.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from config import load_config  # noqa: E402
from eval_track import load_sequence, track_sequence  # noqa: E402
from mano_layer import ManoLayer  # noqa: E402
from model import MNISTModel  # noqa: E402

# bf16 has 8 mantissa bits: a value near 0.5 resolves to about 2^-9.
BF16_EPS = 2.0 ** -9


def build_model(cfg, ckpt, device):
    if ckpt is None:
        model = MNISTModel(cfg)
    else:
        model = MNISTModel.load_from_checkpoint(ckpt, cfg=cfg, map_location=device)
    return model.to(device).eval()


def channel_slices(model):
    """Name -> slice of the rendered stack, in `MODEL.RENDER_CHANNELS` order."""
    out, off = {}, 0
    for name in model.render_channels:
        k = model.RENDER_FACES[name]
        out[name] = slice(off, off + k)
        off += k
    return out


def zero_render_channels(model, which):
    """Force render channels to zero after rasterization, leaving LNES untouched."""
    if which == "none":
        return
    sl = channel_slices(model)
    names = list(sl) if which == "all" else [n for n in which.split(",") if n]
    bad = [n for n in names if n not in sl]
    if bad:
        raise SystemExit(f"--zero {which}: not rendered by this config: {bad}")
    idx = [sl[n] for n in names]
    inner = model._render_prev

    def wrapped(prevpos, betas, camera_K):
        rend = inner(prevpos, betas, camera_K).clone()
        for s in idx:
            rend[..., s] = 0.0
        return rend

    model._render_prev = wrapped


def _render_as(model, channels, prev, betas, camera_K):
    """Rasterize an arbitrary face set without rebuilding the model."""
    old = model.render_channels
    model.render_channels = tuple(channels)
    try:
        return model._render_prev(prev, betas, camera_K)
    finally:
        model.render_channels = old


@torch.no_grad()
def channel_stats(model, cfg, split, device, max_frames):
    """In-mask dynamic range of every rendered channel, measured on GT poses."""
    root = Path(cfg["DATA"]["ROOT"])
    with open(root / "splits.json") as f:
        splits = json.load(f)
    names = list(model.render_channels)
    out = []
    for seq in splits[split]["trials"]:
        _, _, aux, pos51 = load_sequence(root, split, seq)
        betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
        camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
        runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
        frames = np.concatenate([np.arange(a, b) for a, b in runs])
        stride = max(len(frames) // max_frames, 1)
        frames = frames[::stride][:max_frames]

        sil_frac = []
        # One accumulator per sub-channel: within-frame std / spread are what a conv
        # can actually read, across-frame std only says the channel moves over time.
        acc = {n: {"mean": [], "std": [], "lo": [], "hi": [], "spread": []} for n in names}
        support_matches = {n: True for n in names}
        for i0 in range(0, len(frames), 256):
            idx = frames[i0 : i0 + 256]
            prev = torch.from_numpy(pos51[idx]).to(device)
            b, k = model._resolve_betas_K(prev, betas.expand(len(idx), -1),
                                          camera_K.expand(len(idx), -1, -1))
            sil = _render_as(model, ["sil"], prev, b, k)[..., 0]
            m = sil > 0
            sil_frac.append(m.reshape(len(idx), -1).float().mean(dim=1).cpu().numpy())
            for n in names:
                face = _render_as(model, [n], prev, b, k)
                lit = face.abs().amax(dim=-1) > 0
                support_matches[n] &= not bool((lit & ~m).any())
                for c in range(face.shape[-1]):
                    ch = face[..., c]
                    for j in range(len(idx)):
                        vals = ch[j][m[j]]
                        if vals.numel() == 0:
                            continue
                        a = acc[n]
                        a["mean"].append(float(vals.mean()))
                        a["std"].append(float(vals.std()) if vals.numel() > 1 else 0.0)
                        a["lo"].append(float(vals.min()))
                        a["hi"].append(float(vals.max()))
                        a["spread"].append(float(vals.max() - vals.min()))

        sil_frac = np.concatenate(sil_frac)
        rec = {
            "seq": seq,
            "n_frames": int(len(frames)),
            "sil_pixel_frac_mean": float(sil_frac.mean()),
            "channels": {},
        }
        for n in names:
            a = {k2: np.asarray(v) for k2, v in acc[n].items()}
            spread = float(a["spread"].mean())
            rec["channels"][n] = {
                "width": model.RENDER_FACES[n],
                "support_inside_sil": support_matches[n],
                "in_mask_mean": float(a["mean"].mean()),
                "in_mask_std_within_frame": float(a["std"].mean()),
                "in_mask_std_across_frames": float(a["mean"].std()),
                "in_mask_min": float(a["lo"].min()),
                "in_mask_max": float(a["hi"].max()),
                "in_mask_spread_mean": spread,
                "spread_over_bf16_eps": spread / BF16_EPS,
            }
        out.append(rec)
        print(json.dumps(rec, indent=2), flush=True)
    return out


def conv1_channel_norms(ckpt, channels):
    """Per-input-channel L2 norm of the learned stem: what the net actually uses."""
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    w = sd["conv1.weight"].float()  # (out, in, 3, 3)
    norms = [float(w[:, c].norm()) for c in range(w.shape[1])]
    lnes = float(np.mean(norms[:2]))
    labels = ["lnes_off", "lnes_on"]
    for name in channels:
        k = MNISTModel.RENDER_FACES[name]
        labels += [name if k == 1 else f"{name}{i}" for i in range(k)]
    out = {"conv1_in_channels": int(w.shape[1]), "norm": {}, "over_lnes_mean": {}}
    for lab, val in zip(labels, norms):
        out["norm"][lab] = val
        if not lab.startswith("lnes"):
            out["over_lnes_mean"][lab] = val / lnes
    return out


def run_zero(args, cfg, device):
    model = build_model(cfg, args.ckpt, device)
    zero_render_channels(model, args.zero)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])
    with open(root / "splits.json") as f:
        splits = json.load(f)
    # Single rng advanced over sequences in split order: identical to eval_track.run_eval.
    rng = np.random.default_rng(args.seed)

    results = []
    for seq in splits[args.split]["trials"]:
        r = track_sequence(model, mano, cfg, root, args.split, seq, args.step_ms,
                           device, rng, args.init_noise_scale)
        if r:
            results.append(r)
            print(f"{seq}: mpjpe_ra={r['mpjpe_ra_mm']:.3f} n={r['n_frames']}", flush=True)

    n = sum(r["n_frames"] for r in results)
    overall = {
        k: float(sum(r[k] * r["n_frames"] for r in results) / max(n, 1))
        for k in ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
    }
    overall["n_frames"] = n
    return {
        "zeroed": args.zero,
        "checkpoint": str(args.ckpt),
        "step_ms": args.step_ms,
        "overall": overall,
        "per_sequence": {r["seq"]: r["mpjpe_ra_mm"] for r in results},
        "jitter_all_mm_per_step": {r["seq"]: r["jitter_all_mm_per_step"] for r in results},
        "jitter_static_mm_per_step": {r["seq"]: r["jitter_static_mm_per_step"] for r in results},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stats", "zero"], required=True)
    ap.add_argument("--config", default="configs/eventhands_track_render51.yaml")
    ap.add_argument("--ckpt", default=None,
                    help="required for --mode zero; optional for --mode stats")
    ap.add_argument("--split", default="val")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--init-noise-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zero", default="none",
                    help="'none', 'all', or a comma-separated list of channel names")
    ap.add_argument("--max-frames", type=int, default=400,
                    help="GT frames sampled per sequence for the stats mode")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "stats":
        model = build_model(cfg, args.ckpt, device)
        payload = {
            "mode": "stats",
            "checkpoint": str(args.ckpt),
            "render_channels": list(model.render_channels),
            "conv1": conv1_channel_norms(args.ckpt, model.render_channels)
            if args.ckpt else None,
            "per_sequence": channel_stats(model, cfg, args.split, device, args.max_frames),
        }
    else:
        if args.ckpt is None:
            raise SystemExit("--mode zero needs --ckpt")
        payload = run_zero(args, cfg, device)
        payload["mode"] = "zero"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print("wrote", out, flush=True)
    print(json.dumps({k: v for k, v in payload.items() if k != "per_sequence"}, indent=2))


if __name__ == "__main__":
    main()
