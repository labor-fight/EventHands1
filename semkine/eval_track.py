#!/usr/bin/env python3
"""S1 extended recursive evaluator: legacy metrics + buckets + delta-trust, one forward pass.

The protocol is unchanged from `model/eval_track.py` and that is the point. Per valid run the
state is initialised once from ground truth plus noise, then chained `prev <- pred`; ground truth
is read again only by the metrics. The S1 hard gate is that with buckets and delta-trust off this
file reproduces the legacy numbers to within 1e-6, so the reform cannot be confused with an
improvement.

What is added:

* per-step metric arrays kept, so any bucket is a mask away and no second forward pass is needed
* bucket metrics from `semkine/buckets.py`
* the delta-trust arm: `prev + rho * (pred - prev)`, an inference-time knob requiring no
  training. It is here because it is the cheap baseline that S9's routing has to beat: shrinking
  every update by a constant is the zero-parameter special case of "trust the measurement less
  when the evidence is weak", and history records it moving 19.26 -> 18.73 mm for free. A router
  that cannot beat it is not earning its complexity.
* multi-subject splits via the SemKine manifest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                     # noqa: E402
from mano_layer import ManoLayer                    # noqa: E402
from model import MNISTModel                       # noqa: E402
from pose_repr import decode_to_mano_inputs         # noqa: E402

from semkine import buckets as BK                   # noqa: E402
from semkine import metrics as MT                   # noqa: E402
from semkine.dataset import _read_meta51, sequences_for_split   # noqa: E402
from semkine.events import EV_COLS, EventPacketBatch            # noqa: E402

H, W = 180, 240
DRIFT_BUCKET_MS = 5000


def load_sequence(root: Path, legacy_dir: str, seq: str):
    base = str(root / legacy_dir / seq)
    return (
        np.load(base + "_events.npy", mmap_mode="r"),
        np.load(base + "_offsets.npy"),
        np.load(base + "_aux.npz", allow_pickle=True),
        _read_meta51(base + ".meta"),
    )


def _window_events(events, offsets, tsub, end: int, window: int):
    """Events of `[end-window+1, end]` as `(N, 5)` [batch=0, x, y, t_rel_s, p] and count."""
    start = end - window + 1
    a0, a1 = int(offsets[start]), int(offsets[end + 1])
    if a1 <= a0:
        return np.zeros((0, EV_COLS), np.float32)
    ev = np.asarray(events[a0:a1])
    counts = np.diff(offsets[start:end + 2]).astype(np.int64)
    ms_rel = np.repeat(np.arange(window, dtype=np.int64), counts)
    us = ms_rel * 1000
    if tsub is not None:
        us = us + np.asarray(tsub[a0:a1]).astype(np.int64)
    out = np.zeros((len(ev), EV_COLS), np.float32)
    out[:, 1] = ev[:, 0]
    out[:, 2] = ev[:, 1]
    out[:, 3] = us.astype(np.float32) * 1e-6
    out[:, 4] = np.clip(ev[:, 2], 0, 1)
    return out


def make_eval_packet(ev5, prev, betas, camera_K, window_ms: int, device) -> EventPacketBatch:
    n = ev5.shape[0]
    return EventPacketBatch(
        events=torch.from_numpy(ev5).to(device),
        ptr=torch.tensor([0, n], dtype=torch.int64, device=device),
        sequence_id=torch.zeros(1, dtype=torch.int64, device=device),
        t_start_us=torch.zeros(1, dtype=torch.int64, device=device),
        t_end_us=torch.tensor([window_ms * 1000], dtype=torch.int64, device=device),
        delta_t_s=torch.tensor([window_ms * 1e-3], dtype=torch.float32, device=device),
        is_sequence_start=torch.zeros(1, dtype=torch.bool, device=device),
        is_sequence_end=torch.zeros(1, dtype=torch.bool, device=device),
        target=prev.new_zeros(prev.shape),
        prev_state=prev,
        betas=betas,
        camera_K=camera_K,
        lnes=None,
    )


def build_lnes(events, offsets, end: int, window: int) -> np.ndarray:
    """Identical to `model/eval_track.build_lnes`; duplicated so parity is exact by inspection."""
    img = np.zeros((H, W, 2), np.float32)
    start = end - window + 1
    a0, a1 = int(offsets[start]), int(offsets[end + 1])
    if a1 > a0:
        ev = events[a0:a1]
        counts = np.diff(offsets[start : end + 2]).astype(np.int64)
        tval = np.repeat(np.arange(window, dtype=np.float32), counts) / float(window)
        img[ev[:, 1].astype(np.intp), ev[:, 0].astype(np.intp),
            np.clip(ev[:, 2].astype(np.intp), 0, 1)] = tval
    return img


def sample_init_noise(cfg, rng, scale: float) -> np.ndarray:
    track = cfg.get("TRACK", {}) or {}
    noise = np.zeros(51, np.float32)
    noise[0:3] = rng.standard_normal(3) * float(track.get("PREV_NOISE_T", 0.0)) * scale
    noise[3:6] = rng.standard_normal(3) * float(track.get("PREV_NOISE_R", 0.0)) * scale
    noise[6:51] = rng.standard_normal(45) * float(track.get("PREV_NOISE_POSE", 0.0)) * scale
    return noise


def root_align(x: np.ndarray) -> np.ndarray:
    return x - x[..., :1, :]


@torch.no_grad()
def track_sequence(model, mano, cfg, root: Path, legacy_dir: str, seq: str, step_ms: int,
                   device, rng, noise_scale: float, delta_trust: float = 1.0,
                   bucket_manifest: Optional[dict] = None,
                   state_hook=None, active_policy=None,
                   window_ms: Optional[int] = None) -> Optional[dict]:
    """`window_ms` decouples the evidence window from the update interval; it defaults to
    `step_ms`, which is the tying LNES imposes and which every recorded number was measured under."""
    events, offsets, aux, pos51 = load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    use_raw = bool(getattr(model, "encoder_name", ""))
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, camera_K)
    if active_policy is not None and hasattr(active_policy, "begin_sequence"):
        active_policy.begin_sequence(betas, camera_K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    assert cfg["MODEL"]["POSE_REPR"] == "mano_full_axis_angle", "tracking eval expects 51D"

    win = int(window_ms or step_ms)
    preds, gts, elapsed, run_ids = [], [], [], []
    for run_id, (a, b) in enumerate(runs):
        ends = np.arange(a + max(step_ms, win) - 1, b, step_ms, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + sample_init_noise(cfg, rng, noise_scale)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        if state_hook is not None:
            state_hook("run_start", prev_t, None)
        if active_policy is not None:
            active_policy.reset()
        prev_end = int(a)
        need_lnes = (not use_raw) or active_policy is not None
        for end in ends:
            # Rasterising LNES for a raw-event arm costs as much as the whole forward pass and is
            # thrown away, which matters once the step-size sweep multiplies the step count by ten.
            x = (torch.from_numpy(build_lnes(events, offsets, int(end), win))
                 .unsqueeze(0).to(device)) if need_lnes else None
            if use_raw:
                ev5 = _window_events(events, offsets, tsub, int(end), win)
                pred = model.forward_packet(make_eval_packet(ev5, prev_t, betas, camera_K,
                                                             win, device))
            else:
                pred = model(x, prev_t)
            if delta_trust != 1.0:
                # Shrink the step toward the prior. rho = 1 is the identity, so the default
                # path is bit-identical to the legacy evaluator.
                pred = prev_t + delta_trust * (pred - prev_t)
            if active_policy is not None:
                # A genuine skip: the retained coordinates keep the previous value and are fed
                # back as such. Multiplying an update by a mask would leave the model's own
                # `prev + delta` path intact and would not test selective updating at all.
                a0 = int(offsets[int(end) - step_ms + 1])
                a1 = int(offsets[int(end) + 1])
                pred = active_policy(prev_t, pred, pos51[prev_end], pos51[int(end)],
                                     n_events=a1 - a0,
                                     events=events[a0:a1] if a1 > a0 else None,
                                     lnes=x)
            prev_end = int(end)
            if state_hook is not None:
                pred = state_hook("step", prev_t, pred)
            prev_t = pred
            preds.append(pred.cpu().numpy()[0])
            gts.append(pos51[end])
            elapsed.append(int(end - a))
            run_ids.append(run_id)

    if not preds:
        return None
    preds, gts = np.stack(preds), np.stack(gts)
    elapsed = np.asarray(elapsed, dtype=np.int64)
    run_ids = np.asarray(run_ids, dtype=np.int64)

    def mano_fk(params):
        oj, ov = [], []
        for i0 in range(0, len(params), 2048):
            chunk = torch.from_numpy(params[i0 : i0 + 2048]).to(device)
            dec = decode_to_mano_inputs(chunk, "mano_full_axis_angle",
                                        mano.hands_components, mano.hands_mean)
            v, j = mano(betas.expand(len(chunk), -1), dec["global_orient"],
                        dec["local_full_aa"], dec["transl"])
            oj.append(j.cpu().numpy())
            ov.append(v.cpu().numpy())
        return np.concatenate(oj), np.concatenate(ov)

    pj, pv = mano_fk(preds)
    gj, gv = mano_fk(gts)
    per_step = {
        "mpjpe_abs_mm": np.linalg.norm(pj - gj, axis=-1).mean(-1) * 1000,
        "mpvpe_abs_mm": np.linalg.norm(pv - gv, axis=-1).mean(-1) * 1000,
        "mpjpe_ra_mm": np.linalg.norm(root_align(pj) - root_align(gj), axis=-1).mean(-1) * 1000,
        "mpvpe_ra_mm": np.linalg.norm(root_align(pv) - root_align(gv), axis=-1).mean(-1) * 1000,
    }

    drift = {}
    for bkt in np.unique(elapsed // DRIFT_BUCKET_MS):
        m = (elapsed // DRIFT_BUCKET_MS) == bkt
        drift[int(bkt * DRIFT_BUCKET_MS)] = {
            "mpjpe_ra_mm": float(per_step["mpjpe_ra_mm"][m].mean()),
            "mpjpe_abs_mm": float(per_step["mpjpe_abs_mm"][m].mean()),
            "n": int(m.sum()),
        }

    jitter, gt_move = [], []
    for r in np.unique(run_ids):
        idx = np.where(run_ids == r)[0]
        if len(idx) < 2:
            continue
        jitter.append(np.linalg.norm(pj[idx[1:]] - pj[idx[:-1]], axis=-1).mean(-1) * 1000)
        gt_move.append(np.linalg.norm(gj[idx[1:]] - gj[idx[:-1]], axis=-1).mean(-1) * 1000)
    jitter = np.concatenate(jitter) if jitter else np.zeros(0)
    gt_move = np.concatenate(gt_move) if gt_move else np.zeros(0)

    out = {
        "seq": seq,
        "category": str(aux["category"]),
        "n_frames": int(len(preds)),
        "n_runs": int(len(np.unique(run_ids))),
        "drift": drift,
        "jitter_all_mm_per_step": float(jitter.mean()) if len(jitter) else None,
        "gt_move_mm_per_step": float(gt_move.mean()) if len(gt_move) else None,
    }
    for k, v in per_step.items():
        out[k] = float(v.mean())

    if bucket_manifest is not None:
        masks = BK.masks_for_sequence(bucket_manifest, seq, len(preds))
        if masks:
            out["buckets"] = MT.bucket_metrics(per_step, masks,
                                               bucket_manifest.get("min_support", 200))
    return out


def run_eval(args) -> Path:
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MNISTModel.load_from_checkpoint(args.ckpt, cfg=cfg, map_location=device)
    model = model.to(device).eval()
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    root = Path(cfg["DATA"]["ROOT"])

    seqs = sequences_for_split(root, args.split,
                               Path(args.manifest) if args.manifest else None)
    bm = None
    if args.buckets:
        p = Path(args.buckets)
        if not p.exists():
            p = root / "buckets" / f"{args.split}_step{args.step_ms}.json"
        if p.exists():
            bm = BK.load_bucket_manifest(p)
            print(f"buckets from {p}")
        else:
            print(f"WARNING no bucket manifest for split={args.split} step={args.step_ms}")

    rng = np.random.default_rng(args.seed)
    results: List[dict] = []
    for seq, legacy_dir in seqs:
        r = track_sequence(model, mano, cfg, root, legacy_dir, seq, args.step_ms, device,
                           rng, args.init_noise_scale, args.delta_trust, bm)
        if r:
            results.append(r)
            print(f"{seq}: RA={r['mpjpe_ra_mm']:.3f} abs={r['mpjpe_abs_mm']:.2f} "
                  f"n={r['n_frames']}", flush=True)

    n = sum(r["n_frames"] for r in results)
    keys = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
    overall = {k: float(sum(r[k] * r["n_frames"] for r in results) / max(n, 1)) for k in keys}
    overall["n_frames"] = n

    cis = {}
    for k in keys:
        vals, wts = MT.per_sequence_table(results, k)
        cis[k] = MT.sequence_bootstrap([vals[s] for s in vals], [wts[s] for s in vals],
                                       seed=args.seed).as_dict()

    bucket_overall = {}
    if bm is not None:
        for b in BK.BUCKET_NAMES:
            rows = [(r["buckets"][b], r["n_frames"]) for r in results
                    if r.get("buckets", {}).get(b, {}).get("n")]
            tot = sum(e["n"] for e, _ in rows)
            if not tot:
                continue
            bucket_overall[b] = {
                "n": tot,
                "underpowered": tot < bm.get("min_support", 200),
                **{k: float(sum(e[k] * e["n"] for e, _ in rows) / tot)
                   for k in keys if all(e.get(k) is not None for e, _ in rows)},
            }

    summary = {
        "arm": args.arm or cfg["TRAIN"].get("RUN_NAME", "?"),
        "mode": "track" if bool(cfg["MODEL"].get("PREDICT_DELTA", False)) else "absolute",
        "run_name": cfg["TRAIN"].get("RUN_NAME", "?"),
        "checkpoint": str(args.ckpt),
        "split": args.split,
        "step_ms": args.step_ms,
        "init_noise_scale": args.init_noise_scale,
        "delta_trust": args.delta_trust,
        "n_sequences": len(results),
        "overall": overall,
        "overall_ci": cis,
        "buckets_overall": bucket_overall,
        "per_sequence": results,
    }
    out_dir = Path(args.out_dir or cfg["EVAL"]["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}_step{args.step_ms}"
    if args.delta_trust != 1.0:
        tag += f"_dt{args.delta_trust:g}"
    out_json = out_dir / f"track_metrics_{tag}.json"
    out_json.write_text(json.dumps(summary, indent=2))

    print(json.dumps({k: summary[k] for k in summary
                      if k not in ("per_sequence", "buckets_overall")}, indent=2))
    if bucket_overall:
        print("\nbuckets:")
        for b, e in sorted(bucket_overall.items(), key=lambda kv: -kv[1]["n"]):
            flag = " (underpowered)" if e["underpowered"] else ""
            print(f"  {b:22s} n={e['n']:6d} RA={e.get('mpjpe_ra_mm', float('nan')):7.3f}{flag}")
    print("wrote", out_json)
    return out_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--init-noise-scale", type=float, default=1.0)
    ap.add_argument("--delta-trust", type=float, default=1.0,
                    help="rho in prev + rho*(pred-prev); 1.0 is the legacy path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--arm", default=None, help="label for the comparison table")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--buckets", nargs="?", const="auto", default=None)
    run_eval(ap.parse_args())


if __name__ == "__main__":
    main()
