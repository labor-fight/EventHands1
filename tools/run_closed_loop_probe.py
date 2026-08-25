#!/usr/bin/env python3
"""Split an arm's tracking error into its single-step part and its closed-loop part.

Recursive tracking error confounds two things: how good one update is, and how badly the updates
compound when the state the model conditions on is its own output rather than the truth. The two
call for opposite remedies, so measuring them separately is worth one cheap pass.

For each arm this runs the frozen evaluator's loop twice at the same step size on the same windows:

  * `teacher_forced` -- `prev` is the ground truth at the previous step, so every prediction sees a
    correct conditioning state and the error is the single-step error alone.
  * `recursive` -- `prev` is the model's own previous prediction, which is the protocol S1 froze and
    the number every gate is written against.

The ratio is the amplification. An arm whose single-step error matches a control's but whose
recursive error does not has a feedback problem, not a representation problem -- which matters here
because KEG conditions its *event routing* on the previous state through KSSF, a path the dense
control does not have, and `FAILURE_CASES.md` registers self-excitation on that path as a risk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from config import load_config                                    # noqa: E402
from mano_layer import ManoLayer                                  # noqa: E402
from model import MNISTModel                                      # noqa: E402
from semkine import eval_track as ET                              # noqa: E402
from semkine.dataset import sequences_for_split                   # noqa: E402

class _DeltaHook:
    """Collects per-step update magnitudes from the recursive evaluator's own loop.

    `track_sequence` calls `state_hook("step", prev_t, pred)` before chaining, so the norms here
    are exactly the updates the closed loop applied -- no second loop, no behaviour change (the
    hook returns `pred` untouched).
    """

    def __init__(self):
        self.pose, self.trans = [], []

    def __call__(self, tag, prev_t, pred):
        if tag == "step" and pred is not None:
            d = (pred - prev_t)[0]
            self.pose.append(float(d[6:].norm().item()))
            self.trans.append(float(d[:3].norm().item()))
        return pred

    def summary(self):
        if not self.pose:
            return None
        return {
            "pred_step_pose": float(np.mean(self.pose)),
            "pred_step_trans": float(np.mean(self.trans)),
        }


def _fk_joints(params: np.ndarray, mano, betas, device) -> np.ndarray:
    oj = []
    for i0 in range(0, len(params), 2048):
        chunk = torch.from_numpy(params[i0: i0 + 2048]).to(device)
        dec = ET.decode_to_mano_inputs(chunk, "mano_full_axis_angle",
                                       mano.hands_components, mano.hands_mean)
        _, j = mano(betas.expand(len(chunk), -1), dec["global_orient"],
                    dec["local_full_aa"], dec["transl"])
        oj.append(j.cpu().numpy())
    return np.concatenate(oj)


def _corrupt(pose: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    """Displace a 51D state by `scale` times the S1 training noise, in its own units.

    The layout is `[trans(3), root_rot(3), pose(45)]`, and the three blocks are perturbed with the
    same ratios `TRACK.PREV_NOISE_*` uses, so a sweep in `scale` moves along the direction the
    training distribution already covers rather than off it.
    """
    out = pose.copy()
    out[:3] += rng.normal(0.0, 0.005 * scale, 3)
    out[3:6] += rng.normal(0.0, 0.05 * scale, 3)
    out[6:] += rng.normal(0.0, 0.05 * scale, out.shape[0] - 6)
    return out


def _run(model, mano, cfg, root, legacy_dir, seq, step_ms, device, teacher_forced,
         prev_noise: float = 0.0, seed: int = 0):
    """`ET.track_sequence` with an optional ground-truth conditioning state.

    Teacher forcing is injected through the evaluator's own `state_hook` rather than by copying the
    loop, so the two regimes share every other detail: the same windows, the same initialisation
    noise draw, the same metric code.
    """
    if not teacher_forced:
        rng = np.random.default_rng(0)
        hook = _DeltaHook()
        r = ET.track_sequence(model, mano, cfg, root, legacy_dir, seq, step_ms,
                              device, rng, 1.0, 1.0, None, state_hook=hook)
        if r is not None:
            r["delta"] = hook.summary()
        return r

    # Teacher forcing needs the conditioning state replaced without touching the recorded
    # prediction, which the hook cannot express, so run the loop here against the same helpers.
    events, offsets, aux, pos51 = ET.load_sequence(root, legacy_dir, seq)
    tsub_path = root / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    use_raw = bool(getattr(model, "encoder_name", ""))
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    camera_K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(model, "set_hand_context"):
        model.set_hand_context(betas, camera_K)
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)

    rng = np.random.default_rng(seed)
    preds, gts = [], []
    # Update-magnitude bookkeeping for the under-update hypothesis (R4): what the network moved
    # versus what it needed to move, both measured against the same conditioning state.
    dp_pose, dg_pose, dp_tr, dg_tr, dots = [], [], [], [], []
    cond_pairs = []
    for a, b in runs:
        ends = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if not len(ends):
            continue
        prev_idx = int(a)
        for end in ends:
            prev_gt = pos51[prev_idx].copy()
            prev = prev_gt
            if prev_noise:
                prev = _corrupt(prev_gt, rng, prev_noise)
                if len(cond_pairs) < 256:
                    cond_pairs.append((prev_gt, prev.copy()))
            prev_t = torch.from_numpy(prev).view(1, -1).to(device)
            with torch.no_grad():
                if use_raw:
                    ev5 = ET._window_events(events, offsets, tsub, int(end), step_ms)
                    pred = model.forward_packet(ET.make_eval_packet(
                        ev5, prev_t, betas, camera_K, step_ms, device))
                else:
                    x = torch.from_numpy(
                        ET.build_lnes(events, offsets, int(end), step_ms)).unsqueeze(0).to(device)
                    pred = model(x, prev_t)
            p = pred.cpu().numpy()[0]
            d_pred, d_need = p - prev, pos51[end] - prev
            dp_pose.append(np.linalg.norm(d_pred[6:]))
            dg_pose.append(np.linalg.norm(d_need[6:]))
            dp_tr.append(np.linalg.norm(d_pred[:3]))
            dg_tr.append(np.linalg.norm(d_need[:3]))
            # Projection of the applied update onto the needed one: 1 is a perfect step, <1 is
            # under-update, and it separates "too small" from "wrong direction" where the norm
            # ratio alone cannot.
            dots.append(float(np.dot(d_pred[6:], d_need[6:]) /
                              max(np.dot(d_need[6:], d_need[6:]), 1e-12)))
            preds.append(p)
            gts.append(pos51[end])
            prev_idx = int(end)
    if not preds:
        return None
    out = _metrics(np.stack(preds), np.stack(gts), mano, betas, device)
    out["delta"] = {
        "pred_step_pose": float(np.mean(dp_pose)),
        "need_step_pose": float(np.mean(dg_pose)),
        "update_ratio_pose": float(np.mean(dp_pose) / max(np.mean(dg_pose), 1e-9)),
        "alignment_pose": float(np.mean(dots)),
        "pred_step_trans": float(np.mean(dp_tr)),
        "need_step_trans": float(np.mean(dg_tr)),
    }
    if cond_pairs:
        # The noise scale is a training-units knob; convert it to millimetres of conditioning
        # state error through FK so the sensitivity curve and the recursive steady state live on
        # the same axis and the fixed-point comparison (R2) is direct.
        a51 = np.stack([p[0] for p in cond_pairs])
        b51 = np.stack([p[1] for p in cond_pairs])
        ja, jb = (_fk_joints(x, mano, betas, device) for x in (a51, b51))
        out["cond_err_ra_mm"] = float(np.linalg.norm(
            ET.root_align(ja) - ET.root_align(jb), axis=-1).mean() * 1000)
        out["cond_err_abs_mm"] = float(np.linalg.norm(ja - jb, axis=-1).mean() * 1000)
    return out


def _metrics(preds, gts, mano, betas, device):
    """RA- and absolute MPJPE, defined exactly as `track_sequence` defines them."""
    def fk(params):
        oj = []
        for i0 in range(0, len(params), 2048):
            chunk = torch.from_numpy(params[i0: i0 + 2048]).to(device)
            dec = ET.decode_to_mano_inputs(chunk, "mano_full_axis_angle",
                                           mano.hands_components, mano.hands_mean)
            _, j = mano(betas.expand(len(chunk), -1), dec["global_orient"],
                        dec["local_full_aa"], dec["transl"])
            oj.append(j.cpu().numpy())
        return np.concatenate(oj)

    pj, gj = fk(preds), fk(gts)
    return {
        "n_frames": int(len(preds)),
        "mpjpe_abs_mm": float(np.linalg.norm(pj - gj, axis=-1).mean() * 1000),
        "mpjpe_ra_mm": float(np.linalg.norm(ET.root_align(pj) - ET.root_align(gj),
                                            axis=-1).mean() * 1000),
    }


def _pool(per, extra_keys=()) -> dict:
    """Frame-weighted pooling of per-sequence rows, including the delta bookkeeping."""
    n = sum(r["n_frames"] for r in per)
    out = {
        "ra_mm": sum(r["mpjpe_ra_mm"] * r["n_frames"] for r in per) / n,
        "abs_mm": sum(r["mpjpe_abs_mm"] * r["n_frames"] for r in per) / n,
        "n_frames": n,
    }
    for k in extra_keys:
        vals = [(r[k], r["n_frames"]) for r in per if r.get(k) is not None]
        if vals:
            out[k] = float(sum(v * w for v, w in vals) / sum(w for _, w in vals))
    deltas = [r for r in per if r.get("delta")]
    if deltas:
        keys = set().union(*[r["delta"].keys() for r in deltas])
        nd = sum(r["n_frames"] for r in deltas)
        out["delta"] = {k: float(sum(r["delta"].get(k, 0.0) * r["n_frames"]
                                     for r in deltas if k in r["delta"]) / nd) for k in keys}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="LABEL=CKPT:CONFIG")
    ap.add_argument("--route", action="append", default=[], metavar="LABEL=hard|soft",
                    help="per-arm inference-time routing override for the KEG frontend; lets the "
                         "same checkpoint be probed under both routings (same-checkpoint ablation)")
    ap.add_argument("--split", default="val_core")
    ap.add_argument("--step-ms", type=int, default=50)
    ap.add_argument("--prev-noise", default="",
                    help="comma-separated teacher-forcing corruption scales; adds a sensitivity "
                         "sweep of single-step error against the error in the conditioning state")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    routes = dict(s.split("=", 1) for s in a.route)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = {}
    for spec in a.arm:
        label, rest = spec.split("=", 1)
        ckpt, cfg_path = rest.rsplit(":", 1)
        cfg = load_config(cfg_path)
        model = MNISTModel.load_from_checkpoint(
            ckpt, cfg=cfg, map_location=device).to(device).eval()
        mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
        root = Path(cfg["DATA"]["ROOT"])
        seqs = sequences_for_split(root, a.split, None)
        os.environ["EVENTHANDS_KEG_ROUTE"] = routes.get(label, "soft")

        out = {"route": routes.get(label, "soft"), "ckpt": ckpt}
        for forced in (True, False):
            regime = "teacher_forced" if forced else "recursive"
            per = [_run(model, mano, cfg, root, d, s, a.step_ms, device, forced)
                   for s, d in seqs]
            per = [r for r in per if r]
            out[regime] = _pool(per)
        out["amplification"] = out["recursive"]["ra_mm"] / out["teacher_forced"]["ra_mm"]
        print(f"{label}: teacher_forced RA={out['teacher_forced']['ra_mm']:.3f}  "
              f"recursive RA={out['recursive']['ra_mm']:.3f}  "
              f"amplification x{out['amplification']:.2f}", flush=True)

        sens = {}
        for sc in [float(s) for s in a.prev_noise.split(",") if s]:
            per = [_run(model, mano, cfg, root, d, s, a.step_ms, device, True, sc)
                   for s, d in seqs]
            per = [r for r in per if r]
            sens[f"x{sc:g}"] = _pool(per, extra_keys=("cond_err_ra_mm", "cond_err_abs_mm"))
            print(f"    prev-noise x{sc:<5g} cond_err_ra="
                  f"{sens[f'x{sc:g}'].get('cond_err_ra_mm', float('nan')):7.3f}  "
                  f"single-step RA={sens[f'x{sc:g}']['ra_mm']:8.3f}  "
                  f"update_ratio={sens[f'x{sc:g}'].get('delta', {}).get('update_ratio_pose', float('nan')):.3f}",
                  flush=True)
        if sens:
            out["prev_noise_sensitivity"] = sens
        rows[label] = out

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"step_ms": a.step_ms, "split": a.split, "arms": rows},
                                      indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
