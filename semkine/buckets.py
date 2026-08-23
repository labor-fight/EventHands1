#!/usr/bin/env python3
"""S5 motion-activity buckets, folded into S1 because the evaluator needs them from the start.

The metric this project reports is a single average over a validation set in which one anomalous
sequence supplies half the frames. That average cannot answer the question the method is actually
about -- "does routing updates to the observable degrees of freedom help when only one finger
moves" -- so the evaluation is stratified by what the ground truth is doing.

Two design decisions worth stating, both consequences of the `jitter_static` failure recorded in
`ARCHITECTURE_AUDIT.md` §5 (a gate whose support turned out to be one frame):

1. **Thresholds are quantiles of the training split, not hand-picked constants.** A constant like
   "static means below 0.5 mm per step" produced a bucket with one member. Quantiles produce
   buckets whose support is known before any gate is written on them.
2. **Support is reported next to every bucket** and any bucket below `MIN_SUPPORT` is marked
   `underpowered`, which bars it from carrying a claim.

Buckets are not mutually exclusive by construction: `mixed` overlaps `chain`, and `low_event_rate`
is an acquisition property that cuts across all of them. Each is a filter over steps, and each is
reported with its own support.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

#: MANO's 15 local joints in chain order per finger. Index 0 of each triple is the MCP.
FINGER_JOINTS: Dict[str, Tuple[int, int, int]] = {
    "index": (0, 1, 2),
    "middle": (3, 4, 5),
    "pinky": (6, 7, 8),
    "ring": (9, 10, 11),
    "thumb": (12, 13, 14),
}
#: Fingertip indices in the 21-joint OpenPose ordering `ManoLayer` returns.
OPENPOSE_TIPS: Dict[str, int] = {
    "thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20,
}
#: A bucket with fewer steps than this may be reported but may not carry a claim.
MIN_SUPPORT = 200

BUCKET_NAMES = (
    "quiet", "quiet_to_motion", "root_only", "single_finger_any", "multi_finger", "mixed",
    "chain", "occlusion", "low_contrast", "low_event_rate",
) + tuple(f"single_finger_{f}" for f in FINGER_JOINTS)

#: Buckets whose definition is a physical criterion rather than a training-split quantile.
#: `occlusion` is here because the quantile version degenerated: hidden fingertips occur in
#: about 13% of `local` steps and essentially never in `global` ones, so the 85th percentile of
#: the hidden fraction is exactly 0 and `>= 0` selected every step. "At least one of the five
#: fingertips is behind the rest of the hand" is both well defined and adequately supported.
ABSOLUTE_BUCKETS = ("occlusion",)


def _fk(mano, betas: np.ndarray, pos51: np.ndarray, device="cpu", chunk=4096):
    """GT 51D rows -> (joints21, verts) in metres."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
    from pose_repr import decode_to_mano_inputs

    js, vs = [], []
    b = torch.from_numpy(np.asarray(betas, np.float32)).view(1, -1).to(device)
    for i in range(0, len(pos51), chunk):
        t = torch.from_numpy(np.asarray(pos51[i : i + chunk], np.float32)).to(device)
        dec = decode_to_mano_inputs(t, "mano_full_axis_angle", mano.hands_components,
                                    mano.hands_mean)
        v, j = mano(b.expand(len(t), -1), dec["global_orient"], dec["local_full_aa"],
                    dec["transl"])
        js.append(j.detach().cpu().numpy())
        vs.append(v.detach().cpu().numpy())
    return np.concatenate(js), np.concatenate(vs)


def _occluded_tip_fraction(verts: np.ndarray, joints: np.ndarray, camera_K: np.ndarray,
                           render_scale: float, height: int, width: int,
                           depth_tol: float = 8e-3) -> np.ndarray:
    """Fraction of the 5 fingertips hidden behind the rest of the hand, per frame.

    Uses the same point-splat z-buffer the model's `_render_chunk` uses, so "occluded" means the
    same thing here and in the renderer: a tip is hidden when the nearest surface sample at its
    pixel is more than `depth_tol` in front of it.
    """
    fx, fy = camera_K[0, 0] * render_scale, camera_K[1, 1] * render_scale
    cx, cy = camera_K[0, 2] * render_scale, camera_K[1, 2] * render_scale
    n = len(verts)
    out = np.zeros(n, np.float32)
    tips = np.array([OPENPOSE_TIPS[f] for f in FINGER_JOINTS], dtype=np.int64)
    for i in range(n):
        v = verts[i]
        z = np.maximum(v[:, 2], 1e-6)
        u = np.rint(fx * v[:, 0] / z + cx).astype(np.int64)
        w = np.rint(fy * v[:, 1] / z + cy).astype(np.int64)
        ok = (u >= 0) & (u < width) & (w >= 0) & (w < height)
        depth = np.full(height * width, np.inf, np.float32)
        np.minimum.at(depth, w[ok] * width + u[ok], z[ok])
        t = joints[i][tips]
        tz = np.maximum(t[:, 2], 1e-6)
        tu = np.rint(fx * t[:, 0] / tz + cx).astype(np.int64)
        tw = np.rint(fy * t[:, 1] / tz + cy).astype(np.int64)
        vis = (tu >= 0) & (tu < width) & (tw >= 0) & (tw < height)
        hidden = np.zeros(len(tips), bool)
        if vis.any():
            d = depth[tw[vis] * width + tu[vis]]
            hidden[vis] = d < (tz[vis] - depth_tol)
        out[i] = hidden.mean()
    return out


def motion_features(root: Path, seq: str, legacy_dir: str, mano_npz: str,
                    step_ms: int = 50, device: str = "cpu",
                    render_scale: float = 0.375) -> Dict[str, np.ndarray]:
    """Per-evaluation-step ground-truth motion descriptors for one sequence.

    Steps are exactly the ones `eval_track` visits, so a bucket index can be applied to a metric
    array without re-deriving the alignment.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
    from mano_layer import ManoLayer

    from semkine.dataset import _read_meta51

    base = str(root / legacy_dir / seq)
    aux = np.load(base + "_aux.npz", allow_pickle=True)
    offsets = np.load(base + "_offsets.npy")
    pos51 = _read_meta51(base + ".meta")
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    betas = np.asarray(aux["betas"], np.float32)
    K = np.asarray(aux["camera_K"], np.float32).reshape(3, 3)

    ends, run_ids, prev_ends = [], [], []
    for rid, (a, b) in enumerate(runs):
        e = np.arange(a + step_ms - 1, b, step_ms, dtype=np.int64)
        if not len(e):
            continue
        ends.append(e)
        run_ids.append(np.full(len(e), rid, np.int64))
        prev_ends.append(np.r_[a, e[:-1]])
    if not ends:
        return {"n": 0}
    ends = np.concatenate(ends)
    run_ids = np.concatenate(run_ids)
    prev_ends = np.concatenate(prev_ends)

    mano = ManoLayer(mano_npz, add_mean=False).to(device).eval()
    with torch.no_grad():
        j_now, v_now = _fk(mano, betas, pos51[ends], device)
        j_prev, _ = _fk(mano, betas, pos51[prev_ends], device)

    ra_now = j_now - j_now[:, :1]
    ra_prev = j_prev - j_prev[:, :1]
    root_trans_mm = np.linalg.norm(j_now[:, 0] - j_prev[:, 0], axis=-1) * 1000.0
    ra_move_mm = np.linalg.norm(ra_now - ra_prev, axis=-1).mean(-1) * 1000.0

    # Per-finger articulation, measured at the fingertip in the root-aligned frame so it is a
    # millimetre quantity comparable across joints rather than a radian sum.
    tip_move = np.stack(
        [np.linalg.norm(ra_now[:, OPENPOSE_TIPS[f]] - ra_prev[:, OPENPOSE_TIPS[f]], axis=-1)
         * 1000.0 for f in FINGER_JOINTS], axis=1)

    # Per-joint axis-angle change, for the `chain` bucket (which asks how many joints inside one
    # finger moved, a question fingertip displacement cannot answer).
    aa_now = pos51[ends][:, 6:51].reshape(-1, 15, 3)
    aa_prev = pos51[prev_ends][:, 6:51].reshape(-1, 15, 3)
    joint_daa_deg = np.rad2deg(np.linalg.norm(aa_now - aa_prev, axis=-1))

    counts = np.diff(offsets)
    ev = np.array([counts[max(e - step_ms + 1, 0) : e + 1].sum() for e in ends], np.float64)

    with torch.no_grad():
        occ = _occluded_tip_fraction(v_now, j_now, K, render_scale, 180, 240)

    return {
        "n": len(ends),
        "ends": ends,
        "run_ids": run_ids,
        "elapsed_ms": ends - runs[run_ids][:, 0],
        "root_trans_mm": root_trans_mm.astype(np.float32),
        "ra_move_mm": ra_move_mm.astype(np.float32),
        "tip_move_mm": tip_move.astype(np.float32),
        "joint_daa_deg": joint_daa_deg.astype(np.float32),
        "events": ev.astype(np.float64),
        "occluded_tip_frac": occ,
        "fingers": list(FINGER_JOINTS),
    }


#: Quantiles defining the thresholds, registered before any model saw them.
QUANTILES = {
    "quiet_ra": 0.15,        # bottom 15% of articulation -> "quiet" articulation
    "quiet_root": 0.15,      # bottom 15% of root motion
    "active_tip": 0.70,      # a finger is "active" above the 70th pct of tip motion
    "chain_joint": 0.70,     # a joint is "moving" above the 70th pct of |d axis-angle|
    "low_event": 0.15,       # bottom 15% of events per step
    "low_contrast": 0.15,    # bottom 15% of events per mm of ground-truth motion
}


def calibrate_thresholds(feats: Sequence[Dict[str, np.ndarray]]) -> Dict[str, float]:
    """Fit thresholds on the training split only, so val/test buckets are not tuned on."""
    use = [f for f in feats if f.get("n", 0)]
    if not use:
        raise ValueError("no features to calibrate on")
    cat = lambda k: np.concatenate([f[k] for f in use])  # noqa: E731
    ra, rt = cat("ra_move_mm"), cat("root_trans_mm")
    tips = np.concatenate([f["tip_move_mm"] for f in use]).reshape(-1)
    daa = np.concatenate([f["joint_daa_deg"] for f in use]).reshape(-1)
    ev = cat("events")
    occ = cat("occluded_tip_frac")
    total = ra + rt
    contrast = ev / np.maximum(total, 1e-3)
    return {
        "quiet_ra_mm": float(np.quantile(ra, QUANTILES["quiet_ra"])),
        "quiet_root_mm": float(np.quantile(rt, QUANTILES["quiet_root"])),
        "active_tip_mm": float(np.quantile(tips, QUANTILES["active_tip"])),
        "chain_joint_deg": float(np.quantile(daa, QUANTILES["chain_joint"])),
        "low_event_count": float(np.quantile(ev, QUANTILES["low_event"])),
        "low_contrast_ev_per_mm": float(np.quantile(contrast, QUANTILES["low_contrast"])),
        # Diagnostic only: `occlusion` uses the absolute "any tip hidden" rule, so this records
        # what fraction of training steps that rule selects.
        "observed_occlusion_rate": float(np.mean(occ > 0.0)),
        "_quantiles": QUANTILES,
        "_absolute_buckets": list(ABSOLUTE_BUCKETS),
    }


def assign_buckets(f: Dict[str, np.ndarray], th: Dict[str, float]) -> Dict[str, np.ndarray]:
    """Boolean mask per bucket over this sequence's evaluation steps."""
    if not f.get("n", 0):
        return {b: np.zeros(0, bool) for b in BUCKET_NAMES}
    quiet_art = f["ra_move_mm"] <= th["quiet_ra_mm"]
    quiet_root = f["root_trans_mm"] <= th["quiet_root_mm"]
    active = f["tip_move_mm"] > th["active_tip_mm"]        # (N, 5)
    n_active = active.sum(1)
    root_active = ~quiet_root

    out: Dict[str, np.ndarray] = {}
    out["quiet"] = quiet_art & quiet_root
    # `quiet_to_motion`: the step after a quiet step, inside the same run. This is where a
    # tracker that has stopped attending to the input pays for it.
    prev_quiet = np.zeros(f["n"], bool)
    same_run = np.r_[False, f["run_ids"][1:] == f["run_ids"][:-1]]
    prev_quiet[1:] = out["quiet"][:-1]
    out["quiet_to_motion"] = same_run & prev_quiet & ~out["quiet"]
    out["root_only"] = root_active & (n_active == 0)
    out["multi_finger"] = (~root_active) & (n_active >= 2)
    out["mixed"] = root_active & (n_active >= 1)
    single = (~root_active) & (n_active == 1)
    for i, name in enumerate(FINGER_JOINTS):
        out[f"single_finger_{name}"] = single & active[:, i]
    # Per-finger buckets run 100-350 steps per split, under `MIN_SUPPORT`. The union is the
    # bucket a claim may be written on; the five individual ones stay as diagnostics.
    out["single_finger_any"] = single
    # `chain`: at least two joints inside one active finger moved, i.e. the case where a
    # per-finger gate is too coarse and per-joint routing is the claim.
    moving = f["joint_daa_deg"] > th["chain_joint_deg"]     # (N, 15)
    chain = np.zeros(f["n"], bool)
    for i, name in enumerate(FINGER_JOINTS):
        js = list(FINGER_JOINTS[name])
        chain |= active[:, i] & (moving[:, js].sum(1) >= 2)
    out["chain"] = chain
    out["occlusion"] = f["occluded_tip_frac"] > 0.0
    out["low_event_rate"] = f["events"] <= th["low_event_count"]
    contrast = f["events"] / np.maximum(f["ra_move_mm"] + f["root_trans_mm"], 1e-3)
    out["low_contrast"] = contrast <= th["low_contrast_ev_per_mm"]
    return out


def build_bucket_manifest(root: Path, sequences: Sequence[Tuple[str, str]], mano_npz: str,
                          step_ms: int = 50, thresholds: Optional[Dict[str, float]] = None,
                          device: str = "cpu") -> dict:
    """Bucket masks for every step of every sequence, plus support counts.

    `thresholds=None` calibrates on the sequences given, which is correct only when those are
    the training sequences; val/test must be passed the training thresholds.
    """
    feats = {}
    for seq, legacy_dir in sequences:
        feats[seq] = motion_features(root, seq, legacy_dir, mano_npz, step_ms, device)
    th = thresholds or calibrate_thresholds(list(feats.values()))

    per_seq, totals = {}, {b: 0 for b in BUCKET_NAMES}
    for seq, f in feats.items():
        masks = assign_buckets(f, th)
        per_seq[seq] = {
            "n_steps": int(f.get("n", 0)),
            "buckets": {b: np.flatnonzero(m).astype(int).tolist() for b, m in masks.items()},
            "support": {b: int(m.sum()) for b, m in masks.items()},
            "gt_move_mm_mean": float(np.mean(f["ra_move_mm"])) if f.get("n") else 0.0,
            "events_per_step_mean": float(np.mean(f["events"])) if f.get("n") else 0.0,
        }
        for b, m in masks.items():
            totals[b] += int(m.sum())

    return {
        "step_ms": step_ms,
        "thresholds": th,
        "per_sequence": per_seq,
        "totals": totals,
        "min_support": MIN_SUPPORT,
        "underpowered": sorted(b for b, n in totals.items() if n < MIN_SUPPORT),
    }


def load_bucket_manifest(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def masks_for_sequence(manifest: dict, seq: str, n_steps: int) -> Dict[str, np.ndarray]:
    """Rehydrate boolean masks for one sequence, checked against the step count."""
    entry = manifest["per_sequence"].get(seq)
    if entry is None:
        return {}
    if int(entry["n_steps"]) != int(n_steps):
        raise ValueError(
            f"{seq}: bucket manifest has {entry['n_steps']} steps but the evaluator produced "
            f"{n_steps}; regenerate with tools/make_semkine_manifest.py --step-ms"
        )
    out = {}
    for b, idx in entry["buckets"].items():
        m = np.zeros(n_steps, bool)
        if idx:
            m[np.asarray(idx, dtype=np.int64)] = True
        out[b] = m
    return out
