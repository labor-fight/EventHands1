#!/usr/bin/env python3
"""EvRealHands (ERH, DAVIS346 right-hand trials) -> this repo's event-recording format, plus data-consistency checks.

CPU only, GT + events only: no network is run on this data (model readings need a registered evaluation plan).
Raw data is read-only (`RAW`); every product goes under `outputs/tpd/20261105/erh/`.

Conventions (read from /data1/lyq/code/mesh/EvHandPose/core/dataset/dataset.py `EvHandDataset` and verified here, see
`convert_receipt.json`):
  * events: x, y, polarity, timestamp (us, int64) from `event.aedat4`; stable sort by timestamp, t_rel = t - first event;
    undistort `cv2.undistortPoints(xy, K_old, dist, P=K_new)` into the K_new frame, keep points inside 346 x 260
    (strict; the reference's own bound check tests y against the *width* and then clips, reported as a statistic);
    x_out = floor(x_u * 240/346), y_out = floor(y_u * 180/260) on the 240 x 180 grid; ms = t_rel // 1000.
  * labels: frame id i is at `delta_time + i * 1e6/15` us on the t_rel clock.
  * MANO (`annot['mano'][id]`): rot, hand_pose (residual from hands_mean), shape, trans, root_joint, in the world frame;
    event camera frame R_new = R Rot(rot), trans_new = R (trans + root_joint) + T/1000 - root_joint;
    joints_cam = R joints/1000 + T/1000 (annotation joints are in mm).
  * the 51-D state is [t (3), R_root axis-angle (3), hand_pose residual (45)] exactly as in the training data.
  * aux camera_K is stored so that the model's `K * 0.375` is the 240 x 180 pixel intrinsics (see `stored_camera_K`).

opencv is not installed in the training environment; `undistort_points` calls cv2 of another environment through a
subprocess (`CV2_PY`) only for the 346 x 260 pixel look-up table and the 2-D joints (`cv2.undistortPoints` is a per-point
function, so a per-pixel table is identical to calling it on every event).

    python -m tools.tpd.erh_convert all [--workers 6]        # plan, permutation, convert, check, finalize, report
    python -m tools.tpd.erh_convert convert|check|finalize|report [--trials 0 1 ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RAW = Path(os.environ.get("ERH_RAW", "/data1/lyq/code/mesh/data1/EvRealHands"))
OUT = ROOT / "outputs/tpd/20261105/erh"
DATA = OUT / "data"
TRAIN_DATA = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
CV2_PY = os.environ.get("ERH_CV2_PY", "/data1/lyq/miniconda3/envs/EventHands/bin/python")

SRC_W, SRC_H = 346, 260
GRID_W, GRID_H = 240, 180
SX, SY = GRID_W / SRC_W, GRID_H / SRC_H
FPS = 15.0
RENDER_SCALE = 0.375
TIERS = ("3d", "3d_unchecked", "2d")
K_NAMES = ("K", "K_new", "K_old")           # K_old is a diagnostic only, never chosen
K_CANDIDATES = ("K", "K_new")
K_WINDOW_MS = 10                              # events within +- 10 ms of the label time
NULL_SHIFT = (8, 0)
SHIFT_RANGE = 6
DILATE = 2
CLOCK_LAGS = (-150, 150)
CLOCK_SMOOTH_MS = 20
K_EVERY = 10


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(jsonable(obj), f, indent=1)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


# ------------------------------------------------------------------------------------------------ raw annotation


def read_annot(trial):
    with open(RAW / str(trial) / "annot.json") as f:
        return json.load(f)


def tier_of(annot):
    """None for left hands (excluded); '3d' right/normal/annoted, '3d_unchecked' right/normal/not annoted, '2d' right/fast."""
    if annot["hand_type"] != "right":
        return None
    if annot["motion_type"] == "fast":
        return "2d"
    return "3d" if annot["annoted"] else "3d_unchecked"


def plan(force=False):
    """List all 84 trials with hand / tier; cached in `trials.json` (left-hand trials recorded as excluded)."""
    path = OUT / "trials.json"
    if path.exists() and not force:
        return read_json(path)
    rows = {}
    for t in range(84):
        a = read_annot(t)
        rows[t] = {"trial": t, "hand_type": a["hand_type"], "motion_type": a["motion_type"], "annoted": bool(a["annoted"]),
                   "subject_id": a["subject_id"], "scene": a["scene"], "gesture": a["gesture_type"], "tier": tier_of(a),
                   "n_mano": len(a["mano"]), "n_manos": len(a.get("manos", {})), "n_3d_joints": len(a["3d_joints"]),
                   "event_bytes": (RAW / str(t) / "event.aedat4").stat().st_size}
    doc = {"raw": str(RAW), "trials": rows,
           "included": sorted(t for t, r in rows.items() if r["tier"]),
           "excluded_left": sorted(t for t, r in rows.items() if r["tier"] is None),
           "tier_counts": {k: sum(1 for r in rows.values() if r["tier"] == k) for k in TIERS}}
    write_json(path, doc)
    return read_json(path)


def camera_of(annot):
    ev = annot["camera_info"]["event"]
    cam = {k: np.array(ev[k], np.float64) for k in ("R", "K", "K_new", "K_old", "dist")}
    cam["T"] = np.array(ev["T"], np.float64) / 1000.0           # metres
    assert list(ev["image_shape"]) == [SRC_H, SRC_W], ev["image_shape"]
    return cam


# ------------------------------------------------------------------------------------------------ undistortion

_CV2_SNIPPET = """
import sys, numpy as np, cv2
z = np.load(sys.argv[1])
out = cv2.undistortPoints(src=z['xy'].astype(np.float32), cameraMatrix=z['K_old'], distCoeffs=z['dist'], P=z['K_new'])
np.save(sys.argv[2], out.reshape(-1, 2).astype(np.float32))
"""


def undistort_points(xy, K_old, dist, K_new):
    """`cv2.undistortPoints(src=xy, cameraMatrix=K_old, distCoeffs=dist, P=K_new)` as `(N, 2)` float32."""
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    try:
        import cv2
    except ImportError:
        cv2 = None
    if cv2 is not None:
        return cv2.undistortPoints(src=xy, cameraMatrix=np.asarray(K_old, np.float64), distCoeffs=np.asarray(dist, np.float64),
                                   P=np.asarray(K_new, np.float64)).reshape(-1, 2).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(OUT), prefix=".cv2_") as tmp:
        fin, fout = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npy")
        np.savez(fin, xy=xy, K_old=np.asarray(K_old, np.float64), dist=np.asarray(dist, np.float64),
                 K_new=np.asarray(K_new, np.float64))
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
        subprocess.run([CV2_PY, "-I", "-c", _CV2_SNIPPET, fin, fout], check=True, env=env)
        return np.load(fout)


def lut_path(K_old, dist, K_new):
    key = hashlib.sha1(np.concatenate([np.ravel(K_old), np.ravel(dist), np.ravel(K_new)]).astype(np.float64).tobytes()).hexdigest()[:16]
    return OUT / f"undistort_lut_{key}.npy"


def build_lut(K_old, dist, K_new):
    """Undistorted position of every sensor pixel `(260, 346, 2)` float32; cached by camera parameters."""
    path = lut_path(K_old, dist, K_new)
    if path.exists():
        return np.load(path)
    ys, xs = np.mgrid[0:SRC_H, 0:SRC_W]
    und = undistort_points(np.stack([xs.ravel(), ys.ravel()], 1), K_old, dist, K_new).reshape(SRC_H, SRC_W, 2)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(path, und)
    return und


def bin_tables(und):
    """Per-pixel output bin and legality from the undistortion table.

    strict:  0 <= x_u <= 345 and 0 <= y_u <= 259 (used).
    ref:     the reference's check (`y_u <= width - 1`; its out-of-height points are then clipped, not dropped).
    """
    xu, yu = und[..., 0].astype(np.float64), und[..., 1].astype(np.float64)
    legal = (xu >= 0) & (xu <= SRC_W - 1) & (yu >= 0) & (yu <= SRC_H - 1)
    ref_legal = (xu >= 0) & (xu <= SRC_W - 1) & (yu >= 0) & (yu <= SRC_W - 1)
    bx = np.clip(np.floor(xu * SX), 0, GRID_W - 1).astype(np.uint8)
    by = np.clip(np.floor(yu * SY), 0, GRID_H - 1).astype(np.uint8)
    return {"bx": bx, "by": by, "legal": legal, "ref_legal": ref_legal}


def undistorted_to_grid(xy_u):
    """Continuous 240 x 180 grid coordinates of undistorted 346 x 260 points (no floor, no clip)."""
    xy_u = np.asarray(xy_u, np.float64)
    return np.stack([xy_u[..., 0] * SX, xy_u[..., 1] * SY], -1)


def stored_camera_K(K):
    """aux `camera_K`: rows 0/1 of K scaled by (240/346, 180/260), divided by RENDER_SCALE (the model multiplies by it)."""
    K = np.asarray(K, np.float64)
    out = np.eye(3)
    out[0, 0], out[0, 2] = K[0, 0] * SX / RENDER_SCALE, K[0, 2] * SX / RENDER_SCALE
    out[1, 1], out[1, 2] = K[1, 1] * SY / RENDER_SCALE, K[1, 2] * SY / RENDER_SCALE
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------------ events


def read_events(path):
    """Raw DAVIS events (timestamp us int64, x int16, y int16, polarity int8) from an .aedat4 file."""
    from dv import AedatFile
    parts = ([], [], [], [])
    with AedatFile(str(path)) as f:
        for p in f["events"].numpy():
            for lst, name, dt in zip(parts, ("timestamp", "x", "y", "polarity"), (np.int64, np.int16, np.int16, np.int8)):
                lst.append(np.array(p[name], dt))
    return tuple(np.concatenate(lst) for lst in parts)


def bin_events(t, x, y, p, tables):
    """Sorted, undistorted, binned events in the target format.

    Returns dict: events (N, 3) uint8, offsets (n_ms + 1,) int64, tsub (N,) uint16 and statistics.
    """
    n_raw = len(t)
    n_back = int((np.diff(t) < 0).sum())
    max_back = int(-np.diff(t).min()) if n_raw > 1 and n_back else 0
    if n_back:
        order = np.argsort(t, kind="stable")
        t, x, y, p = t[order], x[order], y[order], p[order]
    t0 = int(t[0])
    t_rel = t - t0
    inside = (x >= 0) & (x < SRC_W) & (y >= 0) & (y < SRC_H)
    pix = np.where(inside, y.astype(np.int64) * SRC_W + x.astype(np.int64), 0)
    ok = inside & tables["legal"].ravel()[pix]
    ref_ok = inside & tables["ref_legal"].ravel()[pix]
    n_ms = int(t_rel[-1] // 1000) + 1
    k = np.flatnonzero(ok)
    ev = np.empty((len(k), 3), np.uint8)
    ev[:, 0] = tables["bx"].ravel()[pix[k]]
    ev[:, 1] = tables["by"].ravel()[pix[k]]
    ev[:, 2] = np.clip(p[k], 0, 1)
    tr = t_rel[k]
    tsub = (tr % 1000).astype(np.uint16)
    counts = np.bincount(tr // 1000, minlength=n_ms)
    offsets = np.zeros(n_ms + 1, np.int64)
    offsets[1:] = np.cumsum(counts)
    stats = {"n_raw": n_raw, "n_kept": int(len(k)), "n_unsorted_diffs": n_back, "max_backward_us": max_back,
             "first_event_time_us": t0, "n_ms": n_ms, "duration_s": float(t_rel[-1]) / 1e6,
             "drop_frac_strict": 1.0 - len(k) / n_raw, "drop_frac_reference_rule": 1.0 - float(ref_ok.mean()),
             "clipped_by_reference_frac": float((ref_ok & ~ok).mean()),
             "on_fraction": float(ev[:, 2].mean()) if len(k) else float("nan")}
    return {"events": ev, "offsets": offsets, "tsub": tsub, "stats": stats}


def event_rate_stats(events, offsets, n_windows=200):
    """Rates on the output grid: events per ms, per output pixel per 50 ms, active pixels per 50 ms window."""
    n_ms = len(offsets) - 1
    counts = np.diff(offsets)
    out = {"events_per_ms_mean": float(len(events)) / n_ms, "events_per_ms_median": float(np.median(counts)),
           "events_per_pixel_per_50ms_mean": float(len(events)) * 50.0 / (n_ms * GRID_W * GRID_H)}
    if n_ms > 60:
        starts = np.linspace(0, n_ms - 50, n_windows).astype(np.int64)
        act = []
        for s in starts:
            a0, a1 = int(offsets[s]), int(offsets[s + 50])
            if a1 > a0:
                e = np.asarray(events[a0:a1])
                act.append(len(np.unique(e[:, 1].astype(np.int64) * GRID_W + e[:, 0])))
            else:
                act.append(0)
        act = np.asarray(act, np.float64)
        out["active_pixels_per_50ms_mean"] = float(act.mean())
        out["active_pixel_fraction_per_50ms"] = float(act.mean()) / (GRID_W * GRID_H)
    return out


def source_pixels_per_bin(tables):
    """How many sensor pixels land in each output bin (legal pixels only): the sensor is 2.08 source pixels per bin."""
    cnt = np.zeros((GRID_H, GRID_W), np.int64)
    ok = tables["legal"]
    np.add.at(cnt, (tables["by"][ok].astype(np.int64), tables["bx"][ok].astype(np.int64)), 1)
    cen = cnt[30:150, 40:200]
    ys, xs = np.nonzero(cnt == 0)
    return {"mean": float(cnt.mean()), "median": float(np.median(cnt)), "min": int(cnt.min()), "max": int(cnt.max()),
            "frac_bins_empty": float((cnt == 0).mean()), "frac_bins_ge_3": float((cnt >= 3).mean()),
            "hist": {int(k): int(v) for k, v in zip(*np.unique(cnt, return_counts=True))},
            "central_30_150_x_40_200": {"mean": float(cen.mean()), "frac_empty": float((cen == 0).mean()), "frac_ge_3": float((cen >= 3).mean()),
                                        "hist": {int(k): int(v) for k, v in zip(*np.unique(cen, return_counts=True))}},
            "empty_bins_x_range": [int(xs.min()), int(xs.max())] if len(xs) else None, "empty_bins_y_range": [int(ys.min()), int(ys.max())] if len(ys) else None}


def binning_offset(und, tables):
    """Mean (output bin index - continuous grid coordinate) over the legal pixels (floor binning => about -0.5 px).

    The model compares projected continuous coordinates with integer bin indices; the training data has -0.4375 px in x
    and y (floor(3 x / 8)); this is the same number for the ERH grid.
    """
    ok = tables["legal"]
    cx, cy = und[..., 0].astype(np.float64)[ok] * SX, und[..., 1].astype(np.float64)[ok] * SY
    return {"dx_mean": float((tables["bx"][ok] - cx).mean()), "dy_mean": float((tables["by"][ok] - cy).mean())}


# ------------------------------------------------------------------------------------------------ MANO


_MANO = {}


def get_mano():
    if "m" not in _MANO:
        import torch
        torch.set_num_threads(int(os.environ.get("ERH_TORCH_THREADS", "2")))
        from mano_layer import load_mano
        _MANO["m"] = load_mano().eval()
    return _MANO["m"]


def mano_world_to_camera(rot, trans, root_joint, R, T, target_root=None):
    """MANO (rot, trans) of the dataset's world frame -> event camera frame (EvHandPose `get_annotations`).

    `target_root` is the root joint J0 of the shape the converted parameters will be used with (default: the frame's own
    `root_joint`, i.e. the reference formula). It keeps the wrist where the annotation has it when a trial-constant shape
    replaces per-frame shapes: the posed root joint is J0 + transl before and after the rigid transform.
    """
    from scipy.spatial.transform import Rotation as Rot
    tgt = root_joint if target_root is None else target_root
    R_new = R[None] @ Rot.from_rotvec(np.asarray(rot, np.float64)).as_matrix()
    rot_new = Rot.from_matrix(R_new).as_rotvec()
    trans_new = (np.asarray(trans, np.float64) + root_joint) @ R.T + T - tgt
    return rot_new, trans_new


def fk_raw(betas, global_orient, local_full, transl, chunk=256):
    """Joints `(n, 21, 3)` and vertices `(n, 778, 3)` float32 of our ManoLayer; local_full = absolute 45-D axis-angle."""
    import torch
    mano = get_mano()
    betas = np.asarray(betas, np.float32).reshape(1, -1) if np.ndim(betas) == 1 else np.asarray(betas, np.float32)
    n = len(global_orient)
    J, V = [], []
    with torch.no_grad():
        for i in range(0, n, chunk):
            sl = slice(i, i + chunk)
            b = torch.from_numpy(np.ascontiguousarray(np.broadcast_to(betas, (n, betas.shape[1]))[sl]))
            v, j = mano(b, torch.from_numpy(np.asarray(global_orient[sl], np.float32)),
                        torch.from_numpy(np.asarray(local_full[sl], np.float32)), torch.from_numpy(np.asarray(transl[sl], np.float32)))
            J.append(j.numpy())
            V.append(v.numpy())
    return np.concatenate(J), np.concatenate(V)


def fk51(params51, betas):
    """FK of 51-D states (t, R_root, hand_pose residual) with our decode (adds hands_mean), as the evaluator does."""
    import torch
    from pose_repr import decode_to_mano_inputs
    mano = get_mano()
    p = np.asarray(params51, np.float32)
    with torch.no_grad():
        dec = decode_to_mano_inputs(torch.from_numpy(p), "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
    return fk_raw(betas, dec["global_orient"].numpy(), dec["local_full_aa"].numpy(), dec["transl"].numpy())


def hands_mean():
    return get_mano().hands_mean.numpy().astype(np.float64)


def joint_error_mm(pred_j, ref_j):
    return np.linalg.norm(np.asarray(pred_j, np.float64) - np.asarray(ref_j, np.float64), axis=-1) * 1000.0


def err_summary(e):
    return {"median": float(np.median(e)), "p95": float(np.percentile(e, 95)), "mean": float(e.mean()),
            "per_joint_mean": e.mean(0).tolist(), "n_frames": int(e.shape[0])}


def raw_mano_arrays(annot, ids, key="mano"):
    m = [annot[key][str(i)] for i in ids]
    return {k: np.array([x[k] for x in m], np.float64) for k in ("rot", "hand_pose", "shape", "trans", "root_joint")}


def annotation_joints_cam(annot, ids, cam):
    j = np.array([annot["3d_joints"][str(i)] for i in ids], np.float64) / 1000.0
    return j @ cam["R"].T + cam["T"]


def mano_variants(annot, ids, cam, ann_cam_ours):
    """Errors (mm) of our FK of the converted MANO against the annotation joints (event frame, our order) for the
    candidate conventions: residual vs absolute hand_pose; about-root vs naive translation; `manos` instead of `mano`."""
    a = raw_mano_arrays(annot, ids)
    betas = np.median(a["shape"], 0)                                    # the trial shape that is used (constant in 37 / 40 trials)
    j0_t = root_joint_ours(betas[None])[0]
    rot_n, tr_n = mano_world_to_camera(a["rot"], a["trans"], a["root_joint"], cam["R"], cam["T"], target_root=j0_t)
    hm = hands_mean()
    out = {}

    def run(rot, tr, hp_full):
        j, _ = fk_raw(betas, rot, hp_full, tr)
        return joint_error_mm(j, ann_cam_ours)

    out["A_residual+about_root (used)"] = err_summary(run(rot_n, tr_n, a["hand_pose"] + hm))
    rot_f, tr_f = mano_world_to_camera(a["rot"], a["trans"], a["root_joint"], cam["R"], cam["T"])
    j_f, _ = fk_raw(a["shape"], rot_f, a["hand_pose"] + hm, tr_f)         # reference-faithful: each frame's own shape
    out["F_per_frame_shape (reference)"] = err_summary(joint_error_mm(j_f, ann_cam_ours))
    out["B_absolute_hand_pose"] = err_summary(run(rot_n, tr_n, a["hand_pose"]))
    tr_naive = a["trans"] @ cam["R"].T + cam["T"]
    out["C_naive_translation (no root_joint)"] = err_summary(run(rot_n, tr_naive, a["hand_pose"] + hm))
    out["D_absolute+naive"] = err_summary(run(rot_n, tr_naive, a["hand_pose"]))
    if annot.get("manos"):
        ids_s = [i for i in ids if str(i) in annot["manos"]]
        if ids_s:
            sel = np.array([ids.index(i) for i in ids_s])
            s = raw_mano_arrays(annot, ids_s, key="manos")
            rs, ts = mano_world_to_camera(s["rot"], s["trans"], s["root_joint"], cam["R"], cam["T"])
            j, _ = fk_raw(s["shape"], rs, s["hand_pose"] + hm, ts)
            e_manos = joint_error_mm(j, ann_cam_ours[sel])
            e_mano_same = run(rot_n, tr_n, a["hand_pose"] + hm)[sel]
            out["E_manos_per_frame_shape_same_ids"] = {**err_summary(e_manos), "mano_same_ids_median": float(np.median(e_mano_same)),
                                                       "n_ids_in_manos": len(ids_s), "manos_shape_std_max": float(s["shape"].std(0).max())}
            bm = np.median(s["shape"], 0)
            rs2, ts2 = mano_world_to_camera(s["rot"], s["trans"], s["root_joint"], cam["R"], cam["T"], target_root=root_joint_ours(bm[None])[0])
            j2, _ = fk_raw(bm, rs2, s["hand_pose"] + hm, ts2)
            out["E2_manos_median_shape_same_ids"] = err_summary(joint_error_mm(j2, ann_cam_ours[sel]))
    out["root_joint_vs_ours_max_abs_m"] = float(np.abs(root_joint_ours(a["shape"]) - a["root_joint"]).max())
    out["shape_std_max"] = float(a["shape"].std(0).max())
    out["R_orthonormal_dev"] = float(np.abs(cam["R"].T @ cam["R"] - np.eye(3)).max())
    return out


def root_joint_ours(shapes):
    """J0 of our shaped template per frame, `(n, 3)`."""
    import torch
    mano = get_mano()
    b = torch.from_numpy(np.asarray(shapes, np.float32))
    v = mano.v_template + torch.einsum("bl,mkl->bmk", b, mano.shapedirs)
    return torch.einsum("bik,ji->bjk", v, mano.J_regressor)[:, 0].numpy().astype(np.float64)


def determine_joint_perm(trials, every=25):
    """The permutation between the annotation joint order and our FK joint order, by assignment on pooled cost.

    perm[i] = annotation joint index matched to our joint i, so `ann_ours = ann[:, perm]`.
    Consistency: share of frames whose own per-frame assignment equals the pooled one.
    """
    from scipy.optimize import linear_sum_assignment
    hm = hands_mean()
    cost = np.zeros((21, 21))
    per_frame, n = [], 0
    for t in trials:
        a = read_annot(t)
        cam = camera_of(a)
        ids = sorted(set(int(k) for k in a["mano"]) & set(int(k) for k in a["3d_joints"]))[::every]
        arr = raw_mano_arrays(a, ids)
        rot, tr = mano_world_to_camera(arr["rot"], arr["trans"], arr["root_joint"], cam["R"], cam["T"])
        j, _ = fk_raw(arr["shape"], rot, arr["hand_pose"] + hm, tr)
        ann = annotation_joints_cam(a, ids, cam)
        d = np.linalg.norm(j[:, :, None, :] - ann[:, None, :, :], axis=-1)       # (n, ours i, ann j)
        cost += d.sum(0)
        n += len(ids)
        for f in range(len(ids)):
            per_frame.append(tuple(int(v) for v in linear_sum_assignment(d[f])[1]))
    r, c = linear_sum_assignment(cost / n)
    perm = tuple(int(v) for v in c)
    cons = float(np.mean([p == perm for p in per_frame]))
    distinct = {}
    for p in per_frame:
        distinct[p] = distinct.get(p, 0) + 1
    srt = np.sort(cost / n, axis=1)
    return {"perm": list(perm), "is_identity": perm == tuple(range(21)), "n_frames": n, "consistency": cons,
            "n_distinct_per_frame_perms": len(distinct), "pooled_cost_mm": float(np.mean(cost[r, c] / n) * 1000),
            "min_row_margin_mm": float((srt[:, 1] - srt[:, 0]).min() * 1000), "trials": list(trials)}


# ------------------------------------------------------------------------------------------------ labels and writers


def label_ids(annot, tier):
    if tier == "2d":
        d = annot["2d_joints"]["event"]
        return sorted(int(k) for k, v in d.items() if len(v) == 21)
    return sorted(set(int(k) for k in annot["mano"]) & set(int(k) for k in annot["3d_joints"]))


def label_runs(ids):
    """Runs of consecutive frame ids: list of (first_index, last_index) into `ids`."""
    ids = np.asarray(ids)
    if len(ids) == 0:
        return []
    cut = np.flatnonzero(np.diff(ids) != 1)
    starts = np.concatenate([[0], cut + 1])
    ends = np.concatenate([cut, [len(ids) - 1]])
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def trial_dir(trial, root=None):
    return Path(root or DATA) / f"erh_{trial}"


def trial_base(trial, root=None):
    return trial_dir(trial, root) / f"erh_{trial}"


def write_trial_files(trial, root, events, offsets, tsub, labels, aux, done):
    d = trial_dir(trial, root)
    d.mkdir(parents=True, exist_ok=True)
    b = trial_base(trial, root)
    np.save(f"{b}_events.npy", events)
    np.save(f"{b}_offsets.npy", offsets)
    np.save(f"{b}_tsub.npy", tsub)
    np.savez(f"{b}_labels.npz", **labels)
    np.savez(f"{b}_aux.npz", **aux)
    write_json(f"{b}.done.json", done)


def _str(x):
    return np.array(str(x))


def convert_trial(trial, perm, lut, root=None):
    """Convert one right-hand trial; returns the summary that goes into the receipt."""
    t_start = time.time()
    annot = read_annot(trial)
    tier = tier_of(annot)
    assert tier is not None, f"trial {trial} is a left-hand trial"
    cam = camera_of(annot)
    tables = bin_tables(lut)
    t, x, y, p = read_events(RAW / str(trial) / "event.aedat4")
    binned = bin_events(t, x, y, p, tables)
    del t, x, y, p
    events, offsets, tsub, est = binned["events"], binned["offsets"], binned["tsub"], binned["stats"]
    n_ms = est["n_ms"]
    est.update(event_rate_stats(events, offsets))

    ids = label_ids(annot, tier)
    t_us = annot["delta_time"] + np.asarray(ids, np.float64) * 1e6 / FPS
    label_ms = np.rint(t_us / 1000.0).astype(np.int64)
    inb = (label_ms >= 0) & (label_ms < n_ms)
    n_oob = int((~inb).sum())
    n_before, n_after = int((label_ms < 0).sum()), int((label_ms >= n_ms).sum())
    ids = [i for i, k in zip(ids, inb) if k]
    t_us, label_ms = t_us[inb], label_ms[inb]
    ids_arr = np.asarray(ids, np.int64)
    runs = label_runs(ids)

    labels = {"frame_id": ids_arr, "label_t_us": t_us, "label_ms": label_ms, "tier": _str(tier),
              "joint_perm": np.asarray(perm, np.int64)}
    mano_check = None
    depth = None
    betas_trial = np.zeros(10, np.float32)
    betas_std = None
    betas_source = "pending_subject_mean"
    if tier in ("3d", "3d_unchecked"):
        arr = raw_mano_arrays(annot, ids)
        betas_trial = np.median(arr["shape"], 0).astype(np.float32)
        betas_std = float(arr["shape"].std(0).max())
        betas_source = ("annot['mano'][id]['shape'] (constant over the trial)" if betas_std < 1e-6 else
                        "median over frames of annot['mano'][id]['shape'] (the shape varies per frame in this trial)")
        j0_t = root_joint_ours(betas_trial[None])[0]
        rot_n, tr_n = mano_world_to_camera(arr["rot"], arr["trans"], arr["root_joint"], cam["R"], cam["T"], target_root=j0_t)
        params51 = np.concatenate([tr_n, rot_n, arr["hand_pose"]], 1).astype(np.float32)
        ann = annotation_joints_cam(annot, ids, cam)[:, np.asarray(perm)]
        j, _ = fk51(params51, betas_trial)
        e = joint_error_mm(j, ann)
        mano_check = {"used": err_summary(e), **mano_variants(annot, ids, cam, ann)}
        labels.update(params51=params51, betas_frames=arr["shape"].astype(np.float32), betas_trial=betas_trial,
                      joints3d_cam=ann.astype(np.float32))
        wz = ann[:, 0, 2]
        depth = {"wrist_z_m_p5_p50_p95": np.percentile(wz, [5, 50, 95]).tolist(), "wrist_z_m_min_max": [float(wz.min()), float(wz.max())],
                 "transl_z_m_p5_p50_p95": np.percentile(params51[:, 2], [5, 50, 95]).tolist()}
    else:
        d2 = annot["2d_joints"]["event"]
        raw2 = np.array([d2[str(i)] for i in ids], np.float64).reshape(-1, 2)
        und = undistort_points(raw2, cam["K_old"], cam["dist"], cam["K_new"]).astype(np.float64)
        grid = undistorted_to_grid(und)
        labels.update(joints2d_raw=raw2.reshape(-1, 21, 2).astype(np.float32),
                      joints2d_undist=und.reshape(-1, 21, 2).astype(np.float32),
                      joints2d_grid=grid.reshape(-1, 21, 2).astype(np.float32),
                      joints2d_inbounds=((und[:, 0] >= 0) & (und[:, 0] <= SRC_W - 1) & (und[:, 1] >= 0)
                                         & (und[:, 1] <= SRC_H - 1)).reshape(-1, 21))
        if annot["3d_joints"]:       # present in the fast trials too (not MANO; unused by the reference evaluation)
            ids3 = [i for i in ids if str(i) in annot["3d_joints"]]
            labels.update(joints3d_cam_frame_id=np.asarray(ids3, np.int64),
                          joints3d_cam=(annotation_joints_cam(annot, ids3, cam)[:, np.asarray(perm)]).astype(np.float32))

    K_stored = stored_camera_K(cam["K"])         # provisional (annotation K); `finalize` applies the pre-written rule
    runs_ms = np.array([[label_ms[a], label_ms[b] + 1] for a, b in runs], np.int64).reshape(-1, 2)
    aux = {"betas": betas_trial, "camera_K": K_stored, "camera_K_choice": _str("K"),
           "K_src": cam["K"].astype(np.float32), "K_new_src": cam["K_new"].astype(np.float32),
           "K_old_src": cam["K_old"].astype(np.float32), "dist_src": cam["dist"].astype(np.float32),
           "R_event": cam["R"].astype(np.float32), "T_event_m": cam["T"].astype(np.float32),
           "src_size": np.array([SRC_W, SRC_H]), "grid_size": np.array([GRID_W, GRID_H]), "render_scale": np.float32(RENDER_SCALE),
           "hand_type": _str(annot["hand_type"]), "motion_type": _str(annot["motion_type"]), "scene": _str(annot["scene"]),
           "gesture": _str(annot["gesture_type"]), "subject_id": np.int64(annot["subject_id"]), "tier": _str(tier),
           "n_ms": np.int64(n_ms), "delta_time_us": np.float64(annot["delta_time"]),
           "betas_varying_per_frame": np.bool_(betas_std is not None and betas_std > 1e-6),
           "first_event_time_us": np.int64(est["first_event_time_us"]), "betas_source": _str(betas_source),
           "valid_runs_ms": runs_ms, "category": _str(f"erh_{tier}"), "split": _str("test"), "canonical_hand": _str("right"),
           "params_layout": np.array(["t[0:3]", "R_root[3:6]", "hand_pose_residual_aa[6:51]"])}
    done = {"trial": trial, "tier": tier, "n_events": est["n_kept"], "n_ms": n_ms, "n_labels": len(ids),
            "n_label_time_out_of_range": n_oob, "n_label_runs": len(runs), "betas_std_max": betas_std,
            "raw_event_file": str(RAW / str(trial) / "event.aedat4"), "converter_sha256": sha(__file__),
            "convert_seconds": time.time() - t_start}
    write_trial_files(trial, root, events, offsets, tsub, labels, aux, done)
    n_mano, n_j = len(annot["mano"]), len(annot["3d_joints"])
    run_len = [b - a + 1 for a, b in runs]
    return {"trial": trial, "tier": tier, "subject_id": annot["subject_id"], "scene": annot["scene"],
            "gesture": annot["gesture_type"], "motion": annot["motion_type"], "annoted": bool(annot["annoted"]),
            "events": est, "n_labels": len(ids), "n_mano": n_mano, "n_3d_joints": n_j,
            "n_label_time_out_of_range": n_oob, "n_labels_before_first_event": n_before, "n_labels_after_last_event": n_after,
            "runs": {"n": len(runs), "lengths": run_len},
            "label_ms_first": int(label_ms[0]) if len(ids) else None, "label_ms_last": int(label_ms[-1]) if len(ids) else None,
            "betas_std_max": betas_std, "betas_trial": betas_trial.tolist(), "mano_check": mano_check, "depth": depth,
            "seconds": time.time() - t_start}


def _convert_worker(args):
    trial, perm, lut_path = args
    r = convert_trial(trial, perm, np.load(lut_path))
    print(f"[convert] trial {trial} {r['tier']} events {r['events']['n_kept']:,} labels {r['n_labels']} {r['seconds']:.0f}s", flush=True)
    return r


# ------------------------------------------------------------------------------------------------ loader


_CTX = {}


def _context_class():
    if "c" not in _CTX:
        from tools.pfo.nets import Context

        @dataclass
        class ErhContext(Context):
            labels: object = None

        _CTX["c"] = ErhContext
    return _CTX["c"]


def load(trial, device="cpu", root=None):
    """The recording as a `tools.pfo.nets.Context` (+ `labels`); `gt` is per-ms 51-D, NaN except at the labelled ms."""
    import torch
    ErhContext = _context_class()
    b = trial_base(trial, root)
    events = np.load(f"{b}_events.npy", mmap_mode="r")
    offsets = np.load(f"{b}_offsets.npy")
    tsub = np.load(f"{b}_tsub.npy", mmap_mode="r")
    aux = np.load(f"{b}_aux.npz", allow_pickle=True)
    labels = np.load(f"{b}_labels.npz", allow_pickle=True)
    n_ms = len(offsets) - 1
    gt = np.full((n_ms, 51), np.nan, np.float32)
    if "params51" in labels.files:
        gt[labels["label_ms"]] = labels["params51"]
    betas = torch.tensor(np.asarray(aux["betas"], np.float32), device=device).view(1, -1)
    K = torch.tensor(np.asarray(aux["camera_K"], np.float32), device=device).view(1, 3, 3)
    return ErhContext(f"erh_{trial}", events, offsets, tsub, aux, gt, betas, K, labels)


# ------------------------------------------------------------------------------------------------ checks: geometry


_LATTICE = None


def _lattice(n=4):
    global _LATTICE
    if _LATTICE is None:
        w = [(i, j, n - i - j) for i in range(n + 1) for j in range(n + 1 - i)]
        _LATTICE = np.asarray(w, np.float64) / n
    return _LATTICE


def project_grid(verts, K_model):
    """Vertices (.., 3) metres -> continuous (u, v) on the 240 x 180 grid with the model's intrinsics (`stored K * 0.375`)."""
    K = np.asarray(K_model, np.float64)
    z = verts[..., 2]
    return np.stack([K[0, 0] * verts[..., 0] / z + K[0, 2], K[1, 1] * verts[..., 1] / z + K[1, 2]], -1)


def silhouette(uv, faces, dilate=DILATE):
    """Filled hand silhouette on the 240 x 180 grid (bool), dilated by `dilate` px (Chebyshev).

    Mesh vertices and a barycentric lattice on every face are binned with `floor` (the discretisation the events get),
    closed with a 3 x 3 element, holes filled, then dilated with a (2 dilate + 1)^2 square.
    """
    from scipy import ndimage
    tri = uv[faces]                                                     # (F, 3, 2)
    dense = np.einsum("pk,fkd->fpd", _lattice(), tri).reshape(-1, 2)
    ix, iy = np.floor(dense[:, 0]).astype(np.int64), np.floor(dense[:, 1]).astype(np.int64)
    ok = (ix >= 0) & (ix < GRID_W) & (iy >= 0) & (iy < GRID_H)
    m = np.zeros((GRID_H, GRID_W), bool)
    m[iy[ok], ix[ok]] = True
    m = ndimage.binary_closing(m, structure=np.ones((3, 3), bool))
    m = ndimage.binary_fill_holes(m)
    if dilate:
        m = ndimage.binary_dilation(m, structure=np.ones((2 * dilate + 1, 2 * dilate + 1), bool))
    return m


def inside_fraction(mask, ev, shift=(0, 0)):
    """Events whose bin lies in `mask` shifted by `shift` px (dx, dy); returns (inside, total)."""
    if len(ev) == 0:
        return 0, 0
    ex = ev[:, 0].astype(np.int64) - shift[0]
    ey = ev[:, 1].astype(np.int64) - shift[1]
    ok = (ex >= 0) & (ex < GRID_W) & (ey >= 0) & (ey < GRID_H)
    return int(mask[ey[ok], ex[ok]].sum()), len(ev)


def shift_search(mask, ev, rng=SHIFT_RANGE):
    """Inside counts for every shift in [-rng, rng]^2: events are displaced by (dx, dy) from the silhouette."""
    r = rng
    pad = np.zeros((GRID_H + 2 * r, GRID_W + 2 * r), np.int64)
    pad[r:r + GRID_H, r:r + GRID_W] = mask
    flat = ev[:, 1].astype(np.int64) * GRID_W + ev[:, 0]
    u, c = np.unique(flat, return_counts=True)
    ex, ey = (u % GRID_W), (u // GRID_W)
    d = np.arange(-r, r + 1)
    out = np.zeros((2 * r + 1, 2 * r + 1), np.int64)
    for iy, dy in enumerate(d):
        for ix, dx in enumerate(d):
            sx, sy = ex - dx + r, ey - dy + r
            ok = (sx >= 0) & (sx < pad.shape[1]) & (sy >= 0) & (sy < pad.shape[0])
            out[iy, ix] = int((pad[sy[ok], sx[ok]] * c[ok]).sum())
    return out                                                      # [dy index, dx index]


def check_k_trial(trial, verts, faces, aux, labels, events, offsets):
    """The K-choice rule on one trial (every 10th labelled frame, events within +-10 ms)."""
    res = {}
    n = len(labels["label_ms"])
    frames = np.arange(0, n, K_EVERY)
    names = {"K": aux["K_src"], "K_new": aux["K_new_src"], "K_old": aux["K_old_src"]}
    direct_vs_model = 0.0
    acc = {k: {"inside": 0, "total": 0, "null_inside": 0, "frame_frac": [], "shift": None, "area": []} for k in names}
    n_ms = len(offsets) - 1
    for k, Ksrc in names.items():
        Km = stored_camera_K(Ksrc).astype(np.float64) * RENDER_SCALE          # the model's K (stored x 0.375)
        Kd = np.eye(3)
        Kd[0, 0], Kd[0, 2], Kd[1, 1], Kd[1, 2] = Ksrc[0, 0] * SX, Ksrc[0, 2] * SX, Ksrc[1, 1] * SY, Ksrc[1, 2] * SY
        sh_total = np.zeros((2 * SHIFT_RANGE + 1,) * 2, np.int64)
        for f in frames:
            m0 = int(labels["label_ms"][f])
            a0, a1 = int(offsets[max(m0 - K_WINDOW_MS, 0)]), int(offsets[min(m0 + K_WINDOW_MS + 1, n_ms)])
            ev = np.asarray(events[a0:a1])
            if len(ev) == 0:
                continue
            uv = project_grid(verts[f], Km)
            direct_vs_model = max(direct_vs_model, float(np.abs(uv - project_grid(verts[f], Kd)).max()))
            mask = silhouette(uv, faces)
            ins, tot = inside_fraction(mask, ev)
            nul, _ = inside_fraction(mask, ev, NULL_SHIFT)
            acc[k]["inside"] += ins
            acc[k]["total"] += tot
            acc[k]["null_inside"] += nul
            acc[k]["frame_frac"].append(ins / tot)
            acc[k]["area"].append(float(mask.mean()))
            sh_total += shift_search(mask, ev)
        acc[k]["shift_counts"] = sh_total
    for k, a in acc.items():
        tot = max(a["total"], 1)
        sh = a.pop("shift_counts")
        iy, ix = np.unravel_index(int(np.argmax(sh)), sh.shape)
        ff = np.asarray(a["frame_frac"])
        res[k] = {"inside_frac": a["inside"] / tot, "null_inside_frac": a["null_inside"] / tot, "n_events": a["total"],
                  "n_frames": len(ff), "frame_median": float(np.median(ff)) if len(ff) else None,
                  "mask_area_frac": float(np.mean(a["area"])) if a["area"] else None,
                  "best_shift_dx_dy": [int(ix - SHIFT_RANGE), int(iy - SHIFT_RANGE)],
                  "inside_frac_at_best_shift": float(sh[iy, ix] / tot), "inside_frac_at_zero_shift": float(sh[SHIFT_RANGE, SHIFT_RANGE] / tot)}
    res["model_vs_direct_projection_max_px"] = direct_vs_model
    return res


# ------------------------------------------------------------------------------------------------ checks: clock


def xcorr_lag(rate, speed, lags=CLOCK_LAGS):
    """Pearson correlation of `rate[m]` with `speed[m - L]` for L in lags; NaN entries are skipped.

    L > 0: `speed` has to be delayed by L ms to match `rate`, i.e. the event clock lags the label clock by L ms.
    """
    lo, hi = lags
    L = np.arange(lo, hi + 1)
    c = np.full(len(L), np.nan)
    n = len(rate)
    for i, lag in enumerate(L):
        if lag >= 0:
            a, b = rate[lag:], speed[:n - lag]
        else:
            a, b = rate[:n + lag], speed[-lag:]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 200:
            continue
        a, b = a[ok], b[ok]
        sa, sb = a.std(), b.std()
        if sa > 0 and sb > 0:
            c[i] = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
    return L, c


def speed_series(verts, label_t_us, frame_id, n_ms):
    """Mean vertex speed (m/s) between consecutive labelled frames, linearly interpolated to ms (NaN outside runs/gaps)."""
    dt = np.diff(label_t_us) / 1e6
    ok = np.diff(frame_id) == 1
    sp = np.linalg.norm(np.diff(verts, axis=0), axis=-1).mean(-1) / dt
    mid = (label_t_us[:-1] + label_t_us[1:]) / 2000.0                    # ms, mid-point of each label pair
    out = np.full(n_ms, np.nan)
    for a, b in true_runs(ok):
        if b > a:
            ms = np.arange(int(np.ceil(mid[a])), int(np.floor(mid[b])) + 1)
            ms = ms[(ms >= 0) & (ms < n_ms)]
            out[ms] = np.interp(ms, mid[a:b + 1], sp[a:b + 1])
    return out


def true_runs(ok):
    """Maximal runs of True in a boolean array as (first, last) indices."""
    ok = np.asarray(ok, bool)
    if not ok.any():
        return []
    d = np.diff(np.concatenate([[0], ok.astype(np.int8), [0]]))
    return [(int(a), int(b) - 1) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def check_clock_trial(verts, labels, offsets):
    n_ms = len(offsets) - 1
    rate = np.diff(offsets).astype(np.float64)
    k = np.ones(CLOCK_SMOOTH_MS) / CLOCK_SMOOTH_MS
    rate_s = np.convolve(rate, k, mode="same")
    sp = speed_series(verts.astype(np.float64), labels["label_t_us"], labels["frame_id"], n_ms)
    L, c = xcorr_lag(rate_s, sp)
    i = int(np.nanargmax(c))
    med = float(np.nanmedian(c))
    i0 = int(np.where(L == 0)[0][0])
    thr = c[i] - 0.05
    lo_i = hi_i = i
    while lo_i > 0 and np.isfinite(c[lo_i - 1]) and c[lo_i - 1] >= thr:
        lo_i -= 1
    while hi_i < len(L) - 1 and np.isfinite(c[hi_i + 1]) and c[hi_i + 1] >= thr:
        hi_i += 1
    return {"peak_lag_ms": int(L[i]), "peak_corr": float(c[i]), "corr_at_0": float(c[i0]), "curve_median": med,
            "plateau_ms_within_0.05": [int(L[lo_i]), int(L[hi_i])], "n_ms_valid": int(np.isfinite(sp).sum())}, (L, c)


def check_trial(trial, root=None):
    """K-choice rule + clock for one 3d-tier trial (needs the converted files)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    b = trial_base(trial, root)
    events = np.load(f"{b}_events.npy", mmap_mode="r")
    offsets = np.load(f"{b}_offsets.npy")
    aux = dict(np.load(f"{b}_aux.npz", allow_pickle=True))
    labels = dict(np.load(f"{b}_labels.npz", allow_pickle=True))
    _, verts = fk51(labels["params51"], labels["betas_trial"])
    faces = get_mano().f.numpy()
    t0 = time.time()
    kres = check_k_trial(trial, verts, faces, aux, labels, events, offsets)
    cres, curve = check_clock_trial(verts, labels, offsets)
    return {"trial": trial, "K": kres, "clock": cres, "curve": curve, "seconds": time.time() - t0}


def _check_worker(trial):
    r = check_trial(trial)
    k = r["K"]
    print(f"[check] trial {trial}: inside K {k['K']['inside_frac']:.3f} K_new {k['K_new']['inside_frac']:.3f} "
          f"null {k['K']['null_inside_frac']:.3f}; clock peak {r['clock']['peak_lag_ms']} ms ({r['clock']['peak_corr']:.2f})", flush=True)
    r["curve"] = [np.asarray(r["curve"][0]).tolist(), np.asarray(r["curve"][1]).tolist()]
    return r


# ------------------------------------------------------------------------------------------------ 2-D joint cross-check


def check_k_2d(trial):
    """Independent K evidence on the fast trials: annotation 3-D joints (event frame) projected with K / K_new / K_old vs the
    undistorted 2-D event annotations (px on the 346 x 260 frame). Descriptive; the K rule uses the events."""
    b = trial_base(trial)
    labels = dict(np.load(f"{b}_labels.npz", allow_pickle=True))
    aux = dict(np.load(f"{b}_aux.npz", allow_pickle=True))
    if "joints3d_cam" not in labels:
        return None
    ids2 = list(labels["frame_id"])
    ids3 = list(labels["joints3d_cam_frame_id"])
    common = [i for i in ids3 if i in ids2]
    if not common:
        return None
    u = labels["joints2d_undist"][[ids2.index(i) for i in common]].astype(np.float64)
    j = labels["joints3d_cam"][[ids3.index(i) for i in common]].astype(np.float64)
    out = {"n_frames": len(common)}
    for name, Ks in (("K", aux["K_src"]), ("K_new", aux["K_new_src"]), ("K_old", aux["K_old_src"])):
        pr = np.stack([Ks[0, 0] * j[..., 0] / j[..., 2] + Ks[0, 2], Ks[1, 1] * j[..., 1] / j[..., 2] + Ks[1, 2]], -1)
        e = np.linalg.norm(pr - u, axis=-1)
        d = (u - pr).reshape(-1, 2)
        out[name] = {"median_px": float(np.median(e)), "p90_px": float(np.percentile(e, 90)),
                     "mean_offset_dx_dy": d.mean(0).tolist()}
    return out


# ------------------------------------------------------------------------------------------------ aggregation / report


def read_meta51(path):
    """Per-ms 51-D states of a training recording (`.meta`: int32 ncomps, then float64[51] + 2 bytes per row)."""
    import struct
    with open(path, "rb") as f:
        (n,) = struct.unpack("<i", f.read(4))
        assert n == 51, n
        raw = np.fromfile(f, dtype=np.dtype([("data", np.float64, n), ("m0", np.uint8), ("m1", np.uint8)]))
    return raw["data"].astype(np.float32)


def training_reference():
    """The same event statistics for the recordings the trackers were trained / developed on (read-only)."""
    out = {}
    names = [("val", "zgz_local"), ("val", "zgz_global")] + [("train", f"{s}_local") for s in
             ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf")]
    for split, seq in names:
        base = TRAIN_DATA / split / seq
        if not Path(f"{base}_events.npy").exists():
            continue
        ev = np.load(f"{base}_events.npy", mmap_mode="r")
        off = np.load(f"{base}_offsets.npy")
        s = {"n_events": int(len(ev)), "n_ms": int(len(off) - 1)}
        s["on_fraction"] = float(np.mean(np.asarray(ev[::7, 2])))
        s.update(event_rate_stats(ev, off))
        if Path(f"{base}.meta").exists():
            tz = read_meta51(f"{base}.meta")[::5, 2]
            tz = tz[tz > 0.1]                                    # rows of invalid ms are zero
            s["transl_z_m_p5_p50_p95"] = np.percentile(tz, [5, 50, 95]).tolist()
            s["transl_z_m_min_max"] = [float(tz.min()), float(tz.max())]
        out[seq] = s
    return out


def finalize(k_choice, trials_doc, subject_betas):
    """Apply the K rule's choice to every aux, and fill the 2-D trials' betas from the same subject (3-D trials)."""
    done = []
    for t in trials_doc["included"]:
        b = trial_base(t)
        aux = dict(np.load(f"{b}_aux.npz", allow_pickle=True))
        Ksrc = aux["K_src"] if k_choice == "K" else aux["K_new_src"]
        aux["camera_K"] = stored_camera_K(Ksrc)
        aux["camera_K_choice"] = _str(k_choice)
        if str(aux["tier"]) == "2d":
            sid = int(aux["subject_id"])
            sb = subject_betas.get(sid)
            if sb is not None:
                aux["betas"] = np.asarray(sb["mean"], np.float32)
                aux["betas_source"] = _str(f"mean of subject {sid} trials {sb['trials']} (no MANO fit on this fast trial)")
            else:
                aux["betas"] = np.zeros(10, np.float32)
                aux["betas_source"] = _str("zeros (no 3-D trial of this subject)")
        np.savez(f"{b}_aux.npz", **aux)
        done.append(t)
    return done


WORKERS = {"convert": _convert_worker, "check": _check_worker}


def _pool(kind, args, workers):
    """Run `kind` workers as separate processes (at most `workers` at a time) and collect their JSON results.

    Plain `subprocess`, not `multiprocessing`: no POSIX semaphores, so nothing under /dev/shm.
    """
    tmp = OUT / ".work"
    tmp.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
           "ERH_TORCH_THREADS": "1", "PYTHONWARNINGS": "ignore"}
    pending, running, results = list(enumerate(args)), [], []
    while pending or running:
        while pending and len(running) < workers:
            i, a = pending.pop(0)
            out = tmp / f"{kind}_{i}.json"
            out.unlink(missing_ok=True)
            cmd = [sys.executable, "-m", "tools.tpd.erh_convert", "worker", kind, json.dumps(jsonable(a)), str(out)]
            running.append((subprocess.Popen(cmd, cwd=str(ROOT), env=env), out, a))
        time.sleep(0.2)
        for item in running[:]:
            proc, out, a = item
            if proc.poll() is not None:
                running.remove(item)
                if proc.returncode != 0:
                    for q, _, _ in running:
                        q.terminate()
                    raise RuntimeError(f"{kind} worker failed for {a} (exit {proc.returncode})")
                results.append(read_json(out))
                out.unlink()
    try:
        tmp.rmdir()
    except OSError:
        pass
    return results


def cmd_convert(trials, workers, perm_info):
    doc = plan()
    todo = trials or doc["included"]
    cams = [np.array(read_annot(todo[0])["camera_info"]["event"][k], np.float64) for k in ("K_old", "dist", "K_new")]
    build_lut(*cams)
    path = lut_path(*cams)
    todo = sorted(todo, key=lambda t: -doc["trials"][str(t)]["event_bytes"])        # biggest first
    res = _pool("convert", [(t, perm_info["perm"], str(path)) for t in todo], workers)
    return {r["trial"]: r for r in res}


def cmd_check(trials, workers):
    todo = trials or [t for t in plan()["included"] if plan()["trials"][str(t)]["tier"] == "3d"]
    res = _pool("check", todo, workers)
    return {r["trial"]: r for r in res}


def aggregate(conv, chk, perm_info, k2d, with_reference=True):
    """Receipt content from per-trial results (pure function of its inputs)."""
    doc = plan()
    tiers = {k: [t for t in doc["included"] if doc["trials"][str(t)]["tier"] == k] for k in TIERS}
    rec = {"raw": str(RAW), "converter_sha256": sha(__file__), "excluded_left_hand_trials": doc["excluded_left"],
           "scenes": {str(t): doc["trials"][str(t)]["scene"] for t in doc["included"]},
           "tiers": {k: v for k, v in tiers.items()}, "tier_counts": {k: len(v) for k, v in tiers.items()},
           "joint_perm": perm_info}
    # 1. MANO
    mano = {}
    groups = []
    for tier in ("3d", "3d_unchecked"):
        rows_all = [conv[t] for t in tiers[tier] if t in conv]
        groups.append((tier, [r for r in rows_all if not (r["betas_std_max"] or 0) > 1e-6]))
        groups.append((tier + "_varying_shape", [r for r in rows_all if (r["betas_std_max"] or 0) > 1e-6]))
    for tier, rows in groups:
        if not rows:
            continue
        names = [k for k in rows[0]["mano_check"] if isinstance(rows[0]["mano_check"][k], dict) and "median" in rows[0]["mano_check"][k]]
        agg = {}
        for n in names:
            sel = [r["mano_check"][n] for r in rows if n in r["mano_check"]]
            pj = np.mean([s["per_joint_mean"] for s in sel], 0)
            agg[n] = {"trial_median_of_median_mm": float(np.median([s["median"] for s in sel])),
                      "trial_median_of_p95_mm": float(np.median([s["p95"] for s in sel])),
                      "mean_mm": float(np.average([s["mean"] for s in sel], weights=[s["n_frames"] for s in sel])),
                      "max_trial_median_mm": float(np.max([s["median"] for s in sel])),
                      "per_joint_mean_mm": pj.tolist(), "n_trials": len(sel)}
        agg["root_joint_vs_ours_max_abs_m"] = float(max(r["mano_check"]["root_joint_vs_ours_max_abs_m"] for r in rows))
        agg["shape_std_max"] = float(max(r["mano_check"]["shape_std_max"] for r in rows))
        mano[tier] = agg
    rec["mano"] = {"summary": mano, "per_trial": {t: conv[t]["mano_check"] for t in conv if conv[t]["mano_check"]}}
    # 2. K choice
    if chk:
        med = {k: float(np.median([chk[t]["K"][k]["inside_frac"] for t in chk])) for k in K_NAMES}
        med_null = {k: float(np.median([chk[t]["K"][k]["null_inside_frac"] for t in chk])) for k in K_NAMES}
        choice = max(K_CANDIDATES, key=lambda k: med[k])
        better = int(sum(chk[t]["K"]["K_new"]["inside_frac"] > chk[t]["K"]["K"]["inside_frac"] for t in chk))
        shifts = {k: np.array([chk[t]["K"][k]["best_shift_dx_dy"] for t in chk]) for k in K_NAMES}
        rec["k_choice"] = {"rule": "median over 3d-tier trials of the pooled inside-fraction (every 10th label, events within +-10 ms, "
                                   "silhouette dilated 2 px); candidates K and K_new; K_old diagnostic only",
                           "median_inside_frac": med, "median_null_inside_frac": med_null, "chosen": choice,
                           "n_trials": len(chk), "n_trials_K_new_better": better,
                           "best_shift_median_dx_dy": {k: np.median(v, 0).tolist() for k, v in shifts.items()},
                           "best_shift_zero_frac": {k: float(np.mean(np.all(v == 0, 1))) for k, v in shifts.items()},
                           "model_vs_direct_projection_max_px": float(max(chk[t]["K"]["model_vs_direct_projection_max_px"] for t in chk)),
                           "per_trial": {t: chk[t]["K"] for t in sorted(chk)}}
        pk = [chk[t]["clock"] for t in sorted(chk)]
        rec["clock"] = {"lags_ms": list(CLOCK_LAGS), "smooth_ms": CLOCK_SMOOTH_MS,
                        "sign": "L>0: the event clock lags the label clock by L ms (corr(rate[m], speed[m-L]))",
                        "peak_lag_median_ms": float(np.median([p["peak_lag_ms"] for p in pk])),
                        "peak_lag_min_max_ms": [int(min(p["peak_lag_ms"] for p in pk)), int(max(p["peak_lag_ms"] for p in pk))],
                        "peak_corr_median": float(np.median([p["peak_corr"] for p in pk])),
                        "peak_lag_median_ms_trials_corr_ge_0.5": float(np.median([p["peak_lag_ms"] for p in pk if p["peak_corr"] >= 0.5])),
                        "n_trials_corr_ge_0.5": int(sum(p["peak_corr"] >= 0.5 for p in pk)),
                        "peak_lag_q25_q75_ms": [float(np.percentile([p["peak_lag_ms"] for p in pk], 25)),
                                                float(np.percentile([p["peak_lag_ms"] for p in pk], 75))],
                        "n_trials_peak_within_20ms": int(sum(abs(p["peak_lag_ms"]) <= 20 for p in pk)),
                        "n_trials_zero_inside_plateau": int(sum(p["plateau_ms_within_0.05"][0] <= 0 <= p["plateau_ms_within_0.05"][1] for p in pk)),
                        "corr_at_0_median": float(np.median([p["corr_at_0"] for p in pk])),
                        "per_trial": {t: chk[t]["clock"] for t in sorted(chk)}}
    if k2d:
        rec["k_check_2d_joints"] = k2d
    # 4. event statistics
    ev = {t: conv[t]["events"] for t in conv}
    rec["event_stats"] = {"per_trial": ev, "training_reference": training_reference() if with_reference else {}, "source_pixels_per_bin": None}
    # 7. depth domain (extra): wrist depth of the labelled frames per tier vs the training recordings
    dep = {}
    for k in TIERS:
        rows = [conv[t]["depth"] for t in tiers[k] if t in conv and conv[t].get("depth")]
        if rows:
            allp = np.array([r["transl_z_m_p5_p50_p95"] for r in rows])
            dep[k] = {"n_trials": len(rows), "trial_median_transl_z_m_min_med_max": [float(allp[:, 1].min()), float(np.median(allp[:, 1])), float(allp[:, 1].max())],
                      "p5_of_p5_p95_of_p95_transl_z_m": [float(allp[:, 0].min()), float(allp[:, 2].max())],
                      "wrist_z_m_trial_median_med": float(np.median([r["wrist_z_m_p5_p50_p95"][1] for r in rows])),
                      "per_trial_transl_z_p5_p50_p95": {t: conv[t]["depth"]["transl_z_m_p5_p50_p95"] for t in tiers[k] if t in conv and conv[t].get("depth")}}
    rec["depth"] = dep
    # 5. betas
    by_subject = {}
    for t, r in conv.items():
        if r["tier"] in ("3d", "3d_unchecked"):
            by_subject.setdefault(r["subject_id"], []).append((t, np.asarray(r["betas_trial"], np.float64)))
    subj = {}
    for s, lst in sorted(by_subject.items()):
        B = np.stack([b for _, b in lst])
        subj[s] = {"trials": [t for t, _ in lst], "mean": B.mean(0).tolist(), "max_abs_dev_from_mean": float(np.abs(B - B.mean(0)).max()),
                   "max_pair_L2": float(max(np.linalg.norm(a - b) for a in B for b in B)),
                   "n_distinct_betas": int(len({tuple(np.round(b, 6)) for b in B}))}
    rec["betas"] = {"std_across_frames_max_over_trials": float(max(r["betas_std_max"] for r in conv.values() if r["betas_std_max"] is not None)),
                    "per_trial_std": {t: conv[t]["betas_std_max"] for t in sorted(conv)},
                    "per_subject": subj}
    # 6. labels
    lab = {}
    for k in TIERS:
        rows = [conv[t] for t in tiers[k] if t in conv]
        lens = [x for r in rows for x in r["runs"]["lengths"]]
        lab[k] = {"n_trials": len(rows), "n_labels": int(sum(r["n_labels"] for r in rows)),
                  "n_runs": int(sum(r["runs"]["n"] for r in rows)),
                  "run_length_min_median_max": [int(min(lens)), float(np.median(lens)), int(max(lens))] if lens else None,
                  "n_label_time_out_of_range": int(sum(r["n_label_time_out_of_range"] for r in rows)),
                  "n_before_first_event": int(sum(r["n_labels_before_first_event"] for r in rows)),
                  "n_after_last_event": int(sum(r["n_labels_after_last_event"] for r in rows))}
    rec["labels"] = {"per_tier": lab, "per_trial": {t: {"n_labels": conv[t]["n_labels"], "n_mano": conv[t]["n_mano"],
                                                       "n_3d_joints": conv[t]["n_3d_joints"], "runs": conv[t]["runs"],
                                                       "n_before": conv[t]["n_labels_before_first_event"], "n_after": conv[t]["n_labels_after_last_event"],
                                                       "label_ms_first": conv[t]["label_ms_first"], "label_ms_last": conv[t]["label_ms_last"]}
                                                  for t in sorted(conv)}}
    return rec


def _fmt(x, nd=2):
    return "-" if x is None else f"{x:.{nd}f}"


def write_report(rec, path):
    L = ["# ERH conversion report", "", f"raw: `{rec['raw']}`; excluded left-hand trials: {len(rec['excluded_left_hand_trials'])}",
         f"tiers: {rec['tier_counts']}", ""]
    jp = rec["joint_perm"]
    L += ["## Joint order", f"permutation identity: {jp['is_identity']}; per-frame assignment equals the pooled one in "
          f"{100 * jp['consistency']:.2f} % of {jp['n_frames']} frames ({jp['n_distinct_per_frame_perms']} distinct); "
          f"pooled matched-joint distance {jp['pooled_cost_mm']:.2f} mm; smallest row margin {jp['min_row_margin_mm']:.2f} mm", ""]
    L += ["## 1. MANO conversion vs annotation joints (event frame, mm)", "", "| tier | convention | median of trial medians | median of trial p95 | mean | worst trial median |", "|---|---|---|---|---|---|"]
    for tier, agg in rec["mano"]["summary"].items():
        for n, v in agg.items():
            if isinstance(v, dict) and n != "used":
                L.append(f"| {tier} | {n} | {v['trial_median_of_median_mm']:.2f} | {v['trial_median_of_p95_mm']:.2f} | {v['mean_mm']:.2f} | {v['max_trial_median_mm']:.2f} |")
    for tier, agg in rec["mano"]["summary"].items():
        L.append(f"\n{tier}: root_joint vs ours max abs {agg['root_joint_vs_ours_max_abs_m']:.2e} m; shape std across frames max {agg['shape_std_max']:.2e}; trials {agg['used']['n_trials']}")
        pj = agg["used"]["per_joint_mean_mm"]
        L.append(f"{tier} per-joint mean error (our joint order 0..20): " + " ".join(f"{v:.1f}" for v in pj))
    if "k_choice" in rec:
        kc = rec["k_choice"]
        L += ["", "## 2. K choice (pre-written rule)", "", f"chosen: **{kc['chosen']}** (K_new better in {kc['n_trials_K_new_better']} / {kc['n_trials']} trials); "
              f"model-convention vs direct projection max diff {kc['model_vs_direct_projection_max_px']:.2e} px", "",
              "| K | median inside | median null (+8, 0) | median best shift (dx, dy) | share of trials with best shift (0, 0) |", "|---|---|---|---|---|"]
        for k in K_NAMES:
            L.append(f"| {k} | {kc['median_inside_frac'][k]:.4f} | {kc['median_null_inside_frac'][k]:.4f} | {kc['best_shift_median_dx_dy'][k]} | {kc['best_shift_zero_frac'][k]:.2f} |")
        L += ["", "per trial (inside fraction K / K_new / K_old, null for K, best shift for the chosen K):", "",
              "| trial | scene | K | K_new | K_old | null (K) | best shift (chosen) | inside at best shift |", "|---|---|---|---|---|---|---|---|"]
        for t, v in sorted(kc["per_trial"].items()):
            c = kc["chosen"]
            L.append(f"| {t} | {rec['scenes'][str(t)]} | {v['K']['inside_frac']:.3f} | {v['K_new']['inside_frac']:.3f} | {v['K_old']['inside_frac']:.3f} | "
                     f"{v['K']['null_inside_frac']:.3f} | {v[c]['best_shift_dx_dy']} | {v[c]['inside_frac_at_best_shift']:.3f} |")
    if "k_check_2d_joints" in rec:
        L += ["", "2-D joint cross-check (fast trials; annotation 3-D joints projected vs undistorted 2-D event joints, px):", ""]
        for t, v in rec["k_check_2d_joints"].items():
            if v:
                L.append(f"- trial {t} (n={v['n_frames']}): " + "; ".join(f"{k} median {v[k]['median_px']:.2f} offset {np.round(v[k]['mean_offset_dx_dy'], 2).tolist()}" for k in K_NAMES))
    if "clock" in rec:
        c = rec["clock"]
        L += ["", "## 3. Clock", "", f"{c['sign']}; lags {c['lags_ms']} ms, rate smoothed {c['smooth_ms']} ms",
              f"peak lag median {c['peak_lag_median_ms']:.0f} ms (q25/q75 {c['peak_lag_q25_q75_ms']}, min/max {c['peak_lag_min_max_ms']}), "
              f"median over the {c['n_trials_corr_ge_0.5']} trials with peak corr >= 0.5: {c['peak_lag_median_ms_trials_corr_ge_0.5']:.0f} ms; "
              f"peak corr median {c['peak_corr_median']:.2f}, corr at 0 median {c['corr_at_0_median']:.2f}; "
              f"|peak lag| <= 20 ms in {c['n_trials_peak_within_20ms']} trials; lag 0 inside the peak - 0.05 plateau in {c['n_trials_zero_inside_plateau']} trials", "",
              "| trial | scene | peak lag (ms) | peak corr | corr at 0 | plateau (peak - 0.05) |", "|---|---|---|---|---|---|"]
        for t, v in sorted(c["per_trial"].items()):
            L.append(f"| {t} | {rec['scenes'][str(t)]} | {v['peak_lag_ms']} | {v['peak_corr']:.2f} | {v['corr_at_0']:.2f} | {v['plateau_ms_within_0.05']} |")
    es = rec["event_stats"]
    L += ["", "## 4. Event statistics (240 x 180 grid)", "",
          "| trial | tier | duration s | events (M) | ev/ms mean | ev/ms median | ON frac | dropped strict | ref rule drops / clips | ev/px/50ms | active px/50ms |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t, v in sorted(es["per_trial"].items(), key=lambda kv: int(kv[0])):
        tier = [k for k in TIERS if int(t) in rec["tiers"][k]][0]
        L.append(f"| {t} | {tier} | {v['duration_s']:.1f} | {v['n_kept'] / 1e6:.2f} | {v['events_per_ms_mean']:.0f} | {v['events_per_ms_median']:.0f} | "
                 f"{v['on_fraction']:.3f} | {v['drop_frac_strict']:.4f} | {v['drop_frac_reference_rule']:.4f} / {v['clipped_by_reference_frac']:.4f} | "
                 f"{v['events_per_pixel_per_50ms_mean']:.3f} | {v.get('active_pixels_per_50ms_mean', float('nan')):.0f} |")
    if es.get("floor_binning_offset_px"):
        fo = es["floor_binning_offset_px"]
        L.append(f"\nfloor-binning offset (bin index - continuous coordinate, mean over legal sensor pixels): dx {fo['dx_mean']:.3f} px, dy {fo['dy_mean']:.3f} px (training: -0.4375 px each)")
    if es.get("source_pixels_per_bin"):
        s = es["source_pixels_per_bin"]
        c = s["central_30_150_x_40_200"]
        L.append(f"\nsource pixels per output bin: mean {s['mean']:.2f}, median {s['median']}, min {s['min']}, max {s['max']}, empty bins {100 * s['frac_bins_empty']:.2f} %, "
                 f">= 3: {100 * s['frac_bins_ge_3']:.2f} %, histogram {s['hist']}; central region (rows 30-150, cols 40-200): mean {c['mean']:.2f}, "
                 f"empty {100 * c['frac_empty']:.2f} %, >= 3 {100 * c['frac_ge_3']:.2f} %, histogram {c['hist']}; empty bins span x {s['empty_bins_x_range']}, y {s['empty_bins_y_range']}")
    L += ["", "training / development reference:", "", "| seq | ev/ms mean | ON frac | ev/px/50ms | active px/50ms |", "|---|---|---|---|---|"]
    for sname, v in es["training_reference"].items():
        L.append(f"| {sname} | {v['events_per_ms_mean']:.0f} | {v['on_fraction']:.3f} | {v['events_per_pixel_per_50ms_mean']:.3f} | {v.get('active_pixels_per_50ms_mean', float('nan')):.0f} |")
    L += ["", "## 7. Depth domain (extra)", "", "transl z (m), MANO translation = wrist depth - 0.006 m approx.", "",
          "| set | trials | trial medians min / median / max | overall p5 .. p95 |", "|---|---|---|---|"]
    for k, v in rec["depth"].items():
        a, b2 = v["trial_median_transl_z_m_min_med_max"], v["p5_of_p5_p95_of_p95_transl_z_m"]
        L.append(f"| ERH {k} | {v['n_trials']} | {a[0]:.3f} / {a[1]:.3f} / {a[2]:.3f} | {b2[0]:.3f} .. {b2[1]:.3f} (min of trial p5 .. max of trial p95) |")
    for sname, v in es["training_reference"].items():
        if "transl_z_m_p5_p50_p95" in v:
            L.append(f"| train/dev {sname} | 1 | p50 {v['transl_z_m_p5_p50_p95'][1]:.3f} | {v['transl_z_m_p5_p50_p95'][0]:.3f} .. {v['transl_z_m_p5_p50_p95'][2]:.3f} (min/max {v['transl_z_m_min_max'][0]:.3f} / {v['transl_z_m_min_max'][1]:.3f}) |")
    b = rec["betas"]
    varying = {t: round(r, 3) for t, r in rec["betas"]["per_trial_std"].items() if (r or 0) > 1e-6}
    L += ["", "## 5. Betas", f"std across frames within a trial (max over the 10 shape dims): 0 in {sum(1 for r in rec['betas']['per_trial_std'].values() if r is not None and r <= 1e-6)} trials; "
          f"varying per frame in trials {varying} (their trial betas = median over frames); 2-D trials have no MANO fit", "",
          "| subject | trials | distinct betas | max abs dev from mean | max pairwise L2 |", "|---|---|---|---|---|"]
    for s, v in b["per_subject"].items():
        L.append(f"| {s} | {v['trials']} | {v['n_distinct_betas']} | {v['max_abs_dev_from_mean']:.3f} | {v['max_pair_L2']:.3f} |")
    L += ["", "## 6. Labels", "", "| tier | trials | labels kept | contiguous runs | run length min / median / max | dropped: before first / after last event |", "|---|---|---|---|---|---|"]
    for k, v in rec["labels"]["per_tier"].items():
        L.append(f"| {k} | {v['n_trials']} | {v['n_labels']} | {v['n_runs']} | {v['run_length_min_median_max']} | {v['n_before_first_event']} / {v['n_after_last_event']} |")
    L += ["", "per trial (labels / mano / 3d_joints / runs):", ""]
    for t, v in rec["labels"]["per_trial"].items():
        L.append(f"- {t}: {v['n_labels']} / {v['n_mano']} / {v['n_3d_joints']} / {v['runs']['n']} run(s) lengths {v['runs']['lengths'][:6]}{'...' if v['runs']['n'] > 6 else ''}; "
                 f"labels outside the event stream: {v['n_before']} before the first event, {v['n_after']} after the last")
    Path(path).write_text("\n".join(L) + "\n")


# ------------------------------------------------------------------------------------------------ CLI


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "worker":                      # `worker <kind> <json args> <out.json>` (see `_pool`)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        kind, payload, out = argv[1:4]
        write_json(out, WORKERS[kind](json.loads(payload)))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("plan", "perm", "convert", "check", "finalize", "report", "all"))
    ap.add_argument("--trials", type=int, nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    assert args.workers <= 8, "at most 8 worker processes"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    OUT.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    doc = plan()
    perm_path = OUT / "joint_perm.json"
    cmds = ("plan", "perm", "convert", "check", "finalize", "report") if args.cmd == "all" else (args.cmd,)

    def perm_info():
        if not perm_path.exists():
            raise SystemExit("run `perm` first")
        return read_json(perm_path)

    tpath = OUT / "stage_seconds.json"
    timings = read_json(tpath) if tpath.exists() else {}

    def stamp(name, t0):
        timings[name] = {"seconds": round(time.time() - t0, 1), "workers": args.workers}
        write_json(tpath, timings)

    if "perm" in cmds:
        t0 = time.time()
        trials3 = [t for t in doc["included"] if doc["trials"][str(t)]["tier"] == "3d"]
        write_json(perm_path, determine_joint_perm(trials3))
        print("[perm]", read_json(perm_path)["perm"], flush=True)
        stamp("perm", t0)
    if "convert" in cmds:
        t0 = time.time()
        conv = cmd_convert(args.trials, args.workers, perm_info())
        write_json(OUT / "convert_trials.json", conv)
        stamp("convert", t0)
    if "check" in cmds:
        t0 = time.time()
        chk = cmd_check(args.trials, args.workers)
        np.savez(OUT / "clock_curves.npz", **{f"trial_{t}_{n}": np.asarray(v, dtype=float) for t, r in chk.items() for n, v in zip(("lags", "corr"), r["curve"])})
        write_json(OUT / "check_trials.json", {t: {k: v for k, v in r.items() if k != "curve"} for t, r in chk.items()})
        stamp("check", t0)
    if "finalize" in cmds or "report" in cmds:
        conv = {int(k): v for k, v in read_json(OUT / "convert_trials.json").items()}
        chk = {int(k): v for k, v in read_json(OUT / "check_trials.json").items()} if (OUT / "check_trials.json").exists() else {}
        rec = aggregate(conv, chk, perm_info(), None, with_reference=False)
        by_subject = {s: v for s, v in rec["betas"]["per_subject"].items()}
        if "finalize" in cmds and chk:
            choice = rec["k_choice"]["chosen"]
            finalize(choice, doc, {int(s): v for s, v in by_subject.items()})
            print(f"[finalize] camera_K = {choice}", flush=True)
        k2d = {t: check_k_2d(t) for t in doc["included"] if doc["trials"][str(t)]["tier"] == "2d" and trial_base(t).with_name(f"erh_{t}_labels.npz").exists()}
        rec = aggregate(conv, chk, perm_info(), k2d)
        und = build_lut(*[np.array(read_annot(doc["included"][0])["camera_info"]["event"][k], np.float64) for k in ("K_old", "dist", "K_new")])
        tabs = bin_tables(und)
        rec["event_stats"]["source_pixels_per_bin"] = source_pixels_per_bin(tabs)
        rec["event_stats"]["floor_binning_offset_px"] = binning_offset(und, tabs)
        rec["event_stats"]["floor_binning_offset_training_px"] = {"dx_mean": -0.4375, "dy_mean": -0.4375, "note": "x_out = floor(3 x / 8) of the 640 x 480 source"}
        rec["stage_seconds"] = timings
        rec["wall_seconds_this_invocation"] = time.time() - t_all
        write_json(OUT / "convert_receipt.json", rec)
        write_report(rec, OUT / "convert_report.md")
        print(f"[report] {OUT / 'convert_receipt.json'}", flush=True)
    print(f"[done] {args.cmd} in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
