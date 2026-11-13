#!/usr/bin/env python3
"""EvRealHands (DAVIS346, EvHandPose's real dataset) -> the hand_data51 layout that tools/real/convert.py writes.

Source  /data1/lyq/code/mesh/data1/EvRealHands/<trial>/{event.aedat4, annot.json}, id_seq_all.json.
Output  /data1/lyq/code/mesh/EventHands/data/erh_n346 (undistorted native 346 x 260) and erh_n256 (256 x 192, EventHPE's
        official frame: the same stream scaled by 256 / 346, 192 / 260), each
            {train,val}/<seq>{_events.npy,_offsets.npy,_tsub.npy,.meta,_aux.npz,.done.json}   (as tools/real/convert.py;
            events uint16 on n346, uint8 on n256; dev and test sequences both live in val/, as in splits_real.json)
            + <seq>_labels.npz (frame ids, label ms, 51-D label, annotation 3-D joints in the camera frame, shapes)
            + <seq>_motion.npz (frame-level local / global motion tiers, see below)
            splits_erh.json (splits_real.json's schema), conversion.json, undistort_lut.npy.

Decisions (coordinator, 2026-11-12):
  split    EvHandPose pretrain.yml: train subjects 0 2 4 5 6 8 9, dev (val, val_core) subject 7, test subjects 1 3. Both
           hands (official flip: true): left hands are mirrored to right hands (events x' = 2 K.cx - x_u after the
           undistortion, labels from `manos_flip`, the annotation 3-D joints x -> -x). Checked and unchecked trials. The
           four 2-D-only `fast` trials (52 53 74 75) are excluded.
  labels   right hands: values from `mano`, screened ids = manos & mano & 3d_joints (`manos` is the dataset's quality
           screen; `mano` frames outside it can be hundreds of mm off); left hands: `manos_flip` & 3d_joints (no `mano_flip`
           exists). One betas per sequence = the median of the per-frame shape. hand_pose stays the residual (hands_mean is
           added by the FK); rot / trans / root_joint are moved to the event camera with EvHandPose's about-root formula,
           trans re-based onto the root joint of the sequence's betas. 51-D = [t, R_root, hand_pose residual].
  clock    no label shift. label time = delta_time + id * 1e6 / 15 us on the clock whose zero is the first (sorted) event;
           t0 = the first screened label (hand_data51 convention): events before it / after the last run are dropped.
  runs     per-ms labels by PCHIP (translation) + RotationSpline (root, 15 joints), as tools/prepare_hand_data
           `interpolate_pose_ms`, only inside valid runs. Train: runs = consecutive ids (gap 0). Dev / test: gaps of up to 4
           missing labels are bridged. A run is [first label ms + 1, last label ms + 1): the evaluator's 50 ms endpoint grid
           (ends = a + 49 + 50 k) then lands exactly on a real label ms every 200 ms. Runs shorter than 300 ms are dropped.
           gt_ms_idx / gt_valid = the real labels (gt_valid: inside a kept run), so a scorer can keep label-time endpoints.
  camera   events are undistorted with cv2.undistortPoints(K_old, dist -> P = K_new) (a per-pixel table, built once in the
           cv2 environment), binned by floor into the canvas; legal = inside the canvas. aux camera_K is the native annotation
           K (it fits the undistorted events best, tools/tpd/erh_convert K rule) scaled so that camera_K * (W / 640) is the
           canvas intrinsics: fx, cx scale 640 / 346 on both canvases; the y row uses the canvas's own y scale.
  names    <subject>_t<trial>_<R|Lm>_<scene>_<gesture> (first token = the subject name; Lm = mirrored left hand).
  motion   the dataset has no sequence-level local / global split. EvHandPose's frame-level tiers (scripts/
           generate_fixed_motion_labels.py: 2-D palm / finger motion between a label and its previous consecutive label,
           quiet / local_1..3 / global_1..3 with the thresholds persisted for val8) are recomputed for every label of every
           sequence into <seq>_motion.npz; for the val8 frames they are checked against the official frame_labels.npz.

    python -m tools.erh.convert run --workers 100 [--only 20 21 ...] [--force]
    python -m tools.erh.convert finalize
    python -m tools.erh.convert validate --workers 48
    python -m tools.erh.convert smoke
Training / reading environment: /data1/lyq/miniconda3/envs/EventHandsTrain/bin/python with CUDA_VISIBLE_DEVICES="" (CPU only).
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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

RAW = Path(os.environ.get("ERH_RAW", "/data1/lyq/code/mesh/data1/EvRealHands"))
DATA = Path("/data1/lyq/code/mesh/EventHands/data")
CANVASES = {"n346": (346, 260, "erh_n346"), "n256": (256, 192, "erh_n256")}
MANO = ROOT / "assets" / "mano_right.npz"
CV2_PY = os.environ.get("ERH_CV2_PY", "/data1/lyq/miniconda3/envs/EventHands/bin/python")
TPD_LUT_GLOB = ROOT / "outputs/tpd/20261105/erh"
FIXED_LABELS = Path("/data1/lyq/code/mesh/EvHandPose/data/evrealhands_fixed_labels/val_0_1_4_5_20_21_26_27_v2")
MOTION_MANIFEST_THRESHOLDS = {"quiet_activity_threshold": 1.4988161412853043, "local_ratio_1": 1.25, "local_ratio_2": 1.5,
                              "local_ratio_3": 2.0, "global_ratio_1": 0.8, "global_ratio_2": 2.0 / 3.0, "global_ratio_3": 0.5}
VAL8 = (0, 1, 4, 5, 20, 21, 26, 27)

SRC_W, SRC_H = 346, 260
GRID_W = 640                                   # the K grid of hand_data51 / tools.real (model: camera_K * W / 640)
FPS = 15.0
NAMES = {0: "cj", 1: "hqd", 2: "jjp", 3: "ljh", 4: "qwt", 5: "wbx", 6: "wyg", 7: "zbw", 8: "zyh", 9: "zyr"}
TRAIN_SUBJECTS, DEV_SUBJECTS, TEST_SUBJECTS = (0, 2, 4, 5, 6, 8, 9), (7,), (1, 3)
EXCLUDED = {52: "fast", 53: "fast", 74: "fast", 75: "fast"}
GAP = {"train": 0, "val": 4, "test": 4}        # missing labels bridged inside a run, per role
MIN_RUN_MS = 300
TAIL_US = 3000                                  # labels this close to the end of the stream are dropped
PALM_IDX = np.array([0, 1, 5, 9, 13, 17], np.int64)
FINGER_IDX = np.array([i for i in range(21) if i not in set(PALM_IDX.tolist())], np.int64)


# ------------------------------------------------------------------------------------------------ small helpers
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        v = float(x)
        return v if np.isfinite(v) else None
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(jsonable(obj), indent=1) + "\n")
    os.replace(tmp, path)


def root_of(canvas):
    return DATA / CANVASES[canvas][2]


def role_of(subject_id):
    if subject_id in TRAIN_SUBJECTS:
        return "train"
    if subject_id in DEV_SUBJECTS:
        return "dev"
    if subject_id in TEST_SUBJECTS:
        return "test"
    raise ValueError(subject_id)


def legacy_dir_of(role):
    return "train" if role == "train" else "val"          # splits_real.json keeps dev and test in val/


def seq_name(subject_id, trial, hand, scene, gesture):
    return f"{NAMES[subject_id]}_t{trial:02d}_{'R' if hand == 'right' else 'Lm'}_{scene}_{gesture}"


def read_annot(trial):
    with open(RAW / str(trial) / "annot.json") as f:
        return json.load(f)


def camera_of(annot):
    ev = annot["camera_info"]["event"]
    cam = {k: np.array(ev[k], np.float64) for k in ("R", "K", "K_new", "K_old", "dist")}
    cam["T"] = np.array(ev["T"], np.float64) / 1000.0          # metres
    assert list(ev["image_shape"]) == [SRC_H, SRC_W], ev["image_shape"]
    return cam


def canvas_K(K, canvas):
    """aux camera_K of a canvas: the model multiplies it by W / 640 and must get the canvas intrinsics."""
    W, H, _ = CANVASES[canvas]
    sx, sy, s = W / SRC_W, H / SRC_H, W / GRID_W
    out = np.eye(3)
    out[0, 0], out[0, 2] = K[0, 0] * sx / s, K[0, 2] * sx / s
    out[1, 1], out[1, 2] = K[1, 1] * sy / s, K[1, 2] * sy / s
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------------ undistortion table
_CV2_SNIPPET = """
import sys, numpy as np, cv2
z = np.load(sys.argv[1])
out = cv2.undistortPoints(src=z['xy'].astype(np.float32), cameraMatrix=z['K_old'], distCoeffs=z['dist'], P=z['K_new'])
np.save(sys.argv[2], out.reshape(-1, 2).astype(np.float32))
"""


def build_lut(cam, path):
    """(260, 346, 2) float32: the undistorted position of every sensor pixel, `cv2.undistortPoints(K_old, dist, P=K_new)`
    (EvHandPose `undistortion_points`); the training environment has no cv2, so it runs once in another environment."""
    path = Path(path)
    if path.exists():
        return np.load(path)
    ys, xs = np.mgrid[0:SRC_H, 0:SRC_W]
    xy = np.stack([xs.ravel(), ys.ravel()], 1).astype(np.float32)
    with tempfile.TemporaryDirectory(dir=str(path.parent), prefix=".cv2_") as tmp:
        fin, fout = os.path.join(tmp, "in.npz"), os.path.join(tmp, "out.npy")
        np.savez(fin, xy=xy, K_old=cam["K_old"], dist=cam["dist"], K_new=cam["K_new"])
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"}
        subprocess.run([CV2_PY, "-I", "-c", _CV2_SNIPPET, fin, fout], check=True, env=env)
        lut = np.load(fout).reshape(SRC_H, SRC_W, 2)
    np.save(path, lut)
    return lut


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


# ------------------------------------------------------------------------------------------------ MANO conventions
_MANO_NP = {}


def mano_arrays():
    if "d" not in _MANO_NP:
        d = np.load(MANO, allow_pickle=False)
        _MANO_NP["d"] = (d["v_template"].astype(np.float64), d["shapedirs"].astype(np.float64), d["J_regressor"].astype(np.float64))
    return _MANO_NP["d"]


def root_joint_of(betas):
    """J0 of the shaped template (the LBS pivot), as semkine.dataset.mano_root_joint."""
    v0, sd, jr = mano_arrays()
    v = v0 + np.einsum("l,mkl->mk", np.asarray(betas, np.float64), sd)
    return (jr @ v)[0]


def mano_world_to_camera(rot, trans, root_joint, R, T, target_root):
    """EvHandPose `get_annotations`: world-frame (rot, trans, root_joint) -> event camera frame; `target_root` is the root
    joint of the shape the converted parameters are used with, so the wrist stays where the annotation has it."""
    from scipy.spatial.transform import Rotation as Rot
    R_new = R[None] @ Rot.from_rotvec(np.asarray(rot, np.float64)).as_matrix()
    rot_new = Rot.from_matrix(R_new).as_rotvec()
    trans_new = (np.asarray(trans, np.float64) + root_joint) @ R.T + T - target_root
    return rot_new, trans_new


def indices_change(src=2, dst=1):
    """EvHandPose core/dataset/joint_indices.py (copied): joint order of 'our real dataset' (2) -> smplx (1)."""
    skeleton_ids = {'root': [20, 0, 0], 'thumb1': [3, 13, 1], 'thumb2': [2, 14, 2], 'thumb3': [1, 15, 3], 'thumb4': [0, 16, 4],
                    'index1': [7, 1, 5], 'index2': [6, 2, 6], 'index3': [5, 3, 7], 'index4': [4, 17, 8],
                    'middle1': [11, 4, 9], 'middle2': [10, 5, 10], 'middle3': [9, 6, 11], 'middle4': [8, 18, 12],
                    'ring1': [15, 10, 13], 'ring2': [14, 11, 14], 'ring3': [13, 12, 15], 'ring4': [12, 19, 16],
                    'pinky1': [19, 7, 17], 'pinky2': [18, 8, 18], 'pinky3': [17, 9, 19], 'pinky4': [16, 20, 20]}
    indices = np.array(list(skeleton_ids.values()))
    raw = list(range(21))
    src_id = indices[:, src].tolist()
    dst_id = indices[:, dst].tolist()
    tmp = [dst_id.index(i) for i in raw]
    return [src_id[i] for i in tmp]


# ------------------------------------------------------------------------------------------------ frame-level motion tiers
def rms_2d(d):
    d = np.asarray(d, np.float64)
    return float(np.sqrt(np.mean(np.sum(d * d, axis=-1))))


def motion_scores(prev2d, curr2d):
    """scripts/eval_motion_tier_mpjpe.py `compute_motion_scores` (copied)."""
    glob = rms_2d(curr2d[PALM_IDX] - prev2d[PALM_IDX])
    sp, tp = prev2d[PALM_IDX].mean(0), curr2d[PALM_IDX].mean(0)
    cov = (prev2d[PALM_IDX] - sp).T @ (curr2d[PALM_IDX] - tp)
    u, _, vt = np.linalg.svd(cov)
    rot = vt.T @ u.T
    if np.linalg.det(rot) < 0:
        vt[-1, :] *= -1.0
        rot = vt.T @ u.T
    aligned = (prev2d[FINGER_IDX] - sp) @ rot.T + tp
    loc = rms_2d(curr2d[FINGER_IDX] - aligned)
    return glob, loc


def motion_tiers(annot, cam, left, official_ids, ids):
    """Per label in `ids`: EvHandPose's val8 tier definition, thresholds fixed to the persisted val8 ones. The previous label is
    id - 1 (it must be in the official id set manos & 3d_joints); the joints are the annotation 3-D joints in the event
    frame in smplx order, x mirrored for left hands, projected with the annotation K."""
    perm = indices_change(2, 1)
    cache = {}

    def j2d(i):
        if i not in cache:
            j = np.asarray(annot["3d_joints"][str(i)], np.float64).reshape(21, 3)[perm] / 1000.0
            j = j @ cam["R"].T + cam["T"]
            if left:
                j[:, 0] *= -1.0
            p = j @ cam["K"].T
            cache[i] = p[:, :2] / p[:, 2:3]
        return cache[i]

    n = len(ids)
    scoreable = np.zeros(n, bool)
    gs, ls, ratio = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for k, i in enumerate(ids):
        if (i - 1) in official_ids and i in official_ids:
            g, l = motion_scores(j2d(i - 1), j2d(i))
            scoreable[k] = True
            gs[k], ls[k], ratio[k] = g, l, l / max(g, 1e-6)
    T = MOTION_MANIFEST_THRESHOLDS
    act = np.maximum(gs, ls)
    with np.errstate(invalid="ignore"):
        active = scoreable & (act > T["quiet_activity_threshold"])
        masks = {"local_1": active & (ratio >= T["local_ratio_1"]), "local_2": active & (ratio > T["local_ratio_2"]),
                 "local_3": active & (ratio > T["local_ratio_3"]), "global_1": active & (ratio <= T["global_ratio_1"]),
                 "global_2": active & (ratio < T["global_ratio_2"]), "global_3": active & (ratio < T["global_ratio_3"])}
        group = np.zeros(n, np.int8)                     # 0 unscoreable, 1 quiet, 2 mixed, 3 local, 4 global
        group[scoreable] = 1
        group[active] = 2
        group[active & (ratio >= T["local_ratio_1"])] = 3
        group[active & (ratio <= T["global_ratio_1"])] = 4
    return {"scoreable": scoreable, "global_score": gs, "local_score": ls, "ratio": ratio, "motion_group": group, **masks}


# ------------------------------------------------------------------------------------------------ labels -> runs -> per-ms
def label_groups(ids, gap):
    """Index ranges (lo, hi) of ids with at most `gap` missing frame ids between consecutive labels."""
    ids = np.asarray(ids)
    if len(ids) == 0:
        return []
    cut = np.flatnonzero(np.diff(ids) - 1 > gap)
    lo = np.concatenate([[0], cut + 1])
    hi = np.concatenate([cut, [len(ids) - 1]])
    return [(int(a), int(b)) for a, b in zip(lo, hi)]


def interp_group(ta, tr, go, hp, q_ms):
    """tools.prepare_hand_data.interpolate_pose_ms on one run (PCHIP translation, RotationSpline root + 15 joints), evaluated
    at the given ms with the query time clamped to the labelled span (a ms at most 0.5 ms outside it)."""
    from scipy.interpolate import PchipInterpolator
    from scipy.spatial.transform import Rotation, RotationSpline
    q = np.clip(q_ms.astype(np.float64) / 1000.0, ta[0], ta[-1])
    out = np.zeros((len(q), 51), np.float64)
    for d in range(3):
        out[:, d] = PchipInterpolator(ta, tr[:, d])(q)
    out[:, 3:6] = RotationSpline(ta, Rotation.from_rotvec(go))(q).as_rotvec()
    for j in range(15):
        out[:, 6 + 3 * j:9 + 3 * j] = RotationSpline(ta, Rotation.from_rotvec(hp[:, 3 * j:3 * j + 3]))(q).as_rotvec()
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------------ one trial
def convert_trial(trial, force=False):
    t_start = time.time()
    annot = read_annot(trial)
    hand, subject_id, motion = annot["hand_type"], int(annot["subject_id"]), annot["motion_type"]
    if trial in EXCLUDED or motion == "fast":
        return {"trial": trial, "excluded": True, "reason": EXCLUDED.get(trial, "fast"), "hand": hand, "subject": NAMES[subject_id]}
    left = hand == "left"
    role = role_of(subject_id)
    split_dir = legacy_dir_of(role)
    scene, gesture = annot["scene"], annot["gesture_type"]
    assert scene in ("normal", "highlight", "flash") and gesture in ("fixed", "random")
    seq = seq_name(subject_id, trial, hand, scene, gesture)
    outs = {c: root_of(c) / split_dir for c in CANVASES}
    if not force and all((d / f"{seq}.done.json").exists() for d in outs.values()):
        return json.loads((outs["n346"] / f"{seq}.done.json").read_text())
    for d in outs.values():
        d.mkdir(parents=True, exist_ok=True)

    cam = camera_of(annot)
    ids3 = set(int(k) for k in annot["3d_joints"])
    key = "manos_flip" if left else "mano"
    ids_key = set(int(k) for k in annot[key])
    official_ids = set(int(k) for k in annot["manos_flip" if left else "manos"]) & ids3
    ids_all = sorted(official_ids & ids_key)
    delta = float(annot["delta_time"])

    # ---- events: stable sort (the aedat4 timestamps go backwards in every file), clock zero = the first event
    t, x, y, p = read_events(RAW / str(trial) / "event.aedat4")
    n_raw = len(t)
    n_back = int((np.diff(t) < 0).sum())
    max_back = int(-np.diff(t).min()) if n_raw > 1 and n_back else 0
    if n_back:
        order = np.argsort(t, kind="stable")
        t, x, y, p = t[order], x[order], y[order], p[order]
    t_rel = t - t[0]
    del t
    rel_end_us = int(t_rel[-1])

    # ---- labels inside the stream
    ids = np.array([i for i in ids_all if 0 <= delta + i * 1e6 / FPS < rel_end_us - TAIL_US], np.int64)
    n_oob = len(ids_all) - len(ids)
    assert len(ids) > 10, (trial, len(ids))
    t_lab_us = delta + ids * 1e6 / FPS
    t0_us = int(np.rint(t_lab_us[0]))
    lab_ms = np.rint((t_lab_us - t0_us) / 1000.0).astype(np.int64)
    arr = {k: np.array([annot[key][str(i)][k] for i in ids], np.float64) for k in ("rot", "hand_pose", "shape", "trans", "root_joint")}
    betas = np.median(arr["shape"], 0)
    betas_std = float(arr["shape"].std(0).max())
    j0 = root_joint_of(betas)
    rot_n, tr_n = mano_world_to_camera(arr["rot"], arr["trans"], arr["root_joint"], cam["R"], cam["T"], j0)
    params_lab = np.concatenate([tr_n, rot_n, arr["hand_pose"]], 1)                      # [t, R_root, residual]

    # ---- runs
    gap = GAP["train" if role == "train" else "val"]
    groups = label_groups(ids, gap)
    kept, dropped = [], []
    for lo, hi in groups:
        span = int(lab_ms[hi] - lab_ms[lo])
        (kept if span >= MIN_RUN_MS and hi > lo else dropped).append((lo, hi))
    assert kept, f"trial {trial}: no run >= {MIN_RUN_MS} ms"
    runs = np.array([[lab_ms[lo] + 1, lab_ms[hi] + 1] for lo, hi in kept], np.int64)
    n_ms = int(runs[:, 1].max()) + 1
    assert n_ms * 1000 + t0_us <= rel_end_us, (n_ms, rel_end_us)
    tsec = (t_lab_us - t0_us) / 1e6
    params51 = np.zeros((n_ms, 51), np.float32)
    valid_ms = np.zeros(n_ms, bool)
    self_check = None
    for r, (lo, hi) in enumerate(kept):
        a, b = int(runs[r, 0]), int(runs[r, 1])
        q = np.arange(a, b)
        params51[a:b] = interp_group(tsec[lo:hi + 1], params_lab[lo:hi + 1, 0:3], params_lab[lo:hi + 1, 3:6], params_lab[lo:hi + 1, 6:51], q)
        valid_ms[a:b] = True
        if r == 0:      # the reference tool on the same labels must agree on every ms both define
            from tools.prepare_hand_data import interpolate_pose_ms
            ref, vref = interpolate_pose_ms(tsec[lo:hi + 1], params_lab[lo:hi + 1, 0:3], params_lab[lo:hi + 1, 3:6],
                                            params_lab[lo:hi + 1, 6:51], np.ones(hi - lo + 1, bool), n_ms)
            sel = vref & valid_ms
            self_check = float(np.abs(ref[sel] - params51[sel]).max()) if sel.any() else None
            assert self_check is None or self_check < 1e-6, f"interpolation differs from interpolate_pose_ms: {self_check}"
    in_run = np.zeros(len(ids), bool)
    for lo, hi in kept:
        in_run[lo:hi + 1] = True

    rows = np.arange(int(ids.max()) + 1)
    t_rows = delta + rows * 1e6 / FPS
    gt_ms_idx = np.clip(np.rint((t_rows - t0_us) / 1000.0).astype(np.int64), 0, n_ms - 1)
    gt_timestamp_ns = np.rint(t_rows * 1000.0).astype(np.int64)
    gt_valid = np.zeros(len(rows), bool)
    gt_valid[ids[in_run]] = True
    label_screened = np.zeros(len(rows), bool)
    label_screened[ids] = True

    # ---- annotation 3-D joints in the event frame (our joint order = the annotation order), mirrored for left hands
    j3 = np.array([annot["3d_joints"][str(i)] for i in ids], np.float64).reshape(-1, 21, 3) / 1000.0
    j3 = j3 @ cam["R"].T + cam["T"]
    if left:
        j3[..., 0] *= -1.0
    motion_arrays = motion_tiers(annot, cam, left, official_ids, [int(i) for i in ids])

    # ---- events -> both canvases
    lo_e = int(np.searchsorted(t_rel, t0_us, "left"))
    hi_e = int(np.searchsorted(t_rel, t0_us + n_ms * 1000, "left"))
    rel = t_rel[lo_e:hi_e] - t0_us
    xs_, ys_, ps_ = x[lo_e:hi_e], y[lo_e:hi_e], p[lo_e:hi_e]
    del t_rel, x, y, p
    lut = np.load(root_of("n346") / "undistort_lut.npy").reshape(-1, 2)
    inside = (xs_ >= 0) & (xs_ < SRC_W) & (ys_ >= 0) & (ys_ < SRC_H)
    pix = np.where(inside, ys_.astype(np.int64) * SRC_W + xs_.astype(np.int64), 0)
    u = lut[pix].astype(np.float64)
    ux, uy = u[:, 0], u[:, 1]
    del u, pix
    if left:
        ux = 2.0 * cam["K"][0, 2] - ux                          # mirror about the principal point of K, in the undistorted frame
    legal = inside & (ux >= 0) & (ux < SRC_W) & (uy >= 0) & (uy < SRC_H)
    k = np.flatnonzero(legal)
    rel_k = rel[k]
    ms_k = rel_k // 1000
    tsub = (rel_k - ms_k * 1000).astype(np.uint16)
    counts = np.bincount(ms_k, minlength=n_ms)
    offsets = np.zeros(n_ms + 1, np.int64)
    offsets[1:] = np.cumsum(counts)
    assert len(offsets) == n_ms + 1
    pol = np.clip(ps_[k], 0, 1).astype(np.uint8)
    n_in_window, n_out = int(len(rel)), int(len(k))
    uxk, uyk = ux[k], uy[k]
    del ux, uy, rel, rel_k, ms_k, xs_, ys_, ps_

    ev_stats = {"n_raw": n_raw, "n_in_window": n_in_window, "n_kept": n_out, "n_dropped_outside_canvas": n_in_window - n_out,
                "n_unsorted_diffs": n_back, "max_backward_us": max_back, "stream_duration_s": rel_end_us / 1e6,
                "on_fraction": float(pol.mean()) if n_out else None}
    per_canvas = {}
    for canvas, (W, H, _) in CANVASES.items():
        d = outs[canvas]
        dt = np.uint8 if W <= 256 else np.uint16
        sx, sy = W / SRC_W, H / SRC_H
        ev = np.empty((n_out, 3), dt)
        ev[:, 0] = np.minimum(np.floor(uxk * sx), W - 1)
        ev[:, 1] = np.minimum(np.floor(uyk * sy), H - 1)
        ev[:, 2] = pol
        np.save(d / f"{seq}_events.npy", ev)
        np.save(d / f"{seq}_offsets.npy", offsets)
        np.save(d / f"{seq}_tsub.npy", tsub)
        del ev
        from tools.prepare_hand_data import write_meta
        write_meta(d / f"{seq}.meta", params51)
        aux = {"betas": betas.astype(np.float32), "camera_K": canvas_K(cam["K"], canvas), "valid_ms": valid_ms, "gt_ms_idx": gt_ms_idx,
               "gt_valid": gt_valid, "gt_timestamp_ns": gt_timestamp_ns, "t0_ns": np.int64(t0_us * 1000), "n_ms": np.int64(n_ms),
               "valid_runs_ms": runs.reshape(-1, 2), "category": np.array(scene), "split": np.array(split_dir),
               "canonical_hand": np.array("right"),
               "params_layout": np.array(["t[0:3]", "R_root[3:6]", "hand_pose_residual_aa[6:51]"]),
               "light": np.array(scene), "subject": np.array(NAMES[subject_id]), "copy": np.array("rec"),
               "source_category": np.array(f"erh_{trial}"), "label_shift_ns": np.int64(0),
               # ERH extras (the reader ignores them)
               "hand": np.array("left_mirrored" if left else "right"), "scene": np.array(scene), "gesture": np.array(gesture),
               "label_tier": np.array("checked" if annot["annoted"] else "unchecked"), "trial": np.int64(trial),
               "subject_id": np.int64(subject_id), "role": np.array(role), "label_screened": label_screened,
               "motion_labels_file": np.array(f"{seq}_motion.npz"), "label_fps": np.float64(FPS),
               "run_gap_bridged_labels": np.int64(gap)}
        np.savez_compressed(d / f"{seq}_aux.npz", **aux)
        np.savez_compressed(d / f"{seq}_labels.npz", frame_id=ids, label_ms=lab_ms, in_run=in_run, params51=params_lab.astype(np.float32),
                            joints3d_cam=j3.astype(np.float32), betas_frames=arr["shape"].astype(np.float32),
                            betas_trial=betas.astype(np.float32), tier=np.array("checked" if annot["annoted"] else "unchecked"),
                            hand=np.array("left_mirrored" if left else "right"), label_key=np.array(key))
        np.savez_compressed(d / f"{seq}_motion.npz", frame_id=ids, label_ms=lab_ms, in_run=in_run,
                            thresholds=np.array(json.dumps(MOTION_MANIFEST_THRESHOLDS)), **motion_arrays)
        per_canvas[canvas] = {"events_dtype": str(np.dtype(dt)), "canvas": [W, H]}

    mg = motion_arrays["motion_group"]
    summary = {"seq": seq, "trial": trial, "subject": NAMES[subject_id], "subject_id": subject_id, "role": role, "legacy_dir": split_dir,
               "hand": "left_mirrored" if left else "right", "scene": scene, "gesture": gesture,
               "label_tier": "checked" if annot["annoted"] else "unchecked", "label_key": key,
               "t0_us": t0_us, "n_ms": n_ms, "duration_s": n_ms / 1000.0,
               "n_labels_official_ids": len(official_ids), "n_labels_screened": len(ids_all), "n_labels_dropped_outside_stream": n_oob,
               "n_labels_in_runs": int(in_run.sum()), "n_labels_dropped_short_run": int(sum(hi - lo + 1 for lo, hi in dropped)),
               "n_groups": len(groups), "n_runs": len(kept), "runs_dropped_short": len(dropped),
               "run_ms": [int(b - a) for a, b in runs], "gap_bridged_labels": gap,
               "n_valid_ms": int(valid_ms.sum()), "betas": betas.tolist(), "betas_std_max": betas_std,
               "interp_selfcheck_max_abs_diff": self_check, "events": ev_stats, "canvases": per_canvas,
               "motion_counts": {"unscoreable": int((mg == 0).sum()), "quiet": int((mg == 1).sum()), "mixed": int((mg == 2).sum()),
                                 "local": int((mg == 3).sum()), "global": int((mg == 4).sum())},
               "camera_K": {c: canvas_K(cam["K"], c).tolist() for c in CANVASES},
               "raw_event_file": str(RAW / str(trial) / "event.aedat4"), "tool_sha256": sha(__file__),
               "seconds": round(time.time() - t_start, 1), "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for canvas in CANVASES:
        write_json(outs[canvas] / f"{seq}.done.json", {**summary, "canvas": canvas})
    print(f"[{split_dir}/{seq}] ms={n_ms} events {n_raw} -> {n_out} labels {len(ids)} runs={len(kept)} {summary['seconds']}s", flush=True)
    return summary


def _work(trial_force):
    trial, force = trial_force
    try:
        return convert_trial(trial, force)
    except Exception as e:                                           # noqa: BLE001
        import traceback
        return {"trial": trial, "error": repr(e), "trace": traceback.format_exc()}


# ------------------------------------------------------------------------------------------------ run / finalize
def camera_block(cam):
    return {"K_annotation": cam["K"].tolist(), "K_new": cam["K_new"].tolist(), "K_old": cam["K_old"].tolist(),
            "dist": cam["dist"].tolist(), "R_event": cam["R"].tolist(), "T_event_m": cam["T"].tolist(), "src_size": [SRC_W, SRC_H],
            "mirror_cx": float(cam["K"][0, 2]),
            "undistortion": "cv2.undistortPoints(K_old, dist, P=K_new) per sensor pixel (undistort_lut.npy), floor binning, "
                            "legal = inside the canvas; left hands: x' = 2 K.cx - x_u",
            "aux_camera_K": {c: canvas_K(cam["K"], c).astype(np.float64).tolist() for c in CANVASES},
            "render_scale": {c: CANVASES[c][0] / GRID_W for c in CANVASES}}


def collect(canvas):
    root = root_of(canvas)
    recs = []
    for d in ("train", "val"):
        for f in sorted((root / d).glob("*.done.json")):
            recs.append(json.loads(f.read_text()))
    return recs


def split_blocks(recs):
    def block(sel):
        sel = sorted(sel, key=lambda r: (r["subject"], r["trial"]))
        return {"trials": [r["seq"] for r in sel], "subjects": sorted({r["subject"] for r in sel}),
                "legacy_dir": {r["seq"]: r["legacy_dir"] for r in sel}}
    return {"train": block([r for r in recs if r["role"] == "train"]), "val_core": block([r for r in recs if r["role"] == "dev"]),
            "val": block([r for r in recs if r["role"] == "dev"]), "test": block([r for r in recs if r["role"] == "test"])}


def split_stats(recs):
    out = {}
    for role in ("train", "dev", "test"):
        sel = [r for r in recs if r["role"] == role]
        out[role] = {"sequences": len(sel), "runs": sum(r["n_runs"] for r in sel), "labels_in_runs": sum(r["n_labels_in_runs"] for r in sel),
                     "labels_screened": sum(r["n_labels_screened"] for r in sel), "events": sum(r["events"]["n_kept"] for r in sel),
                     "duration_s": sum(r["duration_s"] for r in sel), "valid_run_s": sum(sum(r["run_ms"]) for r in sel) / 1000.0,
                     "by_hand": {h: sum(1 for r in sel if r["hand"] == h) for h in ("right", "left_mirrored")},
                     "by_tier": {t: sum(1 for r in sel if r["label_tier"] == t) for t in ("checked", "unchecked")},
                     "by_scene": {s: sum(1 for r in sel if r["scene"] == s) for s in ("normal", "highlight", "flash")}}
    return out


def finalize(errors=None, canvases=None):
    cam = camera_of(read_annot(20))
    ids_all = json.loads((RAW / "id_seq_all.json").read_text())
    excluded = []
    for t, why in sorted(EXCLUDED.items()):
        a = read_annot(t)
        excluded.append({"trial": t, "reason": why + " motion (2-D joints only, no MANO)", "hand": a["hand_type"], "subject": NAMES[int(a["subject_id"])]})
    for canvas in canvases or CANVASES:
        root = root_of(canvas)
        recs = collect(canvas)
        sp = {"_protocol": "EvHandPose configs/pretrain.yml split of EvRealHands: train subjects 0 2 4 5 6 8 9, dev (val = val_core) subject 7, "
                           "test subjects 1 3; both hands (left mirrored), checked + unchecked trials, 2-D-only fast trials excluded; "
                           "dev and test sequences live in val/ (splits_real.json convention)",
              **split_blocks(recs)}
        write_json(root / "splits_erh.json", sp)
        W, H, _ = CANVASES[canvas]
        conv_path = root / "conversion.json"
        prior = json.loads(conv_path.read_text()) if conv_path.exists() else {}
        conv = {"tool": "tools/erh/convert.py", "tool_sha256": sha(__file__), "source": str(RAW), "canvas": canvas, "canvas_size": [W, H],
                "source_id_seq_all_sha256": sha(RAW / "id_seq_all.json"),
                "camera": camera_block(cam),
                "split": {"train_subjects": {NAMES[s]: s for s in TRAIN_SUBJECTS}, "dev_subjects": {NAMES[s]: s for s in DEV_SUBJECTS},
                          "test_subjects": {NAMES[s]: s for s in TEST_SUBJECTS}, "run_gap_bridged_labels": GAP, "min_run_ms": MIN_RUN_MS},
                "rules": {"labels": "right: manos & mano & 3d_joints, values from mano; left: manos_flip & 3d_joints; betas = trial median",
                          "clock": "no shift; label us = delta_time + id * 1e6 / 15 on the clock of the first sorted event; t0 = first screened label",
                          "runs": "[first label ms + 1, last label ms + 1); gap bridging train 0 / dev, test 4 labels; < 300 ms dropped",
                          "motion": "no sequence-level local / global in the dataset; frame-level EvHandPose val8 tiers recomputed per label "
                                    "(<seq>_motion.npz)"},
                "excluded_trials": excluded, "n_sequences": len(recs), "split_stats": split_stats(recs), "errors": errors or [],
                "sequences": sorted(recs, key=lambda r: r["seq"]),
                "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        for k in ("validation", "dataset_smoke", "lut_vs_tpd_lut"):
            if k in prior:
                conv[k] = prior[k]
        write_json(conv_path, conv)
        print(f"[finalize {canvas}] {len(recs)} sequences, splits_erh.json + conversion.json written", flush=True)


def cmd_run(args):
    cam = camera_of(read_annot(20))
    for c in CANVASES:
        root_of(c).mkdir(parents=True, exist_ok=True)
    lut = build_lut(cam, root_of("n346") / "undistort_lut.npy")
    np.save(root_of("n256") / "undistort_lut.npy", lut)
    tpd = sorted(TPD_LUT_GLOB.glob("undistort_lut_*.npy"))
    lut_check = None
    if tpd:
        other = np.load(tpd[0])
        lut_check = {"file": str(tpd[0]), "same_shape": list(other.shape) == list(lut.shape), "max_abs_diff_px": float(np.abs(other - lut).max())}
        print("LUT vs the TPD conversion's LUT:", lut_check, flush=True)
    trials = [t for t in range(84) if (not args.only or t in args.only)]
    trials.sort(key=lambda t: -(RAW / str(t) / "event.aedat4").stat().st_size)         # biggest first
    results, errors = [], []
    env_threads = {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    os.environ.update(env_threads)
    with ProcessPoolExecutor(min(args.workers, len(trials))) as ex:
        futs = {ex.submit(_work, (t, args.force)): t for t in trials}
        for f in as_completed(futs):
            r = f.result()
            if "error" in r:
                errors.append({"trial": r["trial"], "error": r["error"]})
                print("ERROR", r["trial"], r["error"], "\n", r.get("trace", ""), flush=True)
            else:
                results.append(r)
    if args.only:
        print(f"done (partial: --only), {len(errors)} errors")
        return 1 if errors else 0
    finalize(errors)
    for c in CANVASES:
        p = root_of(c) / "conversion.json"
        d = json.loads(p.read_text())
        d["lut_vs_tpd_lut"] = lut_check
        write_json(p, d)
    print("DONE", len(results), "results,", len(errors), "errors", flush=True)
    return 1 if errors else 0


# ------------------------------------------------------------------------------------------------ validation
def endpoint_grid(aux):
    """tools/xcl/evaluate.py `endpoint_grid` (same code): (run id, end ms, previous end, first) of the 50 ms packet grid."""
    rows = []
    for rid, (a, b) in enumerate(np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)):
        previous = int(a)
        for end in np.arange(int(a) + 49, int(b), 50, dtype=np.int64):
            rows.append((rid, int(end), previous, previous == int(a)))
            previous = int(end)
    return rows


_V = {}


def _vinit():
    import torch
    torch.set_num_threads(1)
    from mano_layer import load_mano
    from pose_repr import decode_to_mano_inputs
    m = load_mano().eval()
    _V["m"], _V["dec"], _V["torch"] = m, decode_to_mano_inputs, torch


def _fk(params51, betas):
    torch, m = _V["torch"], _V["m"]
    p = torch.from_numpy(np.asarray(params51, np.float32))
    with torch.no_grad():
        dec = _V["dec"](p, "mano_full_axis_angle", m.hands_components, m.hands_mean)
        b = torch.from_numpy(np.broadcast_to(np.asarray(betas, np.float32).reshape(1, -1), (len(p), 10)).copy())
        v, j = m(b, dec["global_orient"], dec["local_full_aa"], dec["transl"])
    return v.numpy().astype(np.float64), j.numpy().astype(np.float64)


def silhouette(uv, faces, W, H, dilate=2):
    """Filled, dilated hand silhouette on the canvas: vertices and a barycentric lattice of every face binned with floor,
    closed (3 x 3), holes filled, dilated by `dilate` px (the discretisation the events get)."""
    from scipy import ndimage
    lat = np.array([(i, j, 4 - i - j) for i in range(5) for j in range(5 - i)], np.float64) / 4.0
    dense = np.einsum("pk,fkd->fpd", lat, uv[faces]).reshape(-1, 2)
    ix, iy = np.floor(dense[:, 0]).astype(np.int64), np.floor(dense[:, 1]).astype(np.int64)
    ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    m = np.zeros((H, W), bool)
    m[iy[ok], ix[ok]] = True
    m = ndimage.binary_closing(m, structure=np.ones((3, 3), bool))
    m = ndimage.binary_fill_holes(m)
    return ndimage.binary_dilation(m, structure=np.ones((2 * dilate + 1, 2 * dilate + 1), bool))


def read_meta(path):
    import struct
    with open(path, "rb") as f:
        (n,) = struct.unpack("<i", f.read(4))
        raw = np.fromfile(f, dtype=np.dtype([("data", np.float64, n), ("m0", np.uint8), ("m1", np.uint8)]))
    assert n == 51 and np.all(raw["m0"] == 4) and np.all(raw["m1"] == 13)
    return raw["data"].astype(np.float32)


def validate_seq(args):
    canvas, seq, split_dir, n_labels = args
    W, H, _ = CANVASES[canvas]
    d = root_of(canvas) / split_dir
    aux = np.load(d / f"{seq}_aux.npz", allow_pickle=True)
    ev = np.load(d / f"{seq}_events.npy", mmap_mode="r")
    off = np.load(d / f"{seq}_offsets.npy")
    tsub = np.load(d / f"{seq}_tsub.npy", mmap_mode="r")
    meta = read_meta(d / f"{seq}.meta")
    n_ms = int(aux["n_ms"])
    checks = {"offsets_len_is_n_ms_plus_1": len(off) == n_ms + 1, "meta_rows_is_n_ms": len(meta) == n_ms, "valid_ms_len": len(aux["valid_ms"]) == n_ms,
              "events_len_matches_offsets": len(ev) == int(off[-1]) and len(tsub) == len(ev),
              "events_in_canvas": bool(len(ev) == 0 or (int(ev[:, 0].max()) < W and int(ev[:, 1].max()) < H)),
              "tsub_lt_1000": bool(len(tsub) == 0 or int(tsub.max()) < 1000), "events_dtype": str(ev.dtype),
              "offsets_monotone": bool(np.all(np.diff(off) >= 0))}
    runs = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
    vm = np.zeros(n_ms, bool)
    for a, b in runs:
        vm[a:b] = True
    checks["valid_ms_equals_runs"] = bool(np.array_equal(vm, np.asarray(aux["valid_ms"], bool)))
    checks["runs_ge_300ms"] = bool(np.all(runs[:, 1] - runs[:, 0] >= MIN_RUN_MS))
    # ---- run alignment: the 50 ms endpoint grid vs the real label ms
    grid = endpoint_grid(aux)
    gt_ms = np.asarray(aux["gt_ms_idx"], np.int64)
    gt_valid = np.asarray(aux["gt_valid"], bool)
    label_ms_set = set(gt_ms[gt_valid].tolist())
    ends = np.array([g[1] for g in grid], np.int64)
    at_label = np.array([e in label_ms_set for e in ends], bool)
    expected = 0
    other = 0
    for rid, (a, b) in enumerate(runs):
        ks = np.arange(a + 49, b, 50)
        for k, e in enumerate(ks):
            mod4 = (k + 1) % 4 == 0
            has = int(e) in label_ms_set
            expected += int(mod4 and has)
            other += int((not mod4) and has)
    align = {"n_endpoints": int(len(ends)), "n_endpoints_at_label_ms": int(at_label.sum()), "n_expected_every_4th_with_label": int(expected),
             "n_endpoints_at_label_ms_off_the_4th_grid": int(other), "n_labels_in_runs": int(gt_valid.sum()),
             "all_runs_start_at_label_ms_plus_1": bool(all(int(a) - 1 in label_ms_set for a, _ in runs)),
             "all_runs_end_at_label_ms_plus_1": bool(all(int(b) - 1 in label_ms_set for _, b in runs))}
    # ---- inside-silhouette fraction of the events at label times (the repo's projection, canvas intrinsics = camera_K * W / 640)
    betas = np.asarray(aux["betas"], np.float32)
    K = np.asarray(aux["camera_K"], np.float64)
    s = W / GRID_W
    fx, fy, cx, cy = K[0, 0] * s, K[1, 1] * s, K[0, 2] * s, K[1, 2] * s
    rows = np.flatnonzero(gt_valid)
    rows = rows[(gt_ms[rows] >= 10) & (gt_ms[rows] + 11 <= n_ms)]
    pick = rows[np.unique(np.linspace(0, len(rows) - 1, min(n_labels, len(rows))).astype(int))] if len(rows) else rows
    if len(pick) and "m" not in _V:
        _vinit()
    faces = _V["m"].f.numpy() if len(pick) else None
    ins = nul = tot = 0
    per_label = []
    for r in pick:
        m0 = int(gt_ms[r])
        v, _ = _fk(meta[m0:m0 + 1], betas)
        v = v[0]
        uv = np.stack([fx * v[:, 0] / v[:, 2] + cx, fy * v[:, 1] / v[:, 2] + cy], 1)
        mask = silhouette(uv, faces, W, H)
        e = np.asarray(ev[int(off[m0 - 10]):int(off[m0 + 11])])
        if len(e) == 0:
            continue
        ex, ey = e[:, 0].astype(np.int64), e[:, 1].astype(np.int64)
        i1 = int(mask[ey, ex].sum())
        exn = ex - 8
        ok = exn >= 0
        i0 = int(mask[ey[ok], exn[ok]].sum())
        ins += i1
        nul += i0
        tot += len(e)
        per_label.append(i1 / len(e))
    return {"seq": seq, "canvas": canvas, "checks": checks, "alignment": align,
            "silhouette": {"n_labels": len(per_label), "events": tot, "inside": ins, "null_inside": nul,
                           "frac": ins / tot if tot else None, "null_frac": nul / tot if tot else None,
                           "median_label_frac": float(np.median(per_label)) if per_label else None}}


def check_motion_against_official(canvas="n346"):
    """The recomputed frame-level tiers must equal EvHandPose's frame_labels.npz on every val8 frame the official file scores."""
    z = np.load(FIXED_LABELS / "frame_labels.npz", allow_pickle=True)
    ids = json.loads((FIXED_LABELS / "label_index.json").read_text()) if (FIXED_LABELS / "label_index.json").exists() else None
    root = root_of(canvas)
    recs = {r["trial"]: r for r in collect(canvas)}
    res = {"frames_official": int(len(z["folder"])), "frames_official_scoreable": int(z["scoreable"].sum()), "per_trial": {}}
    names = ("local_1", "local_2", "local_3", "global_1", "global_2", "global_3")
    tot = {"compared": 0, "mismatch": 0, "score_max_abs_diff": 0.0}
    for t in VAL8:
        if t not in recs:
            continue
        r = recs[t]
        m = np.load(root / r["legacy_dir"] / f"{r['seq']}_motion.npz")
        by_id = {int(f): k for k, f in enumerate(m["frame_id"])}
        sel = np.flatnonzero((z["folder"] == t) & z["scoreable"])
        cmp_ = mis = miss = 0
        mx = 0.0
        for k in sel:
            fid = int(z["frame_id"][k])
            if fid not in by_id:
                miss += 1
                continue
            j = by_id[fid]
            ok = bool(m["scoreable"][j]) and all(bool(m[n][j]) == bool(z[n][k]) for n in names)
            if m["scoreable"][j]:
                mx = max(mx, abs(float(m["global_score"][j]) - float(z["global_score"][k])), abs(float(m["local_score"][j]) - float(z["local_score"][k])))
            cmp_ += 1
            mis += int(not ok)
        res["per_trial"][t] = {"official_scoreable_frames": int(len(sel)), "compared": cmp_, "mismatch": mis, "frame_not_in_our_labels": miss,
                               "score_max_abs_diff": mx}
        tot["frame_not_in_our_labels"] = tot.get("frame_not_in_our_labels", 0) + miss
        tot["compared"] += cmp_
        tot["mismatch"] += mis
        tot["score_max_abs_diff"] = max(tot["score_max_abs_diff"], mx)
    res["total"] = tot
    return res


def cmd_validate(args):
    jobs, meta = [], {}
    for canvas in CANVASES:
        for r in collect(canvas):
            jobs.append((canvas, r["seq"], r["legacy_dir"], args.labels))
            meta[(canvas, r["seq"])] = r
    out = []
    with ProcessPoolExecutor(args.workers) as ex:
        for f in as_completed([ex.submit(validate_seq, j) for j in jobs]):
            out.append(f.result())
    cats = {}
    align_tot = {}
    checks_fail = []
    for r in out:
        m = meta[(r["canvas"], r["seq"])]
        for k, v in r["checks"].items():
            if v is False:
                checks_fail.append((r["canvas"], r["seq"], k))
        for k in ("all_runs_start_at_label_ms_plus_1", "all_runs_end_at_label_ms_plus_1"):
            if not r["alignment"][k]:
                checks_fail.append((r["canvas"], r["seq"], k))
        for key in ((r["canvas"], m["hand"], m["scene"], m["label_tier"]), (r["canvas"], m["hand"], m["scene"], "all_tiers"),
                    (r["canvas"], "all", m["scene"], "all_tiers"), (r["canvas"], "all", "all", "all_tiers")):
            c = cats.setdefault(key, {"sequences": 0, "labels": 0, "events": 0, "inside": 0, "null_inside": 0, "seq_median_fracs": []})
            s = r["silhouette"]
            c["sequences"] += 1
            c["labels"] += s["n_labels"]
            c["events"] += s["events"]
            c["inside"] += s["inside"]
            c["null_inside"] += s["null_inside"]
            if s["median_label_frac"] is not None:
                c["seq_median_fracs"].append(s["median_label_frac"])
        a = align_tot.setdefault((r["canvas"], m["role"]), {"sequences": 0, "endpoints": 0, "at_label": 0, "expected_every_4th": 0, "off_grid_hits": 0, "runs": 0})
        a["sequences"] += 1
        a["endpoints"] += r["alignment"]["n_endpoints"]
        a["at_label"] += r["alignment"]["n_endpoints_at_label_ms"]
        a["expected_every_4th"] += r["alignment"]["n_expected_every_4th_with_label"]
        a["off_grid_hits"] += r["alignment"]["n_endpoints_at_label_ms_off_the_4th_grid"]
        a["runs"] += m["n_runs"]
    cat_rows = []
    for (canvas, hand, scene, tier), c in sorted(cats.items()):
        cat_rows.append({"canvas": canvas, "hand": hand, "scene": scene, "label_tier": tier, "sequences": c["sequences"], "labels": c["labels"],
                         "events": c["events"], "inside_frac": c["inside"] / max(c["events"], 1), "null_shift8_frac": c["null_inside"] / max(c["events"], 1),
                         "median_of_sequence_medians": float(np.median(c["seq_median_fracs"])) if c["seq_median_fracs"] else None})
    motion = check_motion_against_official()
    report = {"labels_per_sequence": args.labels, "categories": cat_rows,
              "run_alignment": [{"canvas": k[0], "role": k[1], **v} for k, v in sorted(align_tot.items())],
              "file_checks_failed": checks_fail, "n_sequence_canvases": len(out), "motion_vs_official_val8": motion,
              "per_sequence": sorted([{"canvas": r["canvas"], "seq": r["seq"], **r["silhouette"], **{"alignment": r["alignment"]}} for r in out],
                                     key=lambda r: (r["canvas"], r["seq"])),
              "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for canvas in CANVASES:
        root = root_of(canvas)
        write_json(root / "validation.json", {**report, "canvas": canvas})
        p = root / "conversion.json"
        d = json.loads(p.read_text())
        d["validation"] = {k: v for k, v in report.items() if k != "per_sequence"}
        d["validation"]["file"] = str(root / "validation.json")
        write_json(p, d)
    print(json.dumps({"file_checks_failed": checks_fail, "motion_vs_official_val8": motion["total"]}, indent=1))
    for r in report["run_alignment"]:
        print(r)
    for r in cat_rows:
        if r["canvas"] == "n346":
            print(r)
    return 1 if checks_fail else 0


# ------------------------------------------------------------------------------------------------ dataset smoke
def describe(it):
    if isinstance(it, dict):
        items = list(it.items())
    elif isinstance(it, (tuple, list)):
        items = list(enumerate(it))
    else:
        items = list(vars(it).items())
    return [(str(k), list(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in items][:20]


def cmd_smoke(args):
    import yaml
    from semkine.dataset import build_dataset, sequences_for_split, splits_manifest
    base = ROOT / "configs/real_20261110/cmp_eho_s3407.yaml"
    cfg = yaml.safe_load(base.read_text())
    comps = np.load(MANO)["hands_components"]
    report = {"base_config": str(base), "canvases": {}}
    for canvas in CANVASES:
        W, H, d = CANVASES[canvas]
        root = root_of(canvas)
        cfg["DATA"].update(ROOT=str(root), SPLITS_MANIFEST=str(root / "splits_erh.json"), WIDTH=W, HEIGHT=H, INPUT_MODE="raw_packed",
                           WINDOW_MIN=30, WINDOW_MAX=300)
        cfg["MODEL"].update(RENDER_W=W, RENDER_H=H, RENDER_SCALE=W / GRID_W)
        rep = {}
        for split in ("train", "val", "test"):
            ds = build_dataset(cfg, split, comps, input_mode="raw_packed")
            seqs = sequences_for_split(root, split, splits_manifest(cfg))
            n = len(ds)
            idx = sorted({0, n // 3, n // 2, n - 1} | set(int(i) for i in np.random.default_rng(3407).integers(0, n, 4)))
            samples = []
            for i in idx:
                it = ds[i]
                samples.append({"index": i, "type": type(it).__name__, "fields": describe(it)})
            rep[split] = {"sequences": len(seqs), "samples_in_index": n, "items_read": idx, "first_item": samples[0]}
            print(f"[smoke {canvas}/{split}] sequences {len(seqs)} index {n} items read {idx}", flush=True)
        report["canvases"][canvas] = rep
    out = Path(args.out) if args.out else None
    for canvas in CANVASES:
        p = root_of(canvas) / "conversion.json"
        d = json.loads(p.read_text())
        d["dataset_smoke"] = {**report["canvases"][canvas], "base_config": report["base_config"], "input_mode": "raw_packed"}
        write_json(p, d)
    if out:
        write_json(out, report)
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=100)
    r.add_argument("--only", type=int, nargs="*")
    r.add_argument("--force", action="store_true")
    sub.add_parser("finalize")
    v = sub.add_parser("validate")
    v.add_argument("--workers", type=int, default=48)
    v.add_argument("--labels", type=int, default=12, help="labels per sequence for the silhouette check")
    s = sub.add_parser("smoke")
    s.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.verb == "run":
        sys.exit(cmd_run(args))
    if args.verb == "finalize":
        finalize()
        return
    if args.verb == "validate":
        sys.exit(cmd_validate(args))
    sys.exit(cmd_smoke(args))


if __name__ == "__main__":
    main()
