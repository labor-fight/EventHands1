#!/usr/bin/env python3
"""Shared helpers of the [XYZ-ASSOC] P1 scripts (CPU only, read-only on the repository).

Protocol constants mirror semkine/eval_track.py (run/end enumeration, 50 ms window) and
model/model.py::_xyz_mesh_forward (render-frame intrinsics s*K with s = RENDER_SCALE = 0.375).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""          # never touch a GPU (all six are training)
import sys
sys.dont_write_bytecode = True
from pathlib import Path

import numpy as np
import torch

REPO = Path("/data1/lyq/code/mesh/EventHands1")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from mano_layer import ManoLayer                                   # noqa: E402
from pose_repr import decode_to_mano_inputs                        # noqa: E402
from semkine.dataset import sequences_for_split                    # noqa: E402
from semkine.eval_track import load_sequence, _window_events       # noqa: E402

H, W = 180, 240
RENDER_SCALE = 0.375
STEP_MS = WIN_MS = 50
CAP = 2048                                   # ENCODER_MAX_NODES
DATA_ROOT = REPO / "data" / "hand_data51"
OUT_DIR = REPO / "outputs" / "xa" / "step1"
TIPS21 = [4, 8, 12, 16, 20]                  # thumb, index, middle, ring, pinky in the 21-joint OpenPose order
BAND_PX, FRONT_PX, Z_TOL, KNN = 16.0, 3.0, 0.01, 16
GROUP_NAMES = ["palm", "index", "middle", "pinky", "ring", "thumb"]
# joint -> finger group (MANO 16-joint LBS order, checked against kintree_table)
JOINT_GROUP = np.array([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], dtype=np.int64)


def get_mano():
    return ManoLayer(REPO / "assets" / "mano_right.npz", add_mean=False).eval()


def sequences():
    """[(seq, legacy_dir)] of the evaluation split (the two zgz sequences), as the evaluator reads them."""
    return sequences_for_split(DATA_ROOT, "val", None)


def enumerate_steps(aux):
    """Evaluation steps exactly as eval_track.track_sequence enumerates them (step_ms = window = 50).

    Returns int64 arrays (run_id, end, prev_end) per step; the window is [end-49, end], the evaluator's
    state chain is pos51[a] -> pos51[end_1] -> ... so prev_end = a for the first step of a run and
    end-50 afterwards.
    """
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    rid, ends_, prevs = [], [], []
    for run_id, (a, b) in enumerate(runs):
        ends = np.arange(a + max(STEP_MS, WIN_MS) - 1, b, STEP_MS, dtype=np.int64)
        if not len(ends):
            continue
        prev_end = int(a)
        for e in ends:
            rid.append(run_id); ends_.append(int(e)); prevs.append(prev_end)
            prev_end = int(e)
    return np.asarray(rid, np.int64), np.asarray(ends_, np.int64), np.asarray(prevs, np.int64)


def fk_batch(mano, betas, params51):
    """`(n, 51)` -> verts `(n, 778, 3)`, joints21 `(n, 21, 3)` float32 torch (camera frame, metres);
    the same chain as model._fk."""
    with torch.no_grad():
        p = torch.from_numpy(np.ascontiguousarray(params51, dtype=np.float32))
        dec = decode_to_mano_inputs(p, "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
        return mano(betas.expand(len(p), -1), dec["global_orient"], dec["local_full_aa"], dec["transl"])


def render_intrinsics_f32(camera_K):
    """fx, fy, cx, cy of the render frame as float32 torch scalars, computed like model._intrinsics."""
    K = torch.tensor(np.asarray(camera_K), dtype=torch.float32).view(1, 3, 3)
    s = RENDER_SCALE
    return K[:, 0, 0] * s, K[:, 1, 1] * s, K[:, 0, 2] * s, K[:, 1, 2] * s


def face_part_table(mano):
    """Hard-LBS joint of every vertex, and of every face (majority of its three vertices; a face whose
    three vertices carry three different joints takes the one with the largest summed LBS weight).
    Returns (vert_joint (778,), face_part (1538,), n_tie_faces)."""
    w = mano.weights.numpy().astype(np.float64)
    hard = w.argmax(1)
    f = mano.f.numpy()
    hj = hard[f]
    a, b, c = hj[:, 0], hj[:, 1], hj[:, 2]
    maj = np.where(a == b, a, np.where(a == c, a, np.where(b == c, b, -1)))
    tie = np.where(maj < 0)[0]
    for i in tie:
        cand = np.unique(hj[i])
        score = [w[f[i], j].sum() for j in cand]
        maj[i] = cand[int(np.argmax(score))]
    return hard, maj.astype(np.int64), int(len(tie))


def load_seq_bundle(seq, legacy_dir):
    events, offsets, aux, pos51 = load_sequence(DATA_ROOT, legacy_dir, seq)
    tp = DATA_ROOT / legacy_dir / f"{seq}_tsub.npy"
    tsub = np.load(tp, mmap_mode="r") if tp.exists() else None
    betas = torch.tensor(aux["betas"], dtype=torch.float32).view(1, -1)
    camera_K = np.asarray(aux["camera_K"], dtype=np.float32).reshape(3, 3)
    return events, offsets, tsub, aux, pos51, betas, camera_K
