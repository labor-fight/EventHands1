"""KA step 1 (docs/EXPERIMENT_RECORDS.md#ka-prereg-20261028): is an event map lifted onto a kinematic atlas of the MANO
surface more invariant to camera / pose nuisance than a similarity-normalised hand-box crop?  CPU, zero training.

Four representations of the same events of one 50 ms window, all given the same previous hand state (51-d, MANO storage
convention; betas; camera K at 640 x 480 scaled to the 240 x 180 event grid):

  img          full-frame counts on a 32 x 24 grid (reference)
  box{16,24}   similarity-normalised crop: centre = centroid of the projected visible vertices of the previous mesh,
               scale = 1.15 x their largest distance from the centre, rotation = projected wrist -> middle-MCP direction
               pointing up; G x G grid over [-1, 1]^2
  atlas_rigid  events lifted to the nearest visible projected vertex (within f r_band / z_median, r_band = 15 mm) of the
               mesh posed with the previous root rotation / translation and the MANO mean finger pose; binned by the
               vertex's atlas cell
  atlas_kin    same lifting on the full previous mesh (articulated fingers)

Atlas (intrinsic, fixed on the zero-shape template): every vertex belongs to the MANO joint with the largest skinning
weight (16 parts). Finger parts: axial position along the bone (joint -> child joint, or -> fingertip vertex for distal
bones) in 4 bins x angle about the bone in 8 bins (reference = palm normal). Palm (joint 0): 4 x 4 grid in the palm
plane x front / back side. 16 x 32 = 512 cells x 2 polarities.

Features: log1p counts, Gaussian blur of sigma = 1 cell (atlas: within a part, angle wraps; palm sides separate),
L2-normalised; similarity = cosine. Every tool verifies `outputs/kin_atlas/protocol.json` before reading data.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
for p in (str(REPO), str(REPO / "tools")):
    if p not in sys.path:
        sys.path.insert(0, p)

from tools.s43_br import geometry as G  # noqa: E402  (read-only: CPU MANO FK + triangle rasterizer)

OUT = REPO / "outputs" / "kin_atlas"
PROTOCOL = OUT / "protocol.json"
PREREG_BEGIN = '<a id="ka-prereg-20261028"></a>'
PREREG_END = "<!-- KA-PREREG:END -->"
ASSET = REPO / "assets" / "mano_right.npz"

EV_HW = (180, 240)
SRC_HW = (480, 640)
R_BAND_M = 0.015
VIS_TOL_M = 0.01
IMG_GRID = (24, 32)                 # (rows, cols)
BOX_GRIDS = (16, 24)
N_AX, N_ANG = 4, 8
N_PARTS = 16
CELLS_PER_PART = N_AX * N_ANG       # palm: 4 (u) x 4 (v) x 2 (side) = 32
REPS = ("img", "box16", "box24", "atlas_rigid", "atlas_kin")
# MANO 16-joint finger chains and their fingertip vertices (npz order: thumb, index, middle, ring, pinky)
FINGER_JOINTS = {"index": (1, 2, 3), "middle": (4, 5, 6), "pinky": (7, 8, 9), "ring": (10, 11, 12), "thumb": (13, 14, 15)}
TIP_VERTEX = {"thumb": 744, "index": 320, "middle": 443, "ring": 554, "pinky": 671}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
OP_TIPS = (4, 8, 12, 16, 20)        # OpenPose fingertips
OP_MIDDLE_MCP = 9


# ----------------------------------------------------------------------------------------------- protocol
def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prereg_text() -> Optional[str]:
    doc = (REPO / "docs" / "EXPERIMENT_RECORDS.md").read_text(encoding="utf-8")
    m = re.search(re.escape(PREREG_BEGIN) + r"(.*?)" + re.escape(PREREG_END), doc, re.S)
    return m.group(1) if m else None


def verify_protocol() -> dict:
    if not PROTOCOL.exists():
        raise SystemExit("outputs/kin_atlas/protocol.json missing: freeze before reading (tools/kin_atlas/freeze.py)")
    proto = json.loads(PROTOCOL.read_text())
    bad = [f for f, h in proto["files"].items() if not (REPO / f).exists() or sha(REPO / f) != h]
    if bad:
        raise SystemExit(f"frozen files changed since the KA freeze: {bad}")
    txt = prereg_text()
    if txt is None or hashlib.sha256(txt.encode()).hexdigest() != proto["prereg_sha256"]:
        raise SystemExit("the KA pre-registration text changed after the freeze")
    return proto


def _np(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save_json(path, obj, exclusive: bool = True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        raise SystemExit(f"{path} exists; preserve it")
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=_np, ensure_ascii=False))
    os.replace(tmp, path)


def force_cpu():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


# ----------------------------------------------------------------------------------------------- data (read-only)
DATA_ROOT = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51")
FIT7 = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq")
SELECT = ("ycy", "ylf")


def base_seqs(subjects=FIT7 + SELECT):
    return [f"{s}_{d}" for s in subjects for d in ("local", "global")]


@lru_cache(maxsize=1)
def _legacy_dirs() -> Dict[str, str]:
    m = json.loads((DATA_ROOT / "splits_semkine.json").read_text())
    out = {}
    for split in ("train", "val_core"):
        if split in m:
            out.update(m[split]["legacy_dir"])
    return out


def _read_meta51(path) -> np.ndarray:
    """`semkine/dataset.py::_read_meta51`, re-implemented without importing the model stack."""
    import struct
    with open(path, "rb") as f:
        (n,) = struct.unpack("<i", f.read(4))
        assert n == 51, n
        dt = np.dtype([("data", np.float64, n), ("m0", np.uint8), ("m1", np.uint8)])
        raw = np.fromfile(f, dtype=dt)
    assert np.all(raw["m0"] == 4) and np.all(raw["m1"] == 13)
    return raw["data"].astype(np.float32)


def load_seq(seq: str):
    """(events (N, 3) uint8 x / y / p mmap, offsets (n_ms + 1), aux npz, pos51 (n_ms, 51)) -- the files
    `semkine/eval_track.py::load_sequence` reads."""
    base = str(DATA_ROOT / _legacy_dirs()[seq] / seq)
    return (np.load(base + "_events.npy", mmap_mode="r"), np.load(base + "_offsets.npy"),
            np.load(base + "_aux.npz", allow_pickle=True), _read_meta51(base + ".meta"))


# ----------------------------------------------------------------------------------------------- MANO / camera
@lru_cache(maxsize=1)
def mano_npz() -> Dict[str, np.ndarray]:
    z = np.load(ASSET, allow_pickle=True)
    return {k: z[k] for k in z.files}


def event_K(K_cam: np.ndarray) -> np.ndarray:
    """Camera K (640 x 480) scaled to the 240 x 180 event grid (pixel-centre convention of tools/s43_br/geometry.py;
    event cell u is exactly coordinate u under the dataset's floor(x * 3 / 8) binning)."""
    return G.scale_intrinsics(np.asarray(K_cam, np.float64), EV_HW, SRC_HW)


def project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    h = points @ K.T
    return h[..., :2] / h[..., 2:3]


def fk(states: np.ndarray, betas: np.ndarray):
    """(verts (n, 778, 3), OpenPose joints (n, 21, 3)) in m."""
    return G.fk_geometry(np.asarray(states, np.float64).reshape(-1, 51), np.asarray(betas, np.float32))


def visible(verts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(V,) bool: 3 x 3 vertex-splat z-buffer on the 240 x 180 grid with 1 cm tolerance (semkine.aura.visible's rule)."""
    H, W = EV_HW
    uv = project(verts, K)
    z = verts[:, 2]
    ui, vi = np.rint(uv[:, 0]).astype(np.int64), np.rint(uv[:, 1]).astype(np.int64)
    zb = np.full(H * W, np.inf)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            uu, vv = ui + dx, vi + dy
            ok = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H) & (z > 0)
            np.minimum.at(zb, vv[ok] * W + uu[ok], z[ok])
    inb = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H) & (z > 0)
    zz = zb[np.clip(vi, 0, H - 1) * W + np.clip(ui, 0, W - 1)]
    return inb & (z <= zz + VIS_TOL_M)


# ----------------------------------------------------------------------------------------------- atlas
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


@lru_cache(maxsize=1)
def atlas_table() -> np.ndarray:
    """(778,) int: atlas cell of every vertex, in [0, 16 * 32). Computed on the zero-shape flat template."""
    z = mano_npz()
    V = z["v_template"].astype(np.float64)
    J = z["J"].astype(np.float64)
    W = z["weights"]
    part = np.argmax(W, axis=1)
    # palm normal: (indexMCP - wrist) x (pinkyMCP - wrist), oriented so the mean pose curls the tips towards +n
    n = _unit(np.cross(J[1] - J[0], J[7] - J[0]))
    flat = np.zeros(51)
    flat[6:] = -z["hands_mean"].reshape(-1)
    v_flat, _ = fk(flat[None], np.zeros(10, np.float32))
    v_mean, _ = fk(np.zeros((1, 51)), np.zeros(10, np.float32))
    tips = np.array(list(TIP_VERTEX.values()))
    if float(((v_mean[0, tips] - v_flat[0, tips]) @ n).mean()) < 0:
        n = -n
    cell = np.zeros(778, np.int64)
    child = {}
    for f, js in FINGER_JOINTS.items():
        child[js[0]] = ("j", js[1])
        child[js[1]] = ("j", js[2])
        child[js[2]] = ("v", TIP_VERTEX[f])
    # palm frame
    e1 = J[4] - J[0]
    e1 = _unit(e1 - (e1 @ n) * n)
    e2 = np.cross(n, e1)
    pv = np.flatnonzero(part == 0)
    du = (V[pv] - J[0]) @ e1
    dv = (V[pv] - J[0]) @ e2
    ub = np.clip(((du - du.min()) / (np.ptp(du) + 1e-12) * 4).astype(int), 0, 3)
    vb = np.clip(((dv - dv.min()) / (np.ptp(dv) + 1e-12) * 4).astype(int), 0, 3)
    side = ((V[pv] - J[0]) @ n > 0).astype(int)
    cell[pv] = 0 * CELLS_PER_PART + ub * 8 + side * 4 + vb
    for j in range(1, N_PARTS):
        idx = np.flatnonzero(part == j)
        if not len(idx):
            continue
        kind, c = child[j]
        end = J[c] if kind == "j" else V[c]
        axis = end - J[j]
        L = np.linalg.norm(axis)
        a = axis / L
        n1 = n - (n @ a) * a
        if np.linalg.norm(n1) < 1e-3:
            n1 = e2 - (e2 @ a) * a
        n1 = _unit(n1)
        n2 = np.cross(a, n1)
        d = V[idx] - J[j]
        t = np.clip((d @ a) / L, 0.0, 0.999999)
        ang = np.arctan2(d @ n2, d @ n1)
        ab = np.clip((t * N_AX).astype(int), 0, N_AX - 1)
        gb = np.clip(((ang + np.pi) / (2 * np.pi) * N_ANG).astype(int), 0, N_ANG - 1)
        cell[idx] = j * CELLS_PER_PART + ab * N_ANG + gb
    return cell


def flex_axes() -> Dict[str, np.ndarray]:
    """Per finger flexion axis in the template frame: b x n, b = first joint -> tip projected on the palm plane, sign
    chosen so that a positive angle moves the tip towards +n (curl). Used by the synthetic generator only."""
    z = mano_npz()
    J = z["J"].astype(np.float64)
    n = _unit(np.cross(J[1] - J[0], J[7] - J[0]))
    flat = np.zeros(51)
    flat[6:] = -z["hands_mean"].reshape(-1)
    v_flat, _ = fk(flat[None], np.zeros(10, np.float32))
    v_mean, _ = fk(np.zeros((1, 51)), np.zeros(10, np.float32))
    tips = np.array(list(TIP_VERTEX.values()))
    if float(((v_mean[0, tips] - v_flat[0, tips]) @ n).mean()) < 0:
        n = -n
    V = z["v_template"].astype(np.float64)
    out = {}
    for f, js in FINGER_JOINTS.items():
        r = V[TIP_VERTEX[f]] - J[js[0]]
        b = _unit(r - (r @ n) * n)
        out[f] = _unit(np.cross(b, n))
    return out


# ----------------------------------------------------------------------------------------------- representations
def _grid_counts(ix: np.ndarray, iy: np.ndarray, p: np.ndarray, rows: int, cols: int) -> np.ndarray:
    out = np.zeros((rows, cols, 2))
    ok = (ix >= 0) & (ix < cols) & (iy >= 0) & (iy < rows)
    np.add.at(out, (iy[ok], ix[ok], p[ok].astype(np.int64)), 1.0)
    return out


def _finish(arr: np.ndarray, blur_fn) -> np.ndarray:
    x = np.log1p(arr)
    x = blur_fn(x)
    v = x.reshape(-1)
    nrm = np.linalg.norm(v)
    return v / nrm if nrm > 0 else v


def _blur2d(x, on=True):
    return gaussian_filter(x, sigma=(1.0, 1.0, 0.0), mode="constant") if on else x


def _blur_atlas(x, on=True):
    if not on:
        return x
    x = x.reshape(N_PARTS, -1, 2).copy()
    palm = x[0].reshape(4, 2, 4, 2)                              # u, side, v, pol
    x[0] = gaussian_filter(palm, sigma=(1.0, 0.0, 1.0, 0.0), mode="nearest").reshape(-1, 2)
    fing = x[1:].reshape(N_PARTS - 1, N_AX, N_ANG, 2)
    fing = gaussian_filter(fing, sigma=(0.0, 1.0, 0.0, 0.0), mode="nearest")
    fing = gaussian_filter(fing, sigma=(0.0, 0.0, 1.0, 0.0), mode="wrap")
    x[1:] = fing.reshape(N_PARTS - 1, -1, 2)
    return x


def lift_counts(ex, ey, ep, verts: np.ndarray, K: np.ndarray, table: np.ndarray) -> Tuple[np.ndarray, float]:
    """Atlas counts (16 * 32, 2) of the events lifted onto the visible vertices of `verts`, and the share of events
    left outside the band."""
    vis = visible(verts, K)
    out = np.zeros((N_PARTS * CELLS_PER_PART, 2))
    if not vis.any() or len(ex) == 0:
        return out, 1.0 if len(ex) else 0.0
    uv = project(verts[vis], K)
    zmed = float(np.median(verts[vis, 2]))
    rho = K[0, 0] * R_BAND_M / zmed
    tree = cKDTree(uv)
    d, k = tree.query(np.stack([ex, ey], 1).astype(np.float64), k=1)
    ok = d <= rho
    cells = table[np.flatnonzero(vis)[k[ok]]]
    np.add.at(out, (cells, ep[ok].astype(np.int64)), 1.0)
    return out, float(1.0 - ok.mean())


def box_counts(ex, ey, ep, verts: np.ndarray, joints: np.ndarray, K: np.ndarray, grid: int) -> np.ndarray:
    vis = visible(verts, K)
    if not vis.any():
        return np.zeros((grid, grid, 2))
    uv = project(verts[vis], K)
    c = uv.mean(0)
    S = 1.15 * float(np.max(np.linalg.norm(uv - c, axis=1))) + 1e-6
    jp = project(joints[[0, OP_MIDDLE_MCP]], K)
    d = _unit(jp[1] - jp[0])
    perp = np.array([-d[1], d[0]])
    q = np.stack([ex, ey], 1).astype(np.float64) - c
    x = (q @ perp) / S
    y = -(q @ d) / S
    ix = np.floor((x + 1) / 2 * grid).astype(np.int64)
    iy = np.floor((y + 1) / 2 * grid).astype(np.int64)
    return _grid_counts(ix, iy, ep, grid, grid)


def representations(ex, ey, ep, prev_state: np.ndarray, betas: np.ndarray, K_cam: np.ndarray,
                    blur: bool = True) -> Dict[str, np.ndarray]:
    """All five feature vectors of one window (see the module docstring) and the atlas off-band shares."""
    K = event_K(K_cam)
    ex, ey, ep = (np.asarray(a) for a in (ex, ey, ep))
    st = np.asarray(prev_state, np.float64).reshape(51)
    rigid = st.copy()
    rigid[6:] = 0.0                                              # residual 0 = MANO mean finger pose
    vv, jj = fk(np.stack([st, rigid]), betas)
    v_kin, v_rig, j_kin = vv[0], vv[1], jj[0]
    table = atlas_table()
    out = {}
    out["img"] = _finish(_grid_counts(np.floor(ex * IMG_GRID[1] / EV_HW[1]).astype(np.int64),
                                      np.floor(ey * IMG_GRID[0] / EV_HW[0]).astype(np.int64), ep, *IMG_GRID),
                         lambda x: _blur2d(x, blur))
    for g in BOX_GRIDS:
        out[f"box{g}"] = _finish(box_counts(ex, ey, ep, v_kin, j_kin, K, g), lambda x: _blur2d(x, blur))
    ar, off_r = lift_counts(ex, ey, ep, v_rig, K, table)
    ak, off_k = lift_counts(ex, ey, ep, v_kin, K, table)
    out["atlas_rigid"] = _finish(ar, lambda x: _blur_atlas(x, blur))
    out["atlas_kin"] = _finish(ak, lambda x: _blur_atlas(x, blur))
    out["_offband_rigid"] = np.array([off_r])
    out["_offband_kin"] = np.array([off_k])
    out["_n_events"] = np.array([len(ex)])
    return out


def perturb_prev(state: np.ndarray, betas: np.ndarray, rng: np.random.Generator, lateral_m: float = 0.010,
                 rot_deg: float = 5.0) -> np.ndarray:
    """Previous state with a 10 mm image-plane (camera x / y) shift and a 5 deg root rotation about a random axis
    (camera frame, about the wrist) -- the descriptive wrong-prev condition."""
    from scipy.spatial.transform import Rotation as R
    s = np.asarray(state, np.float64).reshape(51).copy()
    _, j0 = fk(s[None], betas)
    w = j0[0, 0]
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    Rn = R.from_rotvec(ax * np.radians(rot_deg))
    s[3:6] = (Rn * R.from_rotvec(s[3:6])).as_rotvec()
    _, j1 = fk(s[None], betas)
    s[:3] += w - j1[0, 0]                                        # keep the wrist fixed
    phi = rng.uniform(0, 2 * np.pi)
    s[0] += lateral_m * np.cos(phi)
    s[1] += lateral_m * np.sin(phi)
    return s


# ----------------------------------------------------------------------------------------------- statistics
def auc_from(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(tie) over all pairs."""
    pos, neg = np.asarray(pos, np.float64), np.asarray(neg, np.float64)
    if not len(pos) or not len(neg):
        return float("nan")
    neg_sorted = np.sort(neg)
    lo = np.searchsorted(neg_sorted, pos, side="left")
    hi = np.searchsorted(neg_sorted, pos, side="right")
    return float(((lo + 0.5 * (hi - lo)) / len(neg)).mean())


def boot_ci(values: np.ndarray, groups: np.ndarray, n_boot: int = 2000, seed: int = 20261028):
    values = np.asarray(values, np.float64)
    groups = np.asarray(groups)
    ug = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in ug}
    sums = np.array([values[idx[g]].sum() for g in ug])
    cnts = np.array([len(idx[g]) for g in ug], np.float64)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        bs.append(sums[pick].sum() / cnts[pick].sum())
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
