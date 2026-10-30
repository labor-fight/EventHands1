#!/usr/bin/env python3
"""MV2 / M3 + M8: CPU subtree FK-composite warp and image-of-warped-events (IWE) toolkit.

Object model (docs: tools/mv2/prereg_m3.md, prereg_m8.md):

  * a hand trajectory (`Traj`) is a MANO FK sequence sampled every `FK_STEP` ms (`ManoLayer` through
    `tools/sba/common.py::Ctx`, the evaluator's 51-d decoding), with per-sample projected vertices in render pixels
    (K x 0.375, the model's `_intrinsics`), per-sample vertex visibility (`semkine.aura.visible`, refreshed every
    `VIS_STEP` ms) and, on request, the visible occluding-contour segments (`semkine.sba.contour_entries`, every
    `CONTOUR_STEP` ms);
  * an event (`Events`) is (grid column, grid row, polarity, absolute time in us); its position is the exact source-pixel
    centre of its 240 x 180 grid cell (`semkine.sba.make_grid` table x 0.375: the unbiased convention (ii) of Stage 0
    A1(b), mean offset 0 instead of the network convention's -0.4375 px);
  * `assign` puts each event on a *material point* of the trajectory: the nearest visible vertex (<= `R_NEAR` px) at the
    event's own timestamp, then the best incident front-facing face and its (clamped) barycentric coordinates; the
    residual between the event and the projected material point is carried;
  * `warp` moves the event to a reference time with the FK displacement of its material point
    (p' = p + sum_i b_i (u_i(t_ref) - u_i(t))): a per-subtree composite warp, because every face carries the motion of
    the part (palm / index / middle / pinky / ring / thumb, `Geometry.groups`) its vertices belong to;
  * `splat` / `contrast_stats` build the IWE by bilinear splatting and report variance (the brief's contrast) and a
    scale-free pair concentration; `warp_area_stats` / `cloud_area_ratio` are the Shiba-style collapse diagnostics
    (mean log area change of the warp, area ratio of the warped cloud), `var * area_ratio^2` the area-normalised contrast;
  * `contour_band` separates texture events from contour events (within `CONTOUR_PX` px of the FK silhouette /
    occlusion contour at the event's own time); contour events can be warped by the normal component only
    (aperture-consistent) because the contour generator slides on the surface;
  * null generators (`shuffle_times`): `cluster` = whole time clusters (events sharing one timestamp) exchange their
    timestamp labels across the window (cluster sizes and the common time layers are preserved), `interval` = events
    exchange timestamps only inside their own source-frame interval, `window` = global event-level permutation (kept only
    as a diagnostic: the common time layers make it invalid as a gate null); `thin` (equal-event-count control),
    `swap_polarity` (per-pixel random half or global).

Nothing here reads a result: every function is a pure operator on arrays; `selftest()` runs synthetic checks only.
"""
from __future__ import annotations

import math
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sba import common as CM  # noqa: E402  (tools/sba/common.py: Ctx = ManoLayer + Geometry, read-only)
from semkine import aura as AU  # noqa: E402
from semkine import sba as SB  # noqa: E402

H, W = 180, 240
RS = 0.375                    # render scale: K x 0.375
GROUP_NAMES = ("palm", "index", "middle", "pinky", "ring", "thumb")      # `Geometry.groups` order
FINGERS5 = ("thumb", "index", "middle", "ring", "pinky")
GROUP_OF = {n: i for i, n in enumerate(GROUP_NAMES)}
R_NEAR = 4.0                  # px: events farther than this from every visible vertex are not assigned
CONTOUR_PX = 2.0              # px: contour band half-width
FK_STEP = 2                   # ms between FK samples
VIS_STEP = 6                  # ms between visibility refreshes
CONTOUR_STEP = 4              # ms between contour extractions
VMAX = 10                     # padded incident-face count per vertex
SEED = 20261026


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode()) & 0x7FFFFFFF


# ----------------------------------------------------------------------------------------------------- process globals
_G: Dict[str, object] = {}


def geo() -> dict:
    """CPU singletons of the process: SBA context (MANO layer + Geometry), face tables, grid centres in render px."""
    if not _G:
        torch.set_num_threads(1)
        dev = torch.device("cpu")
        ctx = CM.Ctx(dev)
        faces = ctx.mano.f.long()
        nv = int(ctx.mano.v_template.shape[0])
        vf = -torch.ones(nv, VMAX, dtype=torch.long)
        cnt = torch.zeros(nv, dtype=torch.long)
        for fi, tri in enumerate(faces.tolist()):
            for v in tri:
                c = int(cnt[v])
                if c < VMAX:
                    vf[v, c] = fi
                    cnt[v] += 1
        if int(cnt.max()) > VMAX:
            raise RuntimeError("vertex valence above VMAX")
        grid = SB.make_grid(dev)
        _G.update(ctx=ctx, faces=faces, nv=nv, vf=vf, grid=grid, groups=ctx.geo.groups.long(),
                  topo=ctx.geo.topo, cx=(grid.cx_src * RS).float(), cy=(grid.cy_src * RS).float())
    return _G


class Hand:
    """FK / projection / visibility for one hand shape and camera (CPU)."""

    def __init__(self, betas: np.ndarray, K: np.ndarray):
        g = geo()
        self.ctx = g["ctx"]
        self.betas = torch.tensor(np.asarray(betas, np.float32).reshape(-1)[:10], dtype=torch.float32).view(1, -1)
        Kt = torch.tensor(np.asarray(K, np.float32).reshape(3, 3))
        self.K = Kt
        self.fx, self.fy, self.cx, self.cy = (float(Kt[0, 0] * RS), float(Kt[1, 1] * RS), float(Kt[0, 2] * RS),
                                              float(Kt[1, 2] * RS))

    @torch.no_grad()
    def fk(self, X: torch.Tensor):
        """X (n, 51) -> verts (n, 778, 3), joints (n, 21, 3)."""
        return self.ctx.geo.fk(X.float(), self.betas)

    def uv(self, verts: torch.Tensor) -> torch.Tensor:
        z = verts[..., 2].clamp(min=1e-6)
        return torch.stack([self.fx * verts[..., 0] / z + self.cx, self.fy * verts[..., 1] / z + self.cy], -1)

    @torch.no_grad()
    def visible(self, verts: torch.Tensor) -> torch.Tensor:
        n = verts.shape[0]
        intr = tuple(torch.full((n,), v, dtype=torch.float32) for v in (self.fx, self.fy, self.cx, self.cy))
        return AU.visible(verts, intr)


# ----------------------------------------------------------------------------------------------------- events
@dataclass
class Events:
    ex: np.ndarray            # (n,) grid column 0..239
    ey: np.ndarray            # (n,) grid row 0..179
    p: np.ndarray             # (n,) polarity 0 / 1
    t_us: np.ndarray          # (n,) absolute time, us

    def __len__(self):
        return int(len(self.ex))

    def take(self, idx) -> "Events":
        return Events(self.ex[idx], self.ey[idx], self.p[idx], self.t_us[idx])

    def xy(self) -> np.ndarray:
        """(n, 2) float32 render-pixel positions: exact source-pixel centre of the grid cell x 0.375."""
        g = geo()
        return np.stack([g["cx"].numpy()[self.ex.astype(np.int64)], g["cy"].numpy()[self.ey.astype(np.int64)]], 1)

    @property
    def t_ms(self) -> np.ndarray:
        return self.t_us.astype(np.float64) / 1000.0

    def with_times(self, t_us) -> "Events":
        return Events(self.ex, self.ey, self.p, np.asarray(t_us, np.int64))


def empty_events() -> Events:
    return Events(np.zeros(0, np.int16), np.zeros(0, np.int16), np.zeros(0, np.uint8), np.zeros(0, np.int64))


class Rec:
    """One BASE recording of hand_data51: events (mmap), offsets, sub-ms times, GT per ms, betas, K, valid runs."""

    def __init__(self, seq: str):
        from mv2 import common as MC
        self.seq = seq
        ev, off, aux, pos = MC.load_seq(seq)
        d = MC.seq_dir(seq)
        tpath = MC.DATA_ROOT / d / f"{seq}_tsub.npy"
        self.tsub = np.load(tpath, mmap_mode="r") if tpath.exists() else None
        self.events, self.offsets, self.pos51, self.aux = ev, off, pos, aux
        self.betas = np.asarray(aux["betas"], np.float32).reshape(-1)[:10]
        self.K = np.asarray(aux["camera_K"], np.float32).reshape(3, 3)
        self.runs = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)
        self.n_ms = len(off) - 1
        self.hand = Hand(self.betas, self.K)

    def events_in(self, lo_ms: int, hi_ms: int) -> Events:
        """Events of ms [lo_ms, hi_ms] inclusive (absolute times, us)."""
        a0, a1 = int(self.offsets[lo_ms]), int(self.offsets[hi_ms + 1])
        if a1 <= a0:
            return empty_events()
        ev = np.asarray(self.events[a0:a1])
        cnt = np.diff(self.offsets[lo_ms:hi_ms + 2]).astype(np.int64)
        ms = np.repeat(np.arange(lo_ms, hi_ms + 1, dtype=np.int64), cnt)
        us = ms * 1000
        if self.tsub is not None:
            us = us + np.asarray(self.tsub[a0:a1]).astype(np.int64)
        return Events(ev[:, 0].astype(np.int16), ev[:, 1].astype(np.int16), np.clip(ev[:, 2], 0, 1).astype(np.uint8),
                      us)

    def pose(self, t_ms: np.ndarray) -> np.ndarray:
        """(n, 51) float32 GT at integer ms indices."""
        return np.stack([np.asarray(self.pos51[int(t)], np.float32) for t in t_ms])


# ----------------------------------------------------------------------------------------------------- trajectory
@dataclass
class Traj:
    t0: float                              # ms of sample 0
    dt: float                              # ms between samples
    uv: torch.Tensor                       # (n, 778, 2) render px
    vis: torch.Tensor                      # (n, 778) bool
    verts: torch.Tensor                    # (n, 778, 3)
    X: torch.Tensor                        # (n, 51) poses
    contour: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)   # sample -> (a, b) (m, 2)

    @property
    def n(self) -> int:
        return int(self.uv.shape[0])

    def frac_index(self, t_ms) -> np.ndarray:
        return (np.asarray(t_ms, np.float64) - self.t0) / self.dt


def build_traj(hand: Hand, X: np.ndarray, t0: float, dt: float = FK_STEP, vis_every: Optional[int] = None) -> Traj:
    """FK + projection of poses X (n, 51) sampled at t0 + k dt; visibility refreshed every `vis_every` samples."""
    Xt = torch.as_tensor(np.asarray(X, np.float32))
    verts, _ = hand.fk(Xt)
    uv = hand.uv(verts)
    n = Xt.shape[0]
    ve = max(1, int(round(VIS_STEP / dt))) if vis_every is None else int(vis_every)
    ks = np.arange(0, n, ve)
    vis_k = hand.visible(verts[ks])
    owner = np.minimum(np.arange(n) // ve, len(ks) - 1)
    vis = vis_k[torch.as_tensor(owner)]
    return Traj(float(t0), float(dt), uv, vis, verts, Xt)


def traj_from_rec(rec: Rec, t_lo: int, t_hi: int, perturb=None, dt: float = FK_STEP) -> Traj:
    """GT (or perturbed) trajectory of ms [t_lo, t_hi] sampled every `dt` ms; `perturb(X, t_ms) -> X'` optional."""
    t = np.arange(t_lo, t_hi + 1, dt)
    X = rec.pose(np.round(t).astype(np.int64))
    if perturb is not None:
        X = perturb(X, t)
    return build_traj(rec.hand, X, float(t_lo), dt)


# ----------------------------------------------------------------------------------------------------- contour
@torch.no_grad()
def ensure_contour(traj: Traj, ks: Sequence[int], chunk: int = 24) -> None:
    """Visible occluding-contour segments (render px endpoints) at FK samples `ks` (cached in `traj.contour`)."""
    g = geo()
    todo = [int(k) for k in ks if int(k) not in traj.contour]
    for i in range(0, len(todo), chunk):
        part = todo[i:i + chunk]
        v = traj.verts[part]
        uv_src = traj.uv[part] / RS
        on, _, _ = SB.contour_entries(v, uv_src, g["topo"], g["grid"], g["groups"], 6)
        e0, e1 = g["topo"].edges[:, 0], g["topo"].edges[:, 1]
        for j, k in enumerate(part):
            m = on[j]
            a = traj.uv[k][e0[m]]
            b = traj.uv[k][e1[m]]
            traj.contour[k] = (a, b)


def contour_samples(traj: Traj, t_ms: np.ndarray) -> np.ndarray:
    """FK sample index (multiple of CONTOUR_STEP / dt) used for the contour of events at times `t_ms`."""
    step = max(1, int(round(CONTOUR_STEP / traj.dt)))
    k = np.rint(traj.frac_index(t_ms) / step).astype(np.int64) * step
    return np.clip(k, 0, ((traj.n - 1) // step) * step)


def contour_band(traj: Traj, xy: np.ndarray, t_ms: np.ndarray, px: float = CONTOUR_PX):
    """(is_contour (n,), normal (n, 2), dist (n,)): distance of every event to the nearest visible contour segment at its
    own time (contour sampled every CONTOUR_STEP ms); the unit normal of that segment (zero where none)."""
    n = len(xy)
    kc = contour_samples(traj, t_ms)
    ensure_contour(traj, np.unique(kc))
    dist = np.full(n, np.inf, np.float32)
    nrm = np.zeros((n, 2), np.float32)
    order = np.argsort(kc, kind="stable")
    ks, starts = np.unique(kc[order], return_index=True)
    ends = np.r_[starts[1:], n]
    P = torch.as_tensor(xy)
    for k, s, e in zip(ks, starts, ends):
        idx = order[s:e]
        a, b = traj.contour[int(k)]
        if a.shape[0] == 0:
            continue
        p = P[idx][:, None, :]
        ab = (b - a)[None]
        t = (((p - a[None]) * ab).sum(-1) / (ab * ab).sum(-1).clamp(min=1e-9)).clamp(0, 1)
        d = (p - (a[None] + t[..., None] * ab)).norm(dim=-1)
        dm, j = d.min(1)
        seg = (b - a)[j]
        n_ = torch.stack([-seg[:, 1], seg[:, 0]], -1)
        n_ = n_ / n_.norm(dim=-1, keepdim=True).clamp(min=1e-9)
        dist[idx] = dm.numpy()
        nrm[idx] = n_.numpy()
    return dist <= px, nrm, dist


# ----------------------------------------------------------------------------------------------------- assignment
@dataclass
class Assign:
    face: np.ndarray          # (n,) int64 face id (valid where ok)
    bary: np.ndarray          # (n, 3) float32 clamped barycentric coordinates
    resid: np.ndarray         # (n, 2) event minus projected material point at its own time
    vert: np.ndarray          # (n,) nearest visible vertex id (-1 where none)
    group: np.ndarray         # (n,) part group of that vertex (-1 where none)
    ok: np.ndarray            # (n,) bool: assigned (nearest visible vertex within R_NEAR)
    front: np.ndarray         # (n,) bool: the chosen face is front-facing
    dist: np.ndarray          # (n,) px to the nearest visible vertex


def _bary(p: torch.Tensor, tri: torch.Tensor):
    """p (m, 1, 2), tri (m, c, 3, 2) -> bary (m, c, 3), degenerate (m, c)."""
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01
    deg = den.abs() < 1e-9
    den = torch.where(deg, torch.ones_like(den), den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    area = v0[..., 0] * v1[..., 1] - v0[..., 1] * v1[..., 0]
    return torch.stack([1 - v - w, v, w], -1), deg, area


@torch.no_grad()
def assign(traj: Traj, xy: np.ndarray, t_ms: np.ndarray, r_near: float = R_NEAR) -> Assign:
    """Material point of every event at its own time on the (hypothesised) trajectory."""
    g = geo()
    faces, vf, groups = g["faces"], g["vf"], g["groups"]
    n = len(xy)
    face = np.zeros(n, np.int64)
    bary = np.full((n, 3), 1.0 / 3.0, np.float32)
    resid = np.zeros((n, 2), np.float32)
    vert = np.full(n, -1, np.int64)
    ok = np.zeros(n, bool)
    front = np.zeros(n, bool)
    dist = np.full(n, np.inf, np.float32)
    if n == 0:
        return Assign(face, bary, resid, vert, np.full(0, -1, np.int64), ok, front, dist)
    k = np.clip(np.rint(traj.frac_index(t_ms)).astype(np.int64), 0, traj.n - 1)
    order = np.argsort(k, kind="stable")
    ks, starts = np.unique(k[order], return_index=True)
    ends = np.r_[starts[1:], n]
    P = torch.as_tensor(xy)
    for kk, s, e in zip(ks, starts, ends):
        idx = order[s:e]
        vid = traj.vis[kk].nonzero().flatten()
        if len(vid) == 0:
            continue
        uvk = traj.uv[kk]
        E = P[idx]
        d = torch.cdist(E, uvk[vid])
        dm, j = d.min(1)
        v = vid[j]
        cand = vf[v]                                                  # (m, VMAX) face ids, -1 pad
        valid = cand >= 0
        fv = faces[cand.clamp(min=0)]                                 # (m, VMAX, 3)
        tri = uvk[fv]                                                 # (m, VMAX, 3, 2)
        bc, deg, area = _bary(E[:, None, :], tri)
        is_front = area > 0
        score = bc.min(-1).values
        score = torch.where(is_front, score, score - 10.0)
        score = torch.where(deg | ~valid, torch.full_like(score, -1e9), score)
        best = score.argmax(1)
        ar = torch.arange(len(idx))
        b = bc[ar, best].clamp(min=0)
        b = b / b.sum(-1, keepdim=True).clamp(min=1e-9)
        tri_b = tri[ar, best]
        pm = (b[..., None] * tri_b).sum(1)
        face[idx] = cand[ar, best].numpy()
        bary[idx] = b.numpy()
        resid[idx] = (E - pm).numpy()
        vert[idx] = v.numpy()
        ok[idx] = (dm <= r_near).numpy()
        front[idx] = is_front[ar, best].numpy()
        dist[idx] = dm.numpy()
    grp = np.where(vert >= 0, groups.numpy()[np.clip(vert, 0, None)], -1)
    return Assign(face, bary, resid, vert, grp, ok, front, dist)


# ----------------------------------------------------------------------------------------------------- warp
def _uv_face_at(traj: Traj, fv: torch.Tensor, t_ms: np.ndarray) -> torch.Tensor:
    """(n, 3, 2) projected face vertices at times t_ms (linear in time between FK samples)."""
    f = np.clip(traj.frac_index(t_ms), 0, traj.n - 1 - 1e-9)
    i0 = np.floor(f).astype(np.int64)
    a = torch.as_tensor((f - i0).astype(np.float32))[:, None, None]
    i0t = torch.as_tensor(i0)
    u0 = traj.uv[i0t[:, None], fv]
    u1 = traj.uv[torch.clamp(i0t + 1, max=traj.n - 1)[:, None], fv]
    return (1 - a) * u0 + a * u1


@torch.no_grad()
def displacement(traj: Traj, asg: Assign, t_ev_ms: np.ndarray, t_ref_ms) -> np.ndarray:
    """(n, 2) image-plane displacement of every event's material point from its own time to the reference time(s)."""
    g = geo()
    n = len(t_ev_ms)
    if n == 0:
        return np.zeros((0, 2), np.float32)
    fv = g["faces"][torch.as_tensor(asg.face)]
    tr = np.broadcast_to(np.asarray(t_ref_ms, np.float64), (n,))
    b = torch.as_tensor(asg.bary)[..., None]
    p_ev = (b * _uv_face_at(traj, fv, t_ev_ms)).sum(1)
    p_ref = (b * _uv_face_at(traj, fv, tr)).sum(1)
    d = (p_ref - p_ev).numpy()
    d[~asg.ok] = 0.0
    return d


def warp(xy: np.ndarray, d: np.ndarray, scale: float = 1.0, normals: Optional[np.ndarray] = None,
         use_normal: Optional[np.ndarray] = None) -> np.ndarray:
    """p' = p + scale * d; where `use_normal` (bool) only the component of d along `normals` (aperture-consistent)."""
    dd = d.copy()
    if normals is not None and use_normal is not None and use_normal.any():
        nn = normals[use_normal]
        dd[use_normal] = nn * (nn * d[use_normal]).sum(-1, keepdims=True)
    return (xy + scale * dd).astype(np.float32)


# ----------------------------------------------------------------------------------------------------- IWE
@dataclass
class Box:
    x0: int
    y0: int
    w: int
    h: int


def box_from_points(pts: np.ndarray, margin: float = 6.0) -> Box:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    x0 = int(math.floor(pts[:, 0].min() - margin))
    y0 = int(math.floor(pts[:, 1].min() - margin))
    x1 = int(math.ceil(pts[:, 0].max() + margin))
    y1 = int(math.ceil(pts[:, 1].max() + margin))
    return Box(x0, y0, max(2, x1 - x0 + 1), max(2, y1 - y0 + 1))


def splat(xy: np.ndarray, box: Box, weights: Optional[np.ndarray] = None):
    """Bilinear splat of points on the integer lattice of `box`: image (h, w) float64 and the per-event self term
    sum_c w_c^2 over in-bounds corners (float64, (n,))."""
    n = len(xy)
    img = np.zeros(box.h * box.w, np.float64)
    if n == 0:
        return img.reshape(box.h, box.w), np.zeros(0)
    wt = np.ones(n, np.float64) if weights is None else np.asarray(weights, np.float64)
    x = xy[:, 0].astype(np.float64) - box.x0
    y = xy[:, 1].astype(np.float64) - box.y0
    xf, yf = np.floor(x), np.floor(y)
    fx, fy = x - xf, y - yf
    xi, yi = xf.astype(np.int64), yf.astype(np.int64)
    selfv = np.zeros(n, np.float64)
    for ox, oy, c in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xx, yy = xi + ox, yi + oy
        m = (xx >= 0) & (xx < box.w) & (yy >= 0) & (yy < box.h)
        img += np.bincount(yy[m] * box.w + xx[m], weights=(wt * c)[m], minlength=box.h * box.w)
        selfv[m] += ((wt * c)[m]) ** 2
    return img.reshape(box.h, box.w), selfv


def contrast_stats(xy: np.ndarray, box: Box, weights: Optional[np.ndarray] = None) -> Dict[str, float]:
    """IWE statistics: `var` = variance of the image over the box's pixels (the brief's contrast); `conc` = pair
    concentration P (sum I^2 - self) / N^2 (1 for uniform random placement, scale-free in the event count);
    `n_in` = mass inside the box."""
    img, selfv = splat(xy, box, weights)
    P = float(box.w * box.h)
    tot = float(img.sum())
    s2 = float((img ** 2).sum())
    var = s2 / P - (tot / P) ** 2
    # in-bounds mass: the net mass for unit weights, sum |w| for signed weights (polarity-signed IWE)
    mass = tot if weights is None else float(splat(xy, box, np.abs(np.asarray(weights, np.float64)))[0].sum())
    conc = (s2 - float(selfv.sum())) * P / max(mass ** 2, 1e-12) if mass > 0 else float("nan")
    return {"var": float(var), "conc": float(conc), "n_in": mass}


def cloud_area_ratio(before: np.ndarray, after: np.ndarray) -> float:
    """sqrt(det Cov(after) / det Cov(before)): area ratio of the event cloud (< 1: contraction)."""
    if len(before) < 3:
        return float("nan")
    cb = np.cov(before.astype(np.float64).T) + 1e-9 * np.eye(2)
    ca = np.cov(after.astype(np.float64).T) + 1e-9 * np.eye(2)
    return float(math.sqrt(max(np.linalg.det(ca), 1e-18) / max(np.linalg.det(cb), 1e-18)))


@torch.no_grad()
def warp_area_stats(traj: Traj, asg: Assign, t_ev_ms: np.ndarray, t_ref_ms) -> Dict[str, float]:
    """Divergence / area diagnostics of the warp over the assigned events: mean log area ratio of their faces between
    the event time and the reference time and the fraction of events whose face area shrinks below one half."""
    g = geo()
    sel = np.nonzero(asg.ok)[0]
    if len(sel) == 0:
        return {"mean_log_area": float("nan"), "frac_contract": float("nan"), "n": 0}
    fv = g["faces"][torch.as_tensor(asg.face[sel])]
    tr = np.broadcast_to(np.asarray(t_ref_ms, np.float64), (len(t_ev_ms),))[sel]

    def area(u):
        a, b, c = u[:, 0], u[:, 1], u[:, 2]
        return ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])).abs()

    a_ev = area(_uv_face_at(traj, fv, t_ev_ms[sel])).clamp(min=1e-3)
    a_rf = area(_uv_face_at(traj, fv, tr)).clamp(min=1e-3)
    lr = (a_rf / a_ev).log()
    return {"mean_log_area": float(lr.mean()), "frac_contract": float((lr < math.log(0.5)).float().mean()),
            "n": int(len(sel))}


# ----------------------------------------------------------------------------------------------------- controls
def shuffle_times(t_us: np.ndarray, mode: str, rng: np.random.Generator, interval_us: Optional[float] = None
                  ) -> np.ndarray:
    """Time nulls. `cluster`: clusters (events with an identical timestamp) exchange their time labels across the window;
    `interval`: events exchange timestamps inside their own source-frame interval (blocks of `interval_us`);
    `window`: global event-level permutation (diagnostic only)."""
    t = np.asarray(t_us, np.int64)
    n = len(t)
    if n == 0:
        return t.copy()
    if mode == "cluster":
        uq, inv = np.unique(t, return_inverse=True)
        return uq[rng.permutation(len(uq))][inv]
    if mode == "window":
        return t[rng.permutation(n)]
    if mode == "interval":
        if not interval_us:
            raise ValueError("interval null needs interval_us")
        blk = np.floor(t / float(interval_us)).astype(np.int64)
        order0 = np.argsort(blk, kind="stable")
        order1 = np.lexsort((rng.random(n), blk))
        out = t.copy()
        out[order0] = t[order1]
        return out
    raise ValueError(mode)


def thin_index(n: int, n_keep: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform random subset (sorted) of size min(n, n_keep) -- the equal-event-count control and the event cap."""
    if n <= n_keep:
        return np.arange(n)
    return np.sort(rng.choice(n, size=int(n_keep), replace=False))


def swap_polarity(ev: Events, mode: str, rng: np.random.Generator) -> Events:
    """`pixel_half`: a random half of the pixels flip polarity (kills spatial polarity structure, keeps counts);
    `global`: every polarity flips."""
    if mode == "global":
        return Events(ev.ex, ev.ey, (1 - ev.p).astype(np.uint8), ev.t_us)
    if mode == "pixel_half":
        flip = rng.random(H * W) < 0.5
        f = flip[ev.ey.astype(np.int64) * W + ev.ex.astype(np.int64)]
        return Events(ev.ex, ev.ey, np.where(f, 1 - ev.p, ev.p).astype(np.uint8), ev.t_us)
    raise ValueError(mode)


def uniform_noise_like(ev: Events, box: Box, rng: np.random.Generator) -> Events:
    """Noise control: same count, times and polarities, positions uniform in `box` (grid cells)."""
    n = len(ev)
    ex = np.clip(rng.integers(box.x0, box.x0 + box.w, n), 0, W - 1).astype(np.int16)
    ey = np.clip(rng.integers(box.y0, box.y0 + box.h, n), 0, H - 1).astype(np.int16)
    return Events(ex, ey, ev.p.copy(), ev.t_us.copy())


def add_flash(ev: Events, box: Box, frac: float, rng: np.random.Generator) -> Events:
    """Flash control: add frac * n events at one random timestamp of the window, uniform positions in `box`."""
    n = len(ev)
    m = int(round(frac * n))
    if m == 0 or n == 0:
        return ev
    t = int(ev.t_us[rng.integers(0, n)])
    ex = np.clip(rng.integers(box.x0, box.x0 + box.w, m), 0, W - 1).astype(np.int16)
    ey = np.clip(rng.integers(box.y0, box.y0 + box.h, m), 0, H - 1).astype(np.int16)
    p = rng.integers(0, 2, m).astype(np.uint8)
    cat = lambda a, b: np.concatenate([a, b])           # noqa: E731
    out = Events(cat(ev.ex, ex), cat(ev.ey, ey), cat(ev.p, p), cat(ev.t_us, np.full(m, t, np.int64)))
    o = np.argsort(out.t_us, kind="stable")
    return out.take(o)


# ----------------------------------------------------------------------------------------------------- one-call IWE
def iwe_eval(traj: Traj, ev: Events, box: Box, refs_ms: Sequence[float], member: Optional[np.ndarray] = None,
             channel: Optional[np.ndarray] = None, contour_normal: Optional[np.ndarray] = None,
             use_normal: Optional[np.ndarray] = None, scale: float = 1.0, asg: Optional[Assign] = None,
             xy: Optional[np.ndarray] = None) -> Dict[str, object]:
    """Contrast of the events `member` (default all) warped by `traj` to every reference time and averaged over the
    references (multi-reference). Events not assigned stay at their own position (identity warp). Returns the per-reference
    `var` / `conc`, their means, the assignment (for reuse) and the collapse statistics at the last reference."""
    xy = ev.xy() if xy is None else xy
    t_ms = ev.t_ms
    asg = assign(traj, xy, t_ms) if asg is None else asg
    sel = np.arange(len(ev)) if member is None else np.nonzero(member)[0]
    out = {"asg": asg, "var": [], "conc": [], "n_in": []}
    for tr in refs_ms:
        # displacement only for the selected events (assignment arrays indexed by sel)
        sub = Assign(asg.face[sel], asg.bary[sel], asg.resid[sel], asg.vert[sel], asg.group[sel], asg.ok[sel],
                     asg.front[sel], asg.dist[sel])
        d = displacement(traj, sub, t_ms[sel], tr)
        nn = None if contour_normal is None else contour_normal[sel]
        un = None if use_normal is None else use_normal[sel]
        xw = warp(xy[sel], d, scale, nn, un)
        st = contrast_stats(xw, box)
        out["var"].append(st["var"])
        out["conc"].append(st["conc"])
        out["n_in"].append(st["n_in"])
    out["var_mean"] = float(np.mean(out["var"]))
    out["conc_mean"] = float(np.nanmean(out["conc"]))
    return out


# ----------------------------------------------------------------------------------------------------- synthetic tests
def _synthetic_events(traj: Traj, n: int, rng: np.random.Generator, jitter: float = 0.0):
    """On-surface synthetic events: random visible front faces at random times; position = projected material point.
    Returns (xy float32, t_ms float64, face (n,), bary (n,3))."""
    g = geo()
    faces = g["faces"]
    ts = rng.uniform(traj.t0 + 2, traj.t0 + (traj.n - 1) * traj.dt - 2, n)
    xy, face, bary, keep_t = [], [], [], []
    for t in ts:
        k = int(round((t - traj.t0) / traj.dt))
        vis = traj.vis[k]
        fvv = vis[faces]
        uvk = traj.uv[k][faces]
        a, b, c = uvk[:, 0], uvk[:, 1], uvk[:, 2]
        area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
        cand = (fvv.all(1) & (area > 1.0)).nonzero().flatten()
        if len(cand) == 0:
            continue
        f = int(cand[rng.integers(len(cand))])
        w = rng.dirichlet([2.0, 2.0, 2.0]).astype(np.float32)
        # position at the exact time t (linear between samples)
        fi = (t - traj.t0) / traj.dt
        i0 = int(math.floor(fi))
        i1 = min(i0 + 1, traj.n - 1)
        al = fi - i0
        u = (1 - al) * traj.uv[i0][faces[f]] + al * traj.uv[i1][faces[f]]
        p = (torch.as_tensor(w)[:, None] * u).sum(0).numpy() + rng.normal(0, jitter, 2)
        xy.append(p.astype(np.float32))
        face.append(f)
        bary.append(w)
        keep_t.append(t)
    return np.stack(xy), np.asarray(keep_t), np.asarray(face), np.stack(bary)


def selftest(seq: str = "ycy_local", verbose: bool = True) -> bool:
    """Synthetic unit checks (no recorded events): every check returns pass / fail; nothing numeric is an effect."""
    rng = np.random.default_rng(SEED)
    rec = Rec(seq)
    ra, rb = (int(v) for v in rec.runs[np.argmax(rec.runs[:, 1] - rec.runs[:, 0])])
    # the 400 ms stretch of the longest run with the largest pose change (labels only: no events are read)
    cands = np.arange(ra + 50, rb - 460, 50)
    mot = [float(np.abs(np.asarray(rec.pos51[int(c) + 400]) - np.asarray(rec.pos51[int(c)])).sum()) for c in cands]
    t_lo = int(cands[int(np.argmax(mot))])
    t_hi = t_lo + 400
    traj = traj_from_rec(rec, t_lo, t_hi)
    res_motion = float((traj.uv[-1] - traj.uv[0]).norm(dim=-1).mean())
    res: Dict[str, bool] = {}
    # --- splat conservation and brute-force equality
    box = Box(0, 0, 12, 10)
    pts = np.stack([rng.uniform(1, 9, 50), rng.uniform(1, 7, 50)], 1).astype(np.float32)
    img, selfv = splat(pts, box)
    brute = np.zeros((10, 12))
    for (x, y) in pts.astype(np.float64):
        for gx in (math.floor(x), math.floor(x) + 1):
            for gy in (math.floor(y), math.floor(y) + 1):
                brute[gy, gx] += max(0.0, 1 - abs(x - gx)) * max(0.0, 1 - abs(y - gy))
    res["splat_mass_and_bruteforce"] = bool(abs(img.sum() - 50) < 1e-9 and np.abs(img - brute).max() < 1e-9)
    # --- contrast statistics: concentration of uniform vs piled-up points
    uni = np.stack([rng.uniform(0, 40, 4000), rng.uniform(0, 40, 4000)], 1).astype(np.float32)
    pile = uni.copy()
    pile[:2000] = pile[:2000] * 0.0 + 20.0 + rng.normal(0, 0.3, (2000, 2))
    b2 = Box(0, 0, 41, 41)
    cu, cp = contrast_stats(uni, b2), contrast_stats(pile, b2)
    res["contrast_orders"] = bool(cp["var"] > cu["var"] and cp["conc"] > cu["conc"] and 0.8 < cu["conc"] < 1.5)
    # --- assignment recovers faces of on-surface synthetic events; zero-time-offset warp is the identity
    xy, t_ev, face_true, bary_true = _synthetic_events(traj, 600, rng)
    ev_xy = xy
    asg = assign(traj, ev_xy, t_ev)
    res["assign_mostly_ok"] = bool(asg.ok.mean() > 0.97)
    # projected material point of the assignment at the event time should reproduce the event (small residual)
    res["assign_resid_small"] = bool(np.median(np.linalg.norm(asg.resid[asg.ok], axis=1)) < 0.75)
    d0 = displacement(traj, asg, t_ev, t_ev)
    res["warp_identity_at_own_time"] = bool(np.abs(d0).max() < 1e-4)
    # --- GT warp reconstructs the projected material point at the reference time (within the face ambiguity)
    t_ref = traj.t0 + (traj.n - 1) * traj.dt - 1.0
    d = displacement(traj, asg, t_ev, t_ref)
    xw = warp(ev_xy, d)
    g = geo()
    u_ref = _uv_face_at(traj, g["faces"][torch.as_tensor(face_true)], np.full(len(t_ev), t_ref))
    truth = (torch.as_tensor(bary_true)[..., None] * u_ref).sum(1).numpy()
    err = np.linalg.norm(xw - truth, axis=1)
    res["warp_recovers_material_point"] = bool(np.median(err[asg.ok]) < 1.0)
    # --- contrast: GT warp > identity warp; wrong (reversed) motion < GT
    allpts = np.concatenate([xw, truth])
    bx = box_from_points(allpts, 6)
    c_gt = contrast_stats(xw, bx)["var"]
    c_id = contrast_stats(ev_xy, bx)["var"]
    d_rev = -d
    c_rev = contrast_stats(warp(ev_xy, d_rev), bx)["var"]
    res["test_window_has_motion"] = bool(res_motion > 4.0)
    res["contrast_gt_beats_unwarped_and_reversed"] = bool(c_gt > c_id and c_gt > c_rev)
    # --- shuffled times (cluster) lower the contrast of the GT warp
    t_us = (t_ev * 1000).astype(np.int64)
    t_sh = shuffle_times(t_us, "cluster", np.random.default_rng(1)) / 1000.0
    asg_s = assign(traj, ev_xy, t_sh)
    xs = warp(ev_xy, displacement(traj, asg_s, t_sh, t_ref))
    res["contrast_time_shuffle_lower"] = bool(contrast_stats(xs, bx)["var"] < c_gt)
    # --- null generators preserve what they promise
    tt = np.repeat(np.arange(0, 400, 3), rng.integers(1, 6, len(np.arange(0, 400, 3)))).astype(np.int64) * 1000
    ct = shuffle_times(tt, "cluster", rng)
    ok_cluster = np.array_equal(np.sort(np.unique(ct, return_counts=True)[1]),
                                np.sort(np.unique(tt, return_counts=True)[1]))
    cc_same = all(len(np.unique(ct[tt == v])) == 1 for v in np.unique(tt)[:20])        # cluster stays intact
    it = shuffle_times(tt, "interval", rng, interval_us=16760.0)
    blk0, blk1 = np.floor(tt / 16760.0), np.floor(it / 16760.0)
    ok_interval = bool(np.array_equal(np.sort(it), np.sort(tt)) and
                       all(np.array_equal(np.sort(it[blk0 == b]), np.sort(tt[blk0 == b])) for b in np.unique(blk0)))
    wt = shuffle_times(tt, "window", rng)
    res["null_cluster_preserves_clusters"] = bool(ok_cluster and cc_same)
    res["null_interval_preserves_interval_multiset"] = ok_interval
    res["null_window_preserves_multiset"] = bool(np.array_equal(np.sort(wt), np.sort(tt)))
    # --- polarity swap: count-IWE invariant, signed-IWE sensitive only to the per-pixel version
    ev = Events(rng.integers(0, 240, 3000).astype(np.int16), rng.integers(0, 180, 3000).astype(np.int16),
                rng.integers(0, 2, 3000).astype(np.uint8), np.arange(3000, dtype=np.int64))
    e_g, e_h = swap_polarity(ev, "global", rng), swap_polarity(ev, "pixel_half", rng)
    xyz = ev.xy()
    bz = Box(0, 0, 90, 70)
    sg = lambda e: contrast_stats(xyz, bz, weights=np.where(e.p == 1, 1.0, -1.0))["var"]       # noqa: E731
    res["polarity_global_flip_invariant_signed"] = bool(abs(sg(ev) - sg(e_g)) < 1e-9)
    sgc = lambda e: contrast_stats(xyz, bz, weights=np.where(e.p == 1, 1.0, -1.0))["conc"]       # noqa: E731
    res["polarity_global_flip_invariant_signed_conc"] = bool(abs(sgc(ev) - sgc(e_g)) < 1e-9)
    res["signed_mass_is_abs_mass"] = bool(abs(contrast_stats(xyz, bz, weights=np.where(ev.p == 1, 1.0, -1.0))["n_in"] - contrast_stats(xyz, bz)["n_in"]) < 1e-6)
    res["polarity_count_invariant"] = bool(contrast_stats(xyz, bz)["var"] == contrast_stats(e_h.xy(), bz)["var"])
    res["polarity_pixel_half_changes_signed_only"] = bool(np.array_equal(ev.ex, e_h.ex) and (e_h.p != ev.p).any())
    # --- thinning / flash / noise controls keep counts
    res["thin_count"] = bool(len(thin_index(1000, 300, rng)) == 300 and len(thin_index(100, 300, rng)) == 100)
    fl = add_flash(ev, Box(60, 40, 100, 80), 0.2, rng)
    ns = uniform_noise_like(ev, Box(60, 40, 100, 80), rng)
    res["flash_noise_counts"] = bool(len(fl) == 3600 and len(ns) == 3000 and np.array_equal(ns.t_us, ev.t_us))
    # --- contour band: events placed on the projected silhouette are in the band, far ones are not
    kc = traj.n // 2
    ensure_contour(traj, [kc])
    a_, b_ = traj.contour[kc]
    if a_.shape[0] > 0:
        mid = (0.5 * (a_ + b_)).numpy().astype(np.float32)
        tc = np.full(len(mid), traj.t0 + kc * traj.dt)
        near, nrm, dist = contour_band(traj, mid, tc)
        far, _, _ = contour_band(traj, mid + 20.0, tc)
        res["contour_band_membership"] = bool(near.mean() > 0.98 and far.mean() < 0.9 and
                                              np.allclose(np.linalg.norm(nrm[near], axis=1), 1, atol=1e-4))
    else:
        res["contour_band_membership"] = False
    # --- collapse diagnostics: identity displacement has zero log-area change and unit cloud ratio
    st0 = warp_area_stats(traj, asg, t_ev, t_ev)
    res["area_identity"] = bool(abs(st0["mean_log_area"]) < 1e-5 and abs(cloud_area_ratio(ev_xy, ev_xy) - 1) < 1e-9)
    scaled = (ev_xy - ev_xy.mean(0)) * 0.5 + ev_xy.mean(0)
    res["cloud_area_ratio_contracts"] = bool(abs(cloud_area_ratio(ev_xy, scaled) - 0.25) < 1e-3)
    ok = all(res.values())
    if verbose:
        print("iwe selftest:", {k: ("pass" if v else "FAIL") for k, v in res.items()}, "ALL", "pass" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if selftest() else 1)
