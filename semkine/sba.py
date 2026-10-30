#!/usr/bin/env python3
r"""SBA: swept-boundary alignment of a MANO trajectory to an event window
(docs/EXPERIMENT_RECORDS.md#sba-prereg-20261022). Opt-in research code; nothing in the production path imports it.

The window `(m_prev, m1]` (ms) is treated as one realisation of a spatio-temporal count process on the 240 x 180
event grid. Its mean is the area swept, between consecutive time samples of the trajectory, by the model's own
visible occluding contours (silhouette and self-occlusion contour generators), plus background:

    x(alpha) = x0 + alpha (x1 - x0)                         (additive 51-D, alpha_s at 5 ms bin edges)
    edge e visible on the contour at sample s:  a, b -> a', b'  (projected endpoints, source pixels)
    d(t) = n_out . displacement, swept strip of area  int |d(t)| dl,  sign(d) > 0 = the occluder advances
    lambda_s(u, p) = kappa_sil A^sil_{s,p}(u) + kappa_int A^int_s(u) / 2 + background_s(u, p)
    l(x0, x1) = sum_{s,u,p} N log lambda - lambda          (Poisson; predicted sweep without events costs -lambda)

Strips are integrated with area elements and deposited bilinearly on the 640 x 480 source grid, then binned to
240 x 180 by the dataset's own rule `j = floor(3 i / 8)` (tools/prepare_hand_data.py), which reproduces the 3/3/2
pixel-gain pattern and the 0.44 px offset of the event grid. The pose update is coarse-to-fine Fisher scoring
with finite-difference Jacobians of the mean field (contour sets, visibility and element counts frozen at the
base trajectory of each iteration), Gaussian priors on the start knot, the increment and optional anchors.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

SRC_W, SRC_H = 640, 480
OUT_W, OUT_H = 240, 180
#: temporal bin of the count process (ms)
BIN_MS = 5
#: area-element spacing along the edge and along the sweep (source pixels) and their caps
SPACING = 1.0
MAX_NT, MAX_NR = 16, 48
#: vertex visibility: depth within this of the 3 x 3 vertex-splat z-buffer at its (output) pixel
VIS_TOL = 0.010
#: silhouette / internal test: probe this far (source px) along the outward normal; internal if a farther
#: hand surface is there (depth larger by > INT_DZ)
PROBE_SRC = 4.0
INT_DZ = 0.010
#: output pixels kept around the initial hand box (the likelihood window); outside only feeds the background rate
MARGIN_OUT = 28
#: forearm capsule in the wrist frame (m): starts this far towards the palm, length, radius
CAP_BACK, CAP_LEN, CAP_RAD = 0.01, 0.30, 0.045
#: background floor (events per source pixel per window)
B0_FLOOR = 1e-5
#: finite-difference steps (translation m, rotations rad) and per-iteration trust caps
FD_T, FD_R = 1e-3, 2e-3
CAP_T, CAP_R = 0.02, 0.10
#: Levenberg-Marquardt (damping relative to the Hessian diagonal): initial value, decrease after an accepted
#: step, floor, and escalations (x10) per iteration before the iteration is declared stalled
LM_MU0, LM_DOWN, LM_MU_MIN, LM_TRIES = 1.0, 0.3, 1e-3, 4
#: coarse-to-fine output-pixel pooling per Gauss-Newton iteration
POOLS = (8, 4, 2, 1, 1)
CH_SIL_POS, CH_SIL_NEG, CH_INT = 0, 1, 2
#: knuckles (MCP) of index, middle, ring, little in ManoLayer's joints21, which are in OpenPose order
#: (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring, 17-20 little)
MCP_JOINTS = (5, 9, 13, 17)
#: erratum (2026-10-22): the S0-P0 run used (1, 4, 7, 10) = thumb CMC, thumb tip, index DIP, middle PIP; kept to
#: reproduce outputs/sba/s0 and outputs/sba/p0
MCP_JOINTS_LEGACY = (1, 4, 7, 10)


# ---------------------------------------------------------------------------------------------------- grid
@dataclass
class Grid:
    lut_x: torch.Tensor       # (640,) long: output column of each source column
    lut_y: torch.Tensor       # (480,)
    gain: torch.Tensor        # (180, 240) float: source pixels per output pixel (9 / 6 / 4)
    cx_src: torch.Tensor      # (240,) float: centre of each output column in source coordinates
    cy_src: torch.Tensor      # (180,)


def make_grid(device) -> Grid:
    ix, iy = torch.arange(SRC_W, device=device), torch.arange(SRC_H, device=device)
    lx, ly = (ix * OUT_W) // SRC_W, (iy * OUT_H) // SRC_H
    cnt_x = torch.bincount(lx, minlength=OUT_W).float()
    cnt_y = torch.bincount(ly, minlength=OUT_H).float()
    cx = torch.zeros(OUT_W, device=device).index_add_(0, lx, ix.float()) / cnt_x
    cy = torch.zeros(OUT_H, device=device).index_add_(0, ly, iy.float()) / cnt_y
    return Grid(lx, ly, cnt_y[:, None] * cnt_x[None, :], cx, cy)


# ---------------------------------------------------------------------------------------------------- topology
class Topology:
    """Undirected edges of the MANO mesh with their (<= 2) adjacent faces and the faces' third vertices."""

    def __init__(self, faces: torch.Tensor):
        f = faces.long()
        dev = f.device
        F = f.shape[0]
        he = torch.cat([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])                 # (3F, 2) half edges
        third = torch.cat([f[:, 2], f[:, 0], f[:, 1]])
        fid = torch.arange(F, device=dev).repeat(3)
        lo, hi = he.min(1).values, he.max(1).values
        key = lo * 100000 + hi
        uniq, inv = torch.unique(key, return_inverse=True)
        E = uniq.numel()
        order = torch.argsort(inv, stable=True)
        inv_s = inv[order]
        first = torch.ones_like(inv_s, dtype=torch.bool)
        first[1:] = inv_s[1:] != inv_s[:-1]
        slot = (~first).long()
        if bool(((inv_s[2:] == inv_s[:-2])).any()):
            raise ValueError("non-manifold edge with more than two faces")
        edge_f = torch.full((E, 2), -1, dtype=torch.long, device=dev)
        edge_c = torch.full((E, 2), -1, dtype=torch.long, device=dev)
        edge_f[inv_s, slot] = fid[order]
        edge_c[inv_s, slot] = third[order]
        self.faces = f
        self.edges = torch.stack([uniq // 100000, uniq % 100000], 1)               # (E, 2)
        self.edge_f, self.edge_c = edge_f, edge_c
        self.boundary = edge_f[:, 1] < 0

    @property
    def n_edges(self) -> int:
        return int(self.edges.shape[0])


def project_src(verts: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """verts (..., V, 3) m, K (3, 3) of the 640 x 480 camera -> (..., V, 2) source pixel coordinates."""
    z = verts[..., 2].clamp(min=1e-6)
    return torch.stack([K[0, 0] * verts[..., 0] / z + K[0, 2], K[1, 1] * verts[..., 1] / z + K[1, 2]], -1)


def splat_zbuffer(uv_src: torch.Tensor, z: torch.Tensor, groups: Optional[torch.Tensor] = None,
                  n_groups: int = 1) -> torch.Tensor:
    """(P, V, 2) source px, (P, V) depth -> (P, G, OUT_H * OUT_W) min depth of a 3 x 3 vertex splat per part group
    on the output grid (inf where no vertex). The splat is on the renderer's grid (source coordinates x 0.375)."""
    P, V = uv_src.shape[:2]
    dev = uv_src.device
    HW = OUT_H * OUT_W
    u = (uv_src[..., 0] * (OUT_W / SRC_W)).round().long()
    v = (uv_src[..., 1] * (OUT_H / SRC_H)).round().long()
    gr = torch.zeros(V, dtype=torch.long, device=dev) if groups is None else groups.long()
    zb = torch.full((P * n_groups * HW,), float("inf"), dtype=z.dtype, device=dev)
    pb = (torch.arange(P, device=dev)[:, None] * n_groups + gr[None, :]) * HW
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            uu, vv = u + dx, v + dy
            ok = (uu >= 0) & (uu < OUT_W) & (vv >= 0) & (vv < OUT_H) & (z > 0)
            zb.scatter_reduce_(0, (pb + vv * OUT_W + uu)[ok], z[ok], reduce="amin", include_self=True)
    return zb.view(P, n_groups, HW)


def _out_index(xy_src: torch.Tensor, grid: Grid):
    """Nearest source pixel -> output pixel index (flat) and validity."""
    xi, yi = (xy_src[..., 0] + 0.5).floor().long(), (xy_src[..., 1] + 0.5).floor().long()
    ok = (xi >= 0) & (xi < SRC_W) & (yi >= 0) & (yi < SRC_H)
    j = grid.lut_y[yi.clamp(0, SRC_H - 1)] * OUT_W + grid.lut_x[xi.clamp(0, SRC_W - 1)]
    return j, ok


@torch.no_grad()
def contour_entries(verts: torch.Tensor, uv: torch.Tensor, topo: Topology, grid: Grid,
                    groups: Optional[torch.Tensor] = None, n_groups: int = 1):
    """Visible occluding-contour edges of each pose.

    verts (P, V, 3), uv (P, V, 2) source px. Returns on (P, E) bool, side (P, E) float (+-1, orientation of the
    outward normal relative to rot90(b - a)), internal (P, E) bool (a farther hand surface lies just outside).
    An edge is on the occluding contour when its two faces face opposite ways in image space; the MANO wrist cut
    (boundary edges) is not a physical boundary and is excluded. A contour vertex is hidden only by another part
    group in front of it (`groups` (V,) long): a convex part never hides its own rim, whereas a same-part splat
    test does, because the surface just inside the rim is nearer than any fixed tolerance.
    """
    f = topo.faces
    tri = uv[:, f]                                                        # (P, F, 3, 2)
    a, b, c = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
    area = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    front = area > 0
    f0, f1 = topo.edge_f[:, 0], topo.edge_f[:, 1]
    on = (~topo.boundary)[None] & (front[:, f0] != front[:, f1.clamp(min=0)])
    z = verts[..., 2]
    P, V = z.shape
    gr = torch.zeros(V, dtype=torch.long, device=z.device) if groups is None else groups.long()
    zbg = splat_zbuffer(uv, z, gr, n_groups)                              # (P, G, HW)
    zb = zbg.amin(1)                                                      # (P, HW)
    u = (uv[..., 0] * (OUT_W / SRC_W)).round().long()
    v = (uv[..., 1] * (OUT_H / SRC_H)).round().long()
    inb = (u >= 0) & (u < OUT_W) & (v >= 0) & (v < OUT_H) & (z > 0)
    pix = (v.clamp(0, OUT_H - 1) * OUT_W + u.clamp(0, OUT_W - 1))         # (P, V)
    zz = zbg.gather(2, pix[:, None, :].expand(P, n_groups, V))            # (P, G, V)
    own = torch.nn.functional.one_hot(gr, n_groups).bool().T[None]        # (1, G, V)
    zz = torch.where(own, torch.full_like(zz, float("inf")), zz).amin(1)  # nearest other-part depth
    vis = inb & (z <= zz + VIS_TOL)                                       # (P, V)
    e0, e1 = topo.edges[:, 0], topo.edges[:, 1]
    on &= vis[:, e0] & vis[:, e1]
    pa, pb = uv[:, e0], uv[:, e1]
    pc = uv[:, topo.edge_c[:, 0]]                                         # third vertex of either face (same side)
    ev = pb - pa
    cross = ev[..., 0] * (pc[..., 1] - pa[..., 1]) - ev[..., 1] * (pc[..., 0] - pa[..., 0])
    side = torch.where(cross >= 0, torch.ones_like(cross), -torch.ones_like(cross))
    L = ev.norm(dim=-1).clamp(min=1e-9)
    nrm = -side[..., None] * torch.stack([-ev[..., 1], ev[..., 0]], -1) / L[..., None]
    mid = 0.5 * (pa + pb)
    j, ok = _out_index(mid + PROBE_SRC * nrm, grid)
    zq = zb.gather(1, j.clamp(0, OUT_H * OUT_W - 1))
    zm = 0.5 * (z[:, e0] + z[:, e1])
    internal = ok & torch.isfinite(zq) & (zq > zm + INT_DZ)
    return on, side, internal


# ---------------------------------------------------------------------------------------------------- plan
@dataclass
class Plan:
    """Frozen sweep structure of one trajectory: contour edges per segment and their area elements."""
    seg: torch.Tensor         # (n_e,) segment s (strip from sample s to s+1)
    e0: torch.Tensor          # (n_e,) vertex ids
    e1: torch.Tensor
    side: torch.Tensor        # (n_e,) +-1
    internal: torch.Tensor    # (n_e,) bool
    nt: torch.Tensor          # (n_e,) elements along the edge
    nr: torch.Tensor          # (n_e,) elements along the sweep
    el: torch.Tensor          # (n_el,) entry of each element
    t: torch.Tensor           # (n_el,)
    r: torch.Tensor           # (n_el,)


@torch.no_grad()
def build_plan(verts: torch.Tensor, uv: torch.Tensor, topo: Topology, grid: Grid,
               groups: Optional[torch.Tensor] = None, n_groups: int = 1) -> Plan:
    """verts (S+1, V, 3), uv (S+1, V, 2) of one trajectory (S segments)."""
    S = verts.shape[0] - 1
    dev = uv.device
    on, side, internal = contour_entries(verts[:S], uv[:S], topo, grid, groups, n_groups)  # set at segment start
    s_idx, e_idx = on.nonzero(as_tuple=True)
    e0, e1 = topo.edges[e_idx, 0], topo.edges[e_idx, 1]
    sd = side[s_idx, e_idx]
    a, b = uv[s_idx, e0], uv[s_idx, e1]
    a1, b1 = uv[s_idx + 1, e0], uv[s_idx + 1, e1]
    ev = b - a
    L = ev.norm(dim=-1).clamp(min=1e-9)
    nrm = -sd[:, None] * torch.stack([-ev[:, 1], ev[:, 0]], -1) / L[:, None]
    da, db = (nrm * (a1 - a)).sum(-1), (nrm * (b1 - b)).sum(-1)
    nt = (L / SPACING).ceil().clamp(1, MAX_NT).long()
    nr = (torch.maximum(da.abs(), db.abs()) / SPACING).ceil().clamp(1, MAX_NR).long()
    n_per = nt * nr
    el = torch.repeat_interleave(torch.arange(len(nt), device=dev), n_per)
    start = torch.cumsum(n_per, 0) - n_per
    k = torch.arange(len(el), device=dev) - start[el]
    it, ir = k // nr[el], k % nr[el]
    t = (it.float() + 0.5) / nt[el].float()
    r = (ir.float() + 0.5) / nr[el].float()
    return Plan(s_idx, e0, e1, sd, internal[s_idx, e_idx], nt, nr, el, t, r)


@dataclass
class Crop:
    y0: int
    x0: int
    h: int
    w: int

    @property
    def cells(self) -> int:
        return self.h * self.w


def crop_from(uv: torch.Tensor, margin: int = MARGIN_OUT) -> Crop:
    """Output-pixel box around all projected vertices of `uv` (..., V, 2 source px) plus `margin`."""
    xy = uv.reshape(-1, 2)
    xs, ys = xy[:, 0] * (OUT_W / SRC_W), xy[:, 1] * (OUT_H / SRC_H)
    x0 = int(max(0, math.floor(float(xs.min())) - margin))
    x1 = int(min(OUT_W, math.ceil(float(xs.max())) + margin + 1))
    y0 = int(max(0, math.floor(float(ys.min())) - margin))
    y1 = int(min(OUT_H, math.ceil(float(ys.max())) + margin + 1))
    return Crop(y0, x0, max(1, y1 - y0), max(1, x1 - x0))


def sweep_area(uvh: torch.Tensor, plan: Plan, grid: Grid, crop: Crop, S: int) -> torch.Tensor:
    """Swept area of the plan's edges under each hypothesis trajectory.

    uvh (H, S+1, V, 2) source px. Returns A (H, 3, S, crop.h, crop.w): channels silhouette advancing (d > 0),
    silhouette retreating, internal (unsigned); units = source pixels^2 (so an output pixel collects its 9 / 6 / 4
    source pixels, the dataset's binning gain)."""
    H = uvh.shape[0]
    dev = uvh.device
    out = torch.zeros(H * 3 * S * crop.cells, dtype=uvh.dtype, device=dev)
    if plan.el.numel() == 0:
        return out.view(H, 3, S, crop.h, crop.w)
    a = uvh[:, plan.seg, plan.e0]
    b = uvh[:, plan.seg, plan.e1]
    a1 = uvh[:, plan.seg + 1, plan.e0]
    b1 = uvh[:, plan.seg + 1, plan.e1]
    ev = b - a
    L = ev.norm(dim=-1).clamp(min=1e-9)
    nrm = -plan.side[None, :, None] * torch.stack([-ev[..., 1], ev[..., 0]], -1) / L[..., None]
    da, db = (nrm * (a1 - a)).sum(-1), (nrm * (b1 - b)).sum(-1)
    el = plan.el
    t, r = plan.t[None], plan.r[None]
    d = da[:, el] + t * (db[:, el] - da[:, el])
    q = a[:, el] + t[..., None] * ev[:, el] + (r * d)[..., None] * nrm[:, el]
    dA = (L[:, el] / plan.nt[el].to(L.dtype)) * d.abs() / plan.nr[el].to(L.dtype)
    ch = torch.where(plan.internal[el][None], torch.full_like(d, CH_INT, dtype=torch.long),
                     torch.where(d > 0, torch.full_like(d, CH_SIL_POS, dtype=torch.long),
                                 torch.full_like(d, CH_SIL_NEG, dtype=torch.long)))
    s = plan.seg[el][None].expand_as(d)
    hb = torch.arange(H, device=dev)[:, None].expand_as(d)
    base = ((hb * 3 + ch) * S + s) * crop.cells
    xf, yf = q[..., 0].floor(), q[..., 1].floor()
    wx, wy = q[..., 0] - xf, q[..., 1] - yf
    xi, yi = xf.long(), yf.long()
    for ox, oy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        xx, yy = xi + ox, yi + oy
        ok = (xx >= 0) & (xx < SRC_W) & (yy >= 0) & (yy < SRC_H)
        jx = grid.lut_x[xx.clamp(0, SRC_W - 1)] - crop.x0
        jy = grid.lut_y[yy.clamp(0, SRC_H - 1)] - crop.y0
        ok &= (jx >= 0) & (jx < crop.w) & (jy >= 0) & (jy < crop.h)
        out.index_add_(0, (base + jy * crop.w + jx)[ok], (dA * w)[ok])
    return out.view(H, 3, S, crop.h, crop.w)


# ---------------------------------------------------------------------------------------------------- window
@dataclass
class Window:
    """Events and nuisance rates of one window, on the crop. Times: knots at m_prev (x0) and m1 (x1)."""
    alphas: torch.Tensor      # (S+1,) trajectory parameter of the bin edges
    widths: torch.Tensor      # (S,) bin durations (ms)
    N: torch.Tensor           # (2, S, h, w) counts per polarity (0 = OFF, 1 = ON as stored)
    bkg: torch.Tensor         # (S, h, w) expected background events (both polarities)
    on_frac: float            # ON fraction of the background
    crop: Crop
    n_events: int
    n_crop: int
    b0: float
    b_fa: float
    #: (S, h, w) uniform in-crop field carrying the window's crop events (gain and bin widths): the outlier
    #: component is `Model.outlier` times this (unmodelled hand events: creases, shading, contour misfit)
    uni: Optional[torch.Tensor] = None


def bins_of(m_prev: int, m1: int):
    W = m1 - m_prev
    S = max(1, math.ceil(W / BIN_MS))
    edges = [m_prev + s * BIN_MS for s in range(S)] + [m1]
    alphas = [(e - m_prev) / W for e in edges]
    return S, edges, alphas


def capsule_mask(joints: torch.Tensor, K: torch.Tensor, grid: Grid, mcp: Sequence[int] = MCP_JOINTS) -> torch.Tensor:
    """(OUT_H, OUT_W) bool forearm capsule of one pose (joints (21, 3) m, OpenPose order)."""
    w = joints[0]
    d = w - joints[list(mcp)].mean(0)
    d = d / d.norm().clamp(min=1e-9)
    P = torch.stack([w - CAP_BACK * d, w + CAP_LEN * d])
    p = project_src(P, K)
    zc = w[2].clamp(min=0.05)
    rad = float(K[0, 0]) * CAP_RAD / float(zc)
    cy, cx = grid.cy_src[:, None], grid.cx_src[None, :]
    ab = p[1] - p[0]
    tt = (((cx - p[0, 0]) * ab[0] + (cy - p[0, 1]) * ab[1]) / (ab @ ab).clamp(min=1e-9)).clamp(0, 1)
    dx, dy = cx - (p[0, 0] + tt * ab[0]), cy - (p[0, 1] + tt * ab[1])
    return (dx * dx + dy * dy) <= rad * rad


def make_window(ev_xyp: np.ndarray, ev_ms: np.ndarray, m_prev: int, m1: int, crop: Crop, grid: Grid,
                cap: torch.Tensor, hand_box: Crop, device) -> Window:
    """ev_xyp (n, 3) uint8 output-grid events with their ms (n,), restricted to (m_prev, m1]."""
    S, edges, alphas = bins_of(m_prev, m1)
    W = m1 - m_prev
    widths = torch.tensor(np.diff(edges), dtype=torch.float32, device=device)
    full = torch.zeros(2, S, OUT_H, OUT_W, dtype=torch.float32, device=device)
    n = len(ev_ms)
    if n:
        x = torch.from_numpy(ev_xyp[:, 0].astype(np.int64)).to(device)
        y = torch.from_numpy(ev_xyp[:, 1].astype(np.int64)).to(device)
        p = torch.from_numpy(np.clip(ev_xyp[:, 2], 0, 1).astype(np.int64)).to(device)
        s = torch.from_numpy(np.minimum((ev_ms.astype(np.int64) - m_prev - 1) // BIN_MS, S - 1)).to(device)
        full.view(-1).index_add_(0, ((p * S + s) * OUT_H + y) * OUT_W + x, torch.ones(n, device=device))
    tot = full.sum((0, 1))                                                  # (H, W)
    near = torch.zeros(OUT_H, OUT_W, dtype=torch.bool, device=device)
    near[crop.y0:crop.y0 + crop.h, crop.x0:crop.x0 + crop.w] = True
    far = ~(near | cap)
    g = grid.gain
    b0 = float(tot[far].sum() / g[far].sum().clamp(min=1.0)) if bool(far.any()) else B0_FLOOR
    b0 = max(b0, B0_FLOOR)
    hb = torch.zeros_like(near)
    hb[hand_box.y0:hand_box.y0 + hand_box.h, hand_box.x0:hand_box.x0 + hand_box.w] = True
    cap_only = cap & ~hb
    b_fa = 0.0
    if bool(cap_only.any()):
        b_fa = max(0.0, float(tot[cap_only].sum() / g[cap_only].sum()) - b0)
    on_tot = float(full[1].sum())
    on_frac = on_tot / max(1.0, float(full.sum())) if n else 0.5
    on_frac = min(max(on_frac, 0.05), 0.95)
    sl = (slice(crop.y0, crop.y0 + crop.h), slice(crop.x0, crop.x0 + crop.w))
    rate = (b0 + b_fa * cap[sl].float()) * g[sl]                            # per window
    bkg = rate[None] * (widths / W)[:, None, None]
    Nc = full[:, :, sl[0], sl[1]].contiguous()
    shape = g[sl][None] * (widths / W)[:, None, None]
    uni = shape / shape.sum().clamp(min=1e-12) * float(Nc.sum())
    return Window(torch.tensor(alphas, dtype=torch.float32, device=device), widths, Nc, bkg, on_frac, crop, n,
                  int(Nc.sum()), b0, b_fa, uni)


# ---------------------------------------------------------------------------------------------------- likelihood
@dataclass
class Model:
    kappa_sil: float
    kappa_int: float
    signed: bool = False
    dispersion: float = 1.0
    #: Gaussian blur (output px) of the swept-area field: the contour misfit the model tolerates
    blur: float = 0.0
    #: re-estimate a per-window scale of (kappa_sil, kappa_int) at each solver iteration's base trajectory
    profile: bool = False
    #: outlier fraction: expected events of a uniform in-crop component, as a fraction of the window's crop events
    outlier: float = 0.0


def blur_area(A: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of the last two dims (zero padding, mass kept inside the crop)."""
    if sigma <= 0:
        return A
    r = max(1, int(math.ceil(3 * sigma)))
    k = torch.exp(-0.5 * (torch.arange(-r, r + 1, device=A.device, dtype=A.dtype) / sigma) ** 2)
    k = k / k.sum()
    sh = A.shape
    x = A.reshape(-1, 1, sh[-2], sh[-1])
    x = torch.nn.functional.conv2d(x, k.view(1, 1, 1, -1), padding=(0, r))
    x = torch.nn.functional.conv2d(x, k.view(1, 1, -1, 1), padding=(r, 0))
    return x.view(sh)


def rates(A: torch.Tensor, win: Window, model: Model, sign: int = 1) -> torch.Tensor:
    """A (H, 3, S, h, w) -> lambda (H, 2, S, h, w) if signed else (H, 1, S, h, w)."""
    A = blur_area(A, model.blur)
    ks, ki = model.kappa_sil, model.kappa_int
    bkg = win.bkg if (model.outlier <= 0 or win.uni is None) else win.bkg + model.outlier * win.uni
    if not model.signed:
        lam = ks * (A[:, CH_SIL_POS] + A[:, CH_SIL_NEG]) + ki * A[:, CH_INT] + bkg[None]
        return lam[:, None]
    pos, neg = (A[:, CH_SIL_POS], A[:, CH_SIL_NEG]) if sign > 0 else (A[:, CH_SIL_NEG], A[:, CH_SIL_POS])
    on = ks * pos + 0.5 * ki * A[:, CH_INT] + win.on_frac * bkg[None]
    off = ks * neg + 0.5 * ki * A[:, CH_INT] + (1 - win.on_frac) * bkg[None]
    return torch.stack([off, on], 1)                                        # polarity index 0 = OFF, 1 = ON


def counts(win: Window, model: Model) -> torch.Tensor:
    return win.N.sum(0, keepdim=True) if not model.signed else win.N


def pool(x: torch.Tensor, p: int) -> torch.Tensor:
    """Sum-pool the last two dims by p (zero padding)."""
    if p == 1:
        return x
    h, w = x.shape[-2:]
    ph, pw = (-h) % p, (-w) % p
    if ph or pw:
        x = torch.nn.functional.pad(x, (0, pw, 0, ph))
    sh = x.shape
    x = x.reshape(*sh[:-2], sh[-2] // p, p, sh[-1] // p, p)
    return x.sum((-3, -1))


def loglik(lam: torch.Tensor, N: torch.Tensor, p: int = 1) -> torch.Tensor:
    """Poisson log-likelihood up to constants, per hypothesis: lam (H, ...), N (...)."""
    lp, Np = pool(lam, p), pool(N, p)
    return (Np[None] * torch.log(lp.clamp(min=1e-12))).flatten(1).sum(1) - lp.flatten(1).sum(1)


def profile_kappa(A: torch.Tensor, win: Window, model: Model, sign: int = 1, iters: int = 30) -> torch.Tensor:
    """Per-hypothesis scale k on (kappa_sil, kappa_int) maximising the likelihood (the kappa-free variant)."""
    one = replace(model, profile=False)
    lam_h = rates(A, win, one, sign) - rates(torch.zeros_like(A[:1]), win, one, sign)
    bk = rates(torch.zeros_like(A[:1]), win, one, sign)
    N = counts(win, model)
    k = torch.ones(A.shape[0], dtype=A.dtype, device=A.device)
    Ht = lam_h.flatten(1)
    Bt = bk.flatten(1)
    Nf = N.flatten()[None]
    for _ in range(iters):
        lam = k[:, None] * Ht + Bt
        g1 = (Nf * Ht / lam.clamp(min=1e-12)).sum(1) - Ht.sum(1)
        g2 = -(Nf * Ht * Ht / lam.clamp(min=1e-12) ** 2).sum(1)
        step = torch.where(g2 < -1e-12, -g1 / g2, torch.zeros_like(g1))
        k = (k + step).clamp(min=1e-4, max=1e4)
    return k


# ---------------------------------------------------------------------------------------------------- geometry
class Geometry:
    """MANO FK (float32 ManoLayer with the evaluator's decoding) for trajectories, plus topology and grid."""

    def __init__(self, mano, device):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
        from pose_repr import decode_to_mano_inputs
        self._dec = decode_to_mano_inputs
        self.mano = mano
        self.device = device
        self.topo = Topology(mano.f.to(device))
        self.grid = make_grid(device)
        #: part groups for occlusion: palm, index, middle, pinky, ring, thumb (dominant LBS joint)
        dom = mano.weights.to(device).argmax(1)
        self.groups = torch.tensor([0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5], device=device)[dom]
        self.n_groups = 6

    def fk(self, params: torch.Tensor, betas: torch.Tensor, chunk: int = 4096):
        vs, js = [], []
        for i in range(0, len(params), chunk):
            p = params[i:i + chunk].float()
            dec = self._dec(p, "mano_full_axis_angle", self.mano.hands_components, self.mano.hands_mean)
            v, j = self.mano(betas.float().expand(len(p), -1), dec["global_orient"], dec["local_full_aa"],
                             dec["transl"])
            vs.append(v)
            js.append(j)
        return torch.cat(vs), torch.cat(js)

    def trajectories(self, x0: torch.Tensor, x1: torch.Tensor, alphas: torch.Tensor, betas, K):
        """x0, x1 (H, 51) -> verts (H, S+1, V, 3), uv (H, S+1, V, 2), joints (H, S+1, 21, 3)."""
        H = x0.shape[0]
        X = x0[:, None] + alphas[None, :, None].to(x0.dtype) * (x1 - x0)[:, None]
        v, j = self.fk(X.reshape(-1, 51), betas)
        S1 = alphas.numel()
        v = v.view(H, S1, -1, 3)
        return v, project_src(v, K), j.view(H, S1, -1, 3)


# ---------------------------------------------------------------------------------------------------- solver
@dataclass
class Priors:
    """Gaussian factors in additive 51-D coordinates (precisions)."""
    inc_prec: torch.Tensor                       # (51, 51) precision of the window increment x1 - x0
    start_mean: Optional[torch.Tensor] = None    # (51,) prior mean of x0; None = x0 fixed
    start_prec: Optional[torch.Tensor] = None    # (51,) diagonal precision of x0
    end_mean: Optional[torch.Tensor] = None      # (51,) optional anchor on x1 (e.g. S39's output)
    end_prec: Optional[torch.Tensor] = None      # (51,) diagonal
    pose_mean: Optional[torch.Tensor] = None     # (45,) absolute finger prior on x1[6:]
    pose_prec: Optional[torch.Tensor] = None     # (45, 45)

    def value_grad_hess(self, x0, x1):
        """Negative log prior (scalar), gradient and Hessian w.r.t. the free vector [x0 (if free), x1]."""
        dev, dt = x1.device, torch.float64
        free0 = self.start_mean is not None
        D = 102 if free0 else 51
        g = torch.zeros(D, dtype=dt, device=dev)
        Hm = torch.zeros(D, D, dtype=dt, device=dev)
        val = torch.zeros((), dtype=dt, device=dev)
        x0d, x1d = x0.double(), x1.double()
        o1 = 51 if free0 else 0
        P = self.inc_prec.double()
        inc = x1d - x0d
        val = val + 0.5 * inc @ P @ inc
        g[o1:o1 + 51] += P @ inc
        Hm[o1:o1 + 51, o1:o1 + 51] += P
        if free0:
            g[:51] -= P @ inc
            Hm[:51, :51] += P
            Hm[:51, 51:] -= P
            Hm[51:, :51] -= P
            dx = x0d - self.start_mean.double()
            pr = self.start_prec.double()
            val = val + 0.5 * (pr * dx * dx).sum()
            g[:51] += pr * dx
            Hm[:51, :51] += torch.diag(pr)
        if self.end_mean is not None:
            dx = x1d - self.end_mean.double()
            pr = self.end_prec.double()
            val = val + 0.5 * (pr * dx * dx).sum()
            g[o1:o1 + 51] += pr * dx
            Hm[o1:o1 + 51, o1:o1 + 51] += torch.diag(pr)
        if self.pose_mean is not None:
            dx = x1d[6:] - self.pose_mean.double()
            Pp = self.pose_prec.double()
            val = val + 0.5 * dx @ Pp @ dx
            g[o1 + 6:o1 + 51] += Pp @ dx
            Hm[o1 + 6:o1 + 51, o1 + 6:o1 + 51] += Pp
        return val, g, Hm


def fd_steps(device) -> torch.Tensor:
    e = torch.full((51,), FD_R, device=device)
    e[:3] = FD_T
    return e


@dataclass
class SolveResult:
    x0: torch.Tensor
    x1: torch.Tensor
    loglik: float
    fisher: Optional[torch.Tensor]
    iters: int
    accepted: int
    sign: int
    info: Dict = field(default_factory=dict)


class Evaluator:
    """Mean field and likelihood of trajectories for one window (plan frozen at a base trajectory)."""

    def __init__(self, geo: Geometry, win: Window, model: Model, betas, K, chunk: int = 64):
        self.geo, self.win, self.model, self.betas, self.K = geo, win, model, betas, K
        self.chunk = chunk
        self.S = int(win.widths.numel())

    def plan_at(self, x0: torch.Tensor, x1: torch.Tensor) -> Plan:
        v, uv, _ = self.geo.trajectories(x0[None], x1[None], self.win.alphas, self.betas, self.K)
        return build_plan(v[0], uv[0], self.geo.topo, self.geo.grid, self.geo.groups, self.geo.n_groups)

    def areas(self, X0: torch.Tensor, X1: torch.Tensor, plan: Plan) -> torch.Tensor:
        outs = []
        for i in range(0, X0.shape[0], self.chunk):
            _, uv, _ = self.geo.trajectories(X0[i:i + self.chunk], X1[i:i + self.chunk], self.win.alphas,
                                             self.betas, self.K)
            outs.append(sweep_area(uv, plan, self.geo.grid, self.win.crop, self.S))
        return torch.cat(outs)

    def best_sign(self, A_base: torch.Tensor, p: int) -> int:
        if not self.model.signed:
            return 1
        N = counts(self.win, self.model)
        lp = loglik(rates(A_base, self.win, self.model, 1), N, p)
        ln = loglik(rates(A_base, self.win, self.model, -1), N, p)
        return 1 if float(lp[0]) >= float(ln[0]) else -1

    def loglik(self, A: torch.Tensor, p: int, sign: int, model: Optional[Model] = None) -> torch.Tensor:
        m = self.model if model is None else model
        return loglik(rates(A, self.win, m, sign), counts(self.win, m), p) / m.dispersion


def solve(ev: Evaluator, x0: torch.Tensor, x1: torch.Tensor, priors: Priors, pools: Sequence[int] = POOLS,
          damping: float = LM_MU0, fixed_sign: Optional[int] = None) -> SolveResult:
    """Coarse-to-fine Fisher scoring of the posterior over [x0 (if its prior is given), x1]."""
    dev = x1.device
    free0 = priors.start_mean is not None
    eps = fd_steps(dev).to(x1.dtype)
    D = 102 if free0 else 51
    x0, x1 = x0.clone().float(), x1.clone().float()
    accepted, sign = 0, 1
    fisher = None
    ll = float("nan")
    for it, p in enumerate(pools):
        plan = ev.plan_at(x0, x1)
        I = torch.eye(51, device=dev) * eps[None]
        if free0:
            X0 = torch.cat([x0[None], x0[None] + I, x0[None].expand(51, -1), x0[None] - I, x0[None].expand(51, -1)])
            X1 = torch.cat([x1[None], x1[None].expand(51, -1), x1[None] + I, x1[None].expand(51, -1), x1[None] - I])
        else:
            X0 = x0[None].expand(1 + 2 * 51, -1)
            X1 = torch.cat([x1[None], x1[None] + I, x1[None] - I])
        A = ev.areas(X0, X1, plan)
        sign = fixed_sign if fixed_sign is not None else ev.best_sign(A[:1], p)
        mod = ev.model
        if mod.profile:
            k = float(profile_kappa(A[:1], ev.win, mod, sign)[0])
            mod = replace(mod, kappa_sil=mod.kappa_sil * k, kappa_int=mod.kappa_int * k, profile=False)
        lam = pool(rates(A, ev.win, mod, sign), p).flatten(1).double()           # (1 + 2D, M)
        Np = pool(counts(ev.win, mod), p).flatten().double()
        lam0 = lam[0].clamp(min=1e-12)
        epsD = torch.cat([eps, eps]) if free0 else eps
        J = (lam[1:1 + D] - lam[1 + D:]) / (2 * epsD.double()[:, None])           # (D, M)
        disp = mod.dispersion
        g = (J @ (Np / lam0 - 1.0)) / disp
        F = (J / lam0[None]) @ J.T / disp
        ll = float((Np * torch.log(lam0) - lam0).sum() / disp)
        pv, pg, ph = priors.value_grad_hess(x0, x1)
        Hm0 = F + ph
        diag = torch.diag(Hm0.diagonal().clamp(min=1e-9))
        obj0 = ll - float(pv)
        ok = False
        mu = damping
        for _ in range(LM_TRIES):
            step = _cap(torch.linalg.solve(Hm0 + mu * diag, g - pg), free0).float()
            nx0 = x0 + step[:51] if free0 else x0
            nx1 = x1 + step[-51:]
            # the trial is judged by the true objective: contour set and visibility re-planned at the trial state
            An = ev.areas(nx0[None], nx1[None], ev.plan_at(nx0, nx1))
            ln = float(ev.loglik(An, p, sign, mod)[0])
            pvn, _, _ = priors.value_grad_hess(nx0, nx1)
            if ln - float(pvn) > obj0:
                x0, x1 = nx0, nx1
                ok = True
                accepted += 1
                ll = ln
                damping = max(mu * LM_DOWN, LM_MU_MIN)
                break
            mu *= 10.0
        if not ok:
            damping = mu
        if p == 1:
            fisher = F
        if not ok and p == 1:
            break
    return SolveResult(x0, x1, ll, fisher, len(pools), accepted, sign)


def _cap(step: torch.Tensor, free0: bool) -> torch.Tensor:
    parts = [step[:51], step[51:]] if free0 else [step]
    outs = []
    for s in parts:
        s = s.clone()
        t = s[:3].norm()
        if float(t) > CAP_T:
            s[:3] *= CAP_T / float(t)
        rot = s[3:].view(16, 3)
        n = rot.norm(dim=1, keepdim=True)
        rot = torch.where(n > CAP_R, rot * (CAP_R / n.clamp(min=1e-12)), rot)
        s[3:] = rot.reshape(-1)
        outs.append(s)
    return torch.cat(outs)
