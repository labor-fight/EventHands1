#!/usr/bin/env python3
r"""S37 root innovation (2026-09-29, `docs/S37_ROOT_INNOVATION_PREREG.md`).

The S37 root update is additively separable, `Δr = F(events; a(prev)) + G(prev)`: the linear root
head never sees the previous root, and `prev_mlp` never sees the events, so the blend between the
two is one constant gain for every sample (`docs/S37_ROUTED_READOUT_PREREG.md` §9.1). This module
gives the root a second path whose input is the *innovation measured in the image* -- where the
events sit relative to the previous state's projected silhouette -- and whose geometry only
multiplies that measurement:

    s_i   = SDF_prev(x_i, y_i) / 16 px, clipped to [-1, 1]      (outside > 0)
    n_i   = unit gradient of SDF_prev at the node
    o_i   = (pixel_i - pixel of its routed vertex) / 16 px
    q_j   = sum_i a_ij [s_i n_i, s_i, |s_i|, o_i] / sum_i a_ij  ||  sum_i a_ij / n_live     (j < 16)
    q_16  = the same pooled over all live nodes, coverage 1
    g_j   = [L2_j, L3_j, 1, 0] (part lever arms about the wrist: pixels / 16, cm);  g_16 = [0, 0, 0, 0, 0, 1, 1]

    Δξ = sum_j reshape(W_g g_j, 6 x D) · ψ(q_j),     ψ = Linear(7, H, bias=False) → ReLU → Linear(H, D, bias=False)

ψ has no bias, so q = 0 gives Δξ = 0 exactly whatever the geometry: the previous state cannot move
the root through this path on its own (the S38a failure: geometry added as a feature is read as a
prior and fed back in the closed loop). `W_g` starts at zero, so a warm-started S37 is unchanged
by this path at step 0.

Everything before ψ is `no_grad` geometry. The silhouette is a point splat of the 778 projected
vertices closed by one pixel (dilate x2, erode x1, cross structure), and the signed distance is a
chamfer (1, sqrt 2) relaxation, exact to within ~8% of the Euclidean distance inside the 16 px band
(the information probe `.experiments/s37_debug_20260928/probe_info.py` used scipy's exact EDT).
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

#: pixels per unit of the residual features, = the routing band
RES_PX = 16.0
#: relaxation sweeps of the chamfer distance; beyond this the distance saturates
SDF_ITERS = 24
Q_DIM = 7
G_DIM = 7
_BIG = 1.0e4


def _cross_dilate(m: torch.Tensor) -> torch.Tensor:
    """One binary dilation step with the 4-neighbour cross (scipy's default structure)."""
    p = F.pad(m, (1, 1, 1, 1), value=False)
    return p[:, 1:-1, 1:-1] | p[:, :-2, 1:-1] | p[:, 2:, 1:-1] | p[:, 1:-1, :-2] | p[:, 1:-1, 2:]


def _cross_erode(m: torch.Tensor) -> torch.Tensor:
    """One binary erosion step; outside the frame counts as background (scipy's border_value=0)."""
    p = F.pad(m, (1, 1, 1, 1), value=False)
    return p[:, 1:-1, 1:-1] & p[:, :-2, 1:-1] & p[:, 2:, 1:-1] & p[:, 1:-1, :-2] & p[:, 1:-1, 2:]


def splat_silhouette(uv: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """`(B, H, W)` bool: projected vertices rounded to pixels, closed by one pixel."""
    B = uv.shape[0]
    ui = uv[..., 0].round().long()
    vi = uv[..., 1].round().long()
    ok = (uv[..., 0] >= 0) & (uv[..., 0] < width) & (uv[..., 1] >= 0) & (uv[..., 1] < height)
    b = torch.arange(B, device=uv.device)[:, None].expand_as(ui)
    m = torch.zeros(B, height * width, dtype=torch.bool, device=uv.device)
    idx = (vi.clamp(0, height - 1) * width + ui.clamp(0, width - 1))
    m[b[ok], idx[ok]] = True
    m = m.view(B, height, width)
    m = _cross_dilate(_cross_dilate(m))
    return _cross_erode(m)


def _chamfer(seed: torch.Tensor, iters: int) -> torch.Tensor:
    """Chamfer distance to the `True` pixels of `seed` `(B, H, W)`, saturating after `iters` sweeps."""
    d = torch.where(seed, torch.zeros((), device=seed.device), torch.full((), _BIG, device=seed.device))
    r2 = math.sqrt(2.0)
    steps = ((0, 1, 1.0), (0, -1, 1.0), (1, 0, 1.0), (-1, 0, 1.0),
             (1, 1, r2), (1, -1, r2), (-1, 1, r2), (-1, -1, r2))
    for _ in range(iters):
        p = F.pad(d, (1, 1, 1, 1), value=_BIG)
        H, W = d.shape[-2:]
        best = d
        for dy, dx, w in steps:
            best = torch.minimum(best, p[:, 1 + dy:1 + dy + H, 1 + dx:1 + dx + W] + w)
        d = best
    return d


def silhouette_sdf(mask: torch.Tensor, iters: int = SDF_ITERS) -> Tuple[torch.Tensor, torch.Tensor]:
    """Signed distance (px, outside > 0) to a `(B, H, W)` silhouette and its unit gradient `(B, 2, H, W)`.

    A frame with no silhouette at all gets a saturated positive distance and a zero gradient.
    """
    d_out = _chamfer(mask, iters)                 # 0 on the hand, distance outside it
    d_in = _chamfer(~mask, iters)                 # 0 off the hand, distance inside it
    sdf = (d_out - d_in).clamp(-float(iters), float(iters))
    empty = ~mask.flatten(1).any(1)
    sdf = torch.where(empty[:, None, None], torch.full_like(sdf, float(iters)), sdf)
    # gaussian(sigma=1) then central differences, as in the probe
    k = torch.tensor([math.exp(-0.5 * i * i) for i in (-2, -1, 0, 1, 2)], device=sdf.device)
    k = k / k.sum()
    s = sdf.unsqueeze(1)
    s = F.conv2d(F.pad(s, (2, 2, 0, 0), mode="replicate"), k.view(1, 1, 1, 5))
    s = F.conv2d(F.pad(s, (0, 0, 2, 2), mode="replicate"), k.view(1, 1, 5, 1))
    sp = F.pad(s, (1, 1, 1, 1), mode="replicate")
    gx = 0.5 * (sp[..., 1:-1, 2:] - sp[..., 1:-1, :-2])
    gy = 0.5 * (sp[..., 2:, 1:-1] - sp[..., :-2, 1:-1])
    g = torch.cat([gx, gy], dim=1)
    g = g / g.norm(dim=1, keepdim=True).clamp_min(1e-6)
    return sdf, g


def innovation_features(px: torch.Tensor, py: torch.Tensor, mask: torch.Tensor, a: torch.Tensor,
                        vid: torch.Tensor, uv: torch.Tensor, height: int, width: int
                        ) -> torch.Tensor:
    """`q (B, 17, 7)`: per-part routed residuals and the all-node row (see module docstring).

    px, py, mask: `(B, N)` node pixels and liveness; a: `(B, N, 16)` routing weights;
    vid: `(B, N)` routed vertex; uv: `(B, V, 2)` projected vertices of prev.
    """
    B, N = mask.shape
    J = a.shape[-1]
    q = torch.zeros(B, J + 1, Q_DIM, device=px.device, dtype=torch.float32)
    if N == 0:
        return q
    sil = splat_silhouette(uv, height, width)
    sdf, grad = silhouette_sdf(sil)
    xi = px.round().long().clamp(0, width - 1)
    yi = py.round().long().clamp(0, height - 1)
    flat = yi * width + xi                                                     # (B, N)
    s = sdf.flatten(1).gather(1, flat).clamp(-RES_PX, RES_PX) / RES_PX
    nx = grad[:, 0].flatten(1).gather(1, flat)
    ny = grad[:, 1].flatten(1).gather(1, flat)
    uvv = uv.gather(1, vid.unsqueeze(-1).expand(B, N, 2))
    off = (torch.stack([px, py], -1) - uvv) / RES_PX
    live = mask.to(torch.float32)
    node = torch.stack([s * nx, s * ny, s, s.abs(), off[..., 0], off[..., 1]], -1) * live.unsqueeze(-1)
    aw = a.to(torch.float32) * live.unsqueeze(-1)
    wsum = aw.sum(1)                                                           # (B, J)
    pooled = torch.einsum("bnj,bnc->bjc", aw, node) / wsum.clamp_min(1e-6).unsqueeze(-1)
    n_live = live.sum(1, keepdim=True)                                         # (B, 1)
    cov = wsum / n_live.clamp_min(1.0)
    q[:, :J, :6] = pooled
    q[:, :J, 6] = cov
    q[:, J, :6] = node.sum(1) / n_live.clamp_min(1.0)
    q[:, J, 6] = (n_live[:, 0] > 0).to(torch.float32)
    return q


def lever_arms(verts: torch.Tensor, wrist: torch.Tensor, lbs_weights: torch.Tensor,
               fx, fy, cx, cy) -> torch.Tensor:
    """`g (B, 17, 7)`: rows `[L2 (px/16), L3 (cm), 1, 0]` per part, `[0,0,0,0,0,1,1]` for the all-node row.

    Part centroid = skinning-weighted mean of the vertices; lever = centroid - wrist (MANO joint 0).
    """
    B = verts.shape[0]
    wn = lbs_weights / lbs_weights.sum(0, keepdim=True)
    cen = torch.einsum("vj,bvc->bjc", wn.to(verts.dtype), verts)               # (B, 16, 3)

    def proj(X):
        z = X[..., 2].clamp_min(1e-6)
        return torch.stack([fx[:, None] * X[..., 0] / z + cx[:, None],
                            fy[:, None] * X[..., 1] / z + cy[:, None]], -1)

    L2 = (proj(cen) - proj(wrist[:, None])) / RES_PX
    L3 = (cen - wrist[:, None]) * 100.0
    J = cen.shape[1]
    g = torch.zeros(B, J + 1, G_DIM, device=verts.device, dtype=torch.float32)
    g[:, :J, 0:2] = L2
    g[:, :J, 2:5] = L3
    g[:, :J, 5] = 1.0
    g[:, J, 5] = 1.0
    g[:, J, 6] = 1.0
    return g


class RootInnovationHead(nn.Module):
    """`Δξ (B, 6) = Σ_j reshape(W_g g_j, 6 x D) ψ(q_j)`; exactly zero when `q = 0`."""

    def __init__(self, hidden: int = 32, width: int = 16, out_dim: int = 6):
        super().__init__()
        self.width, self.out_dim = int(width), int(out_dim)
        self.psi1 = nn.Linear(Q_DIM, hidden, bias=False)
        self.psi2 = nn.Linear(hidden, width, bias=False)
        self.geo = nn.Linear(G_DIM, out_dim * width, bias=False)
        nn.init.zeros_(self.geo.weight)

    def forward(self, q: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        u = self.psi2(torch.relu(self.psi1(q)))                                   # (B, R, D)
        M = self.geo(g.to(u.dtype)).view(*g.shape[:2], self.out_dim, self.width)  # (B, R, 6, D)
        return torch.einsum("brow,brw->bo", M, u)
