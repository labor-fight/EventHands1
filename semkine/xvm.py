#!/usr/bin/env python3
r"""XVM: vertex-anchored event motion measurement (docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021).

For every visible vertex i of the anchor state's MANO mesh the operator gathers the events within a depth-scaled
image radius rho_i = f r / z_i, expresses each one in the vertex's projected normal frame R_i = [e1_i, e2_i] (e1_i
the image direction of the vertex normal, i.e. the silhouette normal at contour vertices), and maps the set through a
small continuous kernel to a 2-D displacement m_i (frame R_i, units of rho_i) and a 2x2 information matrix
Lambda_i = L_i L_i^T. A fixed kinematic transport then solves

    Delta = (sum_i B_i^T Lt_i B_i + lam P)^-1 sum_i B_i^T Lt_i mt_i,   mt_i = rho_i R_i m_i,  Lt_i = R_i Lambda_i R_i^T / rho_i^2

with B_i = d pi(V_i) / d x the projected kinematic Jacobian. No absolute depth or absolute pixel coordinate enters the
learned part; the kernel is equivariant to in-plane rotation and depth scaling by construction.

Nothing in the production model imports this module unless MODEL.ENCODER is `xvm` (opt-in).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

#: pair features: xi (2) | |xi| | age / W | measuring (age < 50 ms) | polarity +-1 | inside prev silhouette | |n.d|
PAIR_IN = 8
#: |n . d_hat| below this is a contour vertex
CONTOUR_COS = 0.4


def projection_jacobian(verts: torch.Tensor, fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    """d pi / d X (B, V, 2, 3) of the pinhole projection at camera-frame `verts` (B, V, 3); fx, fy (B,)."""
    X, Y, Z = verts[..., 0], verts[..., 1], verts[..., 2].clamp(min=1e-6)
    Jp = verts.new_zeros(*verts.shape[:-1], 2, 3)
    fx, fy = fx.view(-1, 1), fy.view(-1, 1)
    Jp[..., 0, 0] = fx / Z
    Jp[..., 0, 2] = -fx * X / Z ** 2
    Jp[..., 1, 1] = fy / Z
    Jp[..., 1, 2] = -fy * Y / Z ** 2
    return Jp


def vertex_frames(verts: torch.Tensor, normals: torch.Tensor, Jp: torch.Tensor):
    """Projected normal frame per vertex: R (B, V, 2, 2) with columns e1 (normal direction in the image) and e2
    (e1 rotated +90 deg), and |n . d_hat| (B, V)."""
    dhat = verts / verts.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    cos = (normals * dhat).sum(-1).abs()
    e = (Jp @ normals.unsqueeze(-1)).squeeze(-1)
    nrm = e.norm(dim=-1, keepdim=True)
    e1 = torch.where(nrm > 1e-9, e / nrm.clamp(min=1e-12), torch.tensor([1.0, 0.0], device=e.device, dtype=e.dtype))
    e2 = torch.stack([-e1[..., 1], e1[..., 0]], -1)
    R = torch.stack([e1, e2], -1)                                          # columns e1, e2
    return R, cos


def splat_mask(uv: torch.Tensor, valid: torch.Tensor, H: int, W: int, r: int = 1) -> torch.Tensor:
    """(B, H, W) bool: (2r+1)^2 splat of the projected vertices `uv` (B, V, 2) where `valid` (B, V)."""
    B, V, _ = uv.shape
    m = torch.zeros(B, H * W, dtype=torch.bool, device=uv.device)
    ui, vi = uv[..., 0].round().long(), uv[..., 1].round().long()
    bidx = torch.arange(B, device=uv.device).view(B, 1).expand(B, V)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            uu, vv = ui + dx, vi + dy
            ok = valid & (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
            m[bidx[ok], (vv * W + uu)[ok]] = True
    return m.view(B, H, W)


def gather(uv: torch.Tensor, vmask: torch.Tensor, rho: torch.Tensor, ev_xy: torch.Tensor, ev_mask: torch.Tensor,
           k: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The <= k nearest events within rho of every vertex. uv (B, V, 2), vmask (B, V), rho (B, V) px,
    ev_xy (B, N, 2), ev_mask (B, N). Returns idx (B, V, k) long, ok (B, V, k) bool, d2 (B, V, k)."""
    d2 = torch.cdist(uv.float(), ev_xy.float()).square()                   # (B, V, N)
    d2 = d2.masked_fill(~ev_mask.unsqueeze(1), float("inf"))
    kk = min(int(k), d2.shape[-1])
    val, idx = d2.topk(kk, dim=-1, largest=False)
    ok = (val <= rho.float().unsqueeze(-1) ** 2) & vmask.unsqueeze(-1) & torch.isfinite(val)
    return idx, ok, val


def pair_features(uv, R, rho, cos, ev, idx, ok, mem_ms: float, in_sil):
    """(B, V, k, PAIR_IN). ev: dict with xy (B, N, 2) px, age_ms (B, N), pol (B, N) in {-1, 1};
    in_sil (B, N) float: event pixel inside the anchor silhouette."""
    B, V, K = idx.shape
    flat = idx.reshape(B, -1)

    def take(x):
        return torch.gather(x, 1, flat).view(B, V, K) if x.dim() == 2 else \
            torch.gather(x, 1, flat.unsqueeze(-1).expand(-1, -1, x.shape[-1])).view(B, V, K, x.shape[-1])

    off = take(ev["xy"].float()) - uv.float().unsqueeze(2)                     # (B, V, K, 2) px
    xi = (R.float().transpose(-1, -2).unsqueeze(2) @ off.unsqueeze(-1)).squeeze(-1) / rho.float().view(B, V, 1, 1)
    age = take(ev["age_ms"].float())
    f = torch.cat([xi, xi.norm(dim=-1, keepdim=True), (age / float(mem_ms)).unsqueeze(-1),
                   (age < 50.0).float().unsqueeze(-1), take(ev["pol"].float()).unsqueeze(-1),
                   take(in_sil.float()).unsqueeze(-1), cos.float().view(B, V, 1, 1).expand(B, V, K, 1)], -1)
    return f * ok.unsqueeze(-1).float()


class XVMKernel(nn.Module):
    """Continuous kernel over a vertex's event set -> (m (B, V, 2), L (B, V, 2, 2)) in the vertex frame."""

    def __init__(self, hidden: int = 64):
        super().__init__()
        self.kernel = nn.Sequential(nn.Linear(PAIR_IN, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                    nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, 5))

    def forward(self, feats: torch.Tensor, ok: torch.Tensor, cos: torch.Tensor):
        h = self.kernel(feats)                                                 # (B, V, K, C)
        okf = ok.unsqueeze(-1).to(h.dtype)
        cnt = okf.sum(2)
        mean = (h * okf).sum(2) / cnt.clamp(min=1.0)
        mx = h.masked_fill(~ok.unsqueeze(-1), 0.0).amax(2)
        z = torch.cat([mean, mx, torch.log1p(cnt), cos.to(h.dtype).unsqueeze(-1)], -1)
        o = self.head(z)
        m = o[..., :2]
        l11 = F.softplus(o[..., 2]) + 1e-3
        l22 = F.softplus(o[..., 3]) + 1e-3
        L = torch.zeros(*o.shape[:-1], 2, 2, device=o.device, dtype=o.dtype)
        L[..., 0, 0] = l11
        L[..., 1, 1] = l22
        L[..., 1, 0] = o[..., 4]
        return m, L


def gaussian_nll(m: torch.Tensor, L: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Mean over weighted vertices of 0.5 r^T L L^T r - log det L (information form). t, m (B, V, 2); w (B, V)."""
    r = (m - t).unsqueeze(-1)
    q = (L.transpose(-1, -2) @ r).squeeze(-1)
    nll = 0.5 * (q * q).sum(-1) - torch.log(L[..., 0, 0]) - torch.log(L[..., 1, 1])
    return (nll * w).sum() / w.sum().clamp(min=1.0)


def to_image(m: torch.Tensor, L: torch.Tensor, R: torch.Tensor, rho: torch.Tensor):
    """Vertex-frame measurement -> image-frame displacement (px) and information (1/px^2)."""
    mt = rho.unsqueeze(-1) * (R @ m.unsqueeze(-1)).squeeze(-1)
    Lam = L @ L.transpose(-1, -2)
    Lt = R @ Lam @ R.transpose(-1, -2) / rho.unsqueeze(-1).unsqueeze(-1) ** 2
    return mt, Lt


def transport(B_v: torch.Tensor, mt: torch.Tensor, Lt: torch.Tensor, w: torch.Tensor, P: torch.Tensor,
              lam: float = 1.0) -> torch.Tensor:
    """Information-weighted kinematic solve. B_v (B, V, 2, D), mt (B, V, 2), Lt (B, V, 2, 2), w (B, V) vertex mask,
    P (D, D) prior precision. Returns Delta (B, D)."""
    W = Lt * w.unsqueeze(-1).unsqueeze(-1)
    H = torch.einsum("bvka,bvkl,bvlc->bac", B_v, W, B_v) + lam * P
    g = torch.einsum("bvka,bvkl,bvl->ba", B_v, W, mt)
    return torch.linalg.solve(H, g.unsqueeze(-1)).squeeze(-1)


__all__ = ["CONTOUR_COS", "PAIR_IN", "XVMKernel", "gather", "gaussian_nll", "pair_features", "projection_jacobian",
           "splat_mask", "to_image", "transport", "vertex_frames"]
