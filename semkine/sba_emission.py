#!/usr/bin/env python3
r"""SBA emission features (docs/EXPERIMENT_RECORDS.md#sba-p4-prereg-20261022). Opt-in research code.

Generalises the swept-boundary mean field of `semkine/sba.py` to a linear emission model

    lambda_s(u) = sum_c theta_c F_{c,s}(u) + background_s(u) + eta * uniform_s(u),     theta_c >= 0

with feature channels computed from the model alone along a trajectory:

    silhouette sweep per part group (6)      occluding-contour edges with a background behind them
    internal sweep per part group (6)        occluding-contour edges with a farther hand surface behind them
    material sweep per template patch (P)    visible non-contour edges (both faces front-facing), grouped into P patches
                                             of the MANO template (k-means of rest-pose edge midpoints): creases, nails,
                                             any surface-attached brightness edge, moving with the surface
    shading per part group (6, optional)     per visible surface point, |l . dn| / (a + l . n) with the light l and
                                             ambient a of the recording: the change of Lambertian log-shading of the
                                             material point between the two knots (albedo cancels in the log ratio)

theta is fitted on training subjects by Poisson maximum likelihood (tools/sba/p4_fit.py); the light of a recording is a
per-recording nuisance (fitted on its own events, no labels needed at test time when estimated online).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from . import sba as SB

N_GROUPS = 6


@torch.no_grad()
def edge_patches(geo: "SB.Geometry", n_patches: int = 96, seed: int = 0, iters: int = 50) -> torch.Tensor:
    """(E,) patch id of every mesh edge: k-means (deterministic init) of rest-pose edge midpoints."""
    v = geo.mano.v_template.float().to(geo.device)                           # (V, 3)
    e = geo.topo.edges
    mid = 0.5 * (v[e[:, 0]] + v[e[:, 1]])
    g = torch.Generator(device="cpu").manual_seed(seed)
    c = mid[torch.randperm(len(mid), generator=g)[:n_patches].to(mid.device)].clone()
    for _ in range(iters):
        lab = torch.cdist(mid, c).argmin(1)
        for k in range(n_patches):
            m = lab == k
            if bool(m.any()):
                c[k] = mid[m].mean(0)
    return torch.cdist(mid, c).argmin(1)


@torch.no_grad()
def material_entries(verts: torch.Tensor, uv: torch.Tensor, topo: "SB.Topology", groups, n_groups):
    """Visible non-contour, non-boundary edges whose two faces both face the camera: (P, E) bool."""
    f = topo.faces
    tri = uv[:, f]
    a, b, c = tri[:, :, 0], tri[:, :, 1], tri[:, :, 2]
    area = (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])
    front = area > 0
    f0, f1 = topo.edge_f[:, 0], topo.edge_f[:, 1]
    both = (~topo.boundary)[None] & front[:, f0] & front[:, f1.clamp(min=0)]
    z = verts[..., 2]
    P, V = z.shape
    zbg = SB.splat_zbuffer(uv, z, groups, n_groups)
    zb = zbg.amin(1)
    u = (uv[..., 0] * (SB.OUT_W / SB.SRC_W)).round().long()
    w = (uv[..., 1] * (SB.OUT_H / SB.SRC_H)).round().long()
    inb = (u >= 0) & (u < SB.OUT_W) & (w >= 0) & (w < SB.OUT_H) & (z > 0)
    zz = zb.gather(1, (w.clamp(0, SB.OUT_H - 1) * SB.OUT_W + u.clamp(0, SB.OUT_W - 1)))
    vis = inb & (z <= zz + SB.VIS_TOL)                     # material points: hidden by anything in front
    e0, e1 = topo.edges[:, 0], topo.edges[:, 1]
    return both & vis[:, e0] & vis[:, e1]


@dataclass
class ChanPlan:
    plan: "SB.Plan"
    chan: torch.Tensor          # (n_e,) feature channel of each entry


@torch.no_grad()
def build_chan_plan(verts, uv, geo: "SB.Geometry", patch: Optional[torch.Tensor], n_patches: int) -> ChanPlan:
    """One trajectory (S+1 samples): contour entries (silhouette / internal by group) and material entries (by
    patch), merged into one Plan with a channel per entry."""
    S = verts.shape[0] - 1
    dev = uv.device
    on, side, internal = SB.contour_entries(verts[:S], uv[:S], geo.topo, geo.grid, geo.groups, geo.n_groups)
    s1, e1_ = on.nonzero(as_tuple=True)
    grp = geo.groups[geo.topo.edges[e1_, 0]]
    ch1 = torch.where(internal[s1, e1_], N_GROUPS + grp, grp)
    sd1 = side[s1, e1_]
    if patch is not None:
        mat = material_entries(verts[:S], uv[:S], geo.topo, geo.groups, geo.n_groups)
        s2, e2_ = mat.nonzero(as_tuple=True)
        ch2 = 2 * N_GROUPS + patch[e2_]
        sd2 = torch.ones_like(s2, dtype=sd1.dtype)
        s_idx, e_idx, chan, sd = (torch.cat([s1, s2]), torch.cat([e1_, e2_]), torch.cat([ch1, ch2]),
                                  torch.cat([sd1, sd2]))
    else:
        s_idx, e_idx, chan, sd = s1, e1_, ch1, sd1
    e0, e1 = geo.topo.edges[e_idx, 0], geo.topo.edges[e_idx, 1]
    a, b = uv[s_idx, e0], uv[s_idx, e1]
    a1, b1 = uv[s_idx + 1, e0], uv[s_idx + 1, e1]
    ev = b - a
    L = ev.norm(dim=-1).clamp(min=1e-9)
    nrm = -sd[:, None] * torch.stack([-ev[:, 1], ev[:, 0]], -1) / L[:, None]
    da, db = (nrm * (a1 - a)).sum(-1), (nrm * (b1 - b)).sum(-1)
    nt = (L / SB.SPACING).ceil().clamp(1, SB.MAX_NT).long()
    nr = (torch.maximum(da.abs(), db.abs()) / SB.SPACING).ceil().clamp(1, SB.MAX_NR).long()
    n_per = nt * nr
    el = torch.repeat_interleave(torch.arange(len(nt), device=dev), n_per)
    start = torch.cumsum(n_per, 0) - n_per
    k = torch.arange(len(el), device=dev) - start[el]
    it, ir = k // nr[el], k % nr[el]
    t = (it.float() + 0.5) / nt[el].float()
    r = (ir.float() + 0.5) / nr[el].float()
    internal_e = torch.zeros_like(chan, dtype=torch.bool)
    return ChanPlan(SB.Plan(s_idx, e0, e1, sd, internal_e, nt, nr, el, t, r), chan)


def sweep_channels(uvh: torch.Tensor, cp: ChanPlan, n_ch: int, grid: "SB.Grid", crop: "SB.Crop", S: int):
    """Unsigned swept area per channel: (H, n_ch, S, h, w), source px^2."""
    plan = cp.plan
    H = uvh.shape[0]
    dev = uvh.device
    out = torch.zeros(H * n_ch * S * crop.cells, dtype=uvh.dtype, device=dev)
    if plan.el.numel() == 0:
        return out.view(H, n_ch, S, crop.h, crop.w)
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
    ch = cp.chan[el][None].expand_as(d)
    s = plan.seg[el][None].expand_as(d)
    hb = torch.arange(H, device=dev)[:, None].expand_as(d)
    base = ((hb * n_ch + ch) * S + s) * crop.cells
    xf, yf = q[..., 0].floor(), q[..., 1].floor()
    wx, wy = q[..., 0] - xf, q[..., 1] - yf
    xi, yi = xf.long(), yf.long()
    for ox, oy, w in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        xx, yy = xi + ox, yi + oy
        ok = (xx >= 0) & (xx < SB.SRC_W) & (yy >= 0) & (yy < SB.SRC_H)
        jx = grid.lut_x[xx.clamp(0, SB.SRC_W - 1)] - crop.x0
        jy = grid.lut_y[yy.clamp(0, SB.SRC_H - 1)] - crop.y0
        ok &= (jx >= 0) & (jx < crop.w) & (jy >= 0) & (jy < crop.h)
        out.index_add_(0, (base + jy * crop.w + jx)[ok], (dA * w)[ok])
    return out.view(H, n_ch, S, crop.h, crop.w)


def features(geo: "SB.Geometry", win: "SB.Window", x0: torch.Tensor, x1: torch.Tensor, betas, K,
             patch: Optional[torch.Tensor], n_patches: int, blur: float = 0.0, sigma: Optional[float] = None):
    """Feature channels (n_ch, S, h, w) of one trajectory x0 -> x1 on the window's crop. `sigma`: the mesh scaled
    about the wrist at fixed depth (the two-dimensional scale gauge)."""
    S = int(win.widths.numel())
    v, uv, j = geo.trajectories(x0[None], x1[None], win.alphas, betas, K)
    if sigma is not None and sigma != 1.0:
        w = j[:, :, :1]
        v = w + sigma * (v - w)
        uv = SB.project_src(v, K)
    cp = build_chan_plan(v[0], uv[0], geo, patch, n_patches)
    n_ch = 2 * N_GROUPS + (n_patches if patch is not None else 0)
    F = sweep_channels(uv, cp, n_ch, geo.grid, win.crop, S)[0]
    return SB.blur_area(F, blur)


def n_channels(n_patches: int) -> int:
    return 2 * N_GROUPS + n_patches


# ---------------------------------------------------------------------------------------------------- orthogonal solver
def chan_groups(geo: "SB.Geometry", patch: torch.Tensor, n_patches: int) -> torch.Tensor:
    """(n_ch,) part group of every channel: silhouette / internal channels are their group, a material patch belongs
    to the group of the majority of its edges."""
    grp_edge = geo.groups[geo.topo.edges[:, 0]]
    out = torch.zeros(n_channels(n_patches), dtype=torch.long, device=patch.device)
    out[:N_GROUPS] = torch.arange(N_GROUPS, device=patch.device)
    out[N_GROUPS:2 * N_GROUPS] = torch.arange(N_GROUPS, device=patch.device)
    for i in range(n_patches):
        g = grp_edge[patch == i]
        out[2 * N_GROUPS + i] = torch.bincount(g, minlength=N_GROUPS).argmax() if len(g) else 0
    return out


def sweep_groups(uvh: torch.Tensor, cp: ChanPlan, theta: torch.Tensor, group_of: torch.Tensor, grid, crop, S):
    """Swept area weighted by the learned emissivity theta of each entry's channel, deposited per part group:
    (H, 6, S, h, w) -- the expected hand events per group at unit multipliers."""
    w = theta[cp.chan]
    cp2 = ChanPlan(cp.plan, group_of[cp.chan])
    plan = cp.plan
    H = uvh.shape[0]
    dev = uvh.device
    out = torch.zeros(H * N_GROUPS * S * crop.cells, dtype=uvh.dtype, device=dev)
    if plan.el.numel() == 0:
        return out.view(H, N_GROUPS, S, crop.h, crop.w)
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
    dA = (L[:, el] / plan.nt[el].to(L.dtype)) * d.abs() / plan.nr[el].to(L.dtype) * w[el][None]
    ch = cp2.chan[el][None].expand_as(d)
    s = plan.seg[el][None].expand_as(d)
    hb = torch.arange(H, device=dev)[:, None].expand_as(d)
    base = ((hb * N_GROUPS + ch) * S + s) * crop.cells
    xf, yf = q[..., 0].floor(), q[..., 1].floor()
    wx, wy = q[..., 0] - xf, q[..., 1] - yf
    xi, yi = xf.long(), yf.long()
    for ox, oy, ww in ((0, 0, (1 - wx) * (1 - wy)), (1, 0, wx * (1 - wy)), (0, 1, (1 - wx) * wy), (1, 1, wx * wy)):
        xx, yy = xi + ox, yi + oy
        ok = (xx >= 0) & (xx < SB.SRC_W) & (yy >= 0) & (yy < SB.SRC_H)
        jx = grid.lut_x[xx.clamp(0, SB.SRC_W - 1)] - crop.x0
        jy = grid.lut_y[yy.clamp(0, SB.SRC_H - 1)] - crop.y0
        ok &= (jx >= 0) & (jx < crop.w) & (jy >= 0) & (jy < crop.h)
        out.index_add_(0, (base + jy * crop.w + jx)[ok], (dA * ww)[ok])
    return out.view(H, N_GROUPS, S, crop.h, crop.w)


def profile_nuisance(G: torch.Tensor, win: "SB.Window", iters: int = 40):
    """Per-group multipliers m (6) and outlier eta >= 0 maximising the Poisson likelihood of one hypothesis
    G (6, S, h, w): convex regression on the window's events. Returns (m, eta)."""
    N = win.N.sum(0)
    cell = N > 0
    X = torch.cat([G[:, cell], win.uni[cell][None]]).double()
    Xt = torch.cat([G.flatten(1).sum(1), win.uni.sum()[None]]).double()
    B, Ne = win.bkg[cell].double(), N[cell].double()
    w = torch.full((X.shape[0],), 0.5, dtype=torch.float64, device=G.device)
    for _ in range(iters):
        lam = (w @ X + B).clamp(min=1e-12)
        g = X @ (Ne / lam) - Xt
        H = -(X * (Ne / lam ** 2)) @ X.T - 1e-9 * torch.eye(X.shape[0], dtype=torch.float64, device=G.device)
        w = (w - torch.linalg.solve(H, g)).clamp(min=0.0)
    return w[:N_GROUPS].float(), float(w[N_GROUPS])


@dataclass
class OrthResult:
    x0: torch.Tensor
    x1: torch.Tensor
    sigma: float
    m: torch.Tensor
    eta: float
    accepted: int


class OrthEvaluator:
    """Mean field lambda = sum_g m_g G_g + eta U + B of a trajectory with the scale gauge sigma (mesh about the wrist)."""

    def __init__(self, geo, win, betas, K, theta, group_of, patch, n_patches, blur, chunk=32):
        self.geo, self.win, self.betas, self.K = geo, win, betas, K
        self.theta, self.group_of, self.patch, self.n_p, self.blur = theta, group_of, patch, n_patches, blur
        self.S = int(win.widths.numel())
        self.chunk = chunk

    def _uv(self, X0, X1, sig):
        v, uv, j = self.geo.trajectories(X0, X1, self.win.alphas, self.betas, self.K)
        if not torch.all(sig == 1.0):
            w = j[:, :, :1]
            v = w + sig.view(-1, 1, 1, 1) * (v - w)
            uv = SB.project_src(v, self.K)
        return v, uv

    def plan(self, x0, x1, sig):
        v, uv = self._uv(x0[None], x1[None], torch.tensor([sig], device=x1.device))
        return build_chan_plan(v[0], uv[0], self.geo, self.patch, self.n_p)

    def groups(self, X0, X1, sig, cp):
        outs = []
        for i in range(0, X0.shape[0], self.chunk):
            _, uv = self._uv(X0[i:i + self.chunk], X1[i:i + self.chunk], sig[i:i + self.chunk])
            outs.append(SB.blur_area(sweep_groups(uv, cp, self.theta, self.group_of, self.geo.grid, self.win.crop,
                                                  self.S), self.blur))
        return torch.cat(outs)

    def rates(self, G, m, eta):
        return torch.einsum("g,hg...->h...", m, G) + eta * self.win.uni[None] + self.win.bkg[None]


def solve_orth(ev: OrthEvaluator, x0, x1, priors: "SB.Priors", sigma0: float = 1.0, sigma_sd: float = 0.05,
               free_sigma: bool = True, pools=SB.POOLS, damping: float = SB.LM_MU0) -> OrthResult:
    """Profile-likelihood Fisher scoring: at each iteration the nuisances (m, eta) are re-estimated at the base
    trajectory, then [x0 (if free), x1, log sigma] take one damped step with the Fisher of the Poisson likelihood at
    fixed nuisances (at convergence the update of the pose equals the Schur-complement / efficient-score update)."""
    dev = x1.device
    free0 = priors.start_mean is not None
    eps = SB.fd_steps(dev).to(x1.dtype)
    x0, x1 = x0.clone().float(), x1.clone().float()
    ls = float(np.log(sigma0))
    D = (102 if free0 else 51) + (1 if free_sigma else 0)
    N = ev.win.N.sum(0)
    accepted = 0
    m = torch.ones(N_GROUPS, device=dev)
    eta = 0.3
    for it, p in enumerate(pools):
        sig = float(np.exp(ls))
        cp = ev.plan(x0, x1, sig)
        G0 = ev.groups(x0[None], x1[None], torch.tensor([sig], device=dev), cp)
        m, eta = profile_nuisance(G0[0], ev.win)
        I = torch.eye(51, device=dev) * eps[None]
        X0l, X1l, Sl = [x0[None]], [x1[None]], [sig]
        if free0:
            X0l += [x0[None] + I, x0[None].expand(51, -1), x0[None] - I, x0[None].expand(51, -1)]
            X1l += [x1[None].expand(51, -1), x1[None] + I, x1[None].expand(51, -1), x1[None] - I]
            Sl += [sig] * 204
        else:
            X0l += [x0[None].expand(102, -1)]
            X1l += [x1[None] + I, x1[None] - I]
            Sl += [sig] * 102
        if free_sigma:
            X0l += [x0[None], x0[None]]
            X1l += [x1[None], x1[None]]
            Sl += [sig * float(np.exp(SIG_EPS)), sig * float(np.exp(-SIG_EPS))]
        X0, X1 = torch.cat(X0l), torch.cat(X1l)
        Sg = torch.tensor(Sl, device=dev)
        G = ev.groups(X0, X1, Sg, cp)
        lam = SB.pool(ev.rates(G, m, eta), p).flatten(1).double()
        Np = SB.pool(N, p).flatten().double()
        lam0 = lam[0].clamp(min=1e-12)
        nD = (102 if free0 else 51)
        rows_p = (lam[1:1 + nD] - lam[1 + nD:1 + 2 * nD]) / (2 * (torch.cat([eps, eps]) if free0 else eps).double()[:, None])
        J = rows_p
        if free_sigma:
            rs = (lam[1 + 2 * nD] - lam[2 + 2 * nD]) / (2 * SIG_EPS)
            J = torch.cat([J, rs[None]])
        g = J @ (Np / lam0 - 1.0)
        F = (J / lam0[None]) @ J.T
        ll = float((Np * torch.log(lam0) - lam0).sum())
        pv, pg, ph = priors.value_grad_hess(x0, x1)
        if free_sigma:
            pv = pv + 0.5 * (ls - np.log(sigma0)) ** 2 / sigma_sd ** 2
            pg = torch.cat([pg, torch.tensor([(ls - np.log(sigma0)) / sigma_sd ** 2], dtype=pg.dtype, device=dev)])
            ph = torch.block_diag(ph, torch.tensor([[1.0 / sigma_sd ** 2]], dtype=ph.dtype, device=dev))
        Hm0 = F + ph
        diag = torch.diag(Hm0.diagonal().clamp(min=1e-9))
        obj0 = ll - float(pv)
        ok = False
        mu = damping
        for _ in range(SB.LM_TRIES):
            step = torch.linalg.solve(Hm0 + mu * diag, g - pg)
            pose = SB._cap(step[:nD], free0).float()
            dls = float(step[nD].clamp(-0.05, 0.05)) if free_sigma else 0.0
            nx0 = x0 + pose[:51] if free0 else x0
            nx1 = x1 + pose[-51:]
            nls = ls + dls
            ns = float(np.exp(nls))
            Gn = ev.groups(nx0[None], nx1[None], torch.tensor([ns], device=dev), ev.plan(nx0, nx1, ns))
            lamn = SB.pool(ev.rates(Gn, m, eta), p).flatten().double().clamp(min=1e-12)
            ln = float((Np * torch.log(lamn) - lamn).sum())
            pvn, _, _ = priors.value_grad_hess(nx0, nx1)
            if free_sigma:
                pvn = pvn + 0.5 * (nls - np.log(sigma0)) ** 2 / sigma_sd ** 2
            if ln - float(pvn) > obj0:
                x0, x1, ls = nx0, nx1, nls
                ok = True
                accepted += 1
                damping = max(mu * SB.LM_DOWN, SB.LM_MU_MIN)
                break
            mu *= 10.0
        if not ok:
            damping = mu
            if p == 1:
                break
    return OrthResult(x0, x1, float(np.exp(ls)), m, eta, accepted)


SIG_EPS = 0.01
