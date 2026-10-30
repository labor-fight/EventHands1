#!/usr/bin/env python3
r"""XDI measurement innovation (docs/EXPERIMENT_RECORDS.md#xdi-20261014): an event-to-occluding-contour observation
model of the previous MANO state, and the damped Gauss-Newton step it implies. No parameters, `no_grad`, fp32.

It runs beside the S37 association and does not change it (`stride_sample`, `associate_rays` and `XYZMeshNet` are
untouched): the hit-first association gives every event inside the predicted silhouette a zero residual against its
own hit face (`n . dXYZ = 0`), so it cannot say that the mesh is in the wrong place. Here every event is measured
against the predicted *contour*, from either side.

    V = FK(x)                       camera-frame vertices, metres
    c_v = exp(-(n_v . V_v/|V_v|)^2 / (2 sigma_c^2))      occluding-contour weight (silhouette and inner contours)
    u_v = V_xy / V_z,  m_v ~ (n_x - u n_z, n_y - v n_z)  projected vertex and the contour's 2D normal
    event i (ray d_i, p_i = d_i,xy): v(i) = nearest contour vertex in the image plane
    e_i = m_v . (p_i - u_v)                               point-to-line residual (normalised image units)
    w_i = c_v / (1 + (e_i / s)^2) [ |p_i - u_v| <= band ]  Cauchy weight, background beyond the band
    A_ik = m_v . d u_v / d x_k                            finite differences of FK in the additive 51D parameters
    delta = (sum w A^T A + lambda diag + nu prior)^-1 sum w A^T e,   x <- x + delta   (iterated)

The damping prior pulls the *step* to zero, i.e. an unobservable degree of freedom holds instead of being pulled to
a training mean. The output `delta` is in the same additive 51D convention as the XDL `prev + delta` update.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import torch

from .xyz_mesh import associate_rays, vertex_face_table, vertex_normals

#: finite-difference steps: translation (m), rotations / pose (rad)
FD_T, FD_R = 1e-4, 1e-3
#: damping prior scale of a step per block: translation (m), root (rad), fingers (rad)
PRIOR_T, PRIOR_ROOT, PRIOR_FINGER = 0.05, 0.3, 0.3
#: per-iteration step caps: translation (m), rotations (rad)
CAP_T, CAP_R = 0.1, 0.5
DOF_SETS = {6: tuple(range(6)), 51: tuple(range(51))}


def _steps(dofs: Sequence[int], device) -> torch.Tensor:
    return torch.tensor([FD_T if k < 3 else FD_R for k in dofs], device=device, dtype=torch.float32)


def _prior(dofs: Sequence[int], device) -> torch.Tensor:
    return torch.tensor([PRIOR_T if k < 3 else PRIOR_ROOT if k < 6 else PRIOR_FINGER for k in dofs],
                        device=device, dtype=torch.float32)


@torch.no_grad()
def event_orientation(p: torch.Tensor, mask: torch.Tensor, k: int = 8, rows: int = 8):
    """Local edge structure of every event from its `k` nearest live events in the same packet: the unit normal
    `(B, N, 2)` (least-variance direction of the neighbourhood) and the anisotropy `1 - l_min / l_max` in [0, 1]
    (about 0 for isotropic clutter, 1 on a clean edge). Independent of the hand state."""
    B, N, _ = p.shape
    nrm = torch.zeros(B, N, 2, device=p.device)
    ani = torch.zeros(B, N, device=p.device)
    kk = max(2, min(int(k) + 1, N))
    for b0 in range(0, B, rows):
        b1 = min(B, b0 + rows)
        q = p[b0:b1].float()
        dist = torch.cdist(q, q).masked_fill(~mask[b0:b1].unsqueeze(1), float('inf'))
        nb = dist.topk(kk, dim=-1, largest=False).indices                               # (b, N, k+1), self first
        valid = torch.gather(mask[b0:b1].unsqueeze(1).expand(-1, N, -1), 2, nb).float()
        pts = torch.gather(q.unsqueeze(1).expand(-1, N, -1, -1), 2, nb.unsqueeze(-1).expand(-1, -1, -1, 2))
        cnt = valid.sum(-1, keepdim=True).clamp_min(1.0)
        mu = (pts * valid.unsqueeze(-1)).sum(2) / cnt
        dq = (pts - mu.unsqueeze(2)) * valid.unsqueeze(-1)
        cov = torch.einsum('bnki,bnkj->bnij', dq, dq) / cnt.unsqueeze(-1)
        evals, evecs = torch.linalg.eigh(cov + 1e-12 * torch.eye(2, device=p.device))
        nrm[b0:b1] = evecs[..., :, 0]
        ani[b0:b1] = (1.0 - evals[..., 0] / evals[..., 1].clamp_min(1e-18)).clamp(0, 1) * (cnt.squeeze(-1) >= 3)
    return nrm, ani * mask.float()


_VF_CACHE: Dict[tuple, torch.Tensor] = {}


@torch.no_grad()
def visible_vertices(verts: torch.Tensor, faces: torch.Tensor, tol: float = 2e-3) -> torch.Tensor:
    """`(B, V)` float: 1 where the camera ray to the vertex meets no surface more than `tol` metres in front of it.
    Calls the unchanged S37 ray caster (`associate_rays`, rays through the vertices themselves)."""
    key = (faces.data_ptr(), faces.shape[0], str(faces.device))
    if key not in _VF_CACHE:
        _VF_CACHE[key] = vertex_face_table(faces, verts.shape[1]).to(faces.device)
    B, V, _ = verts.shape
    d = verts.float() / verts[..., 2:3].float().clamp_min(1e-6)
    zero = torch.zeros(B, device=verts.device)
    hit = associate_rays(d, torch.ones(B, V, dtype=torch.bool, device=verts.device), verts, faces, _VF_CACHE[key],
                         band=zero, front_tol=zero, z_tol=0.0, k=16)
    return ((~hit['hit']) | (hit['z'] >= verts[..., 2] - tol)).float()


@torch.no_grad()
def contour_terms(verts: torch.Tensor, faces: torch.Tensor, p: torch.Tensor, mask: torch.Tensor,
                  fpx: torch.Tensor, sigma_c: float, sigma_px: float, band_px: float, c_min: float = 0.3,
                  k: int = 8, rows: int = 8, orient=None, visible: bool = True) -> Dict[str, torch.Tensor]:
    """Soft (EM-style) event-to-contour correspondences at image scale `sigma_px`.

    Every live event takes its `k` nearest contour vertices (contour weight >= c_min) with posteriors
    `pi ~ c_v exp(-|p - u_v|^2 / (2 sigma^2))` (normalised over the k). Returned per pair: `vid, pi, e, m` with
    `e = m_v . (p - u_v)`, shapes `(B, N, k)` / `(B, N, k, 2)`; per event the weight `w (B, N)`: zero beyond `band_px`
    from every contour vertex, times the oriented-matching factor when `orient` is given; plus `u (B, V, 2)`, `z (B, V)`.
    p: `(B, N, 2)` event rays' x/z, y/z; mask `(B, N)`; fpx `(B,)` focal length in render pixels."""
    B, N, _ = p.shape
    V = verts.float()
    z = V[..., 2].clamp_min(1e-6)
    u = V[..., :2] / z.unsqueeze(-1)
    n = vertex_normals(V, faces)
    view = V / V.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    c = torch.exp(-((n * view).sum(-1)).square() / (2.0 * sigma_c ** 2)) * (V[..., 2] > 1e-4)
    if visible:
        c = c * visible_vertices(V, faces)
    m = n[..., :2] - u * n[..., 2:3]
    m = m / m.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    kk = max(1, min(int(k), V.shape[1]))
    vid = torch.zeros(B, N, kk, dtype=torch.long, device=p.device)
    d2 = torch.full((B, N, kk), float('inf'), device=p.device)
    for b0 in range(0, B, rows):
        b1 = min(B, b0 + rows)
        dist = torch.cdist(p[b0:b1].float(), u[b0:b1]).square()                             # (b, N, V)
        dist = dist.masked_fill((c[b0:b1] < c_min).unsqueeze(1), float('inf'))
        top = dist.topk(kk, dim=-1, largest=False)
        vid[b0:b1], d2[b0:b1] = top.indices, top.values
    sig2 = (sigma_px / fpx).square().view(B, 1, 1)
    cv = torch.gather(c.unsqueeze(1).expand(-1, N, -1), 2, vid)
    logit = torch.log(cv.clamp_min(1e-12)) - d2 / (2.0 * sig2)
    logit = logit.masked_fill(~torch.isfinite(d2), float('-inf'))
    pi = torch.softmax(logit, dim=-1).nan_to_num(0.0)
    uv = torch.gather(u.unsqueeze(1).expand(-1, N, -1, -1), 2, vid.unsqueeze(-1).expand(-1, -1, -1, 2))
    mv = torch.gather(m.unsqueeze(1).expand(-1, N, -1, -1), 2, vid.unsqueeze(-1).expand(-1, -1, -1, 2))
    e = (mv * (p.float().unsqueeze(2) - uv)).sum(-1)                                      # (B, N, k)
    band = (band_px / fpx).view(B, 1)
    w = (d2[..., 0].clamp_min(0).sqrt() <= band).float() * mask.float()
    if orient is not None:
        # oriented matching: the event's local edge normal must agree with the matched contour normal
        nrm_e, ani = orient
        mbar = (pi.unsqueeze(-1) * mv).sum(2)
        w = w * ani * (nrm_e * mbar).sum(-1).square()
    return {'vid': vid, 'pi': pi, 'e': e, 'm': mv, 'w': w, 'u': u, 'z': z}


@torch.no_grad()
def robust_cost(verts: torch.Tensor, faces: torch.Tensor, p: torch.Tensor, mask: torch.Tensor, fpx: torch.Tensor,
                sigma_c: float, scale_px: float, band_px: float, ani: torch.Tensor = None) -> torch.Tensor:
    """`(B,)` fine-scale objective: per live event the Cauchy cost `log(1 + (e / s)^2)` of its point-to-line residual
    to the nearest visible contour vertex, saturated at the band, weighted by the state-independent anisotropy."""
    t = contour_terms(verts, faces, p, mask, fpx, sigma_c, scale_px, band_px, k=1)
    s = (scale_px / fpx).view(-1, 1)
    band = (band_px / fpx).view(-1, 1)
    rho = torch.log1p((t['e'][..., 0] / s).square())
    rho = torch.where(t['w'] > 0, torch.minimum(rho, torch.log1p((band / s).square())), torch.log1p((band / s).square()))
    wt = mask.float() if ani is None else ani
    return (wt * rho).sum(1) / wt.sum(1).clamp_min(1e-12)


@torch.no_grad()
def innovation_step(fk: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], x0: torch.Tensor, betas: torch.Tensor,
                    faces: torch.Tensor, p: torch.Tensor, mask: torch.Tensor, fpx: torch.Tensor, dof: int = 51,
                    schedule_px: Sequence[float] = (16.0, 8.0, 4.0, 3.0), sigma_c: float = 0.15,
                    band_px: float = 24.0, lam: float = 0.1, nu: float = 0.01, k: int = 8, orient_k: int = 8,
                    verts0: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """Damped Gauss-Newton (EM-ICP, point to line) on the soft event-to-contour residuals.

    fk(params (M, 51), betas (M, 10)) -> verts (M, 778, 3). x0 `(B, 51)` the previous state; p `(B, N, 2)` event rays
    (x/z, y/z, the render-space rays of `_xyz_mesh_forward`); mask `(B, N)`; fpx `(B,)`. One iteration per entry of
    `schedule_px`, the correspondence scale annealed coarse to fine: at 16 px the hand aligns as a whole, at 3 px
    each finger contour (adjacent finger contours are a few pixels apart). Each pair is weighted by its posterior and
    a Cauchy factor at the current scale. Returns `delta (B, 51)` (zeros outside the solved degrees of freedom),
    `info (B, 51)` = log1p of the final information diagonal per live event, `inlier (B,)` share of live events with
    positive weight, `abs_e_px (B,)` weighted mean |e| in pixels."""
    dofs = DOF_SETS[int(dof)]
    B = x0.shape[0]
    D = len(dofs)
    dev = x0.device
    x = x0.float().clone()
    h = _steps(dofs, dev)
    prior = _prior(dofs, dev)
    noise = (float(schedule_px[-1]) / fpx).square()                                       # (B,) final residual scale^2
    idx = torch.tensor(dofs, device=dev, dtype=torch.long)
    caps = torch.tensor([CAP_T if kd < 3 else CAP_R for kd in dofs], device=dev, dtype=torch.float32)
    live = mask.float().sum(1).clamp_min(1.0)
    info = torch.zeros(B, 51, device=dev)
    inlier = torch.zeros(B, device=dev)
    abs_e = torch.zeros(B, device=dev)
    orient = event_orientation(p, mask, orient_k) if orient_k > 0 else None
    ani = orient[1] if orient is not None else None
    fine_px = float(schedule_px[-1])
    for it, sigma_px in enumerate(schedule_px):
        V = verts0.float() if (it == 0 and verts0 is not None) else fk(x, betas).float()
        t = contour_terms(V, faces, p, mask, fpx, sigma_c, float(sigma_px), band_px, k=k, orient=orient)
        xs = x.unsqueeze(1).repeat(1, D, 1)                                               # (B, D, 51)
        xs[:, torch.arange(D, device=dev), idx] += h
        Vk = fk(xs.reshape(B * D, 51), betas.repeat_interleave(D, 0)).float().view(B, D, -1, 3)
        dV = (Vk - V.unsqueeze(1)) / h.view(1, D, 1, 1)                                   # (B, D, V, 3)
        du = (dV[..., :2] - t['u'].unsqueeze(1) * dV[..., 2:3]) / t['z'].unsqueeze(1).unsqueeze(-1)
        Nk = t['vid'].shape[1] * t['vid'].shape[2]
        g = torch.gather(du, 2, t['vid'].reshape(B, 1, Nk, 1).expand(-1, D, -1, 2))      # (B, D, N*k, 2)
        A = (g * t['m'].reshape(B, 1, Nk, 2)).sum(-1)                                     # (B, D, N*k)
        s = (float(sigma_px) / fpx).view(B, 1, 1)
        wp = (t['w'].unsqueeze(-1) * t['pi'] / (1.0 + (t['e'] / s).square())).reshape(B, Nk)
        H = torch.einsum('bdn,ben->bde', A * wp.unsqueeze(1), A)
        rhs = torch.einsum('bdn,bn->bd', A * wp.unsqueeze(1), t['e'].reshape(B, Nk))
        W = wp.sum(1)
        damp = lam * torch.diagonal(H, dim1=1, dim2=2) + nu * (W * noise).unsqueeze(1) / prior.square() + 1e-12
        step = torch.linalg.solve(H + torch.diag_embed(damp), rhs.unsqueeze(-1)).squeeze(-1)
        step = torch.maximum(torch.minimum(step, caps), -caps)
        step = torch.where(torch.isfinite(step), step, torch.zeros_like(step))
        # LM acceptance on the fine-scale objective: a coarse step is kept only if the fine cost goes down
        # (halved twice before rejection), so a state the events already support is not pulled away
        base = robust_cost(V, faces, p, mask, fpx, sigma_c, fine_px, band_px, ani)
        accepted = torch.zeros_like(step)
        todo = torch.ones(B, dtype=torch.bool, device=dev)
        for frac in (1.0, 0.5, 0.25):
            cand = x.clone()
            cand[:, idx] += frac * step
            cost = robust_cost(fk(cand, betas).float(), faces, p, mask, fpx, sigma_c, fine_px, band_px, ani)
            ok = todo & (cost < base)
            accepted[ok] = frac * step[ok]
            todo = todo & ~ok
        x[:, idx] += accepted
        info[:, idx] = torch.log1p(torch.diagonal(H, dim1=1, dim2=2) / noise.unsqueeze(1).clamp_min(1e-12) / live.unsqueeze(1))
        inlier = (t['w'] > 0).float().sum(1) / live
        abs_e = (wp * t['e'].reshape(B, Nk).abs()).sum(1) / W.clamp_min(1e-12) * fpx
    return {'delta': x - x0.float(), 'info': info, 'inlier': inlier, 'abs_e_px': abs_e}


__all__ = ['DOF_SETS', 'contour_terms', 'event_orientation', 'innovation_step', 'robust_cost', 'visible_vertices']
