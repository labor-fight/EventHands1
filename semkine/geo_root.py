#!/usr/bin/env python3
r"""S38. Depth-aware geometric root evidence: a state's full 3D MANO mesh as the measurement model of
a packet's events (`docs/S38_GEOROOT_PREREG.md`).

    contour   visible silhouette edges of the posed mesh: one adjacent face towards the camera, one
              away, and the edge midpoint is the first surface along its own ray
    residual  s_i = n_i^T (u_i - p_i)        nearest projected contour segment within `band_px`
    Jacobian  a_i = n_i^T dpi/dX(X_i) [ -[X_i - c]_x | I ]          X_i the 3D contour point, c the wrist
    solve     xi = (A^T W A + rho diag(A^T W A))^-1 A^T W s          IRLS (Huber), Gauss-Newton

`xi = (omega, v)` is a camera-frame rotation about the wrist and a translation; on the 51D state it
acts as `R_g <- Exp(omega) R_g`, `t <- t + v` (MANO rotates about J0, which the translation carries to
the wrist, so a rotation about the wrist leaves `t` alone). The contour point's depth `z_i` and its
lever arm `X_i - c`, depth component included, enter only through `a_i`, multiplied by the unknown
correction -- the only way an out-of-plane rotation or a depth change shows in the image. With no
associated events, or with zero residuals, `xi = 0` whatever the geometry: the geometry multiplies the
evidence and never enters as an additive term.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import torch

from .lie import so3_exp, so3_log

#: order of the root correction `xi`: rotation about the wrist (rad), then translation (m), camera frame
DOF = ("rot_x", "rot_y", "rot_z", "t_x", "t_y", "t_z")

#: (events x segments) entries per association chunk
ASSOC_CHUNK = 2_000_000


@dataclass(frozen=True)
class GeoRootParams:
    """Solver settings. These are the pre-registered values; they are not tuned on zgz."""
    band_px: float = 10.0
    huber_px: float = 1.5
    damping: float = 0.1
    irls_iters: int = 3
    #: 2 in the first draft; on events sampled on the true contour (training sequences) two
    #: iterations left the median residual at 1.65 px and every gain below 0.65, eight reach 0.53 px
    gn_iters: int = 8
    min_events: int = 64
    occlusion_tol_m: float = 0.005


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Contour:
    """Visible occluding-contour segments of one posed mesh."""
    pa: torch.Tensor    # (S, 2) projected endpoints, pixels
    pb: torch.Tensor
    xa: torch.Tensor    # (S, 3) the same endpoints in the camera frame, metres
    xb: torch.Tensor
    n: torch.Tensor     # (S, 2) unit outward normal in the image


def edge_topology(faces: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Interior edges `(E, 2)` of a triangle mesh and their two adjacent faces `(E, 2)`. Boundary
    edges -- MANO's wrist opening -- are dropped: they are where the forearm was cut, not a contour."""
    faces = faces.long()
    F = faces.shape[0]
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]).sort(dim=1).values
    fid = torch.arange(F, device=faces.device).repeat(3)
    key = e[:, 0] * (int(faces.max()) + 1) + e[:, 1]
    order = torch.argsort(key, stable=True)
    key, e, fid = key[order], e[order], fid[order]
    pair = (key[1:] == key[:-1]).nonzero().squeeze(1)
    return e[pair], torch.stack([fid[pair], fid[pair + 1]], 1)


def project(X: torch.Tensor, K: Intrinsics) -> torch.Tensor:
    z = X[..., 2].clamp_min(1e-6)
    return torch.stack([K.fx * X[..., 0] / z + K.cx, K.fy * X[..., 1] / z + K.cy], -1)


def raycast_depth(verts: torch.Tensor, faces: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                  K: Intrinsics, chunk: int = 4096) -> torch.Tensor:
    """Depth of the first surface along the camera rays through pixels `(u, v)`, `inf` on a miss.
    Moller-Trumbore against every face; the ray's z component is 1, so the ray parameter is the depth."""
    d = torch.stack([(u - K.cx) / K.fx, (v - K.cy) / K.fy, torch.ones_like(u)], -1).to(verts.dtype)
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    e1, e2, tv = v1 - v0, v2 - v0, -v0
    qv = torch.cross(tv, e1, dim=-1)
    tq = (e2 * qv).sum(-1)
    F = faces.shape[0]
    out = torch.full((d.shape[0],), float("inf"), device=verts.device, dtype=verts.dtype)
    for i0 in range(0, d.shape[0], chunk):
        dd = d[i0:i0 + chunk]
        m = dd.shape[0]
        p = torch.cross(dd[:, None, :].expand(m, F, 3), e2[None].expand(m, F, 3), dim=-1)
        det = (e1[None] * p).sum(-1)
        ok = det.abs() > 1e-12
        inv = 1.0 / torch.where(ok, det, torch.ones_like(det))
        bu = (tv[None] * p).sum(-1) * inv
        bv = (dd @ qv.T) * inv
        t = tq[None] * inv
        hit = ok & (bu >= -1e-6) & (bv >= -1e-6) & (bu + bv <= 1 + 1e-6) & (t > 1e-4)
        out[i0:i0 + m] = torch.where(hit, t, torch.full_like(t, float("inf"))).min(1).values
    return out


def occluding_contour(verts: torch.Tensor, faces: torch.Tensor, edges: torch.Tensor, adj: torch.Tensor,
                      K: Intrinsics, occlusion_tol_m: float) -> Contour:
    """Visible silhouette edges of one posed mesh `(V, 3)` (camera at the origin, looking down +z).

    The outward normal of a segment points away from the projected centroid of its camera-facing face:
    at an occluding contour both adjacent faces project to the inside, the surface folding away."""
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    centroid = (v0 + v1 + v2) / 3.0
    front = (torch.cross(v1 - v0, v2 - v0, dim=-1) * centroid).sum(-1) < 0
    sil = front[adj[:, 0]] != front[adj[:, 1]]
    e, a = edges[sil], adj[sil]
    f_front = torch.where(front[a[:, 0]], a[:, 0], a[:, 1])
    xa, xb = verts[e[:, 0]], verts[e[:, 1]]
    pa, pb = project(xa, K), project(xb, K)
    xm = 0.5 * (xa + xb)
    pm = project(xm, K)
    visible = raycast_depth(verts, faces, pm[:, 0], pm[:, 1], K) >= xm[:, 2] - occlusion_tol_m
    d = pb - pa
    length = d.norm(dim=-1)
    n = torch.stack([d[:, 1], -d[:, 0]], -1) / length.clamp_min(1e-9)[:, None]
    inward = project(centroid[f_front], K) - pm
    n = torch.where(((inward * n).sum(-1) > 0)[:, None], -n, n)
    keep = visible & (length > 1e-6)
    return Contour(pa[keep], pb[keep], xa[keep], xb[keep], n[keep])


def associate(px: torch.Tensor, py: torch.Tensor, C: Contour, band_px: float
              ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Nearest contour segment of every event: `(keep (M,), s (M,), X (M, 3), n (M, 2))`.

    `keep`: within `band_px` of the contour; `s`: signed distance along the segment's outward normal
    (> 0 outside the silhouette); `X`: the 3D contour point at the foot of the perpendicular."""
    M, S = px.shape[0], C.pa.shape[0]
    dev, dt = C.pa.device, C.pa.dtype
    if M == 0 or S == 0:
        return (torch.zeros(M, dtype=torch.bool, device=dev), torch.zeros(M, device=dev, dtype=dt),
                torch.zeros(M, 3, device=dev, dtype=dt), torch.zeros(M, 2, device=dev, dtype=dt))
    P = torch.stack([px, py], -1).to(dt)
    d = C.pb - C.pa
    L2 = (d * d).sum(-1).clamp_min(1e-12)
    dmin, j, tj = [], [], []
    step = max(1, ASSOC_CHUNK // S)
    for i0 in range(0, M, step):
        Pc = P[i0:i0 + step]
        t = (((Pc[:, None, :] - C.pa[None]) * d[None]).sum(-1) / L2[None]).clamp(0.0, 1.0)
        dist2 = ((Pc[:, None, :] - (C.pa[None] + t[..., None] * d[None])) ** 2).sum(-1)
        dm, jj = dist2.min(1)
        dmin.append(dm)
        j.append(jj)
        tj.append(t.gather(1, jj[:, None])[:, 0])
    dmin, j, tj = torch.cat(dmin), torch.cat(j), torch.cat(tj)
    n = C.n[j]
    s = ((P - (C.pa[j] + tj[:, None] * d[j])) * n).sum(-1)
    X = C.xa[j] + tj[:, None] * (C.xb[j] - C.xa[j])
    return dmin.sqrt() <= band_px, s, X, n


def jacobian_rows(X: torch.Tensor, n: torch.Tensor, wrist: torch.Tensor, K: Intrinsics) -> torch.Tensor:
    """`(M, 6)` rows `n^T d pi / d xi` of a camera-frame root twist about `wrist`, columns as `DOF`:
    `delta X = omega x (X - c) + v`, so the rotation part of the row is `(X - c) x g` with
    `g = n^T d pi / d X`."""
    x, y, z = X.unbind(-1)
    iz = 1.0 / z.clamp_min(1e-6)
    g = torch.stack([n[:, 0] * K.fx * iz, n[:, 1] * K.fy * iz,
                     -(n[:, 0] * K.fx * x + n[:, 1] * K.fy * y) * iz * iz], -1)
    return torch.cat([torch.cross(X - wrist, g, dim=-1), g], -1)


def solve_twist(A: torch.Tensor, s: torch.Tensor, P: GeoRootParams) -> torch.Tensor:
    """Robust (Huber IRLS) Marquardt-damped least squares for `xi` `(6,)`, float64; exactly zero with
    fewer than `min_events` rows or with all residuals zero."""
    xi = torch.zeros(6, dtype=torch.float64, device=A.device)
    if A.shape[0] < P.min_events:
        return xi
    A64, s64 = A.double(), s.double()
    eye = torch.eye(6, dtype=torch.float64, device=A.device)
    for _ in range(P.irls_iters):
        r = (s64 - A64 @ xi).abs()
        w = torch.where(r <= P.huber_px, torch.ones_like(r), P.huber_px / r.clamp_min(1e-12))
        Aw = A64 * w[:, None]
        H = Aw.T @ A64
        H = H + P.damping * torch.diag(torch.diagonal(H)) + 1e-12 * eye
        xi = torch.linalg.solve(H, Aw.T @ s64)
    return xi


def apply_twist(params51: torch.Tensor, R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """`R_g <- R R_g`, `t <- t + v` on `(..., 51)` states: a camera-frame rotation about the wrist and a
    translation. Only the six root entries change."""
    out = params51.clone()
    Rg = so3_exp(params51[..., 3:6].double())
    out[..., 3:6] = so3_log(R.double() @ Rg).to(params51.dtype)
    out[..., 0:3] = params51[..., 0:3] + v.to(params51.dtype)
    return out


def interpolate_state(a51: torch.Tensor, b51: torch.Tensor, tau: float) -> torch.Tensor:
    """The state a fraction `tau` of the way from `a51` to `b51` (51,): the root rotation along the
    camera-frame geodesic, everything else linearly."""
    Ra, Rb = so3_exp(a51[3:6].double()), so3_exp(b51[3:6].double())
    out = a51 + tau * (b51 - a51)
    out[3:6] = so3_log(so3_exp(tau * so3_log(Rb @ Ra.T)) @ Ra).to(a51.dtype)
    return out


def twist6(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """`(6,)` float64 `[rotation vector (rad), translation (m)]` of an accumulated twist."""
    return torch.cat([so3_log(R.double()), v.double()])


@torch.no_grad()
def root_correction(fk: Callable[[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]],
                    anchors: torch.Tensor, groups: Sequence[torch.Tensor], px: torch.Tensor,
                    py: torch.Tensor, K: Intrinsics, topo: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                    P: GeoRootParams = GeoRootParams()) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """One persistent root twist explaining every event group against the contour of its anchor state.

    fk: `(N, 51) -> (verts (N, V, 3), joints (N, J, 3))`, camera frame, metres, joint 0 the wrist.
    anchors: `(G, 51)`; groups: `G` index tensors into `px`, `py` (event pixels).
    topo: `(faces, edges, adj)` from :func:`edge_topology`.
    Returns `(R (3, 3), v (3,), n_used)` in float64, applied with :func:`apply_twist`. Gauss-Newton:
    each iteration re-linearises every anchor after the twist found so far; a rotation about the
    updated wrist composes as `R <- Exp(omega) R`, `v <- v + dv`.
    """
    faces, edges, adj = topo
    dev = anchors.device
    R = torch.eye(3, dtype=torch.float64, device=dev)
    v = torch.zeros(3, dtype=torch.float64, device=dev)
    n_used = 0
    for _ in range(P.gn_iters):
        verts, joints = fk(apply_twist(anchors, R, v))
        rows, res = [], []
        for g, idx in enumerate(groups):
            if idx.numel() == 0:
                continue
            C = occluding_contour(verts[g], faces, edges, adj, K, P.occlusion_tol_m)
            keep, s, X, n = associate(px[idx], py[idx], C, P.band_px)
            if bool(keep.any()):
                rows.append(jacobian_rows(X[keep], n[keep], joints[g, 0], K))
                res.append(s[keep])
        n_used = int(sum(r.shape[0] for r in rows))
        if n_used < P.min_events:
            break
        xi = solve_twist(torch.cat(rows), torch.cat(res), P)
        R = so3_exp(xi[:3]) @ R
        v = v + xi[3:]
    return R, v, n_used
