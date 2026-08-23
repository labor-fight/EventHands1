#!/usr/bin/env python3
r"""S8 projection Jacobian: how a pixel moves when the kinematic state moves.

Everything downstream needs this one object. The event residual of S6 is

    r_i(x) = n_i^T ( u_i - pi_K( X_i(x) ) ),

so its derivative is `-n_i^T (d pi / d X) (d X_i / d x)`, and the Fisher information that S9 routes
on is `Lambda = sum_i J_i^T R_i^-1 J_i`. Getting `J` wrong does not produce an obviously broken
result -- it produces a router that confidently updates the wrong joints. Hence this module is
written twice over: an analytic form, and an autograd form used only as a reference, with a test
that they agree and a central-difference check on top of both.

## Conventions

The tangent space is the one S3 fixed, and no other: `x = [xi_root(6) | dphi(45)]`, right
multiplicative, `R <- R Exp(dphi)`. Two consequences that are easy to get wrong.

*The root pivot is `p_root`, not the rest-pose wrist joint.* `state_from_51d` folds the LBS pivot
into `p_root` so that `(R_root, p_root)` is a genuine SE(3) element, which means a root rotation
holds `p_root` fixed. Pivoting at `J_0 + t` instead -- the natural reading of MANO's own transform
chain -- gives a different and inconsistent perturbation.

*The chain rule runs per joint contribution, not through the blended point.* The usual textbook
form writes `dX_i/dphi_k = sum_j w_ij 1[k in anc(j)] omega_k x (X_i - c_k)` using the final blended
position `X_i`. The exact statement uses the contribution `A_j vbar_i` of each joint separately,

    dX_i/d dphi_k = - sum_j w_ij 1[k in anc(j)] skew( A_j vbar_i - c_k ) R^world_k ,

which is what is implemented. On the seam between two parts the two forms differ, because there the
per-joint contributions are genuinely at different places. `jacobian_report` measures the gap
instead of asserting it is small.

## The term the standard formula leaves out

MANO is not rigid LBS. `v_posed = v_shaped + P vec(R_1..15 - I)`, so the *rest* vertex fed to the
skinning matrices is itself a function of the joint rotations, and a joint rotation moves a vertex
through two paths:

    dX_i/d dphi_k = [ d/d dphi_k of the skinning transforms ] vbar_i  +  Rbar_i d vbar_i/d dphi_k .

The second term is absent from every articulated-tracking Jacobian in the hand literature, which
predates pose blendshapes. It is implemented here (`include_posedirs=True`) because autograd
includes it and the two must agree; `jacobian_report` reports its relative size so that dropping it
is a measured decision rather than an oversight.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

from . import lie

#: `[rho(3) | phi(3) | dphi(45)]`
DOF = lie.LieState.DOF
ROOT_TRANS = slice(0, 3)
ROOT_ROT = slice(3, 6)
JOINTS = slice(6, 51)


def ancestor_mask(parents: torch.Tensor) -> torch.Tensor:
    """`(16, 16)` boolean, `mask[k, j]` = joint `k` lies on the chain from the root to `j`.

    Includes `k == j`. Row 0 is therefore all-true, but the root has its own 6-dimensional block
    and the joint blocks only use rows 1..15.
    """
    n = int(parents.shape[0])
    m = torch.zeros(n, n, dtype=torch.bool, device=parents.device)
    for j in range(n):
        k = j
        while k >= 0:
            m[k, j] = True
            k = int(parents[k])
    return m


@dataclass
class ManoFK:
    """Forward kinematics of a `LieState`, in the pieces the Jacobian needs.

    `A` is MANO's own per-joint skinning matrix, so that `X_i = sum_j w_ij A_j vbar_i` reproduces
    `ManoLayer.forward` exactly. `pivot[j]` is the point a right perturbation of joint `j` holds
    fixed, and `Rw[j]` is the frame its rotation axis lives in.
    """

    A: torch.Tensor          # (B, 16, 3, 4)
    Rw: torch.Tensor         # (B, 16, 3, 3)
    pivot: torch.Tensor      # (B, 16, 3)
    v_posed: torch.Tensor    # (B, 778, 3)
    verts: torch.Tensor      # (B, 778, 3)
    R_joints: torch.Tensor   # (B, 15, 3, 3)
    R_root: torch.Tensor     # (B, 3, 3)
    p_root: torch.Tensor     # (B, 3)
    anc: torch.Tensor        # (16, 16)


def forward_kinematics(mano, state: lie.LieState, betas: torch.Tensor) -> ManoFK:
    """Rebuild MANO's transform chain from a `LieState`.

    The one non-obvious step is recovering MANO's joint-0 transform, which carries the rest-pose
    wrist as its translation: `G_0 = [R_root | p_root + R_root j0]`. That identity is what makes
    `p_root` the root pivot, and it is checked against `ManoLayer` in `tests/test_s8_jacobian.py`.
    """
    B = state.batch_size
    dev, dt = state.R_root.device, state.R_root.dtype
    parents = mano.parents

    v_shaped = mano.v_template + torch.einsum("bl,mkl->bmk", betas, mano.shapedirs)
    J = torch.einsum("bik,ji->bjk", v_shaped, mano.J_regressor)          # (B, 16, 3)
    I3 = torch.eye(3, device=dev, dtype=dt)
    pose_feature = (state.R_joints - I3).reshape(B, -1)                  # (B, 135)
    v_posed = v_shaped + torch.einsum("bl,mkl->bmk", pose_feature, mano.posedirs)

    R_all = torch.cat([state.R_root[:, None], state.R_joints], 1)        # (B, 16, 3, 3)
    rel = J.clone()
    rel[:, 1:] -= J[:, parents[1:]]
    # Joint 0's translation in the chain is the rest wrist plus the world offset, which the state
    # stores folded into `p_root`.
    rel = rel.clone()
    rel[:, 0] = state.p_root + (state.R_root @ state.j0[..., None])[..., 0]

    Rw = torch.empty_like(R_all)
    pw = torch.empty(B, 16, 3, device=dev, dtype=dt)
    Rw[:, 0] = R_all[:, 0]
    pw[:, 0] = rel[:, 0]
    for j in range(1, 16):
        p = int(parents[j])
        Rw[:, j] = Rw[:, p] @ R_all[:, j]
        pw[:, j] = pw[:, p] + (Rw[:, p] @ rel[:, j][..., None])[..., 0]

    # A_j = G_j [I | -J_j]: MANO's skinning matrix, which maps the *rest* vertex.
    A = torch.cat([Rw, (pw - (Rw @ J[..., None])[..., 0])[..., None]], -1)  # (B,16,3,4)

    verts = None
    pivot = pw.clone()
    # A right perturbation of the root holds `p_root` fixed, not the chain's joint-0 translation.
    pivot[:, 0] = state.p_root
    return ManoFK(A=A, Rw=Rw, pivot=pivot, v_posed=v_posed, verts=verts,
                  R_joints=state.R_joints, R_root=state.R_root, p_root=state.p_root,
                  anc=ancestor_mask(parents))


def skin(fk: ManoFK, weights: torch.Tensor, batch_index: torch.Tensor,
         vertex_index: torch.Tensor) -> torch.Tensor:
    """`X = sum_j w_j A_j vbar` for a flat list of `(batch, vertex)` pairs."""
    vbar = fk.v_posed[batch_index, vertex_index]                          # (N, 3)
    A = fk.A[batch_index]                                                 # (N, 16, 3, 4)
    P = (A[..., :3] @ vbar[:, None, :, None])[..., 0] + A[..., 3]          # (N, 16, 3)
    return (weights[..., None] * P).sum(1)


def vertex_jacobian(mano, fk: ManoFK, batch_index: torch.Tensor, vertex_index: torch.Tensor,
                    include_posedirs: bool = True, chunk: int = 200_000
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`dX/dx` for a flat list of `(batch, vertex)` pairs.

    Returns the points `(N, 3)` and their Jacobians `(N, 3, 51)`.
    """
    dev, dt = fk.A.device, fk.A.dtype
    N = batch_index.shape[0]
    X_out = torch.empty(N, 3, device=dev, dtype=dt)
    Jc = torch.empty(N, 3, DOF, device=dev, dtype=dt)
    E = torch.stack([lie.skew(torch.eye(3, device=dev, dtype=dt)[a]) for a in range(3)])  # (3,3,3)

    for c0 in range(0, N, chunk):
        sl = slice(c0, min(c0 + chunk, N))
        b, v = batch_index[sl], vertex_index[sl]
        n = b.shape[0]
        w = mano.weights[v]                                               # (n, 16)
        vbar = fk.v_posed[b, v]                                           # (n, 3)
        A = fk.A[b]                                                       # (n, 16, 3, 4)
        Rw = fk.Rw[b]                                                     # (n, 16, 3, 3)
        piv = fk.pivot[b]                                                 # (n, 16, 3)

        P = (A[..., :3] @ vbar[:, None, :, None])[..., 0] + A[..., 3]      # (n, 16, 3)
        X = (w[..., None] * P).sum(1)
        X_out[sl] = X

        # M[k, j] = w_j 1[k on the chain to j]. Summing the per-joint contributions through M is
        # what makes the exact per-contribution form as cheap as the approximate blended one.
        M = w[:, None, :] * fk.anc[None, :, :].to(dt)                     # (n, 16, 16)
        S = M @ P                                                         # (n, 16, 3)
        Wk = M.sum(-1)                                                    # (n, 16)
        arm = S - Wk[..., None] * piv                                     # (n, 16, 3)
        block = -lie.skew(arm.reshape(-1, 3)).reshape(n, 16, 3, 3) @ Rw    # (n, 16, 3, 3)

        Jl = torch.zeros(n, 3, DOF, device=dev, dtype=dt)
        # Root translation: p_root enters X additively through every A_j.
        Jl[:, :, ROOT_TRANS] = fk.R_root[b]
        Jl[:, :, ROOT_ROT] = block[:, 0]
        Jl[:, :, JOINTS] = block[:, 1:].permute(0, 2, 1, 3).reshape(n, 3, 45)

        if include_posedirs:
            # vbar itself depends on the joint rotations. `Mk[:, k]` holds d vec(R_k)/d dphi_k as a
            # (9, 3) matrix, and posedirs maps that to a vertex displacement.
            Rj = fk.R_joints[b]                                           # (n, 15, 3, 3)
            dR = Rj[:, :, None] @ E[None, None]                           # (n, 15, 3, 3, 3)
            Mk = dR.reshape(n, 15, 3, 9).permute(0, 1, 3, 2)              # (n, 15, 9, 3)
            pd = mano.posedirs[v].reshape(n, 3, 15, 9).permute(0, 2, 1, 3)  # (n, 15, 3, 9)
            D = pd @ Mk                                                   # (n, 15, 3, 3)
            Rbar = (w[..., None, None] * A[..., :3]).sum(1)                # (n, 3, 3)
            Jl[:, :, JOINTS] += (Rbar[:, None] @ D).permute(0, 2, 1, 3).reshape(n, 3, 45)

        Jc[sl] = Jl
    return X_out, Jc


def project_jacobian(X: torch.Tensor, camera_K: torch.Tensor, batch_index: torch.Tensor,
                     render_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pinhole projection and its `(N, 2, 3)` derivative, at the render resolution."""
    K = camera_K[batch_index]
    fx, fy = K[:, 0, 0] * render_scale, K[:, 1, 1] * render_scale
    cx, cy = K[:, 0, 2] * render_scale, K[:, 1, 2] * render_scale
    x, y, z = X[:, 0], X[:, 1], X[:, 2].clamp_min(1e-6)
    uv = torch.stack([fx * x / z + cx, fy * y / z + cy], -1)
    zero = torch.zeros_like(fx)
    d = torch.stack([torch.stack([fx / z, zero, -fx * x / z ** 2], -1),
                     torch.stack([zero, fy / z, -fy * y / z ** 2], -1)], -2)
    return uv, d


@dataclass
class PixelJacobian:
    """Per-query projection Jacobian and the contour-normal residual row built from it."""

    X: torch.Tensor          # (N, 3) surface point
    uv: torch.Tensor         # (N, 2) projected pixel
    J_uv: torch.Tensor       # (N, 2, 51) d pixel / d state
    J_r: Optional[torch.Tensor] = None   # (N, 51) d residual / d state
    r: Optional[torch.Tensor] = None     # (N,) residual value


def query_jacobian(mano, fk: ManoFK, camera_K: torch.Tensor, render_scale: float,
                   batch_index: torch.Tensor, face_id: torch.Tensor, bary: torch.Tensor,
                   normal: Optional[torch.Tensor] = None,
                   pixel: Optional[torch.Tensor] = None,
                   include_posedirs: bool = True) -> PixelJacobian:
    """Jacobian at KSSF query points, given `(face_id, bary)` from the field.

    The query point is treated as the barycentric combination of its three vertices, each skinned
    on its own, which is exactly what `kssf.surface_point` computes. Interpolating the rest
    vertices first and skinning once would be a different -- and wrong -- surface.
    """
    f = mano.f.long()[face_id.clamp_min(0)]                               # (N, 3)
    N = f.shape[0]
    b3 = batch_index[:, None].expand(N, 3).reshape(-1)
    X_v, J_v = vertex_jacobian(mano, fk, b3, f.reshape(-1),
                              include_posedirs=include_posedirs)
    w = bary.reshape(-1, 1)
    X = (X_v * w).reshape(N, 3, 3).sum(1)
    Jx = (J_v * w[..., None]).reshape(N, 3, 3, DOF).sum(1)                # (N, 3, 51)

    uv, dpi = project_jacobian(X, camera_K, batch_index, render_scale)
    J_uv = dpi @ Jx                                                       # (N, 2, 51)
    out = PixelJacobian(X=X, uv=uv, J_uv=J_uv)
    if normal is not None and pixel is not None:
        out.r = (normal * (pixel - uv)).sum(-1)
        # d/dx of n^T (u - pi(X)) at fixed n: the normal is evaluated at the previous state, so it
        # is data here, not a function of x.
        out.J_r = -(normal[:, None, :] @ J_uv)[:, 0, :]
    return out


# --------------------------------------------------------------------------- information
#: Column blocks of the tangent space: the 6 root degrees of freedom, then one per joint.
GROUPS: Tuple[Tuple[str, slice], ...] = (("root", slice(0, 6)),) + tuple(
    (f"j{k}", slice(6 + 3 * k, 9 + 3 * k)) for k in range(15))


def fisher(J_r: torch.Tensor, sigma: Optional[torch.Tensor] = None,
           batch_index: Optional[torch.Tensor] = None, n_batch: int = 1) -> torch.Tensor:
    r"""`Lambda = sum_i J_i^T R_i^-1 J_i` from scalar residual rows.

    Accumulated in float64 whatever the input precision. The sum runs over tens of thousands of
    events whose individual contributions are rank one and tiny; in float32 the small eigenvalues
    -- exactly the ones that decide whether a joint is observable -- are lost to accumulation
    error, which would make the router's decisions an artefact of arithmetic.
    """
    Jd = J_r.double()
    if sigma is not None:
        Jd = Jd / sigma.double().clamp_min(1e-12)[:, None]
    if batch_index is None:
        return Jd.T @ Jd
    out = torch.zeros(n_batch, DOF, DOF, device=J_r.device, dtype=torch.float64)
    for b in range(n_batch):
        m = batch_index == b
        if bool(m.any()):
            Jb = Jd[m]
            out[b] = Jb.T @ Jb
    return out


def schur_eliminate(Lam: torch.Tensor, keep: slice, drop: slice,
                    prior_drop: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""Nuisance-adjusted information `Lam_keep - Lam_kd (Lam_dd + P^-1)^-1 Lam_dk`.

    This is the whole point of the root-Schur step in S9. `Lam_keep` alone answers "how much do the
    events say about this finger if the wrist is known exactly", which is the wrong question during
    tracking: the wrist is not known, and much of what looks like finger information is really
    unresolved wrist motion. The Schur complement subtracts exactly the part that the root could
    explain, leaving what the finger alone is responsible for.
    """
    A = Lam[..., keep, keep]
    Bm = Lam[..., keep, drop]
    D = Lam[..., drop, drop]
    if prior_drop is not None:
        D = D + torch.linalg.inv(prior_drop)
    n = D.shape[-1]
    I = torch.eye(n, device=D.device, dtype=D.dtype).expand_as(D)
    # Ridge-stabilised solve rather than an inverse: `D` is singular whenever the root is
    # unobservable, which is a legitimate state of affairs and not an error to raise on.
    eps = 1e-12 * torch.diagonal(D, dim1=-2, dim2=-1).abs().amax(-1)[..., None, None].clamp_min(1.0)
    return A - Bm @ torch.linalg.solve(D + eps * I, Bm.transpose(-1, -2))


def information_gain(Lam: torch.Tensor, prior: Optional[torch.Tensor] = None) -> torch.Tensor:
    r"""`log det(I + P^1/2 Lambda P^1/2)`, the expected entropy reduction of an update.

    With `P = I` this is `log det(I + Lambda)`, which is the right scale-free summary: it saturates
    for directions already well determined and grows only where the prior is still uncertain.
    """
    n = Lam.shape[-1]
    I = torch.eye(n, device=Lam.device, dtype=Lam.dtype).expand_as(Lam)
    M = Lam if prior is None else prior @ Lam
    # Symmetrise before the determinant: `Lam` is symmetric by construction but the product with a
    # prior is not, and `slogdet` on a slightly asymmetric matrix is not the quantity intended.
    M = 0.5 * (M + M.transpose(-1, -2))
    sign, logabs = torch.linalg.slogdet(I + M)
    return torch.where(sign > 0, logabs, torch.full_like(logabs, float("-inf")))


def group_information(Lam: torch.Tensor, eliminate_root: bool = True,
                      prior_root: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Per-group information gain, optionally after eliminating the root as a nuisance."""
    out: Dict[str, torch.Tensor] = {}
    for name, sl in GROUPS:
        if name == "root":
            out[name] = information_gain(Lam[..., sl, sl])
            continue
        if eliminate_root:
            block = schur_eliminate(Lam, sl, GROUPS[0][1], prior_drop=prior_root)
        else:
            block = Lam[..., sl, sl]
        out[name] = information_gain(block)
    return out


# --------------------------------------------------------------------------- reference
def autograd_vertex_jacobian(mano, params51: torch.Tensor, betas: torch.Tensor,
                             batch_index: torch.Tensor, vertex_index: torch.Tensor
                             ) -> torch.Tensor:
    """`dX/dx` by differentiating `ManoLayer` itself. Reference only.

    Differentiating the real layer, rather than a reimplementation, is the point: it is the same
    code path the loss uses, so agreement means the analytic form describes the model actually
    being fitted, pose blendshapes and all.

    Each query point gets its own row of the perturbation, so the state is replicated to one sample
    per point. That is not an optimisation but a correctness requirement: with a shared `delta`,
    `autograd` returns the gradient summed over every point drawn from the same batch element,
    which silently produces a reference that is wrong by a factor of however many points collided.
    Replicating also means the whole Jacobian comes out of three backward passes rather than 51.
    """
    from pose_repr import decode_to_mano_inputs

    dev, dt = params51.device, params51.dtype
    N = batch_index.shape[0]
    j0 = (mano.J_regressor @ (mano.v_template
                              + torch.einsum("bl,mkl->bmk", betas, mano.shapedirs)))[:, 0]
    p_rep = params51[batch_index]
    b_rep = betas[batch_index]
    j0_rep = j0[batch_index]

    delta = torch.zeros(N, DOF, device=dev, dtype=dt, requires_grad=True)
    p = lie.retract_51d(p_rep, delta, mano.hands_mean, j0_rep)
    dec = decode_to_mano_inputs(p, "mano_full_axis_angle",
                                mano.hands_components, mano.hands_mean)
    verts, _ = mano(b_rep, dec["global_orient"], dec["local_full_aa"], dec["transl"])
    sel = verts[torch.arange(N, device=dev), vertex_index]                 # (N, 3)
    cols = []
    for c in range(3):
        gr, = torch.autograd.grad(sel[:, c].sum(), delta, retain_graph=(c < 2))
        cols.append(gr)
    return torch.stack(cols, 1)                                           # (N, 3, DOF)


def jacobian_report(mano, fk: ManoFK, batch_index: torch.Tensor, vertex_index: torch.Tensor
                    ) -> Dict[str, float]:
    """Sizes of the two terms the literature's formula drops, relative to the full Jacobian."""
    _, J_full = vertex_jacobian(mano, fk, batch_index, vertex_index, include_posedirs=True)
    _, J_rigid = vertex_jacobian(mano, fk, batch_index, vertex_index, include_posedirs=False)
    scale = J_full.norm(dim=(-2, -1)).clamp_min(1e-12)
    pose_term = (J_full - J_rigid).norm(dim=(-2, -1))

    # The blended-point approximation: use the final X in place of each joint contribution.
    dev, dt = fk.A.device, fk.A.dtype
    w = mano.weights[vertex_index]
    vbar = fk.v_posed[batch_index, vertex_index]
    A = fk.A[batch_index]
    P = (A[..., :3] @ vbar[:, None, :, None])[..., 0] + A[..., 3]
    X = (w[..., None] * P).sum(1)
    M = w[:, None, :] * fk.anc[None].to(dt)
    Wk = M.sum(-1)
    arm_approx = Wk[..., None] * (X[:, None, :] - fk.pivot[batch_index])
    arm_exact = M @ P - Wk[..., None] * fk.pivot[batch_index]
    n = batch_index.shape[0]
    blk_a = -lie.skew(arm_approx.reshape(-1, 3)).reshape(n, 16, 3, 3) @ fk.Rw[batch_index]
    blk_e = -lie.skew(arm_exact.reshape(-1, 3)).reshape(n, 16, 3, 3) @ fk.Rw[batch_index]
    blend_term = (blk_a - blk_e).norm(dim=(-2, -1)).norm(dim=-1)

    return {
        "n": int(n),
        "jac_norm_median": float(scale.median()),
        "posedirs_rel_median": float((pose_term / scale).median()),
        "posedirs_rel_p99": float(torch.quantile((pose_term / scale).float(), 0.99)),
        "blended_approx_rel_median": float((blend_term / scale).median()),
        "blended_approx_rel_p99": float(torch.quantile((blend_term / scale).float(), 0.99)),
    }
