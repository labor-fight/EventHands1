#!/usr/bin/env python3
r"""S3 Lie group state: SO(3)/SE(3) exp, log, Jacobians, and a lossless 51D adapter.

Why the 51D vector is not a state. `[t, R_root, residual45]` stacks metres and axis-angles in one
array, so adding two of them, averaging them, or putting a Gaussian on them are all
ill-defined -- axis-angle addition is not rotation composition, and the error it makes grows with
the angle. Every stage after this one needs at least one of those operations: S9 measures
information in a tangent space, S12 propagates a covariance, S13 fuses two estimates. So the
state is carried on the manifold and the 51D layout survives only as a serialisation format.

The state is `(T, {R_j})` with `T` in SE(3) and 15 joint rotations in SO(3).

**The root really is an SE(3) element**, which takes one step to see. MANO's root transform is a
rotation about the shape-dependent pivot `j0 = J(beta)[0]` (`ManoLayer.forward` contributes
`j0 - R_g j0` at the root), so

    X_cam = R_g (X_rest - j0) + j0 + t = R_g X_rest + (t + (I - R_g) j0)

which is `X_cam = R X_rest + p` for

    R = R_g,        p = t + (I - R_g) j0                  (51D -> SE(3))
    R_g = R,        t = p - (I - R) j0                    (SE(3) -> 51D)

a bijection for a given `beta`. Skipping the pivot term and writing `p = t` would be the same
mistake the domain-randomisation label mirroring had to avoid, and it is why the adapter requires
`betas` instead of defaulting them.

**Joint rotations are full rotations, not residuals.** The 51D local block is a residual against
`hands_mean`, and MANO applies `Rodrigues(residual_j + mean_j)`, so the state stores
`R_j = Exp(residual_j + mean_j)` and inverts with `residual_j = Log(R_j) - mean_j`. The
round-trip is exact only while `|residual_j + mean_j| < pi`;
:func:`check_axis_angle_range` measures the real margin on real data instead of assuming it.

**Update convention, fixed here and never mixed** -- right multiplication, i.e. perturbations in
the body frame:

    R_j <- R_j Exp(delta phi_j),        T <- T Exp(xi)

Right multiplication is chosen because the S8 kinematic Jacobian columns come out in each joint's
own frame, which is what makes the per-joint routing in S9 meaningful: `delta phi_j` is "how much
this joint rotates about its own axes", independent of where the wrist happens to point. Adding
axis-angles directly (`aa + delta`) is forbidden and :func:`assert_no_axis_angle_addition`
documents the size of the error it would introduce.

Conventions follow Solà, Deray and Atchuthan, "A micro Lie theory for state estimation in
robotics" (2018), whose `Jr`, `Jl` and `V` definitions are the ones used below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

#: Below this angle the series expansions are used instead of the trigonometric closed forms.
#: Chosen so the neglected term is at the float32 round-off level: the first dropped term in
#: `sin(t)/t` is `t^4/120`, which is under 1e-8 for t = 1e-2.
EPS_ANGLE = 1e-4
_TAYLOR_CUT = 1e-2


def skew(v: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 3, 3) with `skew(v) @ w == cross(v, w)`."""
    z = torch.zeros_like(v[..., 0])
    x, y, w = v[..., 0], v[..., 1], v[..., 2]
    return torch.stack([
        torch.stack([z, -w, y], -1),
        torch.stack([w, z, -x], -1),
        torch.stack([-y, x, z], -1),
    ], -2)


def unskew(M: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3), the inverse of :func:`skew` on skew-symmetric input."""
    return torch.stack([M[..., 2, 1], M[..., 0, 2], M[..., 1, 0]], -1)


def _theta(v: torch.Tensor) -> torch.Tensor:
    """`|v|` computed through a clamped square so the gradient at 0 is finite."""
    return (v * v).sum(-1, keepdim=True).clamp_min(1e-30).sqrt()


def so3_exp(phi: torch.Tensor) -> torch.Tensor:
    """Rodrigues: (..., 3) axis-angle -> (..., 3, 3) rotation.

    `R = I + a [phi]x + b [phi]x^2` with `a = sin(t)/t`, `b = (1-cos t)/t^2`; both coefficients
    are replaced by their series near the identity, where the closed forms are `0/0`.
    """
    t = _theta(phi)
    t2 = t * t
    small = t < _TAYLOR_CUT
    a = torch.where(small, 1.0 - t2 / 6.0 + t2 * t2 / 120.0, torch.sin(t) / t)
    b = torch.where(small, 0.5 - t2 / 24.0 + t2 * t2 / 720.0,
                    (1.0 - torch.cos(t)) / t2)
    K = skew(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    return I + a[..., None] * K + b[..., None] * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) -> (..., 3) axis-angle, well conditioned including near `theta = pi`.

    Goes through a quaternion using Shepperd's largest-component selection. The direct formula
    `theta/(2 sin theta) (R - R^T)` loses all precision as `sin theta -> 0` at `theta = pi`,
    which is exactly where a hand joint at full flexion combined with `hands_mean` can land; the
    quaternion route has no such point because one of the four components is always bounded away
    from zero.
    """
    m = R
    tr = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    c = torch.stack([tr, m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]], -1)
    case = c.argmax(-1)

    def q_from(i: int) -> torch.Tensor:
        if i == 0:
            s = (1.0 + tr).clamp_min(1e-20).sqrt() * 2.0
            return torch.stack([0.25 * s,
                                (m[..., 2, 1] - m[..., 1, 2]) / s,
                                (m[..., 0, 2] - m[..., 2, 0]) / s,
                                (m[..., 1, 0] - m[..., 0, 1]) / s], -1)
        if i == 1:
            s = (1.0 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2]).clamp_min(1e-20).sqrt() * 2.0
            return torch.stack([(m[..., 2, 1] - m[..., 1, 2]) / s, 0.25 * s,
                                (m[..., 0, 1] + m[..., 1, 0]) / s,
                                (m[..., 0, 2] + m[..., 2, 0]) / s], -1)
        if i == 2:
            s = (1.0 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2]).clamp_min(1e-20).sqrt() * 2.0
            return torch.stack([(m[..., 0, 2] - m[..., 2, 0]) / s,
                                (m[..., 0, 1] + m[..., 1, 0]) / s, 0.25 * s,
                                (m[..., 1, 2] + m[..., 2, 1]) / s], -1)
        s = (1.0 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2]).clamp_min(1e-20).sqrt() * 2.0
        return torch.stack([(m[..., 1, 0] - m[..., 0, 1]) / s,
                            (m[..., 0, 2] + m[..., 2, 0]) / s,
                            (m[..., 1, 2] + m[..., 2, 1]) / s, 0.25 * s], -1)

    q = q_from(0)
    for i in (1, 2, 3):
        q = torch.where((case == i)[..., None], q_from(i), q)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-20)
    # Canonical hemisphere: q and -q are the same rotation, but only w >= 0 gives |phi| <= pi.
    q = torch.where((q[..., :1] < 0), -q, q)
    w, v = q[..., :1], q[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(n, w.clamp(-1.0, 1.0))
    small = n < 1e-8
    # For small n, angle/n -> 2/w, and w -> 1; avoids 0/0 without branching on the value.
    scale = torch.where(small, 2.0 / w.clamp_min(1e-20), angle / n.clamp_min(1e-20))
    return v * scale


def so3_Jr(phi: torch.Tensor) -> torch.Tensor:
    """Right Jacobian: `Exp(phi + dphi) ~ Exp(phi) Exp(Jr(phi) dphi)`."""
    t = _theta(phi)
    t2 = t * t
    small = t < _TAYLOR_CUT
    a = torch.where(small, -0.5 + t2 / 24.0, -(1.0 - torch.cos(t)) / t2)
    b = torch.where(small, 1.0 / 6.0 - t2 / 120.0, (t - torch.sin(t)) / (t2 * t))
    K = skew(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    return I + a[..., None] * K + b[..., None] * (K @ K)


def so3_Jl(phi: torch.Tensor) -> torch.Tensor:
    """Left Jacobian; `Jl(phi) == Jr(-phi)`."""
    return so3_Jr(-phi)


def _half_cot_coeff(t: torch.Tensor) -> torch.Tensor:
    r"""`1/t^2 - cot(t/2)/(2t)`, the quadratic coefficient of `Jr^-1` and `V^-1`.

    The textbook form is `1/t^2 - (1 + cos t) / (2 t sin t)`, which is written here through the
    half-angle identity `(1 + cos t)/sin t = cot(t/2)`. That is not cosmetic: the original has
    `sin t` in the denominator, so it is singular at `t = pi` (a `0/0` that a naive `clamp_min`
    silently turns into a huge number, and that a sign-blind clamp gets *wrong* for `t > pi`
    where `sin t < 0`). The half-angle form only degenerates at `t = 0` and `t = 2 pi`, and
    `t = pi` -- a joint at half a turn, which real hand poses do approach -- evaluates cleanly to
    `1/pi^2`.

    Valid on `(0, 2 pi)`, which covers everything `so3_log` can return since its range is
    `|phi| <= pi`.
    """
    t2 = t * t
    small = t < _TAYLOR_CUT
    half = 0.5 * t
    # sin(t/2) > 0 throughout (0, 2 pi), so a magnitude clamp cannot flip a sign here.
    cot_half = torch.cos(half) / torch.sin(half).clamp_min(1e-20)
    return torch.where(small, 1.0 / 12.0 + t2 / 720.0, 1.0 / t2 - cot_half / (2.0 * t))


def so3_Jr_inv(phi: torch.Tensor) -> torch.Tensor:
    """Inverse right Jacobian, the map from a group increment to a tangent increment."""
    b = _half_cot_coeff(_theta(phi))
    K = skew(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    return I + 0.5 * K + b[..., None] * (K @ K)


def se3_V(phi: torch.Tensor) -> torch.Tensor:
    """The `V` block of `SE(3)` exp: `Exp([rho, phi]) = [[Exp(phi), V(phi) rho], [0, 1]]`."""
    t = _theta(phi)
    t2 = t * t
    small = t < _TAYLOR_CUT
    a = torch.where(small, 0.5 - t2 / 24.0, (1.0 - torch.cos(t)) / t2)
    b = torch.where(small, 1.0 / 6.0 - t2 / 120.0, (t - torch.sin(t)) / (t2 * t))
    K = skew(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    return I + a[..., None] * K + b[..., None] * (K @ K)


def se3_V_inv(phi: torch.Tensor) -> torch.Tensor:
    b = _half_cot_coeff(_theta(phi))
    K = skew(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(K.shape)
    return I - 0.5 * K + b[..., None] * (K @ K)


def se3_exp(xi: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(..., 6) `[rho(3), phi(3)]` -> `(R, p)`. Translation first, matching `Adjoint` below."""
    rho, phi = xi[..., 0:3], xi[..., 3:6]
    R = so3_exp(phi)
    p = (se3_V(phi) @ rho[..., None])[..., 0]
    return R, p


def se3_log(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """`(R, p)` -> (..., 6) `[rho, phi]`."""
    phi = so3_log(R)
    rho = (se3_V_inv(phi) @ p[..., None])[..., 0]
    return torch.cat([rho, phi], -1)


def se3_adjoint(R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """`Adj(T)` for the `[rho, phi]` ordering: `[[R, [p]x R], [0, R]]`."""
    Z = torch.zeros_like(R)
    top = torch.cat([R, skew(p) @ R], -1)
    bot = torch.cat([Z, R], -1)
    return torch.cat([top, bot], -2)


def _se3_Q(xi: torch.Tensor) -> torch.Tensor:
    r"""The coupling block of the `SE(3)` left Jacobian (Barfoot, *State Estimation*, eq. 7.86).

    `J_l^{SE(3)}([rho, phi]) = [[J_l(phi), Q(rho, phi)], [0, J_l(phi)]]`. `Q` is what makes the
    translational and rotational parts of a rigid increment interact: a rotation about a point
    offset from the origin translates it, so a perturbation of `phi` moves `rho` too. Dropping it
    -- the common shortcut of treating `SE(3)` as `R^3 x SO(3)` -- makes the wrist's covariance
    wrong by a term proportional to the lever arm, which for a hand at arm's length is not small.
    """
    rho, phi = xi[..., 0:3], xi[..., 3:6]
    t = _theta(phi)
    t2, t3, t4, t5 = t * t, t ** 3, t ** 4, t ** 5
    small = t < _TAYLOR_CUT
    s, c = torch.sin(t), torch.cos(t)
    # Series limits at `theta -> 0`, where the closed forms are 0/0.
    c1 = torch.where(small, 1.0 / 6.0 - t2 / 120.0, (t - s) / t3)
    c2 = torch.where(small, 1.0 / 24.0 - t2 / 720.0, (1.0 - t2 / 2.0 - c) / t4)
    c3 = torch.where(small, 1.0 / 120.0 - t2 / 2520.0, (t - s - t3 / 6.0) / t5)
    P, F = skew(rho), skew(phi)
    FP, PF = F @ P, P @ F
    FPF = F @ P @ F
    FF = F @ F
    return (0.5 * P
            + c1[..., None] * (FP + PF + FPF)
            - c2[..., None] * (FF @ P + P @ FF - 3.0 * FPF)
            - 0.5 * (c2 - 3.0 * c3)[..., None] * (FPF @ F + F @ FPF))


def se3_Jl(xi: torch.Tensor) -> torch.Tensor:
    """Left Jacobian of `SE(3)` exp, `(..., 6, 6)`, for the `[rho, phi]` ordering."""
    J = so3_Jl(xi[..., 3:6])
    Z = torch.zeros_like(J)
    return torch.cat([torch.cat([J, _se3_Q(xi)], -1), torch.cat([Z, J], -1)], -2)


def se3_Jr(xi: torch.Tensor) -> torch.Tensor:
    """Right Jacobian of `SE(3)` exp. `J_r(xi) = J_l(-xi)`, exactly as in `SO(3)`."""
    return se3_Jl(-xi)


# --------------------------------------------------------------------------- state
@dataclass
class LieState:
    """`(T, {R_j})` for a batch. `R_root`/`p_root` are the genuine SE(3) pair, pivot folded in."""

    R_root: torch.Tensor    # (B, 3, 3)
    p_root: torch.Tensor    # (B, 3)
    R_joints: torch.Tensor  # (B, 15, 3, 3)
    j0: torch.Tensor        # (B, 3) shape-dependent LBS pivot, part of the model not the state

    @property
    def batch_size(self) -> int:
        return int(self.R_root.shape[0])

    #: tangent dimension: 6 for the root plus 3 per joint
    DOF = 6 + 15 * 3

    def clone(self) -> "LieState":
        return LieState(self.R_root.clone(), self.p_root.clone(),
                        self.R_joints.clone(), self.j0.clone())

    def retract(self, delta: torch.Tensor) -> "LieState":
        """Right-multiplicative update. `delta` is `(B, 51)` = `[xi_root(6), dphi(45)]`.

        This is the only sanctioned way to move the state. `xi_root` is ordered
        `[rho, phi]` to match :func:`se3_exp`.
        """
        assert delta.shape[-1] == self.DOF, (delta.shape, self.DOF)
        dR, dp = se3_exp(delta[..., 0:6])
        R_new = self.R_root @ dR
        p_new = self.p_root + (self.R_root @ dp[..., None])[..., 0]
        dphi = delta[..., 6:].reshape(-1, 15, 3)
        Rj_new = self.R_joints @ so3_exp(dphi)
        return LieState(R_new, p_new, Rj_new, self.j0)

    def local_coordinates(self, other: "LieState") -> torch.Tensor:
        """`Log(self^-1 other)` per block: the tangent vector taking `self` to `other`."""
        Rrel = self.R_root.transpose(-1, -2) @ other.R_root
        prel = (self.R_root.transpose(-1, -2)
                @ (other.p_root - self.p_root)[..., None])[..., 0]
        xi = se3_log(Rrel, prel)
        dphi = so3_log(self.R_joints.transpose(-1, -2) @ other.R_joints)
        return torch.cat([xi, dphi.reshape(-1, 45)], -1)


def state_from_51d(params51: torch.Tensor, hands_mean: torch.Tensor,
                   j0: torch.Tensor) -> LieState:
    """`[t, R_root, residual45]` -> `LieState`. `j0` must be `J(beta)[0]`, not zero."""
    assert params51.shape[-1] == 51, params51.shape
    t = params51[..., 0:3]
    R_root = so3_exp(params51[..., 3:6])
    if j0.dim() == 1:
        j0 = j0.view(1, 3).expand(params51.shape[0], 3)
    I = torch.eye(3, dtype=R_root.dtype, device=R_root.device).expand(R_root.shape)
    p_root = t + ((I - R_root) @ j0[..., None])[..., 0]
    local_full = params51[..., 6:51] + hands_mean.reshape(1, 45).to(params51)
    R_joints = so3_exp(local_full.reshape(-1, 15, 3))
    return LieState(R_root, p_root, R_joints, j0)


def state_to_51d(state: LieState, hands_mean: torch.Tensor) -> torch.Tensor:
    """`LieState` -> `[t, R_root, residual45]`, the exact inverse of :func:`state_from_51d`."""
    I = torch.eye(3, dtype=state.R_root.dtype,
                  device=state.R_root.device).expand(state.R_root.shape)
    t = state.p_root - ((I - state.R_root) @ state.j0[..., None])[..., 0]
    r_root = so3_log(state.R_root)
    local_full = so3_log(state.R_joints).reshape(-1, 45)
    residual = local_full - hands_mean.reshape(1, 45).to(local_full)
    return torch.cat([t, r_root, residual], -1)


def retract_51d(params51: torch.Tensor, delta: torch.Tensor, hands_mean: torch.Tensor,
                j0: torch.Tensor) -> torch.Tensor:
    """Apply a tangent update to a 51D vector by going through the manifold.

    The drop-in replacement for `prev + delta`. Not the same operation: for a 0.3 rad step the
    two differ by several degrees per joint, and the discrepancy compounds over a recursive
    rollout. :func:`assert_no_axis_angle_addition` quantifies it.
    """
    return state_to_51d(state_from_51d(params51, hands_mean, j0).retract(delta), hands_mean)


def local_coordinates_51d(a51: torch.Tensor, b51: torch.Tensor, hands_mean: torch.Tensor,
                          j0: torch.Tensor) -> torch.Tensor:
    """Tangent vector taking `a51` to `b51`, the inverse of :func:`retract_51d` in its second slot.

    The residual a filter should use: `b51 - a51` is not a tangent vector, and for the wrist it is
    not even close, because a difference of axis-angle vectors ignores that the rotation is applied
    on the right.
    """
    A = state_from_51d(a51, hands_mean, j0)
    B = state_from_51d(b51, hands_mean, j0)
    return A.local_coordinates(B)


# --------------------------------------------------------------------------- checks
def check_axis_angle_range(params51: torch.Tensor, hands_mean: torch.Tensor) -> Dict[str, float]:
    """Margin to the `|phi| = pi` wrap, where the 51D round-trip stops being injective."""
    full = params51[..., 6:51] + hands_mean.reshape(1, 45).to(params51)
    n = full.reshape(-1, 15, 3).norm(dim=-1)
    root = params51[..., 3:6].norm(dim=-1)
    import math
    return {
        "joint_max_rad": float(n.max()),
        "joint_margin_to_pi_rad": float(math.pi - n.max()),
        "root_max_rad": float(root.max()),
        "root_margin_to_pi_rad": float(math.pi - root.max()),
        "n_joints_over_pi": int((n >= math.pi).sum()),
        "n_root_over_pi": int((root >= math.pi).sum()),
    }


def geodesic_error(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    """Rotation distance in radians: `|Log(Ra^T Rb)|`."""
    return so3_log(Ra.transpose(-1, -2) @ Rb).norm(dim=-1)


def assert_no_axis_angle_addition(params51: torch.Tensor, delta: torch.Tensor,
                                  hands_mean: torch.Tensor, j0: torch.Tensor
                                  ) -> Dict[str, float]:
    """How wrong `prev + delta` is versus the manifold update, in degrees and millimetres.

    Reported rather than asserted: the point is to record the size of the effect the convention
    change is expected to fix, so a later null result can be read against it.
    """
    exact = retract_51d(params51, delta, hands_mean, j0)
    naive = params51 + delta[..., torch.cat([
        torch.arange(3), torch.arange(3, 6), torch.arange(6, 51)]).to(delta.device)]
    Re = so3_exp(exact[..., 6:51].reshape(-1, 15, 3) + hands_mean.reshape(1, 1, 45)
                 .reshape(1, 15, 3).to(exact))
    Rn = so3_exp(naive[..., 6:51].reshape(-1, 15, 3) + hands_mean.reshape(1, 15, 3).to(naive))
    import math
    g = geodesic_error(Re, Rn)
    return {
        "joint_mean_deg": float(g.mean() * 180.0 / math.pi),
        "joint_max_deg": float(g.max() * 180.0 / math.pi),
        "transl_max_m": float((exact[..., 0:3] - naive[..., 0:3]).norm(dim=-1).max()),
        "root_max_deg": float(geodesic_error(so3_exp(exact[..., 3:6]),
                                             so3_exp(naive[..., 3:6])).max() * 180.0 / math.pi),
    }


def round_trip_report(params51: torch.Tensor, hands_mean: torch.Tensor, j0: torch.Tensor,
                      mano=None, betas: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Errors of `51D -> Lie -> 51D`, in tangent, geodesic and (optionally) mesh terms."""
    st = state_from_51d(params51, hands_mean, j0)
    back = state_to_51d(st, hands_mean)
    out = {
        "max_abs_51d": float((back - params51).abs().max()),
        "max_geodesic_root_rad": float(geodesic_error(so3_exp(params51[..., 3:6]),
                                                      st.R_root).max()),
        "max_geodesic_joint_rad": float(geodesic_error(
            so3_exp((params51[..., 6:51] + hands_mean.reshape(1, 45).to(params51))
                    .reshape(-1, 15, 3)), st.R_joints).max()),
        "max_transl_m": float((back[..., 0:3] - params51[..., 0:3]).abs().max()),
    }
    if mano is not None and betas is not None:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
        from pose_repr import decode_to_mano_inputs

        def verts(p):
            dec = decode_to_mano_inputs(p, "mano_full_axis_angle", mano.hands_components,
                                        mano.hands_mean)
            v, _ = mano(betas.expand(len(p), -1), dec["global_orient"],
                        dec["local_full_aa"], dec["transl"])
            return v

        out["max_mesh_m"] = float((verts(params51) - verts(back)).norm(dim=-1).max())
    return out
