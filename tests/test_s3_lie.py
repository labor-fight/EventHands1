#!/usr/bin/env python3
"""S3 gates: exp/log correctness including the hard branches, and the lossless 51D round trip.

The registered gates are max geodesic <= 1e-6 rad and max mesh displacement <= 1e-5 m on real
ground-truth poses. Everything else here exists to catch the specific ways a Lie implementation
is usually wrong: the `theta -> 0` division, the `theta -> pi` loss of precision in the direct
log formula, and a left/right multiplication mix-up.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from mano_layer import ManoLayer, batch_rodrigues     # noqa: E402
from semkine import lie as L                          # noqa: E402
from semkine.dataset import _read_meta51, mano_root_joint   # noqa: E402

torch.manual_seed(0)
DT = torch.float64


@pytest.fixture(scope="module")
def mano():
    return ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False)


@pytest.fixture(scope="module")
def gt():
    """Real ground-truth poses; synthetic angles would miss the actual operating range."""
    root = Path("data/hand_data51")
    base = str(ROOT / root / "val" / "zgz_local")
    p = _read_meta51(base + ".meta")
    aux = np.load(base + "_aux.npz", allow_pickle=True)
    idx = np.linspace(0, len(p) - 1, 512).astype(int)
    betas = np.asarray(aux["betas"], np.float32)
    return (torch.from_numpy(p[idx]).to(DT),
            torch.from_numpy(betas),
            torch.from_numpy(mano_root_joint(betas, str(ROOT / "assets/mano_right.npz"))).to(DT))


# --------------------------------------------------------------- SO(3) basics
def test_skew_is_cross_product():
    v, w = torch.randn(64, 3, dtype=DT), torch.randn(64, 3, dtype=DT)
    assert torch.allclose(L.skew(v) @ w[..., None], torch.cross(v, w, dim=-1)[..., None])
    assert torch.allclose(L.unskew(L.skew(v)), v)


def test_so3_exp_is_a_rotation():
    phi = torch.randn(256, 3, dtype=DT) * 2.0
    R = L.so3_exp(phi)
    I = torch.eye(3, dtype=DT).expand_as(R)
    assert (R @ R.transpose(-1, -2) - I).abs().max() < 1e-12
    assert (torch.linalg.det(R) - 1.0).abs().max() < 1e-12


def test_so3_exp_matches_mano_rodrigues(mano):
    """The state must agree with the MANO implementation it feeds, not just with theory.

    Tolerance is 1e-7 rather than machine precision because `batch_rodrigues` computes its angle
    as `norm(rot_vecs + 1e-8)`, adding the epsilon to the *vector* instead of to the norm. That
    perturbs both axis and angle by about 1e-8, which is invisible in the float32 the model runs
    in but is a real floor for a float64 comparison. `so3_exp` is the more accurate of the two;
    the point of this test is agreement, and 1e-7 rad is far below the 1e-6 S3 gate.
    """
    phi = torch.randn(512, 3, dtype=DT) * 1.5
    assert (L.so3_exp(phi) - batch_rodrigues(phi)).abs().max() < 1e-7
    assert (L.so3_exp(phi.float()) - batch_rodrigues(phi.float())).abs().max() < 1e-5


@pytest.mark.parametrize("scale", [0.0, 1e-12, 1e-8, 1e-5, 1e-3, 1e-1, 1.0, 3.0])
def test_so3_log_exp_round_trip(scale):
    phi = torch.randn(512, 3, dtype=DT)
    phi = phi / phi.norm(dim=-1, keepdim=True).clamp_min(1e-30) * scale
    if scale == 0.0:
        phi = torch.zeros(512, 3, dtype=DT)
    back = L.so3_log(L.so3_exp(phi))
    assert (back - phi).abs().max() < 1e-11, f"scale={scale}"


@pytest.mark.parametrize("angle", [math.pi - 1e-3, math.pi - 1e-6, math.pi - 1e-9])
def test_so3_log_near_pi(angle):
    """The branch a naive `theta/(2 sin theta)` implementation gets wrong."""
    axis = torch.randn(256, 3, dtype=DT)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    phi = axis * angle
    R = L.so3_exp(phi)
    back = L.so3_log(R)
    assert L.geodesic_error(R, L.so3_exp(back)).max() < 1e-9, f"angle={angle}"


def test_so3_log_at_exactly_pi_is_a_valid_rotation():
    """At pi the axis sign is genuinely ambiguous, so only the rotation must round-trip."""
    axis = torch.tensor([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0],
                         [0.577350269, 0.577350269, 0.577350269]], dtype=DT)
    R = L.so3_exp(axis * math.pi)
    assert L.geodesic_error(R, L.so3_exp(L.so3_log(R))).max() < 1e-7


def test_so3_log_is_in_canonical_hemisphere():
    phi = torch.randn(512, 3, dtype=DT) * 4.0
    assert L.so3_log(L.so3_exp(phi)).norm(dim=-1).max() <= math.pi + 1e-9


def test_so3_exp_gradient_is_finite_at_zero():
    phi = torch.zeros(8, 3, dtype=DT, requires_grad=True)
    L.so3_exp(phi).sum().backward()
    assert torch.isfinite(phi.grad).all()


# --------------------------------------------------------------- Jacobians
def test_right_jacobian_matches_finite_difference():
    """`Exp(phi + d) ~ Exp(phi) Exp(Jr(phi) d)` -- the defining property."""
    phi = torch.randn(64, 3, dtype=DT) * 1.2
    Jr = L.so3_Jr(phi)
    h = 1e-7
    for k in range(3):
        d = torch.zeros(64, 3, dtype=DT)
        d[:, k] = h
        lhs = L.so3_exp(phi + d)
        rhs = L.so3_exp(phi) @ L.so3_exp((Jr @ d[..., None])[..., 0])
        assert (lhs - rhs).abs().max() < 1e-12, k


def test_jacobian_inverse_and_left_right_relation():
    # `Jr_inv` is defined on |phi| < 2 pi; `so3_log` never returns more than pi, so the state
    # only ever needs |phi| <= pi and that is the range tested.
    phi = torch.randn(128, 3, dtype=DT)
    phi = phi / phi.norm(dim=-1, keepdim=True) * torch.rand(128, 1, dtype=DT) * math.pi
    I = torch.eye(3, dtype=DT).expand(128, 3, 3)
    assert (L.so3_Jr(phi) @ L.so3_Jr_inv(phi) - I).abs().max() < 1e-10
    assert (L.so3_Jl(phi) - L.so3_Jr(-phi)).abs().max() < 1e-12
    # Jl(phi) == Exp(phi) Jr(phi)
    assert (L.so3_Jl(phi) - L.so3_exp(phi) @ L.so3_Jr(phi)).abs().max() < 1e-10


@pytest.mark.parametrize("scale", [1e-8, 1e-3, 1.0, 3.0])
def test_jacobians_are_finite_at_all_scales(scale):
    phi = torch.randn(64, 3, dtype=DT)
    phi = phi / phi.norm(dim=-1, keepdim=True) * scale
    for f in (L.so3_Jr, L.so3_Jl, L.so3_Jr_inv, L.se3_V, L.se3_V_inv):
        assert torch.isfinite(f(phi)).all(), f.__name__


# --------------------------------------------------------------- SE(3)
@pytest.mark.parametrize("scale", [0.0, 1e-8, 1e-3, 1.0, 2.5])
def test_se3_exp_log_round_trip(scale):
    xi = torch.randn(256, 6, dtype=DT)
    xi[:, 3:6] = xi[:, 3:6] / xi[:, 3:6].norm(dim=-1, keepdim=True).clamp_min(1e-30) * scale
    R, p = L.se3_exp(xi)
    assert (L.se3_log(R, p) - xi).abs().max() < 1e-10, f"scale={scale}"


def test_se3_exp_matches_matrix_exponential():
    xi = torch.randn(16, 6, dtype=DT) * 0.8
    R, p = L.se3_exp(xi)
    A = torch.zeros(16, 4, 4, dtype=DT)
    A[:, :3, :3] = L.skew(xi[:, 3:6])
    A[:, :3, 3] = xi[:, 0:3]
    E = torch.matrix_exp(A)
    assert (E[:, :3, :3] - R).abs().max() < 1e-10
    assert (E[:, :3, 3] - p).abs().max() < 1e-10


def test_se3_adjoint_is_the_conjugation_map():
    xi = torch.randn(32, 6, dtype=DT) * 0.5
    R, p = L.se3_exp(torch.randn(32, 6, dtype=DT) * 0.7)
    # Adj(T) xi is the tangent of T Exp(xi) T^-1
    Rx, px = L.se3_exp(xi)
    Rc = R @ Rx @ R.transpose(-1, -2)
    pc = (R @ px[..., None])[..., 0] + p - (Rc @ p[..., None])[..., 0]
    got = (L.se3_adjoint(R, p) @ xi[..., None])[..., 0]
    assert (L.se3_log(Rc, pc) - got).abs().max() < 1e-9


# --------------------------------------------------------------- 51D adapter
def test_round_trip_on_real_poses_meets_the_gate(gt, mano):
    """Registered S3 gate: geodesic <= 1e-6 rad and mesh <= 1e-5 m."""
    p, betas, j0 = gt
    m = ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False).to(DT)
    rep = L.round_trip_report(p, m.hands_mean, j0, mano=m, betas=betas[None].to(DT))
    assert rep["max_geodesic_root_rad"] <= 1e-6, rep
    assert rep["max_geodesic_joint_rad"] <= 1e-6, rep
    assert rep["max_mesh_m"] <= 1e-5, rep
    assert rep["max_transl_m"] <= 1e-9, rep


def test_root_pivot_term_is_required(gt, mano):
    """Dropping `(I - R) j0` breaks the round trip, so the pivot is not decoration."""
    p, betas, j0 = gt
    m = ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False).to(DT)
    good = L.state_from_51d(p, m.hands_mean, j0)
    assert (L.state_to_51d(good, m.hands_mean)[:, 0:3] - p[:, 0:3]).abs().max() < 1e-12
    # A state built with j0 = 0 but read back with the real j0 must disagree.
    wrong = L.state_from_51d(p, m.hands_mean, torch.zeros(3, dtype=DT))
    wrong.j0 = j0.view(1, 3).expand(p.shape[0], 3)
    err = (L.state_to_51d(wrong, m.hands_mean)[:, 0:3] - p[:, 0:3]).abs().max()
    assert err > 1e-3, f"pivot term had no effect ({err})"


def test_axis_angle_range_has_margin_to_pi(gt):
    p, _, _ = gt
    m = ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False).to(DT)
    r = L.check_axis_angle_range(p, m.hands_mean)
    assert r["n_joints_over_pi"] == 0 and r["n_root_over_pi"] == 0, r
    assert r["joint_margin_to_pi_rad"] > 0.05, r


def test_state_from_51d_reproduces_mano_vertices(gt):
    """The state's rotations must be the ones MANO would build from the same 51D row."""
    p, betas, j0 = gt
    m = ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False).to(DT)
    st = L.state_from_51d(p, m.hands_mean, j0)
    full = p[:, 6:51] + m.hands_mean.reshape(1, 45)
    # 1e-7, not machine precision: see test_so3_exp_matches_mano_rodrigues for the 1e-8 bias
    # inside MANO's own `batch_rodrigues`.
    assert (st.R_joints - batch_rodrigues(full.reshape(-1, 3)).reshape(-1, 15, 3, 3)
            ).abs().max() < 1e-7
    assert (st.R_root - batch_rodrigues(p[:, 3:6])).abs().max() < 1e-7


def test_retract_is_right_multiplicative(gt):
    p, _, j0 = gt
    m = ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False).to(DT)
    st = L.state_from_51d(p[:32], m.hands_mean, j0)
    d = torch.randn(32, 51, dtype=DT) * 0.05
    new = st.retract(d)
    expect = st.R_joints @ L.so3_exp(d[:, 6:].reshape(-1, 15, 3))
    assert (new.R_joints - expect).abs().max() < 1e-12
    # and NOT left multiplication
    left = L.so3_exp(d[:, 6:].reshape(-1, 15, 3)) @ st.R_joints
    assert (new.R_joints - left).abs().max() > 1e-4


def test_retract_zero_is_the_identity(gt):
    p, _, j0 = gt
    m = ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False).to(DT)
    out = L.retract_51d(p, torch.zeros(len(p), 51, dtype=DT), m.hands_mean, j0)
    assert (out - p).abs().max() < 1e-9


def test_local_coordinates_inverts_retract(gt):
    p, _, j0 = gt
    m = ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False).to(DT)
    st = L.state_from_51d(p[:64], m.hands_mean, j0)
    d = torch.randn(64, 51, dtype=DT) * 0.1
    assert (st.local_coordinates(st.retract(d)) - d).abs().max() < 1e-10


def test_manifold_update_differs_measurably_from_axis_angle_addition(gt):
    """Quantifies what the convention change is worth, so a null result can be read against it."""
    p, _, j0 = gt
    m = ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False).to(DT)
    d = torch.randn(len(p), 51, dtype=DT)
    d[:, 0:3] *= 0.005
    d[:, 3:] *= 0.3          # the large-noise magnitude the tracker is trained against
    rep = L.assert_no_axis_angle_addition(p, d, m.hands_mean, j0)
    assert rep["joint_max_deg"] > 1.0, rep
    assert rep["joint_mean_deg"] > 0.05, rep


def test_batched_and_looped_agree(gt):
    p, _, j0 = gt
    m = ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False).to(DT)
    st = L.state_from_51d(p[:16], m.hands_mean, j0)
    for i in range(16):
        one = L.state_from_51d(p[i : i + 1], m.hands_mean, j0)
        assert (one.R_root - st.R_root[i : i + 1]).abs().max() < 1e-14
        assert (one.R_joints - st.R_joints[i : i + 1]).abs().max() < 1e-14


def test_float32_round_trip_is_still_usable(gt):
    """Training runs in float32/bf16, so the gate has to hold there too, if more loosely."""
    p, betas, j0 = gt
    m = ManoLayer(ROOT / "assets" / "mano_right.npz", add_mean=False)
    rep = L.round_trip_report(p.float(), m.hands_mean, j0.float(), mano=m,
                              betas=betas[None])
    assert rep["max_geodesic_joint_rad"] < 1e-4, rep
    assert rep["max_mesh_m"] < 1e-5, rep
