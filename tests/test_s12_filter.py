#!/usr/bin/env python3
r"""S12 gates for the Lie-manifold filter.

A filter can be wrong in two ways that look nothing alike. It can track badly, which a tracking run
reveals; or it can track well while lying about its own uncertainty, which no accuracy metric
reveals and which makes every downstream use of the covariance -- the S9 router's prior, the S13
anchor's fusion weight -- silently meaningless. The gates here are mostly about the second failure.

What is checked:

* the update is a retraction, never an addition in coordinates
* the covariance stays symmetric positive definite over a long run, which is what Joseph form buys
* the process model reproduces exact constant-velocity motion on the manifold
* the linearised propagation matches the true error transport, to second order
* a stationary measurement makes the covariance converge, and a missing one makes it grow
* the measurement covariance built from Fisher information is ordered the right way: more events
  means a tighter measurement, and an uninformative direction stays at the ceiling
* on a synthetic trajectory with known noise, the filter is calibrated: its predicted standard
  deviation ranks with the realised error, and its normalised innovation squared sits at the
  innovation dimension rather than far below it
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from mano_layer import ManoLayer                                    # noqa: E402
from pose_repr import decode_to_mano_inputs                         # noqa: E402
from semkine import filter as FT                                    # noqa: E402
from semkine import lie                                             # noqa: E402
from semkine.dataset import _read_meta51                            # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOF, NSTATE = FT.DOF, FT.NSTATE


@pytest.fixture(scope="module")
def mano():
    return ManoLayer(REPO / "assets/mano_right.npz", add_mean=False).to(DEV).eval().double()


@pytest.fixture(scope="module")
def ctx(mano):
    """Real poses, plus the `hands_mean` and pivot the 51D adapter needs."""
    m = sorted(glob.glob(str(REPO / "data/hand_data51/*/*.meta")))[0]
    p = _read_meta51(m)
    aux = np.load(m[:-5] + "_aux.npz", allow_pickle=True)
    betas = torch.tensor(aux["betas"], dtype=torch.float64, device=DEV).view(1, -1)
    j0 = (mano.J_regressor @ (mano.v_template
                              + torch.einsum("bl,mkl->bmk", betas, mano.shapedirs)))[:, 0]
    return dict(pos=torch.from_numpy(p).to(DEV).double(), betas=betas,
                hands_mean=mano.hands_mean.double(), j0=j0)


def make_filter(ctx, x0=None, **kw):
    f = FT.LieFilter(**kw)
    f.start((ctx["pos"][0:1] if x0 is None else x0), ctx["hands_mean"], ctx["j0"])
    return f


# ------------------------------------------------------------------- process model
def test_constant_velocity_is_exact_on_the_manifold(ctx):
    """Undamped, unperturbed propagation must equal the closed-form manifold trajectory.

    A single retraction by `T*v` and `n` retractions by `(T/n)*v` agree exactly because each block
    is a one-parameter subgroup, so any discrepancy is an implementation error rather than
    discretisation.
    """
    f = make_filter(ctx, gamma=0.0, sigma_a=0.0)
    v = torch.zeros(DOF, device=DEV, dtype=torch.float64)
    v[3:6] = torch.tensor([0.4, -0.2, 0.1], device=DEV, dtype=torch.float64)
    v[6:9] = torch.tensor([0.3, 0.1, -0.2], device=DEV, dtype=torch.float64)
    v[0:3] = torch.tensor([0.05, 0.0, -0.02], device=DEV, dtype=torch.float64)
    f.v = v.clone()
    x0 = f.x.clone()
    for _ in range(20):
        f.predict(0.05)
    direct = lie.retract_51d(x0, (1.0 * v).view(1, DOF), ctx["hands_mean"], ctx["j0"])
    err = float((f.x - direct).abs().max())
    assert err < 1e-9, f"20 x 50 ms differs from one 1 s step by {err:.3e}"


def test_damping_brings_the_velocity_to_rest(ctx):
    f = make_filter(ctx, gamma=8.0, sigma_a=0.0)
    f.v = torch.ones(DOF, device=DEV, dtype=torch.float64)
    for _ in range(40):
        f.predict(0.05)
    assert float(f.v.abs().max()) < np.exp(-8.0 * 2.0) * 1.01 + 1e-9


@pytest.mark.parametrize("eps", [1e-3, 1e-2, 1e-1, 0.5])
def test_propagation_transports_the_error_exactly(ctx, eps):
    r"""Pose error transport is exact, not linearised, and the test says so at large errors too.

    Perturbing the state by `e`, propagating both copies and reading off the resulting tangent
    difference must reproduce `Phi e` -- and because `e' = Ad(Exp(-u)) e` is an identity rather
    than a first-order expansion, it must do so to machine precision at *any* `e`, including half a
    radian. That is a much sharper gate than second-order agreement as `e` shrinks, and it is the
    reason to prefer the adjoint here: the more familiar `J_r^-1` would satisfy a vanishing-`e`
    test while being wrong by a first-order term in the increment `u`, exactly where a 50 ms step
    lives.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    v = torch.randn(DOF, generator=g, dtype=torch.float64).to(DEV) * 0.8
    e = torch.randn(NSTATE, generator=g, dtype=torch.float64).to(DEV)
    e[DOF:] = 0                                  # a pure pose error, so `Phi`'s top-left block acts
    dt = 0.05
    base = make_filter(ctx, gamma=0.0, sigma_a=0.0)
    base.v = v.clone()
    pert = make_filter(ctx, gamma=0.0, sigma_a=0.0)
    pert.v = v.clone()
    pert.x = lie.retract_51d(pert.x, (eps * e[:DOF]).view(1, DOF), ctx["hands_mean"], ctx["j0"])
    Phi, _ = FT.process_matrices(dt, 0.0, 0.0, DOF, DEV, torch.float64, dt * v)
    base.predict(dt)
    pert.predict(dt)
    true = lie.local_coordinates_51d(base.x, pert.x, ctx["hands_mean"], ctx["j0"]).reshape(DOF)
    pred = (Phi @ (eps * e))[:DOF]
    rel = float((true - pred).norm() / true.norm())
    assert rel < 1e-12, f"eps={eps}: transport is not exact, relative error {rel:.3e}"


def test_small_angle_shortcut_is_measurably_worse(ctx):
    r"""The manifold corrections must matter, or they should not be in the code.

    `Phi = [[I, dt I], [0, ...]]` is the standard shortcut. This checks it is a worse predictor of
    the true error transport than the exact form at a 50 ms step, so the adjoint and right-Jacobian
    blocks are justified by measurement rather than by preference.
    """
    g = torch.Generator(device="cpu").manual_seed(1)
    v = torch.randn(DOF, generator=g, dtype=torch.float64).to(DEV) * 4.0
    e = torch.randn(NSTATE, generator=g, dtype=torch.float64).to(DEV) * 1e-3
    e[DOF:] = 0
    dt, eps = 0.05, 1.0
    base = make_filter(ctx, gamma=0.0, sigma_a=0.0)
    base.v = v.clone()
    pert = make_filter(ctx, gamma=0.0, sigma_a=0.0)
    pert.v = v.clone()
    pert.x = lie.retract_51d(pert.x, (eps * e[:DOF]).view(1, DOF), ctx["hands_mean"], ctx["j0"])
    base.predict(dt)
    pert.predict(dt)
    true = lie.local_coordinates_51d(base.x, pert.x, ctx["hands_mean"], ctx["j0"]).reshape(DOF)
    exact = (FT.process_matrices(dt, 0.0, 0.0, DOF, DEV, torch.float64, dt * v)[0] @ e)[:DOF]
    naive = (FT.process_matrices(dt, 0.0, 0.0, DOF, DEV, torch.float64, None)[0] @ e)[:DOF]
    e_ex = float((true - exact).norm() / true.norm())
    e_na = float((true - naive).norm() / true.norm())
    assert e_ex < 0.05 * e_na, f"exact {e_ex:.3e} is not much better than naive {e_na:.3e}"


def test_se3_right_jacobian_matches_finite_differences():
    r"""`J_r` of `SE(3)` exp, including Barfoot's coupling block, against a central difference.

    `Exp(xi + eps d) = Exp(xi) Exp(J_r(xi) eps d) + O(eps^2)`, so `J_r` column `k` is the tangent
    displacement produced by nudging `xi_k`. The coupling block is the part worth testing: the
    diagonal `SO(3)` blocks would pass with `Q = 0`.
    """
    g = torch.Generator().manual_seed(5)
    for trial in range(4):
        xi = (torch.randn(6, generator=g, dtype=torch.float64) * 0.5).view(1, 6)
        J = lie.se3_Jr(xi)[0]
        R0, p0 = lie.se3_exp(xi)
        num = torch.zeros(6, 6, dtype=torch.float64)
        h = 1e-6
        for k in range(6):
            d = torch.zeros(1, 6, dtype=torch.float64)
            d[0, k] = h
            Rp, pp = lie.se3_exp(xi + d)
            Rm, pm = lie.se3_exp(xi - d)
            lp = lie.se3_log(R0.transpose(-1, -2) @ Rp,
                             (R0.transpose(-1, -2) @ (pp - p0)[..., None])[..., 0])
            lm = lie.se3_log(R0.transpose(-1, -2) @ Rm,
                             (R0.transpose(-1, -2) @ (pm - p0)[..., None])[..., 0])
            num[:, k] = (lp - lm)[0] / (2 * h)
        err = float((J - num).abs().max())
        assert err < 1e-7, f"trial {trial}: J_r off by {err:.3e}\n{J}\n{num}"
        assert float((J[0:3, 3:6]).abs().max()) > 1e-3, "the coupling block is trivially zero"


def test_process_noise_has_the_right_cross_term(ctx):
    _, Q = FT.process_matrices(0.05, 8.0, 60.0, DOF, DEV, torch.float64)
    dt, s2 = 0.05, 60.0 ** 2
    assert abs(float(Q[0, 0]) - s2 * dt ** 3 / 3) < 1e-12
    assert abs(float(Q[0, DOF]) - s2 * dt ** 2 / 2) < 1e-12
    assert abs(float(Q[DOF, DOF]) - s2 * dt) < 1e-12
    w = torch.linalg.eigvalsh(Q)
    assert float(w.min()) > -1e-12, f"Q is not PSD, min eigenvalue {float(w.min()):.3e}"


# ------------------------------------------------------------------- covariance health
def test_covariance_stays_symmetric_positive_definite(ctx):
    """500 steps of predict/update, checking the property Joseph form exists to preserve."""
    rng = np.random.default_rng(0)
    f = make_filter(ctx)
    pos = ctx["pos"]
    worst_asym, worst_eig = 0.0, float("inf")
    for k in range(1, 501):
        f.predict(0.05)
        z = pos[k % len(pos)].view(1, DOF)
        Lam = _random_information(rng, 1.0e4)
        f.update(z, Lam)
        worst_asym = max(worst_asym, float((f.P - f.P.T).abs().max()))
        worst_eig = min(worst_eig, float(torch.linalg.eigvalsh(f.P).min()))
    assert worst_asym < 1e-9, f"covariance lost symmetry: {worst_asym:.3e}"
    assert worst_eig > 0, f"covariance lost positive definiteness: {worst_eig:.3e}"


def _random_information(rng, scale: float) -> torch.Tensor:
    A = torch.tensor(rng.normal(size=(DOF, DOF)), dtype=torch.float64, device=DEV)
    return (A @ A.T) * (scale / DOF)


def test_missing_measurements_grow_the_uncertainty(ctx):
    """The principled form of the zero-event gate: no events means a wider posterior, not the same.

    The existing `ZERO_EVENT_GATE` returns the previous pose bitwise and leaves the tracker's
    confidence untouched. That is the behaviour this replaces, so the growth is checked explicitly.
    """
    f = make_filter(ctx)
    t0 = float(torch.diagonal(f.P[:DOF, :DOF]).sum())
    traces = []
    for _ in range(10):
        f.predict(0.05)
        traces.append(float(torch.diagonal(f.P[:DOF, :DOF]).sum()))
    assert traces[0] > t0
    assert all(traces[i + 1] > traces[i] for i in range(len(traces) - 1)), traces
    assert f.n_measured == 0 and f.n_propagated == 10


def test_measurements_bound_the_uncertainty_at_a_steady_state(ctx):
    r"""Measurements must reach a steady state, and it must be below the unmeasured one.

    Not "the trace shrinks": whether it shrinks depends on where the prior started, and starting a
    filter with an over-tight prior and watching it correctly loosen would fail such a test. The
    real property is that the measured filter converges -- a Riccati fixed point -- and that its
    fixed point is tighter than propagating alone over the same interval, which is the whole reason
    to fuse anything.
    """
    z = ctx["pos"][0].view(1, DOF)
    Lam = torch.eye(DOF, device=DEV, dtype=torch.float64) * 1.0e5
    f = make_filter(ctx, info_divisor=1.0e4)
    tr = []
    for _ in range(200):
        f.predict(0.005)
        f.update(z, Lam)
        tr.append(float(torch.diagonal(f.P[:DOF, :DOF]).sum()))
    tail = np.array(tr[-50:])
    rel = float(tail.std() / max(tail.mean(), 1e-12))
    assert rel < 1e-3, f"no steady state reached: last 50 traces vary by {rel:.3e} relative"

    g = make_filter(ctx, info_divisor=1.0e4)
    for _ in range(200):
        g.predict(0.005)
    free = float(torch.diagonal(g.P[:DOF, :DOF]).sum())
    assert tail.mean() < 0.1 * free, (
        f"fusing barely helped: measured {tail.mean():.4g} vs propagated {free:.4g}")


def test_zero_gain_limit_leaves_the_state_alone(ctx):
    """An infinitely uncertain measurement must move nothing. The sanity end of the gain range."""
    f = make_filter(ctx, sigma_ceil=1e8, sigma_floor=1e-8)
    f.predict(0.05)
    x_before, P_before = f.x.clone(), f.P.clone()
    f.update(ctx["pos"][5].view(1, DOF), None)     # `Lam=None` -> Sigma at the ceiling
    assert float((f.x - x_before).abs().max()) < 1e-6
    assert float((f.P - P_before).abs().max()) < 1e-4


# ------------------------------------------------------------------- measurement covariance
def test_more_information_means_a_tighter_measurement(ctx):
    """`Sigma` must be monotone in the information, in the Loewner order that matters."""
    rng = np.random.default_rng(3)
    A = torch.tensor(rng.normal(size=(400, DOF)), dtype=torch.float64, device=DEV)
    prev = None
    for n in (50, 100, 200, 400):
        Lam = A[:n].T @ A[:n]
        S = FT.sigma_from_information(Lam, 1.0, 1e-8, 1e8, DEV)
        if prev is not None:
            # `prev - S` PSD means every direction got no worse, which is the real statement;
            # comparing traces would let one direction degrade unnoticed.
            w = torch.linalg.eigvalsh(prev - S)
            assert float(w.min()) > -1e-6, f"n={n} loosened some direction by {float(w.min()):.3e}"
        prev = S


def test_unobserved_directions_stay_at_the_ceiling(ctx):
    """A direction the events say nothing about must get the ceiling variance, not a small one."""
    rng = np.random.default_rng(4)
    J = torch.tensor(rng.normal(size=(300, DOF)), dtype=torch.float64, device=DEV)
    J[:, 9:12] = 0.0                              # joint 1 is unobservable in this packet
    S = FT.sigma_from_information(J.T @ J, 1.0, 1e-6, 0.5, DEV)
    assert float(torch.diagonal(S)[9:12].min()) > 0.4, torch.diagonal(S)[9:12]
    assert float(torch.diagonal(S)[0:6].max()) < 0.4


def test_sigma_is_symmetric_positive_definite_within_bounds(ctx):
    rng = np.random.default_rng(5)
    for _ in range(5):
        Lam = _random_information(rng, 1e3)
        S = FT.sigma_from_information(Lam, 2.0, 1e-4, 1.0, DEV)
        assert float((S - S.T).abs().max()) < 1e-12
        w = torch.linalg.eigvalsh(S)
        assert float(w.min()) >= 1e-4 - 1e-12 and float(w.max()) <= 1.0 + 1e-9, (
            float(w.min()), float(w.max()))


# ------------------------------------------------------------------- calibration
def test_filter_is_calibrated_on_a_known_trajectory(ctx):
    r"""The gate that matters: does the covariance mean what it says?

    A trajectory is generated on the manifold with a known constant velocity, and measurements are
    drawn with a known tangent-space noise whose magnitude changes every few steps -- the synthetic
    stand-in for packets of varying event count. Two things are then required of the filter:

    * its normalised innovation squared averages near the innovation dimension. Below it means the
      filter is overstating its uncertainty, far above means it is understating it, and the plan
      registers 90-98% coverage of the chi-square band.
    * its predicted standard deviation ranks with the realised error, Spearman >= 0.30. This is the
      weaker but more useful statement: whatever the absolute scale, the filter must know which
      steps it got wrong, since that ordering is what S13's trigger and S9's prior consume.
    """
    rng = np.random.default_rng(7)
    hm, j0 = ctx["hands_mean"], ctx["j0"]
    x = ctx["pos"][0:1].clone()
    v = torch.tensor(rng.normal(scale=0.3, size=DOF), dtype=torch.float64, device=DEV)
    dt = 0.02
    # Measurement noise per coordinate, and the information matrix a consistent filter should be
    # handed for it: `Lam = Sigma^-1` scaled by `info_divisor`, so the filter's own inversion
    # recovers the truth. This tests the filter, not the choice of `info_divisor`.
    f = make_filter(ctx, x0=x.clone(), gamma=0.0, sigma_a=0.5, p0_pose=1e-3, p0_vel=0.4,
                    info_divisor=1.0, sigma_floor=1e-8, sigma_ceil=1e4)
    f.v = v.clone() + torch.tensor(rng.normal(scale=0.05, size=DOF),
                                   dtype=torch.float64, device=DEV)
    sig_pred, err_real = [], []
    for k in range(400):
        x = lie.retract_51d(x, (dt * v).view(1, DOF), hm, j0)
        s = 0.01 if (k // 20) % 2 == 0 else 0.06        # alternating measurement quality
        noise = torch.tensor(rng.normal(scale=s, size=DOF), dtype=torch.float64, device=DEV)
        z = lie.retract_51d(x, noise.view(1, DOF), hm, j0)
        Lam = torch.eye(DOF, device=DEV, dtype=torch.float64) / s ** 2

        f.predict(dt)
        pre = f.pose_sigma().norm()
        f.update(z, Lam)
        if k > 50:                                       # let the transient settle
            e = lie.local_coordinates_51d(f.x, x, hm, j0).reshape(DOF).norm()
            sig_pred.append(float(pre))
            err_real.append(float(e))

    st = f.stats()
    rho = FT.spearman(np.array(sig_pred), np.array(err_real))
    assert st["nis_mean"] < 3.0 * DOF, f"filter is overconfident: NIS mean {st['nis_mean']:.1f}"
    assert st["nis_mean"] > 0.3 * DOF, f"filter is underconfident: NIS mean {st['nis_mean']:.1f}"
    assert rho >= 0.30, f"predicted sigma does not rank with the error: Spearman {rho:.3f}"


@pytest.mark.parametrize("innov", [0.02, 0.05, 0.2])
def test_root_metric_bias_is_bounded_at_realistic_innovations(ctx, mano, innov):
    r"""How much does the `SE(3)` correction bias the wrist, at the innovation sizes it sees?

    S13 found that averaging two poses under the full `SE(3)` metric displaces the wrist in
    translation, by 5.9 mm when the two disagree by 0.6 rad, and switched the anchor's fusion to a
    decoupled `R^3 x SO(3)` metric for that reason. The filter performs the same kind of averaging
    every step, so the same bias is present -- but at a much smaller scale, because a step's
    innovation is a fraction of a radian rather than most of one.

    This measures it instead of assuming it away. A static truth is measured repeatedly with tangent
    noise supplied in **antithetic pairs** `+n, -n`, so the sample mean of the noise is exactly zero
    rather than merely zero in expectation. That matters: with independent draws, the residual
    offset after a few hundred steps is dominated by the statistical error, `sigma/sqrt(N)`, which
    for centimetre-scale translation noise is millimetres and would be mistaken for a bias. With
    antithetic pairs the statistical term cancels and what remains is the estimator's own
    nonlinearity.

    Translation noise is scaled to rotation at 5 cm per radian, the same exchange rate the S7 oracle
    uses, so "0.05 rad of innovation" means a physically coherent perturbation rather than 5 cm of
    wrist displacement paired with 3 degrees of rotation.
    """
    rng = np.random.default_rng(17)
    hm, j0 = ctx["hands_mean"], ctx["j0"]
    truth = ctx["pos"][20].view(1, DOF)
    f = make_filter(ctx, x0=truth.clone(), gamma=50.0, sigma_a=1e-3,
                    p0_pose=innov, p0_vel=1e-3, info_divisor=1.0,
                    sigma_floor=1e-10, sigma_ceil=1e6)
    Lam = torch.eye(DOF, device=DEV, dtype=torch.float64) / innov ** 2
    for _ in range(200):
        n = torch.tensor(rng.normal(scale=innov, size=DOF), dtype=torch.float64, device=DEV)
        n[0:3] *= 0.05
        for sign in (1.0, -1.0):
            z = lie.retract_51d(truth, (sign * n).view(1, DOF), hm, j0)
            f.predict(0.05)
            f.update(z, Lam)
    d = decode_to_mano_inputs(torch.cat([truth, f.x]), "mano_full_axis_angle",
                              mano.hands_components, mano.hands_mean)
    v, _ = mano(ctx["betas"].expand(2, -1), d["global_orient"], d["local_full_aa"], d["transl"])
    mm = float((v[1] - v[0]).norm(dim=-1).mean()) * 1000
    limit = 0.1 if innov <= 0.05 else 2.0
    assert mm < limit, f"innovation {innov} rad leaves a {mm:.4f} mm bias (limit {limit})"


def test_spearman_matches_scipy():
    rng = np.random.default_rng(11)
    a, b = rng.normal(size=200), rng.normal(size=200)
    b = 0.6 * a + 0.8 * b
    from scipy.stats import spearmanr
    assert abs(FT.spearman(a, b) - float(spearmanr(a, b).statistic)) < 1e-9


def test_filtered_state_never_leaves_the_manifold(ctx):
    """Every emitted state must round-trip through the adapter: the update is a retraction."""
    rng = np.random.default_rng(13)
    f = make_filter(ctx)
    for k in range(100):
        f.predict(0.05)
        f.update(ctx["pos"][k % len(ctx["pos"])].view(1, DOF), _random_information(rng, 1e4))
    rep = lie.round_trip_report(f.x, ctx["hands_mean"], ctx["j0"])
    assert rep["max_geodesic_root_rad"] < 1e-9, rep
    assert rep["max_geodesic_joint_rad"] < 1e-9, rep
    assert torch.isfinite(f.x).all()
