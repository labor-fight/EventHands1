"""DT round, task filter-eval: `semkine.anchored.FilteredTracker` against tiny stand-in trackers (CPU), the filter law of
a root perturbation on SO(3), and the arithmetic of `tools/dt/filter_eval.py` (gain parsing, report numbers, law check).

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_dt_filter.py -q
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "dt")]
from semkine.anchored import FilteredTracker, anchor_blend           # noqa: E402
import filter_eval as FE                                              # noqa: E402


class _Stand(nn.Module):
    """Stand-in for the render-and-compare tracker: `prev + delta(x) + bias` with every operation row-wise. The
    constant bias makes the tracker's own output differ from `prev` even on an empty packet, so a hold on an
    empty packet can only come from the wrapper (the real network's own gate would hide that)."""

    def __init__(self, seed=0, scale=1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("w", scale * 0.05 * torch.randn(51, generator=g))
        self.register_buffer("bias", scale * 0.01 * torch.randn(51, generator=g))
        self.ctx = None

    def set_hand_context(self, betas, camera_K):
        self.ctx = (betas, camera_K)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        s = x.reshape(x.shape[0], -1).mean(dim=1, keepdim=True)
        return prevpos + s * self.w + self.bias


class _NoCtx(nn.Module):
    """A tracker without `set_hand_context` (a state-free stand-in)."""

    def forward(self, x, prevpos, betas=None, camera_K=None):
        return prevpos + 0.01


class _Rec(nn.Module):
    """A tracker that records the `betas` / `camera_K` it is called with (the real network reads them, so the wrapper
    must hand them on, as the very same objects and in the right slots)."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, x, prevpos, betas=None, camera_K=None):
        self.calls.append((betas, camera_K))
        return prevpos + 0.01


def _prev(b, seed=1):
    """(b, 51) states: translation near (0, 0, 0.4), root |phi| in [0.3, 2.3] (< pi), small finger angles."""
    g = torch.Generator().manual_seed(seed)
    p = 0.3 * torch.randn(b, 51, generator=g)
    p[:, :3] = 0.05 * torch.randn(b, 3, generator=g)
    p[:, 2] += 0.4
    axis = torch.randn(b, 3, generator=g)
    p[:, 3:6] = axis / axis.norm(dim=1, keepdim=True) * (0.3 + 2.0 * torch.rand(b, 1, generator=g))
    return p


def _x(b, seed=2):
    g = torch.Generator().manual_seed(seed)
    return 0.1 + torch.rand(b, 6, 7, 2, generator=g)


def _slerp(rv0, rv1, a):
    return Slerp([0.0, 1.0], Rot.from_rotvec(np.stack([rv0, rv1])))([a]).as_rotvec()[0]


def _angle(rv_a, rv_b):
    return (Rot.from_rotvec(rv_a).inv() * Rot.from_rotvec(rv_b)).magnitude()


# ------------------------------------------------------------------------------------ FilteredTracker
@pytest.mark.parametrize("b", [1, 5])
def test_unit_gains_reproduce_the_tracker_bitwise(b):
    trk, x, prev = _Stand().eval(), _x(b), _prev(b)
    out = FilteredTracker(trk, 1.0, 1.0, 1.0).eval()(x, prev)
    assert out.dtype == prev.dtype and out.shape == (b, 51)
    assert torch.equal(out, trk(x, prev))


def test_unit_gains_closed_loop_stays_bitwise_equal_to_the_raw_loop():
    """The project's loops are chaotic (a 1e-6 change moves a result): chained over 40 steps the (1,1,1) filter
    must stay bit-identical to the unwrapped tracker."""
    trk = _Stand(scale=0.1).eval()                 # a slow drift: the root stays inside |phi| < pi for 40 steps
    ft = FilteredTracker(trk, 1.0, 1.0, 1.0).eval()
    raw, flt = _prev(5), _prev(5)
    for i in range(40):
        x = _x(5, seed=100 + i)
        raw, flt = trk(x, raw), ft(x, flt)
        assert torch.equal(raw, flt), i


def test_unit_gains_canonicalise_a_root_beyond_pi():
    """The one case in which (1,1,1) is not bitwise the tracker: `anchor_blend` returns the root as an axis-angle with
    |phi| <= pi, so a tracker output beyond pi (the same rotation in its other representation) comes back as the
    canonical vector. The recorded rt_cnntrack predictions stay below 2.7 rad, so their (1,1,1) evaluations reproduce bitwise."""
    trk, x, prev = _Stand().eval(), _x(1), _prev(1)
    prev[0, 3:6] = torch.tensor([0.0, 0.0, 3.2])                 # |phi| = 3.2 > pi, handed in by the caller
    out = trk(x, prev)
    flt = FilteredTracker(trk, 1.0, 1.0, 1.0).eval()(x, prev)
    assert float(out[0, 3:6].norm()) > math.pi and not torch.equal(flt[:, 3:6], out[:, 3:6])
    assert float(flt[0, 3:6].norm()) <= math.pi + 1e-6
    r_out, r_flt = Rot.from_rotvec(out[0, 3:6].double().numpy()), Rot.from_rotvec(flt[0, 3:6].double().numpy())
    assert (r_out.inv() * r_flt).magnitude() < 1e-5               # the same rotation


@pytest.mark.parametrize("b", [1, 5])
def test_zero_gains_return_prev(b):
    trk, x, prev = _Stand().eval(), _x(b), _prev(b)
    out = FilteredTracker(trk, 0.0, 0.0, 0.0).eval()(x, prev)
    assert torch.equal(out, prev)


@pytest.mark.parametrize("b", [1, 5])
@pytest.mark.parametrize("a", [0.1, 0.25, 0.5, 0.75, 0.9])
def test_root_step_is_the_geodesic_of_size_a(b, a):
    trk, x, prev = _Stand().eval(), _x(b), _prev(b)
    out = trk(x, prev)
    flt = FilteredTracker(trk, a, 1.0, 1.0).eval()(x, prev)
    for i in range(b):
        ref = _slerp(prev[i, 3:6].double().numpy(), out[i, 3:6].double().numpy(), a)
        assert np.allclose(flt[i, 3:6].double().numpy(), ref, atol=1e-6)
        # the angle travelled is a times the angle between the two rotations
        full = _angle(prev[i, 3:6].double().numpy(), out[i, 3:6].double().numpy())
        step = _angle(prev[i, 3:6].double().numpy(), flt[i, 3:6].double().numpy())
        assert abs(step - a * full) < 1e-5
    assert torch.equal(flt[:, :3], out[:, :3]) and torch.equal(flt[:, 6:], out[:, 6:])   # other blocks at gain 1


@pytest.mark.parametrize("b", [1, 5])
def test_each_block_has_its_own_gain(b):
    trk, x, prev = _Stand().eval(), _x(b), _prev(b)
    out = trk(x, prev)
    flt = FilteredTracker(trk, 0.25, 0.5, 0.75).eval()(x, prev)
    assert torch.allclose(flt[:, :3], prev[:, :3] + 0.75 * (out[:, :3] - prev[:, :3]), atol=1e-6)
    assert torch.allclose(flt[:, 6:], prev[:, 6:] + 0.5 * (out[:, 6:] - prev[:, 6:]), atol=1e-6)
    for i in range(b):
        ref = _slerp(prev[i, 3:6].double().numpy(), out[i, 3:6].double().numpy(), 0.25)
        assert np.allclose(flt[i, 3:6].double().numpy(), ref, atol=1e-6)
    # a_trans left out: the translation takes the finger gain
    tied = FilteredTracker(trk, 0.25, 0.5).eval()(x, prev)
    assert torch.allclose(tied[:, :3], prev[:, :3] + 0.5 * (out[:, :3] - prev[:, :3]), atol=1e-6)
    # and the module is `anchor_blend(prev, tracker, ...)` with nothing else
    assert torch.equal(flt, anchor_blend(prev, out, 0.25, 0.5, 0.75))


@pytest.mark.parametrize("b", [1, 5])
def test_empty_packet_returns_prev_bitwise(b):
    trk, prev = _Stand().eval(), _prev(b)
    x = torch.zeros(b, 6, 7, 2)
    assert not torch.equal(trk(x, prev), prev)                  # the stand-in itself does not hold
    for g in [FE.BASE, FE.PRIMARY, *FE.SENSITIVITY, (0.0, 0.0, 0.0), (0.25, 0.75, 0.5)]:   # every triple of the report
        assert torch.equal(FilteredTracker(trk, *g).eval()(x, prev), prev), g


def test_empty_rows_hold_while_other_rows_blend():
    trk, prev, x = _Stand().eval(), _prev(5), _x(5)
    x[0], x[3] = 0.0, 0.0
    ft = FilteredTracker(trk, 0.5, 1.0, 0.5).eval()
    out = ft(x, prev)
    assert torch.equal(out[0], prev[0]) and torch.equal(out[3], prev[3])
    ref = anchor_blend(prev, trk(x, prev), 0.5, 1.0, 0.5)
    for i in (1, 2, 4):
        assert torch.equal(out[i], ref[i])


def test_set_hand_context_is_forwarded():
    trk = _Stand()
    ft = FilteredTracker(trk, 0.5, 1.0, 0.5)
    betas, K = torch.zeros(1, 10), torch.eye(3).view(1, 3, 3)
    ft.set_hand_context(betas, K)
    assert trk.ctx is not None and trk.ctx[0] is betas and trk.ctx[1] is K
    FilteredTracker(_NoCtx(), 0.5, 1.0, 0.5).set_hand_context(betas, K)   # a tracker without the hook: no error


def test_betas_and_camera_K_reach_the_tracker():
    """`forward(x, prev, betas, camera_K)` hands both on to the wrapped tracker, the very same objects, each in its own
    slot; called without them the tracker sees None (the evaluators call `model(x, prev)`)."""
    rec = _Rec()
    ft = FilteredTracker(rec, 0.5, 1.0, 0.5).eval()
    betas, K = torch.zeros(1, 10), torch.eye(3).view(1, 3, 3)
    ft(_x(1), _prev(1), betas=betas, camera_K=K)
    ft(_x(1), _prev(1))
    assert rec.calls[0][0] is betas and rec.calls[0][1] is K
    assert rec.calls[1] == (None, None)


def test_evaluator_interface():
    """What `evalx.run_sequence` reads: a dense-LNES model (empty encoder_name) with `set_hand_context`."""
    ft = FilteredTracker(_Stand(), 0.5, 1.0, 0.5)
    assert ft.encoder_name == "" and hasattr(ft, "set_hand_context")
    assert (ft.a_root, ft.a_rest, ft.a_trans) == (0.5, 1.0, 0.5)
    assert not any(p.requires_grad for p in ft.parameters())      # the filter adds no parameters


def test_batch_one_equals_the_rows_of_a_batch():
    trk, prev, x = _Stand().eval(), _prev(5), _x(5)
    ft = FilteredTracker(trk, 0.5, 1.0, 0.5).eval()
    full = ft(x, prev)
    for i in range(5):
        assert torch.allclose(ft(x[i:i + 1], prev[i:i + 1])[0], full[i], atol=1e-7)


# ------------------------------------------------------------------- the filter law of a root perturbation
class _Contract(nn.Module):
    """Root stand-in with a known response rho that is parallel to every perturbation: the output is the target rotation
    moved the fraction `rho` of the way (geodesic) toward the state it is handed. rho = 0: the output ignores the state
    (a tracker without response); rho = 1: it hands the state back. Fingers and translation pass through."""

    encoder_name = ""

    def __init__(self, target_rv, rho):
        super().__init__()
        self.target, self.rho = np.asarray(target_rv, np.float64), float(rho)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        out = prevpos.clone()
        rt = Rot.from_rotvec(self.target)
        for i in range(prevpos.shape[0]):
            rp = Rot.from_rotvec(prevpos[i, 3:6].double().numpy())
            out[i, 3:6] = torch.from_numpy(Slerp([0.0, 1.0], Rot.concatenate([rt, rp]))([self.rho])[0].as_rotvec()).float()
        return out


def _retention_curve(a, rho, theta_deg, steps=6):
    """Closed loop of `FilteredTracker(_Contract)`: the unperturbed state sits at the target, the perturbed one starts
    `theta_deg` away; retention(k) = angle between the two fed-back states after k steps / theta."""
    target = np.array([0.3, -0.5, 0.2])
    ft = FilteredTracker(_Contract(target, rho), a, 1.0, 1.0).eval()
    x = _x(1)
    s_u = _prev(1)
    s_u[0, 3:6] = torch.from_numpy(target).float()
    axis = np.array([1.0, 2.0, -0.5])
    s_p = s_u.clone()
    s_p[0, 3:6] = torch.from_numpy((Rot.from_rotvec(axis / np.linalg.norm(axis) * np.deg2rad(theta_deg)) * Rot.from_rotvec(target)).as_rotvec()).float()
    ret = []
    for _ in range(steps):
        s_u, s_p = ft(x, s_u), ft(x, s_p)
        ret.append(_angle(s_p[0, 3:6].double().numpy(), s_u[0, 3:6].double().numpy()) / np.deg2rad(theta_deg))
    return np.array(ret)


@pytest.mark.parametrize("a", [0.25, 0.5, 0.75])
@pytest.mark.parametrize("theta", [10.0, 20.0, 90.0])
def test_response_free_tracker_retention_is_one_minus_a_to_the_k(a, theta):
    """The pre-registered debug-gate law (docs/DT_RENDER_TRACK_PREREG.md section 3): with a tracker that does not respond
    to the state the filter alone keeps (1 - a)^k of an injected root rotation after k steps."""
    ret = _retention_curve(a, 0.0, theta)
    assert np.allclose(ret, (1.0 - a) ** np.arange(1, 7), atol=2e-5)


@pytest.mark.parametrize("a", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("rho", [0.0, 0.24, 0.5])
@pytest.mark.parametrize("theta", [10.0, 90.0])
def test_parallel_response_obeys_the_magnitude_law_exactly(a, rho, theta):
    """A tracker whose response is parallel to the perturbation with strength rho: retention(k) = ((1 - a) + a rho)^k."""
    ret = _retention_curve(a, rho, theta)
    assert np.allclose(ret, ((1.0 - a) + a * rho) ** np.arange(1, 7), atol=2e-5)


class _Skew(nn.Module):
    """Root stand-in whose response to a perturbation is NOT parallel to it: for a state R the output root is
    `Exp(rho M e) R_out0`, with e = Log(R R_ref^-1) the rotation of the state away from the reference state and M the
    rotation by `phi_deg` about `n`. At the reference state it returns `R_out0`; the change of the tracker's output is
    f = rho M e exactly."""

    encoder_name = ""

    def __init__(self, ref_rv, out0_rv, rho, phi_deg, n=(0.0, 0.0, 1.0)):
        super().__init__()
        self.r_ref, self.r_out0 = Rot.from_rotvec(ref_rv), Rot.from_rotvec(out0_rv)
        self.rho = float(rho)
        n = np.asarray(n, np.float64)
        self.M = Rot.from_rotvec(n / np.linalg.norm(n) * np.deg2rad(phi_deg)).as_matrix()

    def forward(self, x, prevpos, betas=None, camera_K=None):
        out = prevpos.clone()
        for i in range(prevpos.shape[0]):
            e = (Rot.from_rotvec(prevpos[i, 3:6].double().numpy()) * self.r_ref.inv()).as_rotvec()
            f = self.rho * (self.M @ e)
            out[i, 3:6] = torch.from_numpy((Rot.from_rotvec(f) * self.r_out0).as_rotvec()).float()
        return out


def _skew_branch(a, rho, phi_deg, theta_deg=10.0):
    """One perturbation through `FilteredTracker(_Tap(_Skew))` with `law_trial`: returns (e, f, g, bound)."""
    ref = np.array([0.4, -0.2, 0.7])
    out0 = (Rot.from_rotvec(np.array([0.02, 0.01, -0.025])) * Rot.from_rotvec(ref)).as_rotvec()      # a step of about 2 deg
    tap = FE._Tap(_Skew(ref, out0, rho, phi_deg)).eval()
    net = FilteredTracker(tap, a, 1.0, 1.0).eval()
    prev = _prev(1)[0].numpy().copy()
    prev[3:6] = ref.astype(np.float32)
    e = np.array([1.0, 0.5, -0.3])
    e = e / np.linalg.norm(e) * np.deg2rad(theta_deg)
    _, res = FE.law_trial(net, tap, _x(1), prev, [e])
    return e, res[0]["f"], res[0]["g"], FE.magnitude_bound(rho, a)


def test_law_trial_sees_the_trackers_own_response():
    e, f, g, _ = _skew_branch(0.5, 0.5, 90.0)
    M = Rot.from_rotvec(np.array([0.0, 0.0, 1.0]) * np.deg2rad(90.0)).as_matrix()
    assert np.allclose(f, 0.5 * (M @ e), atol=1e-6)                 # f = rho M e: the response, not the filtered output
    assert np.dot(f, e) / (np.linalg.norm(f) * np.linalg.norm(e)) == pytest.approx(np.dot(e, M @ e) / np.dot(e, e), abs=1e-5)
    assert not np.allclose(g, f, atol=1e-3)                          # the filtered change is not the tracker's


@pytest.mark.parametrize("phi", [0.0, 30.0, 60.0, 90.0, 150.0, 180.0])
@pytest.mark.parametrize("a", [0.25, 0.5, 0.75])
def test_first_step_retention_is_the_vector_sum_and_the_magnitude_law_an_upper_bound(a, phi):
    """The filter adds rotation vectors: the first fed-back output changes by g = (1 - a) e + a f (first order; the
    correction is perpendicular to both and enters |g| at second order), so |g| / theta = |(1 - a) e + a f| / theta, which
    is bounded by (1 - a) + a rho and reaches the bound only when f is parallel to e."""
    rho = 0.5
    e, f, g, bound = _skew_branch(a, rho, phi)
    theta = np.linalg.norm(e)
    assert np.linalg.norm(f) / theta == pytest.approx(rho, abs=5e-6)
    vec = FE.vector_law(e, f, a)
    assert np.linalg.norm(g) / theta == pytest.approx(vec, abs=3e-3)           # the vector form describes the loop
    assert np.linalg.norm(g) / theta <= bound + 3e-3                           # the magnitude form is an upper bound
    if phi == 0.0:
        assert np.linalg.norm(g) / theta == pytest.approx(bound, abs=3e-3)     # ... reached for a parallel response
    if phi >= 90.0:
        assert np.linalg.norm(g) / theta < bound - 0.03                        # ... and clearly exceeded otherwise


def test_vector_law_special_cases_and_the_triangle_bound():
    e = np.array([0.0, 0.0, 0.2])
    assert FE.vector_law(e, 0 * e, 0.5) == pytest.approx(0.5)                  # no response: 1 - a
    assert FE.vector_law(e, e, 0.3) == pytest.approx(1.0)                      # the tracker hands the perturbation back
    assert FE.vector_law(e, 0.24 * e, 0.5) == pytest.approx(0.62)              # parallel: the magnitude form is exact
    assert FE.vector_law(e, -0.5 * e, 0.5) == pytest.approx(0.25)              # opposing response
    assert FE.vector_law(e, np.array([0.1, 0.0, 0.0]), 0.5) == pytest.approx(math.hypot(0.5, 0.25))   # perpendicular, rho = 0.5
    rng = np.random.default_rng(0)
    for _ in range(300):
        e = rng.normal(size=3)
        f = rng.normal(size=3) * rng.uniform(0.0, 1.5)
        a = rng.uniform()
        rho = np.linalg.norm(f) / np.linalg.norm(e)
        assert FE.vector_law(e, f, a) <= FE.magnitude_bound(rho, a) + 1e-12


def test_aggregate_law_rows():
    e0, e1, e2 = np.array([0.0, 0.0, 0.2]), np.array([0.2, 0.0, 0.0]), np.array([0.0, 0.2, 0.0])
    a = 0.5
    f1 = np.array([0.0, 0.1, 0.0])
    rows = [{"e": e0, "f": 0.5 * e0, "g": 0.75 * e0},                          # parallel response, rho 0.5
            {"e": e1, "f": f1, "g": 0.5 * e1 + 0.5 * f1},                      # perpendicular response, rho 0.5
            {"e": e2, "f": np.zeros(3), "g": e2, "held": True},                # empty packet: the filter held prev
            {"e": e0, "f": -0.5 * e0, "g": 0.25 * e0}]                         # opposing response, rho 0.5
    agg = FE.aggregate_law(rows, a)
    pair = [0.75, math.hypot(0.5, 0.25), 1.0, 0.25]
    assert agg["n_trials"] == 4 and agg["n_held"] == 1
    assert agg["k1_cpu_pair"] == pytest.approx(np.mean(pair))
    assert agg["k1_vector_law"] == pytest.approx(np.mean(pair)) and agg["vector_law_max_abs_trial_diff"] < 1e-12
    assert agg["rho_f"] == pytest.approx((0.5 + 0.5 + 1.0 + 0.5) / 4)          # the held trial counts as f = e
    assert agg["cos_f_e_mean"] == pytest.approx((1.0 + 0.0 + 1.0 - 1.0) / 4)   # signed: an opposing response counts negative
    assert agg["cos_f_e_median"] == pytest.approx(0.5)
    assert agg["k1_magnitude_bound_rho_f"] == pytest.approx(FE.magnitude_bound(agg["rho_f"], a))
    assert agg["k1_magnitude_bound_rho_f"] >= agg["k1_vector_law"]


def test_trial_plan_follows_evalx_perturb_trials():
    """Every 20th step of a segment once it is 1000 ms old, provided 20 steps remain in the segment."""
    run = np.array([0] * 60 + [1] * 30)
    elapsed = np.concatenate([49 + 50 * np.arange(60), 49 + 50 * np.arange(30)])
    assert FE.trial_plan(run, elapsed) == [20, 40]                              # the segment-1 candidate (index 80) has no 20 steps left
    assert FE.trial_plan(run, elapsed, horizon=21) == [20]                      # 40 + 21 > 60
    assert FE.trial_plan(run, elapsed, min_elapsed=3000) == []                  # no step is 3 s old
    assert FE.trial_plan(run[:25], elapsed[:25]) == []                          # too short
    axes = FE.trial_axes(3)
    ref = np.random.default_rng(0).standard_normal(3)
    assert axes.shape == (3, 3) and np.allclose(axes[0], ref / np.linalg.norm(ref))
    assert np.allclose(np.linalg.norm(axes, axis=1), 1.0) and np.array_equal(axes, FE.trial_axes(3))


def _fake_sequence(monkeypatch, empty_ends=()):
    """Replace the data access of the law check by a synthetic sequence: no events, a nonzero packet for every step
    except those whose end time is in `empty_ends` (an empty packet)."""
    monkeypatch.setattr(FE.EX.ET, "load_sequence",
                        lambda root, d, s: (None, None, {"betas": np.zeros(10, np.float32), "camera_K": np.eye(3, dtype=np.float32)}, None))
    monkeypatch.setattr(FE.EX.ET, "build_lnes",
                        lambda ev, off, end, step, chans: (np.zeros((6, 7, 2), np.float32) if int(end) in empty_ends
                                                           else np.full((6, 7, 2), 0.3, np.float32)))
    monkeypatch.setattr(FE.EX.EV, "event_channels", lambda cfg: None)


def _fake_stored(tmp_path, target, k1_stored, name="x.npz"):
    """The per-step arrays of a stored evaluation of a closed loop that sits at the target rotation (so a `_Contract`
    tracker responds in parallel to every perturbation), with 2 segments and the stored perturbation retention."""
    run = np.array([0] * 60 + [1] * 50)
    elapsed = np.concatenate([49 + 50 * np.arange(60), 49 + 50 * np.arange(50)])
    pred = np.tile(_prev(1)[0].numpy(), (110, 1))
    pred[:, 3:6] = np.asarray(target, np.float32)
    arrays = {"model|s0|pred": pred, "model|s0|run": run, "model|s0|elapsed": elapsed, "model|s0|end": 49 + 50 * np.arange(110)}
    for th in (10, 20):
        arrays[f"perturb|s0|{th}|div"] = np.full((3, 20), th * k1_stored, np.float32)       # 3 trials: steps 20, 40, 80
    p = tmp_path / name
    np.savez_compressed(p, **arrays)
    return p


@pytest.mark.parametrize("gains", [(0.5, 1.0, 0.5), (0.25, 1.0, 1.0), None])
def test_law_check_tag_on_a_stand_in(monkeypatch, tmp_path, gains):
    """End to end on synthetic data: a tracker with a parallel response rho at a loop that sits at its target, so every
    trial has f = rho e: the retention is (1 - a) + a rho in all of its forms, the response is rho with cosine 1."""
    _fake_sequence(monkeypatch)
    target, rho = np.array([0.3, -0.5, 0.2]), 0.3
    a = 1.0 if gains is None else gains[0]
    k1 = (1.0 - a) + a * rho
    npz = _fake_stored(tmp_path, target, k1)
    out = FE.law_check_tag(_Contract(target, rho), gains, {}, tmp_path, [("s0", "d0")], npz)
    assert set(out) == {"10", "20"}
    for th in ("10", "20"):
        v = out[th]
        assert v["n_trials"] == 3 and v["n_held"] == 0
        for key in ("k1_cpu_pair", "k1_vector_law", "k1_magnitude_bound_rho_f", "k1_stored_gpu", "k1_cpu_vs_stored_pred"):
            assert v[key] == pytest.approx(k1, abs=2e-5), key
        assert v["rho_f"] == pytest.approx(rho, abs=2e-5) and v["cos_f_e_mean"] == pytest.approx(1.0, abs=1e-6)
        assert v["max_abs_trial_diff_vs_stored"] < 5e-5 and v["vector_law_max_abs_trial_diff"] < 5e-5


def test_law_check_tag_counts_empty_packets_as_held(monkeypatch, tmp_path):
    """A trial whose packet is empty is held by the filter (the fed-back output keeps the whole perturbation): it is
    counted as f = e, so it does not break the vector law, and it is reported."""
    target, rho, a = np.array([0.3, -0.5, 0.2]), 0.3, 0.5
    _fake_sequence(monkeypatch, empty_ends=(49 + 50 * 40,))                                # trial at step 40
    k1 = (1.0 - a) + a * rho
    npz = _fake_stored(tmp_path, target, k1)
    v = FE.law_check_tag(_Contract(target, rho), (a, 1.0, 0.5), {}, tmp_path, [("s0", "d0")], npz)["10"]
    assert v["n_trials"] == 3 and v["n_held"] == 1
    assert v["k1_cpu_pair"] == pytest.approx((2 * k1 + 1.0) / 3, abs=2e-5)
    assert v["k1_vector_law"] == pytest.approx(v["k1_cpu_pair"], abs=2e-5)
    assert v["rho_f"] == pytest.approx((2 * rho + 1.0) / 3, abs=2e-5)


def test_discover_perturb_npz(tmp_path):
    np.savez_compressed(tmp_path / "a.npz", **{"model|s|pred": np.zeros(1)})
    assert FE.discover_perturb_npz(tmp_path, "a") is None
    np.savez_compressed(tmp_path / "a_pert.npz", **{"perturb|s|10|div": np.zeros((1, 20))})
    assert FE.discover_perturb_npz(tmp_path, "a") == tmp_path / "a_pert.npz"             # the supplementary evaluation
    np.savez_compressed(tmp_path / "b.npz", **{"perturb|s|10|div": np.zeros((1, 20))})
    np.savez_compressed(tmp_path / "b_pert.npz", **{"perturb|s|10|div": np.zeros((1, 20))})
    assert FE.discover_perturb_npz(tmp_path, "b") == tmp_path / "b.npz"                  # the evaluation itself first
    assert FE.discover_perturb_npz(tmp_path, "c") is None


_STORED = REPO / "outputs" / "dt" / "filter"


@pytest.mark.skipif(not (_STORED / "rt_cnntrack_s3407" / "raw.npz").exists(), reason="stored filter evaluations not present")
@pytest.mark.parametrize("seed", [3407, 3408])
@pytest.mark.parametrize("tag", ["raw", "r0.5_f1.0_t0.5"])
def test_trial_plan_matches_the_stored_trials(seed, tag):
    """The law check replays exactly the trials the evaluation stored: same count per sequence."""
    z = np.load(_STORED / f"rt_cnntrack_s{seed}" / f"{tag}.npz")
    seqs = sorted({k.split("|")[1] for k in z.files if k.startswith("perturb|")})
    assert seqs
    for s in seqs:
        plan = FE.trial_plan(z[f"model|{s}|run"], z[f"model|{s}|elapsed"])
        assert len(plan) == len(z[f"perturb|{s}|10|div"]) == len(z[f"perturb|{s}|20|div"]) > 0


# ------------------------------------------------------------------------------------------ the tool
def test_gain_parsing_and_tags():
    g = FE.parse_gains("1,1,1; 0.5,1,0.5;0.25,1.0,0.25;")
    assert g == [(1.0, 1.0, 1.0), (0.5, 1.0, 0.5), (0.25, 1.0, 0.25)]
    assert [FE.gain_tag(x) for x in g] == ["r1.0_f1.0_t1.0", "r0.5_f1.0_t0.5", "r0.25_f1.0_t0.25"]
    for bad in ("0.5,1", "1.5,1,1", "-0.1,1,1", "nan,1,1", "", "1,1,1;1,1,1", "a,b,c"):
        with pytest.raises(ValueError):
            FE.parse_gains(bad)
    assert FE.BASE == (1.0, 1.0, 1.0) and FE.PRIMARY == (0.5, 1.0, 0.5) and len(FE.SENSITIVITY) == 6


def test_leaf_diffs_and_compare_runs():
    a = {"x": 1.0, "n": {"v": [1.0, 2.0, None], "s": "u"}, "only_a": 3.0, "flag": True}
    b = {"x": 1.0 + 2e-7, "n": {"v": [1.0, 2.0, None], "s": "w"}, "only_b": 5.0, "flag": False}
    d = dict(FE.leaf_diffs(a, b))
    assert set(d) == {"/x", "/n/v[0]", "/n/v[1]", "/n/v[2]"} and abs(d["/x"] - 2e-7) < 1e-12 and d["/n/v[2]"] == 0.0
    assert dict(FE.leaf_diffs({"v": [1, 2]}, {"v": [1, 2, 3]}))["/v"] == float("inf")
    assert dict(FE.leaf_diffs({"v": None}, {"v": 1.0}))["/v"] == float("inf")
    c = FE.compare_runs(a, b)
    assert c["n_leaves"] == 4 and c["n_exact"] == 3 and c["worst_path"] == "/x"


def test_paired_block_ci():
    delta = np.full(100, 0.5)
    blocks = np.repeat(np.arange(10), 10)
    assert FE.paired_block_ci(delta, blocks) == pytest.approx([0.5, 0.5, 0.5])
    rng = np.random.default_rng(0)
    d = rng.normal(0.2, 1.0, 400)
    m, lo, hi = FE.paired_block_ci(d, np.repeat(np.arange(20), 20))
    assert lo < m < hi and m == pytest.approx(d.mean())


def test_block_signs_and_interval_flags():
    delta = np.array([-1.0, -1.0, 2.0, 2.0, -3.0, 1.0])
    blocks = np.array([0, 0, 1, 1, 2, 2])                       # block means -1, +2, -1
    assert FE.block_signs(delta, blocks) == (3, 2)
    assert FE.interval_vs_zero(-0.9, -0.2) == "excludes 0"
    assert FE.interval_vs_zero(0.2, 0.9) == "excludes 0"
    assert FE.interval_vs_zero(-0.781, -0.002) == "touches 0"   # seed 3408's primary RA interval: an end inside the Monte-Carlo error
    assert FE.interval_vs_zero(-0.72, -0.010) == "touches 0"    # ... and an end the resampling seed can still move across the margin
    assert FE.interval_vs_zero(0.004, 0.5) == "touches 0"
    assert FE.interval_vs_zero(-0.78, 0.002) == "covers 0"
    assert FE.interval_vs_zero(-0.909, -0.230) == "excludes 0"  # seed 3407's primary RA interval
    # the one-sided sign tail of the improving blocks: a fair coin gives >= 12 of 14 once in ~150, >= 9 of 14 once in ~5
    assert FE.sign_tail(12, 14) == pytest.approx(106 / 16384) and FE.sign_tail(9, 14) == pytest.approx(3473 / 16384)
    assert FE.sign_tail(0, 5) == 1.0 and FE.sign_tail(5, 5) == pytest.approx(1 / 32)


def _eval_json(ra, glo=None, loc=None, jit=1.0):
    seq = lambda n, r, ep: {"mpjpe_ra_mm": [r, r - 1, r + 1], "mpvpe_ra_mm": [r - 3, 0, 0], "mpjpe_abs_mm": [r + 40, 0, 0],   # noqa: E731
                            "mpvpe_abs_mm": [r + 40, 0, 0], "root_rot_deg": [r / 2, 0, 0], "transl_mm": [r * 3, 0, 0],
                            "n_frames": n, "failure": {"bad_step_frac": ep / n, "episodes": 1 if ep else 0, "fail_time_frac": 0.0,
                                                       "longest_s": 1.0 if ep else 0.0},
                            "motion": {"root_speed_ratio": 1.5, "finger_speed_ratio": 0.5}}
    g, l = (ra if glo is None else glo), (ra if loc is None else loc)
    n = 100 + 300
    return {"model": {"n_frames": n, "overall": {"mpjpe_ra_mm": (100 * g + 300 * l) / n, "mpvpe_ra_mm": 1.0, "mpjpe_abs_mm": 2.0,
                                                 "mpvpe_abs_mm": 2.0, "root_rot_deg": 3.0, "transl_mm": 4.0},
                      "zgz_global": seq(100, g, 0), "zgz_local": seq(300, l, 20), "jitter": {"jit_pred_mm": jit, "acc_err_mm": 2 * jit}}}


def test_summarize_weights_and_deltas():
    s = FE.summarize(_eval_json(0.0, glo=10.0, loc=20.0, jit=1.0))
    assert s["ra_mm"]["global"] == 10.0 and s["ra_mm"]["local"] == 20.0 and s["ra_mm"]["overall"] == pytest.approx(17.5)
    assert s["failures"] == {"episodes": 1, "bad_steps": 20, "bad_step_frac": pytest.approx(20 / 400),
                             "fail_time_frac": 0.0, "longest_s": 1.0}
    assert s["jitter"]["jit_pred_mm"] == 1.0 and s["root_speed_ratio"]["overall"] == 1.5
    s2 = FE.summarize(_eval_json(0.0, glo=9.0, loc=20.0, jit=0.5))
    d = FE.tree_delta(s2, s)
    assert d["ra_mm"]["global"] == -1.0 and d["ra_mm"]["local"] == 0.0 and d["jitter"]["jit_pred_mm"] == -0.5
    assert FE.tree_mean([s, s2])["ra_mm"]["global"] == 9.5
    assert FE.seq_names(_eval_json(1.0)["model"]) == ["zgz_global", "zgz_local"]       # `jitter` is not a sequence


def test_filter_law_arithmetic():
    """The magnitude bound `(1 - a) + a rho` with rho the raw tracker's k1; the gap to it is a gap, not a tracker response."""
    pert = lambda r: {"10": {"retention_k1_k2_k5_k10_k20": [r, 0.1, 0.1, 0.1, 0.1]}}     # noqa: E731
    law = FE.filter_law(pert(0.24), pert(0.58), 0.5)["10"]
    assert law["k1_magnitude_bound"] == pytest.approx(0.62) and law["filter_alone_1_minus_a"] == 0.5
    assert law["k1_raw_rho"] == 0.24 and law["k1_measured"] == 0.58
    assert law["gap_measured_minus_bound"] == pytest.approx(0.58 - 0.62)
    assert not any("implied" in k or "response" in k for k in law)       # the earlier 'implied tracker response' was a misreading
    no_root = FE.filter_law(pert(0.24), pert(0.25), 1.0)["10"]           # no root filter: the bound is rho itself
    assert no_root["k1_magnitude_bound"] == pytest.approx(0.24) and no_root["filter_alone_1_minus_a"] == 0.0
    assert "20" not in FE.filter_law(pert(0.24), {"20": pert(0.3)["10"]}, 0.5)       # a theta missing from the raw run is skipped


def _synthetic_report(tmp_path, law=True):
    """Two seeds-worth of structure with one run: (1,1,1) and the primary, with per-step npz, perturbation and law check."""
    run = "rt_x_s1"
    rd = tmp_path / "filter" / run
    rd.mkdir(parents=True)
    rng = np.random.default_rng(0)
    pert = lambda k1: {th: {"trials": 4, "div_deg": [float(th) * k1] * 20, "excess_deg": [0.0] * 20,   # noqa: E731
                            "retention_k1_k2_k5_k10_k20": [k1, k1 / 2, k1 / 4, k1 / 8, k1 / 16],
                            "excess_k1_k2_k5_k10_k20_deg": [0.0] * 5, "half_life_steps": 0} for th in ("10", "20")}
    for tag, gains, shift, k1 in (("r1.0_f1.0_t1.0", (1.0, 1.0, 1.0), 0.0, 0.24), ("r0.5_f1.0_t0.5", (0.5, 1.0, 0.5), -0.4, 0.58)):
        j = _eval_json(10.0 + shift, glo=9.0 + shift, loc=18.0 + shift)
        j["model"]["jitter"] = {"jit_pred_mm": 7.0 + shift, "acc_err_mm": 8.0 + 4 * shift, "acc_ratio": 2.0 + shift,
                                "acc_err_only_transl_mm": 6.0, "acc_err_only_root_mm": 7.0, "acc_err_only_fingers_mm": 4.0,
                                "rot_acc_pred_deg": 3.0 + shift, "jit_gt_mm": 5.8, "acc_gt_mm": 3.9, "rot_acc_gt_deg": 3.2}
        j.update({"step": 6000, "gains": {"root": gains[0], "rest": gains[1], "trans": gains[2]}, "perturb": pert(k1),
                  "env": {"md5": {"model/model.py": "0" * 32}}, "flags": {"perturb": True}})
        (rd / f"{tag}.json").write_text(json.dumps(j))
        arrays = {}
        for s, n in (("zgz_global", 100), ("zgz_local", 300)):
            arrays[f"model|{s}|end"] = 49 + 50 * np.arange(n)
            arrays[f"model|{s}|mpjpe_ra_mm"] = (10.0 + shift + rng.normal(0, 0.3, n)).astype(np.float32)
            arrays[f"model|{s}|root_rot_deg"] = (5.0 + shift + rng.normal(0, 0.3, n)).astype(np.float32)
        np.savez_compressed(rd / f"{tag}.npz", **arrays)
    if law:
        per = lambda a, rho, pair: {"n_trials": 4, "n_held": 0, "k1_cpu_pair": pair, "rho_f": rho,   # noqa: E731
                                    "cos_f_e_mean": 0.7, "cos_f_e_median": 0.8, "k1_vector_law": pair,
                                    "vector_law_max_abs_trial_diff": 1e-4, "k1_magnitude_bound_rho_f": (1 - a) + a * rho,
                                    "k1_stored_gpu": pair, "k1_cpu_vs_stored_pred": pair, "max_abs_trial_diff_vs_stored": 1e-4}
        tags = {"raw": {"gains": None, "npz": "raw.npz", "per_theta": {"10": per(1.0, 0.24, 0.24), "20": per(1.0, 0.22, 0.22)}},
                "r0.5_f1.0_t0.5": {"gains": [0.5, 1.0, 0.5], "npz": "x.npz",
                                   "per_theta": {"10": per(0.5, 0.24, 0.58), "20": per(0.5, 0.22, 0.57)}}}
        (rd / FE.LAW_CHECK_NAME).write_text(json.dumps({"run": run, "tags": tags}))
    return tmp_path / "filter", run


def test_report_text_states_the_corrected_law(tmp_path):
    """build_report + render_markdown on synthetic evaluations: the magnitude law is presented as an upper bound, the
    'implied tracker response' reading is gone, the bootstrap flags and the block counts are present, the law check is
    included, and the whole report is json-able."""
    in_dir, run = _synthetic_report(tmp_path)
    rep = FE.build_report([run], in_dir, [FE.BASE, FE.PRIMARY], FE.BASE, FE.PRIMARY, tmp_path / "no_semkine")
    json.dumps(rep)
    ci = rep["ra_delta_ci"]["1"]["r0.5_f1.0_t0.5"]
    assert ci["blocks"]["n"] == 3 and ci["blocks"]["n_boot"] == FE.N_BOOT and ci["mpjpe_ra_mm"][0] < 0   # 1 + 2 ten-second blocks
    assert ci["blocks"]["negative"]["mpjpe_ra_mm"] == 3 and FE.interval_vs_zero(*ci["mpjpe_ra_mm"][1:]) == "excludes 0"
    law = rep["filter_law"]["1"]["r0.5_f1.0_t0.5"]["10"]
    assert law["k1_magnitude_bound"] == pytest.approx(0.62) and law["gap_measured_minus_bound"] == pytest.approx(0.58 - 0.62)
    md = FE.render_markdown(rep)
    assert not any("implied" in line.lower() for line in md.splitlines() if line.startswith("|"))      # no table column reads the gap as a response
    assert "upper bound" in md and "magnitude bound" in md
    assert "vector law" in md and "interval vs 0" in md and "blocks with lower RA" in md
    assert "(1 - a_root)^k" in md and "no verdict" in md
    assert "## 5. The registered criteria" in md and "the filter is effective" in md and "mean delta RA -0.4" in md
    assert "The RA gain against sampling noise" in md and "seed 1 excludes 0 (3 of 3 blocks lower)" in md        # in the headline, not only in section 1
    assert rep["law_check"]["1"]["tags"]["raw"]["gains"] is None
    st = FE.law_check_stats(rep)
    assert st["n_rows"] == 4 and st["n_filtered"] == 2 and st["n_bound_holds"] == 2 and st["n_held"] == 0
    rep2 = FE.build_report([run], _synthetic_report(tmp_path / "b", law=False)[0], [FE.BASE, FE.PRIMARY], FE.BASE, FE.PRIMARY, tmp_path)
    md2 = FE.render_markdown(rep2)
    assert rep2["law_check"] == {} and "### 3b." not in md2 and "in 3b" not in md2      # without the CPU replay the section is left out
