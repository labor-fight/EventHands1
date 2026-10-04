"""DT2 round, package B: `semkine.anchored.AdaptiveFilter` (stateful, event-count-adaptive, device-side float64 filter)
against tiny stand-in trackers (CPU), its spec / tag helpers in `tools/dt/filter_eval.py`, and the evalx hooks
(`reset_state` / `get_state` / `set_state` / `takes_event_count`) on a synthetic sequence.

    CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_dt_filter2.py -q
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
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools" / "dt"), str(REPO / "tools" / "tracking")]
from semkine.anchored import AdaptiveFilter, FilteredTracker, GainSchedule, anchor_blend   # noqa: E402
import filter_eval as FE                                                                  # noqa: E402
import filter_sweep_2fold as SW                                                           # noqa: E402

EX = FE.EX
NODES = [[300, 0.3], [3000, 0.5], [30000, 0.8]]


class _Stand(nn.Module):
    """`prev + delta(x) + bias`, every operation row-wise (as tests/test_dt_filter.py's stand-in)."""

    def __init__(self, seed=0, scale=1.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("w", scale * 0.05 * torch.randn(51, generator=g))
        self.register_buffer("bias", scale * 0.01 * torch.randn(51, generator=g))
        self.ctx = None
        self.seen = []

    def set_hand_context(self, betas, camera_K):
        self.ctx = (betas, camera_K)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        self.seen.append((prevpos, betas, camera_K))
        s = x.reshape(x.shape[0], -1).mean(dim=1, keepdim=True)
        return prevpos + s * self.w + self.bias


class _Oracle(nn.Module):
    """A tracker that ignores `prev` and answers with a stored measurement per call (batch 1)."""

    def __init__(self, zs):
        super().__init__()
        self.zs, self.i = zs, 0

    def forward(self, x, prevpos, betas=None, camera_K=None):
        z = self.zs[self.i].to(prevpos.dtype).view(1, -1)
        self.i += 1
        return z


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


def _aa_rot(v):
    return Rot.from_rotvec(np.asarray(v, dtype=np.float64))


GAINS = [(0.5, 1.0, 0.5), (0.25, 0.5, 0.75), (1.0, 1.0, 1.0), (0.8, 1.0, 0.5), (0.0, 0.0, 0.0), (0.35, 0.75, 0.5)]


# -------------------------------------------------------------------------- degenerate parameters = FilteredTracker
@pytest.mark.parametrize("b", [1, 5])
@pytest.mark.parametrize("g", GAINS)
def test_degenerate_device_path_matches_anchor_blend_to_1e6(b, g):
    trk, x, prev = _Stand().eval(), _x(b), _prev(b)
    ref = FilteredTracker(trk, *g).eval()(x, prev)
    out = AdaptiveFilter(trk, *g).eval()(x, prev)
    assert out.dtype == ref.dtype == prev.dtype and out.shape == (b, 51)
    assert float((out - ref).abs().max()) <= 1e-6
    # an event count changes nothing for constant gains (it only decides emptiness)
    out_n = AdaptiveFilter(trk, *g).eval()(x, prev, n_events=torch.full((b,), 1234.0))
    assert torch.equal(out_n, out)


def test_device_path_closed_loop_stays_within_1e5_of_the_host_loop():
    trk = _Stand(scale=0.1).eval()
    host, dev = FilteredTracker(trk, 0.5, 1.0, 0.5).eval(), AdaptiveFilter(trk, 0.5, 1.0, 0.5).eval()
    ph, pd = _prev(5), _prev(5)
    for i in range(40):
        x = _x(5, seed=100 + i)
        ph, pd = host(x, ph), dev(x, pd)
        assert float((ph - pd).abs().max()) <= 1e-5, i


@pytest.mark.parametrize("b", [1, 5])
def test_host_mode_is_filtered_tracker_bitwise_also_in_a_loop(b):
    trk = _Stand(scale=0.1).eval()
    for g in GAINS:
        ft, af = FilteredTracker(trk, *g).eval(), AdaptiveFilter(trk, *g, host=True).eval()
        pf, pa = _prev(b), _prev(b)
        for i in range(25):
            x = _x(b, seed=200 + i)
            pf, pa = ft(x, pf), af(x, pa, n_events=77)
            assert torch.equal(pf, pa), (g, i)


def test_host_mode_needs_constant_gains_and_no_velocity():
    trk = _Stand()
    AdaptiveFilter(trk, 0.5, 1.0, 0.5, host=True)
    with pytest.raises(ValueError):
        AdaptiveFilter(trk, NODES, 1.0, 0.5, host=True)
    with pytest.raises(ValueError):
        AdaptiveFilter(trk, 0.5, 1.0, 0.5, beta_root=0.1, host=True)
    with pytest.raises(ValueError):
        AdaptiveFilter(trk, 0.5, 1.0, 0.5, beta_trans=0.1, host=True)


def test_class_flags_and_hand_context_forwarding():
    trk = _Stand()
    af = AdaptiveFilter(trk, 0.5, 1.0, 0.5)
    assert af.encoder_name == "" and af.takes_event_count is True
    betas, K = torch.zeros(1, 10), torch.eye(3).view(1, 3, 3)
    af.set_hand_context(betas, K)
    assert trk.ctx is not None and trk.ctx[0] is betas and trk.ctx[1] is K
    af.set_hand_context(betas, K)

    class _NoCtx(nn.Module):
        def forward(self, x, prevpos, betas=None, camera_K=None):
            return prevpos + 0.01

    AdaptiveFilter(_NoCtx(), 0.5, 1.0, 0.5).set_hand_context(betas, K)            # no hook on the tracker: no error
    af(_x(1), _prev(1), betas=betas, camera_K=K)
    assert trk.seen[-1][1] is betas and trk.seen[-1][2] is K


def test_the_tracker_always_sees_the_filtered_prev_not_a_prediction():
    """alpha-beta predicts inside the filter only: the network's input `prev` is the caller's (filtered) state."""
    trk = _Stand().eval()
    af = AdaptiveFilter(trk, 0.5, 1.0, 0.5, beta_root=0.5, beta_trans=0.5).eval()
    prev = _prev(1)
    af.reset_state(prev)
    af.set_state({"v_root": torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float64),
                  "v_trans": torch.tensor([[0.01, 0.02, 0.03]], dtype=torch.float64)})
    af(_x(1), prev)
    assert torch.equal(trk.seen[-1][0], prev)


# ----------------------------------------------------------------------------------------- gain schedule
def test_gain_schedule_nodes_clamps_and_log_linear_interpolation():
    gs = GainSchedule(NODES)
    n = torch.tensor([1.0, 100.0, 300.0, 3000.0, 30000.0, 1e6], dtype=torch.float64)
    assert torch.allclose(gs(n).view(-1), torch.tensor([0.3, 0.3, 0.3, 0.5, 0.8, 0.8], dtype=torch.float64), atol=1e-12)
    mid = torch.tensor([math.sqrt(300 * 3000), math.sqrt(3000 * 30000)], dtype=torch.float64)    # log-midpoints
    assert torch.allclose(gs(mid).view(-1), torch.tensor([0.4, 0.65], dtype=torch.float64), atol=1e-12)
    for v in (437.0, 1500.0, 9000.0, 21000.0):                                  # against numpy's own interpolation
        ref = np.interp(math.log10(v), [math.log10(300), math.log10(3000), math.log10(30000)], [0.3, 0.5, 0.8])
        assert abs(float(gs(torch.tensor([v], dtype=torch.float64))) - ref) < 1e-12
    assert gs(torch.tensor([0.0], dtype=torch.float64)).item() == pytest.approx(0.3)          # n < 1 counts as 1
    assert GainSchedule(0.5).const == 0.5 and GainSchedule([(10, 0.7), (100, 0.7)]).const == 0.7
    assert GainSchedule([[3000, 0.5], [300, 0.3]]).nodes == ((300.0, 0.3), (3000.0, 0.5))      # order does not matter
    for bad in ([], [(0, 0.5)], [(10, 1.5)], [(10, 0.5), (10, 0.6)], 1.2, -0.1):
        with pytest.raises(ValueError):
            GainSchedule(bad)


def test_event_count_gain_is_applied_row_by_row():
    trk, x, prev = _Stand().eval(), _x(7), _prev(7)
    n = torch.tensor([100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0, 1e6])
    out = AdaptiveFilter(trk, NODES, 1.0, 0.5).eval()(x, prev, n_events=n)
    z = trk(x, prev)
    xs, ys = [math.log10(300), math.log10(3000), math.log10(30000)], [0.3, 0.5, 0.8]
    for i in range(7):
        a = float(np.interp(math.log10(float(n[i])), xs, ys))
        ref = anchor_blend(prev[i:i + 1], z[i:i + 1], a, 1.0, 0.5)
        assert float((out[i:i + 1] - ref).abs().max()) <= 1e-6, i
    # every gain can be a schedule; fingers and translation too
    out2 = AdaptiveFilter(trk, 0.5, [[1000, 0.2], [100000, 1.0]], [[1000, 0.0], [100000, 0.6]]).eval()(x, prev, n_events=n)
    for i in range(7):
        t = (math.log10(float(n[i])) - 3.0) / 2.0
        af_, at_ = float(np.clip(0.2 + 0.8 * t, 0.2, 1.0)), float(np.clip(0.6 * t, 0.0, 0.6))
        ref = anchor_blend(prev[i:i + 1], z[i:i + 1], 0.5, af_, at_)
        assert float((out2[i:i + 1] - ref).abs().max()) <= 1e-6, i


def test_missing_event_count_is_estimated_by_the_nonzero_lnes_entries():
    trk, x, prev = _Stand().eval(), _x(3), _prev(3)
    x[1, :2] = 0.0                                                      # fewer non-zeros in row 1
    x[2, :, :, 0] = 0.0
    nnz = (x.reshape(3, -1) != 0).sum(1).double()
    f = AdaptiveFilter(trk, [[10, 0.2], [80, 0.9]], 1.0, 0.5).eval()
    assert torch.equal(f(x, prev), f(x, prev, n_events=nnz))
    assert not torch.equal(f(x, prev), f(x, prev, n_events=torch.full((3,), 1e6)))


# --------------------------------------------------------------------------------------- empty packets
@pytest.mark.parametrize("b", [1, 5])
def test_empty_lnes_returns_prev_bitwise_and_decays_the_velocity(b):
    trk, prev = _Stand().eval(), _prev(b)
    x = torch.zeros(b, 6, 7, 2)
    assert not torch.equal(trk(x, prev), prev)                           # the stand-in itself does not hold
    for kw in (dict(), dict(beta_root=0.3, beta_trans=0.3), dict(a_root=NODES)):
        a_root = kw.pop("a_root", 0.5)
        af = AdaptiveFilter(trk, a_root, 1.0, 0.5, **kw).eval()
        af.reset_state(prev)
        v = {"v_root": torch.full((b, 3), 0.04, dtype=torch.float64), "v_trans": torch.full((b, 3), -0.02, dtype=torch.float64)}
        af.set_state(v)
        for n_events in (None, 0, 5000):
            assert torch.equal(af(x, prev, n_events=n_events), prev)
        s = af.get_state()                                              # three empty packets: decay 0.5**3
        if kw:
            assert torch.allclose(s["v_root"], v["v_root"] * 0.125, atol=1e-15)
            assert torch.allclose(s["v_trans"], v["v_trans"] * 0.125, atol=1e-15)


@pytest.mark.parametrize("b", [1, 5])
def test_zero_event_count_returns_prev_even_with_a_nonempty_lnes(b):
    trk, x, prev = _Stand().eval(), _x(b), _prev(b)
    for spec in (dict(a_root=0.5), dict(a_root=NODES, beta_root=0.2, beta_trans=0.2, decay=0.25)):
        af = AdaptiveFilter(trk, a_rest=1.0, a_trans=0.5, **spec).eval()
        af.reset_state(prev)
        assert torch.equal(af(x, prev, n_events=0), prev)
        assert torch.equal(af(x, prev, n_events=torch.zeros(b)), prev)
        assert not torch.equal(af(x, prev, n_events=10), prev)
    # host mode as well
    assert torch.equal(AdaptiveFilter(trk, 0.5, 1.0, 0.5, host=True).eval()(x, prev, n_events=0), prev)


def test_empty_rows_hold_while_other_rows_filter():
    trk, prev, x = _Stand().eval(), _prev(5), _x(5)
    x[0], x[3] = 0.0, 0.0
    n = torch.tensor([50.0, 0.0, 800.0, 900.0, 0.0])                    # rows 1 and 4: zero count; 0 and 3: empty LNES
    af = AdaptiveFilter(trk, NODES, 1.0, 0.5, beta_root=0.2, beta_trans=0.2).eval()
    af.reset_state(prev)
    af.set_state({"v_root": torch.full((5, 3), 0.01, dtype=torch.float64), "v_trans": torch.full((5, 3), 0.002, dtype=torch.float64)})
    out = af(x, prev, n_events=n)
    for i in (0, 1, 3, 4):
        assert torch.equal(out[i], prev[i]), i
    solo = AdaptiveFilter(trk, NODES, 1.0, 0.5, beta_root=0.2, beta_trans=0.2).eval()
    for i in (2,):
        solo.reset_state(prev[i:i + 1])
        solo.set_state({"v_root": torch.full((1, 3), 0.01, dtype=torch.float64), "v_trans": torch.full((1, 3), 0.002, dtype=torch.float64)})
        assert torch.equal(out[i:i + 1], solo(x[i:i + 1], prev[i:i + 1], n_events=n[i:i + 1]))
    s = af.get_state()
    assert torch.allclose(s["v_root"][1], torch.full((3,), 0.005, dtype=torch.float64))          # decayed
    assert not torch.allclose(s["v_root"][2], torch.full((3,), 0.005, dtype=torch.float64))      # updated


# ---------------------------------------------------------------------------------------- alpha-beta
def _rotvec_diff_deg(a, b):
    return float(np.rad2deg((_aa_rot(a).inv() * _aa_rot(b)).magnitude()))


def test_prediction_with_zero_gain_is_prev_plus_velocity():
    """a = 0: x+ = x- = prev (+) v: translation prev_t + v_t, root Exp(v_r) R_prev, fingers prev."""
    trk, x, prev = _Stand().eval(), _x(5), _prev(5)
    af = AdaptiveFilter(trk, 0.0, 0.0, 0.0, beta_root=0.5, beta_trans=0.5).eval()
    af.reset_state(prev)
    vr, vt = 0.05 * torch.randn(5, 3, dtype=torch.float64), 0.01 * torch.randn(5, 3, dtype=torch.float64)
    af.set_state({"v_root": vr, "v_trans": vt})
    out = af(x, prev)
    assert torch.allclose(out[:, :3], prev[:, :3] + vt.float(), atol=1e-6)
    assert torch.equal(out[:, 6:], prev[:, 6:])
    for i in range(5):
        ref = (_aa_rot(vr[i].numpy()) * _aa_rot(prev[i, 3:6].double().numpy())).as_rotvec()
        assert np.abs(out[i, 3:6].double().numpy() - ref).max() < 1e-6


def test_velocity_update_is_beta_times_the_filtered_step_difference():
    trk, x, prev = _Stand().eval(), _x(5), _prev(5)
    z = trk(x, prev)
    af = AdaptiveFilter(trk, 1.0, 1.0, 1.0, beta_root=0.5, beta_trans=0.25).eval()
    af.reset_state(prev)
    out = af(x, prev)                                                   # a = 1, v = 0: x+ = z
    s = af.get_state()
    assert torch.allclose(s["v_trans"], 0.25 * (out[:, :3].double() - prev[:, :3].double()), atol=1e-12)
    for i in range(5):
        d = (_aa_rot(out[i, 3:6].double().numpy()) * _aa_rot(prev[i, 3:6].double().numpy()).inv()).as_rotvec()
        assert np.abs(s["v_root"][i].numpy() - 0.5 * d).max() < 1e-6
    # a second step moves the velocity toward the new difference: v1 = v0 + beta (d1 - v0)
    v0 = s["v_trans"].clone()
    out2 = af(x, out)
    d1 = out2[:, :3].double() - out[:, :3].double()
    # (a = 1, so x+ is z again; with a velocity term the prediction does not enter the output at a = 1)
    assert torch.allclose(af.get_state()["v_trans"], v0 + 0.25 * (d1 - v0), atol=1e-12)
    assert z.shape == (5, 51)


def _truth(n_steps, dt_t=(0.004, -0.002, 0.001), w=(0.03, -0.05, 0.04)):
    """A hand moving at constant velocity: translation linear, root rotating at a constant spatial angular velocity."""
    ts, rs = [], []
    r0 = _aa_rot([0.4, -0.2, 0.7])
    for k in range(n_steps + 1):
        ts.append(np.array([0.0, 0.0, 0.4]) + k * np.array(dt_t))
        rs.append((_aa_rot(np.array(w) * k) * r0).as_rotvec())
    return np.stack(ts), np.stack(rs)


def _run_oracle(n_steps, **kw):
    t, r = _truth(n_steps)
    zs = []
    for k in range(1, n_steps + 1):
        z = torch.zeros(51)
        z[:3], z[3:6] = torch.from_numpy(t[k]).float(), torch.from_numpy(r[k]).float()
        zs.append(z)
    af = AdaptiveFilter(_Oracle(zs), 0.5, 1.0, 0.5, **kw).eval()
    state = torch.zeros(1, 51)
    state[0, :3], state[0, 3:6] = torch.from_numpy(t[0]).float(), torch.from_numpy(r[0]).float()
    af.reset_state(state)
    x = torch.ones(1, 3, 3, 2)
    for k in range(1, n_steps + 1):
        state = af(x, state)
    return (np.linalg.norm(state[0, :3].double().numpy() - t[n_steps]),
            _rotvec_diff_deg(state[0, 3:6].double().numpy(), r[n_steps]), af)


def test_alpha_beta_removes_the_constant_gain_lag_on_a_constant_velocity_motion():
    et0, er0, _ = _run_oracle(60)                                        # constant gain: lag (1 - a) / a * v = v
    et1, er1, af = _run_oracle(60, beta_root=0.3, beta_trans=0.3)
    assert et0 > 0.9 * 0.0046 and er0 > 0.9 * math.degrees(0.0707)       # |v_t| = 0.0046, |w| = 0.0707 rad per step
    assert et1 < 0.1 * et0 and er1 < 0.1 * er0
    s = af.get_state()                                                   # the velocity converged to the true one
    assert np.abs(s["v_trans"][0].numpy() - np.array([0.004, -0.002, 0.001])).max() < 2e-4
    assert np.abs(s["v_root"][0].numpy() - np.array([0.03, -0.05, 0.04])).max() < 2e-3


# ------------------------------------------------------------------------------------------------ state
def test_reset_state_get_state_set_state_roundtrip():
    trk, prev = _Stand(scale=0.1).eval(), _prev(5)
    af = AdaptiveFilter(trk, NODES, 1.0, 0.5, beta_root=0.3, beta_trans=0.2).eval()
    af.reset_state(prev)
    s0 = af.get_state()
    assert s0["v_root"].shape == s0["v_trans"].shape == (5, 3) and not s0["v_root"].any() and not s0["v_trans"].any()
    n = torch.tensor([200.0, 600.0, 2000.0, 9000.0, 40000.0])
    st = prev
    for i in range(6):
        st = af(_x(5, seed=300 + i), st, n_events=n)
    snap, st_snap = af.get_state(), st.clone()
    assert snap["v_root"].abs().max() > 0
    outs = []
    for i in range(4):
        st = af(_x(5, seed=400 + i), st, n_events=n)
        outs.append(st.clone())
    assert not torch.equal(af.get_state()["v_root"], snap["v_root"])      # the snapshot is a copy
    af.set_state(snap)
    st = st_snap
    for i in range(4):
        st = af(_x(5, seed=400 + i), st, n_events=n)
        assert torch.equal(st, outs[i]), i
    # numpy / foreign dtype in, copies out; a state of another batch size is re-initialised
    af2 = AdaptiveFilter(trk, NODES, 1.0, 0.5, beta_root=0.3, beta_trans=0.2).eval()
    af2.set_state({k: v.numpy().astype(np.float64) for k, v in snap.items()})
    assert torch.equal(af2.get_state()["v_root"], snap["v_root"])
    af.reset_state(_prev(2))
    af(_x(5), prev, n_events=n)
    assert af.get_state()["v_root"].shape == (5, 3)


def test_batch_five_equals_five_batch_ones_row_by_row():
    trk = _Stand(scale=0.1).eval()
    spec = dict(a_root=NODES, a_rest=0.9, a_trans=[[500, 0.2], [20000, 0.7]], beta_root=0.25, beta_trans=0.15)
    afb = AdaptiveFilter(trk, **spec).eval()
    rows = [AdaptiveFilter(trk, **spec).eval() for _ in range(5)]
    pb = _prev(5)
    afb.reset_state(pb)
    for i, r in enumerate(rows):
        r.reset_state(pb[i:i + 1])
    ps = [pb[i:i + 1].clone() for i in range(5)]
    n = torch.tensor([150.0, 800.0, 2500.0, 12000.0, 90000.0])
    for k in range(10):
        x = _x(5, seed=500 + k)
        if k == 4:
            x[2], n[2] = 0.0, 0.0                                       # an empty row in the middle of the loop
        pb = afb(x, pb, n_events=n)
        ps = [rows[i](x[i:i + 1], ps[i], n_events=n[i:i + 1]) for i in range(5)]
        for i in range(5):
            assert float((pb[i:i + 1] - ps[i]).abs().max()) <= 1e-7, (k, i)
    sb = afb.get_state()
    for i in range(5):
        for key in ("v_root", "v_trans"):
            assert torch.allclose(sb[key][i:i + 1], rows[i].get_state()[key], atol=1e-12)


# ------------------------------------------------------------------------------------------- specs
def test_canonical_spec_and_module_roundtrip():
    c = AdaptiveFilter.canonical_spec({"a_root": 0.5})
    assert c == {"a_root": 0.5, "a_rest": 1.0, "a_trans": 1.0, "beta_root": 0.0, "beta_trans": 0.0, "decay": 0.5, "host": False}
    c2 = AdaptiveFilter.canonical_spec({"a_root": NODES, "a_rest": 1, "a_trans": 0.5, "beta": 0.2, "name": "x"})
    assert c2["beta_root"] == c2["beta_trans"] == 0.2 and c2["a_trans"] == 0.5 and "name" not in c2
    assert c2["a_root"] == [[300.0, 0.3], [3000.0, 0.5], [30000.0, 0.8]]
    assert AdaptiveFilter.canonical_spec({"a_rest": 0.7})["a_trans"] == 0.7           # the translation ties to the fingers
    af = AdaptiveFilter.from_spec(_Stand(), c2)
    assert af.spec() == c2 and json.loads(json.dumps(af.spec())) == c2
    with pytest.raises(ValueError):
        AdaptiveFilter.canonical_spec({"a_root": 0.5, "gain": 3})
    with pytest.raises(ValueError):
        AdaptiveFilter.canonical_spec({"a_root": 1.5})
    with pytest.raises(ValueError):
        AdaptiveFilter(_Stand(), 0.5, beta_root=1.5)


def test_spec_tag_is_stable_readable_and_distinguishes_specs(tmp_path):
    a = {"a_root": 0.5, "a_trans": 0.5}
    b = {"a_root": 0.5, "a_rest": 1.0, "a_trans": 0.5, "beta_root": 0.0, "decay": 0.5}
    c = {"a_root": 0.5, "a_trans": 0.5, "beta": 0.2}
    d = {"a_root": NODES, "a_trans": 0.5}
    assert FE.spec_tag(a) == FE.spec_tag(b) and FE.spec_tag({**a, "name": "z"}).split("_", 1)[0] == FE.spec_tag(a).split("_", 1)[0]
    tags = [FE.spec_tag(s) for s in (a, c, d, {"a_root": 0.35, "a_trans": 0.5}, {"a_root": [[300, 0.3], [3000, 0.5], [60000, 0.8]], "a_trans": 0.5})]
    assert len(set(tags)) == 5 and len({t.split("_", 1)[0] for t in tags}) == 5
    assert FE.spec_tag(a).endswith("_r0.5_f1_t0.5") and FE.spec_tag(c).endswith("_r0.5_f1_t0.5_b0.2")
    assert FE.spec_tag(d).endswith("_rn0.3-0.5-0.8_f1_t0.5") and FE.spec_tag(a).startswith("af")
    # --filter-spec: a JSON string, a list, or a file
    assert FE.load_filter_specs(json.dumps(a)) == [a]
    assert FE.load_filter_specs(json.dumps([a, c])) == [a, c]
    p = tmp_path / "specs.json"
    p.write_text(json.dumps([d]))
    assert FE.load_filter_specs(str(p)) == [d]
    long = [{"a_root": [[300, 0.3], [3000, 0.5], [30000, 0.8]], "a_rest": 1.0, "a_trans": 0.5 + 0.001 * i} for i in range(20)]
    assert len(json.dumps(long)) > 1000 and FE.load_filter_specs(json.dumps(long)) == long      # a long JSON string is not a file name
    with pytest.raises(ValueError):
        FE.load_filter_specs(json.dumps({"a_root": 2.0}))
    # the historical triple path is untouched
    assert FE.gain_tag((0.5, 1.0, 0.5)) == "r0.5_f1.0_t0.5" and FE.parse_gains("0.5,1,0.5") == [(0.5, 1.0, 0.5)]
    ns = FE.eval_args(True, False, True)
    assert ns.window_ms == EX.STEP == 50 and FE.eval_args(window_ms=100).window_ms == 100


# ----------------------------------------------------------------------------- evalx hooks (synthetic sequence)
@pytest.fixture
def synth(monkeypatch):
    """A 2-segment (0-300 ms, 320-650 ms) synthetic sequence with two empty 50 ms packets, patched into evalx."""
    T = 700
    rng = np.random.default_rng(0)
    per_ms = rng.integers(20, 400, size=T).astype(np.int64)
    per_ms[0:100] = 0                                                    # an empty stretch (the first two packets of segment 1)
    offsets = np.concatenate([[0], np.cumsum(per_ms)]).astype(np.int64)
    events = np.zeros((int(offsets[-1]), 3), np.float32)
    pos = np.tile(_prev(1, seed=9).numpy(), (T, 1)).astype(np.float32)
    pos[:, :3] += np.arange(T)[:, None] * np.array([1e-4, 0, 0], np.float32)
    aux = {"betas": np.zeros(10, np.float32), "camera_K": np.eye(3, dtype=np.float32),
           "valid_runs_ms": np.array([[0, 300], [320, 650]])}
    monkeypatch.setattr(EX.ET, "load_sequence", lambda root, d, seq: (events, offsets, aux, pos))

    def fake_lnes(ev, off, end, window, ch):
        n = int(off[end + 1] - off[end - window + 1])
        return np.full((4, 5, 2), 0.0 if n == 0 else 0.3 + 1e-4 * n, np.float32)
    monkeypatch.setattr(EX.ET, "build_lnes", fake_lnes)
    return {"offsets": offsets, "pos": pos}


class _Spy(AdaptiveFilter):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.log = {"reset": [], "n": [], "set": []}

    def reset_state(self, prev):
        self.log["reset"].append(prev.clone())
        super().reset_state(prev)

    def set_state(self, s):
        self.log["set"].append(s)
        super().set_state(s)

    def forward(self, x, prevpos, betas=None, camera_K=None, n_events=None):
        self.log["n"].append(n_events)
        return super().forward(x, prevpos, betas, camera_K, n_events)


def _cfg():
    return {"TRACK": {}, "DATA": {}}


def test_run_sequence_calls_reset_passes_counts_and_records_state(synth):
    trk = _Stand(scale=0.1).eval()
    spy = _Spy(trk, NODES, 1.0, 0.5, beta_root=0.3, beta_trans=0.3).eval()
    r = EX.run_sequence(spy, _cfg(), Path("/nonexistent"), "d", "s", torch.device("cpu"), np.random.default_rng(0))
    ends = r["end"]
    off = synth["offsets"]
    want_n = [int(off[e + 1] - off[e - 49]) for e in ends]
    assert spy.log["n"] == want_n and list(r["count"]) == want_n
    assert len(spy.log["reset"]) == 2                                    # one reset per segment
    assert len(r["fstate"]) == len(ends) == len(r["prev"])
    first = [i for i in range(len(ends)) if r["elapsed"][i] == 49]
    assert len(first) == 2
    for i in first:                                                      # the snapshot at a segment start is the zero state
        assert not r["fstate"][i]["v_root"].any() and not r["fstate"][i]["v_trans"].any()
    assert any(float(s["v_root"].abs().max()) > 0 for s in r["fstate"])
    assert 0 in want_n                                                   # the synthetic sequence does have empty packets
    # an empty packet returns prev: the next step's prev equals this step's prev
    for i in range(len(ends) - 1):
        if want_n[i] == 0 and r["run"][i] == r["run"][i + 1]:
            assert np.array_equal(r["pred"][i], r["prev"][i]) and np.array_equal(r["prev"][i + 1], r["pred"][i])


def test_run_sequence_noevents_passes_zero_counts_and_holds(synth):
    spy = _Spy(_Stand(scale=0.1).eval(), 0.5, 1.0, 0.5).eval()
    r = EX.run_sequence(spy, _cfg(), Path("/x"), "d", "s", torch.device("cpu"), np.random.default_rng(0), mode="noevents")
    assert set(spy.log["n"]) == {0}
    assert np.array_equal(r["pred"], r["prev"])                          # every packet held
    spy2 = _Spy(_Stand(scale=0.1).eval(), 0.5, 1.0, 0.5).eval()
    EX.run_sequence(spy2, _cfg(), Path("/x"), "d", "s", torch.device("cpu"), np.random.default_rng(0), mode="tf")
    assert all(isinstance(n, int) for n in spy2.log["n"]) and len(spy2.log["n"]) == len(spy.log["n"])


def test_default_models_run_unchanged_through_the_evalx_hooks(synth):
    """A bare model and a FilteredTracker have no hooks: called `model(x, prev)`, no state recorded, results as before."""
    class _Strict(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = _Stand(scale=0.1)

        def forward(self, x, prevpos):                                   # no betas / camera_K / n_events accepted
            return self.inner(x, prevpos)

    r_bare = EX.run_sequence(_Strict().eval(), _cfg(), Path("/x"), "d", "s", torch.device("cpu"), np.random.default_rng(0))
    assert r_bare["fstate"] == []
    ft = FilteredTracker(_Stand(scale=0.1), 0.5, 1.0, 0.5).eval()
    r_ft = EX.run_sequence(ft, _cfg(), Path("/x"), "d", "s", torch.device("cpu"), np.random.default_rng(0))
    assert r_ft["fstate"] == []
    # the same closed loop through AdaptiveFilter(host=True) is bitwise the FilteredTracker's
    af = AdaptiveFilter(_Stand(scale=0.1), 0.5, 1.0, 0.5, host=True).eval()
    r_af = EX.run_sequence(af, _cfg(), Path("/x"), "d", "s", torch.device("cpu"), np.random.default_rng(0))
    assert np.array_equal(r_af["pred"], r_ft["pred"]) and np.array_equal(r_af["prev"], r_ft["prev"])
    # the device path (n_events given by the evaluator) follows the FilteredTracker loop to float32 rounding
    af1 = AdaptiveFilter(_Stand(scale=0.1), 0.5, 1.0, 0.5).eval()
    r1 = EX.run_sequence(af1, _cfg(), Path("/x"), "d", "s", torch.device("cpu"), np.random.default_rng(0))
    assert np.abs(r1["pred"] - r_ft["pred"]).max() < 1e-5
    # the bare stand-in does not hold on empty packets (the filters do), so the two loops differ after the first one
    assert np.abs(r_bare["pred"] - r_ft["pred"]).max() > 1e-4


def test_perturb_trials_branch_from_the_state_snapshot_and_use_the_stored_counts(synth):
    spy = _Spy(_Stand(scale=0.1).eval(), NODES, 1.0, 0.5, beta_root=0.3, beta_trans=0.3).eval()
    dev = torch.device("cpu")
    r = EX.run_sequence(spy, _cfg(), Path("/x"), "d", "s", dev, np.random.default_rng(0))
    before = [{k: v.clone() for k, v in st.items()} for st in r["fstate"]]
    spy.log["n"].clear()
    out = EX.perturb_trials(spy, _cfg(), Path("/x"), "d", "s", dev, r, thetas=(10.0, 20.0), horizon=3, every=1, min_elapsed=100)
    n_trials = out[10.0]["div"].shape[0]
    assert n_trials == 4 and out[10.0]["div"].shape == (n_trials, 3)
    assert len(spy.log["set"]) == 2 * n_trials                           # one set_state per branch (per theta)
    for j in range(n_trials):                                            # the loop's own snapshot, the same for both thetas
        assert spy.log["set"][2 * j] is spy.log["set"][2 * j + 1]
        assert any(spy.log["set"][2 * j] is st for st in r["fstate"])
    assert len(spy.log["n"]) == 2 * n_trials * 3
    assert set(spy.log["n"]) <= {int(c) for c in r["count"]}             # the stored raw counts, as the closed loop used
    for a, b in zip(before, r["fstate"]):                                # branching never touches the stored snapshots
        assert torch.equal(a["v_root"], b["v_root"]) and torch.equal(a["v_trans"], b["v_trans"])
    # a stateful model without snapshots is an error, not a silent stale state
    with pytest.raises(ValueError):
        EX.perturb_trials(spy, _cfg(), Path("/x"), "d", "s", dev, {**r, "fstate": []}, horizon=3, every=1, min_elapsed=100)


class _Const(nn.Module):
    """An absolute tracker: ignores `prev`, always measures `z`."""

    def __init__(self, z):
        super().__init__()
        self.register_buffer("z", torch.as_tensor(z, dtype=torch.float32).view(1, -1))

    def forward(self, x, prevpos, betas=None, camera_K=None):
        return self.z.expand(prevpos.shape[0], -1).clone()


@pytest.mark.parametrize("a", [0.5, 0.3])
@pytest.mark.parametrize("kind", ["stateless", "adaptive", "adaptive_ab"])
def test_perturbation_retention_over_an_absolute_tracker(synth, kind, a):
    """The state sits on the measurement; a 10 deg root perturbation is pulled back geodesically by `a` each step, so the
    divergence from the unperturbed loop is theta (1 - a)^(k+1). With an alpha-beta velocity only the first step is that."""
    dev = torch.device("cpu")
    z = synth["pos"][0]
    trk = _Const(z)
    net = {"stateless": lambda: FilteredTracker(trk, a, 1.0, 0.5), "adaptive": lambda: AdaptiveFilter(trk, a, 1.0, 0.5),
           "adaptive_ab": lambda: AdaptiveFilter(trk, a, 1.0, 0.5, beta_root=0.2, beta_trans=0.2)}[kind]().eval()
    r = EX.run_sequence(net, _cfg(), Path("/x"), "d", "s", dev, np.random.default_rng(0))
    out = EX.perturb_trials(net, _cfg(), Path("/x"), "d", "s", dev, r, thetas=(10.0,), horizon=3, every=1, min_elapsed=100)
    ret = out[10.0]["div"].mean(0) / 10.0
    want = np.array([(1 - a), (1 - a) ** 2, (1 - a) ** 3])
    assert ret.shape == (3,)
    if kind != "adaptive_ab":
        assert np.allclose(ret, want, atol=3e-3), ret
    else:
        # the first step is the same (the snapshot's velocity is 0); then the velocity term reads the correction as a
        # motion and carries on in the same direction, so the perturbation dies faster than (1 - a)^k
        assert abs(ret[0] - want[0]) < 3e-3 and (ret[1:] < want[1:] - 0.01).all(), (ret, want)


# --------------------------------------------------------------------- filter_sweep_2fold: selection logic
def _row(kind, spec, ra, acc, n=(100, 80), runs=("r1",)):
    """A table row: `ra` / `acc` are {fold: value} (same for every run)."""
    return {"kind": kind, "spec": spec, "runs": {r: {"folds": {f: {"n": n[i], "ra": ra[f], "acc_err": acc[f]} for i, f in enumerate("AB")}}
                                                 for r in runs}}


def _table(runs=("r1",)):
    bare = {"A": 10.0, "B": 11.0}
    t = {"raw": _row("raw", None, bare, {"A": 6.0, "B": 6.0}, runs=runs),
         SW.PRIMARY_TAG: _row("host", {"a_root": 0.5}, {"A": 9.8, "B": 10.8}, {"A": 5.0, "B": 5.0}, runs=runs)}
    t["s_a"] = _row("spec", {"a_root": 0.4}, {"A": 9.0, "B": 10.9}, {"A": 5.5, "B": 5.5}, runs=runs)          # best on A
    t["s_b"] = _row("spec", {"a_root": 0.7}, {"A": 9.9, "B": 10.2}, {"A": 5.5, "B": 5.5}, runs=runs)          # best on B
    t["s_bad"] = _row("spec", {"a_root": 1.0}, {"A": 8.0, "B": 8.0}, {"A": 7.0, "B": 5.0}, runs=runs)         # jitterier than bare on A only
    return t


def test_selection_is_per_fold_with_the_acc_constraint_and_reported_out_of_fold():
    t = _table()
    assert SW.admissible(t, "s_bad", ["r1"], "B") and not SW.admissible(t, "s_bad", ["r1"], "A")
    assert SW.ranked(t, ["r1"], "A") == ["s_a", "s_b"]                   # s_bad is out on A; host / raw rows are not specs
    assert SW.ranked(t, ["r1"], "B") == ["s_bad", "s_b", "s_a"]
    sel = SW.select_two_fold(t, ["r1"])
    a, b = sel["selection"]["A"], sel["selection"]["B"]
    assert (a["tag"], a["selected_on"], a["reported_on"]) == ("s_a", "A", "B")
    assert (a["in_fold_ra"], a["out_of_fold_ra"]) == (9.0, 10.9)
    assert (b["tag"], b["selected_on"], b["reported_on"]) == ("s_bad", "B", "A")
    assert (b["in_fold_ra"], b["out_of_fold_ra"]) == (8.0, 8.0)
    # different winners: the one with the smaller out-of-fold RA is recommended
    assert sel["recommended"]["tag"] == "s_bad"
    cv = sel["cv"]
    assert cv["out_of_fold_ra"] == pytest.approx((80 * 10.9 + 100 * 8.0) / 180)
    assert cv["reference"]["raw"]["cv_matched"] == pytest.approx((80 * 11.0 + 100 * 10.0) / 180)
    assert cv["delta_vs"][SW.PRIMARY_TAG] == pytest.approx(cv["out_of_fold_ra"] - (80 * 10.8 + 100 * 9.8) / 180)
    # restricting the candidates changes the selection but never lets a fold's choice see the other fold
    sel2 = SW.select_two_fold(t, ["r1"], among=["s_a", "s_b"])
    assert sel2["selection"]["A"]["tag"] == "s_a" and sel2["selection"]["B"]["tag"] == "s_b"
    assert sel2["recommended"]["tag"] == "s_b"                           # out-of-fold: s_a on B 10.9, s_b on A 9.9
    # the same winner on both folds is recommended as such
    t["s_a"]["runs"]["r1"]["folds"]["B"]["ra"] = 7.0
    assert SW.select_two_fold(t, ["r1"], among=["s_a", "s_b"])["recommended"]["rule"] == "selected on both folds"


def test_selection_needs_every_seed_and_averages_over_them():
    t = _table(runs=("r1", "r2"))
    del t["s_b"]["runs"]["r2"]                                            # run on one seed only: not a candidate over two
    t["s_a"]["runs"]["r2"]["folds"]["A"]["ra"] = 9.4                      # seed mean on A: 9.2
    assert "s_b" not in SW.ranked(t, ["r1", "r2"], "A") and SW.ranked(t, ["r1", "r2"], "A") == ["s_a"]
    assert SW._fold_ra(t["s_a"], ["r1", "r2"], "A") == pytest.approx(9.2)
    assert SW._fold_ra(t["s_b"], ["r1", "r2"], "A") is None
    assert SW.select_two_fold(t, ["r1", "r2"])["selection"]["A"]["in_fold_ra"] == pytest.approx(9.2)
    del t["raw"]
    assert SW.ranked(t, ["r1"], "A") == []                                # no bare reference: nothing is admissible


def test_stage_specs_follow_the_fold_wise_selections(monkeypatch):
    a1 = SW.stage1a_specs()
    assert len(a1) == 13 and len(SW.const_specs()) == 5 and len(SW.event_specs()) == 8
    assert {s["a_trans"] for s in a1} == {0.5} and {s["a_rest"] for s in a1} == {1.0}
    assert len({SW._tag(s) for s in a1}) == 13                            # distinct specs, distinct tags
    ev = SW.event_specs()
    e_a, e_b = ev[0], ev[5]
    runs = ("r1",)
    t = _table()
    for tag in [k for k in t if t[k]["kind"] == "spec"]:
        del t[tag]
    t[SW._tag(e_a)] = _row("spec", e_a, {"A": 9.0, "B": 10.9}, {"A": 5.5, "B": 5.5})                  # best event spec on A
    t[SW._tag(e_b)] = _row("spec", e_b, {"A": 9.9, "B": 10.2}, {"A": 5.5, "B": 5.5})                  # ... on B
    c05 = SW.const_specs()[1]
    t[SW._tag(c05)] = _row("spec", c05, {"A": 8.0, "B": 9.0}, {"A": 5.0, "B": 5.0})                   # best overall, not an event spec
    monkeypatch.setattr(SW, "load_table", lambda in_dir, runs_: t)
    b = SW.stage_specs("1b", Path("."), seed0=runs)
    assert len(b) == 9 and {s["beta_root"] for s in b} == {0.1, 0.2, 0.3} and {s["beta_trans"] for s in b} == {0.1, 0.2, 0.3}
    bases = {SW._tag({k: v for k, v in s.items() if not k.startswith("beta")}) for s in b}
    assert bases == {SW._tag(c05), SW._tag(e_a), SW._tag(e_b)}
    c = SW.stage_specs("1c", Path("."), seed0=runs)                                   # best on both folds: the constant 0.5 spec
    assert len(c) == 1 and c[0]["a_rest"] == 0.8 and c[0]["a_root"] == 0.5 and c[0]["a_trans"] == 0.5
    s2 = SW.stage_specs("2", Path("."), seed0=runs)
    assert {SW._tag(s) for s in s2} == {SW._tag(c05), SW._tag(e_a), SW._tag(e_b)}
    with pytest.raises(ValueError):
        SW.stage_specs("9", Path("."))


def test_fold_sums_pool_and_metrics():
    z = {k: 0.0 for k in SW.SUM_KEYS}
    g = {**z, "n": 10, "ra_sum": 100.0, "n2": 8, "acc_err_sum": 40.0, "acc_pred_sum": 30.0, "acc_gt_sum": 20.0, "n1": 9,
         "spd_pred_sum": 9.0, "spd_gt_sum": 18.0, "k1_n": 2, "k1_sum": 0.5}
    loc = {**z, "n": 30, "ra_sum": 600.0, "n2": 24, "acc_err_sum": 240.0, "acc_pred_sum": 90.0, "acc_gt_sum": 60.0, "n1": 27,
           "spd_pred_sum": 27.0, "spd_gt_sum": 54.0}
    m = SW.fold_metrics(SW.pool({"global": g, "local": loc}))
    assert m["n"] == 40 and m["ra"] == pytest.approx(17.5) and m["acc_err"] == pytest.approx(280 / 32)
    assert m["acc_ratio"] == pytest.approx(1.5) and m["speed_ratio"] == pytest.approx(0.5) and m["k1"] == pytest.approx(0.25)
    assert SW.fold_metrics(SW.pool({"global": z, "local": z}))["ra"] is None
