#!/usr/bin/env python3
r"""Gates for the combined estimator in `semkine/mainline.py`.

The mainline's claim is that routing, filtering and anchoring are one estimator sharing one
information matrix, not three stages stapled together. The gates check the seams, since that is
where a combination of correct parts goes wrong:

* with everything disabled it is the identity, so any measured difference later belongs to a
  mechanism and not to plumbing
* an infinitely informative measurement reproduces the network's output, and an absent one
  reproduces pure propagation; the filter must interpolate between the two rather than doing
  something of its own
* routing by zeroed measurement information leaves the unrouted coordinate at its prior mean *and*
  loosens its covariance, which is the property that distinguishes it from a hard skip
* the anchor moves the state toward its prediction when it fires and not otherwise
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
from semkine import jacobian as JA                                  # noqa: E402
from semkine import kssf as KS                                      # noqa: E402
from semkine import lie                                             # noqa: E402
from semkine import mainline as ML                                  # noqa: E402
from semkine import oracle as OR                                    # noqa: E402
from semkine.dataset import _read_meta51                            # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOF = ML.DOF


@pytest.fixture(scope="module")
def mano():
    return ManoLayer(REPO / "assets/mano_right.npz", add_mean=False).to(DEV).eval().double()


@pytest.fixture(scope="module")
def ctx(mano):
    m = sorted(glob.glob(str(REPO / "data/hand_data51/*/*.meta")))[0]
    p = _read_meta51(m)
    aux = np.load(m[:-5] + "_aux.npz", allow_pickle=True)
    betas = torch.tensor(aux["betas"], dtype=torch.float64, device=DEV).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float64, device=DEV).view(1, 3, 3)
    pos = torch.from_numpy(p).to(DEV).double()
    with torch.no_grad():
        d = decode_to_mano_inputs(pos[:64], "mano_full_axis_angle",
                                  mano.hands_components, mano.hands_mean)
        v, _ = mano(betas.expand(64, -1), d["global_orient"], d["local_full_aa"], d["transl"])
    ok = torch.nonzero(v[..., 2].amin(-1) > 0.10).flatten()
    return dict(pos=pos, betas=betas, K=K, good=int(ok[len(ok) // 2]))


def events_from_motion(mano, ctx, delta, thresh=0.02):
    """Pixels a given tangent motion changes, as a stand-in for one packet."""
    field = KS.KSSF(mano, 180, 240, 0.375).to(DEV).double()
    p51 = ctx["pos"][ctx["good"]].view(1, DOF)
    j0 = (mano.J_regressor @ (mano.v_template
                              + torch.einsum("bl,mkl->bmk", ctx["betas"],
                                             mano.shapedirs)))[:, 0]

    def verts(pp):
        d = decode_to_mano_inputs(pp, "mano_full_axis_angle",
                                  mano.hands_components, mano.hands_mean)
        return mano(ctx["betas"], d["global_orient"], d["local_full_aa"], d["transl"])[0]

    with torch.no_grad():
        f0 = field.rasterize(verts(p51), ctx["K"])
        f1 = field.rasterize(verts(lie.retract_51d(p51, delta, mano.hands_mean, j0)), ctx["K"])
    moved = ((f0.inv_depth - f1.inv_depth).abs() + (f0.visibility - f1.visibility).abs()) > thresh
    _, py, px = torch.nonzero(moved, as_tuple=True)
    return np.stack([px.cpu().numpy(), py.cpu().numpy(),
                     np.zeros(len(px), dtype=np.int64)], -1).astype(np.int64)


def make(mano, ctx, **kw):
    p = ML.SemKinePolicy(mano=mano, step_ms=50, **kw)
    p.begin_sequence(ctx["betas"], ctx["K"])
    p.reset()
    return p


# ----------------------------------------------------------------------------- parity
def test_everything_off_is_the_identity(mano, ctx):
    p = make(mano, ctx, use_filter=False, use_router=False)
    prev = ctx["pos"][0:1]
    pred = prev + 0.05
    out = p(prev, pred, None, None, n_events=0, events=None)
    assert torch.equal(out, pred), "the disabled mainline is not bit-identical to the baseline"
    assert p.rate() == 1.0


def test_an_empty_packet_propagates_instead_of_trusting_the_network(mano, ctx):
    """With no events there is no measurement, so the state must stay where propagation put it.

    This is the principled version of `ZERO_EVENT_GATE`: the previous pose is returned because
    nothing was observed, not because a special case says so.
    """
    p = make(mano, ctx, use_router=False,
             filter_kwargs=dict(p0_pose=1.0, p0_vel=0.0, sigma_a=1e-6,
                                info_divisor=1.0, sigma_floor=1e-12, sigma_ceil=1e6))
    prev = ctx["pos"][0:1]
    pred = ctx["pos"][3:4]
    out = p(prev, pred, None, None, n_events=0, events=None)
    assert float((out - pred).abs().max()) > 1e-6, "the filter jumped to an unmeasured prediction"
    assert float((out - prev).abs().max()) < 1e-6, "propagation moved a zero-velocity state"


def test_filter_interpolates_between_prior_and_measurement(mano, ctx):
    r"""The posterior must lie between the prior and the measurement, and move with the weight."""
    prev = ctx["pos"][0:1]
    pred = ctx["pos"][6:7]
    got = []
    # `info_divisor` is a divisor, so the sequence runs from the loosest measurement to the
    # tightest and the posterior must travel further toward the network's output as it goes.
    for scale in (1e6, 1e4, 1e2):
        p = make(mano, ctx, use_router=False,
                 filter_kwargs=dict(p0_pose=0.05, p0_vel=1e-9, sigma_a=1e-9,
                                    info_divisor=scale, sigma_floor=1e-12, sigma_ceil=1e3))
        d = torch.zeros(1, DOF, device=DEV, dtype=torch.float64)
        d[:, 6:51] = 0.15
        ev = events_from_motion(mano, ctx, d)
        out = p(prev, pred, None, None, n_events=len(ev), events=ev)
        got.append(float((out - prev).abs().sum() / (pred - prev).abs().sum()))
    assert all(0.0 <= g <= 1.001 for g in got), got
    assert got[0] < got[-1], f"the posterior did not move toward the measurement: {got}"


# ----------------------------------------------------------------------------- routing seam
def test_masking_information_is_symmetric_and_kills_the_group(mano, ctx):
    p = make(mano, ctx)
    g = torch.Generator().manual_seed(0)
    A = torch.randn(DOF, DOF, generator=g, dtype=torch.float64).to(DEV)
    Lam = A @ A.T
    active = np.ones(OR.N_GROUPS, dtype=bool)
    active[3] = False
    M = p._mask_information(Lam, active)
    sl = OR.GROUPS[3][1]
    assert float(M[sl, :].abs().max()) == 0.0
    assert float(M[:, sl].abs().max()) == 0.0, "columns were left in place, so the gain can leak"
    assert float((M - M.T).abs().max()) == 0.0
    w = torch.linalg.eigvalsh(M)
    assert float(w.min()) > -1e-9, "masking broke positive semidefiniteness"
    keep = [i for i in range(DOF) if not (sl.start <= i < sl.stop)]
    t = torch.as_tensor(keep, device=DEV)
    assert float((M[t][:, t] - Lam[t][:, t]).abs().max()) == 0.0, "masking touched other groups"


def test_unrouted_coordinates_keep_their_mean_and_lose_confidence(mano, ctx):
    r"""The property that distinguishes routing-as-zero-information from a hard skip.

    A skipped group must (a) stay essentially where propagation predicted, and (b) end the step less
    certain than it began, since nothing observed it. A hard overwrite gives (a) but not (b), and
    (b) is what makes the next step's routing and the anchor trigger meaningful.

    "Essentially" rather than "exactly", and the difference is the mechanism rather than a
    tolerance: a masked group is handed the ceiling variance, not infinity, and the prior couples it
    to groups that *were* observed, so a correct filter nudges it by prior/ceiling -- measured here
    at 2.5e-7 rad, five orders below the routed group's motion. Requiring exact invariance would be
    requiring the filter to ignore its own prior correlations.
    """
    prev = ctx["pos"][ctx["good"]].view(1, DOF)
    pred = prev + 0.1
    p = make(mano, ctx, use_router=False,
             filter_kwargs=dict(p0_pose=0.05, p0_vel=1e-9, sigma_a=0.5,
                                info_divisor=1.0, sigma_floor=1e-12, sigma_ceil=1e3))
    p._started = False
    p._filter.start(prev, mano.hands_mean.to(torch.float64), p._j0)
    p._started = True
    g = torch.Generator().manual_seed(1)
    A = torch.randn(400, DOF, generator=g, dtype=torch.float64).to(DEV)
    Lam = A.T @ A * 1e4
    active = np.ones(OR.N_GROUPS, dtype=bool)
    active[5] = False
    sl = OR.GROUPS[5][1]
    before = float(torch.diagonal(p._filter.P[:DOF, :DOF])[sl].sum())
    p._filter.predict(0.05)
    p._filter.update(pred, p._mask_information(Lam, active), 0.05)
    after = float(torch.diagonal(p._filter.P[:DOF, :DOF])[sl].sum())
    moved = float((p._filter.x[:, sl] - prev[:, sl]).abs().max())
    other = float((p._filter.x[:, OR.GROUPS[6][1]] - prev[:, OR.GROUPS[6][1]]).abs().max())
    assert moved < 1e-4 * other, (
        f"the skipped group moved {moved:.3e}, not negligible against the routed {other:.3e}")
    assert after > before, f"the skipped group's covariance did not grow: {before} -> {after}"
    assert other > 1e-6, "the routed group did not move either, so nothing was tested"


def test_router_reduces_the_update_rate_on_a_real_packet(mano, ctx):
    d = torch.zeros(1, DOF, device=DEV, dtype=torch.float64)
    d[:, 6:15] = 0.2                              # thumb and index only
    ev = events_from_motion(mano, ctx, d)
    if len(ev) < 200:
        pytest.skip(f"only {len(ev)} changed pixels in this pose")
    prev = ctx["pos"][ctx["good"]].view(1, DOF)
    p = make(mano, ctx, router_kwargs=dict(budget=4, gain_on=1.0, gain_off=0.25, min_dwell=0))
    p(prev, prev + 0.05, None, None, n_events=len(ev), events=ev)
    assert p.rate() < 1.0, "the router activated everything"
    assert p.rate() > 0.0


# ----------------------------------------------------------------------------- anchor seam
def test_anchor_pulls_a_drifted_state_back(mano, ctx):
    """With the trigger forced on and a truthful anchor, the state must move toward the truth."""
    truth = ctx["pos"][ctx["good"]].view(1, DOF)
    drift = torch.zeros(1, DOF, device=DEV, dtype=torch.float64)
    drift[:, 3:6] = 0.25
    j0 = (mano.J_regressor @ (mano.v_template
                              + torch.einsum("bl,mkl->bmk", ctx["betas"],
                                             mano.shapedirs)))[:, 0]
    start = lie.retract_51d(truth, drift, mano.hands_mean, j0)
    p = make(mano, ctx, use_router=False,
             anchor_model=lambda _win: truth,
             trigger_kwargs=dict(persistence=1, cooldown=0, nis_ratio_on=0.0,
                                 trace_on=1e9, outside_on=1e9),
             anchor_sigma=1e-6,
             filter_kwargs=dict(p0_pose=0.3, p0_vel=1e-9, sigma_a=1e-6,
                                info_divisor=1e6, sigma_floor=1e-9, sigma_ceil=1e3))
    d = torch.zeros(1, DOF, device=DEV, dtype=torch.float64)
    d[:, 6:51] = 0.1
    ev = events_from_motion(mano, ctx, d)
    out = p(start, start.clone(), None, None, n_events=len(ev), events=ev)
    assert p.n_anchored == 1, p.stats()
    before = float((start - truth).abs().max())
    after = float((out - truth).abs().max())
    assert after < 0.2 * before, f"anchoring moved {before:.4f} to only {after:.4f}"


def test_anchor_does_not_fire_on_a_healthy_step(mano, ctx):
    calls = []
    truth = ctx["pos"][ctx["good"]].view(1, DOF)
    p = make(mano, ctx, use_router=False,
             anchor_model=lambda _win: calls.append(1) or truth,
             trigger_kwargs=dict(persistence=3, cooldown=50, nis_ratio_on=50.0,
                                 trace_on=1e9, outside_on=1e9))
    d = torch.zeros(1, DOF, device=DEV, dtype=torch.float64)
    d[:, 6:51] = 0.05
    ev = events_from_motion(mano, ctx, d)
    for _ in range(5):
        p(truth, truth.clone(), None, None, n_events=len(ev), events=ev)
    assert not calls, "the anchor fired on a well-tracked step"
    assert p.n_anchored == 0


def test_stats_report_every_component(mano, ctx):
    p = make(mano, ctx)
    d = torch.zeros(1, DOF, device=DEV, dtype=torch.float64)
    d[:, 6:51] = 0.1
    ev = events_from_motion(mano, ctx, d)
    prev = ctx["pos"][ctx["good"]].view(1, DOF)
    for _ in range(3):
        prev = p(prev, prev + 0.01, None, None, n_events=len(ev), events=ev)
    st = p.stats()
    for k in ("update_rate", "n_steps", "n_anchored", "filter_n_measured"):
        assert k in st, (k, st)
    assert st["n_steps"] == 3
