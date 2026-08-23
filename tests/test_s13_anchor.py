#!/usr/bin/env python3
r"""S13 gates for the triggered absolute anchor.

Two things must hold before an anchor is worth running in a tracking loop.

**The fusion is a real manifold estimator.** It has to reduce to each endpoint in the appropriate
limit, be independent of which estimate is called first, be a fixed point when the two agree, and
produce a covariance tighter than either input. And it has to be measurably better than the
Euclidean average of the 51-dimensional vectors -- otherwise the manifold machinery is decoration.

**The trigger fires on drift and not on noise.** Checked by injecting a known drift into a state
and requiring the trigger to fire, then holding a clean state and requiring it not to. Plus the
cooldown and persistence behaviour, since without them the trigger degenerates into the constant
blend that history already found neutral.
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
from semkine import anchor as AN                                    # noqa: E402
from semkine import filter as FT                                    # noqa: E402
from semkine import kssf as KS                                      # noqa: E402
from semkine import lie                                             # noqa: E402
from semkine.dataset import _read_meta51                            # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DOF = AN.DOF


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
    j0 = (mano.J_regressor @ (mano.v_template
                              + torch.einsum("bl,mkl->bmk", betas, mano.shapedirs)))[:, 0]
    pos = torch.from_numpy(p).to(DEV).double()
    with torch.no_grad():
        d = decode_to_mano_inputs(pos[:64], "mano_full_axis_angle",
                                  mano.hands_components, mano.hands_mean)
        v, _ = mano(betas.expand(64, -1), d["global_orient"], d["local_full_aa"], d["transl"])
    ok = torch.nonzero(v[..., 2].amin(-1) > 0.10).flatten()
    return dict(pos=pos, betas=betas, K=K, j0=j0, hands_mean=mano.hands_mean.double(),
                good=int(ok[len(ok) // 2]))


def _spd(rng, scale, n=DOF):
    A = torch.tensor(rng.normal(size=(n, n)), dtype=torch.float64, device=DEV)
    return (A @ A.T) / n * scale + scale * 0.1 * torch.eye(n, device=DEV, dtype=torch.float64)


def _mesh(mano, p51, betas):
    d = decode_to_mano_inputs(p51, "mano_full_axis_angle", mano.hands_components, mano.hands_mean)
    v, _ = mano(betas, d["global_orient"], d["local_full_aa"], d["transl"])
    return v


# ----------------------------------------------------------------------- fusion limits
def test_certain_anchor_wins(ctx):
    """As the anchor's covariance goes to zero the fused state must become the anchor."""
    hm, j0 = ctx["hands_mean"], ctx["j0"]
    xa, xb = ctx["pos"][0:1], ctx["pos"][30:31]
    P = torch.eye(DOF, device=DEV, dtype=torch.float64) * 1e-2
    for s in (1e-6, 1e-9, 1e-12):
        S = torch.eye(DOF, device=DEV, dtype=torch.float64) * s
        xf, _ = AN.fuse(xa, P, xb, S, hm, j0, iters=5)
        err = float(lie.local_coordinates_51d(xf, xb, hm, j0).norm())
        assert err < 1e-3 * np.sqrt(s / 1e-6) + 1e-8, f"s={s}: fused is {err:.3e} from the anchor"


def test_certain_tracker_wins(ctx):
    hm, j0 = ctx["hands_mean"], ctx["j0"]
    xa, xb = ctx["pos"][0:1], ctx["pos"][30:31]
    P = torch.eye(DOF, device=DEV, dtype=torch.float64) * 1e-12
    S = torch.eye(DOF, device=DEV, dtype=torch.float64) * 1e-2
    xf, _ = AN.fuse(xa, P, xb, S, hm, j0, iters=5)
    assert float(lie.local_coordinates_51d(xf, xa, hm, j0).norm()) < 1e-6


def test_agreeing_estimates_are_a_fixed_point(ctx):
    rng = np.random.default_rng(0)
    hm, j0 = ctx["hands_mean"], ctx["j0"]
    x = ctx["pos"][10:11]
    xf, Pf = AN.fuse(x, _spd(rng, 1e-2), x, _spd(rng, 1e-2), hm, j0)
    assert float((xf - x).abs().max()) < 1e-10
    assert torch.isfinite(Pf).all()


def test_fusion_is_symmetric_in_its_arguments(ctx):
    """Swapping which estimate is called the tracker must not move the answer.

    A real estimator has no preferred argument. This catches the common bug of linearising at one
    input and never transporting the other's information to the iterate, which yields an answer
    biased toward whichever was passed first.
    """
    rng = np.random.default_rng(1)
    hm, j0 = ctx["hands_mean"], ctx["j0"]
    xa, xb = ctx["pos"][0:1], ctx["pos"][40:41]
    P, S = _spd(rng, 4e-3), _spd(rng, 9e-3)
    x1, P1 = AN.fuse(xa, P, xb, S, hm, j0, iters=8)
    x2, P2 = AN.fuse(xb, S, xa, P, hm, j0, iters=8)
    d = float(lie.local_coordinates_51d(x1, x2, hm, j0).norm())
    assert d < 1e-8, f"fusion depends on argument order by {d:.3e}"
    assert float((P1 - P2).abs().max()) < 1e-8


def test_fused_covariance_is_tighter_than_either(ctx):
    """Information adds: the fused covariance must be below both inputs in the Loewner order."""
    rng = np.random.default_rng(2)
    hm, j0 = ctx["hands_mean"], ctx["j0"]
    xa, xb = ctx["pos"][0:1], ctx["pos"][5:6]
    P, S = _spd(rng, 5e-3), _spd(rng, 5e-3)
    _, Pf = AN.fuse(xa, P, xb, S, hm, j0)
    for M, name in ((P, "tracker"), (S, "anchor")):
        w = torch.linalg.eigvalsh(M - Pf)
        assert float(w.min()) > -1e-9, f"fused is looser than the {name} by {float(w.min()):.3e}"


def test_fusion_beats_the_euclidean_average_on_the_mesh(ctx, mano):
    r"""The manifold machinery has to earn itself, measured where it matters: on the hand.

    Two estimates straddle a known truth with equal covariances, so the correct fusion is the
    manifold midpoint. Averaging the 51-dimensional vectors instead gives a different pose, and the
    gap is reported in millimetres of vertex error rather than in radians, because that is the unit
    the claim is made in.
    """
    hm, j0 = ctx["hands_mean"], ctx["j0"]
    truth = ctx["pos"][ctx["good"]].view(1, DOF)
    # Drift concentrated in the wrist and one finger, which is what real drift looks like and also
    # where coordinate averaging is worst: the error of averaging two rotations in axis-angle grows
    # with the angle between them, so a disagreement spread thinly over 51 coordinates would let
    # the Euclidean shortcut off the hook.
    d = torch.zeros(DOF, device=DEV, dtype=torch.float64)
    d[0:3] = torch.tensor([0.02, -0.01, 0.03], device=DEV, dtype=torch.float64)
    d[3:6] = torch.tensor([0.55, -0.40, 0.30], device=DEV, dtype=torch.float64)
    d[6:9] = torch.tensor([0.50, 0.30, -0.35], device=DEV, dtype=torch.float64)
    xa = AN.retract_metric(truth, d, hm, j0)
    xb = AN.retract_metric(truth, -d, hm, j0)
    I = torch.eye(DOF, device=DEV, dtype=torch.float64) * 1e-2
    xf, _ = AN.fuse(xa, I, xb, I.clone(), hm, j0, iters=10)
    xe = 0.5 * (xa + xb)
    vt = _mesh(mano, truth, ctx["betas"])
    mm = lambda x: float((_mesh(mano, x, ctx["betas"]) - vt).norm(dim=-1).mean()) * 1000  # noqa
    e_f, e_e = mm(xf), mm(xe)
    assert e_f < 1e-6, f"manifold fusion missed the midpoint by {e_f:.4f} mm"
    assert e_e > 1.0, (
        f"the euclidean average is only {e_e:.4f} mm off, so this case does not discriminate")


def test_se3_metric_biases_the_wrist_and_the_decoupled_one_does_not(ctx, mano):
    r"""The root-metric choice, measured rather than asserted.

    Both metrics are correct means -- of different things. The comparison has to be fair, so each
    metric is handed two estimates generated by *its own* retraction, symmetrically about the truth,
    and asked for their mean. The decoupled metric returns the truth exactly. The full `SE(3)`
    metric does not: its `Log` map couples wrist rotation into translation, so the mean is displaced
    even though the inputs were placed symmetrically under that same metric. Since the method's
    error measure is millimetres of vertex position, the bias is reported in millimetres.
    """
    hm, j0 = ctx["hands_mean"], ctx["j0"]
    truth = ctx["pos"][ctx["good"]].view(1, DOF)
    g = torch.Generator(device="cpu").manual_seed(3)
    d = torch.randn(DOF, generator=g, dtype=torch.float64).to(DEV)
    d = d / d.norm() * 0.6
    I = torch.eye(DOF, device=DEV, dtype=torch.float64) * 1e-2
    vt = _mesh(mano, truth, ctx["betas"])
    mm = lambda x: float((_mesh(mano, x, ctx["betas"]) - vt).norm(dim=-1).mean()) * 1000  # noqa
    got = {}
    for metric in ("decoupled", "se3"):
        xa = AN.retract_metric(truth, d, hm, j0, metric)
        xb = AN.retract_metric(truth, -d, hm, j0, metric)
        xf, _ = AN.fuse(xa, I, xb, I.clone(), hm, j0, iters=20, root_metric=metric)
        got[metric] = (xf, mm(xf))
    assert got["decoupled"][1] < 1e-6, f"decoupled fusion is off by {got['decoupled'][1]:.4f} mm"
    assert got["se3"][1] > 1.0, (
        f"the SE(3) bias is only {got['se3'][1]:.4f} mm, so the default needs no justification")
    # The bias is confined to the wrist: the joints are on `SO(3)` under either metric.
    joint_gap = float((got["se3"][0][:, 6:] - got["decoupled"][0][:, 6:]).abs().max())
    assert joint_gap < 1e-6, f"the metrics disagree on the joints by {joint_gap:.3e}"


def test_fusion_recovers_an_injected_drift(ctx, mano):
    r"""The end-to-end claim: a drifted tracker plus a decent anchor lands near the truth.

    Drift is injected as a known tangent offset, the anchor is the truth plus a smaller random
    error, and the fused state must be closer to the truth than the drifted one -- by roughly the
    ratio the two covariances imply, not merely by some amount.
    """
    hm, j0 = ctx["hands_mean"], ctx["j0"]
    truth = ctx["pos"][ctx["good"]].view(1, DOF)
    g = torch.Generator(device="cpu").manual_seed(4)
    drift = torch.randn(DOF, generator=g, dtype=torch.float64).to(DEV)
    drift = drift / drift.norm() * 0.35
    noise = torch.randn(DOF, generator=g, dtype=torch.float64).to(DEV)
    noise = noise / noise.norm() * 0.05
    x_trk = lie.retract_51d(truth, drift.view(1, DOF), hm, j0)
    x_abs = lie.retract_51d(truth, noise.view(1, DOF), hm, j0)
    P = torch.eye(DOF, device=DEV, dtype=torch.float64) * 0.35 ** 2 / DOF
    S = torch.eye(DOF, device=DEV, dtype=torch.float64) * 0.05 ** 2 / DOF
    xf, _ = AN.fuse(x_trk, P, x_abs, S, hm, j0, iters=10)
    vt = _mesh(mano, truth, ctx["betas"])
    mm = lambda x: float((_mesh(mano, x, ctx["betas"]) - vt).norm(dim=-1).mean()) * 1000  # noqa
    before, after = mm(x_trk), mm(xf)
    assert after < 0.3 * before, f"drift {before:.2f} mm only came down to {after:.2f} mm"


# ----------------------------------------------------------------------- outside-silhouette
def test_outside_fraction_separates_a_drifted_state(ctx, mano):
    r"""The geometric trigger must respond to a misplaced hypothesis, not to the filter's opinion.

    Events are synthesised on the true silhouette; the fraction of them falling outside the
    *hypothesised* silhouette is then measured for the true state and for a translated one. The gap
    is what the trigger thresholds on, and it exists without any reference to a covariance.
    """
    truth = ctx["pos"][ctx["good"]].view(1, DOF)
    field = KS.KSSF(mano, 180, 240, 0.375).to(DEV).double()
    with torch.no_grad():
        f_true = field.rasterize(_mesh(mano, truth, ctx["betas"]), ctx["K"])
    band = (f_true.sdf.abs() < 2.0) & (f_true.visibility > 0)
    _, py, px = torch.nonzero(band, as_tuple=True)
    assert len(px) > 200, f"only {len(px)} contour pixels"
    xy = torch.stack([px, py], -1).double()
    b = torch.zeros(len(px), dtype=torch.long, device=DEV)

    frac = {}
    for shift in (0.0, 0.01, 0.02, 0.04, 0.08):
        s = truth.clone()
        s[:, 0] += shift
        with torch.no_grad():
            fb = field.rasterize(_mesh(mano, s, ctx["betas"]), ctx["K"])
        frac[shift] = AN.outside_silhouette_fraction(fb, xy, b)
    assert frac[0.0] < 0.05, f"the true state already looks displaced: {frac[0.0]:.3f}"
    assert all(frac[a] <= frac[b] + 1e-9 for a, b in zip((0.0, 0.01, 0.02, 0.04),
                                                         (0.01, 0.02, 0.04, 0.08))), frac
    # The threshold has to sit where the signal actually separates. Recorded here so the default
    # in `AnchorTrigger` is traceable to a measurement: 1 cm is invisible at a 4 px margin, and
    # the signal only becomes usable past about 2 cm of displacement.
    t = AN.AnchorTrigger()
    assert frac[0.01] < t.outside_off, f"1 cm already disarms nothing: {frac[0.01]:.3f}"
    assert frac[0.04] > t.outside_on, f"4 cm does not cross the threshold: {frac[0.04]:.3f}"


# ----------------------------------------------------------------------- trigger logic
def test_trigger_fires_on_sustained_drift_and_not_on_a_single_spike():
    t = AN.AnchorTrigger(persistence=2, cooldown=5, nis_dim=51)
    assert not t(nis=300.0)                      # armed, not yet fired
    assert not t(nis=10.0)                       # one clean step disarms
    assert not t(nis=300.0)
    assert t(nis=300.0), "two consecutive armed steps should fire"
    assert t.n_fired == 1


def test_cooldown_suppresses_immediate_refiring():
    t = AN.AnchorTrigger(persistence=1, cooldown=4, nis_dim=51)
    fired = [t(nis=1000.0) for _ in range(10)]
    assert fired[0] is True
    assert not any(fired[1:5]), fired
    assert fired[5] is True, fired
    assert t.n_fired == 2


def test_each_signal_can_fire_alone():
    for kw in ({"nis": 1000.0}, {"trace": 10.0}, {"outside": 0.9}):
        t = AN.AnchorTrigger(persistence=1, cooldown=100, nis_dim=51)
        assert t(**kw), kw
        assert t.reasons[0] == list(kw)[0]


def test_quiet_tracking_never_fires():
    rng = np.random.default_rng(9)
    t = AN.AnchorTrigger(persistence=2, cooldown=20, nis_dim=51)
    for _ in range(500):
        # NIS fluctuating around its expectation, a small covariance, few outliers: a healthy run.
        assert not t(nis=float(rng.chisquare(51)), trace=0.01,
                     outside=float(rng.uniform(0.0, 0.2)))
    assert t.n_fired == 0


def test_trigger_consumes_the_filter_statistics(ctx):
    """The three signals must be exactly what `LieFilter` already reports, not a parallel path."""
    f = FT.LieFilter()
    f.start(ctx["pos"][0:1], ctx["hands_mean"], ctx["j0"])
    for k in range(20):
        f.predict(0.05)
        f.update(ctx["pos"][k].view(1, DOF),
                 torch.eye(DOF, device=DEV, dtype=torch.float64) * 1e4)
    st = f.stats()
    assert "nis_mean" in st and f.trace
    t = AN.AnchorTrigger(nis_dim=st["nis_dim"], persistence=1, trace_on=1e9, outside_on=1e9)
    fired = t(nis=f.nis[-1], trace=f.trace[-1])
    assert isinstance(fired, bool)
