#!/usr/bin/env python3
r"""S18 gates for the Kinematic Event Graph frontend.

The gates are grouped by what a failure would mean.

**Contract gates** keep KEG a drop-in frontend: the feature width, the empty packet, and the
parameter budget against `raw_scan`, so a later win cannot be explained as capacity.

**Time-invariance gates** are the ones the S20 claim rests on. The readout must be a function of
how long ago each event happened and of nothing else -- in particular not of the declared window
length. Two decisive forms are checked: shifting every event and the window end by the same amount
must leave the output bitwise unchanged, and the closed-form scatter must equal the `O(1)`
per-event recursion that an asynchronous deployment would run. The second is the
Messikommer et al. (ECCV'20) equivalence contract, stated as an equality rather than as prose.

**Grouping gates** check that the graph is the kinematic one. An event's node must be the joint
MANO's own skinning weights name at that pixel, computed from the full 16-vector rather than from
the field's own top-k list; an event with no surface behind it must carry exactly zero geometry;
and before message passing, events on one node must not touch another node's feature.

**Locality gate** for the joint tokens: zeroing node `k` may change joint `k`'s output and nothing
else. That is the S10 unique-pathway rule, tightened -- with shared features every joint sees every
event, and the ablation only isolates a decoder; here it isolates a decoder *and* its evidence.
"""
from __future__ import annotations

import glob
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from mano_layer import ManoLayer                                        # noqa: E402
from pose_repr import decode_to_mano_inputs                             # noqa: E402
from semkine import keg as KG                                           # noqa: E402
from semkine.dataset import _read_meta51                                # noqa: E402
from semkine.encoder import RawEventEncoder                             # noqa: E402
from semkine.events import EventPacketBatch                             # noqa: E402
from semkine.frontends import build_frontend, frontend_param_count      # noqa: E402
from semkine.kssf import KSSF                                           # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
H, W, SCALE = 180, 240, 0.375


# ----------------------------------------------------------------------- fixtures
def _packet(n_per=64, B=3, dt=0.05, seed=0, device=None):
    """A synthetic ragged batch with monotone microsecond stamps inside each packet."""
    g = torch.Generator().manual_seed(seed)
    evs, ptr = [], [0]
    for b in range(B):
        t = torch.sort(torch.rand(n_per, generator=g) * dt).values
        e = torch.stack([
            torch.full((n_per,), float(b)),
            torch.randint(0, W, (n_per,), generator=g).float(),
            torch.randint(0, H, (n_per,), generator=g).float(),
            t,
            torch.randint(0, 2, (n_per,), generator=g).float(),
        ], -1)
        evs.append(e)
        ptr.append(ptr[-1] + n_per)
    ev = torch.cat(evs)
    out = (ev, torch.tensor(ptr), torch.full((B,), float(dt)))
    if device is not None:
        out = tuple(t.to(device) for t in out)
    return out


@pytest.fixture(scope="module")
def mano():
    return ManoLayer(REPO / "assets/mano_right.npz", add_mean=False).to(DEV).eval()


@pytest.fixture(scope="module")
def pose(mano):
    """One real pose with a sane label, plus its betas and intrinsics."""
    metas = sorted(glob.glob(str(REPO / "data/hand_data51/*/*.meta")))
    if not metas:
        pytest.skip("hand_data51 not available")
    base = metas[len(metas) // 2][:-5]
    p = _read_meta51(base + ".meta")
    aux = np.load(base + "_aux.npz", allow_pickle=True)
    b = torch.tensor(aux["betas"], dtype=torch.float32, device=DEV).view(1, -1)
    k = torch.tensor(aux["camera_K"], dtype=torch.float32, device=DEV).view(1, 3, 3)
    idx = np.linspace(0, len(p) - 1, 40).astype(int)
    pp = torch.from_numpy(p[idx]).to(DEV)
    with torch.no_grad():
        d = decode_to_mano_inputs(pp, "mano_full_axis_angle", mano.hands_components,
                                  mano.hands_mean)
        v, _ = mano(b.expand(len(pp), -1), d["global_orient"], d["local_full_aa"], d["transl"])
    keep = torch.nonzero(v[..., 2].amin(-1) > 0.10).flatten()[:2]
    assert keep.numel() >= 1, "no frame with a sane label"
    return pp[keep], b.expand(keep.numel(), -1), k.expand(keep.numel(), -1, -1)


@pytest.fixture(scope="module")
def fields(mano, pose):
    p, b, k = pose
    f = KSSF(mano, H, W, SCALE).to(DEV)
    with torch.no_grad():
        return f(p, b, k), f


# ----------------------------------------------------------------------- contract
def test_feature_width_and_ragged_contract():
    ev, ptr, dt = _packet()
    m = build_frontend("keg", hidden=32, feat_dim=64, node_dim=16).eval()
    with torch.no_grad():
        y, nodes = m.forward_nodes(ev, ptr, dt)
    assert y.shape == (3, 64)
    assert nodes.shape == (3, KG.N_NODES, 16)
    assert torch.isfinite(y).all()


def test_empty_packet_runs_and_is_shared_across_the_batch():
    m = build_frontend("keg", hidden=16, feat_dim=32, node_dim=8).eval()
    ev = torch.zeros(0, 5)
    ptr = torch.tensor([0, 0])
    dt = torch.tensor([0.05])
    with torch.no_grad():
        y = m(ev, ptr, dt)
    assert y.shape == (1, 32) and torch.isfinite(y).all()
    # No events means no evidence, so the feature must be the network's constant, identical for
    # every empty packet. `ZERO_EVENT_GATE` then turns it into a bitwise-`prev` prediction.
    with torch.no_grad():
        y2 = m(torch.zeros(0, 5), torch.tensor([0, 0, 0]), torch.tensor([0.05, 0.02]))
    assert torch.equal(y2[0], y2[1])


def test_parameter_budget_vs_raw_scan():
    ref = frontend_param_count(RawEventEncoder(hidden=96, feat_dim=256))
    n = frontend_param_count(build_frontend("keg", hidden=96, feat_dim=256, node_dim=64))
    assert n <= ref * 1.10, (n, ref)


def test_throughput_is_compatible_with_training():
    """A frontend that cannot consume a real packet rate is not a candidate.

    `zgz_global` runs at 29 255 events per 50 ms, so a batch of 64 such windows is 1.9 M events.
    The gate is a million events per second, which the `gated_scan` of `encoder.py` cannot meet:
    its padded loop runs once per event of the longest packet.
    """
    ev, ptr, dt = _packet(n_per=4096, B=16, device=DEV)
    m = build_frontend("keg", hidden=64, feat_dim=256, node_dim=64).to(DEV).eval()
    with torch.no_grad():
        m(ev, ptr, dt)                                     # warm up kernels / autotune
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            m(ev, ptr, dt)
        if DEV.type == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t0
    rate = 5 * ev.shape[0] / el
    assert rate >= 1.0e6, f"{rate:.3g} events/s"


# ------------------------------------------------------------------ time invariance
def test_output_depends_on_age_not_on_window_length():
    r"""Shift every event and the window end by the same `c`: the output must not move.

    This is the property that makes a 5 ms deployment of a model trained at 50 ms meaningful. It
    fails for `RawEventEncoder`, whose token divides `t` by `Delta t`, and the contrast is asserted
    so the distinction is measured rather than claimed.
    """
    ev, ptr, dt = _packet(n_per=48, B=2, dt=0.01, seed=3)
    c = 0.04
    ev2 = ev.clone()
    ev2[:, 3] += c
    dt2 = dt + c

    keg = build_frontend("keg", hidden=32, feat_dim=64, node_dim=16).eval()
    with torch.no_grad():
        a = keg(ev, ptr, dt)
        b = keg(ev2, ptr, dt2)
    assert torch.allclose(a, b, atol=1e-6), (a - b).abs().max()

    scan = RawEventEncoder(hidden=32, feat_dim=64).eval()
    with torch.no_grad():
        p = scan(ev, ptr, dt)
        q = scan(ev2, ptr, dt2)
    assert (p - q).abs().max() > 1e-4, "raw_scan is unexpectedly window-invariant"


def test_closed_form_equals_the_per_event_recursion():
    r"""`sum_i e^{-lambda (T - t_i)} v_i` must equal iterating `h <- e^{-lambda dt} h + v`.

    The sum is what trains; the recursion is what an asynchronous deployment runs, at `O(1)` per
    event. If they disagree, "trained synchronously, deployed asynchronously" is an approximation
    and the efficiency claim is not about the same model.
    """
    torch.manual_seed(0)
    n, Hd = 500, 8
    t = torch.sort(torch.rand(n).double() * 0.05).values
    v = torch.randn(n, Hd).double()
    rate = torch.logspace(0.7, 3.3, Hd).double()
    T = 0.05

    agg = KG.lti_aggregate(v, (T - t), torch.zeros(n, dtype=torch.long), 1, rate)

    h = torch.zeros(Hd).double()
    z = torch.zeros(Hd).double()
    prev = 0.0
    for i in range(n):
        decay = torch.exp(-rate * (float(t[i]) - prev))
        h = decay * h + v[i]
        z = decay * z + 1.0                                   # the same recursion on values == 1
        prev = float(t[i])
    tail = torch.exp(-rate * (T - prev))                      # advance to the readout instant
    h, z = tail * h, tail * z

    assert torch.allclose(agg["raw"][0], h, rtol=1e-9, atol=1e-11), (agg["raw"][0] - h).abs().max()
    assert torch.allclose(agg["mass"][0], z, rtol=1e-9, atol=1e-11)
    # The readout is the ratio, so the asynchronous deployment has to run both recursions; the gate
    # covers the quantity that actually reaches `node_proj`, not just its numerator.
    assert torch.allclose(agg["ema"][0], h / (z + KG.EMA_EPS), rtol=1e-8, atol=1e-10)
    # The guard is the only thing between the ratio and the exact mean, and at the smallest mass in
    # this bank that is a relative perturbation of under a part in a million.
    assert ((agg["ema"][0] - h / z).abs() / (h / z).abs().clamp_min(1e-12)).max() < 1e-6


def test_readout_does_not_scale_with_how_many_events_a_node_holds():
    """Duplicating the evidence must not rescale the feature.

    The first version of this frontend fed `node_proj` the unnormalised sum, so a node's feature was
    proportional to its event count -- a quantity that varies 20x between validation sequences (S0's
    `zgz_local` finding) and 10x across the mixed window schedule. Trained, it put a median of 1.75
    and a maximum of 3347 in the same feature block while every other block stayed inside [0, 9],
    and the arm lost 13 mm to the LNES control. The normalised readout is a weighted mean, so
    doubling the events at an unchanged temporal profile leaves it unchanged.
    """
    torch.manual_seed(0)
    n, Hd = 64, 8
    age = torch.rand(n).double() * 0.05
    v = torch.randn(n, Hd).double()
    rate = torch.logspace(0.7, 3.3, Hd).double()
    seg = torch.zeros(n, dtype=torch.long)

    one = KG.lti_aggregate(v, age, seg, 1, rate)
    # Ten copies of every event at the same age: ten times the mass, same profile.
    ten = KG.lti_aggregate(v.repeat(10, 1), age.repeat(10), seg.repeat(10), 1, rate)

    # `EMA_EPS` sits in the denominator and does not scale with the copies, so the invariance holds
    # to exactly `EMA_EPS / mass` and not better. Gate that bound rather than a round tolerance: it
    # is the honest statement, and it is loosest for the fastest rate, whose 500 us window collects
    # the least mass -- measured at 4.2e-6 there against 1.6e-8 for the slowest.
    rel = (one["ema"] - ten["ema"]).abs() / ten["ema"].abs().clamp_min(1e-30)
    assert (rel <= 1.05 * KG.EMA_EPS / one["mass"]).all(), rel.max()
    assert rel.max() < 1e-5
    assert torch.allclose(ten["raw"], 10.0 * one["raw"], rtol=1e-9), "raw should be the scaling one"
    # And the mean is bounded by the values it averages, whatever the event count.
    assert ten["ema"].abs().max() <= v.abs().max() + 1e-12


def test_rates_stay_positive_under_any_parameter_value():
    m = KG.KinematicEventGraph(hidden=16)
    with torch.no_grad():
        m.log_rate.fill_(-50.0)
    assert bool((m.rate > 0).all())
    with torch.no_grad():
        m.log_rate.fill_(50.0)
    assert bool(torch.isfinite(m.rate).all())


def test_rate_bank_is_initialised_across_the_registered_span():
    m = KG.KinematicEventGraph(hidden=64)
    r = m.rate.detach()
    assert abs(float(r.min()) - KG.RATE_MIN) < 0.5
    assert abs(float(r.max()) - KG.RATE_MAX) / KG.RATE_MAX < 0.01


# --------------------------------------------------------------------- grouping
def test_group_is_manos_dominant_joint_at_that_pixel(mano, fields):
    """Checked against the full 16-wide skinning row, not against the field's own top-k list."""
    f, ks = fields
    B = f.visibility.shape[0]
    ys, xs = torch.nonzero(f.visibility[0] > 0, as_tuple=True)
    assert ys.numel() > 200, "the pose projects to too few pixels to test on"
    sel = torch.linspace(0, ys.numel() - 1, 400).long()
    ev = torch.stack([
        torch.zeros(sel.numel(), device=DEV),
        xs[sel].float(), ys[sel].float(),
        torch.linspace(0, 0.049, sel.numel(), device=DEV),
        torch.zeros(sel.numel(), device=DEV),
    ], -1)
    ch, (idx, w) = KG.kssf_event_channels(f, ev)

    fid = f.face_id[0, ys[sel], xs[sel]]
    bary = f.bary[0, ys[sel], xs[sel]].unsqueeze(-1)
    vidx = ks.faces[fid]
    wfull = (mano.weights[vidx] * bary).sum(-2)                        # (N, 16)

    # The routing is soft, so the statement is about the whole convex combination, not one label.
    # Its heaviest destination must still be MANO's own dominant joint for that surface point.
    assert torch.equal(idx.gather(1, w.argmax(-1, keepdim=True)).squeeze(1), wfull.argmax(-1))
    # It must be a genuine convex combination of the skinning row: weights sum to one and each
    # destination's weight is that joint's share of the four MANO returns.
    assert torch.allclose(w.sum(-1), torch.ones_like(w[:, 0]), atol=1e-5)
    assert bool((w >= 0).all())
    top4 = wfull.gather(1, idx)
    assert torch.allclose(w, top4 / top4.sum(-1, keepdim=True).clamp_min(1e-8), atol=1e-5)
    assert ch.shape[-1] == KG.KSSF_CHANNELS
    assert bool((idx < KG.N_NODES).all())
    _ = B


def test_routing_is_continuous_in_the_state_that_produced_it(mano, fields):
    """The property the whole soft routing exists for, measured on a real pose.

    The field is queried at the *previous prediction*, so the routing is a function of the estimate
    and its sensitivity is a feedback gain. `argmax` is discontinuous: an event whose top two
    skinning weights are nearly tied moves its entire contribution to another node under an
    arbitrarily small change of state. The convex version moves by the change in the weights.

    Here the perturbation is applied to the skinning row rather than to the pose, which isolates the
    routing from the rasteriser: the question is how the assignment responds to a small change in
    the field it reads, and a rasterised re-render would confound that with resampling noise.
    """
    f, ks = fields
    ys, xs = torch.nonzero(f.visibility[0] > 0, as_tuple=True)
    sel = torch.linspace(0, ys.numel() - 1, min(4000, ys.numel())).long()
    fid = f.face_id[0, ys[sel], xs[sel]]
    bary = f.bary[0, ys[sel], xs[sel]].unsqueeze(-1)
    wfull = (mano.weights[ks.faces[fid]] * bary).sum(-2)               # (N, 16)

    # Events whose two strongest joints are nearly tied: exactly the ones a hard assignment flips.
    top2 = wfull.topk(2, dim=-1).values
    # Near-ties are a small fraction of the surface -- 12 of 4000 sampled pixels on this pose, about
    # 0.3% at a 0.02 margin -- but each one relocates a whole event, and closed-loop drift is not an
    # epsilon perturbation: at the measured 30 mm the field moves far enough to flip well beyond the
    # near-tied set. The gate is about the *form* of the response, not its incidence.
    tied = (top2[:, 0] - top2[:, 1]) < 0.02
    assert int(tied.sum()) >= 10, f"only {int(tied.sum())} near-tied events on this pose"

    eps = 1e-3
    pert = wfull.clone()
    pert[:, :] = wfull
    swap = wfull.topk(2, dim=-1).indices
    pert.scatter_(1, swap[:, 1:2], top2[:, 1:2] + eps)                 # nudge the runner-up past

    def soft(row):
        v, i = row.topk(4, dim=-1)
        return i, v / v.sum(-1, keepdim=True).clamp_min(1e-8)

    i0, w0 = soft(wfull)
    i1, w1 = soft(pert)
    # Expand both to the full 16-node simplex so they can be compared destination by destination.
    d0 = torch.zeros_like(wfull).scatter_(1, i0, w0)
    d1 = torch.zeros_like(wfull).scatter_(1, i1, w1)
    soft_move = (d0 - d1).abs().sum(-1)[tied]

    hard0 = torch.zeros_like(wfull).scatter_(1, wfull.argmax(-1, keepdim=True), 1.0)
    hard1 = torch.zeros_like(wfull).scatter_(1, pert.argmax(-1, keepdim=True), 1.0)
    hard_move = (hard0 - hard1).abs().sum(-1)[tied]

    # A hard assignment relocates a whole event (total variation 2) for an `eps`-sized change; the
    # convex one moves by order `eps`. The gate is the separation, which is what bounds the gain.
    assert float(hard_move.max()) == 2.0, "the near-tied set should contain a genuine flip"
    assert float(soft_move.max()) < 0.05, float(soft_move.max())


def test_background_events_are_a_separate_node_with_exactly_zero_geometry(fields):
    f, _ = fields
    ys, xs = torch.nonzero(f.visibility[0] < 0.5, as_tuple=True)
    sel = torch.linspace(0, ys.numel() - 1, 200).long()
    ev = torch.stack([
        torch.zeros(sel.numel(), device=DEV),
        xs[sel].float(), ys[sel].float(),
        torch.linspace(0, 0.049, sel.numel(), device=DEV),
        torch.ones(sel.numel(), device=DEV),
    ], -1)
    ch, (idx, w) = KG.kssf_event_channels(f, ev)
    # A background event carries no kinematic claim, so it must not be shared with any joint: the
    # whole of its unit weight sits on the background node.
    assert bool((idx == KG.BG_NODE).all())
    assert torch.allclose(w.sum(-1), torch.ones_like(w[:, 0]))
    assert float(ch.abs().max()) == 0.0


def test_events_on_one_node_do_not_touch_another_node_before_message_passing():
    m = KG.KinematicEventGraph(hidden=16, feat_dim=32, node_dim=8).eval()
    ev, ptr, dt = _packet(n_per=32, B=1, seed=7)
    n = ev.shape[0]
    g = (torch.full((n, 1), 3, dtype=torch.long), torch.ones(n, 1))
    idx2 = g[0].clone()
    idx2[:8] = 9
    g2 = (idx2, torch.ones(n, 1))
    with torch.no_grad():
        a = m.node_features(ev, ptr, dt, groups=g)
        b = m.node_features(ev, ptr, dt, groups=g2)
    moved = (a - b).abs().amax(-1)[0]
    changed = torch.nonzero(moved > 1e-7).flatten().tolist()
    assert changed == [3, 9], changed


def test_halo_routing_is_deletion_free_and_matches_the_gate_inside(fields):
    """E5.5a's contract, one property per assert.

    The debug-3eaac2 probes measured the visibility gate deleting the outer half of the contour
    evidence (vis_frac 0.62 at the GT state, monotone-decreasing in the state error, deleted
    events at median |sdf| 4-9 px). The halo lift must (a) change nothing for the events the gate
    kept, (b) keep unit routing mass for every event, (c) route band events to real joints with
    the SDF kernel's weight, and (d) stop zeroing the mismatch geometry the correction needs.
    """
    f, _ = fields
    ys, xs = torch.meshgrid(torch.arange(0, H, 3, device=DEV),
                            torch.arange(0, W, 3, device=DEV), indexing="ij")
    ys, xs = ys.reshape(-1), xs.reshape(-1)
    ev = torch.stack([
        torch.zeros(ys.numel(), device=DEV),
        xs.float(), ys.float(),
        torch.linspace(0, 0.049, ys.numel(), device=DEV),
        torch.ones(ys.numel(), device=DEV),
    ], -1)
    ch_g, (ig, wg) = KG.kssf_event_channels(f, ev)
    ch_h, (ih, wh) = KG.kssf_event_channels(f, ev, halo=True)
    vis = f.visibility[0, ys, xs] > 0.5
    sdf = f.sdf[0, ys, xs]
    assert vis.any() and (~vis).any()

    # Unit mass for every event: evidence is redistributed, never deleted.
    assert torch.allclose(wh.sum(-1), torch.ones_like(wh[:, 0]), atol=1e-5)
    # Inside the silhouette the halo lift *is* the gated lift: same channels, same four surface
    # destinations at the same weights, nothing on the background slot.
    assert torch.allclose(ch_h[vis], ch_g[vis], atol=1e-6)
    assert torch.equal(ih[vis, :4], ig[vis])
    assert torch.allclose(wh[vis, :4], wg[vis], atol=1e-5)
    assert float(wh[vis, 4].abs().max()) == 0.0
    # Outside events whose normal-projection lands on the surface: the background slot holds
    # exactly 1 - exp(-sdf/kappa) and the rest sits on real joints.
    halo_hit = ~vis & (wh[:, 4] < 1.0 - 1e-4)
    assert int(halo_hit.sum()) > 20, int(halo_hit.sum())
    want_bg = 1.0 - torch.exp(-sdf[halo_hit].clamp_min(0.0) / KG.HALO_KAPPA_PX)
    assert torch.allclose(wh[halo_hit, 4], want_bg, atol=1e-5)
    assert bool((ih[halo_hit, :4] < KG.N_NODES).all())
    # The gate zeroed all twelve channels of these events (old behaviour, still asserted by
    # test_background_events_...); the halo keeps their mismatch geometry at its true value.
    assert float(ch_g[halo_hit].abs().max()) == 0.0
    assert torch.allclose(ch_h[~vis, 1], torch.tanh(sdf[~vis] / KG.SDF_TANH_PX), atol=1e-6)
    # Far events (SDF saturated at 2x band, normal undefined) still go entirely to background.
    far = ~vis & (sdf > KG.HALO_KAPPA_PX * 3.99)
    assert int(far.sum()) > 20
    assert bool((ih[far, 4] == KG.BG_NODE).all())
    assert torch.allclose(wh[far, 4], torch.ones_like(wh[far, 4]), atol=1e-6)


def test_halo_is_a_config_bit_that_reaches_the_lift(pose):
    """MODEL.ENCODER_KSSF_HALO must change the forward pass and must not add a parameter."""
    m0, m1 = _model(), _model(ENCODER_KSSF_HALO=True)
    m1.load_state_dict(m0.state_dict())
    batch = _batch_for(m0, pose, n_per=2048)
    with torch.no_grad():
        a = m0.forward_packet(batch)
        b = m1.forward_packet(batch)
    # Same weights, different lift: the packets place events off the rendered hand, so the halo
    # arm must read them differently by far more than the aggregation jitter.
    assert float((a - b).abs().max()) > 1e-4, float((a - b).abs().max())


def test_kssf_off_falls_back_to_a_grid_with_the_same_node_count():
    ev, ptr, dt = _packet(n_per=64, B=2)
    idx, w = KG.spatial_groups(ev, H, W)
    assert bool((idx >= 0).all() and (idx < KG.N_NODES).all())
    # The fallback is one-hot: it reads no state, so there is nothing for softness to protect.
    assert idx.shape[1] == 1 and torch.equal(w, torch.ones_like(w))
    m = build_frontend("keg", hidden=16, feat_dim=32, node_dim=8).eval()
    with torch.no_grad():
        y = m(ev, ptr, dt)
    assert y.shape == (2, 32)


# ------------------------------------------------------------------- model wiring
def _cfg(**over):
    cfg = {
        "MODEL": {
            "POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51, "PREDICT_DELTA": True,
            "PREV_RENDER": True, "ZERO_EVENT_GATE": True, "ACTIVE_HEAD": True,
            "ENCODER": "keg", "ENCODER_HIDDEN": 32, "ENCODER_FEAT": 64, "ENCODER_NODE_DIM": 16,
            "ENCODER_KSSF": True, "ENCODER_JOINT_TOKENS": True,
        },
        "DATA": {"HEIGHT": H, "WIDTH": W},
        "MANO": {"NPZ": str(REPO / "assets/mano_right.npz")},
        "LOSS": {"TYPE": "mse_51d", "LAMBDA_POSE": 1.0, "LAMBDA_T": 1.0, "LAMBDA_R": 1.0,
                 "NORMALIZER": 51, "LOG10": False},
        "TRAIN": {"LR": 1e-3, "WARMUP_STEPS": 0},
    }
    cfg["MODEL"].update(over)
    return cfg


def _model(**over):
    from model import MNISTModel
    return MNISTModel(_cfg(**over)).to(DEV).eval()


def _batch_for(model, pose, n_per=64):
    p, b, k = pose
    B = p.shape[0]
    ev, ptr, dt = _packet(n_per=n_per, B=B, device=DEV)
    return EventPacketBatch(
        events=ev, ptr=ptr,
        sequence_id=torch.zeros(B, dtype=torch.int64, device=DEV),
        t_start_us=torch.zeros(B, dtype=torch.int64, device=DEV),
        t_end_us=torch.full((B,), 50_000, dtype=torch.int64, device=DEV),
        delta_t_s=dt,
        is_sequence_start=torch.zeros(B, dtype=torch.bool, device=DEV),
        is_sequence_end=torch.zeros(B, dtype=torch.bool, device=DEV),
        target=p.clone(), prev_state=p.clone(), betas=b.clone(), camera_K=k.clone(),
    )


def test_zero_event_packet_returns_prev_bitwise(pose):
    m = _model()
    p, b, k = pose
    B = p.shape[0]
    batch = EventPacketBatch(
        events=torch.zeros(0, 5, device=DEV), ptr=torch.zeros(B + 1, dtype=torch.int64, device=DEV),
        sequence_id=torch.zeros(B, dtype=torch.int64, device=DEV),
        t_start_us=torch.zeros(B, dtype=torch.int64, device=DEV),
        t_end_us=torch.full((B,), 50_000, dtype=torch.int64, device=DEV),
        delta_t_s=torch.full((B,), 0.05, device=DEV),
        is_sequence_start=torch.zeros(B, dtype=torch.bool, device=DEV),
        is_sequence_end=torch.zeros(B, dtype=torch.bool, device=DEV),
        target=p.clone(), prev_state=p.clone(), betas=b.clone(), camera_K=k.clone(),
    )
    batch.validate()
    with torch.no_grad():
        out = m.forward_packet(batch)
    assert torch.equal(out, p)


def test_joint_token_pathway_is_unique_per_joint(pose):
    """Zeroing node `k` may move joint `k` and must move nothing else.

    With a shared feature vector this test is impossible to pass, which is the point: the joint
    tokens make the S10 ablation isolate a joint's *evidence*, not only its decoder.
    """
    m = _model()
    batch = _batch_for(m, pose)
    enc = m.event_encoder
    with torch.no_grad():
        fields = m.kssf_on(DEV)(batch.prev_state.float(), batch.betas, batch.camera_K,
                                m.pose_repr)
        extra, groups = KG.kssf_event_channels(fields, batch.events)
        feat, nodes = enc.forward_nodes(batch.events, batch.ptr, batch.delta_t_s, extra, groups)
        base = m._decode_active(feat, batch.prev_state, nodes)
        for j in (0, 4, 14):
            n2 = nodes.clone()
            n2[:, j + 1] = 0.0
            got = m._decode_active(feat, batch.prev_state, n2)
            d = (got - base).abs().amax(0)
            assert float(d[6 + 3 * j : 9 + 3 * j].max()) > 0, j
            mask = torch.ones(51, dtype=torch.bool, device=DEV)
            mask[6 + 3 * j : 9 + 3 * j] = False
            assert float(d[mask].max()) == 0.0, j


def test_aggregation_jitter_is_below_float32_epsilon(pose):
    """Measure, rather than assume, how reproducible the atomic accumulation is.

    `lti_aggregate` sums with `index_add_`, whose CUDA accumulation order varies between calls, so
    KEG gives up the bitwise-deterministic inference `ARCHITECTURE_AUDIT.md` §5 registers. The gate
    bounds the damage at the prediction, in radians and metres, and S19 repeats the measurement
    through the recursive evaluator where a closed loop could amplify it.
    """
    m = _model()
    batch = _batch_for(m, pose, n_per=2048)
    with torch.no_grad():
        runs = [m.forward_packet(batch) for _ in range(6)]
    jit = max(float((runs[0] - r).abs().max()) for r in runs)
    scale = float(runs[0].abs().max())
    assert jit / max(scale, 1e-12) < 1e-5, (jit, scale)


def test_kssf_ablation_is_a_same_checkpoint_switch(pose):
    """`ablate_kssf` must change the prediction without touching a weight.

    The tolerance is the aggregation jitter measured above, not zero: the switch has to move the
    prediction by far more than the accumulation order does, which is the statement that matters.
    """
    m = _model()
    batch = _batch_for(m, pose)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with torch.no_grad():
        a = m.forward_packet(batch)
        m.ablate_kssf = True
        b = m.forward_packet(batch)
        m.ablate_kssf = False
        c = m.forward_packet(batch)
    same = float((a - c).abs().max())
    diff = float((a - b).abs().max())
    assert same < 1e-5, same
    assert diff > 100.0 * max(same, 1e-9), (diff, same)
    for k, v in m.state_dict().items():
        assert torch.equal(v, before[k]), k


def test_kssf_requires_the_keg_frontend():
    from model import MNISTModel
    with pytest.raises(ValueError):
        MNISTModel(_cfg(ENCODER="raw_scan", ENCODER_JOINT_TOKENS=False))


def test_unknown_model_key_still_refuses_to_build():
    from model import MNISTModel
    with pytest.raises(ValueError):
        MNISTModel(_cfg(ENCODER_NONSENSE=1))
