"""CPU tests: semkine/s45sm_ctx.py (router context, docs/EXPERIMENT_RECORDS.md#s45sm-r6c-prereg-20261031)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import eval_track as ET  # noqa: E402
from semkine import s45rt_router as RT  # noqa: E402
from semkine import s45sm as SM  # noqa: E402
from semkine import s45sm_ctx as CTX  # noqa: E402
from semkine import s45sm_fast as FAST  # noqa: E402
from semkine import xag as XAG  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y  # noqa: E402

NPZ = ROOT / "assets/mano_right.npz"
T0P = ROOT / "outputs/s45rt/20261031/t0p/t0p_final.ckpt"


def packets(seed=0, n_per=(900, 0, 400, 1500), earlier=(0, 0, 0, 0), span=0.05):
    """Packets of `n_per` events in [0, span) s; with `earlier`, the context batch: `earlier` events before the packet,
    then the packet's own events shifted by the context's extra duration (0.1 s)."""
    g = torch.Generator().manual_seed(seed)
    cur, ctx, ptr, cptr = [], [], [0], [0]
    for b, (n, m) in enumerate(zip(n_per, earlier)):
        ev = torch.zeros(n, 5)
        ev[:, EV_BATCH] = b
        ev[:, EV_X] = torch.randint(80, 140, (n,), generator=g).float()
        ev[:, EV_Y] = torch.randint(60, 110, (n,), generator=g).float()
        ev[:, EV_T] = torch.sort(torch.rand(n, generator=g) * span).values
        ev[:, EV_P] = (torch.rand(n, generator=g) > 0.5).float()
        pre = torch.zeros(m, 5)
        pre[:, EV_BATCH] = b
        pre[:, EV_X] = torch.randint(20, 220, (m,), generator=g).float()
        pre[:, EV_Y] = torch.randint(10, 170, (m,), generator=g).float()
        pre[:, EV_T] = torch.sort(torch.rand(m, generator=g) * 0.1).values
        pre[:, EV_P] = (torch.rand(m, generator=g) > 0.5).float()
        shifted = ev.clone()
        shifted[:, EV_T] += 0.1 if m else 0.0
        cur.append(ev)
        ctx.append(torch.cat([pre, shifted]))
        ptr.append(ptr[-1] + n)
        cptr.append(cptr[-1] + m + n)
    B = len(n_per)
    dt = torch.full((B,), span)
    cdt = torch.tensor([span + (0.1 if m else 0.0) for m in earlier])
    return (torch.cat(cur), torch.tensor(ptr), dt), {"events": torch.cat(ctx), "ptr": torch.tensor(cptr), "delta_t_s": cdt}


class Batch:
    def __init__(self, events, ptr, dt):
        self.events, self.ptr, self.delta_t_s = events, ptr, dt


def nets():
    torch.manual_seed(0)
    sr = SM.build(NPZ).eval()
    t0p, _ = RT.load(T0P)
    return {"sr": sr, "t0p": t0p.eval()}


@torch.no_grad()
def test_context_equal_to_packet_is_the_fast_forward():
    (events, ptr, dt), _ = packets(1)
    src, mask = XM.stride_sample(ptr, SM.MAX_NODES)
    for kind, net in nets().items():
        ref = FAST.router_forward(net, events, ptr, dt, src, mask)
        got = CTX.router_forward_ctx(net, ptr, {"events": events, "ptr": ptr, "delta_t_s": dt}, src, mask)
        assert set(ref) == set(got)
        for k in ref:
            assert torch.equal(ref[k], got[k]), (kind, k)


def test_context_nodes_are_the_packet_events():
    (events, ptr, dt), ctx = packets(2, n_per=(700, 0, 300), earlier=(500, 40, 0))
    src, mask = XM.stride_sample(ptr, SM.MAX_NODES)
    idx = CTX.context_nodes(ptr, ctx["ptr"], src, mask)
    a, b = events[src[mask]], ctx["events"][idx[mask]]
    assert torch.equal(a[:, [EV_BATCH, EV_X, EV_Y, EV_P]], b[:, [EV_BATCH, EV_X, EV_Y, EV_P]])
    shift = torch.where(b[:, EV_BATCH] == 2, 0.0, 0.1)                       # packet 2 has no earlier events
    assert torch.allclose(b[:, EV_T] - a[:, EV_T], shift, atol=1e-6)
    assert int(idx[~mask].abs().sum()) == 0


@torch.no_grad()
def test_context_changes_the_routes():
    (events, ptr, dt), ctx = packets(3, n_per=(800, 600), earlier=(2500, 1800))
    src, mask = XM.stride_sample(ptr, SM.MAX_NODES)
    net = nets()["sr"]
    ref = FAST.router_forward(net, events, ptr, dt, src, mask)
    got = CTX.router_forward_ctx(net, ptr, ctx, src, mask)
    assert not torch.equal(ref["z"][mask], got["z"][mask])


def test_check_tail():
    ev5 = np.zeros((3, 5), np.float32)
    ev5[:, 1:3] = [[1, 2], [3, 4], [5, 6]]
    ev5[:, 3] = [0.001, 0.02, 0.049]
    ev5[:, 4] = [1, 0, 1]
    ctx = np.zeros((5, 5), np.float32)
    ctx[:2, 1:3] = [[9, 9], [8, 8]]
    ctx[2:] = ev5
    ctx[2:, 3] += 0.15
    CTX.check_tail(ev5, ctx, 200)
    with pytest.raises(RuntimeError):
        CTX.check_tail(ev5, ctx, 150)
    bad = ctx.copy()
    bad[-1, 4] = 0
    with pytest.raises(RuntimeError):
        CTX.check_tail(ev5, bad, 200)


def fake_sequence(ms=1000, seed=0):
    rng = np.random.default_rng(seed)
    counts = rng.integers(0, 6, ms)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n = int(offsets[-1])
    events = np.stack([rng.integers(0, 240, n), rng.integers(0, 180, n), rng.integers(0, 2, n)], 1).astype(np.int16)
    return events, offsets


def test_eval_patch_attaches_the_clipped_context():
    events, offsets = fake_sequence()
    native = ET._window_events
    prev, betas, K = torch.zeros(1, 51), torch.zeros(1, 10), torch.eye(3).view(1, 3, 3)
    CTX.install_eval(200)
    try:
        CTX._STATE["runs"] = [(0, 600), (700, 1000)]
        for end, want in ((49, None), (99, 100), (249, 200), (599, 200), (749, None), (899, 200)):
            ev5 = ET._window_events(events, offsets, None, end, 50)
            batch = ET.make_eval_packet(ev5, prev, betas, K, 50, "cpu")
            assert np.array_equal(batch.events.numpy(), native(events, offsets, None, end, 50))
            ctx = CTX.context_of(batch)
            if want is None:
                assert ctx is None
                continue
            assert ctx["ms"] == want and float(ctx["delta_t_s"]) == pytest.approx(want * 1e-3)
            assert np.array_equal(ctx["events"].numpy(), native(events, offsets, None, end, want))
            assert ctx["ptr"].tolist() == [0, len(ctx["events"])]
        other = ET._window_events(events, offsets, None, 299, 50)
        batch = ET.make_eval_packet(other.copy(), prev, betas, K, 50, "cpu")        # not the events just cut
        assert CTX.context_of(batch) is None
        with pytest.raises(RuntimeError):
            ET._window_events(events, offsets, None, 650, 50)                 # between the runs
    finally:
        CTX.uninstall_eval()
    assert ET._window_events is native and CTX.installed() is None


def test_routing_wrapper(tmp_path):
    torch.manual_seed(0)
    net = SM.build(NPZ)
    ckpt = tmp_path / "sr.ckpt"
    SM.save_checkpoint(net, ckpt)
    (events, ptr, dt), ctx = packets(4, n_per=(800, 0, 500), earlier=(2000, 300, 900))
    src, mask = XM.stride_sample(ptr, SM.MAX_NODES)
    native = CTX._NATIVE["fast"]
    CTX.install_routing()
    try:
        FAST.install("sr", ckpt, "pfa", 0.5)
        assert XAG.sample_sources is FAST.predicted_sources and XAG.sample_sources is not native
        plain = Batch(events, ptr, dt)
        ref = native(plain, src, mask)
        got = XAG.sample_sources(plain, src, mask)                             # no context: falls through
        assert all(torch.equal(ref[k], got[k]) for k in ref)
        same = Batch(events.clone(), ptr, dt)
        setattr(same.events, CTX.ATTR, {"events": events, "ptr": ptr, "delta_t_s": dt, "ms": 50})
        got = XAG.sample_sources(same, src, mask)                              # context = packet: same routes
        assert all(torch.equal(ref[k], got[k]) for k in ref)
        longer = Batch(events.clone(), ptr, dt)
        setattr(longer.events, CTX.ATTR, {**ctx, "ms": 150})
        got = XAG.sample_sources(longer, src, mask)
        assert not torch.equal(ref["xag_face"][mask], got["xag_face"][mask])
    finally:
        FAST.uninstall()
        CTX.uninstall_routing()
    assert SM.predicted_sources is CTX._NATIVE["sm"] and FAST.predicted_sources is native
    assert XAG.sample_sources is SM._NATIVE["sample_sources"]


def test_selection_rule():
    from tools.s45sm import selection as SEL

    def probe(values):
        return {"by_context": {str(c): {"ylf_local": {"occupancy_overlap_patch": o, "part_cross_share": x}}
                               for c, (o, x) in values.items()}}
    base = {50: (0.467, 0.294)}
    best, rows = SEL.rule_ctx(probe({**base, 100: (0.480, 0.290), 150: (0.490, 0.300), 200: (0.495, 0.297),
                                 300: (0.495, 0.299)}))
    assert best == 200 and not rows["100"]["qualifies"] and rows["150"]["cross_ok"] is False
    assert rows["300"]["qualifies"]                                         # tie on occupancy: the shorter window
    best, _ = SEL.rule_ctx(probe({**base, 100: (0.47, 0.29), 150: (0.48, 0.29), 200: (0.486, 0.29), 300: (0.5, 0.31)}))
    assert best is None
