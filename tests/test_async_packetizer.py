#!/usr/bin/env python3
"""P0.1 gates: the packetizer must conserve events across calls.

`should_emit` (S15, `test_s15_packetizer.py`) tests the *policy*. These tests are about the
*plumbing*, which is what a caller-owned asynchronous runtime actually depends on: a stream
arrives in arbitrary chunks, and the split must not change the answer. Every input event has to
leave through exactly one emitted packet or stay in the buffer, once, in order -- otherwise no
latency or equivalence claim downstream is meaningful, because the sparse arm would be scoring on
fewer events than the dense control it is compared against.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from semkine.packetizer import AdaptivePacketizer                           # noqa: E402


def _stream(n=500, seed=0, rate_us=40):
    """A synthetic stream: strictly increasing microsecond stamps, random pixels/polarity."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.integers(1, rate_us, size=n)).astype(np.int64)
    ev = np.stack([rng.integers(0, 240, n), rng.integers(0, 180, n),
                   rng.integers(0, 2, n)], -1).astype(np.float32)
    return ev, t


def _drain(p, out):
    """All events the packetizer has accounted for: emitted packets plus whatever it still holds."""
    got = [pk.events for pk in out if len(pk.events)]
    held = p.buffered_events()
    if len(held):
        got.append(held)
    return np.concatenate(got, 0) if got else np.zeros((0, 4), np.float32)


def test_chunked_push_matches_one_shot():
    """push(A); push(B) must emit exactly what push(concat(A, B)) emits."""
    ev, t = _stream(400)
    kw = dict(min_contour=37, min_dt_us=1_000, max_dt_us=50_000)

    whole = AdaptivePacketizer(**kw)
    ref = whole.push(ev, t)

    split = AdaptivePacketizer(**kw)
    cut = 173
    got = split.push(ev[:cut], t[:cut]) + split.push(ev[cut:], t[cut:])


    assert len(got) == len(ref)
    for a, b in zip(ref, got):
        assert a.t_start_us == b.t_start_us and a.t_end_us == b.t_end_us
        assert np.array_equal(a.events, b.events)


@pytest.mark.parametrize("n_chunks", [1, 2, 3, 5, 7, 11, 20])
def test_conservation_under_arbitrary_chunking(n_chunks):
    """Emitted + buffered == input, event for event, in order, for any chunking."""
    ev, t = _stream(600, seed=n_chunks)
    p = AdaptivePacketizer(min_contour=53, min_dt_us=1_000, max_dt_us=50_000)
    bounds = np.linspace(0, len(t), n_chunks + 1).astype(int)
    out = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        out += p.push(ev[a:b], t[a:b])
    got = _drain(p, out)


    assert len(got) == len(t), f"lost {len(t) - len(got)} of {len(t)} events"
    assert np.array_equal(got[:, :3], ev)
    assert np.array_equal(got[:, 3].astype(np.int64), t)


def test_packet_intervals_are_contiguous_and_half_open():
    """Packet k's t_end is packet k+1's t_start: no gap, no overlap, no double-counted event."""
    ev, t = _stream(800, seed=3)
    p = AdaptivePacketizer(min_contour=29, min_dt_us=500, max_dt_us=50_000)
    out = p.push(ev, t)
    assert len(out) >= 3
    for a, b in zip(out[:-1], out[1:]):
        assert a.t_end_us == b.t_start_us
    for pk in out:
        ts = pk.events[:, 3].astype(np.int64)
        assert bool((ts >= pk.t_start_us).all()) and bool((ts < pk.t_end_us).all())


def test_max_dt_emits_through_a_quiet_interval():
    """A silent gap longer than max_dt must still produce a packet, at the deadline."""
    ev = np.zeros((3, 3), np.float32)
    t = np.array([0, 10, 200_000], dtype=np.int64)
    p = AdaptivePacketizer(min_contour=10_000, min_dt_us=1_000, max_dt_us=50_000)
    out = p.push(ev, t)
    assert len(out) >= 1
    assert out[0].t_end_us == 50_000
    assert len(out[0].events) == 2


def test_min_dt_holds_a_burst():
    """A burst that meets the contour budget inside min_dt stays buffered."""
    ev = np.zeros((200, 3), np.float32)
    t = np.arange(200, dtype=np.int64)          # 200 events inside 200 us
    p = AdaptivePacketizer(min_contour=10, min_dt_us=5_000, max_dt_us=50_000)
    out = p.push(ev, t)
    assert out == []
    assert len(p.buffered_events()) == 200


def test_support_proxy_counts_groups_of_contour_events_only():
    ev, t = _stream(300, seed=5, rate_us=200)
    mask = np.zeros(len(t), bool)
    mask[::2] = True
    groups = np.zeros(len(t), np.int64)
    groups[100:] = 1
    p = AdaptivePacketizer(proxy="support", min_contour=20, min_groups=2,
                           min_dt_us=1_000, max_dt_us=50_000)
    out = p.push(ev, t, contour_mask=mask, group_ids=groups)
    for pk in out:
        assert pk.n_groups >= 2 or pk.dt_s * 1e6 >= 50_000
    got = _drain(p, out)
    assert len(got) == len(t)


def test_reset_clears_and_empty_push_is_a_no_op():
    ev, t = _stream(100, seed=7)
    p = AdaptivePacketizer(min_contour=1_000, min_dt_us=1_000, max_dt_us=50_000)
    p.push(ev, t)
    assert len(p.buffered_events()) == 100
    before = (p.n_emitted, p.n_held)
    assert p.push(np.zeros((0, 3), np.float32), np.zeros(0, np.int64)) == []
    assert len(p.buffered_events()) == 100 and (p.n_emitted, p.n_held) == before
    p.reset()
    assert len(p.buffered_events()) == 0


def test_single_event_and_cross_sequence():
    """A lone event stays open until the stream is closed: nothing may split a microsecond."""
    p = AdaptivePacketizer(min_contour=1, min_dt_us=0, max_dt_us=50_000)
    assert p.push(np.zeros((1, 3), np.float32), np.array([5], np.int64)) == []
    out = p.flush()
    assert len(out) == 1 and len(out[0].events) == 1
    assert len(p.buffered_events()) == 0
    p.reset()
    # A new sequence starts its own clock; the old t0 must not leak in as a 50 ms deadline.
    out = p.push(np.zeros((2, 3), np.float32), np.array([10_000_000, 10_000_010], np.int64))
    assert out == [] or out[0].t_start_us == 10_000_000


def test_non_monotonic_timestamps_raise():
    """Out-of-order input is an upstream bug; silently sorting it hides the bug."""
    with pytest.raises(ValueError):
        AdaptivePacketizer().push(np.zeros((3, 3), np.float32),
                                  np.array([0, 100, 50], np.int64))
    p = AdaptivePacketizer(min_contour=10_000, min_dt_us=1_000, max_dt_us=50_000)
    p.push(np.zeros((1, 3), np.float32), np.array([1_000], np.int64))
    with pytest.raises(ValueError):
        p.push(np.zeros((1, 3), np.float32), np.array([500], np.int64))
