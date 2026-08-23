#!/usr/bin/env python3
"""S15 gates: emit on information, wait out a quiet interval, never emit below min_dt."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from semkine.packetizer import AdaptivePacketizer                           # noqa: E402


def test_emits_when_contour_budget_is_met():
    p = AdaptivePacketizer(min_contour=10, min_dt_us=1000, max_dt_us=50_000)
    assert p.should_emit(5_000, 10, 3)
    assert not p.should_emit(5_000, 9, 3)


def test_max_dt_forces_a_packet_even_if_quiet():
    p = AdaptivePacketizer(min_contour=10_000, max_dt_us=50_000, min_dt_us=1_000)
    assert p.should_emit(50_000, 0, 0)
    assert not p.should_emit(49_999, 0, 0)


def test_min_dt_blocks_a_burst():
    p = AdaptivePacketizer(min_contour=1, min_dt_us=2_000, max_dt_us=50_000)
    assert not p.should_emit(1_000, 100, 5)


def test_support_proxy_requires_groups():
    p = AdaptivePacketizer(proxy="support", min_contour=5, min_groups=2,
                           min_dt_us=1000, max_dt_us=50_000)
    assert not p.should_emit(5_000, 20, 1)
    assert p.should_emit(5_000, 20, 2)
