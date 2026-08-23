#!/usr/bin/env python3
"""S16 gates: frontends share a contract and stay inside the parameter budget."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from semkine.frontends import (                                            # noqa: E402
    AEGNNLite, build_frontend, frontend_param_count,
)
from semkine.encoder import RawEventEncoder                                # noqa: E402


def _batch():
    ev = torch.tensor([
        [0, 10, 20, 0.00, 1],
        [0, 11, 20, 0.01, 0],
        [0, 12, 21, 0.02, 1],
        [1, 4, 5, 0.00, 1],
        [1, 5, 5, 0.03, 1],
    ], dtype=torch.float32)
    ptr = torch.tensor([0, 3, 5])
    dt = torch.tensor([0.05, 0.05])
    return ev, ptr, dt


def test_all_frontends_emit_the_same_feature_width():
    ev, ptr, dt = _batch()
    for name in ("raw_scan", "sparse_cell", "aegnn_lite"):
        m = build_frontend(name, hidden=32, feat_dim=64).eval()
        with torch.no_grad():
            y = m(ev, ptr, dt)
        assert y.shape == (2, 64), (name, y.shape)


def test_aegnn_empty_packet_is_zero():
    m = AEGNNLite(hidden=16, feat_dim=32).eval()
    ev = torch.zeros(0, 5)
    ptr = torch.tensor([0, 0])
    dt = torch.tensor([0.05])
    with torch.no_grad():
        y = m(ev, ptr, dt)
    assert y.shape == (1, 32)
    assert torch.equal(y, torch.zeros_like(y))


def test_parameter_budget_vs_scan():
    ref = frontend_param_count(RawEventEncoder(hidden=96, feat_dim=256))
    for name in ("sparse_cell", "aegnn_lite"):
        n = frontend_param_count(build_frontend(name, hidden=96, feat_dim=256))
        # The battle is allowed 10%; the fallback is allowed to be smaller.
        assert n <= ref * 1.10, (name, n, ref)
