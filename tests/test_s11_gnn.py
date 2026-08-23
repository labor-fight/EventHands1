#!/usr/bin/env python3
"""S11 gates: kinematic GNN capacity, unique pathway, and tree locality."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from semkine.gnn import KinematicGNN, s10_head_count                       # noqa: E402


def test_capacity_within_ten_percent():
    g = KinematicGNN(feat_dim=256, hidden=64, n_layers=2)
    n = g.parameter_count()
    ref = s10_head_count(256, 64)
    assert n <= ref * 1.10, (n, ref)


def test_ablation_freezes_fingers():
    g = KinematicGNN(feat_dim=32, hidden=16).eval()
    feat = torch.randn(3, 32)
    prev = torch.randn(3, 51)
    with torch.no_grad():
        live = g(feat, prev)
        dead = g(feat, prev, ablate_joints=True)
    assert torch.equal(dead[:, 6:], torch.zeros_like(dead[:, 6:]))
    assert torch.equal(dead[:, :6], live[:, :6])


def test_output_shape_and_root_independence_from_finger_prev():
    g = KinematicGNN(feat_dim=32, hidden=16).eval()
    feat = torch.randn(2, 32)
    prev = torch.zeros(2, 51)
    with torch.no_grad():
        a = g(feat, prev)
        prev2 = prev.clone()
        prev2[:, 6:] = 0.4
        b = g(feat, prev2)
    # Root head reads only the trunk feature, so changing finger prev must not move the root.
    assert torch.allclose(a[:, :6], b[:, :6], atol=1e-6)
    assert a.shape == (2, 51)
