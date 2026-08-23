#!/usr/bin/env python3
"""S14 gates: LM solve stays on the active set and damps a singular system."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from semkine.gn import active_tangent_mask, lm_solve                       # noqa: E402
from semkine import oracle as OR                                           # noqa: E402


def test_inactive_coordinates_stay_zero():
    g = torch.Generator().manual_seed(0)
    J = torch.randn(40, 51, generator=g)
    r = torch.randn(40, generator=g)
    active = np.zeros(OR.N_GROUPS, dtype=bool)
    active[0] = True
    active[3] = True
    keep = active_tangent_mask(active)
    dx = lm_solve(J, r, damp=1e-2, keep=keep)
    assert float(dx[~keep].abs().max()) == 0.0
    assert float(dx[keep].abs().max()) > 0.0


def test_rank_deficient_system_does_not_raise():
    """A hidden joint contributes no rows: the solve must still return a finite step."""
    J = torch.zeros(5, 51)
    J[:, 0] = 1.0
    r = torch.ones(5)
    keep = torch.ones(51, dtype=torch.bool)
    dx = lm_solve(J, r, damp=1e-3, keep=keep)
    assert torch.isfinite(dx).all()
    assert float(dx[0].abs()) > 0.0


def test_empty_active_set_is_zero():
    J = torch.randn(10, 51)
    r = torch.randn(10)
    dx = lm_solve(J, r, 1e-2, torch.zeros(51, dtype=torch.bool))
    assert float(dx.abs().max()) == 0.0
