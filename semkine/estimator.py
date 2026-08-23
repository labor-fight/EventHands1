#!/usr/bin/env python3
r"""S10 active per-joint estimator: the unique-pathway head plus a routing mask.

The architecture lives on `MNISTModel` (`ACTIVE_HEAD=true`): the trunk emits features, a root
head writes the six root coordinates, and each of the fifteen joints has a decoder that is the
*only* path to that joint. This module does not invent a second head. It exposes the inference
contract the plan requires:

* `decode` assembles the 51D output
* `ablate` silences every joint decoder on the *same* checkpoint, which must freeze the fingers
  exactly (tested in `tests/test_s10_active_head.py`)
* `apply_route` writes only the groups a router selected, so S9 and S10 share a coordinate
  grouping rather than inventing a second one

A learned router is not trained here. Warm-up is the schedule the plan registers: soft (all
joints) → oracle mask → the parsed RDOR mask. The schedule is a training concern; this file
only applies a mask it is given.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from . import oracle as OR


def apply_route(prev: torch.Tensor, pred: torch.Tensor, active: np.ndarray) -> torch.Tensor:
    """Keep `pred` on active groups and `prev` on the rest. The skip is real, not a scale."""
    return OR.apply_active(prev, pred, active)


def ablate_joint_heads(model, on: bool = True) -> None:
    """Same-checkpoint non-laziness switch. Fingers must become exactly `prev` when on."""
    if not getattr(model, "active_head", False):
        raise RuntimeError("ablate_joint_heads requires MODEL.ACTIVE_HEAD")
    model.ablate_joint_heads = bool(on)


class SoftToParsedSchedule:
    """Registered warm-up: all-active → oracle → parsed mask.

    Fractions are of *optimizer steps*, not epochs, so a change of batch size does not move
    the transition. After `oracle_frac` the caller is expected to pass a parsed mask; this
    object only tells it which source to trust.
    """

    def __init__(self, soft_frac: float = 0.15, oracle_frac: float = 0.35):
        if not 0.0 <= soft_frac <= oracle_frac <= 1.0:
            raise ValueError("need 0 <= soft_frac <= oracle_frac <= 1")
        self.soft_frac = float(soft_frac)
        self.oracle_frac = float(oracle_frac)

    def source(self, step: int, max_steps: int) -> str:
        t = step / max(max_steps, 1)
        if t < self.soft_frac:
            return "soft"
        if t < self.oracle_frac:
            return "oracle"
        return "parsed"
