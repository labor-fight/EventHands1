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

This module applies the mask supplied by the caller.
"""
from __future__ import annotations

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
