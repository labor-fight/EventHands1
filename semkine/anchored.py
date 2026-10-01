#!/usr/bin/env python3
r"""Root tracking: absolute measurement + residual tracking (docs/S37_ROOT_TRACKING_VERDICT.md).

A tracker that updates its own previous output (`prev + delta`) has the better single step and the
worse closed loop: every arm of that kind measured in this round (S37, CNN-delta, CNN-track) is
2-3 deg better than a per-packet absolute regressor on the root when handed the true state, and
1-2 deg worse once it feeds itself, because the gain on `prev` its prev-noise curriculum taught it
is too trusting for its own errors. Re-anchoring the fed-back state on an absolute measurement of
the same packet every step keeps the first and removes the second:

    rotation      R = R_trk Exp(a_r Log(R_trk^T R_abs))
    translation, fingers   x = (1 - a_f) x_trk + a_f x_abs

`a = 0` is the tracker's own loop, `a = 1` the absolute arm. The two estimates must not share their
features: one network with both heads (`MODEL.ABS_TRACK`) makes the heads' errors coincide and gains
nothing, while two separately trained networks do.
"""
from __future__ import annotations

import torch
from torch import nn

from .lie import so3_exp, so3_log


def anchor_blend(x_trk: torch.Tensor, x_abs: torch.Tensor, a_root: float, a_rest: float) -> torch.Tensor:
    """`(B, 51)` blend of a tracked and an absolute pose: geodesic on the root rotation (columns 3:6),
    linear on translation and fingers. Computed in fp32."""
    rest = (1.0 - a_rest) * x_trk + a_rest * x_abs
    with torch.autocast(device_type=x_trk.device.type, enabled=False):
        Rt = so3_exp(x_trk[:, 3:6].float())
        Ra = so3_exp(x_abs[:, 3:6].float())
        rot = so3_log(Rt @ so3_exp(a_root * so3_log(Rt.transpose(-1, -2) @ Ra)))
    return torch.cat([rest[:, :3], rot.to(rest.dtype), rest[:, 6:]], dim=-1)


class AnchoredTracker(nn.Module):
    """Two trained dense (LNES) arms run on the same packet; the blend is the output and the next state.

    `abs_model` ignores `prev` (state-free); `trk_model` is a `prev + delta` arm. An event-free packet
    holds `prev` (the absolute arm has nothing to measure there)."""

    #: dense LNES input, as `MNISTModel` without an ENCODER (the evaluators read this)
    encoder_name = ""

    def __init__(self, abs_model: nn.Module, trk_model: nn.Module, a_root: float, a_rest: float):
        super().__init__()
        self.abs_model, self.trk_model = abs_model, trk_model
        self.a_root, self.a_rest = float(a_root), float(a_rest)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        x_abs = self.abs_model(x, prevpos, betas=betas, camera_K=camera_K)
        x_trk = self.trk_model(x, prevpos, betas=betas, camera_K=camera_K)
        out = anchor_blend(x_trk, x_abs, self.a_root, self.a_rest)
        empty = x.reshape(x.shape[0], -1).abs().sum(dim=1, keepdim=True) <= 0
        return torch.where(empty, prevpos.to(out.dtype), out)
