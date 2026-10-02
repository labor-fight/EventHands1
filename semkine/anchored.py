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
features: one network with both heads (the retired `MODEL.ABS_TRACK`, commit a555857) made the heads'
errors coincide and gained nothing. Trained to the full budget the delta arm becomes absolute-like and
two-network anchoring adds no more than a second absolute network would (docs/S37_ROOT_TRACKING_VERDICT.md);
`CausalFilter` below -- the "no motion" tracker -- is the form the round kept.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

def _quat(aa: np.ndarray) -> np.ndarray:
    """axis-angle (B, 3) -> unit quaternion (B, 4), w first."""
    th = np.linalg.norm(aa, axis=-1, keepdims=True)
    return np.concatenate([np.cos(0.5 * th), np.sin(0.5 * th) * (aa / np.maximum(th, 1e-12))], axis=-1)


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a.T
    bw, bx, by, bz = b.T
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def _axis_angle(q: np.ndarray) -> np.ndarray:
    """unit quaternion -> axis-angle with |phi| <= pi (the w >= 0 hemisphere)."""
    q = np.where(q[:, :1] < 0, -q, q)
    v = q[:, 1:]
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * (2.0 * np.arctan2(n, q[:, :1]) / np.maximum(n, 1e-12))


def _pq(a):
    """axis-angle (3 floats) -> unit quaternion [w, x, y, z]; plain Python floats (float64)."""
    th = math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])
    if th < 1e-12:
        return [1.0, 0.5 * a[0], 0.5 * a[1], 0.5 * a[2]]
    s = math.sin(0.5 * th) / th
    return [math.cos(0.5 * th), a[0] * s, a[1] * s, a[2] * s]


def _pmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw]


def _paa(q):
    """unit quaternion -> axis-angle with |phi| <= pi (the w >= 0 hemisphere), as `_axis_angle`."""
    if q[0] < 0:
        q = [-v for v in q]
    n = math.sqrt(q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    k = 2.0 * math.atan2(n, q[0]) / n if n > 1e-12 else 2.0
    return [q[1] * k, q[2] * k, q[3] * k]


def root_step(r, ref_q, prev, gain: float):
    """S38 root measurement on one packet, in float64 Python arithmetic: the measured rotation
    R = Exp(r) R_ref (`ref_q` its quaternion), then for `gain` < 1 the geodesic step of that size from
    `prev` toward it, `R_prev Exp(gain Log(R_prev^T R))` -- `anchor_blend`'s slerp. Axis-angle in and
    out. One packet's worth of scalars: at batch 1 numpy's per-call dispatch costs ten times the math."""
    q = _pmul(_pq(r), ref_q)
    if gain < 1.0:
        qp = _pq(prev)
        rel = _paa(_pmul([qp[0], -qp[1], -qp[2], -qp[3]], q))
        q = _pmul(qp, _pq([gain * rel[0], gain * rel[1], gain * rel[2]]))
    return _paa(q)


def anchor_blend(x_trk: torch.Tensor, x_abs: torch.Tensor, a_root: float, a_rest: float,
                 a_trans=None) -> torch.Tensor:
    """`(B, 51)` blend of a tracked and an absolute pose: geodesic on the root rotation (columns 3:6),
    `R_trk Exp(a_root Log(R_trk^T R_abs))` as a quaternion slerp; linear on the fingers (`a_rest`) and
    the translation (`a_trans`, default `a_rest`).

    Inference only (no gradient): computed in float64 numpy on the host and returned on `x_trk`'s
    device. A per-packet blend of 51 numbers is a few dozen element-wise ops; on the GPU at batch 1
    they cost ~1-3 ms of kernel launches, here ~0.1 ms including the two transfers."""
    t = x_trk.detach().double().cpu().numpy()
    a = x_abs.detach().double().cpu().numpy()
    out = (1.0 - a_rest) * t + a_rest * a
    at = a_rest if a_trans is None else a_trans
    out[:, :3] = (1.0 - at) * t[:, :3] + at * a[:, :3]
    qt = _quat(t[:, 3:6])
    rel = _axis_angle(_qmul(qt * np.array([1.0, -1.0, -1.0, -1.0]), _quat(a[:, 3:6])))
    out[:, 3:6] = _axis_angle(_qmul(qt, _quat(a_root * rel)))
    return torch.from_numpy(out).to(device=x_trk.device, dtype=x_trk.dtype)


class AnchoredTracker(nn.Module):
    """Two trained dense (LNES) arms run on the same packet; the blend is the output and the next state.

    `abs_model` ignores `prev` (state-free); `trk_model` is a `prev + delta` arm. An event-free packet
    holds `prev` (the absolute arm has nothing to measure there)."""

    #: dense LNES input, as `MNISTModel` without an ENCODER (the evaluators read this)
    encoder_name = ""

    def __init__(self, abs_model: nn.Module, trk_model: nn.Module, a_root: float, a_rest: float,
                 a_trans=None):
        super().__init__()
        self.abs_model, self.trk_model = abs_model, trk_model
        self.a_root, self.a_rest = float(a_root), float(a_rest)
        #: translation gain; None ties it to the fingers' (the depth is what tracking measures best)
        self.a_trans = None if a_trans is None else float(a_trans)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        x_abs = self.abs_model(x, prevpos, betas=betas, camera_K=camera_K)
        x_trk = self.trk_model(x, prevpos, betas=betas, camera_K=camera_K)
        out = anchor_blend(x_trk, x_abs, self.a_root, self.a_rest, self.a_trans)
        empty = x.reshape(x.shape[0], -1).abs().sum(dim=1, keepdim=True) <= 0
        return torch.where(empty, prevpos.to(out.dtype), out)


class CausalFilter(nn.Module):
    """A state-free arm with the lightest possible state: the fed-back output is the previous output
    moved toward this packet's measurement by constant gains (geodesic on the root, linear on fingers
    and translation, `anchor_blend` with the previous output as the prior). `a = 1` is the arm itself.
    An event-free packet holds `prev`. This is the zero-parameter form of the anchoring above, with
    the tracker replaced by "no motion"."""

    encoder_name = ""

    def __init__(self, abs_model: nn.Module, a_root: float, a_rest: float, a_trans: float = 1.0):
        super().__init__()
        self.abs_model = abs_model
        self.a_root, self.a_rest, self.a_trans = float(a_root), float(a_rest), float(a_trans)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        meas = self.abs_model(x, prevpos, betas=betas, camera_K=camera_K)
        prev = prevpos.to(meas.dtype)
        out = anchor_blend(prev, meas, self.a_root, self.a_rest, self.a_trans)
        empty = x.reshape(x.shape[0], -1).abs().sum(dim=1, keepdim=True) <= 0
        return torch.where(empty, prev, out)
