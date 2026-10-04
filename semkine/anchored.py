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


class FilteredTracker(nn.Module):
    """A `prev + delta` tracker with a constant-gain causal filter on its own output (DT round,
    docs/DT_RENDER_TRACK_PREREG.md): the fed-back state is `prev` moved toward the tracker's output by
    fixed gains, `anchor_blend(prev, out, ...)` -- geodesic on the root rotation, linear on fingers and
    translation, i.e. `prev + a (out - prev)` with the root on SO(3). `a = 1` is the tracker itself;
    `a_rest` is the finger gain and `a_trans` the translation gain (default: the finger gain). An
    event-free packet holds `prev`. This is the retired delta-trust (which applied one gain to all 51
    numbers) with the root handled on the manifold and the three blocks gained separately."""

    encoder_name = ""

    def __init__(self, trk_model: nn.Module, a_root: float, a_rest: float, a_trans=None):
        super().__init__()
        self.trk_model = trk_model
        self.a_root, self.a_rest = float(a_root), float(a_rest)
        self.a_trans = None if a_trans is None else float(a_trans)

    def set_hand_context(self, betas, camera_K):
        """The evaluators hand the sequence's betas / K to the model under test; forward them."""
        if hasattr(self.trk_model, "set_hand_context"):
            self.trk_model.set_hand_context(betas, camera_K)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        out = self.trk_model(x, prevpos, betas=betas, camera_K=camera_K)
        prev = prevpos.to(out.dtype)
        res = anchor_blend(prev, out, self.a_root, self.a_rest, self.a_trans)
        empty = x.reshape(x.shape[0], -1).abs().sum(dim=1, keepdim=True) <= 0
        return torch.where(empty, prev, res)


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


# ---------------------------------------------------------------------------------------------------------------
# DT2 round, package B (docs/DT2_PREREG.md): a stateful, event-count-adaptive, device-side causal filter.
# `FilteredTracker` above is untouched; `AdaptiveFilter` generalises it: gains that depend on the packet's raw
# event count, an optional alpha-beta velocity term on the root (SO(3) tangent) and the translation, a float64
# torch implementation that stays on the device, and (for the constant-gain, no-velocity case) a host switch that
# calls `anchor_blend` itself and so reproduces `FilteredTracker` bit for bit.
# ---------------------------------------------------------------------------------------------------------------
def _t_quat(aa: torch.Tensor) -> torch.Tensor:
    """axis-angle (B, 3) -> unit quaternion (B, 4), w first (`_quat` on the device; small angles by the series)."""
    th = torch.linalg.norm(aa, dim=-1, keepdim=True)
    small = th < 1e-8
    s = torch.where(small, 0.5 - th * th / 48.0, torch.sin(0.5 * th) / torch.where(small, torch.ones_like(th), th))
    return torch.cat([torch.cos(0.5 * th), aa * s], dim=-1)


def _t_qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def _t_conj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _t_log(q: torch.Tensor) -> torch.Tensor:
    """unit quaternion -> axis-angle with |phi| <= pi (the w >= 0 hemisphere), as `_axis_angle`."""
    q = torch.where(q[..., :1] < 0, -q, q)
    w, v = q[..., :1], q[..., 1:]
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    small = n < 1e-12
    k = torch.where(small, 2.0 / w, 2.0 * torch.atan2(n, w) / torch.where(small, torch.ones_like(n), n))
    return v * k


class GainSchedule(nn.Module):
    """A gain in [0, 1] as a function of the packet's raw event count. A number is a constant; a list of
    `(n, gain)` nodes is piecewise linear in log10(n) between the nodes and constant beyond the first and the
    last (so `[(300, .3), (3000, .5), (30000, .8)]` is .3 below 300 events, .8 above 30000, and linear in the
    decade between). Counts below 1 are treated as 1 (an empty packet never reaches the gain: it holds)."""

    def __init__(self, spec):
        super().__init__()
        if isinstance(spec, (int, float)) and not isinstance(spec, bool):
            nodes = [(1.0, float(spec))]
        else:
            nodes = sorted((float(n), float(g)) for n, g in spec)
        if not nodes:
            raise ValueError("a gain schedule needs a number or at least one (n, gain) node")
        if any(n <= 0 for n, _ in nodes) or len({n for n, _ in nodes}) != len(nodes):
            raise ValueError(f"gain schedule nodes need distinct positive event counts: {nodes}")
        if any(not 0.0 <= g <= 1.0 for _, g in nodes):
            raise ValueError(f"every gain must lie in [0, 1]: {nodes}")
        self.nodes = tuple(nodes)
        #: the constant, or None for a schedule that varies with the count
        self.const = nodes[0][1] if len({g for _, g in nodes}) == 1 else None
        if self.const is None:
            self.register_buffer("lx", torch.tensor([math.log10(n) for n, _ in nodes], dtype=torch.float64), persistent=False)
            self.register_buffer("ly", torch.tensor([g for _, g in nodes], dtype=torch.float64), persistent=False)

    def spec(self):
        return self.const if self.const is not None else [[n, g] for n, g in self.nodes]

    def forward(self, n: torch.Tensor):
        """`n` (B,) float64 event counts -> a Python float (constant schedule) or a (B, 1) float64 tensor."""
        if self.const is not None:
            return self.const
        lx, ly = self.lx.to(n.device), self.ly.to(n.device)
        v = torch.minimum(torch.maximum(torch.log10(n.clamp_min(1.0)), lx[0]), lx[-1])
        i = torch.searchsorted(lx, v, right=True).clamp(1, lx.numel() - 1)
        x0, x1, y0, y1 = lx[i - 1], lx[i], ly[i - 1], ly[i]
        return (y0 + (v - x0) / (x1 - x0) * (y1 - y0)).unsqueeze(1)


def _canon_gain(g):
    """A gain spec in its canonical (JSON-stable) form: a float, or sorted `[[n, gain], ...]`."""
    if isinstance(g, (int, float)) and not isinstance(g, bool):
        GainSchedule(g)                                                  # validates the range
        return float(g)
    nodes = sorted([float(n), float(v)] for n, v in g)
    GainSchedule(nodes)
    return nodes


class AdaptiveFilter(nn.Module):
    """`FilteredTracker` with event-count-dependent gains, an optional alpha-beta velocity and a device-side float64
    implementation (docs/DT2_PREREG.md). Per packet, with `z` the tracker's output given `prev` (the previous
    FILTERED state -- the network never sees a predicted state), `n` the packet's raw event count:

        x- = prev (+) v              translation: prev_t + v_t;  root: Exp(v_r) R_prev (left, camera frame);  fingers: prev
        x+ = x- (+) a(n) (z (-) x-)  translation / fingers: linear;  root: geodesic, `R- Exp(a Log(R-^T R_z))`
        v  <- v + beta (x+ (-) prev - v)   the filtered state's step difference, exponentially averaged

    The gains `a_root`, `a_rest` (fingers) and `a_trans` (default: the finger gain) are numbers or `(n, gain)` node
    lists (`GainSchedule`). `beta_root` / `beta_trans` = 0 (default) switch the velocity off: x- = prev and the module is
    `FilteredTracker` with event-dependent gains. An empty packet (LNES all zero, or `n_events == 0`) returns `prev`
    exactly and the velocity decays by `decay` (default 0.5). Rows are independent: any batch B, one row of
    velocity state each (`reset_state(prev)`, `get_state()`, `set_state(s)`; a state that does not fit the batch is
    re-initialised to zero).

    `n_events` (call argument; int / float / (B,) tensor) is the evaluators' raw event count of the packet
    (`evalx.run_sequence` passes it because `takes_event_count` is True). Without it the count is estimated by the
    number of non-zero LNES entries, a lower bound of the events (several events can hit one pixel).

    `host=True` (constant gains, no velocity only) calls `anchor_blend` on the host -- the very code `FilteredTracker`
    runs -- and so reproduces it bit for bit; the default device path matches `anchor_blend` to float32 rounding
    (<= 1e-6; tests/test_dt_filter2.py) because torch and numpy do not share their transcendental functions.
    Inference only (the filter detaches the tracker's output)."""

    encoder_name = ""
    takes_event_count = True
    SPEC_KEYS = ("a_root", "a_rest", "a_trans", "beta_root", "beta_trans", "decay", "host")

    def __init__(self, trk_model: nn.Module, a_root=1.0, a_rest=1.0, a_trans=None, beta_root: float = 0.0,
                 beta_trans: float = 0.0, decay: float = 0.5, host: bool = False):
        super().__init__()
        self.trk_model = trk_model
        self.g_root = GainSchedule(_canon_gain(a_root))
        self.g_rest = GainSchedule(_canon_gain(a_rest))
        self.g_trans = GainSchedule(_canon_gain(a_rest if a_trans is None else a_trans))
        for nm, v in (("beta_root", beta_root), ("beta_trans", beta_trans), ("decay", decay)):
            if not 0.0 <= float(v) <= 1.0:
                raise ValueError(f"{nm} must lie in [0, 1], got {v}")
        self.beta_root, self.beta_trans, self.decay = float(beta_root), float(beta_trans), float(decay)
        self.host = bool(host)
        if self.host and (any(g.const is None for g in (self.g_root, self.g_rest, self.g_trans))
                          or self.beta_root > 0 or self.beta_trans > 0):
            raise ValueError("host=True is the constant-gain, no-velocity filter (anchor_blend itself)")
        self.register_buffer("_v_root", torch.zeros(0, 3, dtype=torch.float64), persistent=False)
        self.register_buffer("_v_trans", torch.zeros(0, 3, dtype=torch.float64), persistent=False)

    # ---- spec <-> module
    @classmethod
    def canonical_spec(cls, spec: dict) -> dict:
        """A filter spec (a dict, as in JSON) with every key explicit and normalised: `a_root`, `a_rest`, `a_trans`
        (number or `[[n, gain], ...]`), `beta_root`, `beta_trans`, `decay`, `host`. `beta` sets both betas; an absent
        `a_trans` is the finger gain; `name` (a label) is not part of the spec."""
        s = dict(spec)
        s.pop("name", None)
        if "beta" in s:
            b = s.pop("beta")
            s.setdefault("beta_root", b)
            s.setdefault("beta_trans", b)
        bad = set(s) - set(cls.SPEC_KEYS)
        if bad:
            raise ValueError(f"unknown filter spec keys {sorted(bad)}; known: {cls.SPEC_KEYS}")
        out = {"a_root": _canon_gain(s.get("a_root", 1.0)), "a_rest": _canon_gain(s.get("a_rest", 1.0))}
        out["a_trans"] = _canon_gain(s["a_trans"]) if s.get("a_trans") is not None else out["a_rest"]
        out.update(beta_root=float(s.get("beta_root", 0.0)), beta_trans=float(s.get("beta_trans", 0.0)),
                   decay=float(s.get("decay", 0.5)), host=bool(s.get("host", False)))
        return out

    @classmethod
    def from_spec(cls, trk_model: nn.Module, spec: dict) -> "AdaptiveFilter":
        return cls(trk_model, **cls.canonical_spec(spec))

    def spec(self) -> dict:
        return {"a_root": self.g_root.spec(), "a_rest": self.g_rest.spec(), "a_trans": self.g_trans.spec(),
                "beta_root": self.beta_root, "beta_trans": self.beta_trans, "decay": self.decay, "host": self.host}

    def set_hand_context(self, betas, camera_K):
        """The evaluators hand the sequence's betas / K to the model under test; forward them."""
        if hasattr(self.trk_model, "set_hand_context"):
            self.trk_model.set_hand_context(betas, camera_K)

    # ---- per-row state: the velocity
    def reset_state(self, prev: torch.Tensor):
        """Zero velocity for each of `prev`'s rows (a new segment). Consumes no random numbers."""
        b, dev = prev.shape[0], prev.device
        self._v_root = torch.zeros(b, 3, dtype=torch.float64, device=dev)
        self._v_trans = torch.zeros(b, 3, dtype=torch.float64, device=dev)

    def get_state(self) -> dict:
        """A copy of the state (rows = batch): `{"v_root": (B, 3), "v_trans": (B, 3)}` float64 on the device."""
        return {"v_root": self._v_root.clone(), "v_trans": self._v_trans.clone()}

    def set_state(self, state: dict):
        dev = self._v_root.device
        self._v_root = torch.as_tensor(state["v_root"], dtype=torch.float64).to(dev).clone()
        self._v_trans = torch.as_tensor(state["v_trans"], dtype=torch.float64).to(dev).clone()

    def _state_for(self, b: int, dev):
        if self._v_root.shape[0] != b:
            self._v_root = torch.zeros(b, 3, dtype=torch.float64, device=dev)
            self._v_trans = torch.zeros(b, 3, dtype=torch.float64, device=dev)
        elif self._v_root.device != dev:
            self._v_root, self._v_trans = self._v_root.to(dev), self._v_trans.to(dev)

    # ---- the filter
    def forward(self, x, prevpos, betas=None, camera_K=None, n_events=None):
        z = self.trk_model(x, prevpos, betas=betas, camera_K=camera_K)
        prev = prevpos.to(z.dtype)
        b = x.shape[0]
        empty = x.reshape(b, -1).abs().sum(dim=1, keepdim=True) <= 0
        n = None
        if n_events is not None:
            n = torch.as_tensor(n_events, dtype=torch.float64, device=x.device).reshape(-1)
            n = n.expand(b) if n.numel() == 1 else n
            empty = empty | (n <= 0).unsqueeze(1)
        if self.host:
            res = anchor_blend(prev, z, self.g_root.const, self.g_rest.const, self.g_trans.const)
            return torch.where(empty, prev, res)
        if n is None and any(g.const is None for g in (self.g_root, self.g_rest, self.g_trans)):
            n = (x.reshape(b, -1) != 0).sum(dim=1).to(torch.float64)
        a_r, a_f, a_t = self.g_root(n), self.g_rest(n), self.g_trans(n)
        self._state_for(b, x.device)
        p, zz = prev.detach().to(torch.float64), z.detach().to(torch.float64)
        t0, q_prev = p[:, :3], _t_quat(p[:, 3:6])
        t_pred = t0 + self._v_trans if self.beta_trans > 0 else t0
        q_pred = _t_qmul(_t_quat(self._v_root), q_prev) if self.beta_root > 0 else q_prev
        t_new = (1.0 - a_t) * t_pred + a_t * zz[:, :3]
        f_new = (1.0 - a_f) * p[:, 6:] + a_f * zz[:, 6:]
        rel = _t_log(_t_qmul(_t_conj(q_pred), _t_quat(zz[:, 3:6])))
        q_new = _t_qmul(q_pred, _t_quat(a_r * rel))
        r_new = _t_log(q_new)
        if self.beta_root > 0:
            v = self._v_root + self.beta_root * (_t_log(_t_qmul(q_new, _t_conj(q_prev))) - self._v_root)
            self._v_root = torch.where(empty, self.decay * self._v_root, v)
        if self.beta_trans > 0:
            v = self._v_trans + self.beta_trans * ((t_new - t0) - self._v_trans)
            self._v_trans = torch.where(empty, self.decay * self._v_trans, v)
        out = torch.cat([t_new, r_new, f_new], dim=1).to(z.dtype)
        return torch.where(empty, prev, out)
