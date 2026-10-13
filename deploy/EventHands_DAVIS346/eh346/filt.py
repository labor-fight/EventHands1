"""The S39 adaptive filter (DT2 AdaptiveFilter with the ramp1000 gate), float64 on the device.

state+ = blend(state, z, a(n)):  translation and fingers linearly, root rotation along the SO(3) geodesic,
a(n) piecewise linear in log10(n) of the raw event count of the last 50 ms (configs/s39.json, "filter").
An empty window (no LNES event or n <= 0) holds the state exactly. Same operations as the training repository's
`semkine/anchored.py` (AdaptiveFilter without velocity), so it reproduces the server bit for bit.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def t_quat(aa):
    th = torch.linalg.norm(aa, dim=-1, keepdim=True)
    small = th < 1e-8
    s = torch.where(small, 0.5 - th * th / 48.0, torch.sin(0.5 * th) / torch.where(small, torch.ones_like(th), th))
    return torch.cat([torch.cos(0.5 * th), aa * s], dim=-1)


def t_qmul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], dim=-1)


def t_conj(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def t_log(q):
    q = torch.where(q[..., :1] < 0, -q, q)
    w, v = q[..., :1], q[..., 1:]
    n = torch.linalg.norm(v, dim=-1, keepdim=True)
    small = n < 1e-12
    k = torch.where(small, 2.0 / w, 2.0 * torch.atan2(n, w) / torch.where(small, torch.ones_like(n), n))
    return v * k


class GainSchedule(nn.Module):
    """A gain in [0, 1]: a number, or `[[n, gain], ...]` nodes, piecewise linear in log10(n), constant outside."""

    def __init__(self, spec):
        super().__init__()
        nodes = [(1.0, float(spec))] if isinstance(spec, (int, float)) else sorted((float(n), float(g)) for n, g in spec)
        self.const = nodes[0][1] if len({g for _, g in nodes}) == 1 else None
        if self.const is None:
            self.register_buffer("lx", torch.tensor([math.log10(n) for n, _ in nodes], dtype=torch.float64),
                                 persistent=False)
            self.register_buffer("ly", torch.tensor([g for _, g in nodes], dtype=torch.float64), persistent=False)

    def forward(self, n):
        if self.const is not None:
            return self.const
        lx, ly = self.lx.to(n.device), self.ly.to(n.device)
        v = torch.minimum(torch.maximum(torch.log10(n.clamp_min(1.0)), lx[0]), lx[-1])
        i = torch.searchsorted(lx, v, right=True).clamp(1, lx.numel() - 1)
        x0, x1, y0, y1 = lx[i - 1], lx[i], ly[i - 1], ly[i]
        return (y0 + (v - x0) / (x1 - x0) * (y1 - y0)).unsqueeze(1)


def per_step_spec(spec: dict, step_ms: int, base_ms: int = 50) -> dict:
    """Gains converted to another step length with the same time constant for a held target:
    a' = 1 - (1 - a)^(step / 50). Used only when the output step is not 50 ms (see README, "output rate")."""
    if step_ms == base_ms:
        return spec
    f = lambda g: 1.0 - (1.0 - float(g)) ** (step_ms / base_ms)                  # noqa: E731
    out = dict(spec)
    for k in ("a_root", "a_rest", "a_trans"):
        v = spec[k]
        out[k] = f(v) if isinstance(v, (int, float)) else [[n, f(g)] for n, g in v]
    return out


class AdaptiveFilter(nn.Module):
    def __init__(self, spec: dict):
        super().__init__()
        self.g_root = GainSchedule(spec["a_root"])
        self.g_rest = GainSchedule(spec["a_rest"])
        self.g_trans = GainSchedule(spec.get("a_trans", spec["a_rest"]))

    def forward(self, lnes, prevpos, z, n_events):
        """lnes (B, H, W, 2), prev (B, 51), network output z (B, 51), n_events (number or (B,)) -> filtered (B, 51)."""
        prev = prevpos.to(z.dtype)
        b = lnes.shape[0]
        empty = lnes.reshape(b, -1).abs().sum(dim=1, keepdim=True) <= 0
        n = torch.as_tensor(n_events, dtype=torch.float64, device=lnes.device).reshape(-1)
        n = n.expand(b) if n.numel() == 1 else n
        empty = empty | (n <= 0).unsqueeze(1)
        a_r, a_f, a_t = self.g_root(n), self.g_rest(n), self.g_trans(n)
        p, zz = prev.detach().to(torch.float64), z.detach().to(torch.float64)
        t0, q_prev = p[:, :3], t_quat(p[:, 3:6])
        t_new = (1.0 - a_t) * t0 + a_t * zz[:, :3]
        f_new = (1.0 - a_f) * p[:, 6:] + a_f * zz[:, 6:]
        rel = t_log(t_qmul(t_conj(q_prev), t_quat(zz[:, 3:6])))
        q_new = t_qmul(q_prev, t_quat(a_r * rel))
        r_new = t_log(q_new)
        out = torch.cat([t_new, r_new, f_new], dim=1).to(z.dtype)
        return torch.where(empty, prev, out)


def combine(outs):
    """Several networks' outputs (each (1, 51)) -> one, as the evaluated seed ensemble: translation and fingers mean,
    root = tangent-space mean of the rotations about the first one, float64 on the host."""
    if len(outs) == 1:
        return outs[0]
    zz = torch.cat([o.detach() for o in outs], dim=0).double().cpu()
    q = t_quat(zz[:, 3:6])
    ref = q[:1].expand_as(q)
    rel = t_log(t_qmul(t_conj(ref), q))
    r = t_log(t_qmul(q[:1], t_quat(rel.mean(dim=0, keepdim=True))))[0]
    out = torch.cat([zz[:, :3].mean(dim=0), r, zz[:, 6:].mean(dim=0)]).view(1, -1)
    return out.to(device=outs[0].device, dtype=outs[0].dtype)
