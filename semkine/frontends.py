#!/usr/bin/env python3
r"""S16 frontend battle: one arm at a time, parameter budget within 10%.

The plan's novelty defence says the frontend is not a contribution. These modules exist so a
reviewer who asks "why not AEGNN / SAST / a PointNet++" gets a measured answer rather than a
citation. Each arm is a *minimum mechanism*, not a framework port:

* `lnes`         the existing dense surface (not reimplemented)
* `raw_scan`     S2 gated scan
* `sparse_cell`  S2 fallback
* `aegnn_lite`   k-NN graph in (x, y, t), two message-passing layers, global pool
                 -- no detection head, no voxelization, no async framework

SAST / FARSE / SNN thought-arms are represented by the same graph module with a different
neighbour rule rather than by importing those repositories. rpg_asynet is not used.

A battle run trains one frontend against the frozen S1 Track+domrand protocol, same loss, same
seeds, parameter count within 10% of `raw_scan`. Winner is the one whose paired CI on recursive
RA excludes zero in its favour; a tie keeps LNES.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .encoder import TOKEN_DIM, RawEventEncoder, SparseCellEncoder, event_tokens
from .events import EV_BATCH, EV_T, EV_X, EV_Y


class AEGNNLite(nn.Module):
    """Two-layer k-NN graph on event tokens. Pool is mean+max over the packet."""

    def __init__(self, height: int = 180, width: int = 240, hidden: int = 96,
                 feat_dim: int = 256, k: int = 16, t_scale: float = 80.0,
                 extra_channels: int = 0):
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.k = int(k)
        self.t_scale = float(t_scale)
        self.feat_dim = int(feat_dim)
        in_dim = TOKEN_DIM + int(extra_channels)
        self.enc = nn.Linear(in_dim, hidden)
        # Single linear maps keep the arm inside the ±10% parameter budget of `raw_scan`.
        self.msg = nn.Linear(2 * hidden, hidden)
        self.upd = nn.Linear(2 * hidden, hidden)
        self.proj = nn.Sequential(nn.Linear(2 * hidden, feat_dim), nn.ReLU(inplace=True),
                                  nn.Linear(feat_dim, feat_dim))

    def _knn(self, events: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
        """For each event, indices of its k nearest neighbours in the same packet. `(N, k)`."""
        n = events.shape[0]
        if n == 0:
            return events.new_zeros((0, self.k), dtype=torch.long)
        xyz = torch.stack([
            events[:, EV_X] / self.width,
            events[:, EV_Y] / self.height,
            events[:, EV_T] * self.t_scale,
        ], -1)
        idx = torch.zeros(n, self.k, dtype=torch.long, device=events.device)
        B = int(ptr.numel() - 1)
        for b in range(B):
            a, z = int(ptr[b]), int(ptr[b + 1])
            if z <= a:
                continue
            p = xyz[a:z]
            d = torch.cdist(p, p)
            d.fill_diagonal_(1e6)
            kk = min(self.k, max(z - a - 1, 1))
            nbr = d.topk(kk, largest=False).indices
            if kk < self.k:
                pad = nbr[:, :1].expand(-1, self.k).clone()
                pad[:, :kk] = nbr
                nbr = pad
            idx[a:z] = nbr + a
        return idx

    def forward(self, events, ptr, delta_t_s, extra: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        B = int(ptr.numel() - 1)
        tok = event_tokens(events, ptr, delta_t_s, self.height, self.width)
        if extra is not None and extra.shape[0]:
            tok = torch.cat([tok, extra], -1)
        if events.shape[0] == 0:
            return events.new_zeros((B, self.feat_dim))
        h = torch.relu(self.enc(tok))
        nbr = self._knn(events, ptr)
        src = h[nbr]                                                      # (N, k, H)
        msg = torch.relu(self.msg(torch.cat([h.unsqueeze(1).expand_as(src), src], -1))).mean(1)
        h = torch.relu(self.upd(torch.cat([h, msg], -1)))
        out = events.new_zeros(B, 2 * h.shape[-1])
        for b in range(B):
            a, z = int(ptr[b]), int(ptr[b + 1])
            if z <= a:
                continue
            hh = h[a:z]
            out[b] = torch.cat([hh.mean(0), hh.max(0).values], 0)
        return self.proj(out)


def _pick(kw, *names):
    return {k: kw[k] for k in names if k in kw}


def build_frontend(kind: str, **kw) -> nn.Module:
    kind = (kind or "raw_scan").lower()
    shared = _pick(kw, "height", "width", "hidden", "feat_dim", "extra_channels")
    if kind in ("raw_scan", "scan", "raw"):
        return RawEventEncoder(**shared)
    if kind in ("sparse_cell", "fallback", "cell"):
        return SparseCellEncoder(**shared, **_pick(kw, "cell"))
    if kind in ("aegnn_lite", "aegnn", "graph"):
        return AEGNNLite(**shared, **_pick(kw, "k", "t_scale"))
    raise ValueError(f"unknown frontend {kind!r}")


def frontend_param_count(mod: nn.Module) -> int:
    return sum(p.numel() for p in mod.parameters())
