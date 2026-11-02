#!/usr/bin/env python3
r"""S2 raw-event tokens: the seven-scalar event token the xyz_mesh relation encoder and the S45-line routers read.

(The recurrent scan encoder and the sparse-cell fallback that once lived here were removed with the S2 / S36 raw-event
encoder family; the token and its helpers are what the S37-XYZ / S45 line uses.)

Event token, seven scalars, in this order, for every event:

    x/W, y/H, 2p-1, t_rel/Δt, log(1 + Δt_ie / ε), SAE_same, SAE_opp

`Δt_ie` is the time since the previous event *in the same packet*. SAE_same / SAE_opp are the
time since the last event of the same / opposite polarity at the same pixel, scaled into
`[0, 1]` by the packet duration. Microsecond stamps (`*_tsub.npy`) make `Δt_ie` a real interval
rather than a millisecond bin; without them the fifth coordinate collapses to a step function.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from .events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y  # noqa: F401  (EV_BATCH is re-exported to the S45-line modules)

TOKEN_DIM = 7
TOKEN_NAMES = ("x_norm", "y_norm", "polarity", "t_norm", "log_dt_ie", "sae_same", "sae_opp")
EPS_DT = 1e-6


def _packet_dt(delta_t_s: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Per-event packet duration, `(N,)`, used to normalise time coordinates."""
    return delta_t_s.clamp_min(EPS_DT)[events[:, EV_BATCH].long()]


def inter_event_dt(events: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """Time since the previous event in the same packet, `(N,)`. First event of a packet is 0."""
    t = events[:, EV_T]
    dt = torch.zeros_like(t)
    if events.shape[0] == 0:
        return dt
    # Vectorised: a difference is intra-packet iff the two events share a batch index.
    same = events[1:, EV_BATCH] == events[:-1, EV_BATCH]
    dt[1:] = torch.where(same, (t[1:] - t[:-1]).clamp_min(0.0), torch.zeros_like(t[1:]))
    return dt


def _pixel_keys(events: torch.Tensor, height: int, width: int) -> torch.Tensor:
    xs = events[:, EV_X].long().clamp(0, width - 1)
    ys = events[:, EV_Y].long().clamp(0, height - 1)
    return events[:, EV_BATCH].long() * (height * width) + ys * width + xs


def sae_times(events: torch.Tensor, ptr: torch.Tensor, height: int, width: int,
              fallback: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Time since last same/opposite polarity at the same pixel, in seconds.

    Vectorised via a stable sort on `(packet, pixel, polarity)`. A pixel that has never fired
    this polarity in the packet returns the packet duration, so the feature stays in `[0, 1]`
    after dividing by `Δt`.

    `fallback` overrides that "never fired" value with a constant number of seconds. The packet
    duration is the right default for a feature that is about to be divided by `Δt`, but it makes
    the feature depend on the window length, which S18 must not do: a rate-invariant encoder needs
    every input channel to be a function of absolute time differences alone.
    """
    n = events.shape[0]
    if n == 0:
        z = events.new_zeros(0)
        return z, z
    B = int(ptr.numel() - 1)
    span_b = events.new_zeros(B)
    counts = ptr[1:] - ptr[:-1]
    ok = counts > 0
    # No `.any().item()`: empty packets keep span 0; indices for those rows are dummies.
    safe_first = torch.where(ok, ptr[:-1], torch.zeros_like(ptr[:-1]))
    safe_last = torch.where(ok, ptr[1:] - 1, torch.zeros_like(ptr[:-1]))
    span_b = torch.where(
        ok, (events[safe_last, EV_T] - events[safe_first, EV_T]).clamp_min(EPS_DT), span_b)
    span = span_b[events[:, EV_BATCH].long()]

    pix = _pixel_keys(events, height, width)
    pol = (events[:, EV_P] > 0.5).long()
    ts = events[:, EV_T]
    t_us = (ts * 1e6).long()
    idx = torch.arange(n, device=events.device)
    # Packets here are ≤300 ms; a fixed 1 s ceiling avoids `t_us.max().item()` (a device sync).
    TMAX_US = 1_000_000

    def _prev_same() -> torch.Tensor:
        key = pix * 2 + pol
        order = torch.argsort(key * (n + 1) + idx, stable=True)
        k, t, orig = key[order], ts[order], idx[order]
        prev = torch.zeros_like(t)
        if n > 1:
            match = k[1:] == k[:-1]
            prev[1:] = torch.where(match, (t[1:] - t[:-1]).clamp_min(0.0), torch.zeros_like(t[1:]))
        out = torch.zeros_like(ts)
        out[orig] = prev
        return out

    def _prev_opp() -> torch.Tensor:
        """Last earlier opposite-polarity event at the same pixel, via searchsorted.

        Every tensor here is a fixed `(n,)`: no `nonzero`, no boolean gathers. Those have
        data-dependent output shapes, which forces the host to drain the CUDA queue before it
        can allocate, and at batch 1 the S36 latency audit measured the stalls costing more
        than the arithmetic. Wrong-polarity events ride along as `-1` sentinel keys that sort
        to the front and can never equal a real query (`pix >= 0`), which also makes an empty
        polarity side correct without a branch.
        """
        out = torch.zeros_like(ts)
        q_all = pix * TMAX_US + t_us
        for src_pol, dst_pol in ((0, 1), (1, 0)):
            is_dst = pol == dst_pol
            key = torch.where(is_dst, q_all, torch.full_like(q_all, -1))
            order = torch.argsort(key)
            key_sorted = key[order]
            cand_pix = torch.where(is_dst, pix, torch.full_like(pix, -1))[order]
            cand_t = ts[order]
            # Last key strictly below the query; landing on a sentinel means "no candidate".
            pos = torch.searchsorted(key_sorted, q_all).sub(1).clamp(min=0)
            hit = (pol == src_pol) & (cand_pix[pos] == pix) & (cand_t[pos] < ts)
            out = torch.where(hit, ts - cand_t[pos], out)
        return out

    same_dt = _prev_same()
    opp_raw = _prev_opp()
    never = span if fallback is None else torch.full_like(span, float(fallback))
    same = torch.where(same_dt > 0, same_dt, never)
    opp = torch.where(opp_raw > 0, opp_raw, never)
    return same, opp


def event_tokens(events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor,
                 height: int, width: int, eps: float = EPS_DT) -> torch.Tensor:
    """`(N, 7)` tokens from a ragged event batch. Empty input returns `(0, 7)`."""
    n = events.shape[0]
    if n == 0:
        return events.new_zeros((0, TOKEN_DIM))
    dt = _packet_dt(delta_t_s, events)
    dt_ie = inter_event_dt(events, ptr)
    sae_s, sae_o = sae_times(events, ptr, height, width)
    tok = torch.stack([
        events[:, EV_X] / float(width),
        events[:, EV_Y] / float(height),
        2.0 * events[:, EV_P] - 1.0,
        events[:, EV_T] / dt,
        torch.log1p(dt_ie / eps),
        (sae_s / dt).clamp(0.0, 1.0),
        (sae_o / dt).clamp(0.0, 1.0),
    ], dim=-1)
    return tok
