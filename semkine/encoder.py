#!/usr/bin/env python3
r"""S2 RawEvent-Track: event-token recurrent encoder and its sparse-cell fallback.

The paper contribution is not this frontend. Ev2Hands (3DV'24) and EventEgoHands (2025) already
put a PointNet++ on an event cloud for a hand mesh. The encoder exists so later stages can query
KSSF and route information at event coordinates rather than on a dense 180x240 surface that
throws away 70-88% of the events at 50 ms. Two attempts are budgeted; if both
miss the registered parity gate the mainline falls back to LNES and KSSF/RDOR query the dense
features instead.

Event token, seven scalars, in this order, for every event:

    x/W, y/H, 2p-1, t_rel/Δt, log(1 + Δt_ie / ε), SAE_same, SAE_opp

`Δt_ie` is the time since the previous event *in the same packet*. SAE_same / SAE_opp are the
time since the last event of the same / opposite polarity at the same pixel, scaled into
`[0, 1]` by the packet duration. Microsecond stamps (`*_tsub.npy`) make `Δt_ie` a real interval
rather than a millisecond bin; without them the fifth coordinate collapses to a step function
and the recurrent state has nothing to decay against.

The recurrent update is a gated exponential scan, one packet at a time (Zubić et al., CVPR'24
SSM-events, the scan, not the rest of the paper):

    a_i = exp(-softplus(γ) Δt_ie) ⊙ σ(W_f x_i)
    h_i = a_i ⊙ h_{i-1} + (1 - a_i) ⊙ tanh(W_g x_i)

Packets are independent, so the scan is *segmented* on `ptr`. A zero-event packet produces the
zero vector and, with `PREDICT_DELTA`, the output is bitwise the previous state -- the same
contract `ZERO_EVENT_GATE` enforces on LNES.

The fallback (`sparse_cell`) scatters the same tokens into a coarse grid and mixes only occupied
cells. It is forbidden from allocating a dense `H x W` convolution: that would be LNES with extra
steps.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y

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


def query_render(render: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour lookup of a `(B, H, W, C)` render at event pixels, `(N, C)`."""
    if events.shape[0] == 0:
        return render.new_zeros((0, render.shape[-1]))
    b = events[:, EV_BATCH].long()
    x = events[:, EV_X].long().clamp(0, render.shape[2] - 1)
    y = events[:, EV_Y].long().clamp(0, render.shape[1] - 1)
    return render[b, y, x]


def pad_to_packets(values: torch.Tensor, ptr: torch.Tensor, fill: float = 0.0
                   ) -> Tuple[torch.Tensor, torch.Tensor]:
    """`(N, D)` ragged -> `(B, T, D)` padded plus a boolean `(B, T)` valid mask."""
    B = int(ptr.numel() - 1)
    counts = ptr[1:] - ptr[:-1]
    T = int(counts.max().item()) if B else 0
    D = values.shape[-1] if values.ndim == 2 else 1
    if T == 0 or values.shape[0] == 0:
        return values.new_zeros((B, 0, D)), torch.zeros(B, 0, dtype=torch.bool, device=values.device)
    packed = values.new_full((B, T, D), fill)
    valid = torch.zeros(B, T, dtype=torch.bool, device=values.device)
    for b in range(B):
        a, z = int(ptr[b]), int(ptr[b + 1])
        if z > a:
            packed[b, : z - a] = values[a:z]
            valid[b, : z - a] = True
    return packed, valid


def gated_scan(tokens: torch.Tensor, dt_ie: torch.Tensor, ptr: torch.Tensor,
               w_forget: nn.Linear, w_input: nn.Linear, log_decay: torch.Tensor
               ) -> torch.Tensor:
    """Segmented gated exponential scan. `tokens (N, D)` -> hidden `(N, H)`.

    Implemented as a padded sequential scan so every packet's events stay causal and a later
    packet cannot see an earlier one. Padding is a compute convenience; it is not part of the
    data contract and does not change the values on valid events.
    """
    if tokens.shape[0] == 0:
        return tokens.new_zeros((0, w_input.out_features))
    forget = torch.sigmoid(w_forget(tokens))
    inp = torch.tanh(w_input(tokens))
    decay = torch.exp(-torch.nn.functional.softplus(log_decay) * dt_ie.unsqueeze(-1))
    a = decay * forget
    b = (1.0 - forget) * inp
    a_p, valid = pad_to_packets(a, ptr, fill=0.0)
    b_p, _ = pad_to_packets(b, ptr, fill=0.0)
    B, T, H = a_p.shape
    h = tokens.new_zeros(B, H)
    outs = []
    for t in range(T):
        nxt = a_p[:, t] * h + b_p[:, t]
        # Padding sits after the packet ended; leaving `h` untouched keeps the last real event.
        h = torch.where(valid[:, t].unsqueeze(-1), nxt, h)
        outs.append(h)
    if not outs:
        return tokens.new_zeros((0, H))
    stacked = torch.stack(outs, 1)                                        # (B, T, H)
    # Unpack back to ragged order.
    pieces = []
    for bi in range(B):
        n = int((ptr[bi + 1] - ptr[bi]).item())
        if n:
            pieces.append(stacked[bi, :n])
    return torch.cat(pieces, 0) if pieces else tokens.new_zeros((0, H))


def pool_packets(hidden: torch.Tensor, ptr: torch.Tensor) -> torch.Tensor:
    """Last + mean + max over each packet's hidden states, `(B, 3H)`. Empty packet -> 0."""
    B = int(ptr.numel() - 1)
    H = hidden.shape[-1] if hidden.ndim == 2 and hidden.shape[0] else 0
    if H == 0:
        # Infer H from an empty but well-shaped tensor, or fall back to 0.
        H = hidden.shape[-1] if hidden.ndim == 2 else 0
        return hidden.new_zeros((B, 3 * max(H, 1)))[:, : 0 if H == 0 else 3 * H]
    out = hidden.new_zeros(B, 3 * H)
    for b in range(B):
        a, z = int(ptr[b]), int(ptr[b + 1])
        if z <= a:
            continue
        h = hidden[a:z]
        out[b, :H] = h[-1]
        out[b, H:2 * H] = h.mean(0)
        out[b, 2 * H:] = h.max(0).values
    return out


class RawEventEncoder(nn.Module):
    """Tokenise, scan, pool, project. Output is a feature vector, not a pose.

    The pose head lives in `MNISTModel` (or the S10 joint decoders) so this module cannot sneak a
    second path to the finger angles. Distillation, if any, is applied to the *pose*, not here.
    """

    def __init__(self, height: int = 180, width: int = 240, hidden: int = 128,
                 feat_dim: int = 256, extra_channels: int = 0):
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.hidden = int(hidden)
        self.feat_dim = int(feat_dim)
        in_dim = TOKEN_DIM + int(extra_channels)
        self.w_forget = nn.Linear(in_dim, hidden)
        self.w_input = nn.Linear(in_dim, hidden)
        self.log_decay = nn.Parameter(torch.tensor(2.0))
        self.proj = nn.Sequential(
            nn.Linear(3 * hidden, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def tokens(self, events, ptr, delta_t_s, extra: Optional[torch.Tensor] = None):
        tok = event_tokens(events, ptr, delta_t_s, self.height, self.width)
        if extra is not None:
            tok = torch.cat([tok, extra], dim=-1)
        return tok

    def forward(self, events, ptr, delta_t_s, extra: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        B = int(ptr.numel() - 1)
        if events.shape[0] == 0:
            return events.new_zeros((B, self.feat_dim))
        tok = self.tokens(events, ptr, delta_t_s, extra)
        dt_ie = inter_event_dt(events, ptr)
        hidden = gated_scan(tok, dt_ie, ptr, self.w_forget, self.w_input, self.log_decay)
        return self.proj(pool_packets(hidden, ptr))


class SparseCellEncoder(nn.Module):
    """Fallback: scatter tokens into a coarse grid and mix only occupied cells.

    No dense `H x W` convolution is allocated. Mixing is a per-cell MLP applied to the scattered
    features, then a global pool. Occupied cells of one packet cannot see another packet.
    """

    def __init__(self, height: int = 180, width: int = 240, cell: int = 16,
                 hidden: int = 64, feat_dim: int = 256, extra_channels: int = 0):
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.cell = int(cell)
        self.gh = (self.height + self.cell - 1) // self.cell
        self.gw = (self.width + self.cell - 1) // self.cell
        self.n_cells = self.gh * self.gw
        self.feat_dim = int(feat_dim)
        in_dim = TOKEN_DIM + int(extra_channels)
        self.cell_mlp = nn.Sequential(
            nn.Linear(in_dim + 1, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
        )
        self.proj = nn.Sequential(
            nn.Linear(2 * hidden, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, feat_dim),
        )

    def forward(self, events, ptr, delta_t_s, extra: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        B = int(ptr.numel() - 1)
        tok = event_tokens(events, ptr, delta_t_s, self.height, self.width)
        if extra is not None and extra.shape[0]:
            tok = torch.cat([tok, extra], dim=-1)
        D = tok.shape[-1] if tok.ndim == 2 and tok.shape[0] else TOKEN_DIM
        acc = tok.new_zeros(B, self.n_cells, D)
        cnt = tok.new_zeros(B, self.n_cells, 1)
        if events.shape[0]:
            b = events[:, EV_BATCH].long()
            cx = (events[:, EV_X] / self.cell).long().clamp(0, self.gw - 1)
            cy = (events[:, EV_Y] / self.cell).long().clamp(0, self.gh - 1)
            idx = b * self.n_cells + cy * self.gw + cx
            acc = acc.reshape(B * self.n_cells, D)
            cnt = cnt.reshape(B * self.n_cells, 1)
            acc.index_add_(0, idx, tok)
            cnt.index_add_(0, idx, torch.ones(idx.shape[0], 1, device=tok.device, dtype=tok.dtype))
            acc = acc.view(B, self.n_cells, D)
            cnt = cnt.view(B, self.n_cells, 1)
        feat = torch.cat([acc / cnt.clamp_min(1.0), cnt.clamp_max(64.0) / 64.0], dim=-1)
        mixed = self.cell_mlp(feat)                                       # (B, G, H)
        occupied = (cnt.squeeze(-1) > 0).to(mixed.dtype).unsqueeze(-1)
        mixed = mixed * occupied
        denom = occupied.sum(1).clamp_min(1.0)
        mean = mixed.sum(1) / denom
        peak = mixed.max(1).values
        return self.proj(torch.cat([mean, peak], dim=-1))


def build_encoder(kind: str, **kw) -> nn.Module:
    kind = (kind or "none").lower()
    if kind in ("raw_scan", "scan", "raw"):
        return RawEventEncoder(**kw)
    if kind in ("sparse_cell", "fallback", "cell"):
        return SparseCellEncoder(**kw)
    raise ValueError(f"unknown encoder {kind!r}; expected raw_scan or sparse_cell")
