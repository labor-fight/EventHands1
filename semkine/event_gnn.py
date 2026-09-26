#!/usr/bin/env python3
r"""S36. AEGNN on the events themselves, in the one form this repo's measured negatives allow.

What is borrowed from AEGNN (Schaefer et al., CVPR'22, `uzh-rpg/aegnn`)
-----------------------------------------------------------------------
* **The events are the nodes.** Not pixel tiles, not MANO joints. `CellGNN` called itself a graph
  and was a 3x3 convolution with two free taps (S35); this module has a data-dependent adjacency,
  so the word means something here.
* **Edges carry relative geometry.** AEGNN's `SplineConv` makes the filter a function of the
  relative position of the two nodes. Here the same information enters as an explicit edge feature
  `dp = (dx, dy, dt)` fed to the message MLP, which is the cheap version of the same idea and keeps
  the operator anisotropic -- a neighbour above and a neighbour to the left are not interchangeable.
* **Causal edges only.** Every edge runs from an *earlier* event to a later one, which is the
  precondition for AEGNN's real claim: a new event only recomputes the sub-graph it touched.

What is deliberately *not* borrowed, and why
--------------------------------------------
* **No `torch.cdist` over the packet.** `AEGNNLite` did that and is a settled structural NO-GO:
  `O(N^2)` with up to 29 000 events in a 50 ms window is 8.4e8 pairwise distances per sample.
  Neighbours are searched in a *temporal window* of `window` preceding events instead, which is
  `O(N * window)` and exact whenever the true neighbour radius is inside that window -- events far
  apart in time are never spatio-temporal neighbours once `t` is normalised.
* **No detection head, no voxelisation, no async runtime.** `docs/...MASTER_VERDICT...` is explicit
  that latency may not be claimed without a runtime whose prefix output matches the synchronous
  one. This module makes async *possible* (causal edges) and claims nothing.
* **No state-conditioned graph.** The adjacency is built from `(x, y, t)` alone. KEG routed events
  to joints using the previous pose and its loop gain went to 0.910 against the dense arm's 0.449;
  the registered design law is that evidence *structure* stays state-free and the state enters as
  values and at the readout. So the previous render rides along as two extra node channels sampled
  at each event's own pixel -- comparative, exactly as in the arm that actually tracks -- and never
  as a choice of where an event is binned.

What is kept from the tracking recipe, unchanged
------------------------------------------------
Everything outside this file: `PREDICT_DELTA`, `PREVPOS_EMBED`, `PREV_RENDER`, `ZERO_EVENT_GATE`,
the S10 active heads and the low-event-rate domain randomisation that moved recursive RA from
19.26 mm to 12.77 mm. This module only replaces what produces the packet feature vector, so a win
or a loss is attributable to the graph and not to the loop around it.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
from torch import nn

from .encoder import TOKEN_DIM, TOKEN_NAMES, event_tokens
from .events import EV_X, EV_Y

#: (dx, dy, dt) on every edge.
EDGE_DIM = 3
#: which token column carries the in-packet normalised timestamp
T_COL = TOKEN_NAMES.index("t_norm")


def gather_node_features(h: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather whole feature rows without expanding node indices over channels.

    In deterministic CUDA backward, expanded ``gather`` indices can materialize
    coordinates for every channel. ``index_select`` keeps channels as a slice,
    while selecting exactly the same rows (including repeated neighbours).
    """
    B, N, C = h.shape
    offset = torch.arange(B, device=idx.device).reshape(B, *([1] * (idx.ndim - 1))) * N
    rows = (idx + offset).reshape(-1)
    values = h.reshape(B * N, C)
    # CUDA's deterministic index-select backward otherwise rounds each repeated
    # addition in BF16/FP16. Accumulate in FP32 and cast once, as the original
    # expanded gather's scalar reduction does. Forward selected values are exact.
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    return values.index_select(0, rows).to(h.dtype).reshape(*idx.shape, C)


class EdgeConv(nn.Module):
    r"""One round of message passing with the edge geometry in the message.

        m_ij = relu( W [ h_j - h_i ; dp_ij ] ) ,   h_i <- h_i + mean_{j in N(i)} m_ij

    `h_j - h_i` is DGCNN's local difference and `dp_ij` is AEGNN's relative position. `dp` is what
    keeps the operator anisotropic -- without it a neighbour above and a neighbour to the left are
    interchangeable, which is exactly the degeneracy that made `CellGNN` a convolution.

    The node's own state is carried by the residual rather than by a third block in the concat.
    That is not cosmetic: the per-edge tensor is the memory ceiling of this module, and `C + 3`
    instead of `2C + 3` is what let `ENCODER_MAX_NODES` rise past the point where the wider form
    ran out of 44 GiB.
    """

    def __init__(self, dim: int, compact_gather: bool = False):
        super().__init__()
        self.lin = nn.Linear(dim + EDGE_DIM, dim)
        self.compact_gather = bool(compact_gather)

    def forward(self, h: torch.Tensor, idx: torch.Tensor, dp: torch.Tensor,
                emask: torch.Tensor) -> torch.Tensor:
        B, N, C = h.shape
        k = idx.shape[-1]
        if self.compact_gather:
            hj = gather_node_features(h, idx)
        else:
            hj = h.gather(1, idx.reshape(B, N * k, 1).expand(B, N * k, C)).reshape(B, N, k, C)
        m = torch.relu(self.lin(torch.cat([hj - h.unsqueeze(2), dp], dim=-1)))
        m = m * emask.unsqueeze(-1)
        return m.sum(2) / emask.sum(2, keepdim=True).clamp_min(1.0)


class EventGNN(nn.Module):
    """`(events, ptr, delta_t_s[, extra])` in, one feature vector per packet out.

    The pose head stays in `MNISTModel`: this module may not contain a second path to a finger
    angle, same contract as every other frontend here.
    """

    #: node attribute sets. `token7` is the S36 token (x, y, p, t, log inter-event dt, two
    #: surface-of-active-events ages); `raw4` is the event itself, (x, y, p, t), nothing derived.
    NODE_ATTRS = {"token7": TOKEN_DIM, "raw4": 4}
    #: per-node edge summary width: mean (dx, dy, dt) over the node's edges and their mean length
    EDGE_SUMMARY_DIM = 4

    def __init__(self, height: int = 180, width: int = 240, hidden: int = 96,
                 feat_dim: int = 512, k: int = 8, n_layers: int = 3,
                 max_nodes: int = 768, window: int = 32, t_scale: float = 1.0,
                 extra_channels: int = 0, node_attrs: str = "token7", readout: bool = True):
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.hidden, self.feat_dim = int(hidden), int(feat_dim)
        self.k, self.window = int(k), int(window)
        self.max_nodes = int(max_nodes)
        #: how many pixels one normalised time unit is worth when measuring "near"
        self.t_scale = float(t_scale)
        if node_attrs not in self.NODE_ATTRS:
            raise ValueError(f"node_attrs must be one of {sorted(self.NODE_ATTRS)}, got {node_attrs!r}")
        self.node_attrs = node_attrs
        self.in_dim = self.NODE_ATTRS[node_attrs] + int(extra_channels)
        self.embed = nn.Linear(self.in_dim, self.hidden)
        self.layers = nn.ModuleList([EdgeConv(self.hidden) for _ in range(int(n_layers))])
        # mean + max readout. Not built when the consumer reads the nodes directly (S37 mesh
        # query): DDP runs with find_unused_parameters=False, so an unused head is an error.
        self.readout = bool(readout)
        if self.readout:
            self.proj = nn.Sequential(
                nn.Linear(2 * self.hidden, self.feat_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.feat_dim, self.feat_dim),
            )

    # ------------------------------------------------------------------ nodes
    def _sample(self, events: torch.Tensor, ptr: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniform-stride subsample to at most `max_nodes` per packet.

        Stride rather than a random or head-of-packet cut: a packet spans up to 300 ms and the
        readout has to see the whole window, so dropping the tail would silently shorten it. The
        cap is what keeps memory flat as the event rate rises, which is the property `AEGNNLite`
        did not have.
        """
        dev = events.device
        N = self.max_nodes
        counts = (ptr[1:] - ptr[:-1]).clamp(min=0)
        n = counts.clamp(max=N)
        pos = torch.arange(N, device=dev)
        mask = pos.unsqueeze(0) < n.unsqueeze(1)                       # (B, N)
        step = counts.to(torch.float32) / n.clamp(min=1).to(torch.float32)
        src = ptr[:-1].unsqueeze(1) + (pos.unsqueeze(0).to(torch.float32)
                                       * step.unsqueeze(1)).long()
        src = torch.minimum(src, (ptr[1:] - 1).clamp(min=0).unsqueeze(1))
        return src.clamp(min=0), mask

    # ------------------------------------------------------------------ graph
    def _edges(self, p: torch.Tensor, mask: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """k nearest earlier neighbours inside a window of `window` preceding events.

        Returns `(idx, dp, emask)`. Candidates are positions `i-1 .. i-window` of the *time-sorted*
        node list, so an edge can only ever point from past to future and the cost is
        `O(N * window)` instead of the `O(N^2)` that made `AEGNNLite` unusable.
        """
        B, N, _ = p.shape
        dev = p.device
        off = torch.arange(1, self.window + 1, device=dev)
        cand = torch.arange(N, device=dev).unsqueeze(1) - off.unsqueeze(0)   # (N, window)
        ok = cand >= 0
        cand = cand.clamp(min=0)
        cand_b = cand.unsqueeze(0).expand(B, N, self.window)
        pj = p.gather(1, cand_b.reshape(B, N * self.window, 1).expand(B, N * self.window, 3)
                      ).reshape(B, N, self.window, 3)
        d = (pj - p.unsqueeze(2)).pow(2).sum(-1)
        valid = ok.unsqueeze(0) & mask.gather(1, cand_b.reshape(B, -1)).reshape(
            B, N, self.window) & mask.unsqueeze(-1)
        d = d.masked_fill(~valid, float("inf"))
        kk = min(self.k, self.window)
        near = d.topk(kk, dim=-1, largest=False)
        idx = cand_b.gather(2, near.indices)
        emask = torch.isfinite(near.values).to(p.dtype)
        dp = (p.gather(1, idx.reshape(B, N * kk, 1).expand(B, N * kk, 3)).reshape(B, N, kk, 3)
              - p.unsqueeze(2)) * emask.unsqueeze(-1)
        return idx, dp, emask

    # ---------------------------------------------------------------- forward
    def forward(self, events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor,
                extra: Optional[torch.Tensor] = None, return_nodes: bool = False):
        """Packet feature vector, `(B, feat_dim)` (`None` when built without a readout).

        With `return_nodes`, also the per-node features after message passing, a per-node summary
        of the edges that fed them, and where the nodes sit:
        `(out, h (B, N, hidden), g (B, N, 4), px (B, N), py (B, N), mask (B, N))`, pixels in the
        event frame. This is the hand-off for a readout that anchors the previous state's geometry
        to the graph (S37): the graph and the features are computed exactly as without the flag --
        the state has not entered yet -- and the pooled `out` is bitwise the S36 readout.
        """
        B = int(ptr.numel() - 1)
        dev = events.device
        wdtype = self.embed.weight.dtype
        if events.shape[0] == 0:
            z = torch.zeros(B, self.feat_dim, device=dev, dtype=wdtype) if self.readout else None
            if return_nodes:
                e = torch.zeros(B, 0, device=dev, dtype=torch.float32)
                return (z, torch.zeros(B, 0, self.hidden, device=dev, dtype=wdtype),
                        torch.zeros(B, 0, self.EDGE_SUMMARY_DIM, device=dev, dtype=torch.float32),
                        e, e, torch.zeros(B, 0, device=dev, dtype=torch.bool))
            return z

        tok = event_tokens(events, ptr, delta_t_s, self.height, self.width)
        if self.node_attrs == "raw4":
            tok = tok[:, :4]                      # (x_norm, y_norm, polarity, t_norm), T_COL kept
        if extra is not None and extra.shape[0]:
            tok = torch.cat([tok, extra], dim=-1)

        src, mask = self._sample(events, ptr)
        N = mask.shape[1]
        flat = src.reshape(-1)
        feat = tok[flat].reshape(B, N, self.in_dim) * mask.unsqueeze(-1)
        # Normalised coordinates the graph is built in. `t_norm` already lives in [0, 1] within the
        # packet; `t_scale` says how many frame-widths one packet-length of time is worth.
        ev = events[flat].reshape(B, N, -1)
        p = torch.stack([ev[..., EV_X] / self.width,
                         ev[..., EV_Y] / self.height,
                         feat[..., T_COL] * self.t_scale], dim=-1) * mask.unsqueeze(-1)

        idx, dp, emask = self._edges(p.to(torch.float32), mask)
        h = torch.relu(self.embed(feat)) * mask.unsqueeze(-1)
        for layer in self.layers:
            h = (h + layer(h, idx, dp.to(h.dtype), emask.to(h.dtype))) * mask.unsqueeze(-1)

        any_node = mask.any(1, keepdim=True).to(h.dtype)
        out = None
        if self.readout:
            live = mask.sum(1, keepdim=True).clamp_min(1.0).to(h.dtype)
            mean = h.sum(1) / live
            peak = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
            out = self.proj(torch.cat([mean, peak * any_node], dim=-1))
            # An event-free packet must return exactly zero so `MODEL.ZERO_EVENT_GATE` gates an
            # update that was already nothing.
            out = out * any_node
        if return_nodes:
            # Edge summary per node: the mean relative position of the events it read (in graph
            # coordinates: frame widths / heights, packet-normalised time) and their mean length.
            # This is the graph's own local-motion reading, handed to the readout unpooled.
            n_e = emask.sum(-1, keepdim=True).clamp_min(1.0)                         # (B, N, 1)
            dp32 = dp.to(torch.float32)
            g = torch.cat([(dp32 * emask.unsqueeze(-1)).sum(2) / n_e,
                           (dp32.norm(dim=-1) * emask).sum(-1, keepdim=True) / n_e], dim=-1)
            g = g * mask.unsqueeze(-1)
            return (out, h, g, ev[..., EV_X].to(torch.float32) * mask,
                    ev[..., EV_Y].to(torch.float32) * mask, mask)
        return out
