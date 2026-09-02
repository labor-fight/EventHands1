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

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim + EDGE_DIM, dim)

    def forward(self, h: torch.Tensor, idx: torch.Tensor, dp: torch.Tensor,
                emask: torch.Tensor) -> torch.Tensor:
        B, N, C = h.shape
        k = idx.shape[-1]
        hj = h.gather(1, idx.reshape(B, N * k, 1).expand(B, N * k, C)).reshape(B, N, k, C)
        m = torch.relu(self.lin(torch.cat([hj - h.unsqueeze(2), dp], dim=-1)))
        m = m * emask.unsqueeze(-1)
        return m.sum(2) / emask.sum(2, keepdim=True).clamp_min(1.0)


class EventGNN(nn.Module):
    """`(events, ptr, delta_t_s[, extra[, queries]])` in, one feature vector per packet out;
    with `queries` also one readout feature per query point (S38 joint readout).

    The pose head stays in `MNISTModel`: this module may not contain a second path to a finger
    angle, same contract as every other frontend here.
    """

    #: initial spatial kernel width of the S38 joint readout, in pixels: half a finger
    #: length, so a query sees its own digit and not the whole hand
    QUERY_SIGMA0_PX = 16.0

    def __init__(self, height: int = 180, width: int = 240, hidden: int = 96,
                 feat_dim: int = 512, k: int = 8, n_layers: int = 3,
                 max_nodes: int = 768, window: int = 32, t_scale: float = 1.0,
                 extra_channels: int = 0, joint_queries: int = 0, attn_pool: int = 0):
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.hidden, self.feat_dim = int(hidden), int(feat_dim)
        self.k, self.window = int(k), int(window)
        self.max_nodes = int(max_nodes)
        #: how many pixels one normalised time unit is worth when measuring "near"
        self.t_scale = float(t_scale)
        self.in_dim = TOKEN_DIM + int(extra_channels)
        self.embed = nn.Linear(self.in_dim, self.hidden)
        self.layers = nn.ModuleList([EdgeConv(self.hidden) for _ in range(int(n_layers))])
        # S41. Content-based attention pooling: `attn_pool` learned query vectors replace
        # mean+max as the readout. The selection weights depend on the node *features* alone
        # -- the queries are constants -- so this is on the legal side of the S38/KEG law
        # (the previous state may not steer evidence selection; here it cannot). Queries are
        # zero-init: every head starts as exact mean pooling and sharpens only if the loss
        # asks it to, so the arm deforms continuously from the baseline readout.
        self.attn_pool = int(attn_pool)
        pooled = (self.attn_pool if self.attn_pool else 2) * self.hidden
        if self.attn_pool:
            self.pool_q = nn.Parameter(torch.zeros(self.attn_pool, self.hidden))
            self.pool_key = nn.Linear(self.hidden, self.hidden)
        self.proj = nn.Sequential(
            nn.Linear(pooled, self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim),
        )
        # S38. One spatial-kernel readout per query point (the projected LBS joints of the
        # previous state). The parameter is created only when asked for: an always-present
        # but sometimes-unused parameter breaks DDP's gradient contract and changes every
        # older checkpoint's key set.
        self.joint_queries = int(joint_queries)
        if self.joint_queries:
            self.query_log_sigma = nn.Parameter(
                torch.full((self.joint_queries,), math.log(self.QUERY_SIGMA0_PX)))

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

    # --------------------------------------------------------------- readout
    def _query_readout(self, h: torch.Tensor, px: torch.Tensor, py: torch.Tensor,
                       mask: torch.Tensor, any_node: torch.Tensor,
                       queries: torch.Tensor) -> torch.Tensor:
        """S38. One feature per query point: a softmax spatial kernel over the nodes.

        `w_ij = softmax_i(-d^2(node_i, query_j) / 2 sigma_j^2)` with a learnable per-query
        sigma. This is where -- and only where -- the previous state is allowed to steer
        aggregation: the graph and the messages upstream never saw it as structure (the KEG
        law), and the readout is the measured bottleneck (nodes 512 -> 4096 moved recursive
        RA by 0.09 mm while the pooled vector stayed 512-D).

        Dead nodes are filled with a finite -1e4 rather than -inf: an event-free packet then
        takes a uniform softmax over rows of `h` that the mask already zeroed, so the readout
        is exactly zero without a NaN path, and `ZERO_EVENT_GATE` keeps its contract.
        """
        sig2 = (self.query_log_sigma.exp() ** 2).view(1, -1, 1).to(torch.float32)
        d2 = ((px.unsqueeze(1) - queries[..., 0].unsqueeze(-1)).square()
              + (py.unsqueeze(1) - queries[..., 1].unsqueeze(-1)).square())      # (B, J, N)
        logit = (-0.5 * d2 / sig2).masked_fill(~mask.unsqueeze(1), -1e4)
        w = torch.softmax(logit, dim=-1).to(h.dtype)
        return torch.bmm(w, h) * any_node.unsqueeze(-1)

    # ---------------------------------------------------------------- forward
    def forward(self, events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor,
                extra: Optional[torch.Tensor] = None,
                queries: Optional[torch.Tensor] = None):
        """Packet feature vector; with `queries` (B, J, 2) pixel points, also (B, J, hidden)."""
        B = int(ptr.numel() - 1)
        dev = events.device
        if queries is not None and not self.joint_queries:
            raise RuntimeError("queries passed but the module was built without joint_queries")
        if events.shape[0] == 0:
            z = torch.zeros(B, self.feat_dim, device=dev, dtype=self.embed.weight.dtype)
            if queries is None:
                return z
            return z, torch.zeros(B, queries.shape[1], self.hidden, device=dev, dtype=z.dtype)

        tok = event_tokens(events, ptr, delta_t_s, self.height, self.width)
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
        if self.attn_pool:
            # (B, K, N) logits; dead nodes get a finite -1e4 (same recipe as the S38
            # readout) so an event-free packet is a uniform softmax over zeroed rows
            # and the readout is exactly zero without a NaN path.
            logit = torch.einsum("qc,bnc->bqn", self.pool_q.to(h.dtype),
                                 self.pool_key(h)) / math.sqrt(self.hidden)
            logit = logit.masked_fill(~mask.unsqueeze(1), -1e4)
            w = torch.softmax(logit, dim=-1)
            pooled = torch.bmm(w, h).reshape(h.shape[0], -1)
        else:
            live = mask.sum(1, keepdim=True).clamp_min(1.0).to(h.dtype)
            mean = h.sum(1) / live
            peak = h.masked_fill(~mask.unsqueeze(-1), -1e4).max(1).values
            pooled = torch.cat([mean, peak * any_node], dim=-1)
        out = self.proj(pooled)
        # An event-free packet must return exactly zero so `MODEL.ZERO_EVENT_GATE` gates an update
        # that was already nothing.
        out = out * any_node
        if queries is None:
            return out
        nodes = self._query_readout(
            h, ev[..., EV_X].to(torch.float32) * mask,
            ev[..., EV_Y].to(torch.float32) * mask, mask, any_node,
            queries.to(torch.float32))
        return out, nodes
