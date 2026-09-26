#!/usr/bin/env python3
r"""X1. A sparse spatial hierarchy over the packet's events that cannot read the state.

Why this module exists
----------------------
S37's nodes see about 2-3 ms of events each (k nearest among the 32 causal predecessors in time
order) and nothing in its graph reaches hand scale, so the arm has no estimate of the pose that is
independent of the previous state: once a closed-loop error leaves the association basin nothing
brings it back (`research_state/lit/CANDIDATES.md` §1, E2/E3/E4). X1 asks the one question that
decides whether a sparse method can supply that missing measurement: does an encoder that sees only
the current packet, has no input through which a state could enter, and aggregates spatially up to
the whole hand, give an absolute 51D pose close to the dense CNN's? This file is that encoder and
nothing else; the pose head stays in `MNISTModel`, as for every frontend here.

Structure
---------
* **Nodes are S37's nodes.** The seven-scalar `event_tokens` and the same uniform-stride sample to
  at most `max_nodes` per packet (the sampler is `EventGNN._sample` itself), static `(B, N0)` shape
  plus a live mask.
* **Distances are in pixels.** `(x_px, y_px, t_norm * t_px)`: one packet-length of time is worth
  `t_px` pixels (default 10), so neighbourhoods are spatial with a mild preference for events close
  in time. Edge features are `dp = (dx / 16, dy / 16, dt_norm)`.
* **Level 0** (`N0`, width `C = hidden`): `Linear(7 -> C) + ReLU`, then `n_layers` residual
  `EdgeConv` rounds on the k nearest *live* nodes of the whole packet (not a temporal window).
* **Set abstraction x3** (`N0 -> N0/4 -> N0/16 -> N0/64`, widths `2C, 4C, 8C`): centroids are every
  fourth node of the previous level in time order; each centroid groups its k nearest live nodes of
  the previous level and max-pools `MLP([h_j, (p_j - p_c) / r_l])`. Levels 1 and 2 add one residual
  `EdgeConv` among their own nodes.
* **Readout**: masked mean and max over level 3, `Linear(16C -> feat) + ReLU + Linear(feat -> feat)`.

Constraints the design keeps, and why
-------------------------------------
* **No state.** There is no `prev` argument and `extra` node channels are refused: a rendered prev at
  each event would turn the "absolute" branch into another conditional one, which is exactly the
  property X1 is meant to exclude.
* **Nothing grows with the canvas.** Every neighbour search is a `cdist` among at most `max_nodes`
  nodes of one packet, chunked over the batch; no tensor is `H x W`. Memory and compute follow the
  node count, which the sampler caps, not the sensor resolution.
* **Static shapes.** Centroids are a fixed stride of the node list rather than farthest-point
  sampling, groups are kNN rather than ball queries, and dead nodes are masked instead of dropped, so
  one batch size is one set of shapes (the precondition for a CUDA graph later).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from .encoder import TOKEN_DIM, TOKEN_NAMES, event_tokens
from .event_gnn import EdgeConv, EventGNN, gather_node_features
from .events import EV_X, EV_Y

#: which token column carries the in-packet normalised timestamp
T_COL = TOKEN_NAMES.index("t_norm")


def knn(query: torch.Tensor, q_mask: torch.Tensor, source: torch.Tensor, s_mask: torch.Tensor,
        k: int, exclude_self: bool, chunk: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """The `k` nearest live `source` nodes of every `query` node, per packet. Indices only.

    Squared distances are summed from exact per-axis coordinate differences, not from the
    `|a|^2 + |b|^2 - 2ab` expansion: at pixel coordinates in the thousands the expansion cancels
    catastrophically, so a translated or re-framed packet would get a different neighbour set. With
    differences, an integer translation leaves every distance -- and therefore every index -- bitwise
    unchanged. (`torch.cdist(..., "donot_use_mm_for_euclid_dist")` computes the same differences but
    took 90 % of the training forward at 512 packets.)

    The `(chunk, Nq, Ns)` distance block is the largest tensor of the module, so the batch is walked
    in chunks under `no_grad` with autocast off (indices are not differentiable, and bf16 distances
    would tie neighbours that are not tied).

    Returns `(idx (B, Nq, kk) long, valid (B, Nq, kk) bool)`, `kk = min(k, Ns)`. A slot is invalid
    when it points at a dead source, at the query itself (`exclude_self`, which requires the query
    and source sets to be the same list), or belongs to a dead query; invalid slots point at index 0
    so every gather stays in range, and they are masked downstream.
    """
    B, Nq, _ = query.shape
    Ns = source.shape[1]
    kk = min(int(k), Ns)
    with torch.no_grad(), torch.autocast(device_type=query.device.type, enabled=False):
        q, s = query.float(), source.float()
        idx_parts, dist_parts = [], []
        for b0 in range(0, B, max(int(chunk), 1)):
            b1 = min(B, b0 + max(int(chunk), 1))
            qc, sc = q[b0:b1], s[b0:b1]
            d = (qc[:, :, None, 0] - sc[:, None, :, 0]).square_()
            for c in range(1, qc.shape[-1]):
                d += (qc[:, :, None, c] - sc[:, None, :, c]).square_()
            d.masked_fill_(~s_mask[b0:b1].unsqueeze(1), float("inf"))
            if exclude_self:
                d.diagonal(dim1=1, dim2=2).fill_(float("inf"))
            near = d.topk(kk, dim=-1, largest=False)
            idx_parts.append(near.indices)
            dist_parts.append(near.values)
        idx = torch.cat(idx_parts, 0)
        valid = torch.isfinite(torch.cat(dist_parts, 0)) & q_mask.unsqueeze(-1)
        idx = idx.masked_fill(~valid, 0)
    return idx, valid


class SetAbstraction(nn.Module):
    r"""One PointNet++ set-abstraction step with kNN groups:

        f_c = max_{j in kNN(c)} MLP([h_j ; (p_j - p_c) / r])

    The MLP ends in a ReLU, so every message is non-negative and zero-filling the invalid slots before
    the max is an exact masked max; a centroid without a single valid member comes out exactly zero.
    `r` only normalises the relative position into O(1): with kNN groups there is no ball radius.
    """

    def __init__(self, c_in: int, c_out: int, radius_px: float):
        super().__init__()
        self.radius_px = float(radius_px)
        self.mlp = nn.Sequential(
            nn.Linear(c_in + 3, c_out), nn.ReLU(inplace=True),
            nn.Linear(c_out, c_out), nn.ReLU(inplace=True),
        )

    def forward(self, h: torch.Tensor, pos: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor,
                centre: torch.Tensor) -> torch.Tensor:
        """`h (B, Np, C)`, `pos (B, Np, 3)` of the previous level; `idx / valid (B, Nc, kk)` its
        members per centroid; `centre (B, Nc, 3)`. Returns `(B, Nc, c_out)`."""
        B, Nc, kk = idx.shape
        C = h.shape[-1]
        flat = idx.reshape(B, Nc * kk, 1)
        hj = gather_node_features(h, idx)
        pj = pos.gather(1, flat.expand(B, Nc * kk, 3)).reshape(B, Nc, kk, 3)
        rel = ((pj - centre.unsqueeze(2)) * (1.0 / self.radius_px)).to(h.dtype)
        m = self.mlp(torch.cat([hj, rel], dim=-1))
        m = m * valid.unsqueeze(-1).to(m.dtype)
        # `max(...).values` rather than `amax`: its backward keeps only the argmax indices, not a
        # second copy of the (B, Nc, kk, c_out) message tensor.
        return m.max(dim=2).values


class EventHierEncoder(nn.Module):
    """`(events, ptr, delta_t_s[, extra])` in, one feature vector per packet out.

    Same call contract as `EventGNN` (an event-free packet returns exactly zero, so a downstream gate
    sees "no update"), but no node hand-off: `return_nodes` is refused, because which level's nodes a
    routed readout should read is a design question of its own and not something to default here.
    """

    #: every set abstraction keeps one node in this many, in time order
    STRIDE = 4
    #: number of set-abstraction steps; the level widths are hidden * (1, 2, 4, 8)
    N_SA = 3
    #: pixels per unit of the `dx`, `dy` edge features (the spec's `dp = (dx/16, dy/16, dt_norm)`)
    EDGE_PX = 16.0
    #: relative-position normaliser of SA1..SA3. SA_l groups from level l-1, whose node density is
    #: 4^(l-1) times lower than level 0's, so in a 2-D event set its kNN radius grows about 2x per
    #: level; doubling r keeps the relative positions O(1). SA1 uses the edge scale.
    SA_RADIUS_PX = (16.0, 32.0, 64.0)

    def __init__(self, height: int = 180, width: int = 240, hidden: int = 64,
                 feat_dim: int = 512, k: int = 16, n_layers: int = 2, max_nodes: int = 2048,
                 t_px: float = 10.0, extra_channels: int = 0, knn_chunk: int = 32):
        super().__init__()
        if int(extra_channels):
            raise ValueError("event_hier is the state-free absolute branch: it takes no rendered "
                             "node channels (set MODEL.PREV_RENDER: false)")
        if not float(t_px) > 0.0:
            raise ValueError(f"t_px must be positive (it divides dt in the edge features), "
                             f"got {t_px}")
        self.height, self.width = int(height), int(width)
        self.hidden, self.feat_dim = int(hidden), int(feat_dim)
        self.k, self.n_layers = int(k), int(n_layers)
        self.max_nodes = int(max_nodes)
        #: how many pixels one packet-length of time is worth in the neighbour metric
        self.t_px = float(t_px)
        #: packets per `cdist` block in the neighbour search (memory, not semantics)
        self.knn_chunk = int(knn_chunk)
        #: token width, read by tools that build a matching `extra` (always zero columns here)
        self.in_dim = TOKEN_DIM
        widths = [self.hidden * (2 ** i) for i in range(self.N_SA + 1)]
        self.widths = tuple(widths)
        self.embed = nn.Linear(TOKEN_DIM, widths[0])
        self.level0 = nn.ModuleList([EdgeConv(widths[0], compact_gather=True) for _ in range(self.n_layers)])
        self.sa = nn.ModuleList([SetAbstraction(widths[i], widths[i + 1], self.SA_RADIUS_PX[i])
                                 for i in range(self.N_SA)])
        # one EdgeConv round on levels 1 .. N_SA-1; the last level goes straight to the readout
        self.level_convs = nn.ModuleList([EdgeConv(widths[i + 1], compact_gather=True) for i in range(self.N_SA - 1)])
        self.proj = nn.Sequential(
            nn.Linear(2 * widths[-1], self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim),
        )

    # ------------------------------------------------------------------ nodes
    def _sample(self, events: torch.Tensor, ptr: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """S37's node set, bitwise: `EventGNN._sample` reads nothing but `self.max_nodes`."""
        return EventGNN._sample(self, events, ptr)

    def level_sizes(self) -> Tuple[int, ...]:
        """Static node count of every level, `(N0, N1, N2, N3)`."""
        sizes = [self.max_nodes]
        for _ in range(self.N_SA):
            sizes.append(-(-sizes[-1] // self.STRIDE))
        return tuple(sizes)

    def _edge_dp(self, pos: torch.Tensor, idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """`(dx / 16, dy / 16, dt_norm)` from every node to each neighbour, zero on invalid slots."""
        B, N, kk = idx.shape
        pj = pos.gather(1, idx.reshape(B, N * kk, 1).expand(B, N * kk, 3)).reshape(B, N, kk, 3)
        d = pj - pos.unsqueeze(2)
        dp = torch.cat([d[..., :2] * (1.0 / self.EDGE_PX), d[..., 2:] * (1.0 / self.t_px)], dim=-1)
        return dp * valid.unsqueeze(-1).to(dp.dtype)

    def _graph(self, pos: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`(idx, dp, emask)` of one level: the k nearest live nodes of the same level, the node
        itself excluded (its own state rides the EdgeConv residual). Built once per level, since the
        positions do not change between that level's EdgeConv rounds."""
        idx, valid = knn(pos, mask, pos, mask, self.k, exclude_self=True, chunk=self.knn_chunk)
        return idx, self._edge_dp(pos, idx, valid).to(dtype), valid.to(dtype)

    # ---------------------------------------------------------------- forward
    def forward(self, events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor,
                extra: Optional[torch.Tensor] = None, return_nodes: bool = False) -> torch.Tensor:
        """Packet feature vector, `(B, feat_dim)`; exactly zero for a packet without events."""
        if return_nodes:
            raise NotImplementedError("event_hier has no node hand-off; it is a pooled encoder")
        if extra is not None and extra.ndim == 2 and extra.shape[-1] > 0:
            raise ValueError("event_hier reads no extra node channels (it must stay state-free)")
        B = int(ptr.numel() - 1)
        dev = events.device
        if events.shape[0] == 0:
            return torch.zeros(B, self.feat_dim, device=dev, dtype=self.embed.weight.dtype)

        tok = event_tokens(events, ptr, delta_t_s, self.height, self.width)
        src, mask = self._sample(events, ptr)
        N = mask.shape[1]
        flat = src.reshape(-1)
        m0 = mask.unsqueeze(-1)
        feat = tok[flat].reshape(B, N, TOKEN_DIM) * m0
        ev = events[flat].reshape(B, N, -1)
        # Neighbour metric in pixels. Dead slots are zeroed; they never become neighbours because
        # the search masks dead sources, and they never become centroids of a live group.
        pos = torch.stack([ev[..., EV_X], ev[..., EV_Y], feat[..., T_COL] * self.t_px],
                          dim=-1).to(torch.float32) * m0

        h = torch.relu(self.embed(feat)) * m0          # bool mask: keeps the autocast dtype
        idx, dp, emask = self._graph(pos, mask, h.dtype)
        for layer in self.level0:
            h = (h + layer(h, idx, dp, emask)) * m0

        for i, sa in enumerate(self.sa):
            # Centroids: every STRIDE-th node of the previous level. The live nodes are a prefix of
            # every level (the sampler fills slots in order), so node 0 of a packet with at least one
            # event is live at every level and no level of a non-empty packet is empty.
            centre = pos[:, ::self.STRIDE]
            c_mask = mask[:, ::self.STRIDE]
            gidx, gvalid = knn(centre, c_mask, pos, mask, self.k, exclude_self=False,
                               chunk=self.knn_chunk)
            h = sa(h, pos, gidx, gvalid, centre)
            pos, mask = centre, c_mask
            h = h * mask.unsqueeze(-1)
            if i < len(self.level_convs):
                idx, dp, emask = self._graph(pos, mask, h.dtype)
                h = (h + self.level_convs[i](h, idx, dp, emask)) * mask.unsqueeze(-1)

        # h >= 0 everywhere (ReLU messages, masked max, residual sums of ReLU means) and dead nodes
        # are exactly 0, so the plain max over nodes is the masked max.
        any_node = mask.any(1, keepdim=True).to(h.dtype)
        live = mask.sum(1, keepdim=True).clamp_min(1).to(h.dtype)
        mean = h.sum(1) / live
        peak = h.max(1).values
        out = self.proj(torch.cat([mean, peak], dim=-1))
        # An event-free packet inside a non-empty batch must also return exactly zero.
        return out * any_node
