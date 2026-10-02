#!/usr/bin/env python3
r"""S38. A sparse multi-scale event encoder: submanifold sparse convolutions on the packet's active sites.

Why (docs/S38_ROOT_TRACKING_VERDICT.md, diagnosis)
--------------------------------------------------
The S37 event graph does not measure absolute hand orientation: its pooled feature decodes the root
rotation no better than an event histogram (13-15 deg), and trained end to end on the absolute target
(x1001 E7) it still reaches only 13-14 deg, where ResNet18 on LNES reaches 9-10 deg on the same
budget (docs/S37_ROOT_TRACKING_VERDICT.md section 2.1). Two properties of `EventGNN` explain that
without invoking capacity:

* every node sees the 8 nearest of the 32 *temporally* preceding sampled events (three hops span
  ~1.9 ms), and the next stage is the global mean / max pool -- no stage ever sees the layout of the
  hand at an intermediate scale;
* the 2048 nodes are a time-uniform stride over the events, i.e. proportional to event density: at
  the median zgz_global packet 3.1e4 events land on only 3.3e3 pixels, so fast edges are sampled
  many times and slow parts of the silhouette rarely.

What this module is
-------------------
* **Sites, not events.** Level 0 is the set of occupied `cell` x `cell` pixel cells of the packet
  (4 x 4 by default, at most 45 x 60). Each site is summarised by what LNES keeps per pixel -- the
  newest normalised time of each polarity -- plus log counts per polarity, the mean time, the mean
  sub-cell position of its events and the cell centre. Every event contributes; there is no node cap,
  and the site count is bounded by the sensor (349 at the median zgz_global packet of 3.1e4 events,
  84 at the median zgz_local packet).
* **Submanifold sparse convolution** (Graham, Engelcke & van der Maaten, CVPR'18): a 3 x 3 kernel
  evaluated only at occupied sites and reading only occupied neighbours, so the set of sites never
  grows. A 2 x 2 stride-2 sparse convolution builds each next level from its occupied children only.
  Three levels by default (stride `cell`, 2`cell`, 4`cell` = 4, 8, 16 px). This is the operator of the asynchronous
  sparse CNNs for events (Messikommer et al., ECCV'20): an incoming event changes one level-0 site and,
  per layer, only the sites inside the kernel of a changed site, so the structure admits per-event
  updates. As for S36 / S37, the module makes asynchronous updates possible and claims nothing about
  an asynchronous runtime.
* **No dense H x W tensor is allocated.** Neighbours are found by binary search over the sorted site
  keys (the hash table of a sparse-convolution library, as a sorted array).
* **Deterministic.** Sums are sorted segment reductions, never float atomics: the closed loop
  amplifies last-bit differences (the S36 latency audit measured 0.15 mm from a reordered sum).

Interface: the `EventGNN` one. `(events, ptr, delta_t_s)` -> `(B, feat_dim)`; with `return_nodes`
also the per-node features the S37 routed readout pools by joint. The nodes are the level-0 sites
(`cell` px cells), each carrying its own feature and those of all its ancestors, at the cell centre in
event-frame pixels. At batch 1 the cost is the number of operator launches, not arithmetic, so the
structure is kept to three levels and every integer index computation is shared or vectorised.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from .events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y

#: per-site input: log1p n_pos, log1p n_neg, newest t pos, newest t neg, mean t, mean sub-cell offset
#: (x, y), cell centre (x, y)
IN_DIM = 9


class _Level:
    """The occupied cells of one level as sorted unique keys over a grid with a one-cell empty border,
    `key = (b * (H + 2) + y + 1) * (W + 2) + x + 1`. Because the border is never occupied, the 3 x 3
    neighbours of every cell are `key + const` -- no bounds test, no wrap into the next row or packet --
    and a packet's cells are the key range `[b S, (b + 1) S)`, `S = (H + 2) (W + 2)`."""

    __slots__ = ("key", "H", "W", "_bxy", "nbr")

    def __init__(self, key: torch.Tensor, H: int, W: int):
        self.key, self.H, self.W = key, int(H), int(W)
        self._bxy = None
        self.nbr = None

    @staticmethod
    def make_key(b: torch.Tensor, y: torch.Tensor, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        return (b * (H + 2) + y + 1) * (W + 2) + x + 1

    def bxy(self):
        """`(b, y + 1, x + 1)`: packet and bordered-grid coordinates of every cell (decoded once)."""
        if self._bxy is None:
            S = (self.H + 2) * (self.W + 2)
            b = torch.div(self.key, S, rounding_mode="floor")
            r = self.key - b * S
            yq = torch.div(r, self.W + 2, rounding_mode="floor")
            self._bxy = (b, yq, r - yq * (self.W + 2))
        return self._bxy

    def segments(self, B: int) -> torch.Tensor:
        """Cells per packet, by binary search over the packet-major keys (no read-back to the host)."""
        S = (self.H + 2) * (self.W + 2)
        edges = torch.searchsorted(self.key, torch.arange(B + 1, device=self.key.device) * S)
        return edges[1:] - edges[:-1]

    def neighbours(self):
        """`(idx (M, 9), valid (M, 9, 1) float)`: the row of each 3 x 3 neighbour (row-major offsets)
        and whether that cell is occupied. An empty neighbour points at the site itself and is masked:
        pointing every hole at one shared zero row made that row the target of millions of atomic adds
        in the backward pass. All nine offsets go through one binary search (at batch 1 the cost is
        operator launches, not arithmetic)."""
        if self.nbr is None:
            M = self.key.numel()
            q = self.key.unsqueeze(1) + _offsets(self.W + 2, self.key.device)            # (M, 9)
            pos = torch.searchsorted(self.key, q).clamp_(max=max(M - 1, 0))
            hit = self.key[pos] == q
            own = torch.arange(M, device=pos.device).unsqueeze(1)
            self.nbr = (torch.where(hit, pos, own), hit.unsqueeze(-1).float())
        return self.nbr

    def parent(self):
        """The next level (stride 2) and, per site, `(row of its parent, slot 0..3 inside it)`."""
        Hp, Wp = (self.H + 1) // 2, (self.W + 1) // 2
        b, yq, xq = self.bxy()
        yq1, xq1 = yq + 1, xq + 1                       # bordered y + 2: parent row (y // 2) + 1 = yq1 // 2
        pk = (b * (Hp + 2) + torch.div(yq1, 2, rounding_mode="floor")) * (Wp + 2) \
            + torch.div(xq1, 2, rounding_mode="floor")
        key, inv = torch.unique(pk, sorted=True, return_inverse=True)
        return _Level(key, Hp, Wp), inv, (yq1 % 2) * 2 + (xq1 % 2)


_OFFSETS = {}


def _offsets(row: int, device) -> torch.Tensor:
    """The nine key offsets of a 3 x 3 neighbourhood on a grid `row` keys wide (cached per device)."""
    k = (int(row), str(device))
    if k not in _OFFSETS:
        _OFFSETS[k] = torch.tensor([dy * row + dx for dy in (-1, 0, 1) for dx in (-1, 0, 1)],
                                   dtype=torch.long, device=device)
    return _OFFSETS[k]


class SubmConv(nn.Module):
    """3 x 3 submanifold sparse convolution, BN over the active sites, ReLU; residual when the width
    is kept. The kernel is one `Linear` over the nine gathered neighbours (an empty one reads zero)."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.lin = nn.Linear(9 * cin, cout, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, h: torch.Tensor, nbr) -> torch.Tensor:
        if h.shape[0] == 0:
            return h.new_zeros(0, self.lin.out_features)
        idx, valid = nbr
        M, C = h.shape
        # index_select, not h[idx]: the advanced-indexing backward accumulates through a sorted
        # index_put and took 90 % of a training step; index_select's backward is an index_add
        g = torch.index_select(h, 0, idx.reshape(-1)).reshape(M, 9, C) * valid.to(h.dtype)
        y = self.bn(self.lin(g.reshape(M, 9 * C)))
        return torch.relu(y + h) if y.shape == h.shape else torch.relu(y)


class SparseDown(nn.Module):
    """2 x 2 stride-2 sparse convolution: each parent reads its (up to four) occupied children."""

    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.lin = nn.Linear(4 * cin, cout, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, h: torch.Tensor, n_parent: int, inv: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        if h.shape[0] == 0:
            return h.new_zeros(0, self.lin.out_features)
        buf = h.new_zeros(n_parent * 4, h.shape[1])
        buf[inv * 4 + slot] = h                     # one child per slot: no accumulation, no atomics
        return torch.relu(self.bn(self.lin(buf.reshape(n_parent, -1))))


def _seg_pool(h: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """`[mean || max]` per packet over its sites; zero for a packet without sites."""
    h32 = h.float()
    mean = torch.segment_reduce(h32, "mean", lengths=lengths, axis=0, unsafe=True, initial=0.0)
    peak = torch.segment_reduce(h32, "max", lengths=lengths, axis=0, unsafe=True, initial=0.0)
    return torch.cat([mean, peak], dim=-1).to(h.dtype)


class SparsePyramid(nn.Module):
    """`(events, ptr, delta_t_s)` in, one feature vector per packet out (and the routed-readout nodes)."""

    #: the routed readout's per-node edge summary; this encoder has no edges, it hands over zeros
    EDGE_SUMMARY_DIM = 4

    def __init__(self, height: int = 180, width: int = 240, hidden: int = 128, feat_dim: int = 512,
                 cell: int = 4, channels: Sequence[int] = (48, 96, 192), blocks: Sequence[int] = (1, 2, 2),
                 extra_channels: int = 0, readout: bool = True, nodes: bool = True):
        super().__init__()
        if extra_channels:
            raise ValueError("sparse_pyramid reads the events only (no rendered node channels)")
        if len(channels) != len(blocks) or len(channels) < 2:
            raise ValueError("sparse_pyramid: one block count per level, at least two levels")
        self.height, self.width, self.cell = int(height), int(width), int(cell)
        self.hidden, self.feat_dim = int(hidden), int(feat_dim)
        self.channels = c = tuple(int(v) for v in channels)
        self.n_levels = L = len(c)
        self.h0 = (self.height + self.cell - 1) // self.cell
        self.w0 = (self.width + self.cell - 1) // self.cell
        self.embed = nn.Sequential(nn.Linear(IN_DIM, c[0], bias=False), nn.BatchNorm1d(c[0]), nn.ReLU(inplace=True))
        self.stages = nn.ModuleList([nn.ModuleList([SubmConv(c[i], c[i]) for _ in range(int(blocks[i]))])
                                     for i in range(L)])
        self.downs = nn.ModuleList([SparseDown(c[i], c[i + 1]) for i in range(L - 1)])
        # routed-readout nodes: a level-0 site's own feature and all its ancestors'. Not built when no
        # consumer reads nodes (an absolute arm): unused parameters are an error under DDP and in the gate.
        self.nodes = bool(nodes)
        if self.nodes:
            # LayerNorm first: the inputs are BN + ReLU outputs, all positive, and an all-positive input
            # let Adam's warmup steps push every ReLU of the next layer below zero for good (measured: 1 of
            # 512 readout units alive after 500 steps without it)
            self.node_proj = nn.Sequential(nn.LayerNorm(sum(c)), nn.Linear(sum(c), self.hidden), nn.ReLU(inplace=True))
        self.readout = bool(readout)
        #: the [mean || max] pool the readout projects, kept after each forward (`EventGNN.pooled`)
        self.pooled_dim = 2 * (c[-2] + c[-1])
        self.pooled = None
        if self.readout:
            # [mean || max] over the sites of the two coarsest levels, layer-normalised (see node_proj). GELU,
            # not ReLU: with a ReLU here the 500-step gate found 0-16 % of the 512 units alive even after the
            # LayerNorm, the packet feature constant and the absolute root measurement with it
            self.proj = nn.Sequential(nn.LayerNorm(2 * (c[-2] + c[-1])),
                                      nn.Linear(2 * (c[-2] + c[-1]), self.feat_dim), nn.GELU(),
                                      nn.Linear(self.feat_dim, self.feat_dim))

    # ------------------------------------------------------------------ level 0
    def _level0(self, events: torch.Tensor, delta_t_s: torch.Tensor):
        b = events[:, EV_BATCH].long()
        xs, ys = events[:, EV_X], events[:, EV_Y]
        x = torch.div(xs.long(), self.cell, rounding_mode="floor").clamp(0, self.w0 - 1)
        y = torch.div(ys.long(), self.cell, rounding_mode="floor").clamp(0, self.h0 - 1)
        tn = (events[:, EV_T] / delta_t_s.clamp_min(1e-6)[b]).clamp(0.0, 1.0)
        pol = (events[:, EV_P] > 0.5).float()
        # sub-cell position in [0, 1): where inside the cell the events fall
        sub = torch.stack([xs - x * self.cell, ys - y * self.cell], -1) * (1.0 / self.cell)
        key = _Level.make_key(b, y, x, self.h0, self.w0)
        order = torch.argsort(key, stable=True)
        uniq, cnt = torch.unique_consecutive(key[order], return_counts=True)
        tn, pol = tn[order], pol[order]
        sums = torch.segment_reduce(torch.cat([pol.unsqueeze(-1), tn.unsqueeze(-1), sub[order]], -1), "sum",
                                    lengths=cnt, axis=0, unsafe=True)
        newest = torch.segment_reduce(torch.stack([tn * pol, tn * (1.0 - pol)], -1), "max",
                                      lengths=cnt, axis=0, unsafe=True)
        n = cnt.float().unsqueeze(-1)
        lv = _Level(uniq, self.h0, self.w0)
        _, yq, xq = lv.bxy()
        centre = torch.stack([(xq.float() - 0.5) / self.w0, (yq.float() - 0.5) / self.h0], -1)
        f = torch.cat([torch.log1p(sums[:, :1]), torch.log1p(n - sums[:, :1]), newest,
                       sums[:, 1:] / n, centre], dim=-1)
        return lv, f

    # ---------------------------------------------------------------- forward
    def forward(self, events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor,
                extra: Optional[torch.Tensor] = None, return_nodes: bool = False):
        """Packet feature `(B, feat_dim)`; with `return_nodes` the `EventGNN` tuple
        `(out, h (B, N, hidden), g (B, N, 4) zeros, px (B, N), py (B, N), mask (B, N))`, pixels in
        the event frame. An event-free packet returns exactly zero and no live node."""
        if extra is not None and extra.shape[0]:
            raise ValueError("sparse_pyramid reads the events only")
        B = int(ptr.numel() - 1)
        dev = events.device
        wdtype = self.embed[0].weight.dtype
        if events.shape[0] == 0:
            self.pooled = torch.zeros(B, self.pooled_dim, device=dev, dtype=wdtype)
            z = torch.zeros(B, self.feat_dim, device=dev, dtype=wdtype) if self.readout else None
            if return_nodes:
                e = torch.zeros(B, 0, device=dev, dtype=torch.float32)
                return (z, torch.zeros(B, 0, self.hidden, device=dev, dtype=wdtype),
                        torch.zeros(B, 0, self.EDGE_SUMMARY_DIM, device=dev, dtype=torch.float32),
                        e, e, torch.zeros(B, 0, device=dev, dtype=torch.bool))
            return z

        lv, f = self._level0(events, delta_t_s)
        h = self.embed(f.to(wdtype))
        levels, feats, links = [lv], [], []
        for i in range(self.n_levels):
            for blk in self.stages[i]:
                h = blk(h, lv.neighbours())
            feats.append(h)
            if i < self.n_levels - 1:
                nxt, inv, slot = lv.parent()
                h = self.downs[i](h, nxt.key.numel(), inv, slot)
                links.append(inv)
                lv = nxt
                levels.append(lv)

        out = None
        if self.readout:
            if B == 1:          # one packet with events: every site is its own, no segment bookkeeping
                self.pooled = torch.cat([torch.cat([feats[i].mean(0, keepdim=True), feats[i].amax(0, keepdim=True)], -1)
                                         for i in (-2, -1)], dim=-1)
                out = self.proj(self.pooled)
            else:
                seg = levels[-1].segments(B)
                live = (seg > 0).unsqueeze(1)
                self.pooled = torch.cat([_seg_pool(feats[-2], levels[-2].segments(B)), _seg_pool(feats[-1], seg)],
                                        dim=-1) * live.to(feats[-1].dtype)
                out = self.proj(self.pooled) * live.to(feats[-1].dtype)
        if not return_nodes:
            return out
        if not self.nodes:
            raise RuntimeError("this sparse_pyramid was built without nodes")

        # level-0 sites as routed-readout nodes, each with its ancestors' features, padded per packet
        up, cols = None, [feats[0]]
        for i in range(1, self.n_levels):
            up = links[0] if up is None else links[i - 1][up]
            cols.append(torch.index_select(feats[i], 0, up))
        hn = self.node_proj(torch.cat(cols, dim=-1))
        l0 = levels[0]
        b0, yq, xq = l0.bxy()
        s = float(self.cell)
        cx = (xq.float() - 1.0) * s + (s - 1.0) / 2.0          # cell centre, event-frame pixels
        cy = (yq.float() - 1.0) * s + (s - 1.0) / 2.0
        M0 = l0.key.numel()
        if B == 1:                                           # one packet: nothing to pad
            mask = torch.ones(1, M0, device=dev, dtype=torch.bool)
            g = torch.zeros(1, M0, self.EDGE_SUMMARY_DIM, device=dev, dtype=torch.float32)
            return out, hn.unsqueeze(0), g, cx.unsqueeze(0), cy.unsqueeze(0), mask
        n0 = l0.segments(B)
        N = int(n0.max())
        col = torch.arange(M0, device=dev) - (torch.cumsum(n0, 0) - n0)[b0]
        hpad = hn.new_zeros(B, N, self.hidden)
        hpad[b0, col] = hn
        mask = torch.zeros(B, N, device=dev, dtype=torch.bool)
        mask[b0, col] = True
        px = torch.zeros(B, N, device=dev, dtype=torch.float32)
        py = torch.zeros(B, N, device=dev, dtype=torch.float32)
        px[b0, col] = cx
        py[b0, col] = cy
        g = torch.zeros(B, N, self.EDGE_SUMMARY_DIM, device=dev, dtype=torch.float32)
        return out, hpad, g, px, py, mask
