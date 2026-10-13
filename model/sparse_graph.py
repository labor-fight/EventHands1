"""S43: the sparse trunk with static shapes, so that a CUDA graph can replay it (docs/EXPERIMENT_RECORDS.md#s43-prereg,
deviation 9).

`sparse_trunk.SparseExecutor` evaluates the trunk on exactly the active sites, but every one of its ~300 kernels is
issued by the host and its index building synchronises with the host 28 times per forward (nonzero / unique): at batch
1 the GPU is busy 0.76 ms of a 4.0 ms forward. `StaticSparseTrunk` computes the very same function with tensor shapes
that depend only on a capacity *bucket*:

    level masks   m_0 = sites with a non-zero input channel; m_{l+1} = 2x2 cell occupancy of m_l (dense maps, tiny)
    site lists    idx_l = top-K_l of m_l (K_l = the bucket's capacity of level l; the K_l - n_l surplus rows are "padding":
                  sites with m_l = 0, marked invalid); grid_l maps a site to its row, -1 off the active set
    taps          each conv / pool gathers its k x k neighbourhood through grid_l, an inactive / outside / padding
                  neighbour reading the zero row
    readout       mean and area pooling over the VALID rows of level 4

A padding row computes garbage that no valid row ever reads (grid_l is -1 there), and it is excluded from the readout;
so for every bucket whose capacities hold the active sets, the output equals `SparseExecutor` / the masked dense trunk
(tests/test_s43_sparse.py, <= 1e-4). The only host decision is the bucket; `GraphedSparseTrunk` records one CUDA graph
per bucket on a static input buffer and replays it.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from sparse_trunk import SparseExecutor, SparseResNet

__all__ = ["BUCKETS", "StaticSparseTrunk", "GraphedSparseTrunk"]

#: capacity buckets (levels 0..4), from the active-set sizes of the zgz evaluation steps at W = 200 ms
#: (A0 median 1879, p90 6540, max 9444; A1 max 2728; A2 max 805; A3 max 240; A4 max 75)
BUCKETS: Tuple[Tuple[int, ...], ...] = ((2560, 1024, 384, 128, 64), (5120, 1920, 640, 192, 80), (10240, 3072, 1024, 256, 96))
_PAD = 3        # grid padding: covers kernels up to 7 x 7 (the stem) with padding 3
_EXTRA = 7      # bottom / right padding: the last output cell of a stride-2 layer may read 2 rows past a ceil-sized input


class StaticSparseTrunk:
    """Static-shape evaluation of `(adapter conv1, SparseResNet)` in eval mode on an (H, W, C) input image, batch 1, fp32.
    No host synchronisation inside `run`; `counts(img)` gives the per-level active-set sizes (a (5,) device tensor) for
    the bucket choice."""

    def __init__(self, adapter: nn.Conv2d, trunk: SparseResNet, height: int = 180, width: int = 240,
                 buckets: Sequence[Sequence[int]] = BUCKETS) -> None:
        self.ex = SparseExecutor(adapter, trunk)                      # folded weights, (k*k*cin, cout) layout
        self.trunk = trunk
        self.device = adapter.weight.device
        self.buckets = tuple(tuple(int(k) for k in b) for b in buckets)
        if any(len(b) != 5 for b in self.buckets):
            raise ValueError("a bucket is five level capacities")
        self.hw: List[Tuple[int, int]] = [(height, width)]
        for _ in range(4):
            h, w = self.hw[-1]
            self.hw.append(((h + 1) // 2, (w + 1) // 2))
        self._ar = {k: torch.arange(k, device=self.device) for b in self.buckets for k in b}
        self._tap_off: Dict[Tuple[int, int], torch.Tensor] = {}

    # ---- geometry
    def masks(self, img: torch.Tensor) -> List[torch.Tensor]:
        m = (img.abs().amax(dim=-1) > 0).to(torch.float32)[None, None]
        ms = [m]
        for _ in range(4):
            m = F.max_pool2d(m, 2, 2, ceil_mode=True)
            if self.trunk.dilate:
                d = self.trunk.dilate
                m = F.max_pool2d(m, 2 * d + 1, 1, d)
            ms.append(m)
        return [x[0, 0] for x in ms]

    def counts(self, img: torch.Tensor) -> torch.Tensor:
        return torch.stack([m.sum() for m in self.masks(img)]).to(torch.int64)

    def bucket_for(self, counts: Sequence[int]) -> Optional[int]:
        """Index of the smallest bucket holding `counts` (host ints), None if none does."""
        for i, b in enumerate(self.buckets):
            if all(int(c) <= k for c, k in zip(counts, b)):
                return i
        return None

    def _level(self, m: torch.Tensor, cap: int, lvl: int):
        """(idx (cap,), valid (cap,) bool, padded grid (Hp*Wp,) long, Wp) of mask `m` (h, w) with capacity `cap`."""
        h, w = self.hw[lvl]
        flat = m.reshape(-1)
        idx = torch.topk(flat, cap, sorted=True).indices                      # active sites first (value 1), then padding
        valid = flat[idx] > 0
        grid = torch.full((h * w,), -1, dtype=torch.long, device=m.device)
        grid.scatter_(0, idx, torch.where(valid, self._ar[cap], torch.full_like(idx, -1)))
        gp = F.pad(grid.view(h, w), (_PAD, _EXTRA, _PAD, _EXTRA), value=-1)
        return idx, valid, gp.reshape(-1), gp.shape[1]

    def _taps(self, src, dst_idx: torch.Tensor, dst_lvl: int, k: int, stride: int, pad: int) -> torch.Tensor:
        gp, wp, n_src = src
        h, w = self.hw[dst_lvl]
        r, c = dst_idx // w, dst_idx % w
        key = (k, wp)
        off = self._tap_off.get(key)
        if off is None:
            d = torch.arange(k, device=dst_idx.device)
            off = (d[:, None] * wp + d[None, :]).reshape(1, -1)
            self._tap_off[key] = off
        base = (r * stride - pad + _PAD) * wp + (c * stride - pad + _PAD)
        idx = gp[base[:, None] + off]
        return torch.where(idx >= 0, idx, torch.full_like(idx, n_src))

    # ---- the function
    @torch.no_grad()
    def run(self, img: torch.Tensor, bucket: int) -> torch.Tensor:
        """The 51 trunk logits of `img` (H, W, C) evaluated with the capacities of `bucket` (static shapes)."""
        caps = self.buckets[bucket]
        ex = self.ex
        ms = self.masks(img)
        lv = [self._level(m, cap, l) for l, (m, cap) in enumerate(zip(ms, caps))]
        src = lambda l: (lv[l][2], lv[l][3], caps[l])                               # noqa: E731
        apply = ex._apply
        h0, w0 = self.hw[0]
        f0 = img.reshape(h0 * w0, -1)[lv[0][0]].float()                             # padding rows are zero sites
        x = apply(ex.adapter, f0, self._taps(src(0), lv[0][0], 0, 3, 1, 1))
        if self.trunk.coord:
            r0, c0 = lv[0][0] // w0, lv[0][0] % w0
            u = (c0.float() + 0.5) * (2.0 / w0) - 1.0
            v = (r0.float() + 0.5) * (2.0 / h0) - 1.0
            x = torch.cat([x, u[:, None], v[:, None]], dim=1)
        x = torch.relu(apply(ex.stem, x, self._taps(src(0), lv[1][0], 1, 7, 2, 3)))
        x = F.pad(x, (0, 0, 0, 1))[self._taps(src(1), lv[2][0], 2, 3, 2, 1)].amax(dim=1)
        cur = 2
        for stage in ex.blocks:
            for blk in stage:
                nxt = cur + 1 if blk["stride"] == 2 else cur
                y = torch.relu(apply(blk["c1"], x, self._taps(src(cur), lv[nxt][0], nxt, 3, blk["stride"], 1)))
                y = apply(blk["c2"], y, self._taps(src(nxt), lv[nxt][0], nxt, 3, 1, 1))
                idn = x if blk["down"] is None else apply(blk["down"], x, self._taps(src(cur), lv[nxt][0], nxt, 1, blk["stride"], 0))
                x = torch.relu(y + idn)
                cur = nxt
        valid4 = lv[4][1].to(x.dtype)[:, None]
        s = (x * valid4).sum(dim=0)
        n = valid4.sum().clamp(min=1.0)
        feats = [s / n]
        if self.trunk.pool == "mean_area":
            h4, w4 = self.hw[4]
            feats.append(s / float(h4 * w4))
        self.last_feats = torch.cat(feats)
        return torch.addmm(ex.fc_b[None], self.last_feats[None], ex.fc_w.t())[0]


class GraphedSparseTrunk:
    """`StaticSparseTrunk` with one CUDA graph per bucket, replayed on a static input buffer `img_buf` (H, W, C).

        g = GraphedSparseTrunk(model.conv1, model.rn)
        g.img_buf.copy_(img); out, feats, bucket = g(counts_host)      # counts_host: StaticSparseTrunk.counts(img).tolist()

    The graphs are recorded lazily (`render_fast.capture_checked`: eager warm-up under sync-debug "error" first). The
    returned tensors are the graph's static outputs (clone them if they must survive the next replay)."""

    def __init__(self, adapter: nn.Conv2d, trunk: SparseResNet, height: int = 180, width: int = 240,
                 buckets: Sequence[Sequence[int]] = BUCKETS) -> None:
        self.st = StaticSparseTrunk(adapter, trunk, height, width, buckets)
        c_in = adapter.in_channels
        self.img_buf = torch.zeros(height, width, c_in, device=self.st.device)
        self._graphs: Dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]] = {}

    def graph(self, bucket: int):
        if bucket not in self._graphs:
            from render_fast import capture_checked
            saved = self.img_buf.clone()
            g, out = capture_checked(lambda: self.st.run(self.img_buf, bucket))
            self._graphs[bucket] = (g, out, self.st.last_feats)
            self.img_buf.copy_(saved)
        return self._graphs[bucket]

    def __call__(self, counts_host: Sequence[int]):
        b = self.st.bucket_for(counts_host)
        if b is None:
            return None, None, None
        g, out, feats = self.graph(b)
        g.replay()
        return out, feats, b
