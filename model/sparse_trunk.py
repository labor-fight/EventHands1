#!/usr/bin/env python3
"""S43: a sparse render-and-compare trunk for the 51-d tracker (opt-in, `MODEL.CNN_BACKBONE: sparse_resnet18_l3*`).

The dense tracker's trunk is `rn(conv1(x))` on the full 180 x 240 image x = [LNES | render of the previous state].
Almost all of that image is zero: the LNES is non-zero only where an event fell inside the evidence window, the
render only where a vertex of the previous mesh projects. This module computes the same kind of ResNet-18-l3 trunk
**only on those sites** -- a sparse function, not a dense network with a mask on its output:

    level 0 (1 px)    active sites A0 = {pixels with a non-zero input channel}  (events in the window U mesh points)
                      3x3 submanifold adapter (the model's `conv1`), output kept on A0 only
    level 1 (2 px)    A1 = 2x2 cells of A0 holding at least one active site; 7x7 stride-2 conv reads A0, writes A1
    level 2 (4 px)    A2 = cells of A1; 3x3 stride-2 max-pool over A1; layer1 (2 BasicBlocks, submanifold 3x3) on A2
    level 3 (8 px)    A3 = cells of A2; layer2 (stride-2 block + block) on A3
    level 4 (16 px)   A4 = cells of A3; layer3 on A4
    readout           [mean over A4 | sum over A4 / (H4 W4)] -> fc -> 51

Every convolution reads its input only on active sites (inactive sites are exactly zero) and writes only on its
output's active set; the active set of a strided layer is the cell occupancy of its input's active set, so an event
changes at most one cell per level plus the kernel neighbourhoods around it (bounded footprint). BatchNorm statistics
are taken over active sites only (`MaskedBN2d`), so the function does not depend on how much empty frame surrounds the
hand. The optional coordinate planes (CoordConv, `coord=True`) give the trunk the absolute image position the
translation-equivariant convolutions do not have.

Two implementations of the same function:

* `SparseResNet.forward(x, m)` -- training: dense cuDNN convolutions on zero-filled maps, masked after every layer.
  Masking is exact: a convolution of a map that is zero off the active set, evaluated on the active set, is the
  submanifold / strided sparse convolution.
* `SparseExecutor` -- inference on the active sites alone (gather -> GEMM per layer, BatchNorm folded, no dense map
  ever allocated except the per-level index grid). It reports the multiply-accumulates actually performed.
  `tests/test_s43_sparse.py` pins the two against each other (max |diff| <= 1e-4 on the 51 outputs).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["MaskedBN2d", "SparseBasicBlock", "SparseResNet", "SparseExecutor", "cell_mask", "input_mask"]


def cell_mask(m: torch.Tensor) -> torch.Tensor:
    """Active set of a stride-2 level: 2x2 cells holding an active site (ceil, like every stride-2 layer here)."""
    return F.max_pool2d(m, 2, 2, ceil_mode=True)


def input_mask(x: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, 1, H, W) float mask of the sites with any non-zero channel."""
    return (x.detach().abs().amax(dim=1, keepdim=True) > 0).to(x.dtype)


class MaskedBN2d(nn.BatchNorm2d):
    """BatchNorm2d whose batch statistics are taken over the active sites `m` only; output is zero off `m`.

    Eval mode is the plain affine map with the running statistics (so it folds into the preceding convolution).
    Running variance is updated with the unbiased estimate, as in `nn.BatchNorm2d`."""

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:          # type: ignore[override]
        mx = m.to(x.dtype)
        if self.training:
            xm = x * mx
            n = m.sum(dtype=torch.float32).clamp(min=2.0)
            s1 = xm.sum(dim=(0, 2, 3), dtype=torch.float32)
            s2 = (xm * xm).sum(dim=(0, 2, 3), dtype=torch.float32)
            mean = s1 / n
            var = (s2 / n - mean * mean).clamp(min=0.0)
            if self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    mom = 0.1 if self.momentum is None else float(self.momentum)
                    self.running_mean.mul_(1.0 - mom).add_(mom * mean.detach())
                    self.running_var.mul_(1.0 - mom).add_(mom * var.detach() * (n / (n - 1.0)))
        else:
            mean, var = self.running_mean.float(), self.running_var.float()
        scale = self.weight.float() * torch.rsqrt(var + self.eps)
        shift = self.bias.float() - mean * scale
        y = x * scale.to(x.dtype)[None, :, None, None] + shift.to(x.dtype)[None, :, None, None]
        return y * mx

    def folded(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """(scale, shift) of the eval-mode affine map, float32."""
        scale = self.weight.float() * torch.rsqrt(self.running_var.float() + self.eps)
        return scale, self.bias.float() - self.running_mean.float() * scale


class SparseBasicBlock(nn.Module):
    """torchvision's BasicBlock on active sites: conv3x3(stride) -> MBN -> ReLU -> conv3x3 -> MBN, + (1x1 stride conv
    -> MBN | identity), ReLU. `m_out` is the output level's active set (the cell occupancy of `m_in` if strided)."""

    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = MaskedBN2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = MaskedBN2d(cout)
        self.stride = int(stride)
        if stride != 1 or cin != cout:
            self.down_conv = nn.Conv2d(cin, cout, 1, stride, 0, bias=False)
            self.down_bn = MaskedBN2d(cout)
        else:
            self.down_conv = None
            self.down_bn = None

    def forward(self, x: torch.Tensor, m_out: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x), m_out))
        y = self.bn2(self.conv2(y), m_out)
        idn = x if self.down_conv is None else self.down_bn(self.down_conv(x), m_out)
        return F.relu(y + idn)


class SparseResNet(nn.Module):
    """The ResNet-18 "l3" trunk (stem 7x7/2, max-pool 3x3/2, three stages of two BasicBlocks, widths 64/128/256) as a
    sparse function of the active sites; `forward(x, m)` with x (B, C_in, H, W) zero off m (B, 1, H, W).

    `coord`: append the two normalised image coordinates (on active sites) to the stem input. `pool`: "mean_area"
    concatenates the mean over the last level's active cells and their sum over the full level grid (the latter keeps
    the extent of the active set, i.e. apparent hand size, which the mean alone normalises away)."""

    takes_mask = True

    def __init__(self, num_classes: int, widths: Sequence[int] = (64, 128, 256), layers: Sequence[int] = (2, 2, 2),
                 in_ch: int = 3, coord: bool = True, pool: str = "mean_area", dilate: int = 0) -> None:
        super().__init__()
        # `dilate` d > 0: the active set of every strided level (1..4) is its cell occupancy grown by d cells (Chebyshev),
        # so each level also computes a rim of context around the event / mesh support, as a dense trunk does everywhere
        self.dilate = int(dilate)
        widths, layers = tuple(int(w) for w in widths), tuple(int(n) for n in layers)
        if len(widths) != 3 or len(layers) != 3 or min(layers) < 1 or min(widths) < 1:
            raise ValueError(f"three stage widths and three positive block counts, got {widths} / {layers}")
        if pool not in ("mean_area", "mean"):
            raise ValueError(f"pool must be mean_area or mean, got {pool!r}")
        self.widths, self.layers, self.coord, self.pool = widths, layers, bool(coord), pool
        self.conv1 = nn.Conv2d(in_ch + (2 if coord else 0), widths[0], 7, 2, 3, bias=False)
        self.bn1 = MaskedBN2d(widths[0])
        stages: List[nn.ModuleList] = []
        cin = widths[0]
        for i, (c, n) in enumerate(zip(widths, layers)):
            blocks = [SparseBasicBlock(cin, c, 1 if i == 0 else 2)]
            blocks += [SparseBasicBlock(c, c, 1) for _ in range(n - 1)]
            stages.append(nn.ModuleList(blocks))
            cin = c
        self.layer1, self.layer2, self.layer3 = stages
        self.fc = nn.Linear(widths[-1] * (2 if pool == "mean_area" else 1), num_classes)
        for mod in self.modules():                       # torchvision's ResNet initialisation
            if isinstance(mod, nn.Conv2d):
                nn.init.kaiming_normal_(mod.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(mod, nn.BatchNorm2d):
                nn.init.constant_(mod.weight, 1)
                nn.init.constant_(mod.bias, 0)

    @staticmethod
    def coord_planes(h: int, w: int, device, dtype) -> torch.Tensor:
        """(1, 2, h, w): pixel centres mapped to [-1, 1] (x then y)."""
        u = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * (2.0 / w) - 1.0
        v = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * (2.0 / h) - 1.0
        return torch.stack([u[None, :].expand(h, w), v[:, None].expand(h, w)])[None].to(dtype)

    def masks(self, m0: torch.Tensor) -> List[torch.Tensor]:
        """Active sets of levels 0..4."""
        ms = [m0]
        for _ in range(4):
            m = cell_mask(ms[-1])
            if self.dilate:
                m = F.max_pool2d(m, 2 * self.dilate + 1, 1, self.dilate)
            ms.append(m)
        return ms

    def forward(self, x: torch.Tensor, m0: torch.Tensor, return_masks: bool = False):
        feats, ms = self.pooled(x, m0)
        out = self.fc(feats)
        return (out, ms) if return_masks else out

    def pooled(self, x: torch.Tensor, m0: torch.Tensor):
        """(pooled readout vector (B, F) float32, the level masks): everything before `fc`."""
        ms = self.masks(m0.to(x.dtype))
        if self.coord:
            cp = self.coord_planes(x.shape[2], x.shape[3], x.device, x.dtype)
            x = torch.cat([x, cp.expand(x.shape[0], -1, -1, -1) * ms[0]], dim=1)
        y = F.relu(self.bn1(self.conv1(x), ms[1]))
        y = F.max_pool2d(y, 3, 2, 1) * ms[2]
        for lvl, stage in zip((2, 3, 4), (self.layer1, self.layer2, self.layer3)):
            for blk in stage:
                y = blk(y, ms[lvl])
        m4 = ms[4]
        s = y.sum(dim=(2, 3), dtype=torch.float32)
        n = m4.sum(dim=(2, 3), dtype=torch.float32).clamp(min=1.0)
        feats = [s / n]
        if self.pool == "mean_area":
            feats.append(s / float(m4.shape[2] * m4.shape[3]))
        return torch.cat(feats, dim=1), ms


# ----------------------------------------------------------------------------------------------------------------------
# inference on active sites only
# ----------------------------------------------------------------------------------------------------------------------

class _Level:
    """Active sites of one level: coords (N, 2) long (row, col), a (H, W) int grid holding each site's row (-1 off)."""

    __slots__ = ("coords", "grid", "h", "w", "padded")

    def __init__(self, coords: torch.Tensor, h: int, w: int) -> None:
        self.coords, self.h, self.w = coords, int(h), int(w)
        self.padded = {}
        self.grid = torch.full((h, w), -1, dtype=torch.long, device=coords.device)
        self.grid[coords[:, 0], coords[:, 1]] = torch.arange(len(coords), device=coords.device)

    def child(self, dilate: int = 0) -> "_Level":
        """The stride-2 level: unique 2x2 cells (ceil), grown by `dilate` cells (Chebyshev) inside the level's grid."""
        h, w = (self.h + 1) // 2, (self.w + 1) // 2
        r, c = self.coords[:, 0] // 2, self.coords[:, 1] // 2
        if dilate:
            d = torch.arange(-dilate, dilate + 1, device=r.device)
            r = (r[:, None, None] + d[None, :, None]).expand(-1, -1, len(d)).reshape(-1)
            c = (c[:, None, None] + d[None, None, :]).expand(-1, len(d), -1).reshape(-1)
            ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
            r, c = r[ok], c[ok]
        cell = torch.unique(r * w + c)
        return _Level(torch.stack([cell // w, cell % w], dim=1), h, w)


def _taps(src: _Level, dst: _Level, k: int, stride: int, pad: int) -> torch.Tensor:
    """(N_dst, k*k) row of the input site read by each kernel tap (N_src = "zero row" for an inactive / outside tap),
    taps in the (dy, dx) row-major order of a conv weight's (kh, kw) layout. One gather from the source's index grid
    padded with -1 (the outside of the frame), so no per-tap bounds test is needed."""
    key = (k, pad)
    gp = src.padded.get(key)
    if gp is None:
        gp = F.pad(src.grid, (pad, k, pad, k), value=-1)                     # covers r, c in [-pad, h - 1 + k]
        src.padded[key] = gp
    d = torch.arange(k, device=dst.coords.device)
    off = (d[:, None] * gp.shape[1] + d[None, :]).reshape(1, -1)              # (1, k*k) flat tap offsets
    base = (dst.coords[:, 0:1] * stride) * gp.shape[1] + dst.coords[:, 1:2] * stride
    idx = gp.reshape(-1)[base + off]
    return torch.where(idx >= 0, idx, torch.full_like(idx, len(src.coords)))


class SparseExecutor:
    """Eval-mode `(adapter conv1, SparseResNet)` evaluated on active sites only (fp32, batch 1).

    `run(img)` with img (H, W, C) the tracker's input image [LNES | render] returns the 51 trunk logits and the
    multiply-accumulates performed (only taps that read an active site are counted -- the work a sparse kernel
    does). Weights are copied and BatchNorm folded at construction; rebuild after changing the model."""

    def __init__(self, adapter: nn.Conv2d, trunk: SparseResNet) -> None:
        if trunk.training:
            raise ValueError("SparseExecutor folds BatchNorm: put the model in eval mode first")
        self.trunk = trunk
        self.adapter = self._conv(adapter.weight, adapter.bias)
        self.stem = self._conv(trunk.conv1.weight, None, trunk.bn1)
        self.blocks = []
        for stage in (trunk.layer1, trunk.layer2, trunk.layer3):
            st = []
            for blk in stage:
                st.append({
                    "c1": self._conv(blk.conv1.weight, None, blk.bn1), "c2": self._conv(blk.conv2.weight, None, blk.bn2),
                    "down": None if blk.down_conv is None else self._conv(blk.down_conv.weight, None, blk.down_bn),
                    "stride": blk.stride})
            self.blocks.append(st)
        self.fc_w = trunk.fc.weight.detach().float()
        self.fc_b = trunk.fc.bias.detach().float()

    @staticmethod
    def _conv(w: torch.Tensor, b: Optional[torch.Tensor], bn: Optional[MaskedBN2d] = None) -> Dict[str, torch.Tensor]:
        w = w.detach().float()
        cout, cin, kh, kw = w.shape
        bias = torch.zeros(cout, device=w.device) if b is None else b.detach().float()
        if bn is not None:
            scale, shift = bn.folded()
            w = w * scale[:, None, None, None]
            bias = bias * scale + shift
        # (k*k*cin, cout) with taps outermost, matching the gathered (N, k*k, cin) layout
        return {"w": w.permute(2, 3, 1, 0).reshape(kh * kw * cin, cout).contiguous(), "b": bias, "k": kh, "cin": cin,
                "cout": cout}

    @staticmethod
    def _apply(conv, feats: torch.Tensor, idx: torch.Tensor):
        """Gathered conv: feats (N_src, cin), idx (N_dst, k*k) -> (N_dst, cout). The last row of the padded source is the
        zero row every inactive / outside tap points at."""
        g = F.pad(feats, (0, 0, 0, 1))[idx].reshape(idx.shape[0], -1)
        return torch.addmm(conv["b"], g, conv["w"])

    @torch.no_grad()
    def run(self, img: torch.Tensor, count_macs: bool = True) -> Tuple[torch.Tensor, int, Dict[str, int]]:
        h, w, _ = img.shape
        act = img.abs().amax(dim=-1) > 0
        coords = act.nonzero()
        lv0 = _Level(coords, h, w)
        f0 = img[coords[:, 0], coords[:, 1]].float()
        cache, valid = {}, []

        def taps(src, dst, k, stride, pad):
            key = (id(src), id(dst), k, stride)
            if key not in cache:
                cache[key] = _taps(src, dst, k, stride, pad)
            idx = cache[key]
            if count_macs:
                valid.append((idx, len(src.coords)))
            return idx

        def conv(c, x, idx):
            y = self._apply(c, x, idx)
            if count_macs:
                valid[-1] = valid[-1] + (c["cin"] * c["cout"],)
            return y

        x = conv(self.adapter, f0, taps(lv0, lv0, 3, 1, 1))
        if self.trunk.coord:
            u = (coords[:, 1].float() + 0.5) * (2.0 / w) - 1.0
            v = (coords[:, 0].float() + 0.5) * (2.0 / h) - 1.0
            x = torch.cat([x, u[:, None], v[:, None]], dim=1)
        dl = self.trunk.dilate
        lv1 = lv0.child(dl)
        x = torch.relu(conv(self.stem, x, taps(lv0, lv1, 7, 2, 3)))
        lv2 = lv1.child(dl)
        # 3x3 / 2 max-pool: inactive in-frame sites are 0 and every active value is >= 0, so the max over the active
        # neighbours (with a 0 floor, the zero row) is the dense max-pool of the zero-filled map
        x = F.pad(x, (0, 0, 0, 1))[_taps(lv1, lv2, 3, 2, 1)].amax(dim=1)
        cur = lv2
        for stage in self.blocks:
            for blk in stage:
                nxt = cur.child(dl) if blk["stride"] == 2 else cur
                y = torch.relu(conv(blk["c1"], x, taps(cur, nxt, 3, blk["stride"], 1)))
                y = conv(blk["c2"], y, taps(nxt, nxt, 3, 1, 1))
                idn = x if blk["down"] is None else conv(blk["down"], x, taps(cur, nxt, 1, blk["stride"], 0))
                x = torch.relu(y + idn)
                cur = nxt
        s = x.sum(dim=0)
        feats = [s / max(len(cur.coords), 1)]
        if self.trunk.pool == "mean_area":
            feats.append(s / float(cur.h * cur.w))
        self.last_feats = torch.cat(feats)                                  # the pooled readout (for MODEL.TRANSL_HEAD)
        out = torch.addmm(self.fc_b[None], self.last_feats[None], self.fc_w.t())[0]
        macs = 0
        if count_macs:                                                      # one host synchronisation, at the end
            counts = torch.stack([(idx < n).sum() for idx, n, _ in valid]).tolist()
            macs = int(sum(c * cc for c, (_, _, cc) in zip(counts, valid))) + self.fc_w.numel()
        sizes = {"A0": len(lv0.coords), "A1": len(lv1.coords), "A2": len(lv2.coords), "A4": len(cur.coords)}
        return out, macs, sizes
