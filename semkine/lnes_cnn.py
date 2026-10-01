#!/usr/bin/env python3
r"""Root tracking C37: the dense LNES trunk as an S37 frontend.

The question this answers is the one the S37 line kept circling: is the global-rotation error a
property of the readout and the loop, or of the event graph that feeds them? x1001 E7 trained the
graph encoder on absolute targets and it still read root rotation 4-5 deg worse than ResNet18 on
LNES. Here everything S37 does after the encoder stays as it is -- the FK/LBS routing of `prev`, the
per-joint evidence `[mean, max, coverage]`, the root head on `[f; e_0..e_15]`, the fifteen joint
decoders, `prev_mlp`, `prev + delta`, the zero-event gate -- and only the encoder is the dense one:

    events -> LNES (2 planes, built on the GPU, bit-for-bit `semkine.eval_track.build_lnes`)
           -> 2->3 conv adapter + ResNet18 (the dense arm's trunk)
           -> f   = fc(avgpool(layer4))                          (B, feat_dim)   the pooled vector
           -> nodes = layer2 cells (stride 8, 23 x 30, 128 ch)   at their pixel centres, masked to
                      cells that hold at least one event

`hidden` is fixed at 128 (layer2's width), so the evidence, root head and joint decoders have exactly
S37's shapes. The state never enters the encoder: as in S37 it is a routing choice at the readout.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from .events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y


class LnesCNN(nn.Module):
    #: same per-node edge-summary width as `EventGNN` (zeros here: a cell has no edges)
    EDGE_SUMMARY_DIM = 4
    #: the node map is layer2 of ResNet18: stride 8, 128 channels
    NODE_STRIDE, NODE_DIM = 8, 128

    def __init__(self, height: int = 180, width: int = 240, hidden: int = 128, feat_dim: int = 512,
                 extra_channels: int = 0, readout: bool = True):
        super().__init__()
        if int(hidden) != self.NODE_DIM:
            raise ValueError(f"lnes_cnn nodes are ResNet18 layer2 cells ({self.NODE_DIM} channels): "
                             f"set MODEL.ENCODER_HIDDEN {self.NODE_DIM}, got {hidden}")
        if extra_channels:
            raise ValueError("lnes_cnn reads events only; the state enters at the routed readout")
        if not readout:
            raise ValueError("lnes_cnn always has its pooled readout")
        self.height, self.width = int(height), int(width)
        self.hidden, self.feat_dim = int(hidden), int(feat_dim)
        self.conv1 = nn.Conv2d(2, 3, kernel_size=3, padding=1)
        self.rn = models.resnet18(num_classes=self.feat_dim)
        gh = -(-self.height // self.NODE_STRIDE)
        gw = -(-self.width // self.NODE_STRIDE)
        ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
        c = (self.NODE_STRIDE - 1) / 2.0
        self.register_buffer("node_px", (xs.flatten().float() * self.NODE_STRIDE + c), persistent=False)
        self.register_buffer("node_py", (ys.flatten().float() * self.NODE_STRIDE + c), persistent=False)

    def lnes(self, events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor) -> torch.Tensor:
        """`(B, 2, H, W)`: per (pixel, polarity) the latest event's whole-millisecond offset over the
        window length, 0 where nothing fired -- `build_lnes`'s "later overwrites earlier" on
        time-ordered events is a max."""
        B = int(ptr.numel() - 1)
        H, W = self.height, self.width
        out = torch.zeros(B * 2 * H * W, device=events.device, dtype=torch.float32)
        if events.shape[0]:
            b = events[:, EV_BATCH].long()
            x = events[:, EV_X].long().clamp(0, W - 1)
            y = events[:, EV_Y].long().clamp(0, H - 1)
            p = events[:, EV_P].long().clamp(0, 1)
            ms = torch.div(torch.round(events[:, EV_T].double() * 1e6), 1000, rounding_mode="floor")
            win = torch.round(delta_t_s.double() * 1000.0)[b]
            val = (ms.float() / win.float())
            out.scatter_reduce_(0, ((b * 2 + p) * H + y) * W + x, val, reduce="amax")
        return out.view(B, 2, H, W)

    def forward(self, events: torch.Tensor, ptr: torch.Tensor, delta_t_s: torch.Tensor,
                extra: Optional[torch.Tensor] = None, return_nodes: bool = False):
        """Pooled `(B, feat_dim)`; with `return_nodes` the `EventGNN` hand-off
        `(out, h (B, N, 128), g (B, N, 4), px (B, N), py (B, N), mask (B, N))`."""
        B = int(ptr.numel() - 1)
        img = self.lnes(events, ptr, delta_t_s)
        dt = self.conv1.weight.dtype
        rn = self.rn
        x = rn.maxpool(rn.relu(rn.bn1(rn.conv1(self.conv1(img.to(dt))))))
        f2 = rn.layer2(rn.layer1(x))
        out = rn.fc(torch.flatten(rn.avgpool(rn.layer4(rn.layer3(f2))), 1))
        any_ev = (ptr[1:] > ptr[:-1]).unsqueeze(1).to(out.dtype)
        out = out * any_ev                        # an event-free packet returns exactly zero
        if not return_nodes:
            return out
        occ = (img > 0).any(1, keepdim=True).float()
        occ = F.max_pool2d(occ, self.NODE_STRIDE, self.NODE_STRIDE, ceil_mode=True)
        mask = occ.flatten(1) > 0                                               # (B, N)
        h = f2.flatten(2).transpose(1, 2) * mask.unsqueeze(-1).to(f2.dtype)    # (B, N, 128)
        N = h.shape[1]
        g = torch.zeros(B, N, self.EDGE_SUMMARY_DIM, device=h.device, dtype=torch.float32)
        px = self.node_px.unsqueeze(0).expand(B, N) * mask
        py = self.node_py.unsqueeze(0).expand(B, N) * mask
        return out, h, g, px, py, mask
