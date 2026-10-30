#!/usr/bin/env python3
r"""XKF direct branch (docs/EXPERIMENT_RECORDS.md#xkf-20261014): a measurement of the current hand that never reads the
previous state.

Borrowed structure:
* E-3DPSM (CVPR 2026, arXiv 2604.08543): a direct pose estimate from the event window, fused with an incremental
  estimate; here the incremental estimate is the unchanged S37 XYZ path and the fusion is `semkine/xkf.py`.
* HandDGP (ECCV 2024, arXiv 2407.15844): camera-space translation from 2D keypoints and root-relative 3D joints by a
  differentiable closed-form least-squares solve, so depth follows from geometry (apparent size given beta and K) and
  is not regressed against the training depth range.
* CLIFF (ECCV 2022): the camera enters as per-pixel ray-slope planes (`MNISTModel._cam_planes`).

Input `(B, 5, H, W)` = long-window LNES (2, normalised last timestamp per polarity) + ray slopes x/z, y/z (2) +
log(W / 200 ms) (1). Trunk: an independent ResNet18-l3 (`backbones.ScaledResNet`), its own features (a trunk shared
with the incremental heads made both errors coincide, `MODEL.ABS_TRACK`, retired).

Heads: root rotation (6D), 15 finger local axis-angle residuals (the 51D state convention), per-keypoint confidence
(global pool -> fc); 21 OpenPose keypoint heatmaps at 1/4 resolution from layer1 + upsampled layer2 / layer3, read by
soft-argmax in event pixels. Translation: `dgp_translation`.
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from backbones import ScaledResNet, resnet_forward_taps

N_KP = 21
IN_CH = 5
#: global head: root 6D + 45 finger AA + 21 keypoint confidence logits
GLOBAL_OUT = 6 + 45 + N_KP
#: heatmap stride in event pixels (conv1 /2, maxpool /2)
HM_STRIDE = 4


def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 6D -> rotation matrix `(B, 3, 3)` (columns from Gram-Schmidt of the two 3-vectors)."""
    a1, a2 = x[..., 0:3], x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def dgp_translation(joints0: torch.Tensor, uv: torch.Tensor, w: torch.Tensor, ridge: float = 1e-8,
                    return_covariance: bool = False):
    """HandDGP-style global positioning. joints0 `(B, J, 3)` camera-frame joints with zero translation, uv `(B, J, 2)`
    normalised image coordinates (x/z, y/z), w `(B, J)` positive weights. Returns t `(B, 3)` minimising
    sum_j w_j [ (u_j (Z_j + t_z) - (X_j + t_x))^2 + (v_j (Z_j + t_z) - (Y_j + t_y))^2 ]: a 3x3 linear solve in fp32.
    Optionally also returns the residual-scaled inverse normal matrix, a geometric uncertainty diagnostic in m^2.
    XKF's fusion still uses constant empirical residual moments, not this input-dependent covariance."""
    with torch.autocast(device_type=joints0.device.type, enabled=False):
        J, uv, w = joints0.float(), uv.float(), w.float()
        u, v = uv[..., 0], uv[..., 1]
        zero, one = torch.zeros_like(u), torch.ones_like(u)
        a1 = torch.stack([-one, zero, u], -1)                                   # (B, J, 3)
        a2 = torch.stack([zero, -one, v], -1)
        b1 = J[..., 0] - u * J[..., 2]
        b2 = J[..., 1] - v * J[..., 2]
        A = torch.cat([a1, a2], 1)                                              # (B, 2J, 3)
        b = torch.cat([b1, b2], 1)
        ww = torch.cat([w, w], 1)
        AtA = torch.einsum('bni,bn,bnj->bij', A, ww, A)
        Atb = torch.einsum('bni,bn,bn->bi', A, ww, b)
        tr = torch.diagonal(AtA, dim1=1, dim2=2).sum(-1).clamp_min(1e-12)
        AtA = AtA + (ridge * tr).view(-1, 1, 1) * torch.eye(3, device=J.device)
        t = torch.linalg.solve(AtA, Atb.unsqueeze(-1)).squeeze(-1)
        if not return_covariance:
            return t
        residual = torch.einsum('bni,bi->bn', A, t) - b
        variance = (ww * residual.square()).sum(-1) / max(1, A.shape[1] - 3)
        covariance = torch.linalg.inv(AtA) * variance[:, None, None]
        return t, covariance


class DirectBranch(nn.Module):
    def __init__(self, height: int = 180, width: int = 240, fpn: int = 64):
        super().__init__()
        self.height, self.width = int(height), int(width)
        self.adapter = nn.Conv2d(IN_CH, 3, kernel_size=3, padding=1)
        self.trunk = ScaledResNet(GLOBAL_OUT, layers=(2, 2, 2, 0))
        self.lat1 = nn.Conv2d(64, fpn, 1)
        self.lat2 = nn.Conv2d(128, fpn, 1)
        self.lat3 = nn.Conv2d(256, fpn, 1)
        self.hm = nn.Sequential(nn.Conv2d(fpn, fpn, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(fpn, N_KP, 1))
        #: soft-argmax inverse temperature (learned, log-parameterised)
        self.log_beta = nn.Parameter(torch.tensor(math.log(4.0)))

    def soft_argmax(self, heat: torch.Tensor) -> torch.Tensor:
        """`(B, K, h, w)` -> `(B, K, 2)` event-pixel coordinates (x, y) of the cell-centre expectation."""
        B, K, h, w = heat.shape
        p = torch.softmax(self.heatmap_logits(heat).flatten(2), dim=-1).view(B, K, h, w)
        xs = (torch.arange(w, device=heat.device, dtype=torch.float32) + 0.5) * HM_STRIDE - 0.5
        ys = (torch.arange(h, device=heat.device, dtype=torch.float32) + 0.5) * HM_STRIDE - 0.5
        return torch.stack([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)

    def heatmap_logits(self, heat: torch.Tensor) -> torch.Tensor:
        """The same temperature is used for the coordinate readout and heatmap supervision."""
        return heat.float() * self.log_beta.exp()

    def heatmap_loss(self, heat: torch.Tensor, target_px: torch.Tensor, visible: torch.Tensor,
                     sigma_px: float = 6.0) -> torch.Tensor:
        """Mean KL from a normalized Gaussian target (6 event pixels = 1.5 heatmap cells).

        Coordinate-only supervision permits broad/multimodal heatmaps whose mean happens to match the target.
        The KL term also constrains their spatial support, which is needed for stable DGP keypoint geometry.
        """
        _, _, h, w = heat.shape
        xs = (torch.arange(w, device=heat.device, dtype=torch.float32) + 0.5) * HM_STRIDE - 0.5
        ys = (torch.arange(h, device=heat.device, dtype=torch.float32) + 0.5) * HM_STRIDE - 0.5
        logqx = F.log_softmax(-0.5 * ((xs - target_px[..., 0:1]) / sigma_px).square(), dim=-1)
        logqy = F.log_softmax(-0.5 * ((ys - target_px[..., 1:2]) / sigma_px).square(), dim=-1)
        qx, qy = logqx.exp(), logqy.exp()
        logp = F.log_softmax(self.heatmap_logits(heat).flatten(2), dim=-1).view_as(heat)
        cross_entropy = -torch.einsum('bkh,bkw,bkhw->bk', qy, qx, logp)
        entropy = -(qx * logqx).sum(-1) - (qy * logqy).sum(-1)
        kl = (cross_entropy - entropy).clamp_min(0.0)
        return (kl * visible).sum() / visible.sum().clamp_min(1.0)

    def forward(self, lnes: torch.Tensor, cam: torch.Tensor, window_ms: torch.Tensor) -> Dict[str, torch.Tensor]:
        """lnes `(B, H, W, 2)` float in [0, 1]; cam `(B, H, W, 2)` ray slopes; window_ms `(B,)`."""
        B = lnes.shape[0]
        logw = torch.log(window_ms.float().clamp_min(1.0) / 200.0).view(B, 1, 1, 1).expand(B, 1, self.height, self.width)
        x = torch.cat([lnes.permute(0, 3, 1, 2).float(), cam.permute(0, 3, 1, 2).float(), logw], 1)
        x = self.adapter(x.to(self.adapter.weight.dtype))
        g, taps = resnet_forward_taps(self.trunk, x, ("layer1", "layer2", "layer3"))
        l1, l2, l3 = taps["layer1"], taps["layer2"], taps["layer3"]
        f = self.lat3(l3)
        f = F.interpolate(f, size=l2.shape[-2:], mode="bilinear", align_corners=False) + self.lat2(l2)
        f = F.interpolate(f, size=l1.shape[-2:], mode="bilinear", align_corners=False) + self.lat1(l1)
        heat = self.hm(f)
        kp = self.soft_argmax(heat)
        g = g.float()
        return {"root6d": g[:, 0:6], "fingers": g[:, 6:51], "kp_logit": g[:, 51:51 + N_KP], "kp_px": kp,
                "heat": heat}


__all__ = ["DirectBranch", "GLOBAL_OUT", "HM_STRIDE", "N_KP", "dgp_translation", "matrix_to_rot6d", "rot6d_to_matrix"]
