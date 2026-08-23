#!/usr/bin/env python3
"""
Pose representation adapters for EventHands absolute-pose baselines.

Paper notation:  theta = [t, R, alpha]
Repo 12D layout: [alpha6, t3, R3]   (kept for checkpoint compatibility)
51D meta layout: [t3, R3, residual45]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass(frozen=True)
class Slices:
    pose_repr: str
    output_dim: int

    @property
    def local(self) -> slice:
        if self.pose_repr == "mano_pca6":
            return slice(0, 6)
        if self.pose_repr == "mano_full_axis_angle":
            return slice(6, 51)
        raise ValueError(self.pose_repr)

    @property
    def transl(self) -> slice:
        if self.pose_repr == "mano_pca6":
            return slice(6, 9)
        if self.pose_repr == "mano_full_axis_angle":
            return slice(0, 3)
        raise ValueError(self.pose_repr)

    @property
    def root(self) -> slice:
        if self.pose_repr == "mano_pca6":
            return slice(9, 12)
        if self.pose_repr == "mano_full_axis_angle":
            return slice(3, 6)
        raise ValueError(self.pose_repr)


def get_slices(pose_repr: str, output_dim: int) -> Slices:
    if pose_repr == "mano_pca6":
        assert output_dim == 12
    elif pose_repr == "mano_full_axis_angle":
        assert output_dim == 51
    else:
        raise ValueError(pose_repr)
    return Slices(pose_repr=pose_repr, output_dim=output_dim)


def meta51_to_target(
    params51: torch.Tensor,
    pose_repr: str,
    components: torch.Tensor,
) -> torch.Tensor:
    """
    Convert meta row(s) [t3, R3, residual45] into network target layout.
    params51: (..., 51)
    components: (45, 45) hands_components
    """
    t = params51[..., 0:3]
    R = params51[..., 3:6]
    residual = params51[..., 6:51]
    if pose_repr == "mano_full_axis_angle":
        return params51
    if pose_repr == "mano_pca6":
        C6 = components[:6]  # (6, 45)
        # alpha = residual @ pinv(C6)
        pinv = torch.linalg.pinv(C6)
        alpha = residual @ pinv
        return torch.cat([alpha, t, R], dim=-1)
    raise ValueError(pose_repr)


def decode_to_mano_inputs(
    pred: torch.Tensor,
    pose_repr: str,
    components: torch.Tensor,
    hands_mean: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Network output -> MANO inputs.
    Returns dict with transl, global_orient, local_full_aa (45, mean already added).
    """
    sl = get_slices(pose_repr, pred.shape[-1])
    transl = pred[..., sl.transl]
    global_orient = pred[..., sl.root]
    local = pred[..., sl.local]
    if pose_repr == "mano_pca6":
        # residual45 = alpha6 @ C6
        residual = local @ components[:6]
    else:
        residual = local
    local_full = residual + hands_mean.view(*([1] * (residual.ndim - 1)), -1)
    return {
        "transl": transl,
        "global_orient": global_orient,
        "local_full_aa": local_full,
        "residual45": residual,
    }


def axis_angle_to_quaternion(aa: torch.Tensor) -> torch.Tensor:
    """Axis-angle (..., 3) -> unit quaternion (..., 4), w first.

    The half-angle factor sin(|r|/2)/|r| is replaced by its Taylor expansion near
    the identity, and the norm is taken from a clamped square so neither the value
    nor its gradient blows up at r = 0 -- which is exactly where an untrained 51D
    head sits, and where a plain ``norm`` would hand back NaN gradients.
    """
    sq = (aa * aa).sum(dim=-1, keepdim=True)
    theta = sq.clamp(min=1e-24).sqrt()
    half = 0.5 * theta
    sin_over_theta = torch.where(sq < 1e-12, 0.5 - sq / 48.0, torch.sin(half) / theta)
    return torch.cat([torch.cos(half), aa * sin_over_theta], dim=-1)


def split_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    pose_repr: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (L_local, L_root, L_transl) as elementwise MSE means."""
    sl = get_slices(pose_repr, pred.shape[-1])
    l_local = torch.nn.functional.mse_loss(pred[..., sl.local], target[..., sl.local])
    l_root = torch.nn.functional.mse_loss(pred[..., sl.root], target[..., sl.root])
    l_t = torch.nn.functional.mse_loss(pred[..., sl.transl], target[..., sl.transl])
    return l_local, l_root, l_t
