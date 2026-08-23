#!/usr/bin/env python3
"""Pure-torch MANO right-hand layer (778 verts / 21 OpenPose joints)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


def batch_rodrigues(rot_vecs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Axis-angle (B, 3) -> rotation matrices (B, 3, 3)."""
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat(
        [zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1
    ).view(batch_size, 3, 3)
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def _transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """(B, 3, 3), (B, 3, 1) -> (B, 4, 4)."""
    pad = torch.zeros((R.shape[0], 1, 4), dtype=R.dtype, device=R.device)
    pad[:, :, -1] = 1.0
    return torch.cat([torch.cat([R, t], dim=2), pad], dim=1)


class ManoLayer(nn.Module):
    """
    Inputs:
      betas:          (B, 10)
      global_orient:  (B, 3) axis-angle
      hand_pose:      (B, 45) FULL local axis-angle (hands_mean already added)
                      OR residual if add_mean=True
      transl:         (B, 3)
    Outputs:
      verts:   (B, 778, 3)
      joints:  (B, 21, 3) OpenPose order
    """

    def __init__(
        self,
        mano_npz: Union[str, Path],
        add_mean: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        data = np.load(mano_npz, allow_pickle=False)
        self.register_buffer("v_template", torch.tensor(data["v_template"], dtype=dtype))
        self.register_buffer("shapedirs", torch.tensor(data["shapedirs"], dtype=dtype))
        self.register_buffer("posedirs", torch.tensor(data["posedirs"], dtype=dtype))
        self.register_buffer("J_regressor", torch.tensor(data["J_regressor"], dtype=dtype))
        self.register_buffer("weights", torch.tensor(data["weights"], dtype=dtype))
        self.register_buffer("f", torch.tensor(data["f"], dtype=torch.long))
        self.register_buffer(
            "kintree_table", torch.tensor(data["kintree_table"], dtype=torch.long)
        )
        self.register_buffer("hands_mean", torch.tensor(data["hands_mean"], dtype=dtype))
        self.register_buffer(
            "hands_components", torch.tensor(data["hands_components"], dtype=dtype)
        )
        self.register_buffer(
            "fingertip_vertex_ids",
            torch.tensor(data["fingertip_vertex_ids"], dtype=torch.long),
        )
        self.register_buffer(
            "mano16_tips_to_openpose21",
            torch.tensor(data["mano16_tips_to_openpose21"], dtype=torch.long),
        )
        parents = data["kintree_table"][0].copy()
        parents[0] = -1
        self.register_buffer("parents", torch.tensor(parents, dtype=torch.long))
        self.add_mean = add_mean
        self.num_joints = 16

    def decode_pca(self, alpha: torch.Tensor, ncomps: int = 6) -> torch.Tensor:
        """PCA coeffs (B, ncomps) -> residual45 (B, 45)."""
        C = self.hands_components[:ncomps]  # (ncomps, 45)
        return alpha @ C

    def project_pca(self, residual45: torch.Tensor, ncomps: int = 6) -> torch.Tensor:
        """residual45 (B, 45) -> PCA coeffs (B, ncomps) via pinv(C[:ncomps])."""
        C = self.hands_components[:ncomps]  # (ncomps, 45)
        # alpha = residual @ pinv(C) with C shape (ncomps, 45)
        # pinv(C) is (45, ncomps); residual @ pinv = (B, ncomps)
        pinv = torch.linalg.pinv(C)
        return residual45 @ pinv

    def forward(
        self,
        betas: torch.Tensor,
        global_orient: torch.Tensor,
        hand_pose: torch.Tensor,
        transl: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        hand_pose: (B, 45) full local AA if add_mean=False (default for GT residual+mean),
                   or residual if add_mean=True.
        """
        batch = betas.shape[0]
        device, dtype = betas.device, betas.dtype

        pose_hand = hand_pose
        if self.add_mean:
            pose_hand = pose_hand + self.hands_mean.view(1, -1)

        full_pose = torch.cat([global_orient, pose_hand], dim=1)  # (B, 48)

        v_shaped = self.v_template + torch.einsum("bl,mkl->bmk", betas, self.shapedirs)
        J = torch.einsum("bik,ji->bjk", v_shaped, self.J_regressor)  # (B, 16, 3)

        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(batch, 16, 3, 3)
        ident = torch.eye(3, dtype=dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view(batch, -1)  # (B, 135)
        v_posed = v_shaped + torch.einsum("bl,mkl->bmk", pose_feature, self.posedirs)

        # Global rigid transforms
        rel_joints = J.clone()
        rel_joints[:, 1:] -= J[:, self.parents[1:]]
        transforms_mat = _transform_mat(
            rot_mats.view(-1, 3, 3), rel_joints.reshape(-1, 3, 1)
        ).view(batch, 16, 4, 4)

        transform_chain = [transforms_mat[:, 0]]
        for i in range(1, self.num_joints):
            transform_chain.append(
                torch.matmul(transform_chain[self.parents[i]], transforms_mat[:, i])
            )
        transforms = torch.stack(transform_chain, dim=1)  # (B, 16, 4, 4)

        posed_joints = transforms[:, :, :3, 3]  # (B, 16, 3)

        joints_homogen = torch.cat(
            [J, torch.zeros((batch, 16, 1), dtype=dtype, device=device)], dim=2
        )
        rest_mats = torch.matmul(transforms, joints_homogen.unsqueeze(-1))
        joint_rel = transforms.clone()
        joint_rel[:, :, :3, 3] = transforms[:, :, :3, 3] - rest_mats[:, :, :3, 0]

        T = torch.einsum("bnj,bjkl->bnkl", self.weights.expand(batch, -1, -1), joint_rel)
        rest_shape_h = torch.cat(
            [
                v_posed,
                torch.ones((batch, v_posed.shape[1], 1), dtype=dtype, device=device),
            ],
            dim=2,
        )
        verts = torch.matmul(T, rest_shape_h.unsqueeze(-1)).squeeze(-1)[:, :, :3]

        tips = verts[:, self.fingertip_vertex_ids]  # (B, 5, 3) thumb,index,middle,ring,pinky
        joints16_tips = torch.cat([posed_joints, tips], dim=1)  # (B, 21, 3)
        joints21 = joints16_tips[:, self.mano16_tips_to_openpose21]

        if transl is not None:
            verts = verts + transl.unsqueeze(1)
            joints21 = joints21 + transl.unsqueeze(1)

        return verts, joints21


def load_mano(
    mano_npz: Union[str, Path, None] = None, add_mean: bool = False
) -> ManoLayer:
    if mano_npz is None:
        mano_npz = Path(__file__).resolve().parents[1] / "assets" / "mano_right.npz"
    return ManoLayer(mano_npz, add_mean=add_mean)
