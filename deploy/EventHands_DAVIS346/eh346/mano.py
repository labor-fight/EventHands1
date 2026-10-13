"""MANO right-hand layer (778 vertices, 21 OpenPose joints), pure torch.

Identical math to the training repository's `model/mano_layer.py` (same operations in the same order), so the
previous-state render reproduces the server bit for bit. The MANO buffers are loaded from the weight file
(they are part of every S39 checkpoint); see LICENSE_NOTE.md for the MANO licence.

The 51-D tracking state is   [transl (3) | global_orient (3, axis-angle) | hand_pose (45, axis-angle, relative to
MANO `hands_mean`)]   in the camera frame (x right, y down, z forward, metres). It is the smplx MANO parameterisation
with `flat_hand_mean=False`: verts = LBS(betas, global_orient, hands_mean + hand_pose) + transl.
"""
from __future__ import annotations

import torch
from torch import nn

BUFFERS = ("v_template", "shapedirs", "posedirs", "J_regressor", "weights", "f", "kintree_table", "hands_mean",
           "hands_components", "fingertip_vertex_ids", "mano16_tips_to_openpose21", "parents")

#: OpenPose 21-joint skeleton (parent, child) for drawing
SKELETON = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


def batch_rodrigues(rot_vecs: torch.Tensor) -> torch.Tensor:
    """Axis-angle (B, 3) -> rotation matrices (B, 3, 3)."""
    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype
    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle
    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view(batch_size, 3, 3)
    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
    return ident + sin * K + (1 - cos) * torch.bmm(K, K)


def _transform_mat(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    pad = torch.zeros((R.shape[0], 1, 4), dtype=R.dtype, device=R.device)
    pad[:, :, -1] = 1.0
    return torch.cat([torch.cat([R, t], dim=2), pad], dim=1)


class Mano(nn.Module):
    """`forward(betas (B, 10), global_orient (B, 3), hand_pose_full (B, 45), transl (B, 3)) -> (verts, joints21)`."""

    def __init__(self):
        super().__init__()
        shapes = {"v_template": (778, 3), "shapedirs": (778, 3, 10), "posedirs": (778, 3, 135), "J_regressor": (16, 778),
                  "weights": (778, 16), "f": (1538, 3), "kintree_table": (2, 16), "hands_mean": (45,),
                  "hands_components": (45, 45), "fingertip_vertex_ids": (5,), "mano16_tips_to_openpose21": (21,),
                  "parents": (16,)}
        for k in BUFFERS:
            dt = torch.long if k in ("f", "kintree_table", "fingertip_vertex_ids", "mano16_tips_to_openpose21",
                                     "parents") else torch.float32
            self.register_buffer(k, torch.zeros(shapes[k], dtype=dt))
        self._parents_py = None
        self.num_joints = 16

    def _parents(self):
        if self._parents_py is None:
            self._parents_py = [int(p) for p in self.parents.cpu()]
        return self._parents_py

    def forward(self, betas, global_orient, hand_pose_full, transl=None):
        batch = betas.shape[0]
        device, dtype = betas.device, betas.dtype
        full_pose = torch.cat([global_orient, hand_pose_full], dim=1)
        v_shaped = self.v_template + torch.einsum("bl,mkl->bmk", betas, self.shapedirs)
        J = torch.einsum("bik,ji->bjk", v_shaped, self.J_regressor)
        rot_mats = batch_rodrigues(full_pose.view(-1, 3)).view(batch, 16, 3, 3)
        ident = torch.eye(3, dtype=dtype, device=device)
        pose_feature = (rot_mats[:, 1:] - ident).view(batch, -1)
        v_posed = v_shaped + torch.einsum("bl,mkl->bmk", pose_feature, self.posedirs)
        rel_joints = J.clone()
        rel_joints[:, 1:] -= J[:, self.parents[1:]]
        transforms_mat = _transform_mat(rot_mats.view(-1, 3, 3), rel_joints.reshape(-1, 3, 1)).view(batch, 16, 4, 4)
        parents = self._parents()
        chain = [transforms_mat[:, 0]]
        for i in range(1, self.num_joints):
            chain.append(torch.matmul(chain[parents[i]], transforms_mat[:, i]))
        transforms = torch.stack(chain, dim=1)
        posed_joints = transforms[:, :, :3, 3]
        joints_h = torch.cat([J, torch.zeros((batch, 16, 1), dtype=dtype, device=device)], dim=2)
        rest = torch.matmul(transforms, joints_h.unsqueeze(-1))
        joint_rel = transforms.clone()
        joint_rel[:, :, :3, 3] = transforms[:, :, :3, 3] - rest[:, :, :3, 0]
        T = torch.einsum("bnj,bjkl->bnkl", self.weights.expand(batch, -1, -1), joint_rel)
        rest_h = torch.cat([v_posed, torch.ones((batch, v_posed.shape[1], 1), dtype=dtype, device=device)], dim=2)
        verts = torch.matmul(T, rest_h.unsqueeze(-1)).squeeze(-1)[:, :, :3]
        tips = verts[:, self.fingertip_vertex_ids]
        joints21 = torch.cat([posed_joints, tips], dim=1)[:, self.mano16_tips_to_openpose21]
        if transl is not None:
            verts = verts + transl.unsqueeze(1)
            joints21 = joints21 + transl.unsqueeze(1)
        return verts, joints21

    def from_state(self, state51: torch.Tensor, betas: torch.Tensor):
        """(verts (B, 778, 3), joints (B, 21, 3)) of 51-D states (B, 51) in the camera frame."""
        hand = state51[:, 6:51] + self.hands_mean.view(1, -1)
        return self(betas, state51[:, 3:6], hand, state51[:, 0:3])


def state_from_mano(transl, global_orient, hand_pose, flat_hand_mean: bool, hands_mean) -> torch.Tensor:
    """A 51-D state from MANO parameters (camera frame). `hand_pose` (45) is the full axis-angle pose when
    `flat_hand_mean` is True (smplx `flat_hand_mean=True`), else the residual on MANO's mean pose (smplx default)."""
    t = torch.as_tensor(transl, dtype=torch.float32).view(-1, 3)
    g = torch.as_tensor(global_orient, dtype=torch.float32).view(-1, 3)
    h = torch.as_tensor(hand_pose, dtype=torch.float32).view(-1, 45)
    if flat_hand_mean:
        h = h - torch.as_tensor(hands_mean, dtype=torch.float32).view(1, 45).to(h.device)
    return torch.cat([t, g, h], dim=1)
