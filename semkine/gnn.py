#!/usr/bin/env python3
r"""S11 kinematic-tree GNN: a battle arm against the per-joint MLP of S10.

The question is whether the fingers should talk to each other while decoding. S10's decoders are
independent by design -- that is what makes the same-checkpoint ablation isolate a joint. A GNN
on the MANO parent tree lets a distal joint see its ancestors' features, which is the inductive
bias articulated tracking has always used, at the cost of making the ablation leak: zeroing one
node no longer leaves the others untouched.

This arm is therefore optional. It must not replace S10 on the mainline unless a paired
comparison on the selected loss beats the MLP by more than the 1.1 mm cross-training floor.
Parameter count is kept within 10% of the S10 head so a win cannot be explained as capacity.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn


def mano_edges(parents: torch.Tensor) -> torch.Tensor:
    """Undirected tree edges as `(2, E)` index pairs over the 16 MANO joints."""
    child = torch.arange(1, parents.shape[0], device=parents.device)
    par = parents[1:].clamp(min=0)
    e = torch.stack([par, child], 0)
    return torch.cat([e, e.flip(0)], 1)


class TreeConv(nn.Module):
    """One round of message passing: each node reads a linear function of itself and its neighbours."""

    def __init__(self, dim: int):
        super().__init__()
        self.lin_self = nn.Linear(dim, dim)
        self.lin_msg = nn.Linear(dim, dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, h: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        # h: (B, 16, D)
        src, dst = edges[0], edges[1]
        msg = h[:, src]
        agg = h.new_zeros(h.shape)
        agg.index_add_(1, dst, msg)
        deg = h.new_zeros(h.shape[:2]).index_add_(1, dst, torch.ones_like(msg[:, :, 0]))
        agg = agg / deg.clamp_min(1.0).unsqueeze(-1)
        return self.act(self.lin_self(h) + self.lin_msg(agg))


class KinematicGNN(nn.Module):
    """Shared trunk features + previous joint angles in, 51D tangent update out.

    Node 0 is the root (6D written by a separate linear map, matching S10). Nodes 1..15 each
    emit a 3-vector. The GNN is the *only* path from the trunk to a finger angle, same unique-
    pathway rule as S10, except that pathway now includes the tree.
    """

    def __init__(self, feat_dim: int = 256, hidden: int = 64, n_layers: int = 2,
                 parents: Optional[torch.Tensor] = None):
        super().__init__()
        self.hidden = int(hidden)
        self.in_proj = nn.Linear(feat_dim + 3, hidden)
        self.layers = nn.ModuleList([TreeConv(hidden) for _ in range(n_layers)])
        self.root_head = nn.Linear(feat_dim, 6)
        self.joint_out = nn.Linear(hidden, 3)
        if parents is None:
            # MANO right-hand default parent vector (wrist, then three joints per finger).
            parents = torch.tensor([-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14])
        self.register_buffer("parents", parents.long())
        self.register_buffer("edges", mano_edges(self.parents))

    def forward(self, feat: torch.Tensor, prevpos: torch.Tensor,
                ablate_joints: bool = False) -> torch.Tensor:
        B = feat.shape[0]
        prev_j = prevpos[:, 6:51].reshape(B, 15, 3)
        # The root node is conditioned on a zero placeholder so it does not steal finger angles.
        root_pad = feat.new_zeros(B, 1, 3)
        nodes = torch.cat([root_pad, prev_j], 1)                          # (B, 16, 3)
        h = self.in_proj(torch.cat([feat[:, None, :].expand(B, 16, -1), nodes], -1))
        for layer in self.layers:
            h = h + layer(h, self.edges)
        root = self.root_head(feat)
        if ablate_joints:
            joints = feat.new_zeros(B, 45)
        else:
            joints = self.joint_out(h[:, 1:]).reshape(B, 45)
        return torch.cat([root, joints], -1)

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def s10_head_count(feat_dim: int = 256, hidden: int = 64) -> int:
    """Parameter count of the S10 unique-pathway head, for the ±10% capacity gate."""
    root = feat_dim * 6 + 6
    one = (feat_dim + 3) * hidden + hidden + hidden * 3 + 3
    return root + 15 * one
