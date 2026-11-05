"""EVF networks (docs/EXPERIMENT_RECORDS.md#evf-prereg-20261104): node features -> 51-D increment parameters.

EVFPointNet (starting point A, the point version):
  h_i     = phi(node features_i)                              shared MLP, one node at a time
  w_ik    = softmax_i(att(h_i)_k)  over the packet's nodes      17 queries (the 16 MANO joints + one free query, the
                                                               counterpart of F's 16 LBS slots + residual slot): a soft
                                                               pooling, no node belongs to any query
  slot_k  = sum_i w_ik val(h_i),  mass_k = log(1 + sum_i sigmoid(att(h_i)_k))
  params  = fc2(relu(fc1([slots, mass, packet statistics, prev48?])))   fc2 zero-initialised: the initial output is prev
Every per-node quantity depends on that node and the virtual frame only, and the pooling is a softmax / sum over the
nodes, so the asynchronous runtime can add nodes one at a time (online softmax; `semkine/evf_async.py`).

compose(prev, params, ...): the increment. Root translation by a projective head: params[0:2] move the projected root
joint by params * s pixels (s = the hand scale, pixels per 10 cm at the root depth), params[2] is log(z1 / z0); the new
root position is the back-projection, Delta t = p1 - p0 (MANO's translation moves the root joint rigidly). No absolute
depth enters the network: lateral steps are measured in units of the hand's own size and depth steps as a ratio.
params[3:51] add to the root rotation and finger axis-angles (as B2). Output = prev + increment.
"""
from __future__ import annotations

import torch
import torch.nn as nn

N_SLOTS = 17
LOG_DEPTH_MAX = 1.0
MASK_FILL = -1.0e4


class EVFPointNet(nn.Module):
    def __init__(self, node_dim: int, packet_dim: int, hidden: int = 96, layers: int = 3, slot_dim: int = 32,
                 readout_hidden: int = 256, prev48: bool = True):
        super().__init__()
        mods, d = [], int(node_dim)
        for _ in range(int(layers)):
            mods += [nn.Linear(d, int(hidden)), nn.ReLU(inplace=True)]
            d = int(hidden)
        self.phi = nn.Sequential(*mods)
        self.att = nn.Linear(d, N_SLOTS)
        self.val = nn.Linear(d, int(slot_dim))
        self.prev48 = bool(prev48)
        self.fc1 = nn.Linear(N_SLOTS * int(slot_dim) + N_SLOTS + int(packet_dim) + (48 if self.prev48 else 0),
                             int(readout_hidden))
        self.fc2 = nn.Linear(int(readout_hidden), 51)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def node(self, feat):
        """Per-node part: `(logits (..., 17), values (..., slot_dim))` in float32."""
        h = self.phi(feat)
        return self.att(h).float(), self.val(h).float()

    def pool(self, logit, value, mask):
        m = mask.unsqueeze(-1)
        w = torch.softmax(logit.masked_fill(~m, MASK_FILL), dim=1) * m
        w = w / w.sum(1, keepdim=True).clamp_min(1e-12)
        slots = torch.einsum("bnk,bnd->bkd", w, value)
        mass = torch.log1p((torch.sigmoid(logit) * m).sum(1))
        return slots, mass

    def readout(self, slots, mass, packet, prev48=None):
        parts = [slots.flatten(1), mass, packet.float()]
        if self.prev48:
            parts.append(prev48.float())
        return self.fc2(torch.relu(self.fc1(torch.cat(parts, -1))))

    def forward(self, feat, mask, packet, prev48=None):
        logit, value = self.node(feat)
        slots, mass = self.pool(logit, value, mask)
        return self.readout(slots, mass, packet, prev48).float()


def compose(prev: torch.Tensor, params: torch.Tensor, root: torch.Tensor, scale: torch.Tensor, fx, fy, cx, cy):
    """`prev + increment` (see the module docstring); `root (B, 3)` the root joint of prev in camera metres."""
    with torch.autocast(device_type=prev.device.type, enabled=False):
        prev, params, root = prev.float(), params.float(), root.float()
        z0 = root[:, 2].clamp_min(0.05)
        u0 = fx * root[:, 0] / z0 + cx
        v0 = fy * root[:, 1] / z0 + cy
        u1 = u0 + params[:, 0] * scale
        v1 = v0 + params[:, 1] * scale
        z1 = z0 * torch.exp(params[:, 2].clamp(-LOG_DEPTH_MAX, LOG_DEPTH_MAX))
        p1 = torch.stack([(u1 - cx) / fx * z1, (v1 - cy) / fy * z1, z1], -1)
        p0 = torch.stack([(u0 - cx) / fx * z0, (v0 - cy) / fy * z0, z0], -1)
        return prev + torch.cat([p1 - p0, params[:, 3:51]], -1)


__all__ = ["EVFPointNet", "N_SLOTS", "compose"]
