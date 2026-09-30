#!/usr/bin/env python3
r"""S37. Route event-graph nodes to MANO joints by the previous state's FK geometry, then pool.

The graph never sees the state (S36 law: evidence *structure* stays state-free). The state enters
at the readout only, as a per-node responsibility vector over the 16 MANO joints:

    a_i = W[v(i)] * 1[d_i <= band],   v(i) = the front-most of the `front_k` projected vertices
                                             nearest to node i, d_i = its pixel distance

where `W` is the 778 x 16 linear-blend-skinning matrix -- row `v` lists, with weights, the joints
that move vertex `v`, so `a_i` is the model's own answer to "which joints could have produced an
event at this pixel". Taking the front-most of a few nearest vertices stands in for a z-buffer:
the 2D-nearest vertex may belong to the back of the hand.

Per-joint evidence is then a weighted mean, a hard-assignment max and a coverage scalar:

    e_j = [ sum_i a_ij h_i / sum_i a_ij  ||  max_{i: argmax_j a_i = j} h_i  ||  sum_i a_ij / n_live ]

Joints that no node reaches get exactly zero, so an event-free packet produces zero evidence and
`MODEL.ZERO_EVENT_GATE` keeps its bitwise-`prev` contract. Both functions are pure so the
contracts in `tests/test_s37_routed_readout.py` can pin them without a model.
"""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

#: rows of the (nodes x 778) distance block per chunk; bounds the transient at ~100 MB
ROUTE_CHUNK = 32768


#: a farther vertex may replace the 2D-nearest one as "the surface under this pixel" only if it
#: is within this many pixels of it -- about one projected vertex spacing -- and closer to the
#: camera; otherwise the nearest vertex wins (the front test is for front/back surfaces of the
#: same spot, not for jumping to another finger)
FRONT_TOL_PX = 3.0


def route_front_vertex_lbs(px: torch.Tensor, py: torch.Tensor, mask: torch.Tensor,
                           uv: torch.Tensor, z: torch.Tensor, lbs_weights: torch.Tensor,
                           band_px: float = 16.0, front_k: int = 8
                           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-node joint responsibilities from the previous state's projected mesh.

    px, py: `(B, N)` node pixels in the event frame; mask: `(B, N)` live nodes.
    uv: `(B, V, 2)` projected vertex pixels; z: `(B, V)` vertex depth (metres).
    lbs_weights: `(V, J)` skinning matrix.

    Returns `(a (B, N, J) float32, d (B, N) pixel distance to the chosen vertex, v (B, N) long)`;
    `a` is zero for dead nodes and for nodes farther than `band_px` from the hand.
    """
    B, N = mask.shape
    V, J = lbs_weights.shape
    dev = px.device
    a = torch.zeros(B, N, J, device=dev, dtype=torch.float32)
    d_out = torch.full((B, N), float("inf"), device=dev, dtype=torch.float32)
    v_out = torch.zeros(B, N, device=dev, dtype=torch.long)
    keep = mask.reshape(-1)
    if not bool(keep.any()):
        return a, d_out, v_out
    flat = keep.nonzero().squeeze(1)                                  # live rows in (B*N)
    b_idx = flat // N
    ex = px.reshape(-1)[flat].float()
    ey = py.reshape(-1)[flat].float()
    u = uv[..., 0].float()
    v = uv[..., 1].float()
    zf = z.float()
    k = max(1, min(int(front_k), V))
    d_all, v_all = [], []
    for i0 in range(0, int(flat.numel()), ROUTE_CHUNK):
        sl = slice(i0, i0 + ROUTE_CHUNK)
        b = b_idx[sl]
        d2 = (u[b] - ex[sl, None]).square() + (v[b] - ey[sl, None]).square()     # (m, V)
        near = d2.topk(k, dim=-1, largest=False)
        dist = near.values.clamp(min=0.0).sqrt()                                   # (m, k) sorted
        # front test among the vertices under (about) the same pixel as the nearest one
        same_spot = dist <= dist[:, :1] + FRONT_TOL_PX
        zk = zf[b].gather(1, near.indices).masked_fill(~same_spot, float("inf"))
        front = zk.argmin(dim=-1, keepdim=True)                                    # (m, 1)
        vid = near.indices.gather(1, front).squeeze(1)                             # (m,)
        d_all.append(dist.gather(1, front).squeeze(1))
        v_all.append(vid)
    d = torch.cat(d_all)
    vid = torch.cat(v_all)
    gate = (d <= float(band_px)).to(torch.float32)
    a.reshape(-1, J)[flat] = lbs_weights.to(torch.float32)[vid] * gate.unsqueeze(1)
    d_out.reshape(-1)[flat] = d
    v_out.reshape(-1)[flat] = vid
    return a, d_out, v_out


def pool_joint_evidence(h: torch.Tensor, a: torch.Tensor, mask: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-joint evidence `(B, J, 2C+1)` = [weighted mean, hard-assignment max, coverage].

    h: `(B, N, C)` node features; a: `(B, N, J)` responsibilities (zero = not routed);
    mask: `(B, N)`. Returns `(e, count)` with `count (B, J)` = nodes routed to each joint.
    """
    B, N, C = h.shape
    J = a.shape[-1]
    a = a.to(h.dtype) * mask.unsqueeze(-1).to(h.dtype)
    wsum = a.sum(1)                                                    # (B, J)
    mean = torch.einsum("bnj,bnc->bjc", a, h) / wsum.clamp_min(1e-6).unsqueeze(-1)
    mean = mean * (wsum > 0).unsqueeze(-1).to(h.dtype)

    routed = a.sum(-1) > 0                                             # (B, N)
    j_hard = a.argmax(-1)                                              # (B, N)
    src = h.masked_fill(~routed.unsqueeze(-1), -1e4)
    peak = torch.full((B, J, C), -1e4, device=h.device, dtype=h.dtype)
    peak = peak.scatter_reduce(1, j_hard.unsqueeze(-1).expand(B, N, C), src, reduce="amax")
    count = torch.zeros(B, J, device=h.device, dtype=torch.long)
    count.scatter_add_(1, j_hard, routed.long())
    peak = peak * (count > 0).unsqueeze(-1).to(h.dtype)

    live = mask.sum(1, keepdim=True).clamp_min(1).to(h.dtype)
    coverage = (wsum / live).unsqueeze(-1)
    return torch.cat([mean, peak, coverage], dim=-1), count


class PerJointRootFusion(nn.Module):
    """Root update `sum_j MLP_j([f; e_j])`: one MLP per joint, each with its own weights.

    The sixteen MLPs run as one batched matmul pair rather than a loop of sixteen `nn.Sequential`s:
    at batch 1 the loop is kernel-launch bound (+1.85 ms per packet measured on real packets).
    Slice `j` of every parameter is initialised exactly like an `nn.Linear` of that shape.
    """

    def __init__(self, feat_dim: int, ev_dim: int, hidden: int, out_dim: int = 6, n_joints: int = 16):
        super().__init__()
        d = int(feat_dim) + int(ev_dim)
        self.n_joints, self.in_dim, self.hidden, self.out_dim = int(n_joints), d, int(hidden), int(out_dim)
        self.w1 = nn.Parameter(torch.empty(self.n_joints, d, self.hidden))
        self.b1 = nn.Parameter(torch.empty(self.n_joints, self.hidden))
        self.w2 = nn.Parameter(torch.empty(self.n_joints, self.hidden, self.out_dim))
        self.b2 = nn.Parameter(torch.empty(self.n_joints, self.out_dim))
        with torch.no_grad():
            for j in range(self.n_joints):
                for w, b, fan_in in ((self.w1, self.b1, d), (self.w2, self.b2, self.hidden)):
                    bound = 1.0 / math.sqrt(fan_in)
                    w[j].uniform_(-bound, bound)
                    b[j].uniform_(-bound, bound)

    def head(self, j: int, f: torch.Tensor, e_j: torch.Tensor) -> torch.Tensor:
        """Joint `j`'s own contribution `(B, out_dim)`."""
        h = torch.relu(torch.cat([f, e_j], dim=-1) @ self.w1[j] + self.b1[j])
        return h @ self.w2[j] + self.b2[j]

    def forward(self, f: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        """f: `(B, F)` pooled vector; e: `(B, J, E)` routed evidence -> `(B, out_dim)`."""
        x = torch.cat([f.unsqueeze(1).expand(-1, e.shape[1], -1), e], dim=-1)          # (B, J, F+E)
        h = torch.relu(torch.einsum("bjd,jdh->bjh", x, self.w1) + self.b1)
        return torch.einsum("bjh,jho->bo", h, self.w2) + self.b2.sum(0)
