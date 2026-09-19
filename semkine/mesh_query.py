#!/usr/bin/env python3
r"""S37 mesh query: the previous state's MANO mesh reads the event graph and drives the joints.

The state enters the network in exactly one form -- the FK mesh of `prev` -- and the graph is
never conditioned on it. A fixed set of Q query vertices (stratified over the 16 joints by their
dominant skinning weight) is projected into the event frame; each visible query vertex gathers
the graph's node features `h_i`, the graph's own local-motion reading `g_i` (mean edge vector)
and the offset of the events it read, with a fixed spatial kernel:

    w_qi = softmax_i( -d_qi^2 / 2 sigma^2 ) over live nodes with d_qi <= band,   d in pixels
    v_q  = [ sum_i w_qi h_i  ||  sum_i w_qi g_i  ||  sum_i w_qi (p_i - u_q) / band  ||  mass_q ]

`mass_q` is the share of live nodes inside the band. Vertex evidence is then carried to joints by
the skinning weights of the query vertices, so joint j's evidence is a weighted mean over "its"
surface, and only over vertices that actually saw events:

    e_j = sum_q W[q, j] has_q v_q / sum_q W[q, j] has_q,     cov_j = sum_q W[q, j] has_q / sum_q W[q, j]

A joint whose surface saw no event has e_j = 0 and cov_j = 0; the decoder gates its update with
`cov_j > 0`, so an unobserved joint does not move. That is the per-joint form of the zero-event
identity. The kernel width is a constant on purpose: a learnable width under the noise curriculum
was measured to flatten to the whole frame (the S38 retraction), and KEG's learned routing put the
loop gain at 0.91. Nothing here has a parameter the optimizer could use to widen the kernel.

Occlusion: a query vertex on the back of the hand is dropped when another mesh vertex within
`front_px` of it is closer to the camera by more than `z_tol` (a cheap z-test against the 778
projected vertices, no rasterizer).
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn


def stratified_query_vertices(lbs_weights: torch.Tensor, n_tokens: int) -> torch.Tensor:
    """`n_tokens` vertex ids, the same number per joint, taken from the vertices that joint
    dominates (largest weight on it), in decreasing weight. Deterministic."""
    V, J = lbs_weights.shape
    per = max(1, n_tokens // J)
    dom = lbs_weights.argmax(-1)
    picked = []
    for j in range(J):
        cand = (dom == j).nonzero().squeeze(1)
        if cand.numel() == 0:
            cand = lbs_weights[:, j].topk(per).indices
        order = lbs_weights[cand, j].argsort(descending=True)
        cand = cand[order]
        if cand.numel() < per:                              # pad from the joint's next-best vertices
            extra = lbs_weights[:, j].topk(per * 4).indices
            extra = extra[~torch.isin(extra, cand)][: per - cand.numel()]
            cand = torch.cat([cand, extra])
        picked.append(cand[:per])
    return torch.cat(picked)


class MeshQuery(nn.Module):
    def __init__(self, lbs_weights: torch.Tensor, n_tokens: int = 192, band_px: float = 16.0,
                 sigma_px: float = 8.0, front_px: float = 3.0, z_tol: float = 0.01,
                 node_dim: int = 128, edge_dim: int = 4):
        super().__init__()
        W = lbs_weights.detach().float()
        q_idx = stratified_query_vertices(W, n_tokens)
        self.register_buffer("q_idx", q_idx, persistent=False)
        self.register_buffer("lbs_q", W[q_idx], persistent=False)                # (Q, J)
        self.n_joints = int(W.shape[1])
        self.band_px, self.sigma_px = float(band_px), float(sigma_px)
        self.front_px, self.z_tol = float(front_px), float(z_tol)
        self.node_dim, self.edge_dim = int(node_dim), int(edge_dim)
        #: per-vertex evidence width: h, g, mean offset (2), mass (1)
        self.vert_dim = self.node_dim + self.edge_dim + 3
        #: per-joint evidence width: vertex evidence + coverage
        self.ev_dim = self.vert_dim + 1

    @torch.no_grad()
    def visible(self, uv: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """`(B, Q)` bool: query vertex not occluded by a closer mesh vertex under the same pixel."""
        uq, zq = uv[:, self.q_idx], z[:, self.q_idx]                              # (B, Q, 2), (B, Q)
        d2 = (uq[:, :, None, 0] - uv[:, None, :, 0]).square() + \
             (uq[:, :, None, 1] - uv[:, None, :, 1]).square()                     # (B, Q, V)
        near = d2 <= self.front_px ** 2
        zmin = z[:, None, :].masked_fill(~near, float("inf")).amin(-1)             # (B, Q)
        return zq <= zmin + self.z_tol

    def attention(self, px: torch.Tensor, py: torch.Tensor, mask: torch.Tensor,
                  uv: torch.Tensor, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed spatial kernel from query vertices to live nodes: `w (B, Q, N)` rows summing to 1
        where the vertex saw anything, and `inband (B, Q, N)`."""
        uq = uv[:, self.q_idx]                                                    # (B, Q, 2)
        d2 = (uq[:, :, None, 0] - px[:, None, :]).square() + \
             (uq[:, :, None, 1] - py[:, None, :]).square()                        # (B, Q, N)
        inband = (d2 <= self.band_px ** 2) & mask[:, None, :] & self.visible(uv, z)[:, :, None]
        logit = (-0.5 * d2 / self.sigma_px ** 2).masked_fill(~inband, -1e4)
        w = torch.softmax(logit, dim=-1) * inband.any(-1, keepdim=True).to(logit.dtype)
        return w, inband

    def forward(self, h: torch.Tensor, g: torch.Tensor, px: torch.Tensor, py: torch.Tensor,
                mask: torch.Tensor, uv: torch.Tensor, z: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Returns `(e (B, J, ev_dim), mesh (B, ev_dim), cov (B, J), stats)`."""
        B, N, _ = h.shape
        w, inband = self.attention(px.float(), py.float(), mask, uv.float(), z.float())
        w = w.to(h.dtype)
        uq = uv[:, self.q_idx].to(h.dtype)                                        # (B, Q, 2)
        p = torch.stack([px, py], dim=-1).to(h.dtype)                             # (B, N, 2)
        live = mask.sum(1).clamp_min(1).to(h.dtype)                               # (B,)
        feat = torch.cat([h, g.to(h.dtype), p], dim=-1)                           # (B, N, C+4+2)
        agg = torch.bmm(w, feat)                                                  # (B, Q, C+4+2)
        has = (inband.sum(-1) > 0)                                                # (B, Q)
        mean_p = agg[..., -2:]
        offset = (mean_p - uq) / self.band_px * has.unsqueeze(-1).to(h.dtype)
        mass = (inband.sum(-1).to(h.dtype) / live[:, None]).unsqueeze(-1)         # (B, Q, 1)
        v = torch.cat([agg[..., :-2], offset, mass], dim=-1)                      # (B, Q, vert_dim)

        Wq = self.lbs_q.to(h.dtype) * has.unsqueeze(-1).to(h.dtype)               # (B, Q, J) seen only
        num = torch.einsum("bqj,bqd->bjd", Wq, v)
        den = Wq.sum(1)                                                           # (B, J)
        e = num / den.clamp_min(1e-6).unsqueeze(-1) * (den > 0).unsqueeze(-1).to(h.dtype)
        cov = den / self.lbs_q.sum(0).to(h.dtype).clamp_min(1e-6)[None, :]        # (B, J)
        e = torch.cat([e, cov.unsqueeze(-1)], dim=-1)                             # (B, J, ev_dim)

        n_has = has.sum(1).clamp_min(1).to(h.dtype)
        mesh = (v * has.unsqueeze(-1).to(h.dtype)).sum(1) / n_has[:, None]
        mesh = torch.cat([mesh, (has.sum(1).to(h.dtype) / has.shape[1])[:, None]], dim=-1)
        stats = {"mq_frac_vertices_seen": (has.sum(1).to(torch.float32) / has.shape[1]).mean(),
                 "mq_frac_nodes_used": (inband.any(1).sum(1).to(torch.float32) / live).mean(),
                 "mq_joints_seen": (cov > 0).sum(1).to(torch.float32).mean()}
        return e, mesh, cov, stats

    def node_responsibility(self, px: torch.Tensor, py: torch.Tensor, mask: torch.Tensor,
                            uv: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """`(B, N, J)`: how much of each node's attention mass lands on each joint (probes)."""
        w, _ = self.attention(px.float(), py.float(), mask, uv.float(), z.float())
        return torch.einsum("bqn,qj->bnj", w, self.lbs_q)
