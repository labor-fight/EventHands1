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

from typing import Tuple

import torch

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


def surface_patches(v_template: torch.Tensor, n_patches: int = 64) -> torch.Tensor:
    """`(V,)` long: a fixed partition of the MANO vertices into `n_patches` surface patches --
    deterministic farthest-point sampling on the rest-pose template (start at vertex 0), every
    vertex assigned to its nearest centre. A property of the asset, independent of the state."""
    P = v_template.detach().float().cpu()
    centres = [0]
    dmin = (P - P[0]).norm(dim=1)
    for _ in range(int(n_patches) - 1):
        nxt = int(dmin.argmax())
        centres.append(nxt)
        dmin = torch.minimum(dmin, (P - P[nxt]).norm(dim=1))
    return torch.cdist(P, P[centres]).argmin(1)


#: channels per patch of the coverage map: routed-node share, visible-vertex share, band-node
#: share, mean band offset (du, dv) in bands; plus one global channel (visible fraction)
COVMAP_CHANNELS = 5


def coverage_map(px: torch.Tensor, py: torch.Tensor, mask: torch.Tensor, dist: torch.Tensor,
                 vid: torch.Tensor, uv: torch.Tensor, vis: torch.Tensor, patch: torch.Tensor,
                 band_px: float, n_patches: int, edge_px: float = 2.0
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""S39. The footprint comparison the root needs, on the surface: per patch, where the routed
    events are against where the visible prev surface is.

    For patch `k` (a fixed set of vertices, `surface_patches`):

        share_k  = #routed live nodes whose vertex is in k / #routed live nodes
        vis_k    = #visible vertices in k / #visible vertices
        band_k   = #routed nodes in k farther than `edge_px` from their vertex / #routed nodes
        (du, dv)_k = mean pixel offset (node - vertex) of those band nodes / band_px

    plus one global scalar, #visible vertices / V. `share_k - vis_k` is the discrepancy between
    the event footprint and the prev footprint on that patch: a rotation or translation error of
    prev shows up as an antisymmetric pattern over the patches (events pile up on one side, the
    surface is bare on the other), which the per-joint mean/max/coverage pool averages away.
    Every channel is evidence x visibility: a packet that routes nothing gives exactly zero
    (`any_routed` = 0), and no geometric quantity enters additively.

    Measured before it was built (debug session 2026-09-21, routed 3407, zgz closed loop): a ridge
    readout of these 257 numbers removes a third of the root rotation error the loop carries
    (10.5 -> 6.9 deg, cos 0.82) and two thirds of the x/y translation error (R^2 0.66 / 0.75) on
    the dense sequence; the root head of the routed arm removes none (10.6 -> 10.7 deg).

    px, py: `(B, N)` node pixels; mask: `(B, N)` live; dist, vid: `(B, N)` from
    `route_front_vertex_lbs`; uv: `(B, V, 2)`; vis: `(B, V)` bool visible vertices;
    patch: `(V,)` long. Returns `(feat (B, n_patches * COVMAP_CHANNELS + 1) float32,
    any_routed (B, 1) float32)`.
    """
    B, N = mask.shape
    V = vis.shape[1]
    K = int(n_patches)
    dev = px.device
    routed = (dist <= float(band_px)) & mask.bool()                                   # (B, N)
    routed_f = routed.to(torch.float32)
    pk = patch.to(dev)[vid.clamp(0, V - 1)]                                            # (B, N)
    boff = torch.arange(B, device=dev).unsqueeze(1) * K
    key = (pk + boff).reshape(-1)
    n_routed = routed_f.sum(1)                                                          # (B,)
    share = torch.zeros(B * K, device=dev, dtype=torch.float32)
    share.index_add_(0, key, routed_f.reshape(-1))
    share = share.view(B, K) / n_routed.clamp_min(1.0).unsqueeze(1)
    vk = torch.zeros(B * K, device=dev, dtype=torch.float32)
    vkey = (patch.to(dev).unsqueeze(0).expand(B, V) + boff).reshape(-1)
    vk.index_add_(0, vkey, vis.to(torch.float32).reshape(-1))
    n_vis = vis.to(torch.float32).sum(1)
    vis_share = vk.view(B, K) / n_vis.clamp_min(1.0).unsqueeze(1)
    band = routed & (dist >= float(edge_px))
    band_f = band.to(torch.float32)
    du = (px.float() - uv[..., 0].gather(1, vid.clamp(0, V - 1))) * band_f
    dv = (py.float() - uv[..., 1].gather(1, vid.clamp(0, V - 1))) * band_f
    bn = torch.zeros(B * K, device=dev, dtype=torch.float32)
    bn.index_add_(0, key, band_f.reshape(-1))
    bdu = torch.zeros(B * K, device=dev, dtype=torch.float32)
    bdu.index_add_(0, key, du.reshape(-1))
    bdv = torch.zeros(B * K, device=dev, dtype=torch.float32)
    bdv.index_add_(0, key, dv.reshape(-1))
    bn, bdu, bdv = bn.view(B, K), bdu.view(B, K), bdv.view(B, K)
    band_share = bn / n_routed.clamp_min(1.0).unsqueeze(1)
    mdu = bdu / bn.clamp_min(1.0) / float(band_px)
    mdv = bdv / bn.clamp_min(1.0) / float(band_px)
    any_routed = (n_routed > 0).to(torch.float32).unsqueeze(1)
    feat = torch.cat([share, vis_share, band_share, mdu, mdv, (n_vis / V).unsqueeze(1)], dim=1)
    return feat * any_routed, any_routed


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
