#!/usr/bin/env python3
r"""S37 mesh graph (the user's drawing, 2026-09-18): the previous state's full MANO mesh *is* the
graph, the events are its observations, skinning weights pool the graph into the joints.

    prev 51D --MANO FK--> 778 posed vertices = nodes, faces = edges
    events   --nearest visible vertex (<= band px, else background)--> per-node observations
    EdgeConv x L on the mesh --> per-vertex features h_v
    LBS pooling  e_j = [ sum_v W_vj has_v h_v / sum_v W_vj has_v || max_{v: argmax W_v = j, has_v} h_v
                         || cov_j = sum_v W_vj has_v / sum_v W_vj vis_v ]
    joint head k reads e_{k+1} (+ its prev angle); root reads e_0..e_15 in order + the background
    Delta (51D) --> x_k = prev + Delta --> next FK

Against `fk_graph` (16 joint nodes + 192 sampled vertices + background, typed kNN / LBS / kinematic
edges, joint nodes read directly) this arm keeps every vertex, takes the edges from the mesh faces
(true 1-ring, max degree 8), and puts the joint aggregation outside the graph as a fixed
skinning-weight pool instead of learned LBS-type edges. Two things follow from having every
vertex as a node: the back of the hand projects onto the same pixels as the front, so nodes need a
visibility test before events are handed to them (`visible_vertices` = back-face culling on the
posed mesh + a point-splat z-buffer for occlusion by another part, the sparse stand-in for a
rasterizer); and there are ~4x more nodes than events per node, so the pool carries a per-joint
coverage scalar and pools only over vertices that saw an event. The state is the 51D MANO parameter vector, not joint positions: FK needs the
rotations (twist about a bone is invisible in joint positions), so the joint heads regress
per-joint axis-angle deltas and the root head the root rotation / translation delta.

Nothing geometric enters the node features (same law as `fk_graph`): a node's input is only what
its events say. The graph, the visibility and the pooling weights are fixed functions of the
previous state and of the MANO asset; the optimizer cannot widen any of them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F

from .fk_graph import E_MESH, N_JOINTS

#: events per chunk of the (events x nodes) distance block; ~150 MB at 778 nodes
ASSIGN_CHUNK = 50_000


# --------------------------------------------------------------------------- graph spec
@dataclass
class MeshGraphSpec:
    """Fixed graph over the full MANO mesh; built once from the asset.

    Nodes `0..777` are the vertices (asset order), node `778` is the background. Edges are the
    face 1-ring, both directions, typed `E_MESH`; the background node has no edges (it reaches the
    output through the root head only)."""

    vert_joint: torch.Tensor    # (778,) long, argmax skinning joint of every vertex
    idx: torch.Tensor           # (N, K) long neighbour table (self-index where padded)
    etype: torch.Tensor         # (N, K, 3) float one-hot edge type (zero where padded)
    emask: torch.Tensor         # (N, K) float, 1 where a real edge
    n_verts: int
    k: int

    @property
    def n_nodes(self) -> int:
        return self.n_verts + 1

    @property
    def background(self) -> int:
        return self.n_verts

    @staticmethod
    def from_mano(mano) -> "MeshGraphSpec":
        W = mano.weights.detach().float().cpu()            # (778, 16)
        faces = mano.f.detach().long().cpu()               # (1538, 3)
        V = int(W.shape[0])
        e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        e = torch.unique(torch.cat([e, e.flip(1)]), dim=0)
        e = e[e[:, 0] != e[:, 1]]
        deg = torch.bincount(e[:, 0], minlength=V)
        K = int(deg.max())
        N = V + 1
        idx = torch.arange(N).unsqueeze(1).repeat(1, K)
        etype = torch.zeros(N, K, 3)
        emask = torch.zeros(N, K)
        # e is sorted by (source, target); the running rank within each source is the slot
        order = torch.arange(e.shape[0])
        first = torch.cat([torch.zeros(1, dtype=torch.long), deg.cumsum(0)[:-1]])
        slot = order - first[e[:, 0]]
        idx[e[:, 0], slot] = e[:, 1]
        etype[e[:, 0], slot, E_MESH] = 1.0
        emask[e[:, 0], slot] = 1.0
        return MeshGraphSpec(vert_joint=W.argmax(1), idx=idx, etype=etype, emask=emask,
                             n_verts=V, k=K)


# --------------------------------------------------------------------------- visibility
@torch.no_grad()
def facing_camera(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """`(B, V)` bool: the vertex's area-weighted normal points toward the camera at the origin.

    Back-face culling on the posed mesh -- exact for the far side of a closed surface (the back of
    a finger, the back of the palm), blind to occlusion by another part (a finger in front of the
    palm), which `zbuffer_visible` handles.
    """
    v0, v1, v2 = verts[:, faces[:, 0]], verts[:, faces[:, 1]], verts[:, faces[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1).double()                              # (B, F, 3)
    # float64 accumulation: a float32 `index_add_` sums in the atomics' data-race order, and a
    # vertex whose normal is grazing (n . p ~ 0) could flip visibility between two runs of the same
    # state -- the recursive evaluator amplifies exactly that (fk_graph `_segment_sums`, 0.0755 mm)
    n = torch.zeros(verts.shape, device=verts.device, dtype=torch.float64)
    for i in range(3):
        n.index_add_(1, faces[:, i], fn)
    return (n * verts.double()).sum(-1) < 0                                           # view dir = -p


@torch.no_grad()
def zbuffer_visible(uv: torch.Tensor, z: torch.Tensor, height: int, width: int,
                    front_px: float = 1.0, z_tol: float = 0.01) -> torch.Tensor:
    """`(B, V)` bool: the vertex projects inside the frame and no vertex within `front_px`
    (Chebyshev, in pixels) of its pixel is closer to the camera by more than `z_tol` metres.

    A point-splat z-buffer (nearest depth per pixel) min-pooled over a `(2 front_px + 1)^2`
    window, then read back at every vertex: the same test `MeshQuery.visible` does with a dense
    `(V, V)` distance block, at `O(B H W)` instead of `O(B V^2)`. Vertices outside the frame are
    not visible: no event can fall on them.
    """
    B, V = z.shape
    zf = z.float()
    ui = uv[..., 0].float().round().long()
    vi = uv[..., 1].float().round().long()
    inside = (zf > 1e-6) & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    pix = (vi.clamp(0, height - 1) * width + ui.clamp(0, width - 1))                 # (B, V)
    zbuf = torch.full((B, height * width), float("inf"), device=z.device, dtype=torch.float32)
    zbuf.scatter_reduce_(1, pix, zf.masked_fill(~inside, float("inf")), reduce="amin")
    r = int(round(float(front_px)))
    if r > 0:
        zbuf = -F.max_pool2d(-zbuf.view(B, 1, height, width), kernel_size=2 * r + 1, stride=1,
                             padding=r).view(B, height * width)
    zref = zbuf.gather(1, pix)                                                        # (B, V)
    return inside & (zf <= zref + float(z_tol))


@torch.no_grad()
def nearest_node_lut(uv: torch.Tensor, node_mask: torch.Tensor, height: int, width: int,
                     band_px: float) -> torch.Tensor:
    """`(B, H, W)` long: the nearest unmasked node of every pixel, or `M` (= background) when
    none lies within `band_px`.

    Jump flooding (Rong & Tan, I3D 2006) on the pixel grid: the node ids are splatted at their
    rounded pixels, then every pixel looks at its eight neighbours at strides 16, 8, 4, 2, 1 and
    adopts the neighbour's node if that node is closer to the pixel *centre* than its own. Five
    passes reach 31 px, enough for a 16 px band. The cost is `O(B H W)` -- independent of both the
    node count and the event count -- where the brute-force `(events x nodes)` block of
    `fk_graph.assign_and_observe` is 4.6x slower at 778 nodes than at 209 (1.83 s against 0.40 s
    per 20 M events, measured 2026-09-18). Per-pixel exactness against the brute force is checked
    in `tests/test_s37_mesh_graph.py`.
    """
    B, M, _ = uv.shape
    dev = uv.device
    uvf = uv.float()
    ui = uvf[..., 0].round().long()
    vi = uvf[..., 1].round().long()
    ok = node_mask.bool() & (ui >= 0) & (ui < width) & (vi >= 0) & (vi < height)
    flat = (torch.arange(B, device=dev).unsqueeze(1) * (height * width)
            + vi.clamp(0, height - 1) * width + ui.clamp(0, width - 1))                # (B, M)
    ids = torch.arange(M, device=dev).unsqueeze(0).expand(B, M)
    seed = torch.full((B * height * width,), M, device=dev, dtype=torch.long)
    seed.scatter_reduce_(0, flat[ok], ids[ok], reduce="amin")                            # collisions: lowest id
    seed = seed.view(B, height, width)
    # node pixel table with a sentinel row (index M) far away, so "no node" never wins
    uv_ext = torch.cat([uvf, torch.full((B, 1, 2), 1e6, device=dev)], dim=1)             # (B, M+1, 2)
    py, px = torch.meshgrid(torch.arange(height, device=dev, dtype=torch.float32),
                            torch.arange(width, device=dev, dtype=torch.float32), indexing="ij")

    def d2_of(s):
        p = uv_ext.gather(1, s.reshape(B, -1, 1).expand(-1, -1, 2)).view(B, height, width, 2)
        return (p[..., 0] - px).square() + (p[..., 1] - py).square()

    best = d2_of(seed)
    for k in (16, 8, 4, 2, 1):
        pad = F.pad(seed, (k, k, k, k), value=M)
        for dy in (-k, 0, k):
            for dx in (-k, 0, k):
                if dy == 0 and dx == 0:
                    continue
                cand = pad[:, k + dy: k + dy + height, k + dx: k + dx + width]
                d2c = d2_of(cand)
                better = d2c < best
                seed = torch.where(better, cand, seed)
                best = torch.where(better, d2c, best)
    return seed.masked_fill(best > float(band_px) ** 2, M)


@torch.no_grad()
def visible_vertices(verts: torch.Tensor, uv: torch.Tensor, z: torch.Tensor, faces: torch.Tensor,
                     height: int, width: int, front_px: float = 1.0, z_tol: float = 0.01
                     ) -> torch.Tensor:
    """`(B, V)` bool: faces the camera *and* passes the z-buffer test. Measured against a triangle
    z-buffer on 300 training states (2026-09-18): the two tests together leave ~5 back vertices
    per hand marked visible (the z-test alone at a 7 x 7 window: 39; normals alone: 13) and hide
    ~70 silhouette vertices whose events the neighbouring front vertex 1-2 px away receives."""
    return facing_camera(verts, faces) & zbuffer_visible(uv, z, height, width, front_px, z_tol)


# --------------------------------------------------------------------------- pooling
def lbs_pool_evidence(h: torch.Tensor, lbs_weights: torch.Tensor, vis: torch.Tensor,
                      has: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Skinning-weight pool of vertex features into per-joint evidence.

    h: `(B, V, C)` vertex features after message passing; lbs_weights: `(V, J)`; vis, has:
    `(B, V)` bool -- vertex visible / vertex saw at least one event.

    Returns `(e (B, J, 2C+1), count (B, J))` with
        e_j = [ sum_v W_vj has_v h_v / sum_v W_vj has_v   (weighted mean over observed vertices)
             || max over {v : argmax_j W_v = j, has_v} h_v (hard-part peak)
             || cov_j = sum_v W_vj has_v / sum_v W_vj vis_v (share of the joint's visible surface
                                                             that saw an event) ]
    and `count_j` = number of observed vertices whose dominant joint is j. A joint none of whose
    vertices saw an event has e_j = 0 exactly, so an event-free packet gives zero evidence.
    """
    B, V, C = h.shape
    J = lbs_weights.shape[1]
    W = lbs_weights.to(h.dtype)
    has_f = has.to(h.dtype)
    a = W.unsqueeze(0) * has_f.unsqueeze(-1)                                          # (B, V, J)
    wsum = a.sum(1)                                                                   # (B, J)
    mean = torch.einsum("bvj,bvc->bjc", a, h) / wsum.clamp_min(1e-6).unsqueeze(-1)
    mean = mean * (wsum > 0).unsqueeze(-1).to(h.dtype)

    j_hard = W.argmax(-1).unsqueeze(0).expand(B, V)                                   # (B, V)
    src = h.masked_fill(~has.unsqueeze(-1), -1e4)
    peak = torch.full((B, J, C), -1e4, device=h.device, dtype=h.dtype)
    peak = peak.scatter_reduce(1, j_hard.unsqueeze(-1).expand(B, V, C), src, reduce="amax")
    count = torch.zeros(B, J, device=h.device, dtype=torch.long)
    count.scatter_add_(1, j_hard, has.long())
    peak = peak * (count > 0).unsqueeze(-1).to(h.dtype)

    vsum = (W.unsqueeze(0) * vis.to(h.dtype).unsqueeze(-1)).sum(1)                    # (B, J)
    cov = (wsum / vsum.clamp_min(1e-6)) * (vsum > 0).to(h.dtype)
    return torch.cat([mean, peak, cov.unsqueeze(-1)], dim=-1), count


@torch.no_grad()
def assign_events_by_lut(events: torch.Tensor, lut: torch.Tensor, background: int) -> torch.Tensor:
    """`(E,)` long node id per event, read off `nearest_node_lut`'s table at the event's rounded
    pixel; events outside the frame go to `background` (the node count, 778 for MANO)."""
    from .events import EV_BATCH, EV_X, EV_Y
    H, W = lut.shape[-2:]
    b = events[:, EV_BATCH].long()
    x = events[:, EV_X].float().round().long().clamp(0, W - 1)
    y = events[:, EV_Y].float().round().long().clamp(0, H - 1)
    return lut[b, y, x]


__all__ = ["ASSIGN_CHUNK", "MeshGraphSpec", "N_JOINTS", "assign_events_by_lut", "facing_camera",
           "lbs_pool_evidence", "nearest_node_lut", "visible_vertices", "zbuffer_visible"]
