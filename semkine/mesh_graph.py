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

S38 (2026-09-20, `docs/S38_MESH3D_PREREG.md`) relaxes that law in one direction: *relative* 3D
geometry of the previous state may enter as **edge** features -- the vector to a mesh neighbour
(`edge_vectors`), the lever arm of a vertex about the root joint (`RigidNode`), the position of a
part relative to the root joint (`part_positions`). The graph, the assignment, the visibility and
the pool stay as above. What it buys: the S37 mesh graph's message `relu(W [h_j - h_i ; 1, 0, 0])`
is permutation-invariant over the 1-ring (a learned isotropic Laplacian), and reading a rotation
from a displacement field is bilinear in the field and the lever arm, `omega ~ sum_i r_i x delta_i`,
with `r_i` in the *camera* frame -- a quantity the S37 evidence path never sees and a single linear
root head cannot form. The S37 result that motivates this (RA-vs-rotation correlation 0.96, seeds
15 deg / 28 deg apart) is in `docs/S37_MESHGRAPH_PREREG.md` section 7.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F
from torch import nn

from .fk_graph import E_MESH, N_JOINTS

#: metres -> centimetres: the unit every S38 geometric feature is expressed in (edges ~0.8,
#: lever arms up to ~18), so they sit in the range of the O(1) observation channels
GEO_SCALE = 100.0

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


# --------------------------------------------------------------------------- 3D geometry (S38)
@torch.no_grad()
def edge_vectors(verts: torch.Tensor, idx: torch.Tensor, emask: torch.Tensor,
                 scale: float = GEO_SCALE) -> torch.Tensor:
    """`(B, N, K, 3)`: the 3D vector `X_j - X_i` along every real edge of the neighbour table, in
    the camera frame, metres x `scale`; zero on padded slots and on the background row.

    This is the edge feature that replaces the constant one-hot edge type of the S37 mesh graph
    (edge type `E_MESH` is the only type there, so `EdgeConv`'s three edge channels were carrying
    `[1, 0, 0]` on every edge). With it the message `relu(W [h_j - h_i ; X_j - X_i])` knows in which
    direction on the surface the neighbour lies -- the AEGNN / S36 `dp` role, in 3D. Gather-only,
    so bit-reproducible across runs.
    """
    B, V, _ = verts.shape
    N, K = idx.shape
    X = verts.float()
    if N > V:                                       # the background node has no position
        X = torch.cat([X, X.new_zeros(B, N - V, 3)], dim=1)
    Xj = X.gather(1, idx.reshape(1, N * K, 1).expand(B, N * K, 3)).reshape(B, N, K, 3)
    dp = (Xj - X.unsqueeze(2)) * emask.to(X.dtype).view(1, N, K, 1)
    return dp * float(scale)


@torch.no_grad()
def lever_arms(verts: torch.Tensor, pivot: torch.Tensor, scale: float = GEO_SCALE) -> torch.Tensor:
    """`(B, V, 3)`: `X_i - pivot` in the camera frame, metres x `scale`. The pivot is the posed
    MANO root joint, the point the root rotation delta acts about."""
    return (verts.float() - pivot.float().unsqueeze(1)) * float(scale)


@torch.no_grad()
def part_positions(verts: torch.Tensor, lbs_weights: torch.Tensor, pivot: torch.Tensor,
                   scale: float = GEO_SCALE) -> torch.Tensor:
    """`(B, J, 3)`: the skinning-weighted centroid of every part, `sum_v W_vj X_v / sum_v W_vj`,
    relative to the pivot, metres x `scale` -- the place in space the pooled evidence `e_j` of
    `lbs_pool_evidence` speaks for. Over *all* vertices (a part's position does not depend on
    what was seen). Same normalised weights as `FKGraphSpec.lbs_wn`."""
    W = lbs_weights.float()
    wn = W / W.sum(0, keepdim=True).clamp_min(1e-8)                                   # (V, J)
    c = torch.einsum("vj,bvc->bjc", wn, verts.float())
    return (c - pivot.float().unsqueeze(1)) * float(scale)


class RigidNode(nn.Module):
    r"""One node for the whole hand, fed by every vertex that saw an event, with the vertex's 3D
    lever arm on the edge:

        g = MLP( mean_{i : has_i} relu( W [ h_i ; r_i ] ) ),      r_i = (X_i - X_root) * scale

    Exactly zero when no vertex qualifies, so an event-free packet (and the `ablate_evidence`
    gate, which clears `has`) contributes nothing to the root. This is the long-range pathway the
    S37 mesh graph lacked (three 1-ring hops reach ~3 cm; wrist to fingertip is 22 hops) *and* the
    place a rotation becomes computable: a rotation about the root moves vertex `i` by
    `omega x r_i`, so the per-edge nonlinearity over `[h_i ; r_i]` can form the moment
    `r_i x delta_i` before the mean -- a statistic the sixteen skinning-weighted part means of
    `lbs_pool_evidence` average away inside each part. The mean is a masked sum (no atomics), so
    the recursive evaluator sees the same bits every run.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self.msg = nn.Linear(self.hidden + 3, self.hidden)
        self.out = nn.Sequential(nn.Linear(self.hidden, self.hidden), nn.ReLU(inplace=True))

    def forward(self, h: torch.Tensor, r: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h: `(B, V, C)` vertex features; r: `(B, V, 3)` lever arms; mask: `(B, V)` bool.
        Returns `(B, C)`."""
        m = torch.relu(self.msg(torch.cat([h, r.to(h.dtype)], dim=-1)))
        mf = mask.to(h.dtype).unsqueeze(-1)
        g = (m * mf).sum(1) / mf.sum(1).clamp_min(1.0)
        return self.out(g) * mask.any(1, keepdim=True).to(h.dtype)


class PartLever(nn.Module):
    r"""The part-level analogue of `RigidNode`, for the root head alone (S38b):

        u_j = relu( W [ e_j ; r_j ] ) * seen_j,      r_j = (part centroid_j - X_root) * scale

    one shared linear map over the sixteen `(evidence, lever arm)` pairs, no pooling -- the root
    head reads `u_0..u_15` in joint order next to the evidences it already read. `seen_j` is
    "the part saw an event" (`e_j != 0`), so an unseen part and the `ablate_evidence` gate give
    exactly zero here as they do in the pool. The graph is untouched: this is the smallest change
    that lets the root form "orientation-dependent coefficient x displacement", which a linear
    layer over the sixteen means cannot.
    """

    def __init__(self, ev_dim: int, hidden: int):
        super().__init__()
        self.ev_dim, self.hidden = int(ev_dim), int(hidden)
        self.lin = nn.Linear(self.ev_dim + 3, self.hidden)

    def forward(self, e: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """e: `(B, J, ev_dim)` pooled evidences; r: `(B, J, 3)` part lever arms. `(B, J, hidden)`."""
        seen = (e.abs().sum(-1, keepdim=True) > 0).to(e.dtype)
        return torch.relu(self.lin(torch.cat([e, r.to(e.dtype)], dim=-1))) * seen


__all__ = ["ASSIGN_CHUNK", "GEO_SCALE", "MeshGraphSpec", "N_JOINTS", "PartLever", "RigidNode",
           "assign_events_by_lut", "edge_vectors", "facing_camera", "lbs_pool_evidence",
           "lever_arms", "nearest_node_lut", "part_positions", "visible_vertices",
           "zbuffer_visible"]
