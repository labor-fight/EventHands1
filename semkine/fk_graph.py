#!/usr/bin/env python3
r"""S37. The previous state's MANO FK *is* the graph; the events are its observations.

The FK does exactly two things here. It fixes the graph -- 16 joint nodes, 192 surface-vertex
nodes sampled per joint, one background node, with edges along the mesh, along the skinning
weights and along the kinematic tree -- and it says where each node projects, so every event can be
handed to the node under it. Nothing geometric enters the features: a node's input is only what
the events assigned to it say,

    [ log-count share, mean offset (du, dv)/r, offset spread/r, mean time, polarity balance,
      local flow (b_u, b_v) ]

where `(b_u, b_v)` is the closed-form least-squares slope of the events' offsets against their
timestamps -- the first-order motion the events report at that node -- and the edge feature is the
edge *type* (mesh / skinning / kinematic, one-hot), which happens to be exactly `EdgeConv`'s three
edge channels. This is the sparse dual of render-and-compare: instead of drawing the state and
reading it at the events, the events are read at the state.

All events of a packet are used (an `index_add_` over the whole stream, no node subsampling), and
the graph is the same 209 nodes for every packet, so the cost is linear in the event count with a
tiny constant and no k-NN over events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .event_gnn import EdgeConv
from .events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y

N_JOINTS = 16
OBS_DIM = 8
#: edge types, one-hot on EdgeConv's three edge channels
E_MESH, E_LBS, E_KIN = 0, 1, 2
#: events per chunk of the (events x nodes) distance block; ~170 MB at 208 nodes
ASSIGN_CHUNK = 200_000


# --------------------------------------------------------------------------- graph spec
@dataclass
class FKGraphSpec:
    """Fixed graph over the previous state's FK; built once from the MANO asset."""

    vert_ids: torch.Tensor      # (V,) long, sampled vertex ids in node order 16..16+V-1
    vert_joint: torch.Tensor    # (V,) long, argmax skinning joint of each sampled vertex
    lbs_wn: torch.Tensor        # (16, 778) normalised skinning weights^T: joint node = surface centroid
    idx: torch.Tensor           # (N, K) long neighbour table (self-index where padded)
    etype: torch.Tensor         # (N, K, 3) float one-hot edge type (zero where padded)
    emask: torch.Tensor         # (N, K) float, 1 where a real edge
    n_verts: int
    k: int

    @property
    def n_nodes(self) -> int:           # joints + vertices + background
        return N_JOINTS + self.n_verts + 1

    @property
    def background(self) -> int:
        return N_JOINTS + self.n_verts

    @staticmethod
    def from_mano(mano, n_verts: int = 192, k_mesh: int = 6, k_total: int = 8) -> "FKGraphSpec":
        W = mano.weights.detach().float().cpu()            # (778, 16)
        V = mano.v_template.detach().float().cpu()         # (778, 3)
        parents = mano.kintree_table[0].detach().cpu().clone()
        parents[0] = -1
        parents = parents.long()
        per = n_verts // N_JOINTS
        assert per * N_JOINTS == n_verts, "n_verts must be a multiple of 16"
        arg = W.argmax(1)
        vert_ids, vert_joint = [], []
        for j in range(N_JOINTS):
            cand = (arg == j).nonzero().squeeze(1)
            if cand.numel() < per:                       # not the case for MANO, but stay safe
                extra = W[:, j].argsort(descending=True)
                extra = extra[~torch.isin(extra, cand)][: per - cand.numel()]
                cand = torch.cat([cand, extra])
            vert_ids.append(_farthest_point(V[cand], per, start=int(W[cand, j].argmax()), ids=cand))
            vert_joint.append(torch.full((per,), j, dtype=torch.long))
        vert_ids = torch.cat(vert_ids)
        vert_joint = torch.cat(vert_joint)
        N = N_JOINTS + n_verts + 1
        bg = N - 1
        nbrs = [[] for _ in range(N)]                    # (neighbour, type)

        # vertex <-> vertex: rest-pose 3D k-NN among the sampled vertices
        P = V[vert_ids]
        d = torch.cdist(P, P)
        d.fill_diagonal_(float("inf"))
        knn = d.topk(k_mesh, dim=1, largest=False).indices
        for i in range(n_verts):
            for jn in knn[i].tolist():
                nbrs[N_JOINTS + i].append((N_JOINTS + jn, E_MESH))
        # vertex <-> joint: the two largest skinning weights of the vertex
        top2 = W[vert_ids].topk(2, dim=1).indices
        for i in range(n_verts):
            for j in top2[i].tolist():
                nbrs[N_JOINTS + i].append((j, E_LBS))
        # joint <-> joint: kinematic tree, both directions
        for j in range(N_JOINTS):
            p = int(parents[j])
            if p >= 0:
                nbrs[j].append((p, E_KIN))
                nbrs[p].append((j, E_KIN))
        # background <-> wrist and the five finger bases
        for j in (0, 1, 4, 7, 10, 13):
            nbrs[bg].append((j, E_KIN))
            nbrs[j].append((bg, E_KIN))
        # joint <- its member vertices, by skinning weight, until K is full
        for j in range(N_JOINTS):
            room = k_total - len(nbrs[j])
            if room > 0:
                order = W[vert_ids, j].argsort(descending=True)
                for i in order.tolist():
                    if room == 0:
                        break
                    if (N_JOINTS + i, E_LBS) not in nbrs[j]:
                        nbrs[j].append((N_JOINTS + i, E_LBS))
                        room -= 1
        idx = torch.arange(N).unsqueeze(1).repeat(1, k_total)
        etype = torch.zeros(N, k_total, 3)
        emask = torch.zeros(N, k_total)
        for n in range(N):
            lst = nbrs[n][:k_total]
            for s, (m, t) in enumerate(lst):
                idx[n, s] = m
                etype[n, s, t] = 1.0
                emask[n, s] = 1.0
        lbs_wn = (W / W.sum(0, keepdim=True).clamp(min=1e-8)).T.contiguous()
        return FKGraphSpec(vert_ids=vert_ids, vert_joint=vert_joint, lbs_wn=lbs_wn, idx=idx,
                           etype=etype, emask=emask, n_verts=n_verts, k=k_total)

    def node_points(self, verts: torch.Tensor) -> torch.Tensor:
        """3D node positions `(B, 16 + V, 3)` for posed vertices `(B, 778, 3)`: joint surface
        centroids first, then the sampled vertices. The background node has no position."""
        joints = torch.einsum("jv,bvc->bjc", self.lbs_wn.to(verts), verts.float())
        return torch.cat([joints, verts.float()[:, self.vert_ids.to(verts.device)]], dim=1)


def _farthest_point(P: torch.Tensor, n: int, start: int, ids: torch.Tensor) -> torch.Tensor:
    """Deterministic farthest-point sampling of `n` rows of `P`; returns the chosen global ids."""
    chosen = [start]
    dmin = (P - P[start]).norm(dim=1)
    for _ in range(n - 1):
        nxt = int(dmin.argmax())
        chosen.append(nxt)
        dmin = torch.minimum(dmin, (P - P[nxt]).norm(dim=1))
    return ids[torch.tensor(chosen, dtype=torch.long)]


# --------------------------------------------------------------------------- observations
def _segment_sums(key: torch.Tensor, cols: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Per-bin column sums, bit-reproducible across runs.

    A float32 `index_add_` sums in the atomics' data-race order, and the recursive evaluator
    amplifies that ~1e-7 jitter past the 0.05 mm reproduction gate that pins every recorded number
    (measured on this arm: 0.0755 mm selection-vs-rerun drift). Accumulating in float64 and casting
    back removes it: the order still varies but float64 rounding is ~1e-13, far below the float32
    step, so every run casts to the same float32 (verified 5/5 identical, 1 ms at 3M events).
    """
    out = torch.zeros(n_bins, cols.shape[1], device=cols.device, dtype=torch.float64)
    if cols.shape[0]:
        out.index_add_(0, key, cols.double())
    return out.float()


def assign_and_observe(events: torch.Tensor, ptr: torch.Tensor, node_uv: torch.Tensor,
                       delta_t_s: torch.Tensor, band_px: float,
                       chunk: int = ASSIGN_CHUNK,
                       node_mask: Optional[torch.Tensor] = None,
                       assign_pre: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Hand every event to the node under it and summarise each node's events.

    events: `(E, 5)` [batch, x, y, t_rel_s, p]; ptr: `(B+1,)`; node_uv: `(B, M, 2)` projected pixels
    of the M = 16 + V positioned nodes; delta_t_s: `(B,)` packet length. Events farther than
    `band_px` from every node go to the background node (index M), whose reference point is the
    projected hand centroid and whose offset scale is 4 r. `node_mask (B, M)`, when given, marks
    the nodes an event may be handed to (the mesh graph passes its visible vertices); a masked
    node never wins the nearest search and so never sees an event. The centroid is over all nodes.
    `assign_pre (E,)`, when given, is the node of every event already decided (M = background) and
    the nearest-node search is skipped -- the mesh graph gets it from a per-pixel lookup table.

    Returns `(obs (B, M+1, OBS_DIM) float32, assign (E,) long)`.
    """
    B, M, _ = node_uv.shape
    N = M + 1
    dev = events.device
    obs = torch.zeros(B, N, OBS_DIM, device=dev, dtype=torch.float32)
    E = int(events.shape[0])
    assign = torch.full((E,), M, device=dev, dtype=torch.long)
    if E == 0:
        return obs, assign
    r = float(band_px)
    uv = node_uv.float()
    centroid = uv.mean(1)                                            # (B, 2)
    blocked = None if node_mask is None else ~node_mask.bool()       # (B, M)
    b_all = events[:, EV_BATCH].long()
    x_all = events[:, EV_X].float()
    y_all = events[:, EV_Y].float()
    t_all = events[:, EV_T].float() / delta_t_s.float().clamp_min(1e-6)[b_all]
    p_all = events[:, EV_P].float() * 2.0 - 1.0
    # sums per (packet, node): 1, t, t^2, du, dv, t*du, t*dv, du^2, dv^2, p
    S = torch.zeros(B * N, 10, device=dev, dtype=torch.float32)
    if assign_pre is not None:
        # the nearest-node search was done elsewhere (a pixel lookup table); only the offsets
        node = assign_pre.long()
        inband = node < M
        ref = torch.where(inband.unsqueeze(1), uv[b_all, node.clamp(max=M - 1)], centroid[b_all])
        scale = torch.where(inband, torch.full_like(x_all, r), torch.full_like(x_all, 4.0 * r))
        du_all = (x_all - ref[:, 0]) / scale
        dv_all = (y_all - ref[:, 1]) / scale
        assign = node
    else:
        du_all = torch.empty(E, device=dev)
        dv_all = torch.empty(E, device=dev)
        for i0 in range(0, E, chunk):
            sl = slice(i0, i0 + chunk)
            b = b_all[sl]
            d2 = (uv[b, :, 0] - x_all[sl, None]).square() + (uv[b, :, 1] - y_all[sl, None]).square()
            if blocked is not None:
                d2 = d2.masked_fill(blocked[b], float("inf"))
            dmin, node = d2.min(dim=1)
            inband = dmin <= r * r
            node = torch.where(inband, node, torch.full_like(node, M))
            ref = torch.where(inband.unsqueeze(1), uv[b, node.clamp(max=M - 1)], centroid[b])
            scale = torch.where(inband, torch.full_like(dmin, r), torch.full_like(dmin, 4.0 * r))
            du_all[sl] = (x_all[sl] - ref[:, 0]) / scale
            dv_all[sl] = (y_all[sl] - ref[:, 1]) / scale
            assign[sl] = node
    t = t_all
    key = b_all * N + assign
    cols = torch.stack([torch.ones_like(t), t, t * t, du_all, dv_all, t * du_all, t * dv_all,
                        du_all * du_all, dv_all * dv_all, p_all], dim=1)
    S = _segment_sums(key, cols, B * N).view(B, N, 10)
    n = S[..., 0]
    live = n > 0
    inv_n = 1.0 / n.clamp_min(1.0)
    mean_t = S[..., 1] * inv_n
    mean_du = S[..., 3] * inv_n
    mean_dv = S[..., 4] * inv_n
    var = (S[..., 7] * inv_n - mean_du.square()).clamp_min(0) + (S[..., 8] * inv_n - mean_dv.square()).clamp_min(0)
    spread = var.sqrt()
    mean_p = S[..., 9] * inv_n
    den = n * S[..., 2] - S[..., 1].square()
    ok = (n >= 3) & (den > 1e-6)
    b_u = torch.where(ok, (n * S[..., 5] - S[..., 1] * S[..., 3]) / den.clamp_min(1e-6), torch.zeros_like(den))
    b_v = torch.where(ok, (n * S[..., 6] - S[..., 1] * S[..., 4]) / den.clamp_min(1e-6), torch.zeros_like(den))
    total = (ptr[1:] - ptr[:-1]).clamp_min(1).float().to(dev)          # (B,)
    share = torch.log1p(n) / torch.log1p(total).unsqueeze(1)
    obs = torch.stack([share, mean_du, mean_dv, spread, mean_t, mean_p,
                       b_u.clamp(-4.0, 4.0), b_v.clamp(-4.0, 4.0)], dim=-1)
    return obs * live.unsqueeze(-1).float(), assign


# --------------------------------------------------------------------------- encoder
class FKGraphEncoder(nn.Module):
    """Observation embedding (+ learned node identity) followed by `EdgeConv` rounds on the fixed
    FK graph. Returns `(B, N, hidden)`; the caller reads joints `0..15` and the background node."""

    def __init__(self, spec: FKGraphSpec, hidden: int = 128, n_layers: int = 3, node_id: bool = True,
                 obs_dim: int = OBS_DIM):
        super().__init__()
        self.hidden = int(hidden)
        self.n_nodes = spec.n_nodes
        self.obs_embed = nn.Linear(int(obs_dim), self.hidden)
        self.node_id = nn.Embedding(self.n_nodes, self.hidden) if node_id else None
        if self.node_id is not None:
            nn.init.normal_(self.node_id.weight, std=0.02)
        self.layers = nn.ModuleList([EdgeConv(self.hidden) for _ in range(int(n_layers))])
        self.register_buffer("idx", spec.idx.clone(), persistent=False)
        self.register_buffer("etype", spec.etype.clone(), persistent=False)
        self.register_buffer("emask", spec.emask.clone(), persistent=False)

    def forward(self, obs: torch.Tensor, edge_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`edge_feat (B, N, K, 3)`, when given, rides on `EdgeConv`'s three edge channels instead
        of the fixed one-hot edge type -- the S38 mesh graph passes the 3D vector to each
        neighbour (`semkine.mesh_graph.edge_vectors`). Without it the behaviour is S37's."""
        B = obs.shape[0]
        h = self.obs_embed(obs.to(self.obs_embed.weight.dtype))
        if self.node_id is not None:
            h = h + self.node_id.weight.unsqueeze(0).to(h.dtype)
        h = torch.relu(h)
        idx = self.idx.unsqueeze(0).expand(B, -1, -1)
        if edge_feat is None:
            dp = self.etype.unsqueeze(0).expand(B, -1, -1, -1).to(h.dtype)
        else:
            dp = edge_feat.to(h.dtype)
        emask = self.emask.unsqueeze(0).expand(B, -1, -1).to(h.dtype)
        for layer in self.layers:
            h = h + layer(h, idx, dp, emask)
        return h
