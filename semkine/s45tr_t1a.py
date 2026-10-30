r"""S45-TR arm T1a: event-graph node features replace token7 in the relation record
(docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030).

S45's relation encoder reads, per matched record, `[token7, dXYZ, zn, valid, hit]` (13 values); token7 describes the
event alone. T1a replaces token7 by the event's node feature after three rounds of causal kNN message passing over the
packet's sampled events (`semkine/event_gnn.py`, the S36 EventGNN operator: hidden 96, k 8, window 32, 3 layers,
t_scale 1, node attributes = token7). The graph is built from `(x/W, y/H, t_norm)` alone, so it carries no state; the
association, dXYZ / zn / valid / hit, phi, the 778 pooling, the MeshGNN, the readout and the additive 51D update are
S45's. The graph trains end to end with the pose loss; there is no auxiliary loss.

The node features are computed from the sampled tokens S45 already hands `XYZMeshNet.forward` (the same stride
sample, `stride_sample(ptr, 2048)`), which is bit for bit what `EventGNN.forward` computes for those nodes: its graph
coordinates are `(tok[0], tok[1], tok[T_COL] * t_scale)` and its node attributes are the tokens.

Parameters: the EventGNN embed / message layers are new; the relation encoder's first layer is rebuilt for its
`96 + 6` inputs, with S45's initial bias. Both are built after S45's modules inside an isolated RNG scope, so every
other initial value and every later random draw are S45's. The unmatched-event summary (residual slot) reads the same
node features padded with zeros, as S45 pads token7.

Opt-in only: `install(gnn_cfg)` replaces `semkine.xyz_mesh.XYZMeshNet` for the process before the model is built.
"""
from __future__ import annotations

import torch
from torch import nn

from . import xyz_mesh as X
from .encoder import TOKEN_DIM
from .event_gnn import T_COL, EventGNN

_ORIGINAL = X.XYZMeshNet


def edge_round(lin: nn.Linear, h, idx, dp, emask):
    """One `EdgeConv` round with `lin` (same formula as `EdgeConv.forward`); the neighbour gather is a row
    `index_select` on the flattened nodes. Amendment 3 (docs/EXPERIMENT_RECORDS.md#s45tr-amend3-20261030): under
    deterministic algorithms the backward of `EdgeConv`'s channel-expanded `gather` sorts a (B, N*k, C) index (an
    18 GiB buffer at 512 x 2048 nodes); the backward of `index_select` sorts a (B*N*k,) row index."""
    B, N, C = h.shape
    k = idx.shape[-1]
    rows = (idx + (torch.arange(B, device=idx.device) * N).view(B, 1, 1)).reshape(-1)
    hj = h.reshape(B * N, C).index_select(0, rows).view(B, N, k, C)
    m = torch.relu(lin(torch.cat([hj - h.unsqueeze(2), dp], dim=-1)))
    m = m * emask.unsqueeze(-1)
    return m.sum(2) / emask.sum(2, keepdim=True).clamp_min(1.0)


def node_features(gnn: EventGNN, tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """`(B, N, hidden)` EventGNN node features of already-sampled tokens `(B, N, 7)` (zero where `mask` is False).

    With gradients enabled each message-passing round is recomputed in the backward pass (activation checkpointing,
    amendment 3): the per-edge tensors of three rounds at 512 x 2048 nodes would otherwise stay resident next to S45's.
    The values and gradients are those of the plain rounds (tests/test_s45tr_t1a_r2.py)."""
    from torch.utils.checkpoint import checkpoint
    m = mask.unsqueeze(-1)
    feat = tok * m
    p = torch.stack([feat[..., 0], feat[..., 1], feat[..., T_COL] * gnn.t_scale], dim=-1) * m
    idx, dp, emask = gnn._edges(p.to(torch.float32), mask)
    h = torch.relu(gnn.embed(feat.to(gnn.embed.weight.dtype))) * m
    for layer in gnn.layers:
        args = (layer.lin, h, idx, dp.to(h.dtype), emask.to(h.dtype))
        upd = checkpoint(edge_round, *args, use_reentrant=False) if torch.is_grad_enabled() else edge_round(*args)
        h = (h + upd) * m
    return h


class T1aXYZMeshNet(_ORIGINAL):
    """S45 with token7 replaced by event-graph node features in the relation record."""

    GNN = None  # set by install()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        g = self.GNN
        if g is None:
            raise RuntimeError("s45tr_t1a.install() must run before the model is built")
        if self.contour or self.sign or self.residual_mode != "legacy" or self.xag_gnn != "legacy" or not self.geom:
            raise ValueError("T1a is defined on the S45 recipe (hit_first, legacy residual / GNN, geometry)")
        old = self.rel[0]
        with torch.random.fork_rng(devices=[]):
            self.evgnn = EventGNN(height=180, width=240, hidden=int(g["HIDDEN"]), feat_dim=int(g["HIDDEN"]),
                                  k=int(g["K"]), n_layers=int(g["LAYERS"]), max_nodes=2048, window=int(g["WINDOW"]),
                                  t_scale=float(g["T_SCALE"]), node_attrs="token7", readout=False)
            new = nn.Linear(int(g["HIDDEN"]) + self.rel_in - TOKEN_DIM, old.out_features)
        with torch.no_grad():
            new.bias.copy_(old.bias)
        self.rel[0] = new
        self.rel_in = int(g["HIDDEN"]) + self.rel_in - TOKEN_DIM

    def forward(self, tok, d, mask, verts, assoc, prev48, *args, **kwargs):
        return super().forward(node_features(self.evgnn, tok, mask), d, mask, verts, assoc, prev48, *args, **kwargs)


def install(gnn_cfg) -> None:
    """Route the model constructor to `T1aXYZMeshNet` for this process (idempotent)."""
    T1aXYZMeshNet.GNN = dict(gnn_cfg)
    X.XYZMeshNet = T1aXYZMeshNet


def installed() -> bool:
    return X.XYZMeshNet is T1aXYZMeshNet


__all__ = ["T1aXYZMeshNet", "edge_round", "install", "installed", "node_features"]
