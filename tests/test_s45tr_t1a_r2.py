"""S45-TR amendment 3 (docs/EXPERIMENT_RECORDS.md#s45tr-amend3-20261030): the row-gather round equals EventGNN's own
EdgeConv round, and the checkpointed node features equal the plain ones in values and gradients."""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from semkine import s45tr_t1a as T  # noqa: E402
from semkine.event_gnn import EventGNN  # noqa: E402
from tools.s45tr.common import T1A_GNN  # noqa: E402


def graph(seed=0, B=3, N=300):
    torch.manual_seed(seed)
    g = EventGNN(height=180, width=240, hidden=T1A_GNN["HIDDEN"], feat_dim=T1A_GNN["HIDDEN"], k=T1A_GNN["K"],
                 n_layers=T1A_GNN["LAYERS"], max_nodes=2048, window=T1A_GNN["WINDOW"], t_scale=1.0, readout=False)
    tok = torch.rand(B, N, 7)
    tok[..., 3] = torch.sort(torch.rand(B, N), -1).values
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[1, 200:] = False
    mask[2, :] = False
    return g, tok * mask.unsqueeze(-1), mask


def test_row_gather_round_equals_edgeconv():
    g, tok, mask = graph()
    h = torch.randn(3, 300, T1A_GNN["HIDDEN"])
    p = torch.stack([tok[..., 0], tok[..., 1], tok[..., 3]], -1) * mask.unsqueeze(-1)
    idx, dp, emask = g._edges(p, mask)
    for layer in g.layers:
        assert torch.allclose(T.edge_round(layer.lin, h, idx, dp, emask), layer(h, idx, dp, emask), atol=1e-6, rtol=0)


def plain(g, tok, mask):
    m = mask.unsqueeze(-1)
    p = torch.stack([tok[..., 0], tok[..., 1], tok[..., 3] * g.t_scale], -1) * m
    idx, dp, emask = g._edges(p, mask)
    h = torch.relu(g.embed(tok * m)) * m
    for layer in g.layers:
        h = (h + layer(h, idx, dp, emask)) * m
    return h


def test_checkpointed_features_and_gradients_equal_plain():
    g, tok, mask = graph(1)
    w = torch.randn(3, 300, T1A_GNN["HIDDEN"])
    a = T.node_features(g, tok, mask)
    (a * w).sum().backward()
    ga = {n: p.grad.clone() for n, p in g.named_parameters()}
    g.zero_grad()
    b = plain(g, tok, mask)
    (b * w).sum().backward()
    assert torch.allclose(a, b, atol=1e-6, rtol=0)
    for n, p in g.named_parameters():
        assert torch.allclose(ga[n], p.grad, atol=1e-5, rtol=1e-5), n
    with torch.no_grad():
        assert torch.allclose(T.node_features(g, tok, mask), b, atol=1e-6, rtol=0)
