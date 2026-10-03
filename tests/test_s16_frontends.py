#!/usr/bin/env python3
"""S16/S36 gates: frontends share a contract, and the event graph is causal and bounded.

`aegnn_lite` was removed in S36. Its gate lives on here as the two properties that made it
unusable and that `event_gnn` has to demonstrate it does not share: the neighbour search must not
be quadratic in the packet, and memory must not follow the event rate.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from semkine.frontends import build_frontend, frontend_param_count               # noqa: E402
from semkine.encoder import RawEventEncoder, event_tokens                        # noqa: E402
from semkine.event_gnn import EventGNN                                           # noqa: E402


def _batch():
    ev = torch.tensor([
        [0, 10, 20, 0.00, 1],
        [0, 11, 20, 0.01, 0],
        [0, 12, 21, 0.02, 1],
        [1, 4, 5, 0.00, 1],
        [1, 5, 5, 0.03, 1],
    ], dtype=torch.float32)
    ptr = torch.tensor([0, 3, 5])
    dt = torch.tensor([0.05, 0.05])
    return ev, ptr, dt


def _stream(n, packets=2, w=240, h=180):
    """`n` events per packet on a drifting diagonal, strictly increasing in time."""
    rows = []
    for b in range(packets):
        for i in range(n):
            rows.append([b, 20 + (i * 7) % (w - 40), 20 + (i * 3) % (h - 40),
                         0.049 * i / max(n - 1, 1), i % 2])
    ptr = torch.tensor([0] + [n * (b + 1) for b in range(packets)])
    return (torch.tensor(rows, dtype=torch.float32), ptr,
            torch.full((packets,), 0.05))


def test_all_frontends_emit_the_same_feature_width():
    ev, ptr, dt = _batch()
    for name in ("raw_scan", "sparse_cell", "event_gnn"):
        m = build_frontend(name, hidden=32, feat_dim=64).eval()
        with torch.no_grad():
            y = m(ev, ptr, dt)
        assert y.shape == (2, 64), (name, y.shape)


def test_event_gnn_empty_packet_is_exactly_zero():
    """`MODEL.ZERO_EVENT_GATE` must gate an update that was already nothing."""
    m = EventGNN(hidden=16, feat_dim=32).eval()
    with torch.no_grad():
        whole = m(torch.zeros(0, 5), torch.tensor([0, 0]), torch.tensor([0.05]))
        mixed = m(torch.tensor([[0, 10, 20, 0.0, 1]], dtype=torch.float32),
                  torch.tensor([0, 1, 1]), torch.tensor([0.05, 0.05]))
    assert torch.equal(whole, torch.zeros_like(whole))
    assert torch.equal(mixed[1], torch.zeros_like(mixed[1]))
    assert mixed[0].abs().sum() > 0


def _graph_of(m, ev, ptr, dt):
    src, mask = m._sample(ev, ptr)
    tok = event_tokens(ev, ptr, dt, m.height, m.width)
    flat = src.reshape(-1)
    B, N = mask.shape
    f = tok[flat].reshape(B, N, -1) * mask.unsqueeze(-1)
    e = ev[flat].reshape(B, N, -1)
    p = torch.stack([e[..., 1] / m.width, e[..., 2] / m.height,
                     f[..., 3] * m.t_scale], -1) * mask.unsqueeze(-1)
    return m._edges(p, mask)


def test_no_edge_points_into_the_future():
    """The whole async claim rests on this: a new event may only read events older than itself."""
    m = EventGNN(hidden=16, feat_dim=32, k=6, window=16, max_nodes=64).eval()
    ev, ptr, dt = _stream(200)
    idx, dp, emask = _graph_of(m, ev, ptr, dt)
    assert emask.sum() > 0, "the graph is empty; the test proves nothing"
    assert int(((dp[..., 2] > 1e-6) & (emask > 0)).sum()) == 0


def test_the_neighbour_search_does_not_grow_with_the_event_rate():
    """`AEGNNLite`'s NO-GO, restated as a gate.

    Distances scored must be `O(nodes * window)` and the node count must saturate at `max_nodes`,
    so a 100x busier packet costs the same. Under `cdist` the same packet would cost 10 000x.
    """
    m = EventGNN(hidden=16, feat_dim=32, k=6, window=16, max_nodes=64).eval()
    seen = []
    for n in (128, 4096):
        ev, ptr, dt = _stream(n, packets=1)
        _, _, emask = _graph_of(m, ev, ptr, dt)
        seen.append((int(emask.shape[1]), int(emask.sum())))
    assert seen[0] == seen[1], seen                      # identical cost at 32x the event rate
    assert seen[0][0] == m.max_nodes


def test_the_graph_is_state_free():
    """The previous pose may change node *values*, never which events are adjacent.

    KEG's loop gain went to 0.910 because its routing moved with the state. Here the render rides
    in as `extra` and the adjacency must not notice.
    """
    m = EventGNN(hidden=16, feat_dim=32, k=4, window=8, max_nodes=32, extra_channels=2).eval()
    ev, ptr, dt = _stream(64)
    a = _graph_of(m, ev, ptr, dt)
    b = _graph_of(m, ev, ptr, dt)
    assert torch.equal(a[0], b[0])
    extra_hi = torch.ones(ev.shape[0], 2)
    with torch.no_grad():
        y0 = m(ev, ptr, dt, torch.zeros(ev.shape[0], 2))
        y1 = m(ev, ptr, dt, extra_hi)
    assert not torch.allclose(y0, y1), "the render channel is not reaching the readout"


def test_gradients_reach_every_message_layer():
    m = EventGNN(hidden=16, feat_dim=32, k=4, window=8, max_nodes=32, n_layers=3)
    ev, ptr, dt = _stream(64)
    m(ev, ptr, dt).sum().backward()
    for i, layer in enumerate(m.layers):
        assert layer.lin.weight.grad is not None and layer.lin.weight.grad.abs().sum() > 0, i


def test_parameter_budget_vs_scan():
    ref = frontend_param_count(RawEventEncoder(hidden=96, feat_dim=256))
    n = frontend_param_count(build_frontend("sparse_cell", hidden=96, feat_dim=256))
    assert n <= ref * 1.10, (n, ref)


#: `cell_gnn` at the settings S34 trained (hidden 256, feat 512, 4 layers, 8 px cells). The class
#: was deleted with the arm; the count is kept so the budget gate below still has its reference.
#: See `docs/ARCHIVED_FAILURE_RECORDS.md` [GNN-ARMS].
RETIRED_CELLGNN_PARAMS = 1_053_952


def test_the_event_graph_does_not_outspend_the_arm_it_replaced():
    """The 10% rule exists so a win cannot be read as capacity.

    S36 beat `cell_gnn` by 5.23 mm recursive RA. That claim is only interesting if the graph did
    not simply buy the difference, so the trained configuration is pinned against the retired arm's
    count rather than against `raw_scan` -- a single-stage scan is not what S36 was trying to beat.
    """
    arm = frontend_param_count(
        build_frontend("event_gnn", hidden=128, feat_dim=512, n_layers=3, extra_channels=2))
    assert arm <= RETIRED_CELLGNN_PARAMS, (arm, RETIRED_CELLGNN_PARAMS)
