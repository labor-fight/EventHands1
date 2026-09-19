#!/usr/bin/env python3
"""S37 mesh query: the contract tests.

  * the query vertex set is stratified: every joint owns the same number of vertices;
  * a query vertex gathers only nodes inside the band, and only if it is not occluded;
  * joints whose surface saw no event get exactly zero evidence and zero coverage;
  * the decoder is gated: an unobserved joint's update is exactly zero; the root reads all
    sixteen joints and the mesh aggregate; finger k has no gradient from other joints;
  * the previous state enters through its FK mesh only: no prev_mlp, no prev angles in the heads;
  * an event-free packet returns `prev` bitwise;
  * the config differs from S36 only in the keys that define the arm and the run names.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from model import MNISTModel  # noqa: E402
from semkine.events import EV_BATCH, EV_X, EV_Y, EV_T, EV_P  # noqa: E402
from semkine.mesh_query import MeshQuery, stratified_query_vertices  # noqa: E402

HIDDEN = 32


def _cfg():
    return {
        "MODEL": {
            "POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51, "MANO_NCOMPS": 45,
            "PREDICT_DELTA": True, "PREVPOS_EMBED": False, "PREV_RENDER": False,
            "MESH_QUERY": True, "MESH_QUERY_TOKENS": 64, "MESH_QUERY_BAND_PX": 16.0,
            "MESH_QUERY_SIGMA_PX": 8.0, "ENCODER_NODE_ATTRS": "raw4", "ZERO_EVENT_GATE": True,
            "RENDER_H": 180, "RENDER_W": 240, "RENDER_SCALE": 0.375,
            "ENCODER": "event_gnn", "ENCODER_HIDDEN": HIDDEN, "ENCODER_FEAT": 64,
            "ENCODER_LAYERS": 2, "ENCODER_K": 4, "ENCODER_MAX_NODES": 64, "ENCODER_WINDOW": 8,
            "ACTIVE_HEAD": True, "ACTIVE_HIDDEN": 16,
        },
        "LOSS": {"LAMBDA_POSE": 450.0, "LAMBDA_T": 30000.0, "LAMBDA_R": 60.0, "NORMALIZER": 51, "LOG10": True},
        "TRAIN": {"LR": 1e-3, "WARMUP_STEPS": 0},
        "DATA": {"HEIGHT": 180, "WIDTH": 240},
        "MANO": {"NPZ": "assets/mano_right.npz"},
    }


@dataclasses.dataclass
class _Packet:
    events: torch.Tensor
    ptr: torch.Tensor
    prev_state: torch.Tensor
    target: torch.Tensor
    betas: torch.Tensor
    camera_K: torch.Tensor
    delta_t_s: torch.Tensor
    counts: torch.Tensor
    lnes = None


def _packet(events, n_packets=1):
    counts = torch.bincount(events[:, EV_BATCH].long(), minlength=n_packets)
    ptr = torch.cat([torch.zeros(1, dtype=torch.long), counts.cumsum(0)])
    return _Packet(events=events, ptr=ptr, prev_state=torch.zeros(n_packets, 51),
                   target=torch.zeros(n_packets, 51), betas=torch.zeros(n_packets, 10),
                   camera_K=None, delta_t_s=torch.full((n_packets,), 0.05), counts=counts)


def _events(rows):
    ev = torch.zeros(len(rows), 5)
    for i, (b, x, y, t, p) in enumerate(rows):
        ev[i, EV_BATCH], ev[i, EV_X], ev[i, EV_Y], ev[i, EV_T], ev[i, EV_P] = b, x, y, t, p
    return ev


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = MNISTModel(_cfg())
    m.eval()
    return m


def _test_prev():
    prev = torch.zeros(1, 51)
    prev[0, 2] = 0.5
    return prev


# ------------------------------------------------------------------ query set and kernel
def test_query_vertices_are_stratified_over_joints():
    W = torch.rand(778, 16)
    W = W / W.sum(1, keepdim=True)
    q = stratified_query_vertices(W, 192)
    assert q.shape == (192,) and q.unique().numel() == 192
    dom = W[q].argmax(-1)
    assert all(int((dom == j).sum()) == 12 for j in range(16))


def test_kernel_respects_band_and_occlusion():
    # 32 vertices, joint j dominates vertices j and j + 16 -> 16 distinct query vertices
    W = 0.02 * torch.ones(32, 16)
    W[torch.arange(32), torch.arange(32) % 16] = 1.0
    W = W / W.sum(1, keepdim=True)
    mq = MeshQuery(W, n_tokens=16, band_px=16.0, sigma_px=8.0, node_dim=4, edge_dim=4)
    Q = mq.q_idx.numel()
    assert Q == 16 and mq.q_idx.unique().numel() == 16
    uv = torch.zeros(1, 32, 2)
    uv[0, :, 0] = torch.arange(32) * 40.0         # vertices 40 px apart along x
    uv[0, :, 1] = 50.0
    z = torch.full((1, 32), 0.5)
    q0 = int(mq.q_idx[0])
    # a non-query mesh vertex right under query 0 but closer to the camera -> query 0 is occluded
    occluder = [v for v in range(32) if v not in set(mq.q_idx.tolist())][0]
    uv[0, occluder] = uv[0, q0]
    z[0, occluder] = 0.4
    px = uv[0, mq.q_idx, 0].clone().view(1, -1)      # one node exactly on every query vertex
    py = uv[0, mq.q_idx, 1].clone().view(1, -1)
    mask = torch.ones(1, Q, dtype=torch.bool)
    w, inband = mq.attention(px, py, mask, uv, z)
    assert w.shape == (1, Q, Q)
    assert torch.all(w[0, 0] == 0), "an occluded query vertex must gather nothing"
    for qi in range(1, Q):
        assert float(w[0, qi].sum()) == pytest.approx(1.0, abs=1e-5)
        assert int(w[0, qi].argmax()) == qi                     # its own node dominates
        far = (px[0] - uv[0, mq.q_idx[qi], 0]).abs() > 16.0
        assert torch.all(w[0, qi][far] == 0), "nodes outside the band get no weight"


def test_unseen_joints_have_zero_evidence_and_zero_coverage(model):
    prev = _test_prev()
    betas, K = model._resolve_betas_K(prev, None, None)
    uv, z = model._project_prev(prev, betas, K)
    mq = model.mesh_query_mod
    # nodes far from every projected vertex: nothing is seen
    px = torch.full((1, 5), 2.0)
    py = torch.full((1, 5), 2.0)
    mask = torch.ones(1, 5, dtype=torch.bool)
    h = torch.randn(1, 5, HIDDEN)
    g = torch.randn(1, 5, 4)
    e, mesh, cov, stats = mq(h, g, px, py, mask, uv, z)
    assert torch.all(e == 0) and torch.all(mesh == 0) and torch.all(cov == 0)
    # one node on a visible query vertex: only joints skinned by a query vertex that saw the
    # node (any visible query vertex within the band) get evidence; every other joint stays zero
    vis = mq.visible(uv, z)[0]
    qi = int(vis.nonzero().squeeze(1)[0])
    px = uv[0, mq.q_idx[qi], 0].view(1, 1)
    py = uv[0, mq.q_idx[qi], 1].view(1, 1)
    one = torch.ones(1, 1, dtype=torch.bool)
    e, mesh, cov, _ = mq(h[:, :1], g[:, :1], px, py, one, uv, z)
    _, inband = mq.attention(px, py, one, uv, z)
    seen_q = inband[0].any(-1)                                   # query vertices that saw the node
    assert bool(seen_q[qi])
    skinned = (mq.lbs_q[seen_q] > 0).any(0)
    assert torch.all(cov[0, skinned] > 0)
    assert torch.all(e[0, ~skinned] == 0) and torch.all(cov[0, ~skinned] == 0)
    assert torch.all(mesh != 0) or bool(mesh.abs().sum() > 0)


# ------------------------------------------------------------------ decoder
def test_decoder_is_gated_and_has_no_bypass(model):
    torch.manual_seed(1)
    mq = model.mesh_query_mod
    e = torch.randn(1, 16, mq.ev_dim, requires_grad=True)
    mesh = torch.randn(1, mq.ev_dim, requires_grad=True)
    cov = torch.ones(1, 16)
    cov[0, 5] = 0.0                                      # joint 5 (finger decoder 4) unobserved
    out = model._decode_mesh(e, mesh, cov)
    assert torch.all(out[0, 6 + 3 * 4: 9 + 3 * 4] == 0), "an unobserved joint must not move"
    for k in range(15):
        if k == 4:
            continue
        g_e, g_m = torch.autograd.grad(out[0, 6 + 3 * k: 9 + 3 * k].sum(), (e, mesh), retain_graph=True)
        others = torch.ones(16, dtype=torch.bool)
        others[k + 1] = False
        assert torch.all(g_e[0, others] == 0) and torch.all(g_m == 0), f"finger {k} reads beyond its joint"
        assert g_e[0, k + 1].abs().sum() > 0
    g_e, g_m = torch.autograd.grad(out[0, :6].sum(), (e, mesh))
    assert torch.all(g_e[0].abs().sum(-1) > 0) and g_m.abs().sum() > 0


def test_prev_enters_only_through_the_fk_mesh(model):
    assert not model.prevpos_embed and not hasattr(model, "prev_mlp")
    for head in model.joint_heads:
        assert head[0].in_features == model.mesh_query_mod.ev_dim      # no prev angles appended
    assert model.event_encoder.in_dim == 4 and not model.event_encoder.readout


def test_empty_packet_returns_prev_bitwise(model):
    pk = _packet(_events([]), n_packets=2)
    pk.prev_state = torch.randn(2, 51)
    out = model.forward_packet(pk)
    assert torch.equal(out, pk.prev_state)


def test_forward_packet_end_to_end(model):
    prev = _test_prev()
    betas, K = model._resolve_betas_K(prev, None, None)
    uv, z = model._project_prev(prev, betas, K)
    vis = model.mesh_query_mod.visible(uv, z)[0]
    qs = model.mesh_query_mod.q_idx[vis][:24]
    rows = [(0, float(uv[0, q, 0].round()), float(uv[0, q, 1].round()), 0.001 * n, n % 2)
            for n, q in enumerate(qs.tolist())]
    rows += [(0, 2.0, 2.0, 0.03, 1)]
    pk = _packet(_events(rows))
    pk.prev_state = prev
    out = model.forward_packet(pk)
    assert out.shape == (1, 51) and torch.isfinite(out).all()
    assert float(model.route_stats["mq_joints_seen"]) >= 1
    model.ablate_evidence = True
    try:
        out_abl = model.forward_packet(pk)
    finally:
        model.ablate_evidence = False
    assert torch.equal(out_abl, prev), "with the evidence zeroed nothing is observed, so nothing moves"
    assert not torch.equal(out, out_abl)


# ------------------------------------------------------------------ config
def test_s37_meshq_config_diff_against_s36():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s36_eventgnn_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / "s37_meshq_s3407.yaml"),
         "--allow", "MODEL.PREV_RENDER", "MODEL.PREVPOS_EMBED", "MODEL.MESH_QUERY",
         "MODEL.MESH_QUERY_TOKENS", "MODEL.MESH_QUERY_BAND_PX", "MODEL.MESH_QUERY_SIGMA_PX",
         "MODEL.ENCODER_NODE_ATTRS", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
