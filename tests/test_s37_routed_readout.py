#!/usr/bin/env python3
"""S37 (routed readout). State-free event graph + FK-routed per-joint readout: the contract tests.

What must hold:

  * routing: a node on a projected vertex receives that vertex's LBS row, a node far from the
    hand receives zero, and of two vertices under one pixel the front-most wins;
  * pooling: joints no node reaches get exactly zero evidence;
  * no bypass: finger decoder k has zero gradient from the pooled vector and from every other
    joint's evidence; the root reads all sixteen;
  * the tracking contract survives: an event-free packet returns `prev` bitwise;
  * `ROUTED_READOUT=false` leaves the S36 parameter set untouched (its checkpoint still loads);
  * the arm differs from S36 in the two readout-defining keys and run naming, nowhere else.
"""
from __future__ import annotations

import dataclasses
import json
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
from semkine.routed_readout import pool_joint_evidence, route_front_vertex_lbs  # noqa: E402

HIDDEN = 32


def _cfg():
    return {
        "MODEL": {
            "POSE_REPR": "mano_full_axis_angle",
            "OUTPUT_DIM": 51,
            "MANO_NCOMPS": 45,
            "PREDICT_DELTA": True,
            "PREVPOS_EMBED": True,
            "PREV_RENDER": False,
            "ROUTED_READOUT": True,
            "ROUTE_BAND_PX": 16.0,
            "ROUTE_FRONT_K": 8,
            "ZERO_EVENT_GATE": True,
            "RENDER_H": 180,
            "RENDER_W": 240,
            "RENDER_SCALE": 0.375,
            "ENCODER": "event_gnn",
            "ENCODER_HIDDEN": HIDDEN,
            "ENCODER_FEAT": 64,
            "ENCODER_LAYERS": 2,
            "ENCODER_K": 4,
            "ENCODER_MAX_NODES": 64,
            "ENCODER_WINDOW": 8,
            "ACTIVE_HEAD": True,
            "ACTIVE_HIDDEN": 16,
        },
        "LOSS": {"LAMBDA_POSE": 450.0, "LAMBDA_T": 30000.0, "LAMBDA_R": 60.0,
                 "NORMALIZER": 51, "LOG10": True},
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
    """A previous state whose hand sits half a metre in front of the camera (layout
    `[transl(3), global rot(3), local pose(45)]`; all-zeros would put the wrist at z = 0)."""
    prev = torch.zeros(1, 51)
    prev[0, 2] = 0.5
    return prev


def _projected(model, prev):
    betas, K = model._resolve_betas_K(prev, None, None)
    verts, _ = model._fk(prev, betas)
    fx, fy, cx, cy = model._intrinsics(K)
    x, y, z = verts[0].unbind(-1)
    return fx * x / z + cx, fy * y / z + cy, z


# ------------------------------------------------------------------ routing (pure function)
def test_route_gives_the_lbs_row_of_the_vertex_under_the_node_and_zero_far_away():
    torch.manual_seed(0)
    W = torch.rand(5, 16)
    W = W / W.sum(1, keepdim=True)
    uv = torch.tensor([[[10.0, 10.0], [50.0, 50.0], [50.0, 50.0], [90.0, 20.0], [120.0, 100.0]]])
    z = torch.tensor([[0.5, 0.6, 0.4, 0.5, 0.5]])          # vertex 2 is in front of vertex 1
    px = torch.tensor([[10.0, 50.0, 200.0, 0.0]])
    py = torch.tensor([[10.0, 50.0, 170.0, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    a, d, v = route_front_vertex_lbs(px, py, mask, uv, z, W, band_px=16.0, front_k=3)
    assert a.shape == (1, 4, 16)
    assert torch.allclose(a[0, 0], W[0]) and int(v[0, 0]) == 0 and float(d[0, 0]) == 0.0
    # two vertices under the same pixel: the front-most (smaller z) wins
    assert torch.allclose(a[0, 1], W[2]) and int(v[0, 1]) == 2
    # far from every vertex: no responsibility
    assert torch.all(a[0, 2] == 0) and float(d[0, 2]) > 16.0
    # dead node: no responsibility, whatever its (zeroed) pixel says
    assert torch.all(a[0, 3] == 0)


def test_pooling_leaves_unreached_joints_exactly_zero():
    torch.manual_seed(0)
    B, N, C, J = 1, 6, 8, 16
    h = torch.randn(B, N, C)
    a = torch.zeros(B, N, J)
    a[0, 0, 3] = 0.7
    a[0, 0, 5] = 0.3
    a[0, 1, 3] = 1.0
    mask = torch.ones(B, N, dtype=torch.bool)
    mask[0, 5] = False
    e, count = pool_joint_evidence(h, a, mask)
    assert e.shape == (B, J, 2 * C + 1)
    reached = {3, 5}
    for j in range(J):
        if j not in reached:
            assert torch.all(e[0, j] == 0), j
    want_mean = (0.7 * h[0, 0] + 1.0 * h[0, 1]) / 1.7
    assert torch.allclose(e[0, 3, :C], want_mean, atol=1e-6)
    # hard-assignment max: node 0 argmax is joint 3, node 1 is joint 3 -> joint 5 gets no max
    assert torch.allclose(e[0, 3, C:2 * C], torch.maximum(h[0, 0], h[0, 1]))
    assert int(count[0, 3]) == 2 and int(count[0, 5]) == 0
    assert torch.all(e[0, 5, C:2 * C] == 0)          # no hard-assigned node -> zero max
    assert float(e[0, 3, -1]) == pytest.approx(1.7 / 5)   # coverage over the 5 live nodes


# ------------------------------------------------------------------ heads
def test_finger_heads_read_only_their_own_evidence_and_root_reads_everything(model):
    torch.manual_seed(1)
    feat = torch.randn(1, 64, requires_grad=True)
    evidence = torch.randn(1, 16, 2 * HIDDEN + 1, requires_grad=True)
    prev = _test_prev()
    out = model._decode_active(feat, prev, evidence)
    for k in range(15):
        g_feat, g_ev = torch.autograd.grad(out[0, 6 + 3 * k: 9 + 3 * k].sum(), (feat, evidence),
                                           retain_graph=True)
        assert torch.all(g_feat == 0), f"finger {k} reads the pooled vector"
        others = torch.ones(16, dtype=torch.bool)
        others[k + 1] = False
        assert torch.all(g_ev[0, others] == 0), f"finger {k} reads another joint's evidence"
        assert g_ev[0, k + 1].abs().sum() > 0
    g_feat, g_ev = torch.autograd.grad(out[0, :6].sum(), (feat, evidence))
    assert g_feat.abs().sum() > 0
    assert torch.all(g_ev[0].abs().sum(-1) > 0), "root must read all sixteen evidence rows"


def test_empty_packet_returns_prev_bitwise(model):
    pk = _packet(_events([]), n_packets=2)
    pk.prev_state = torch.randn(2, 51)
    out = model.forward_packet(pk)
    assert torch.equal(out, pk.prev_state)


def test_forward_packet_routes_events_on_the_hand(model):
    prev = _test_prev()
    u, v, z = _projected(model, prev)
    ok = (u >= 0) & (u < 240) & (v >= 0) & (v < 180)
    idx = ok.nonzero().squeeze(1)[:24]
    rows = [(0, float(u[i].round()), float(v[i].round()), 0.001 * n, n % 2) for n, i in enumerate(idx)]
    rows += [(0, 2.0, 2.0, 0.03, 1), (0, 237.0, 177.0, 0.031, 0)]          # two far corners
    pk = _packet(_events(rows))
    pk.prev_state = prev
    out = model.forward_packet(pk)
    assert out.shape == (1, 51) and torch.isfinite(out).all()
    assert 0.5 < float(model.route_stats["route_frac_routed"]) < 1.0    # on-hand routed, corners not
    assert float(model.route_stats["route_joints_hit"]) >= 1
    model.ablate_evidence = True
    try:
        out_abl = model.forward_packet(pk)
    finally:
        model.ablate_evidence = False
    assert not torch.equal(out, out_abl), "the evidence must reach the output"


# ------------------------------------------------------------------ compatibility and config
def test_routed_off_keeps_the_s36_parameter_set():
    from config import load_config
    cfg = load_config(str(ROOT / "configs/semkine/s36_eventgnn_s3407.yaml"))
    m = MNISTModel(cfg)
    assert not m.routed
    sel = ROOT / "outputs/semkine/s36_eventgnn_s3407/selection_val_core_step50.json"
    if not sel.exists():
        pytest.skip("S36 zgz run not present on this machine")
    ckpt = ROOT / json.loads(sel.read_text())["selected"]["ckpt"]
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    assert set(sd) == set(m.state_dict()), "ROUTED_READOUT=false changed the S36 key set"


def test_s37_config_differs_from_s36_only_in_the_readout():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s36_eventgnn_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / "s37_routed_s3407.yaml"),
         "--allow", "MODEL.PREV_RENDER", "MODEL.ROUTED_READOUT", "MODEL.ROUTE_BAND_PX",
         "MODEL.ROUTE_FRONT_K", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
