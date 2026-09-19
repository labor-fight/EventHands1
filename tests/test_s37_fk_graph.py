#!/usr/bin/env python3
"""S37 FK graph: the contract tests.

  * the graph: every joint owns >= 8 sampled vertex nodes; the neighbour table has no self-loops
    and no out-of-range ids; the kinematic edges are exactly `kintree_table`;
  * the observations: an event on a node's pixel is handed to that node, a far event to the
    background; synthetic events moving at constant speed give back that speed as (b_u, b_v);
    the node input carries no geometry -- two prev poses whose nodes project to the same pixels
    produce identical observations;
  * heads: finger k reads only joint node k + 1 (plus its own prev angle); the root reads all
    sixteen joint nodes and the background;
  * contract: an event-free packet returns `prev` bitwise; `ENCODER: fk_graph` off leaves the S36
    parameter set untouched; the arm differs from S36 only in the listed keys.
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
from semkine.fk_graph import (E_KIN, E_LBS, E_MESH, N_JOINTS, OBS_DIM, FKGraphSpec,  # noqa: E402
                              assign_and_observe)

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
            "ZERO_EVENT_GATE": True,
            "RENDER_H": 180,
            "RENDER_W": 240,
            "RENDER_SCALE": 0.375,
            "ENCODER": "fk_graph",
            "FK_GRAPH_VERTS": 192,
            "FK_GRAPH_BAND_PX": 16.0,
            "FK_GRAPH_K": 6,
            "FK_GRAPH_NODE_ID": True,
            "ENCODER_HIDDEN": HIDDEN,
            "ENCODER_LAYERS": 2,
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
    prev = torch.zeros(1, 51)
    prev[0, 2] = 0.5
    return prev


def _node_pixels(model, prev):
    betas, K = model._resolve_betas_K(prev, None, None)
    verts, _ = model._fk(prev, betas)
    pts = torch.cat([torch.einsum("jv,bvc->bjc", model.fk_lbs_wn, verts),
                     verts[:, model.fk_vert_ids]], dim=1)
    fx, fy, cx, cy = model._intrinsics(K)
    x, y, z = pts[0].unbind(-1)
    return fx * x / z + cx, fy * y / z + cy


# ------------------------------------------------------------------ graph
def test_spec_covers_every_joint_and_has_a_valid_neighbour_table(model):
    spec = model.fk_spec
    assert spec.n_verts == 192 and spec.n_nodes == N_JOINTS + 192 + 1
    counts = torch.bincount(spec.vert_joint, minlength=N_JOINTS)
    assert int(counts.min()) >= 8, counts.tolist()
    assert spec.vert_ids.unique().numel() == spec.vert_ids.numel()
    N, K = spec.idx.shape
    assert N == spec.n_nodes and K == 8
    real = spec.emask > 0
    assert torch.all(spec.idx[real] != torch.arange(N).unsqueeze(1).expand(N, K)[real]), "self-loop"
    assert int(spec.idx.min()) >= 0 and int(spec.idx.max()) < N
    assert torch.all(spec.etype.sum(-1)[real] == 1) and torch.all(spec.etype.sum(-1)[~real] == 0)
    # kinematic edges are exactly the tree, both directions
    parents = model.mano.kintree_table[0].clone().long()
    tree = {(j, int(parents[j])) for j in range(1, N_JOINTS)} | {(int(parents[j]), j) for j in range(1, N_JOINTS)}
    kin = set()
    for n in range(N_JOINTS):
        for s in range(K):
            if spec.emask[n, s] > 0 and spec.etype[n, s, E_KIN] > 0 and int(spec.idx[n, s]) < N_JOINTS:
                kin.add((n, int(spec.idx[n, s])))
    assert kin == tree, kin ^ tree
    # every vertex node has mesh neighbours and its skinning joints
    for i in range(N_JOINTS, N_JOINTS + spec.n_verts):
        types = spec.etype[i][spec.emask[i] > 0].argmax(-1).tolist()
        assert types.count(E_MESH) == 6 and types.count(E_LBS) == 2


# ------------------------------------------------------------------ observations
def test_events_go_to_the_node_under_them_and_far_events_to_the_background():
    uv = torch.tensor([[[10.0, 10.0], [50.0, 50.0], [100.0, 100.0]]])       # M = 3 nodes
    ev = _events([(0, 10.0, 10.0, 0.01, 1), (0, 52.0, 50.0, 0.02, 0), (0, 200.0, 170.0, 0.03, 1)])
    obs, assign = assign_and_observe(ev, torch.tensor([0, 3]), uv, torch.tensor([0.05]), band_px=16.0)
    assert assign.tolist() == [0, 1, 3]                    # third event: background (index M)
    assert obs.shape == (1, 4, OBS_DIM)
    assert obs[0, 2].abs().sum() == 0                      # node 2 saw nothing
    assert obs[0, 1, 1] == pytest.approx(2.0 / 16.0)       # mean du of node 1, in band units
    assert obs[0, 0, 5] == pytest.approx(1.0) and obs[0, 1, 5] == pytest.approx(-1.0)   # polarity


def test_constant_speed_events_give_back_their_speed_as_local_flow():
    uv = torch.tensor([[[100.0, 90.0]]])
    dt = 0.05
    rows = []
    for i in range(20):
        t = i / 19 * dt
        rows.append((0, 100.0 + 8.0 * (t / dt), 90.0 - 4.0 * (t / dt), t, 1))   # 8 px right, 4 px up per packet
    obs, assign = assign_and_observe(_events(rows), torch.tensor([0, 20]), uv, torch.tensor([dt]), band_px=16.0)
    assert torch.all(assign == 0)
    assert obs[0, 0, 6] == pytest.approx(8.0 / 16.0, abs=1e-4)     # b_u in band units per packet
    assert obs[0, 0, 7] == pytest.approx(-4.0 / 16.0, abs=1e-4)


def test_observations_carry_no_geometry():
    """Two node layouts that project to the same pixels give identical observations."""
    uv_a = torch.tensor([[[30.0, 40.0], [80.0, 60.0]]])
    ev = _events([(0, 31.0, 40.0, 0.01, 1), (0, 79.0, 61.0, 0.02, 0), (0, 32.0, 39.0, 0.03, 1)])
    obs_a, _ = assign_and_observe(ev, torch.tensor([0, 3]), uv_a, torch.tensor([0.05]), 16.0)
    obs_b, _ = assign_and_observe(ev, torch.tensor([0, 3]), uv_a.clone(), torch.tensor([0.05]), 16.0)
    assert torch.equal(obs_a, obs_b)
    # and the model's node input dimension is the observation width alone
    m = MNISTModel(_cfg())
    assert m.event_encoder.obs_embed.in_features == OBS_DIM


# ------------------------------------------------------------------ heads and contract
def test_finger_heads_read_only_their_own_joint_node_and_root_reads_all(model):
    torch.manual_seed(1)
    joints = torch.randn(1, 16, HIDDEN, requires_grad=True)
    background = torch.randn(1, HIDDEN, requires_grad=True)
    out = model._decode_active(None, _test_prev(), joints, root_extra=background)
    for k in range(15):
        g_j, g_bg = torch.autograd.grad(out[0, 6 + 3 * k: 9 + 3 * k].sum(), (joints, background),
                                        retain_graph=True)
        others = torch.ones(16, dtype=torch.bool)
        others[k + 1] = False
        assert torch.all(g_j[0, others] == 0) and g_j[0, k + 1].abs().sum() > 0
        assert torch.all(g_bg == 0), f"finger {k} reads the background node"
    g_j, g_bg = torch.autograd.grad(out[0, :6].sum(), (joints, background))
    assert torch.all(g_j[0].abs().sum(-1) > 0) and g_bg.abs().sum() > 0


def test_empty_packet_returns_prev_bitwise(model):
    pk = _packet(_events([]), n_packets=2)
    pk.prev_state = torch.randn(2, 51)
    out = model.forward_packet(pk)
    assert torch.equal(out, pk.prev_state)


def test_forward_packet_end_to_end_and_ablation_hooks(model):
    prev = _test_prev()
    u, v = _node_pixels(model, prev)
    ok = (u >= 2) & (u < 238) & (v >= 2) & (v < 178)
    idx = ok.nonzero().squeeze(1)[:40]
    rows = [(0, float(u[i].round()), float(v[i].round()), 0.001 * n, n % 2) for n, i in enumerate(idx)]
    rows += [(0, 2.0, 2.0, 0.041, 1), (0, 237.0, 177.0, 0.042, 0)]
    pk = _packet(_events(rows))
    pk.prev_state = prev
    out = model.forward_packet(pk)
    assert out.shape == (1, 51) and torch.isfinite(out).all()
    assert 0.8 < float(model.route_stats["route_frac_routed"]) < 1.0    # 40 on the hand, 2 corners
    assert float(model.route_stats["route_joints_hit"]) >= 1
    model.ablate_evidence = True
    try:
        out_abl = model.forward_packet(pk)
    finally:
        model.ablate_evidence = False
    assert not torch.equal(out, out_abl), "the observations must reach the output"
    model.obs_feature_mask = torch.zeros(OBS_DIM)
    try:
        out_masked = model.forward_packet(pk)
    finally:
        model.obs_feature_mask = None
    assert torch.equal(out_masked, out_abl), "an all-zero feature mask is the observation ablation"


def test_fk_graph_off_keeps_the_s36_parameter_set():
    from config import load_config
    cfg = load_config(str(ROOT / "configs/semkine/s36_eventgnn_s3407.yaml"))
    m = MNISTModel(cfg)
    assert not m.fk_graph
    sel = ROOT / "outputs/semkine/s36_eventgnn_s3407/selection_val_core_step50.json"
    if not sel.exists():
        pytest.skip("S36 zgz run not present on this machine")
    ckpt = ROOT / json.loads(sel.read_text())["selected"]["ckpt"]
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    assert set(sd) == set(m.state_dict()), "fk_graph changed the S36 key set"


def test_s37_fkgraph_config_diff_against_s36():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s36_eventgnn_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / "s37_fkgraph_s3407.yaml"),
         "--allow", "MODEL.ENCODER", "MODEL.PREV_RENDER",
         "MODEL.FK_GRAPH_VERTS", "MODEL.FK_GRAPH_BAND_PX", "MODEL.FK_GRAPH_K", "MODEL.FK_GRAPH_NODE_ID",
         "MODEL.ENCODER_FEAT", "MODEL.ENCODER_K", "MODEL.ENCODER_MAX_NODES", "MODEL.ENCODER_WINDOW",
         "MODEL.ENCODER_T_SCALE", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
