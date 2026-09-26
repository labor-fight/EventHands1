#!/usr/bin/env python3
"""S37 mesh graph: the contract tests.

  * the graph: every one of the 778 vertices is a node, the edge set is exactly the face 1-ring
    (both directions), no self-loops, K = max degree = 8, the background node has no edges;
  * visibility: of two vertices under the same pixel only the nearer is visible; a vertex outside
    the frame is not; on the real rest-pose hand a substantial fraction of the mesh is hidden;
  * assignment: an event on a hidden vertex's pixel is handed to the nearest visible node (or the
    background), never to the hidden one; the observations carry no geometry;
  * pooling: a joint none of whose vertices saw an event has exactly zero evidence; the weighted
    mean is the skinning-weighted mean over observed vertices; coverage is in [0, 1];
  * heads: finger k reads only evidence row k + 1 (plus its own prev angle); the root reads all
    sixteen rows and the background node;
  * contract: an event-free packet returns `prev` bitwise; observations reach the output and the
    ablation hook silences them; the arm differs from S37 fkgraph only in the listed keys.
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
from semkine.fk_graph import E_MESH, OBS_DIM, assign_and_observe  # noqa: E402
from semkine.mesh_graph import (MeshGraphSpec, assign_events_by_lut, facing_camera,  # noqa: E402
                                lbs_pool_evidence, nearest_node_lut, visible_vertices,
                                zbuffer_visible)

HIDDEN = 32


def _cfg(**model_over):
    m = {
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
        "ENCODER": "mesh_graph",
        "MESH_GRAPH_BAND_PX": 16.0,
        "MESH_GRAPH_FRONT_PX": 1.0,
        "MESH_GRAPH_Z_TOL": 0.01,
        "MESH_GRAPH_OBS_FLOW": False,
        "MESH_GRAPH_NODE_ID": True,
        "ENCODER_HIDDEN": HIDDEN,
        "ENCODER_LAYERS": 2,
        "ACTIVE_HEAD": True,
        "ACTIVE_HIDDEN": 16,
    }
    m.update(model_over)
    return {
        "MODEL": m,
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


def _projected(model, prev):
    betas, K = model._resolve_betas_K(prev, None, None)
    verts, _ = model._fk(prev, betas)
    uv, z = model._project_verts(verts, K)
    vis = visible_vertices(verts, uv, z, model.mano.f, model.render_h, model.render_w,
                           model.mg_front_px, model.mg_z_tol)
    return uv[0], z[0], vis[0]


# ------------------------------------------------------------------ graph
def test_spec_is_the_face_one_ring_over_every_vertex(model):
    spec = model.mg_spec
    V = model.mano.weights.shape[0]
    assert spec.n_verts == V == 778 and spec.n_nodes == V + 1 and spec.background == V
    N, K = spec.idx.shape
    assert N == spec.n_nodes and K == 8 == spec.k
    real = spec.emask > 0
    assert torch.all(spec.idx[real] != torch.arange(N).unsqueeze(1).expand(N, K)[real]), "self-loop"
    assert int(spec.idx.min()) >= 0 and int(spec.idx.max()) < N
    assert torch.all(spec.etype[real].argmax(-1) == E_MESH)
    assert torch.all(spec.etype.sum(-1)[real] == 1) and torch.all(spec.etype.sum(-1)[~real] == 0)
    assert spec.emask[spec.background].sum() == 0, "the background node has no edges"
    # exactly the undirected face edges, both directions
    faces = model.mano.f.long()
    want = set()
    for a, b, c in faces.tolist():
        for s, t in ((a, b), (b, c), (c, a)):
            want.add((s, t))
            want.add((t, s))
    have = {(n, int(spec.idx[n, s])) for n in range(V) for s in range(K) if spec.emask[n, s] > 0}
    assert have == want, len(have ^ want)
    assert torch.equal(spec.vert_joint, model.mano.weights.argmax(1))


# ------------------------------------------------------------------ visibility
def test_zbuffer_keeps_only_the_nearer_of_two_vertices_under_one_pixel():
    uv = torch.tensor([[[50.0, 40.0], [51.0, 40.0], [120.0, 90.0], [-5.0, 10.0]]])
    z = torch.tensor([[0.40, 0.43, 0.50, 0.30]])
    vis = zbuffer_visible(uv, z, 180, 240, front_px=1.0, z_tol=0.01)
    assert vis.tolist() == [[True, False, True, False]]     # 3 cm behind -> hidden; off-frame -> hidden
    vis = zbuffer_visible(uv, z, 180, 240, front_px=1.0, z_tol=0.05)
    assert vis.tolist() == [[True, True, True, False]]      # within tolerance -> both visible
    vis = zbuffer_visible(uv, z, 180, 240, front_px=0.0, z_tol=0.01)
    assert vis.tolist() == [[True, True, True, False]]      # no window: different pixels, both win


def test_backface_culling_follows_the_face_normal():
    # one triangle in front of the camera (origin, looking down +z); its normal (v1-v0)x(v2-v0)
    faces = torch.tensor([[0, 1, 2]])
    tri = torch.tensor([[[0.0, 0.0, 0.5], [0.1, 0.0, 0.5], [0.0, 0.1, 0.5]]])   # normal +z: away
    assert facing_camera(tri, faces).tolist() == [[False, False, False]]
    tri_flipped = tri[:, [0, 2, 1]]                                                # normal -z: toward
    assert facing_camera(tri_flipped, faces).tolist() == [[True, True, True]]


def test_a_real_hand_hides_about_half_its_mesh(model):
    # The mean pose at 0.5 m is a steeply foreshortened, self-occluding view: 0.33 visible here
    # (0.52 averaged over 300 training states), and the index MCP sits behind the middle finger,
    # which a triangle z-buffer confirms (0 of its 33 vertices visible). So: a fraction in range,
    # most parts with a few visible vertices, and the culled half agreeing with the z-buffer's
    # choice (MANO's winding is outward).
    prev = _test_prev()
    _, _, vis = _projected(model, prev)
    frac = float(vis.float().mean())
    assert 0.25 < frac < 0.7, frac
    per_joint = torch.bincount(model.mg_spec.vert_joint[vis], minlength=16)
    assert int((per_joint >= 3).sum()) >= 12, per_joint.tolist()
    betas, K = model._resolve_betas_K(prev, None, None)
    verts, _ = model._fk(prev, betas)
    uv, z = model._project_verts(verts, K)
    facing = facing_camera(verts, model.mano.f)[0]
    zb = zbuffer_visible(uv, z, 180, 240, 1.0, 0.01)[0]
    assert 0.4 < float(facing.float().mean()) < 0.65
    assert float((facing & zb).sum() / facing.sum()) > 0.6


# ------------------------------------------------------------------ assignment
def test_events_never_reach_a_hidden_node():
    uv = torch.tensor([[[10.0, 10.0], [11.0, 10.0], [100.0, 100.0]]])   # nodes 0/1 share a spot
    mask = torch.tensor([[False, True, True]])
    ev = _events([(0, 10.0, 10.0, 0.01, 1), (0, 100.0, 101.0, 0.02, 0), (0, 200.0, 170.0, 0.03, 1)])
    obs, assign = assign_and_observe(ev, torch.tensor([0, 3]), uv, torch.tensor([0.05]), 16.0,
                                     node_mask=mask)
    assert assign.tolist() == [1, 2, 3]                    # hidden node 0 skipped; far event -> background
    assert obs[0, 0].abs().sum() == 0
    assert obs[0, 1, 0] > 0 and obs[0, 1, 1] == pytest.approx(-1.0 / 16.0)   # du relative to node 1
    # every node hidden: everything goes to the background
    obs, assign = assign_and_observe(ev, torch.tensor([0, 3]), uv, torch.tensor([0.05]), 16.0,
                                     node_mask=torch.zeros(1, 3, dtype=torch.bool))
    assert assign.tolist() == [3, 3, 3]
    # without a mask the behaviour is the fkgraph one
    _, assign = assign_and_observe(ev, torch.tensor([0, 3]), uv, torch.tensor([0.05]), 16.0)
    assert assign.tolist() == [0, 2, 3]


def test_lut_matches_brute_force_on_visible_pixels_and_sends_out_of_frame_to_background():
    uv = torch.tensor([[[10.0, 10.0], [11.0, 10.0], [100.0, 100.0]]])
    mask = torch.tensor([[False, True, True]])
    lut = nearest_node_lut(uv, mask, 180, 240, 16.0)
    assert lut.shape == (1, 180, 240)
    assert int(lut[0, 10, 10]) == 1                    # hidden node 0 skipped; 1 px to node 1
    assert int(lut[0, 100, 100]) == 2
    assert int(lut[0, 170, 200]) == 3                  # (200, 170) farther than 16 px -> background
    ev = _events([(0, 10.0, 10.0, 0.01, 1), (0, 100.0, 101.0, 0.02, 0),
                  (0, 200.0, 170.0, 0.03, 1)])
    assign = assign_events_by_lut(ev, lut, background=3)
    _, brute = assign_and_observe(ev, torch.tensor([0, 3]), uv, torch.tensor([0.05]), 16.0,
                                  node_mask=mask)
    assert assign.tolist() == brute.tolist() == [1, 2, 3]


def test_lut_rejects_invalid_coordinates_before_rounding_and_clamping():
    # Every edge pixel has a real nearby vertex: clamping must not turn an invalid event into
    # hand evidence. In particular, -0.1 rounds to an otherwise valid pixel at zero.
    uv = torch.tensor([[[0.0, 2.0], [4.0, 2.0], [2.0, 0.0], [2.0, 4.0]]])
    lut = nearest_node_lut(uv, torch.ones(1, 4, dtype=torch.bool), 5, 5, 2.0)
    xy = [(-1.0, 2.0), (-0.1, 2.0), (5.0, 2.0), (5.1, 2.0),
          (2.0, -1.0), (2.0, -0.1), (2.0, 5.0), (2.0, 5.1),
          (float("nan"), 2.0), (float("inf"), 2.0), (float("-inf"), 2.0),
          (2.0, float("nan")), (2.0, float("inf")), (2.0, float("-inf"))]
    ev = _events([(0, x, y, 0.01, 1) for x, y in xy])
    assigned = assign_events_by_lut(ev, lut, background=4)
    assert assigned.tolist() == [4] * len(xy)


def test_lut_keeps_every_valid_pixel_and_subpixel_lookup_unchanged():
    # Distinct LUT values make any accidental pixel or packet change observable.
    lut = torch.arange(2 * 4 * 5).reshape(2, 4, 5)
    xy = [(float(x), float(y)) for y in range(4) for x in range(5)]
    xy += [(0.49, 0.51), (1.5, 2.5), (4.9, 2.0), (2.0, 3.9)]
    ev = _events([(b, x, y, 0.01, 1) for b in range(2) for x, y in xy])
    original = lut[ev[:, EV_BATCH].long(), ev[:, EV_Y].round().long().clamp(0, 3),
                   ev[:, EV_X].round().long().clamp(0, 4)]
    assert torch.equal(assign_events_by_lut(ev, lut, background=40), original)
    assert assign_events_by_lut(ev[:0], lut, background=40).shape == (0,)


def test_model_observation_is_six_channels_without_flow_and_eight_with(model):
    assert model.mg_obs_dim == OBS_DIM - 2
    assert model.event_encoder.obs_embed.in_features == OBS_DIM - 2
    m8 = MNISTModel(_cfg(MESH_GRAPH_OBS_FLOW=True))
    assert m8.mg_obs_dim == OBS_DIM and m8.event_encoder.obs_embed.in_features == OBS_DIM


# ------------------------------------------------------------------ pooling
def test_lbs_pool_is_the_skinning_weighted_mean_over_observed_vertices():
    torch.manual_seed(2)
    V, J, C = 6, 3, 4
    W = torch.tensor([[1.0, 0.0, 0.0], [0.7, 0.3, 0.0], [0.0, 1.0, 0.0],
                      [0.0, 0.6, 0.4], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    h = torch.randn(1, V, C)
    vis = torch.tensor([[True, True, True, True, True, False]])
    has = torch.tensor([[True, True, False, False, False, False]])
    e, count = lbs_pool_evidence(h, W, vis, has)
    assert e.shape == (1, J, 2 * C + 1) and count.tolist() == [[2, 0, 0]]
    mean0 = (1.0 * h[0, 0] + 0.7 * h[0, 1]) / 1.7
    assert torch.allclose(e[0, 0, :C], mean0, atol=1e-6)
    assert torch.allclose(e[0, 0, C:2 * C], torch.maximum(h[0, 0], h[0, 1]), atol=1e-6)
    assert e[0, 0, -1] == pytest.approx(1.7 / 1.7)               # all of joint 0's visible surface seen
    assert torch.allclose(e[0, 1, :C], h[0, 1], atol=1e-6)        # joint 1: only vertex 1 (w=0.3) seen
    assert e[0, 1, C:2 * C].abs().sum() == 0                      # no vertex is *dominated* by joint 1 and seen
    assert e[0, 1, -1] == pytest.approx(0.3 / 1.9)                # 0.3 of its visible 1.9
    assert e[0, 2].abs().sum() == 0, "joint 2 saw nothing -> exactly zero"
    assert torch.all((e[..., -1] >= 0) & (e[..., -1] <= 1))


# ------------------------------------------------------------------ heads and contract
def test_finger_heads_read_only_their_own_evidence_and_root_reads_all(model):
    torch.manual_seed(1)
    ev_dim = 2 * HIDDEN + 1
    e = torch.randn(1, 16, ev_dim, requires_grad=True)
    background = torch.randn(1, HIDDEN, requires_grad=True)
    out = model._decode_active(None, _test_prev(), e, root_extra=background)
    for k in range(15):
        g_e, g_bg = torch.autograd.grad(out[0, 6 + 3 * k: 9 + 3 * k].sum(), (e, background),
                                        retain_graph=True)
        others = torch.ones(16, dtype=torch.bool)
        others[k + 1] = False
        assert torch.all(g_e[0, others] == 0) and g_e[0, k + 1].abs().sum() > 0
        assert torch.all(g_bg == 0), f"finger {k} reads the background node"
    g_e, g_bg = torch.autograd.grad(out[0, :6].sum(), (e, background))
    assert torch.all(g_e[0].abs().sum(-1) > 0) and g_bg.abs().sum() > 0


def test_empty_packet_returns_prev_bitwise(model):
    pk = _packet(_events([]), n_packets=2)
    pk.prev_state = torch.randn(2, 51)
    out = model.forward_packet(pk)
    assert torch.equal(out, pk.prev_state)


def test_forward_packet_end_to_end_and_ablation_hooks(model):
    prev = _test_prev()
    uv, _, vis = _projected(model, prev)
    u, v = uv.unbind(-1)
    ok = vis & (u >= 2) & (u < 238) & (v >= 2) & (v < 178)
    idx = ok.nonzero().squeeze(1)[::4][:60]
    rows = [(0, float(u[i].round()), float(v[i].round()), 0.0005 * n, n % 2) for n, i in enumerate(idx)]
    rows += [(0, 2.0, 2.0, 0.041, 1), (0, 237.0, 177.0, 0.042, 0)]
    pk = _packet(_events(rows))
    pk.prev_state = prev
    out = model.forward_packet(pk)
    assert out.shape == (1, 51) and torch.isfinite(out).all()
    assert 0.9 < float(model.route_stats["route_frac_routed"]) < 1.0     # 60 on the hand, 2 corners
    assert float(model.route_stats["route_joints_hit"]) >= 4
    assert 0.3 < float(model.route_stats["mg_frac_visible"]) < 0.8
    model.ablate_evidence = True
    try:
        out_abl = model.forward_packet(pk)
    finally:
        model.ablate_evidence = False
    assert not torch.equal(out, out_abl), "the observations must reach the output"
    # a packet whose events all miss the hand: every joint's evidence is exactly zero (no vertex
    # saw anything), only the background node and the prev paths remain; still a finite update
    far = _packet(_events([(0, 2.0, 2.0, 0.041, 1), (0, 237.0, 177.0, 0.042, 0)]))
    far.prev_state = prev
    out_far = model.forward_packet(far)
    assert float(model.route_stats["route_frac_routed"]) == 0.0
    assert float(model.route_stats["route_joints_hit"]) == 0.0
    assert torch.isfinite(out_far).all() and not torch.equal(out_far, prev)


def test_s37_meshgraph_config_diff_against_fkgraph():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s37_fkgraph_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / "s37_meshgraph_s3407.yaml"),
         "--allow", "MODEL.ENCODER",
         "MODEL.FK_GRAPH_VERTS", "MODEL.FK_GRAPH_BAND_PX", "MODEL.FK_GRAPH_K", "MODEL.FK_GRAPH_NODE_ID",
         "MODEL.MESH_GRAPH_BAND_PX", "MODEL.MESH_GRAPH_FRONT_PX", "MODEL.MESH_GRAPH_Z_TOL",
         "MODEL.MESH_GRAPH_OBS_FLOW", "MODEL.MESH_GRAPH_NODE_ID",
         "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
