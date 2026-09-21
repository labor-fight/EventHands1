#!/usr/bin/env python3
"""S38 mesh3d / rootlever: the contract tests (docs/S38_MESH3D_PREREG.md).

  * backward compatibility: with every S38 knob off the mesh graph is the S37 arm -- same
    parameters, same output bitwise -- so the S37 checkpoints still load and reproduce;
  * edge vectors: exactly X_j - X_i on the real edges (camera frame, centimetres), zero on padded
    slots and on the background row; they rotate with the mesh and ignore its translation;
  * anisotropy: the S37 encoder is invariant to permuting a node's neighbour slots and blind to
    where the neighbours are; with edge vectors the output depends on the neighbour geometry;
  * rigid node: the masked mean of relu(W[h ; r]) through the output MLP; exactly zero when no
    vertex qualifies; lever arms are about the root joint, so translating the hand leaves them;
  * part positions: skinning-weighted centroids about the root joint, 16 x 3; the part lever term
    is relu(W[e_j ; r_j]) per part, exactly zero for an unseen part;
  * heads: fingers read only their own evidence row and never the root's extras (background,
    rigid node, part lever terms); the root reads all of them;
  * contract: an event-free packet returns `prev` bitwise; `ablate_evidence` zeroes the rigid node
    and still changes the output; a packet that misses the hand leaves the rigid node at zero;
    each config differs from S37 meshgraph only in its listed keys.
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
from semkine.mesh_graph import (GEO_SCALE, PartLever, RigidNode, edge_vectors,  # noqa: E402
                                lever_arms, part_positions, visible_vertices)

HIDDEN = 32
LEVER = 12


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


MESH3D = dict(MESH_GRAPH_EDGE_GEO=True, MESH_GRAPH_RIGID_NODE=True, MESH_GRAPH_GEO_SCALE=100.0)
ROOTLEVER = dict(MESH_GRAPH_ROOT_LEVER=LEVER, MESH_GRAPH_GEO_SCALE=100.0)


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


def _test_prev():
    prev = torch.zeros(1, 51)
    prev[0, 2] = 0.5
    return prev


def _build(seed=0, **over):
    torch.manual_seed(seed)
    m = MNISTModel(_cfg(**over))
    m.eval()
    return m


@pytest.fixture(scope="module")
def mesh3d():
    return _build(0, **MESH3D)


@pytest.fixture(scope="module")
def rootlever():
    return _build(0, **ROOTLEVER)


@pytest.fixture(scope="module")
def s37():
    return _build(0)


def _hand_packet(model, prev, n=60):
    """Events on visible vertices of `prev`'s hand plus two far corners."""
    betas, K = model._resolve_betas_K(prev, None, None)
    verts, _ = model._fk(prev, betas)
    uv, z = model._project_verts(verts, K)
    vis = visible_vertices(verts, uv, z, model.mano.f, model.render_h, model.render_w,
                           model.mg_front_px, model.mg_z_tol)[0]
    u, v = uv[0].unbind(-1)
    ok = vis & (u >= 2) & (u < 238) & (v >= 2) & (v < 178)
    idx = ok.nonzero().squeeze(1)[::4][:n]
    rows = [(0, float(u[i].round()), float(v[i].round()), 0.0005 * k, k % 2) for k, i in enumerate(idx)]
    rows += [(0, 2.0, 2.0, 0.041, 1), (0, 237.0, 177.0, 0.042, 0)]
    pk = _packet(_events(rows))
    pk.prev_state = prev
    return pk


# ------------------------------------------------------------------ backward compatibility
def test_all_knobs_off_is_the_s37_mesh_graph_bitwise(s37):
    assert s37.rigid_node is None and s37.part_lever is None and not s37.mg_edge_geo
    assert isinstance(s37.root_head, torch.nn.Linear)
    keys = set(s37.state_dict())
    assert not any(k.startswith(("rigid_node", "part_lever")) for k in keys)
    assert s37.root_head.in_features == 16 * (2 * HIDDEN + 1) + HIDDEN
    pk = _hand_packet(s37, _test_prev())
    out = s37.forward_packet(pk)
    # the encoder without an edge feature is the old call
    obs = torch.randn(2, s37.mg_spec.n_nodes, s37.mg_obs_dim)
    assert torch.equal(s37.event_encoder(obs), s37.event_encoder(obs, None))
    assert torch.equal(out, s37.forward_packet(pk))


def test_s38_modules_only_appear_when_asked(mesh3d, rootlever, s37):
    ev_dim = 2 * HIDDEN + 1
    assert isinstance(mesh3d.rigid_node, RigidNode) and mesh3d.mg_edge_geo and mesh3d.part_lever is None
    assert isinstance(mesh3d.root_head, torch.nn.Linear)
    assert mesh3d.root_head.in_features == 16 * ev_dim + HIDDEN + HIDDEN
    assert rootlever.rigid_node is None and not rootlever.mg_edge_geo
    assert isinstance(rootlever.part_lever, PartLever) and rootlever.mg_root_lever == LEVER
    assert isinstance(rootlever.root_head, torch.nn.Linear)
    assert rootlever.root_head.in_features == 16 * ev_dim + HIDDEN + 16 * LEVER
    # the S38a change adds no parameter to the encoder: the edge channels already existed
    assert sum(p.numel() for p in mesh3d.event_encoder.parameters()) == \
        sum(p.numel() for p in s37.event_encoder.parameters())
    # neither arm is a capacity change in disguise: both stay within a few % of S37
    n37 = sum(p.numel() for p in s37.parameters())
    for m in (mesh3d, rootlever):
        n = sum(p.numel() for p in m.parameters())
        assert n37 <= n < 1.25 * n37, (n, n37)


# ------------------------------------------------------------------ edge vectors
def test_edge_vectors_are_the_neighbour_offsets_in_centimetres(mesh3d):
    spec = mesh3d.mg_spec
    prev = _test_prev()
    betas, _ = mesh3d._resolve_betas_K(prev, None, None)
    verts, _ = mesh3d._fk(prev, betas)
    dp = edge_vectors(verts, spec.idx, spec.emask, GEO_SCALE)
    assert dp.shape == (1, spec.n_nodes, spec.k, 3)
    real = spec.emask > 0
    for n, s in real.nonzero()[::97].tolist():
        want = (verts[0, int(spec.idx[n, s])] - verts[0, n]) * 100.0
        assert torch.allclose(dp[0, n, s], want, atol=1e-5)
    assert dp[0][~real].abs().sum() == 0, "padded slots carry no geometry"
    assert dp[0, spec.background].abs().sum() == 0, "the background node has no edges"
    # MANO edges are millimetres to a couple of centimetres
    lengths = dp[0][real].norm(dim=-1)
    assert 0.2 < float(lengths.median()) < 1.5 and float(lengths.max()) < 4.0, lengths.median()


def test_edge_vectors_rotate_with_the_mesh_and_ignore_translation(mesh3d):
    spec = mesh3d.mg_spec
    torch.manual_seed(3)
    verts = torch.randn(1, spec.n_verts, 3)
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    if torch.det(R) < 0:
        R[:, 0] = -R[:, 0]
    t = torch.tensor([0.3, -0.2, 0.7])
    dp = edge_vectors(verts, spec.idx, spec.emask)
    dp_rt = edge_vectors(verts @ R.T + t, spec.idx, spec.emask)
    assert torch.allclose(dp_rt, dp @ R.T, atol=1e-4)
    assert torch.allclose(edge_vectors(verts + t, spec.idx, spec.emask), dp, atol=1e-5)


def test_edge_geometry_makes_the_operator_anisotropic(s37, mesh3d):
    """S37: permuting a node's neighbour slots leaves its output unchanged and the output does not
    depend on where the neighbours are. S38a: same features, different neighbour geometry ->
    different output."""
    spec = s37.mg_spec
    torch.manual_seed(4)
    obs = torch.randn(1, spec.n_nodes, s37.mg_obs_dim)
    h0 = s37.event_encoder(obs)
    # permute the slots of every node (same neighbour set) -- both encoders must not care
    perm = torch.randperm(spec.k)
    enc = s37.event_encoder
    idx0, et0, em0 = enc.idx.clone(), enc.etype.clone(), enc.emask.clone()
    enc.idx, enc.etype, enc.emask = idx0[:, perm], et0[:, perm], em0[:, perm]
    try:
        h_perm = enc(obs)
    finally:
        enc.idx, enc.etype, enc.emask = idx0, et0, em0
    assert torch.allclose(h_perm, h0, atol=1e-5)
    # S38a: two meshes with the same topology and the same observations, different geometry
    va = torch.randn(1, spec.n_verts, 3)
    vb = torch.randn(1, spec.n_verts, 3)
    enc3 = mesh3d.event_encoder
    ha = enc3(obs, edge_vectors(va, spec.idx, spec.emask))
    hb = enc3(obs, edge_vectors(vb, spec.idx, spec.emask))
    assert not torch.allclose(ha, hb, atol=1e-4), "edge geometry must reach the features"
    # ... and the same geometry twice is the same answer (no randomness, gather-only)
    assert torch.equal(ha, enc3(obs, edge_vectors(va, spec.idx, spec.emask)))


# ------------------------------------------------------------------ rigid node and lever arms
def test_rigid_node_is_the_masked_mean_and_exactly_zero_when_empty():
    torch.manual_seed(5)
    node = RigidNode(HIDDEN).eval()
    B, V = 2, 7
    h = torch.randn(B, V, HIDDEN)
    r = torch.randn(B, V, 3)
    mask = torch.tensor([[True, False, True, True, False, False, False],
                         [False] * V])
    g = node(h, r, mask)
    assert g.shape == (B, HIDDEN)
    m = torch.relu(node.msg(torch.cat([h[0], r[0]], -1)))
    want = node.out(m[mask[0]].mean(0, keepdim=True))
    assert torch.allclose(g[0], want[0], atol=1e-6)
    assert g[1].abs().sum() == 0, "no observed vertex -> exactly zero (not relu(bias))"
    # the lever arm reaches the message: same features, different arms -> different node
    g2 = node(h, r + 1.0, mask)
    assert not torch.allclose(g2[0], g[0], atol=1e-5)


def test_lever_arms_and_part_positions_are_about_the_root_joint(mesh3d):
    prev = _test_prev()
    betas, _ = mesh3d._resolve_betas_K(prev, None, None)
    verts, joints = mesh3d._fk(prev, betas)
    pivot = joints[:, 0]
    r = lever_arms(verts, pivot)
    assert r.shape == (1, 778, 3)
    assert torch.allclose(r[0], (verts[0] - pivot[0]) * 100.0, atol=1e-5)
    # the wrist's own vertices sit near the pivot, a fingertip far from it (MANO hand ~ 18 cm)
    wrist_v = mesh3d.mano.weights[:, 0].argmax()
    tip = mesh3d.mano.fingertip_vertex_ids[2]                                  # middle fingertip
    assert float(r[0, wrist_v].norm()) < 6.0 and 10.0 < float(r[0, tip].norm()) < 25.0
    # translating the hand leaves lever arms and part positions unchanged
    t = torch.tensor([[0.05, -0.02, 0.1]])
    assert torch.allclose(lever_arms(verts + t.unsqueeze(1), pivot + t), r, atol=1e-4)
    parts = part_positions(verts, mesh3d.mano.weights, pivot)
    assert parts.shape == (1, 16, 3)
    W = mesh3d.mano.weights
    want0 = (W[:, 0] / W[:, 0].sum()) @ verts[0]
    assert torch.allclose(parts[0, 0], (want0 - pivot[0]) * 100.0, atol=1e-4)
    assert torch.allclose(part_positions(verts + t.unsqueeze(1), W, pivot + t), parts, atol=1e-4)
    # sixteen distinct places: the parts are spread over the hand, not collapsed on the pivot
    assert float(parts[0].norm(dim=-1).max()) > 8.0


def test_part_lever_is_per_part_shared_and_zero_for_unseen_parts():
    torch.manual_seed(6)
    ev_dim = 2 * HIDDEN + 1
    pl = PartLever(ev_dim, LEVER).eval()
    e = torch.randn(2, 16, ev_dim)
    e[0, 3] = 0.0                                   # part 3 of packet 0 saw nothing
    e[1] = 0.0                                      # packet 1 saw nothing at all
    r = torch.randn(2, 16, 3)
    u = pl(e, r)
    assert u.shape == (2, 16, LEVER)
    want = torch.relu(pl.lin(torch.cat([e[0, 5], r[0, 5]])))
    assert torch.allclose(u[0, 5], want, atol=1e-6)
    assert u[0, 3].abs().sum() == 0 and u[1].abs().sum() == 0, "unseen part -> exactly zero"
    # the lever arm reaches the term; the same W serves every part
    assert not torch.allclose(pl(e, r + 1.0)[0, 5], u[0, 5], atol=1e-5)
    assert sum(p.numel() for p in pl.parameters()) == (ev_dim + 3) * LEVER + LEVER


# ------------------------------------------------------------------ heads
def _root_extra_dim(model):
    return HIDDEN + (HIDDEN if model.mg_rigid else 0) + 16 * model.mg_root_lever


@pytest.mark.parametrize("arm", ["mesh3d", "rootlever"])
def test_fingers_never_read_the_root_extras_and_root_reads_everything(arm, request):
    model = request.getfixturevalue(arm)
    torch.manual_seed(1)
    ev_dim = 2 * HIDDEN + 1
    e = torch.randn(1, 16, ev_dim, requires_grad=True)
    extra = torch.randn(1, _root_extra_dim(model), requires_grad=True)
    out = model._decode_active(None, _test_prev(), e, root_extra=extra)
    for k in range(15):
        g_e, g_x = torch.autograd.grad(out[0, 6 + 3 * k: 9 + 3 * k].sum(), (e, extra),
                                       retain_graph=True)
        others = torch.ones(16, dtype=torch.bool)
        others[k + 1] = False
        assert torch.all(g_e[0, others] == 0) and g_e[0, k + 1].abs().sum() > 0
        assert torch.all(g_x == 0), f"finger {k} reads the root's extras"
    g_e, g_x = torch.autograd.grad(out[0, :6].sum(), (e, extra))
    assert torch.all(g_e[0].abs().sum(-1) > 0)
    # every block of the root's extra input is read: background | rigid node | part lever terms
    blocks = [(0, HIDDEN)]
    if model.mg_rigid:
        blocks.append((HIDDEN, 2 * HIDDEN))
    if model.mg_root_lever:
        blocks.append((_root_extra_dim(model) - 16 * LEVER, _root_extra_dim(model)))
    for a, b in blocks:
        assert g_x[0, a:b].abs().sum() > 0, (a, b)


# ------------------------------------------------------------------ contract
@pytest.mark.parametrize("arm", ["mesh3d", "rootlever"])
def test_empty_packet_returns_prev_bitwise(arm, request):
    model = request.getfixturevalue(arm)
    pk = _packet(_events([]), n_packets=2)
    pk.prev_state = torch.randn(2, 51)
    pk.prev_state[:, 2] = 0.5
    out = model.forward_packet(pk)
    assert torch.equal(out, pk.prev_state)


def test_mesh3d_forward_packet_end_to_end_and_ablation(mesh3d):
    prev = _test_prev()
    pk = _hand_packet(mesh3d, prev)
    out = mesh3d.forward_packet(pk)
    assert out.shape == (1, 51) and torch.isfinite(out).all()
    assert 0.9 < float(mesh3d.route_stats["route_frac_routed"]) < 1.0
    assert float(mesh3d.route_stats["route_joints_hit"]) >= 4
    assert float(mesh3d.route_stats["mg_rigid_mass"]) >= 30          # ~60 events on distinct vertices
    assert torch.equal(out, mesh3d.forward_packet(pk)), "same packet, same bits"
    # ablation: observations and `has` cleared -> the rigid node is exactly zero and the output
    # changes; the geometry (edge vectors, lever arms) stays, it is the prior not the evidence
    mesh3d.ablate_evidence = True
    try:
        out_abl = mesh3d.forward_packet(pk)
        assert float(mesh3d.route_stats["mg_rigid_mass"]) == 0.0      # "the mesh saw nothing"
    finally:
        mesh3d.ablate_evidence = False
    assert not torch.equal(out, out_abl)
    # a packet that misses the hand: no vertex saw an event -> every evidence and the rigid node
    # are zero; the update is still finite and not `prev` (background node + prev_mlp remain)
    far = _packet(_events([(0, 2.0, 2.0, 0.041, 1), (0, 237.0, 177.0, 0.042, 0)]))
    far.prev_state = prev
    out_far = mesh3d.forward_packet(far)
    assert float(mesh3d.route_stats["route_joints_hit"]) == 0.0
    assert float(mesh3d.route_stats["mg_rigid_mass"]) == 0.0
    assert torch.isfinite(out_far).all() and not torch.equal(out_far, prev)


def test_rigid_node_is_zero_under_ablation_inside_the_model(mesh3d):
    prev = _test_prev()
    pk = _hand_packet(mesh3d, prev)
    seen = {}
    hook = mesh3d.rigid_node.register_forward_hook(lambda m, i, o: seen.__setitem__("g", o.detach().clone()))
    try:
        mesh3d.forward_packet(pk)
        g_live = seen["g"]
        mesh3d.ablate_evidence = True
        try:
            mesh3d.forward_packet(pk)
        finally:
            mesh3d.ablate_evidence = False
        g_abl = seen["g"]
    finally:
        hook.remove()
    assert g_live.abs().sum() > 0 and g_abl.abs().sum() == 0


def test_rootlever_forward_packet_end_to_end_and_lever_terms_follow_the_pool(rootlever):
    prev = _test_prev()
    pk = _hand_packet(rootlever, prev)
    seen = {}
    hook = rootlever.part_lever.register_forward_hook(
        lambda m, i, o: seen.update(e=i[0].detach().clone(), r=i[1].detach().clone(), u=o.detach().clone()))
    try:
        out = rootlever.forward_packet(pk)
        assert out.shape == (1, 51) and torch.isfinite(out).all()
        assert "mg_rigid_mass" not in rootlever.route_stats
        assert torch.equal(out, rootlever.forward_packet(pk))
        # the lever terms are exactly zero for the parts the pool found unseen, nonzero otherwise
        unseen = seen["e"][0].abs().sum(-1) == 0
        assert torch.all(seen["u"][0][unseen].abs().sum(-1) == 0)
        assert torch.all(seen["u"][0][~unseen].abs().sum(-1) > 0) and int((~unseen).sum()) >= 4
        # the part centroids are about the root joint: the wrist part sits close to it
        assert float(seen["r"][0, 0].norm()) < float(seen["r"][0].norm(dim=-1).max())
        rootlever.ablate_evidence = True
        try:
            out_abl = rootlever.forward_packet(pk)
        finally:
            rootlever.ablate_evidence = False
        assert seen["u"].abs().sum() == 0, "ablation: no evidence, no lever term"
        assert not torch.equal(out, out_abl)
    finally:
        hook.remove()
    # a packet that misses the hand: every evidence is zero -> every lever term is zero, and the
    # update is the S37 one (background node + prev_mlp), finite and not `prev`
    far = _packet(_events([(0, 2.0, 2.0, 0.041, 1), (0, 237.0, 177.0, 0.042, 0)]))
    far.prev_state = prev
    out_far = rootlever.forward_packet(far)
    assert float(rootlever.route_stats["route_joints_hit"]) == 0.0
    assert torch.isfinite(out_far).all() and not torch.equal(out_far, prev)


def test_s38_mesh3d_config_diff_against_meshgraph():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s37_meshgraph_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / "s38_mesh3d_s3407.yaml"),
         "--allow", "MODEL.MESH_GRAPH_EDGE_GEO", "MODEL.MESH_GRAPH_RIGID_NODE",
         "MODEL.MESH_GRAPH_GEO_SCALE",
         "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_s38_rootlever_config_diff_against_meshgraph():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s37_meshgraph_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / "s38_rootlever_s3407.yaml"),
         "--allow", "MODEL.MESH_GRAPH_ROOT_LEVER", "MODEL.MESH_GRAPH_GEO_SCALE",
         "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
