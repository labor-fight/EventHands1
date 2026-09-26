#!/usr/bin/env python3
"""S39 surface coverage map for the root (docs/S39_COVMAP_PREREG.md): the contract tests.

  * partition: every vertex in exactly one of 64 patches, every patch non-empty, deterministic;
  * coverage map: shares sum to one, a packet that routes nothing gives exactly zero, no
    geometric quantity enters -- the same events under a translated prev give the same visible
    shares but a shifted event-share pattern (the footprint discrepancy the root is meant to read),
    and the map differs between a rotated and an unrotated prev for the same events;
  * heads: fingers never read the coverage map, the root does; knob off = the S37 routed arm
    (parameter set identical, S37 checkpoints load strictly);
  * contract: empty packet returns prev bitwise; `ablate_covmap` zeroes the map alone and changes
    the root output but not the finger outputs; `ablate_evidence` zeroes it too; the config differs
    from S37 routed only in the four covmap keys.
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
from semkine.mesh_graph import visible_vertices  # noqa: E402
from semkine.routed_readout import (COVMAP_CHANNELS, coverage_map, route_front_vertex_lbs,  # noqa: E402
                                    surface_patches)

HIDDEN = 32
PATCHES = 64
COV_HIDDEN = 24


def _cfg(covmap=True, **over):
    m = {
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
        "ENCODER_MAX_NODES": 256,
        "ENCODER_WINDOW": 8,
        "ACTIVE_HEAD": True,
        "ACTIVE_HIDDEN": 16,
    }
    if covmap:
        m.update({"ROOT_COVMAP": True, "ROOT_COVMAP_PATCHES": PATCHES,
                  "ROOT_COVMAP_HIDDEN": COV_HIDDEN, "ROOT_COVMAP_EDGE_PX": 2.0})
    m.update(over)
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


def _test_prev():
    prev = torch.zeros(1, 51)
    prev[0, 2] = 0.5
    return prev


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = MNISTModel(_cfg())
    m.eval()
    return m


def _geometry(model, prev):
    betas, K = model._resolve_betas_K(prev, None, None)
    verts, _ = model._fk(prev, betas)
    uv, z = model._project_verts(verts, K)
    vis = visible_vertices(verts, uv, z, model.mano.f, 180, 240)
    return uv, z, vis


def _hand_packet(model, prev, n=120, seed=0):
    """Events on visible vertices of `prev`'s hand (a few px of jitter) plus two far corners."""
    torch.manual_seed(seed)
    uv, z, vis = _geometry(model, prev)
    idx = vis[0].nonzero().squeeze(1)
    idx = idx[torch.randperm(len(idx))[:n]]
    xy = uv[0, idx] + torch.randn(len(idx), 2) * 1.5
    rows = [(0, float(xy[i, 0]), float(xy[i, 1]), 0.0004 * i, i % 2) for i in range(len(idx))]
    rows += [(0, 2.0, 2.0, 0.041, 1), (0, 237.0, 177.0, 0.042, 0)]
    pk = _packet(_events(rows))
    pk.prev_state = prev
    return pk


# ------------------------------------------------------------------ partition and map
def test_surface_patches_partition_every_vertex_deterministically(model):
    p = surface_patches(model.mano.v_template, PATCHES)
    assert p.shape == (778,) and int(p.min()) == 0 and int(p.max()) == PATCHES - 1
    sizes = torch.bincount(p, minlength=PATCHES)
    assert int(sizes.min()) >= 1 and int(sizes.max()) <= 60, sizes.tolist()   # fingertips are small patches
    assert torch.equal(p, surface_patches(model.mano.v_template, PATCHES))
    assert torch.equal(model.covmap_patch, p)


def test_coverage_map_shares_sum_to_one_and_nothing_routed_is_exactly_zero(model):
    prev = _test_prev()
    uv, z, vis = _geometry(model, prev)
    # nodes: 30 on the hand, 5 far from it, 5 dead
    idx = vis[0].nonzero().squeeze(1)[:30]
    px = torch.cat([uv[0, idx, 0], torch.full((10,), 5.0)]).unsqueeze(0)
    py = torch.cat([uv[0, idx, 1], torch.full((10,), 5.0)]).unsqueeze(0)
    mask = torch.ones(1, 40, dtype=torch.bool)
    mask[0, 35:] = False
    a, dist, vid = route_front_vertex_lbs(px, py, mask, uv, z, model.mano.weights, 16.0, 8)
    feat, any_r = coverage_map(px, py, mask, dist, vid, uv, vis, model.covmap_patch, 16.0, PATCHES)
    assert feat.shape == (1, PATCHES * COVMAP_CHANNELS + 1) and any_r.tolist() == [[1.0]]
    share, vis_share, band = feat[0, :PATCHES], feat[0, PATCHES:2 * PATCHES], feat[0, 2 * PATCHES:3 * PATCHES]
    assert share.sum() == pytest.approx(1.0, abs=1e-5), "event shares over the patches"
    assert vis_share.sum() == pytest.approx(1.0, abs=1e-5), "visible-surface shares"
    assert float(band.sum()) <= 1.0 + 1e-5 and float(feat[0, -1]) == pytest.approx(float(vis.float().mean()))
    # nothing on the hand: every entry exactly zero, gate zero
    far = torch.full((1, 6), 3.0)
    a2, d2, v2 = route_front_vertex_lbs(far, far, torch.ones(1, 6, dtype=torch.bool), uv, z,
                                        model.mano.weights, 16.0, 8)
    f2, g2 = coverage_map(far, far, torch.ones(1, 6, dtype=torch.bool), d2, v2, uv, vis,
                          model.covmap_patch, 16.0, PATCHES)
    assert f2.abs().sum() == 0 and g2.tolist() == [[0.0]]


def test_coverage_map_reads_the_footprint_discrepancy_not_the_geometry(model):
    """Same events, prev shifted by 8 px in x: the visible-surface shares barely move (same hand
    seen from the same side), the event shares move against them -- that discrepancy is the
    observation. Same events, same prev: identical map (a fixed function)."""
    prev = _test_prev()
    pk = _hand_packet(model, prev)
    uv, z, vis = _geometry(model, prev)
    ev = pk.events
    px, py = ev[:, EV_X].unsqueeze(0), ev[:, EV_Y].unsqueeze(0)
    mask = torch.ones(1, ev.shape[0], dtype=torch.bool)
    a, d, v = route_front_vertex_lbs(px, py, mask, uv, z, model.mano.weights, 16.0, 8)
    f0, _ = coverage_map(px, py, mask, d, v, uv, vis, model.covmap_patch, 16.0, PATCHES)
    f0b, _ = coverage_map(px, py, mask, d, v, uv, vis, model.covmap_patch, 16.0, PATCHES)
    assert torch.equal(f0, f0b)
    shifted = prev.clone()
    shifted[0, 0] += 8.0 * 0.5 / (603.45 * 0.375)          # 8 px at 0.5 m with the render focal
    uv2, z2, vis2 = _geometry(model, shifted)
    a2, d2, v2 = route_front_vertex_lbs(px, py, mask, uv2, z2, model.mano.weights, 16.0, 8)
    f1, _ = coverage_map(px, py, mask, d2, v2, uv2, vis2, model.covmap_patch, 16.0, PATCHES)
    vis_move = (f1[0, PATCHES:2 * PATCHES] - f0[0, PATCHES:2 * PATCHES]).abs().sum()
    ev_move = (f1[0, :PATCHES] - f0[0, :PATCHES]).abs().sum()
    band_move = (f1[0, 2 * PATCHES:3 * PATCHES] - f0[0, 2 * PATCHES:3 * PATCHES]).abs().sum()
    # measured 2026-09-21 for 4 / 8 / 16 px: vis 0.05 / 0.08 / 0.10 (perspective only), events
    # 0.52 / 0.72 / 1.10, band 0.38 / 0.57 / 0.84 -- the events move against the surface
    assert float(vis_move) < 0.15, float(vis_move)
    assert float(ev_move) > 5 * float(vis_move) and float(band_move) > 4 * float(vis_move), \
        (float(vis_move), float(ev_move), float(band_move))
    # a rotated prev also changes the map for the same events
    rot = prev.clone()
    rot[0, 5] = 0.3
    uv3, z3, vis3 = _geometry(model, rot)
    a3, d3, v3 = route_front_vertex_lbs(px, py, mask, uv3, z3, model.mano.weights, 16.0, 8)
    f3, _ = coverage_map(px, py, mask, d3, v3, uv3, vis3, model.covmap_patch, 16.0, PATCHES)
    assert not torch.allclose(f3, f0, atol=1e-3)


# ------------------------------------------------------------------ heads and contract
def test_fingers_never_read_the_coverage_map_and_root_does(model):
    torch.manual_seed(1)
    ev_dim = 2 * HIDDEN + 1
    feat = torch.randn(1, 64, requires_grad=True)
    e = torch.randn(1, 16, ev_dim, requires_grad=True)
    cov = torch.randn(1, COV_HIDDEN, requires_grad=True)
    out = model._decode_active(feat, _test_prev(), e, root_extra=cov)
    for k in range(15):
        g_cov, = torch.autograd.grad(out[0, 6 + 3 * k: 9 + 3 * k].sum(), (cov,), retain_graph=True)
        assert torch.all(g_cov == 0), f"finger {k} reads the coverage map"
    g_cov, = torch.autograd.grad(out[0, :6].sum(), (cov,))
    assert g_cov.abs().sum() > 0
    assert model.root_head.in_features == 64 + 16 * ev_dim + COV_HIDDEN
    assert model.covmap_mlp[0].in_features == PATCHES * COVMAP_CHANNELS + 1


def test_covmap_off_is_the_s37_routed_arm():
    torch.manual_seed(0)
    off = MNISTModel(_cfg(covmap=False))
    assert not off.root_covmap and not hasattr(off, "covmap_mlp") and not hasattr(off, "covmap_patch")
    assert off.root_head.in_features == 64 + 16 * (2 * HIDDEN + 1)
    on = MNISTModel(_cfg())
    extra = set(on.state_dict()) - set(off.state_dict())
    assert extra == {"covmap_mlp.0.weight", "covmap_mlp.0.bias"}, extra
    # knob needs the routed arm
    with pytest.raises(ValueError):
        MNISTModel(_cfg(ROUTED_READOUT=False))


def test_empty_packet_returns_prev_bitwise(model):
    pk = _packet(_events([]), n_packets=2)
    pk.prev_state = torch.randn(2, 51)
    pk.prev_state[:, 2] = 0.5
    out = model.forward_packet(pk)
    assert torch.equal(out, pk.prev_state)


def test_forward_packet_end_to_end_and_covmap_ablation(model):
    prev = _test_prev()
    pk = _hand_packet(model, prev)
    out = model.forward_packet(pk)
    assert out.shape == (1, 51) and torch.isfinite(out).all()
    assert float(model.route_stats["covmap_any"]) == 1.0
    assert 0.9 < float(model.route_stats["route_frac_routed"]) <= 1.0
    assert torch.equal(out, model.forward_packet(pk)), "same packet, same bits"
    model.ablate_covmap = True
    try:
        out_nc = model.forward_packet(pk)
    finally:
        model.ablate_covmap = False
    assert not torch.equal(out_nc[:, :6], out[:, :6]), "the map must reach the root"
    assert torch.equal(out_nc[:, 6:], out[:, 6:]), "the map must not reach the fingers"
    model.ablate_evidence = True
    try:
        out_ne = model.forward_packet(pk)
    finally:
        model.ablate_evidence = False
    assert not torch.equal(out_ne, out)
    # events that all miss the hand: nothing routed -> the map is exactly zero and the root's
    # extra input is exactly zero (the gate), while the update stays finite and not `prev`
    far = _packet(_events([(0, 2.0, 2.0, 0.041, 1), (0, 237.0, 177.0, 0.042, 0)]))
    far.prev_state = prev
    seen = {}
    hook = model.covmap_mlp.register_forward_hook(lambda m, i, o: seen.__setitem__("cov_in", i[0].detach().clone()))
    try:
        out_far = model.forward_packet(far)
    finally:
        hook.remove()
    assert float(model.route_stats["covmap_any"]) == 0.0 and seen["cov_in"].abs().sum() == 0
    assert torch.isfinite(out_far).all() and not torch.equal(out_far, prev)


def test_s39_config_differs_from_s37_routed_only_in_the_covmap():
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s37_routed_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / "s39_covmap_s3407.yaml"),
         "--allow", "MODEL.ROOT_COVMAP", "MODEL.ROOT_COVMAP_PATCHES", "MODEL.ROOT_COVMAP_HIDDEN",
         "MODEL.ROOT_COVMAP_EDGE_PX", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
