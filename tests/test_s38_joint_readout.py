#!/usr/bin/env python3
"""S38. Joint-query readout: the contract tests.

The arm replaces the joint decoders' shared pooled input with one spatial-kernel readout per
LBS joint, centred on the previous state's projected FK. What must hold, each a lesson this
repo has already paid for:

  * the graph and the messages stay state-free -- queries may steer the *readout* only
    (KEG, loop gain 0.910): the pooled feature is bitwise identical with and without queries;
  * the readout is spatially selective (or it is global pooling with extra steps);
  * a joint's local feature is its decoder's only evidence path (KSGN: an ignorable bypass
    gets ignored) -- enforced by construction, checked here via the head input width;
  * the tracking contract survives: an event-free packet returns `prev` bitwise.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from model import MNISTModel  # noqa: E402
from semkine.events import EV_BATCH, EV_X, EV_Y, EV_T, EV_P  # noqa: E402


def _cfg():
    return {
        "MODEL": {
            "POSE_REPR": "mano_full_axis_angle",
            "OUTPUT_DIM": 51,
            "MANO_NCOMPS": 45,
            "PREDICT_DELTA": True,
            "PREVPOS_EMBED": True,
            "PREV_RENDER": False,
            "PREV_FK_DIRECT": True,
            "ENCODER_JOINT_READOUT": True,
            "ZERO_EVENT_GATE": True,
            "RENDER_H": 180,
            "RENDER_W": 240,
            "RENDER_SCALE": 0.375,
            "ENCODER": "event_gnn",
            "ENCODER_HIDDEN": 32,
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
    return _Packet(
        events=events,
        ptr=ptr,
        prev_state=torch.zeros(n_packets, 51),
        target=torch.zeros(n_packets, 51),
        betas=torch.zeros(n_packets, 10),
        camera_K=None,
        delta_t_s=torch.full((n_packets,), 0.05),
        counts=counts,
    )


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


def _spread_events(n=48):
    torch.manual_seed(3)
    return _events([(0, float(torch.randint(20, 220, (1,))), float(torch.randint(20, 160, (1,))),
                     i * 1e-3, i % 2) for i in range(n)])


def test_the_pooled_feature_ignores_the_queries(model):
    """State steers the readout only: `feat` must be bitwise identical with/without queries."""
    ev = _spread_events()
    pk = _packet(ev)
    pk.prev_state = _test_prev()
    betas, K = model._resolve_betas_K(pk.prev_state, pk.betas, None)
    with torch.no_grad():
        verts, joints = model._fk(pk.prev_state, betas)
        extra = model._fk_extra(ev, pk.ptr, verts, joints, K)
        queries = model._joint_queries(verts, K)
        feat_alone = model.event_encoder(ev, pk.ptr, pk.delta_t_s, extra)
        feat_q, nodes = model.event_encoder(ev, pk.ptr, pk.delta_t_s, extra, queries)
    assert torch.equal(feat_alone, feat_q)
    assert nodes.shape == (1, 16, model.event_encoder.hidden)


def test_the_readout_is_spatially_selective(model):
    """Queries at different pixels must read different mixtures of the nodes."""
    ev = _events([(0, 30.0, 30.0, 0.01, 1), (0, 31.0, 30.0, 0.02, 0),
                  (0, 200.0, 150.0, 0.03, 1), (0, 201.0, 150.0, 0.04, 0)])
    pk = _packet(ev)
    pk.prev_state = _test_prev()
    betas, K = model._resolve_betas_K(pk.prev_state, pk.betas, None)
    with torch.no_grad():
        verts, joints = model._fk(pk.prev_state, betas)
        extra = model._fk_extra(ev, pk.ptr, verts, joints, K)
        # 16 queries as the sigma parameter demands: half on one event cluster, half on
        # the other, 170 px apart.
        queries = torch.tensor([[30.0, 30.0]] * 8 + [[200.0, 150.0]] * 8).view(1, 16, 2)
        feat, nodes = model.event_encoder(ev, pk.ptr, pk.delta_t_s, extra, queries)
    assert not torch.allclose(nodes[0, 0], nodes[0, 15]), \
        "queries 170 px apart read the same feature: the kernel is not selective"
    # Same query pixel, same readout: the kernel is a pure function of geometry.
    assert torch.equal(nodes[0, 0], nodes[0, 7])


def test_empty_packet_returns_prev_bitwise(model):
    pk = _packet(_events([]), n_packets=2)
    pk.prev_state = torch.randn(2, 51)
    out = model.forward_packet(pk)
    assert torch.equal(out, pk.prev_state)


def test_forward_packet_end_to_end_and_head_width(model):
    ev = _spread_events()
    pk = _packet(ev)
    pk.prev_state = _test_prev() + 0.01 * torch.randn(1, 51)
    out = model.forward_packet(pk)
    assert out.shape == (1, 51)
    assert torch.isfinite(out).all()
    # The joint decoders read `hidden + 3` wide input: the local feature is the only
    # evidence path, there is no fallback onto the pooled vector.
    hid = model.event_encoder.hidden
    for head in model.joint_heads:
        assert head[0].in_features == hid + 3


def test_joint_readout_requires_fk_direct():
    cfg = _cfg()
    cfg["MODEL"]["PREV_FK_DIRECT"] = False
    with pytest.raises(ValueError, match="PREV_FK_DIRECT"):
        MNISTModel(cfg)


def test_queries_are_the_16_lbs_joints_in_decoder_order(model):
    """Query k+1 must move when joint k's angle moves, and the wrist query must not."""
    prev = _test_prev()
    betas, K = model._resolve_betas_K(prev, None, None)
    with torch.no_grad():
        verts, _ = model._fk(prev, betas)
        q0 = model._joint_queries(verts, K)
        bent = prev.clone()
        bent[0, 6:9] = 0.8  # joint 0 of the 45D local block
        verts_b, _ = model._fk(bent, betas)
        q1 = model._joint_queries(verts_b, K)
    assert q0.shape == (1, 16, 2)
    d_wrist = (q1[0, 0] - q0[0, 0]).norm()
    d_joint = (q1[0, 1] - q0[0, 1]).norm()
    # Bending a finger joint moves that joint's own query and (up to LBS blending
    # leakage) not the wrist's.
    assert d_joint > 1.0, f"joint query moved only {float(d_joint):.3f} px"
    assert d_wrist < d_joint / 5, f"wrist moved {float(d_wrist):.3f} vs joint {float(d_joint):.3f}"


def test_s38_config_diff_is_the_readout_only():
    """The arm must differ from S37 in the readout switch and run naming, nowhere else."""
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "diff_configs.py"),
            str(ROOT / "configs" / "semkine" / "s37_fkdirect_s3407.yaml"),
            str(ROOT / "configs" / "semkine" / "s38_jointreadout_s3407.yaml"),
            "--allow", "MODEL.ENCODER_JOINT_READOUT",
            "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
