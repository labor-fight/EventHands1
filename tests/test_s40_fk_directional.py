#!/usr/bin/env python3
"""S40. Directional FK channels: the contract tests.

Two channels join the four of S37: the unit direction from the event to its nearest
projected vertex, gated by `g_surf`. What must hold:

  * the width is 6 and the encoder consumes it;
  * the direction points *toward* the hand and is gated to ~0 far away;
  * near the surface the 1 px divisor clamp fades the direction instead of blowing it up;
  * the config differs from S37 in the switch and run naming only.
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
            "FK_DIRECTIONAL": True,
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


def _projections(model, prev):
    betas, K = model._resolve_betas_K(prev, None, None)
    verts, joints = model._fk(prev, betas)
    fx, fy, cx, cy = model._intrinsics(K)
    x, y, z = verts[0].unbind(-1)
    u = fx * x / z.clamp(min=1e-6) + cx
    v = fy * y / z.clamp(min=1e-6) + cy
    return verts, joints, K, u, v


def test_direction_points_toward_the_hand(model):
    prev = _test_prev()
    verts, joints, K, u, v = _projections(model, prev)
    # An event strictly to the left of *every* projected vertex: its nearest vertex is to
    # the right, so the (gated) direction must have du > 0.
    i = int(u.argmin())
    e = _events([(0, float(u[i]) - 10.0, float(v[i]), 0.01, 1)])
    pk = _packet(e)
    extra = model._fk_extra(e, pk.ptr, verts, joints, K)
    assert extra.shape == (1, 6)
    assert float(extra[0, 4]) > 0.05, f"du*g_surf should be positive, got {float(extra[0, 4])}"
    assert abs(float(extra[0, 4])) <= 1.0 and abs(float(extra[0, 5])) <= 1.0


def test_direction_is_gated_far_away_and_clamped_on_surface(model):
    prev = _test_prev()
    verts, joints, K, u, v = _projections(model, prev)
    corners = [(2.0, 2.0), (238.0, 2.0), (2.0, 178.0), (238.0, 178.0)]
    d2 = [min((cu - u).square().add((cv - v).square()).min() for _ in [0]) for cu, cv in corners]
    far = corners[int(torch.tensor(d2).argmax())]
    on = (float(u[0].round()), float(v[0].round()))
    e = _events([(0, far[0], far[1], 0.01, 1), (0, on[0], on[1], 0.02, 0)])
    pk = _packet(e)
    extra = model._fk_extra(e, pk.ptr, verts, joints, K)
    # Far corner: g_surf gates the direction to ~0.
    assert abs(float(extra[0, 4])) < 0.05 and abs(float(extra[0, 5])) < 0.05
    # On the surface (d < 1 px): the clamp keeps |direction| <= d/1px, i.e. it fades out
    # rather than normalising sub-pixel noise to unit length.
    d_on = float((u - on[0]).square().add((v - on[1]).square()).min().sqrt())
    assert (extra[1, 4].square() + extra[1, 5].square()).sqrt() <= max(d_on, 1e-6) + 1e-4


def test_forward_packet_consumes_six_channels(model):
    from semkine.encoder import TOKEN_DIM
    assert model.event_encoder.in_dim == TOKEN_DIM + 6
    torch.manual_seed(2)
    ev = _events([(0, float(torch.randint(20, 220, (1,))), float(torch.randint(20, 160, (1,))),
                   i * 1e-3, i % 2) for i in range(24)])
    pk = _packet(ev)
    pk.prev_state = _test_prev()
    out = model.forward_packet(pk)
    assert out.shape == (1, 51) and torch.isfinite(out).all()


def test_empty_packet_returns_prev_bitwise(model):
    pk = _packet(torch.zeros(0, 5))
    pk.prev_state = _test_prev()
    out = model.forward_packet(pk)
    assert torch.equal(out, pk.prev_state)


def test_s40_config_diff_is_the_switch_only():
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s37_fkdirect_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / "s40_fkdir6_s3407.yaml"),
         "--allow", "MODEL.FK_DIRECTIONAL",
         "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
