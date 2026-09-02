#!/usr/bin/env python3
"""S37. Render-free FK conditioning: the contract tests.

The arm replaces the rasterized `sil`/`inv` node channels with four analytic values measured
against the projected MANO FK of the previous state. What must hold:

  * the features discriminate on-hand from off-hand events (or they are not a silhouette);
  * only rows the encoder will gather are written, and they land on the sampled indices;
  * the tracking contract survives: an event-free packet returns `prev` bitwise;
  * the arm differs from S36 in the conditioning path and nowhere else.
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


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = MNISTModel(_cfg())
    m.eval()
    return m


def _test_prev():
    """A previous state whose hand sits half a metre in front of the camera.

    All-zeros would put the wrist at the optical centre (z = 0), where the projection
    degenerates and vertices land everywhere in the frame -- the layout is
    `[transl(3), global rot(3), local pose(45)]`.
    """
    prev = torch.zeros(1, 51)
    prev[0, 2] = 0.5
    return prev


def _projected_hand(model, prev):
    """(on-hand pixel, far-corner pixel) straight from the model's own FK + intrinsics."""
    betas, K = model._resolve_betas_K(prev, None, None)
    verts, _ = model._fk(prev, betas)
    fx, fy, cx, cy = model._intrinsics(K)
    x, y, z = verts[0].unbind(-1)
    u = (fx * x / z.clamp(min=1e-6) + cx).round()
    v = (fy * y / z.clamp(min=1e-6) + cy).round()
    ok = (u >= 0) & (u < model.render_w) & (v >= 0) & (v < model.render_h)
    assert ok.any(), "the test pose must project inside the render frame"
    i = int(ok.nonzero()[0])
    corners = torch.tensor([[2.0, 2.0], [238.0, 2.0], [2.0, 178.0], [238.0, 178.0]])
    d = (corners[:, 0:1] - u[ok][None, :]).square() + (corners[:, 1:2] - v[ok][None, :]).square()
    far = corners[int(d.min(dim=1).values.argmax())]
    assert float(d.min(dim=1).values.max()).__pow__(0.5) > 4 * model.FK_TAU_SURF
    return (float(u[i]), float(v[i])), (float(far[0]), float(far[1]))


def test_fk_extra_discriminates_on_hand_from_off_hand(model):
    prev = _test_prev()
    (u, v), far = _projected_hand(model, prev)
    ev = _events([(0, u, v, 0.01, 1), (0, far[0], far[1], 0.02, 0)])
    pk = _packet(ev)
    pk.prev_state = prev
    betas, K = model._resolve_betas_K(pk.prev_state, pk.betas, None)
    verts, joints = model._fk(pk.prev_state, betas)
    extra = model._fk_extra(ev, pk.ptr, verts, joints, K)
    assert extra.shape == (2, 4)
    g_on, g_off = float(extra[0, 0]), float(extra[1, 0])
    assert g_on > 0.9, f"event on a projected vertex must read ~1, got {g_on}"
    assert g_off < 0.05, f"event far from the hand must read ~0, got {g_off}"
    # The gated depth behaves like `inv`: positive on the hand, ~0 off it.
    assert float(extra[0, 1]) > 0.0
    assert float(extra[1, 1]) < 0.05
    # All four channels stay in [0, 1] -- same range discipline as the raster faces.
    assert float(extra.min()) >= 0.0 and float(extra.max()) <= 1.0


def test_fk_extra_writes_only_sampled_rows(model):
    torch.manual_seed(1)
    n = 4 * model.event_encoder.max_nodes  # force subsampling
    ev = _events([(0, float(torch.randint(0, 240, (1,))), float(torch.randint(0, 180, (1,))),
                   i * 1e-4, i % 2) for i in range(n)])
    pk = _packet(ev)
    pk.prev_state = _test_prev()
    betas, K = model._resolve_betas_K(pk.prev_state, pk.betas, None)
    verts, joints = model._fk(pk.prev_state, betas)
    extra = model._fk_extra(ev, pk.ptr, verts, joints, K)
    src, mask = model.event_encoder._sample(ev, pk.ptr)
    kept = src.reshape(-1)[mask.reshape(-1)]
    written = (extra.abs().sum(-1) > 0).nonzero().squeeze(-1)
    assert set(written.tolist()) <= set(kept.tolist())
    # The sampled indices are pairwise distinct, so the scatter has one writer per row.
    assert kept.unique().numel() == kept.numel()


def test_empty_packet_returns_prev_bitwise(model):
    pk = _packet(_events([]), n_packets=2)
    pk.prev_state = torch.randn(2, 51)
    out = model.forward_packet(pk)
    assert torch.equal(out, pk.prev_state)


def test_forward_packet_end_to_end(model):
    prev = _test_prev()
    (u, v), _far = _projected_hand(model, prev)
    ev = _events([(0, u, v, 0.001 * i, i % 2) for i in range(32)])
    pk = _packet(ev)
    pk.prev_state = prev + 0.01 * torch.randn(1, 51)
    out = model.forward_packet(pk)
    assert out.shape == (1, 51)
    assert torch.isfinite(out).all()


def test_prev_fk_and_prev_render_are_mutually_exclusive():
    cfg = _cfg()
    cfg["MODEL"]["PREV_RENDER"] = True
    with pytest.raises(ValueError, match="mutually exclusive"):
        MNISTModel(cfg)


def test_s37_config_diff_is_the_conditioning_path_only():
    """The arm must differ from S36 in MODEL.PREV_* and run naming, nowhere else."""
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "diff_configs.py"),
            str(ROOT / "configs" / "semkine" / "s36_eventgnn_s3407.yaml"),
            str(ROOT / "configs" / "semkine" / "s37_fkdirect_s3407.yaml"),
            "--allow", "MODEL.PREV_RENDER", "MODEL.PREV_FK_DIRECT",
            "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
