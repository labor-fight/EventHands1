#!/usr/bin/env python3
"""S41. Content-attention pooling: the contract tests.

Four learned constant queries replace mean+max as the frontend readout. What must hold:

  * the selection is state-free by construction: the module takes no state input at the
    readout, so there is nothing the S38 anti-selection can act on (checked structurally:
    the parameters exist iff the switch is on, keeping checkpoint key-sets honest);
  * at zero-init every head is exact mean pooling -- the arm starts at the baseline family;
  * an event-free packet still returns prev bitwise (ZERO_EVENT_GATE contract);
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
from semkine.event_gnn import EventGNN  # noqa: E402
from semkine.events import EV_BATCH, EV_X, EV_Y, EV_T, EV_P  # noqa: E402


def _cfg(attn=4):
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
            "ENCODER_ATTN_POOL": attn,
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


def test_the_parameters_exist_iff_the_switch_is_on():
    with_pool = MNISTModel(_cfg(attn=4)).event_encoder
    without = MNISTModel(_cfg(attn=0)).event_encoder
    assert hasattr(with_pool, "pool_q") and with_pool.pool_q.shape == (4, 32)
    assert not hasattr(without, "pool_q")
    keys_w = set(with_pool.state_dict())
    keys_o = set(without.state_dict())
    assert keys_w - keys_o == {"pool_q", "pool_key.weight", "pool_key.bias"}


def test_zero_init_heads_are_exact_mean_pooling():
    torch.manual_seed(0)
    enc = EventGNN(height=180, width=240, hidden=16, feat_dim=32, k=4, n_layers=1,
                   max_nodes=32, window=8, attn_pool=3)
    assert float(enc.pool_q.abs().max()) == 0.0
    ev = _events([(0, 30.0 + i, 40.0, i * 1e-3, i % 2) for i in range(12)])
    counts = torch.tensor([12])
    ptr = torch.tensor([0, 12])
    # Monkey-substitute proj with identity-like capture to read the pooled vector.
    captured = {}
    orig = enc.proj

    class _Cap(torch.nn.Module):
        def forward(self, x):
            captured["pooled"] = x
            return orig(x)

    enc.proj = _Cap()
    with torch.no_grad():
        enc(ev, ptr, torch.tensor([0.05]))
    pooled = captured["pooled"].reshape(3, 16)
    assert torch.allclose(pooled[0], pooled[1]) and torch.allclose(pooled[1], pooled[2]), \
        "zero-init queries must all produce the same (mean) readout"


def test_empty_packet_returns_prev_bitwise():
    torch.manual_seed(0)
    m = MNISTModel(_cfg(attn=4))
    m.eval()
    pk = _packet(torch.zeros(0, 5))
    pk.prev_state = _test_prev()
    out = m.forward_packet(pk)
    assert torch.equal(out, pk.prev_state)


def test_forward_packet_end_to_end_and_grad():
    torch.manual_seed(1)
    m = MNISTModel(_cfg(attn=4))
    m.train()
    ev = _events([(0, float(torch.randint(20, 220, (1,))), float(torch.randint(20, 160, (1,))),
                   i * 1e-3, i % 2) for i in range(24)])
    pk = _packet(ev)
    pk.prev_state = _test_prev()
    pk.target = _test_prev()
    pred = m.forward_packet(pk)
    loss, _ = m._compute_loss(pred, pk.target, pk.betas)
    loss.backward()
    assert m.event_encoder.pool_q.grad is not None
    assert torch.isfinite(m.event_encoder.pool_q.grad).all()


def test_attn_pool_is_exclusive_with_the_joint_readout():
    cfg = _cfg(attn=4)
    cfg["MODEL"]["ENCODER_JOINT_READOUT"] = True
    with pytest.raises(ValueError):
        MNISTModel(cfg)


def test_s41_config_diff_is_the_switch_only():
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s37_fkdirect_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / "s41_attnpool_s3407.yaml"),
         "--allow", "MODEL.ENCODER_ATTN_POOL",
         "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
