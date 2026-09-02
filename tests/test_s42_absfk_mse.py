#!/usr/bin/env python3
"""S42. The absolute-FK regulariser on mse_51d: the contract tests.

The graft takes S39's one measured win (abs MPJPE -2.51 mm, attributable to the metre-scale
absolute-FK term) and adds it, weighted, to the loss that wins recursive RA. What must hold:

  * weight 0 is bit-identical to the plain mse (every existing arm stays untouched);
  * a rigid z-shift -- invisible to wrist-aligned metrics, exactly what the term exists
    for -- is charged at `weight * shift` metres;
  * gradient flows through the FK to the prediction;
  * the config differs from S37 in the weight and run naming only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from model import MNISTModel  # noqa: E402


def _cfg(w=0.5):
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
        "LOSS": {"ABS_FK_WEIGHT": w, "LAMBDA_POSE": 450.0, "LAMBDA_T": 30000.0,
                 "LAMBDA_R": 60.0, "NORMALIZER": 51, "LOG10": True},
        "TRAIN": {"LR": 1e-3, "WARMUP_STEPS": 0},
        "DATA": {"HEIGHT": 180, "WIDTH": 240},
        "MANO": {"NPZ": "assets/mano_right.npz"},
    }


def _state(z=0.5):
    s = torch.zeros(1, 51)
    s[0, 2] = z
    return s


def test_weight_zero_is_bit_identical_to_plain_mse():
    torch.manual_seed(0)
    m0 = MNISTModel(_cfg(w=0.0))
    y = _state()
    pred = y.clone()
    pred[0, 2] += 0.05
    pred[0, 7] += 0.1
    loss0, parts0 = m0._compute_loss(pred, y, torch.zeros(1, 10))
    assert "loss_abs_fk" not in parts0
    manual = (450.0 * parts0["mano_loss"] + 60.0 * parts0["rot_loss"]
              + 30000.0 * parts0["pos_loss"]) / 51.0
    assert torch.equal(loss0, manual)


def test_a_rigid_shift_is_charged_at_weight_times_metres():
    torch.manual_seed(0)
    m5 = MNISTModel(_cfg(w=0.5))
    torch.manual_seed(0)
    m0 = MNISTModel(_cfg(w=0.0))
    y = _state()
    pred = y.clone()
    pred[0, 2] += 0.04
    betas = torch.zeros(1, 10)
    l5, p5 = m5._compute_loss(pred, y, betas)
    l0, _ = m0._compute_loss(pred, y, betas)
    assert abs(float(p5["loss_abs_fk"]) - 0.04) < 1e-4
    assert abs(float(l5 - l0) - 0.5 * 0.04) < 1e-4, \
        f"the weighted abs term must be the only gap, got {float(l5 - l0)}"


def test_gradient_reaches_the_prediction_through_fk():
    torch.manual_seed(0)
    m = MNISTModel(_cfg(w=0.5))
    y = _state()
    pred = y.clone().requires_grad_(True)
    loss, _ = m._compute_loss(pred + 0.01, y, torch.zeros(1, 10))
    loss.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    assert float(pred.grad.abs().sum()) > 0


def test_s42_config_diff_is_the_weight_only():
    import subprocess
    r = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "diff_configs.py"),
         str(ROOT / "configs" / "semkine" / "s37_fkdirect_s3407.yaml"),
         str(ROOT / "configs" / "semkine" / "s42_absfk_s3407.yaml"),
         "--allow", "LOSS.ABS_FK_WEIGHT",
         "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR"],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
