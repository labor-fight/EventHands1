#!/usr/bin/env python3
"""S39. SO(3)+FK loss with the §13 remedies: the contract tests.

The arm swaps the elementwise 51D MSE for `so3_trans_fk` on the S37 EventGNN line. The §13
pit this must not refall into: the bare loss left absolute translation unconstrained
(SmoothL1's β=1 m puts centimetre errors in the vanishing-gradient quadratic region, and the
FK term is root-relative), degrading non-aligned MPJPE 63.7 -> 83.6 mm. The remedies --
TRANS_BETA=0.01 and ABS_FK_WEIGHT=1.0 -- were implemented in S4 but never trained; what is
checked here is that they are actually *wired*, not merely present in the YAML:

  * the loss is zero at the ground truth (both branches decode through the same MANO);
  * TRANS_BETA reaches the SmoothL1: a 1 cm error costs ~100x more at β=0.01 than β=1;
  * ABS_FK_WEIGHT reaches the sum: a rigid z-shift (invisible to root-relative FK) is paid;
  * the packet path (PREV_FK_DIRECT conditioning) trains one step under the new loss.
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


def _cfg(trans_beta=0.01, abs_fk_weight=1.0):
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
        "LOSS": {
            "TYPE": "so3_trans_fk",
            "ROT_WEIGHT": 1.0,
            "TRANS_WEIGHT": 1.0,
            "FK_WEIGHT": 2.0,
            "TRANS_BETA": trans_beta,
            "ABS_FK_WEIGHT": abs_fk_weight,
            "LAMBDA_POSE": 450.0,
            "LAMBDA_T": 30000.0,
            "LAMBDA_R": 60.0,
            "NORMALIZER": 51,
            "LOG10": True,
        },
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


def _state(z=0.5):
    # Layout: [transl 0:3, global rot 3:6, local residual 6:51].
    s = torch.zeros(1, 51)
    s[0, 2] = z
    return s


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = MNISTModel(_cfg())
    m.eval()
    return m


def test_the_loss_is_zero_at_the_ground_truth(model):
    y = _state()
    y[0, 7] = 0.3  # a bent joint, so the FK terms see articulation
    loss, parts = model._compute_loss(y.clone(), y, torch.zeros(1, 10))
    assert float(loss) < 1e-6, f"loss at GT is {float(loss)}"
    for k in ("loss_rot", "loss_trans", "loss_fk", "loss_abs_fk"):
        assert float(parts[k]) < 1e-6, f"{k} at GT is {float(parts[k])}"


def test_trans_beta_reaches_the_smooth_l1():
    """A 1 cm translation error must cost ~100x more at β=0.01 than at the legacy β=1."""
    torch.manual_seed(0)
    sharp = MNISTModel(_cfg(trans_beta=0.01, abs_fk_weight=0.0))
    torch.manual_seed(0)
    legacy = MNISTModel(_cfg(trans_beta=1.0, abs_fk_weight=0.0))
    y = _state()
    pred = y.clone()
    pred[0, 0] += 0.01  # 1 cm x error: pure translation, rotation identical
    betas = torch.zeros(1, 10)
    t_sharp = float(sharp._compute_loss(pred, y, betas)[1]["loss_trans"])
    t_legacy = float(legacy._compute_loss(pred, y, betas)[1]["loss_trans"])
    # SmoothL1 at err=0.01: β=0.01 -> err - β/2 = 0.005 ; β=1 -> err²/2β ≈ 5e-5 (per-elem mean /3)
    assert t_sharp > 50 * t_legacy, f"β not wired: {t_sharp} vs {t_legacy}"


def test_abs_fk_weight_pays_for_a_rigid_shift():
    """A rigid +5 cm z-shift is invisible to root-relative FK; only L_absFK may charge it."""
    torch.manual_seed(0)
    with_abs = MNISTModel(_cfg(abs_fk_weight=1.0))
    torch.manual_seed(0)
    without = MNISTModel(_cfg(abs_fk_weight=0.0))
    y = _state()
    pred = y.clone()
    pred[0, 2] += 0.05
    betas = torch.zeros(1, 10)
    p_with = with_abs._compute_loss(pred, y, betas)[1]
    p_without = without._compute_loss(pred, y, betas)[1]
    assert float(p_with["loss_fk"]) < 1e-6, "rigid shift leaked into root-relative FK"
    assert abs(float(p_with["loss_abs_fk"]) - 0.05) < 1e-4
    # Same value logged either way; only the weight decides whether the total pays it.
    total_gap = float(with_abs._compute_loss(pred, y, betas)[0]) - float(
        without._compute_loss(pred, y, betas)[0])
    assert abs(total_gap - 0.05) < 1e-3, f"ABS_FK_WEIGHT not in the sum: gap {total_gap}"


def test_the_packet_path_trains_one_step(model):
    """forward_packet -> so3 loss -> backward, on the exact S39 conditioning path."""
    torch.manual_seed(1)
    ev = _events([(0, float(torch.randint(20, 220, (1,))), float(torch.randint(20, 160, (1,))),
                   i * 1e-3, i % 2) for i in range(32)])
    pk = _packet(ev)
    pk.prev_state = _state()
    pk.target = _state()
    pk.target[0, 7] = 0.2
    m = MNISTModel(_cfg())
    m.train()
    pred = m.forward_packet(pk)
    loss, parts = m._compute_loss(pred, pk.target, pk.betas)
    assert torch.isfinite(loss), f"non-finite loss {float(loss)}"
    loss.backward()
    g = [p.grad.abs().sum() for p in m.event_encoder.parameters() if p.grad is not None]
    assert g and float(sum(g)) > 0, "no gradient reached the event encoder"


def test_s39_config_diff_is_the_loss_only():
    """The arm must differ from S37 in the LOSS block and run naming, nowhere else."""
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "diff_configs.py"),
            str(ROOT / "configs" / "semkine" / "s37_fkdirect_s3407.yaml"),
            str(ROOT / "configs" / "semkine" / "s39_so3fk_s3407.yaml"),
            "--allow", "LOSS.TYPE", "LOSS.ROT_WEIGHT", "LOSS.TRANS_WEIGHT",
            "LOSS.FK_WEIGHT", "LOSS.TRANS_BETA", "LOSS.ABS_FK_WEIGHT",
            "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr


def test_s39_seeds_differ_only_in_seed():
    import subprocess

    r = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "diff_configs.py"),
            str(ROOT / "configs" / "semkine" / "s39_so3fk_s3407.yaml"),
            str(ROOT / "configs" / "semkine" / "s39_so3fk_s3408.yaml"),
            "--allow", "SEED", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "EVAL.OUTPUT_DIR",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stdout + r.stderr
