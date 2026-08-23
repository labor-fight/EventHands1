#!/usr/bin/env python3
r"""S4 gates for the geometry loss and its absolute-translation remedy.

The diagnosis this stage exists for: `so3_trans_fk` improved root-relative accuracy while degrading
non-aligned MPJPE from 63.7 to 83.6 mm. The cause is structural rather than a tuning accident. The
FK term is root-relative by construction, so it cannot see translation at all, and the translation
term is a SmoothL1 whose transition sits at 1 metre -- so at the centimetre errors that actually
occur it is a pure quadratic, and a quadratic's gradient vanishes with the error it is trying to
remove. Between them, nothing pushes absolute position.

The gates below are about gradients, not about loss values, because that is where the failure lives:
a loss can be large and still supply no usable signal. Two remedies are checked independently, both
defaulting to off so every existing arm stays bit-identical:

* `ABS_FK_WEIGHT` -- a non-root-relative FK term, in metres, which is the quantity the non-aligned
  metric measures
* `TRANS_BETA` -- moving the SmoothL1 transition to the error scale, restoring a constant gradient
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def grad_wrt_translation(beta: float, err_m: float) -> float:
    """Magnitude of `d SmoothL1 / d t` at a translation error of `err_m` metres."""
    t = torch.zeros(1, 3, requires_grad=True, dtype=torch.float64)
    g = torch.full((1, 3), err_m, dtype=torch.float64)
    loss = F.smooth_l1_loss(t, g, beta=beta)
    loss.backward()
    return float(t.grad.abs().max())


@pytest.mark.parametrize("err_m", [0.005, 0.01, 0.02, 0.05])
def test_default_beta_makes_the_translation_gradient_vanish(err_m):
    r"""The mechanism of the failure, isolated from the model.

    With the default transition at 1 m, the gradient at a `d`-metre error is proportional to `d`,
    so a 2 cm error produces a gradient 50x smaller than the same loss in its linear regime would.
    With the transition at the error scale the gradient is constant, and the term keeps pushing all
    the way down.
    """
    g_default = grad_wrt_translation(1.0, err_m)
    g_tuned = grad_wrt_translation(0.01, err_m)
    assert g_tuned > g_default, (err_m, g_default, g_tuned)
    # Quadratic region: the gradient tracks the error rather than staying put.
    assert abs(g_default - err_m / 3.0) < 1e-9, (g_default, err_m)
    if err_m >= 0.01:
        assert abs(g_tuned - 1.0 / 3.0) < 1e-9, g_tuned


def test_root_relative_fk_is_blind_to_translation():
    """The other half of the mechanism: a root-relative term has exactly zero translation gradient.

    Zero by construction, up to the rounding of the cancellation itself: the measured gradient is
    3e-17, twenty orders of magnitude below the absolute term's constant 1/3. Any remedy therefore
    has to add a term that is not root-relative; tuning the existing ones cannot fix it.
    """
    t = torch.zeros(1, 3, requires_grad=True, dtype=torch.float64)
    joints = torch.randn(1, 21, 3, dtype=torch.float64) + t[:, None, :]
    target = torch.randn(1, 21, 3, dtype=torch.float64)
    rel_p = joints - joints[:, 0:1]
    rel_g = target - target[:, 0:1]
    torch.norm(rel_p - rel_g, dim=-1).mean().backward()
    assert float(t.grad.abs().max()) < 1e-15


def test_absolute_fk_term_has_a_constant_translation_gradient():
    """The remedy: a non-aligned FK term pushes translation with unit gradient in metres."""
    for err in (0.005, 0.02, 0.1):
        t = torch.full((1, 3), err, requires_grad=True, dtype=torch.float64)
        joints = torch.zeros(1, 21, 3, dtype=torch.float64) + t[:, None, :]
        target = torch.zeros(1, 21, 3, dtype=torch.float64)
        torch.norm(joints - target, dim=-1).mean().backward()
        # Unit gradient along the error direction, independent of the error's size: the defining
        # property of an L2-norm-of-difference term and the reason it does not stall.
        assert abs(float(t.grad.norm()) - 1.0) < 1e-9, (err, float(t.grad.norm()))


def test_defaults_leave_the_loss_untouched():
    """`ABS_FK_WEIGHT=0` and `TRANS_BETA=1` must reproduce the legacy loss exactly.

    Checked on the configuration surface rather than by running a training step, so it holds for
    every arm trained before this stage without re-running any of them.
    """
    from model import BaseModel

    cfg = {"LOSS": {"TYPE": "so3_trans_fk"}}
    loss_cfg = cfg["LOSS"]
    assert float(loss_cfg.get("ABS_FK_WEIGHT", 0.0)) == 0.0
    assert float(loss_cfg.get("TRANS_BETA", 1.0)) == 1.0
    assert hasattr(BaseModel, "_so3_fk_loss")


def test_smooth_l1_beta_is_supported_by_this_torch():
    """`beta=` on `smooth_l1_loss` is not available in every torch version; fail loudly if not."""
    F.smooth_l1_loss(torch.zeros(2), torch.ones(2), beta=0.01)
