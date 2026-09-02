"""The retention penalty must be measurable, live, and matched to the probe it comes from.

This project has already trained two arms to completion that were silent copies of the arm they were
meant to differ from, because config keys reached nothing (`docs/debug_e55b_unroll_20260825.md`). A
penalty whose weight is read but whose gradient never arrives is the same failure with a different
name, so each of these asserts one link in the chain rather than the end-to-end accuracy claim:

* the key is whitelisted and lands on the attribute;
* the penalty is the probe's `gain_rand` squared, verified against an analytic case;
* it contributes gradient to the parameters;
* at weight zero it costs nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from model import MNISTModel                                      # noqa: E402


H, W = 180, 240
MANO_NPZ = REPO / "assets/mano_right.npz"

pytestmark = pytest.mark.skipif(not MANO_NPZ.exists(), reason="mano_right.npz not available")


def _cfg(w=0.0, scale=1.0, **track):
    return {
        "MODEL": {"POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51,
                  "PREDICT_DELTA": True, "PREVPOS_EMBED": True,
                  "PREV_RENDER": True, "ZERO_EVENT_GATE": True,
                  "RENDER_H": H, "RENDER_W": W, "RENDER_SCALE": 0.375},
        "DATA": {"HEIGHT": H, "WIDTH": W},
        "MANO": {"NPZ": str(MANO_NPZ)},
        "TRACK": {"PREV_NOISE_T": 0.01, "PREV_NOISE_R": 0.05, "PREV_NOISE_POSE": 0.05,
                  "GAIN_REG_W": w, "GAIN_REG_SCALE": scale, **track},
        "LOSS": {"TYPE": "mse_51d", "LAMBDA_POSE": 1.0, "LAMBDA_T": 1.0, "LAMBDA_R": 1.0,
                 "NORMALIZER": 51, "LOG10": False},
        "TRAIN": {"LR": 1e-3, "WARMUP_STEPS": 0},
    }


def test_key_is_whitelisted_and_read():
    m = MNISTModel(_cfg(w=0.25, scale=2.0))
    assert m.gain_reg_w == 0.25 and m.gain_reg_scale == 2.0
    # An unknown TRACK key must still be refused, so the whitelist has not been widened by accident.
    bad = _cfg()
    bad["TRACK"]["GAIN_REG_TYPO"] = 1.0
    with pytest.raises(ValueError, match="unknown TRACK keys"):
        MNISTModel(bad)


def test_delta_uses_the_curriculum_scales():
    m = MNISTModel(_cfg(w=0.1))
    ref = torch.zeros(4096, 51)
    d = m._sample_cond_delta(ref)
    # Per-block standard deviations must match PREV_NOISE_*; a single scale for all 51 entries would
    # make the penalty measure a direction distribution the tracker never meets.
    assert d[:, 0:3].std().item() == pytest.approx(0.01, rel=0.15)
    assert d[:, 3:6].std().item() == pytest.approx(0.05, rel=0.15)
    assert d[:, 6:51].std().item() == pytest.approx(0.05, rel=0.15)
    # And the scale knob has to actually scale.
    assert MNISTModel(_cfg(w=0.1, scale=3.0))._sample_cond_delta(ref)[:, 0:3].std().item() \
        == pytest.approx(0.03, rel=0.15)


def test_penalty_is_measured_in_root_aligned_joint_space():
    """The metric is the whole term, so pin it rather than the plumbing.

    A parameter-space ratio and a root-aligned-joint-space ratio are different objectives, and the
    first one measurably drove joint-space retention the wrong way. Here `f` returns the perturbed
    conditioning state unchanged, so the numerator and denominator are the *same* joint-space
    displacement and the retention must be exactly 1 -- which is true in joint space but not in the
    51-D parameter space, so this test fails if the metric regresses.
    """
    m = MNISTModel(_cfg(w=1.0))
    prev = torch.randn(8, 51) * 0.05
    captured = {}

    def fwd(x, prevpos, **kw):
        captured["prevpos"] = prevpos
        return prevpos

    m.forward = fwd                                               # type: ignore[assignment]
    packed = (torch.zeros(8, 8, 8, 2), prev, torch.zeros(8, 51), None, None)
    assert m._gain_penalty(packed, prev).item() == pytest.approx(1.0, rel=1e-4)
    # And the perturbation really was applied, so the 1.0 is not a degenerate 0/0.
    assert not torch.allclose(captured["prevpos"], prev)


def test_penalty_ignores_pure_global_translation():
    """Root alignment discards global translation, matching the probe. A translation-only output
    change must therefore contribute no retention, whereas a parameter-space ratio would score it."""
    m = MNISTModel(_cfg(w=1.0))
    prev = torch.randn(8, 51) * 0.05
    shift = torch.zeros(8, 51)
    shift[:, 0:3] = 0.02                                          # 20 mm of pure global translation

    m.forward = lambda x, prevpos, **kw: prevpos + shift          # type: ignore[assignment]
    packed = (torch.zeros(8, 8, 8, 2), prev, torch.zeros(8, 51), None, None)
    # f(x+d) - f(x) = d in joint space once the constant shift is root-aligned away, so still 1.0.
    assert m._gain_penalty(packed, prev + shift).item() == pytest.approx(1.0, rel=1e-4)


def test_penalty_requires_the_mano_forward_path():
    bad = _cfg(w=1.0)
    bad["MODEL"]["PREV_RENDER"] = False
    m = MNISTModel(bad)
    prev = torch.randn(2, 51) * 0.05
    with pytest.raises(ValueError, match="PREV_RENDER"):
        m._gain_penalty((torch.zeros(2, 8, 8, 2), prev, torch.zeros(2, 51), None, None), prev)


def test_penalty_reaches_the_parameters_and_is_free_when_off():
    m = MNISTModel(_cfg(w=1.0))
    # The render path concatenates a RENDER_H x RENDER_W image onto the LNES, so this has to be
    # full size; the metric tests above stub `forward` and can stay small. The events must also be
    # non-zero, or ZERO_EVENT_GATE takes the strict identity path and there is nothing to grade.
    x = torch.rand(2, H, W, 2)               # LNES is (B, H, W, 2)
    prev = torch.randn(2, 51) * 0.1
    packed = (x, prev, torch.zeros(2, 51), None, None)
    pred = m(x, prev)
    pen = m._gain_penalty(packed, pred)
    assert torch.isfinite(pen) and pen.requires_grad
    m.zero_grad()
    pen.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert grads, "the retention penalty produced no gradient anywhere"

    off = MNISTModel(_cfg(w=0.0))
    assert off.gain_reg_w == 0.0


def test_penalty_forward_leaves_batchnorm_statistics_alone():
    """The second forward must not write to the 20 BatchNorm buffers.

    It is fed a deliberately corrupted conditioning state, so letting it update the running
    statistics makes inference normalise with numbers that are part real and part perturbation.
    Measured cost when it did: recursive RA 16.64 -> 18.8-19.7 mm at every lambda, worst at the
    smallest lambda, because the damage came from the extra forward and not the penalty weight.
    """
    m = MNISTModel(_cfg(w=1.0))
    m.train()
    x, prev = torch.rand(2, H, W, 2), torch.randn(2, 51) * 0.1
    pred = m(x, prev)

    def checksum():
        return m._bn_checksum()

    before = checksum()
    m._gain_penalty((x, prev, torch.zeros(2, 51), None, None), pred)
    assert checksum() == before, "the penalty forward moved the BatchNorm running statistics"
    # The guard must be scoped: an ordinary forward still has to update them.
    m(x, prev)
    assert checksum() != before, "the guard leaked and disabled BatchNorm tracking entirely"


def test_zero_event_packets_stay_an_identity_update():
    """The penalty must not open a path that updates the state on no evidence.

    With ZERO_EVENT_GATE the output is exactly the conditioning state when there are no events, so
    the perturbed and unperturbed branches differ by exactly the perturbation and the retention is
    1 by construction -- and, more importantly, no gradient exists to push it anywhere else.
    """
    m = MNISTModel(_cfg(w=1.0))
    prev = torch.randn(2, 51) * 0.1
    x = torch.zeros(2, H, W, 2)
    pred = m(x, prev)
    assert torch.allclose(pred, prev, atol=1e-6)
    pen = m._gain_penalty((x, prev, torch.zeros(2, 51), None, None), pred)
    assert pen.item() == pytest.approx(1.0, rel=1e-4)
