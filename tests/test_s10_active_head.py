#!/usr/bin/env python3
r"""S10 gates for the per-joint active head.

The history this stage is built against: a zero-initialised additive per-joint head placed alongside
a direct 51D regression was measured to contribute 0.16 mm, i.e. nothing. The diagnosis was that the
direct path absorbs the gradient and the bypass is free to stay idle. So the design requirement is
architectural rather than a matter of loss weighting -- the per-joint decoders have to be the only
way a finger angle can be produced.

That requirement is testable without training, and these gates test it:

* the trunk emits features, not a pose, and there is no parameter path from the trunk to a joint's
  output except through that joint's decoder
* each joint's decoder affects that joint and no other
* zeroing the decoders at inference on the *same* checkpoint freezes the fingers exactly, which is
  what makes the ablation decisive instead of suggestive
* the default configuration is untouched, so every previously trained arm is bit-identical
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))

from model import MNISTModel                                        # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(active: bool, **model_kw):
    cfg = {
        "MODEL": {"POSE_REPR": "mano_full_axis_angle", "OUTPUT_DIM": 51,
                  "PREDICT_DELTA": True, "ACTIVE_HEAD": active, **model_kw},
        "MANO": {"NPZ": str(REPO / "assets/mano_right.npz")},
        "LOSS": {"TYPE": "mse_51d", "LAMBDA_POSE": 60.0, "LAMBDA_T": 30000.0,
                 "LAMBDA_R": 60.0, "NORMALIZER": 12, "LOG10": True},
        "DATA": {"HEIGHT": 180, "WIDTH": 240},
    }
    return MNISTModel(cfg).to(DEV).eval()


@pytest.fixture(scope="module")
def batch():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(4, 180, 240, 2, generator=g).to(DEV)
    prev = (torch.randn(4, 51, generator=g) * 0.1).to(DEV)
    return x, prev


def test_default_config_is_unchanged(batch):
    """`ACTIVE_HEAD` absent must give the legacy architecture, parameter for parameter."""
    m = build(False)
    assert not m.active_head
    assert m.rn.fc.out_features == 51
    assert not hasattr(m, "joint_heads")
    x, prev = batch
    with torch.no_grad():
        out = m(x, prev)
    assert out.shape == (4, 51)


def test_trunk_emits_features_not_a_pose(batch):
    m = build(True, ACTIVE_FEAT_DIM=128)
    assert m.rn.fc.out_features == 128, "the trunk still emits a pose, so a direct path exists"
    assert m.root_head.out_features == 6
    assert len(m.joint_heads) == 15
    x, prev = batch
    with torch.no_grad():
        out = m(x, prev)
    assert out.shape == (4, 51)


@pytest.mark.parametrize("k", list(range(15)))
def test_each_joint_has_exactly_one_pathway(batch, k):
    r"""Perturbing decoder `k`'s parameters must move joint `k`'s output and nothing else.

    This is the property the old additive head lacked, and the reason it could be lazy: with a
    direct 51D regression in parallel, joint `k`'s output had two parents and the network could
    starve one of them. Here the gradient of joint `k` flows through exactly one head.
    """
    m = build(True)
    x, prev = batch
    with torch.no_grad():
        base = m(x, prev)
        for p in m.joint_heads[k].parameters():
            p.add_(0.1)
        after = m(x, prev)
    d = (after - base).abs().amax(0)
    sl = slice(6 + 3 * k, 9 + 3 * k)
    assert float(d[sl].max()) > 1e-4, f"joint {k}'s decoder does not reach its own output"
    others = torch.cat([d[:6 + 3 * k], d[9 + 3 * k:]])
    assert float(others.max()) < 1e-12, (
        f"joint {k}'s decoder leaked into another coordinate by {float(others.max()):.3e}")


def test_root_head_does_not_touch_the_fingers(batch):
    m = build(True)
    x, prev = batch
    with torch.no_grad():
        base = m(x, prev)
        for p in m.root_head.parameters():
            p.add_(0.1)
        after = m(x, prev)
    d = (after - base).abs().amax(0)
    assert float(d[:6].max()) > 1e-4
    assert float(d[6:].max()) < 1e-12, "the root head writes finger angles"


def test_ablation_freezes_the_fingers_exactly(batch):
    r"""The same-checkpoint non-laziness check, and what makes it decisive.

    With `PREDICT_DELTA` the output is `prev + head`, so silencing the decoders must return the
    previous finger angles *bit-exactly* while leaving the root free to move. Anything less than
    exact would mean some other path is contributing, and the ablation would no longer isolate the
    head's contribution.
    """
    m = build(True)
    x, prev = batch
    with torch.no_grad():
        live = m(x, prev)
        m.ablate_joint_heads = True
        dead = m(x, prev)
    assert torch.equal(dead[:, 6:], prev[:, 6:]), "the ablated fingers are not exactly the prior"
    assert torch.equal(dead[:, :6], live[:, :6]), "the ablation changed the root as well"
    assert float((live[:, 6:] - prev[:, 6:]).abs().max()) > 1e-4, \
        "the live head does not move the fingers, so the ablation tests nothing"


def test_gradients_reach_every_decoder(batch):
    """No decoder may be structurally starved: all fifteen must receive gradient from the loss."""
    m = build(True).train()
    x, prev = batch
    out = m(x, prev)
    out.pow(2).mean().backward()
    for k, head in enumerate(m.joint_heads):
        g = head[0].weight.grad
        assert g is not None and float(g.abs().max()) > 0, f"decoder {k} received no gradient"


def test_unknown_model_keys_are_still_rejected():
    """The config guard must keep working, or a mistyped ablation would silently do nothing."""
    with pytest.raises(ValueError):
        build(True, ACTIVE_HEADS=True)
