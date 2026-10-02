"""U1a readout contracts; no dataset, checkpoint, or training run required.

These tests remove the zero-output initialization when testing dependencies:
otherwise a decoder that reads prev by mistake would also appear state-free.
The full forward_packet / MANO residual45 integration is covered separately.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "model")]

from semkine.lie import so3_exp, so3_log  # noqa: E402
from semkine.u1a_readout import U1aReadout, copy_shared_to_untied  # noqa: E402


def _readout(mode="shared", *, live=False, dtype=torch.float32):
    torch.manual_seed(31)
    readout = U1aReadout(pooled_dim=11, feat_dim=7, evidence_dim=5,
                         hidden=19, type_dim=3, joint_dim=4, mode=mode).to(dtype=dtype)
    if live:
        # Nonzero, small, nonsymmetric heads exercise every permitted field.
        heads = [readout.shared_head] if mode == "shared" else readout.node_heads
        with torch.no_grad():
            for head in heads:
                head[-1].weight.normal_(0.0, 0.07)
                head[-1].bias.normal_(0.0, 0.02)
    return readout


def _inputs(batch=4, *, dtype=torch.float32, grad=False):
    g = torch.Generator().manual_seed(71)
    shapes = ((batch, 11), (batch, 7), (batch, 16, 5), (batch, 51))
    return tuple(torch.randn(s, generator=g, dtype=dtype).requires_grad_(grad) for s in shapes)


def _manual_copy(shared, untied):
    # An independent equality control, not the implementation's copy helper.
    untied.type_embed.load_state_dict(shared.type_embed.state_dict())
    untied.joint_embed.load_state_dict(shared.joint_embed.state_dict())
    for head in untied.node_heads:
        head.load_state_dict(shared.shared_head.state_dict())


@pytest.mark.parametrize("mode", ["shared", "untied"])
def test_initialization_has_seventeen_zero_raw_measurements(mode):
    readout = _readout(mode)
    out = readout(*_inputs())
    assert tuple(o.shape for o in out) == ((4, 3), (4, 3), (4, 45))
    assert all(torch.count_nonzero(o) == 0 for o in out)
    # Zero finger residual is the label coordinate; the decoder adds no MANO mean.
    assert torch.equal(out[2], torch.zeros(4, 45))


def test_shared_and_identical_untied_heads_have_equal_outputs_and_input_jacobians():
    shared = _readout("shared", live=True, dtype=torch.float64)
    untied = _readout("untied", live=True, dtype=torch.float64)
    _manual_copy(shared, untied)
    a, b = _inputs(dtype=torch.float64, grad=True), _inputs(dtype=torch.float64, grad=True)
    ya, yb = shared(*a), untied(*b)
    for u, v in zip(ya, yb):
        torch.testing.assert_close(u, v, atol=1e-12, rtol=1e-12)
    # Unequal coordinate weights prevent cancellation between the 17 slots.
    weights = torch.linspace(0.1, 1.3, 51, dtype=torch.float64)
    loss_a = (torch.cat(ya, -1) * weights).sum()
    loss_b = (torch.cat(yb, -1) * weights).sum()
    ga = torch.autograd.grad(loss_a, a, retain_graph=True)
    gb = torch.autograd.grad(loss_b, b, retain_graph=True)
    for u, v in zip(ga, gb):
        torch.testing.assert_close(u, v, atol=1e-12, rtol=1e-12)
        assert torch.isfinite(u).all() and u.abs().sum() > 0
    # Tying weights must sum the slot gradients, rather than drop or average them.
    shared_parameters = tuple(shared.parameters())
    untied_parameters = tuple(untied.parameters())
    grads_a = dict(zip(dict(shared.named_parameters()), torch.autograd.grad(loss_a, shared_parameters)))
    grads_b = dict(zip(dict(untied.named_parameters()), torch.autograd.grad(loss_b, untied_parameters)))
    for identity in ("type_embed.weight", "joint_embed.weight"):
        torch.testing.assert_close(grads_a[identity], grads_b[identity], atol=1e-12, rtol=1e-12)
    for name in dict(shared.shared_head.named_parameters()):
        summed = sum(grads_b[f"node_heads.{slot}.{name}"] for slot in range(17))
        torch.testing.assert_close(grads_a[f"shared_head.{name}"], summed, atol=1e-12, rtol=1e-12)


def test_copy_helper_matches_the_independent_copy_and_keeps_heads_untied():
    shared = _readout("shared", live=True)
    expected, actual = _readout("untied"), _readout("untied")
    _manual_copy(shared, expected)
    copy_shared_to_untied(shared, actual)
    for key, value in expected.state_dict().items():
        assert torch.equal(value, actual.state_dict()[key]), key
    x = _inputs()
    before = actual(*x)
    with torch.no_grad():
        actual.node_heads[7][-1].bias[0] += 0.25
    after = actual(*x)
    assert torch.equal(before[0], after[0]) and torch.equal(before[1], after[1])
    difference = after[2] - before[2]
    expected_difference = torch.zeros_like(difference)
    expected_difference[:, 3 * (7 - 2)] = 0.25
    torch.testing.assert_close(difference, expected_difference, atol=1e-7, rtol=0)


@pytest.mark.parametrize("mode", ["shared", "untied"])
@pytest.mark.parametrize("field", [1, 2, 3], ids=["feat", "evidence", "prev"])
def test_root_and_every_finger_ignore_state_conditioned_fields_even_with_nonfinite_values(mode, field):
    readout = _readout(mode, live=True)
    x = _inputs()
    reference = readout(*x)
    altered = list(x)
    altered[field] = x[field] * -9.0 + 23.0
    finite = readout(*altered)
    assert torch.equal(reference[1], finite[1]) and torch.equal(reference[2], finite[2])
    assert not torch.allclose(reference[0], finite[0]), "translation lost its permitted field"
    altered[field] = torch.full_like(x[field], float("nan"))
    nonfinite = readout(*altered)
    assert torch.equal(reference[1], nonfinite[1]) and torch.equal(reference[2], nonfinite[2])
    assert torch.isfinite(nonfinite[1]).all() and torch.isfinite(nonfinite[2]).all()


@pytest.mark.parametrize("mode", ["shared", "untied"])
def test_each_absolute_measurement_responds_to_pooled_events_and_has_no_state_gradient(mode):
    readout = _readout(mode, live=True)
    x = _inputs(grad=True)
    _, root, fingers = readout(*x)
    for measurement in (root, *fingers.split(3, dim=-1)):
        gradients = torch.autograd.grad(measurement.square().sum(), x, retain_graph=True)
        assert torch.isfinite(gradients[0]).all() and gradients[0].abs().sum() > 0
        assert all(torch.count_nonzero(g) == 0 for g in gradients[1:])
    changed = list(x)
    changed[0] = x[0].detach().flip(-1) + torch.arange(11) * 0.3
    _, root2, fingers2 = readout(*changed)
    assert not torch.allclose(root, root2)
    assert ((fingers - fingers2).reshape(4, 15, 3).abs().sum((0, 2)) > 1e-5).all()


@pytest.mark.parametrize("mode", ["shared", "untied"])
def test_translation_reads_full_prev_and_no_raw_pool_bypass(mode):
    readout = _readout(mode, live=True)
    x = _inputs(grad=True)
    t, _, _ = readout(*x)
    pool_grad, feat_grad, evidence_grad, prev_grad = torch.autograd.grad(t.square().sum(), x)
    assert torch.count_nonzero(pool_grad) == 0
    assert feat_grad.abs().sum() > 0 and evidence_grad.abs().sum() > 0
    # All 51 prior coordinates are allowed, not only the previous translation.
    assert (prev_grad.abs().sum(0) > 1e-8).all()
    changed = list(x)
    changed[0] = torch.full_like(x[0], float("nan"))
    assert torch.equal(t, readout(*changed)[0])


@pytest.mark.parametrize("mode", ["shared", "untied"])
def test_packet_batch_does_not_normalize_or_message_across_rows(mode):
    readout = _readout(mode, live=True).train()
    x = _inputs()
    batch = readout(*x)
    for row in range(4):
        alone = readout(*(field[row:row + 1] for field in x))
        for u, v in zip(batch, alone):
            torch.testing.assert_close(u[row:row + 1], v, atol=2e-7, rtol=1e-6)
    changed = [field.clone() for field in x]
    for field in changed:
        field[1:] = field[1:] * 100.0 + 800.0
    changed_out = readout(*changed)
    for u, v in zip(batch, changed_out):
        assert torch.equal(u[0], v[0]), "another packet changed this packet's readout"


def test_slot_order_is_translation_root_then_fifteen_ordered_residual_joints():
    readout = _readout("untied")
    with torch.no_grad():
        for slot, head in enumerate(readout.node_heads):
            head[-1].bias.copy_(torch.tensor([slot, slot + 0.1, -slot - 0.2]))
    t, root, fingers = readout(*_inputs(batch=1))
    torch.testing.assert_close(t, torch.tensor([[0.0, 0.1, -0.2]]))
    torch.testing.assert_close(root, torch.tensor([[1.0, 1.1, -1.2]]))
    expected = torch.tensor([[s, s + 0.1, -s - 0.2] for s in range(2, 17)])
    torch.testing.assert_close(fingers.reshape(15, 3), expected)


def _s38_filter_harness():
    """Use the real S38 methods without allocating an encoder or MANO layer."""
    from model import MNISTModel
    from semkine.anchored import _pq

    class Harness(nn.Module):
        _abs_root = MNISTModel._abs_root
        _abs_fingers = MNISTModel._abs_fingers

        def __init__(self):
            super().__init__()
            ref = torch.tensor([1.568033, 0.992783, 1.368705])
            self.register_buffer("root_ref_R", so3_exp(ref))
            self.root_ref_q = _pq(ref.tolist())
            self.root_filter_gain = 0.5
            self.finger_filter_gain = 0.5

    return Harness()


@pytest.mark.parametrize("mode", ["shared", "untied"])
def test_raw_readout_composes_left_of_root_ref_and_keeps_s38_filters(mode):
    readout = _readout(mode, live=True)
    x = _inputs()
    _, raw_root, raw_fingers = readout(*x)
    prev = x[3] * 0.15
    counts = torch.tensor([80, 30, 0, 60])
    live = counts > 0
    harness = _s38_filter_harness().train()
    rotation = harness._abs_root(raw_root, prev, counts, 170)
    finger = harness._abs_fingers(raw_fingers, prev, counts, 170)
    measurement_R = so3_exp(raw_root) @ harness.root_ref_R
    torch.testing.assert_close(so3_exp(rotation[live]), measurement_R[live], atol=5e-7, rtol=1e-6)
    # A noncommuting test would detect reversing Exp(raw) and the reference.
    assert (measurement_R[live] - harness.root_ref_R @ so3_exp(raw_root[live])).abs().max() > 1e-3
    assert torch.equal(finger[live], raw_fingers[live])
    assert torch.equal(rotation[~live], prev[~live, 3:6])
    assert torch.equal(finger[~live], prev[~live, 6:])

    harness.eval()
    filtered_root = harness._abs_root(raw_root, prev, counts, 170)
    prior_R = so3_exp(prev[:, 3:6])
    innovation = so3_log(prior_R.transpose(-1, -2) @ measurement_R)
    expected_R = prior_R @ so3_exp(0.5 * innovation)
    torch.testing.assert_close(so3_exp(filtered_root[live]), expected_R[live], atol=5e-7, rtol=1e-6)
    filtered_fingers = harness._abs_fingers(raw_fingers, prev, counts, 170)
    torch.testing.assert_close(filtered_fingers[live], 0.5 * (prev[live, 6:] + raw_fingers[live]))
    assert torch.equal(filtered_root[~live], prev[~live, 3:6])
    assert torch.equal(filtered_fingers[~live], prev[~live, 6:])
    # The batch-one fast path uses the actual host event count.
    assert torch.equal(harness._abs_root(raw_root[:1], prev[:1], counts[:1], 0), prev[:1, 3:6])
    assert torch.equal(harness._abs_fingers(raw_fingers[:1], prev[:1], counts[:1], 0), prev[:1, 6:])


@pytest.mark.parametrize("mode", ["shared", "untied"])
@pytest.mark.parametrize("live_head", [False, True], ids=["zero-init", "nonzero-head"])
def test_training_gradients_through_actual_root_composition_are_finite(mode, live_head):
    readout = _readout(mode, live=live_head)
    x = _inputs(grad=True)
    t, root, fingers = readout(*x)
    harness = _s38_filter_harness().train()
    counts = torch.full((4,), 50)
    rotation = harness._abs_root(root, x[3], counts, 200)
    measured_fingers = harness._abs_fingers(fingers, x[3], counts, 200)
    target_R = so3_exp(torch.tensor([[0.1, -0.3, 0.2]]).expand(4, -1))
    loss = (so3_exp(rotation) - target_R).square().mean()
    loss = loss + (measured_fingers - 0.3).square().mean() + (t + 0.1).square().mean()
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in readout.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    for field in x:
        assert field.grad is not None and torch.isfinite(field.grad).all()
    if live_head:
        assert x[0].grad.abs().sum() > 0
    heads = [readout.shared_head] if mode == "shared" else readout.node_heads
    assert all(head[-1].weight.grad.abs().sum() > 0 for head in heads)
