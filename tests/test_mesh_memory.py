"""Contracts for explicit, event-driven vertex memory.

These checks concern state retention and trainability, not reconstruction accuracy.
The memory receives an observation mask explicitly: a zero-valued measurement is
different from a missing measurement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semkine.mesh_memory import EventDrivenVertexMemory  # noqa: E402


VERTICES = 5
HIDDEN = 6


@pytest.fixture
def memory():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        return EventDrivenVertexMemory(HIDDEN, n_vertices=VERTICES)


def _inputs(batch_size=2):
    generator = torch.Generator().manual_seed(29)
    features = torch.randn(batch_size, VERTICES, HIDDEN, generator=generator)
    previous = torch.randn(batch_size, VERTICES, HIDDEN, generator=generator)
    observed = torch.zeros(batch_size, VERTICES, dtype=torch.bool)
    observed[:, 1] = True
    observed[-1, 3] = True
    return features, observed, previous


def test_unobserved_vertices_keep_nonzero_history_without_mutating_inputs(memory):
    features, observed, previous = _inputs()
    original_features, original_previous = features.clone(), previous.clone()

    next_state, innovation = memory(features, observed, previous)

    assert next_state.shape == innovation.shape == previous.shape
    assert torch.equal(next_state[~observed], previous[~observed])
    assert torch.count_nonzero(innovation[~observed]) == 0
    torch.testing.assert_close(innovation[observed],
                               next_state[observed] - previous[observed])
    assert not torch.equal(next_state[observed], previous[observed])
    assert torch.equal(features, original_features)
    assert torch.equal(previous, original_previous)


def test_empty_observation_packet_holds_history_and_emits_no_innovation(memory):
    features, observed, previous = _inputs()
    observed.zero_()

    next_state, innovation = memory(features, observed, previous)

    assert torch.equal(next_state, previous)
    assert torch.count_nonzero(innovation) == 0
    assert torch.isfinite(next_state).all() and torch.isfinite(innovation).all()


def test_missing_previous_state_is_zero_initialized_and_not_cached(memory):
    features, observed, previous = _inputs()
    baseline_state, baseline_innovation = memory(features, observed, None)
    zero_state, zero_innovation = memory(features, observed, torch.zeros_like(previous))
    torch.testing.assert_close(baseline_state, zero_state)
    torch.testing.assert_close(baseline_innovation, zero_innovation)
    assert torch.count_nonzero(baseline_state[~observed]) == 0

    # An unrelated stream must not become this call's implicit history.
    memory(features * 3, torch.ones_like(observed), previous + 7)
    repeated_state, repeated_innovation = memory(features, observed, None)
    assert torch.equal(repeated_state, baseline_state)
    assert torch.equal(repeated_innovation, baseline_innovation)


def test_batch_members_and_streams_are_independent(memory):
    features, observed, previous = _inputs()
    combined = memory(features, observed, previous)
    for batch_index in range(features.shape[0]):
        independent = memory(features[batch_index:batch_index + 1],
                             observed[batch_index:batch_index + 1],
                             previous[batch_index:batch_index + 1])
        for together, separate in zip(combined, independent):
            torch.testing.assert_close(together[batch_index:batch_index + 1], separate,
                                       rtol=1e-5, atol=1e-6)

    changed_features, changed_previous = features.clone(), previous.clone()
    changed_features[1] += 100
    changed_previous[1] -= 100
    changed = memory(changed_features, observed, changed_previous)
    for original, perturbed in zip(combined, changed):
        assert torch.equal(original[0], perturbed[0])


def test_two_step_loss_reaches_both_observed_packets_and_memory_parameters(memory):
    first_features, observed, previous = _inputs(batch_size=1)
    first_features.requires_grad_()
    second_features = (first_features.detach() * 0.4 + 0.3).requires_grad_()
    previous.requires_grad_()

    first_state, _ = memory(first_features, observed, previous)
    _, second_innovation = memory(second_features, observed, first_state)
    second_innovation.square().sum().backward()

    for features in (first_features, second_features):
        assert features.grad is not None and torch.isfinite(features.grad).all()
        assert torch.count_nonzero(features.grad[observed]) > 0
        assert torch.count_nonzero(features.grad[~observed]) == 0
    assert previous.grad is not None and torch.isfinite(previous.grad).all()
    assert torch.count_nonzero(previous.grad[observed]) > 0
    parameters = [p for p in memory.parameters() if p.requires_grad]
    assert parameters
    for parameter in parameters:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_bf16_autocast_preserves_fp32_history_exactly(memory):
    features, observed, previous = _inputs()
    # These values are deliberately not representable in bfloat16.
    previous = previous * 0.00314159 + 0.1234567
    assert not torch.equal(previous, previous.to(torch.bfloat16).float())

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        next_state, innovation = memory(features, observed, previous)
        empty_state, empty_innovation = memory(features, torch.zeros_like(observed), previous)

    assert next_state.dtype == previous.dtype == torch.float32
    assert torch.equal(next_state[~observed], previous[~observed])
    assert torch.count_nonzero(innovation[~observed]) == 0
    assert torch.equal(empty_state, previous)
    assert torch.count_nonzero(empty_innovation) == 0


def test_zero_features_with_observation_mask_are_still_a_measurement(memory):
    features = torch.zeros(1, VERTICES, HIDDEN, requires_grad=True)
    previous = torch.full_like(features, 0.7)
    observed = torch.zeros(1, VERTICES, dtype=torch.bool)
    observed[:, 2] = True

    next_state, innovation = memory(features, observed, previous)
    assert not torch.equal(next_state[observed], previous[observed])
    assert torch.equal(next_state[~observed], previous[~observed])
    assert torch.count_nonzero(innovation[~observed]) == 0
    innovation.square().sum().backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad[observed]) > 0
    assert torch.count_nonzero(features.grad[~observed]) == 0
