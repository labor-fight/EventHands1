"""Stateful full-mesh tracking contracts, independent of pose accuracy."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from semkine.event_guided_mesh import (  # noqa: E402
    EventGuidedMeshEncoder,
    MeshVertexState,
    build_event_guided_graph,
)


@pytest.fixture(autouse=True, scope="module")
def _threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


def _fixture(batch=2, weights=None, layers=2):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(71)
        encoder = EventGuidedMeshEncoder(
            torch.eye(6) if weights is None else weights, hidden=8, n_layers=layers,
            k=3, candidates=5, memory=True, geometry_neighbors=1)
        vertices = torch.randn(batch, 6, 3) * 0.01
        observations = torch.rand(batch, 6, 6)
        observations[..., 0] = 0
        observations[:, 1, 0] = 1
        if batch > 1:
            observations[1, 4, 0] = 1
        state = MeshVertexState(torch.randn(batch, 6, 8),
                                torch.ones(batch, 6, dtype=torch.bool))
    return encoder, observations, vertices, state


def test_only_current_observed_vertices_commit_memory_and_open_joint_heads():
    encoder, observations, vertices, state = _fixture()
    original = state.hidden.clone()
    evidence, joint_support, next_state = encoder.forward_step(observations, vertices, state)
    observed = observations[..., 0] > 0
    assert encoder.last_support.all(), "old history remains available as graph context"
    assert torch.equal(encoder.last_observed, observed)
    assert torch.equal(joint_support, observed), "identity LBS must not activate old-only joints"
    assert torch.equal(next_state.hidden[~observed], state.hidden[~observed])
    assert torch.count_nonzero(encoder.last_innovation[~observed]) == 0
    assert torch.count_nonzero(evidence[~observed]) == 0
    assert not torch.equal(next_state.hidden[observed], state.hidden[observed])
    assert torch.equal(state.hidden, original)
    assert next_state.seen.all()


def test_empty_packet_holds_nonzero_history_exactly_under_autocast_with_all_grad_paths():
    encoder, observations, vertices, state = _fixture()
    observations.zero_()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        evidence, joint_support, next_state = encoder.forward_step(observations, vertices, state)
    assert next_state.hidden.dtype == state.hidden.dtype == torch.float32
    assert torch.equal(next_state.hidden, state.hidden)
    assert torch.equal(next_state.seen, state.seen)
    assert not joint_support.any()
    assert torch.count_nonzero(evidence) == 0
    assert torch.count_nonzero(encoder.last_innovation) == 0
    evidence.sum().backward()
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name


def test_lbs_reads_updated_memory_only_at_currently_observed_vertices():
    weights = torch.tensor([[1., 0.], [.75, .25], [.5, .5],
                            [0., 1.], [.3, .7], [.1, .9]])
    encoder, observations, vertices, state = _fixture(weights=weights)
    evidence, support, next_state = encoder.forward_step(observations, vertices, state)
    innovation = torch.where((observations[..., 0] > 0).unsqueeze(-1),
                             next_state.hidden - state.hidden, torch.zeros_like(state.hidden))
    readout = torch.where((observations[..., 0] > 0).unsqueeze(-1),
                         next_state.hidden, torch.zeros_like(state.hidden))
    expected = torch.einsum("vj,bvc->bjc", weights / weights.sum(0), readout)
    torch.testing.assert_close(evidence, expected)
    torch.testing.assert_close(encoder.last_innovation, innovation)
    torch.testing.assert_close(encoder.last_readout_features, readout)
    assert support.all()
    historical = torch.einsum("vj,bvc->bjc", weights / weights.sum(0), next_state.hidden)
    assert not torch.allclose(evidence, historical)


def test_observed_nonzero_gru_fixed_point_remains_readable_without_innovation():
    encoder, observations, vertices, _ = _fixture(batch=1)
    # A valid, exact GRU fixed point: all gates have zero weights; the candidate
    # has constant value tanh(b_n), and the old hidden state equals that value.
    # This verifies readout semantics without fitting any parameters to data.
    with torch.no_grad():
        for parameter in encoder.memory.parameters():
            parameter.zero_()
        encoder.memory.cell.bias_ih[2 * encoder.hidden:].fill_(0.5)
    fixed_value = torch.tanh(encoder.memory.cell.bias_ih[2 * encoder.hidden:].detach())
    state = MeshVertexState(fixed_value.view(1, 1, -1).expand(1, 6, -1).clone(),
                            torch.ones(1, 6, dtype=torch.bool))
    evidence, support, next_state = encoder.forward_step(observations, vertices, state)
    observed = observations[..., 0] > 0
    assert torch.equal(next_state.hidden, state.hidden)
    assert torch.count_nonzero(encoder.last_innovation) == 0
    assert torch.equal(support, observed)
    assert torch.equal(evidence[observed], state.hidden[observed])
    assert torch.count_nonzero(evidence[observed]) > 0
    assert torch.count_nonzero(evidence[~observed]) == 0
    empty = observations.clone()
    empty[..., 0] = 0
    empty_evidence, empty_support, held = encoder.forward_step(empty, vertices, next_state)
    assert torch.equal(held.hidden, state.hidden)
    assert torch.count_nonzero(empty_evidence) == 0
    assert not empty_support.any()


def test_cold_start_seen_marks_only_actual_observations_and_no_history_is_cached():
    encoder, observations, vertices, state = _fixture()
    result = encoder.forward_step(observations, vertices)
    assert torch.equal(result[2].seen, observations[..., 0] > 0)
    assert torch.count_nonzero(result[2].hidden[~result[2].seen]) == 0
    encoder.forward_step(observations * 2, vertices, state)
    repeated = encoder.forward_step(observations, vertices)
    assert torch.equal(result[0], repeated[0])
    assert torch.equal(result[2].hidden, repeated[2].hidden)
    ordinary = encoder(observations, vertices)
    assert torch.equal(ordinary[0], result[0])
    assert torch.equal(ordinary[1], result[1])


def test_graph_context_does_not_mark_receivers_as_seen_or_update_unobserved_memory():
    encoder, observations, vertices, state = _fixture(batch=1)
    previous_seen = torch.zeros_like(state.seen)
    previous_seen[:, 0] = True
    state = MeshVertexState(state.hidden * previous_seen.unsqueeze(-1), previous_seen)
    _, _, next_state = encoder.forward_step(observations, vertices, state)
    expected_seen = previous_seen | (observations[..., 0] > 0)
    assert torch.equal(next_state.seen, expected_seen)
    assert encoder.last_support.sum() > expected_seen.sum()
    assert torch.count_nonzero(next_state.hidden[~expected_seen]) == 0


def test_history_can_inform_a_newly_observed_neighbour_without_moving_stored_vertices():
    encoder, observations, vertices, _ = _fixture(batch=1, layers=1)
    # Fully connected graph ensures vertex 1 can read vertex 0's local history.
    encoder.k = 5
    previous = torch.zeros(1, 6, 8)
    seen = torch.zeros(1, 6, dtype=torch.bool)
    seen[:, 0] = True
    empty_state = MeshVertexState(previous.clone(), seen)
    previous[:, 0] = 4
    populated_state = MeshVertexState(previous, seen)
    baseline = encoder.forward_step(observations, vertices, empty_state)
    changed = encoder.forward_step(observations, vertices, populated_state)
    assert not torch.allclose(baseline[2].hidden[:, 1], changed[2].hidden[:, 1])
    assert torch.equal(changed[2].hidden[:, 0], populated_state.hidden[:, 0])


def test_two_step_gradient_reaches_both_packets_and_gru_parameters_and_detach_is_explicit():
    encoder, observations, vertices, _ = _fixture(batch=1)
    first = observations.clone().requires_grad_()
    second = observations.clone()
    second[..., 1:] *= .4
    second.requires_grad_()
    _, _, state = encoder.forward_step(first, vertices)
    evidence, _, next_state = encoder.forward_step(second, vertices, state)
    evidence.square().sum().backward()
    for inputs in (first, second):
        assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
        assert inputs.grad.abs().sum() > 0
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    detached = next_state.detach()
    assert not detached.hidden.requires_grad
    assert torch.equal(detached.hidden, next_state.hidden)
    assert torch.equal(detached.seen, next_state.seen)


def test_batch_members_have_independent_memory_and_seen_maps():
    encoder, observations, vertices, state = _fixture()
    combined = encoder.forward_step(observations, vertices, state)
    for b in range(2):
        independent = encoder.forward_step(
            observations[b:b + 1], vertices[b:b + 1],
            MeshVertexState(state.hidden[b:b + 1], state.seen[b:b + 1]))
        torch.testing.assert_close(combined[0][b:b + 1], independent[0])
        torch.testing.assert_close(combined[2].hidden[b:b + 1], independent[2].hidden)
        assert torch.equal(combined[2].seen[b:b + 1], independent[2].seen)


def test_memory_state_validates_shape_and_seen_dtype():
    encoder, observations, vertices, state = _fixture()
    with pytest.raises(ValueError, match="hidden"):
        encoder.forward_step(observations, vertices, MeshVertexState(state.hidden[:, :-1], state.seen))
    with pytest.raises(ValueError, match="seen"):
        encoder.forward_step(observations, vertices, MeshVertexState(state.hidden, state.seen.float()))
    encoder_without_memory = EventGuidedMeshEncoder(torch.eye(6), hidden=8)
    with pytest.raises(RuntimeError, match="memory=True"):
        encoder_without_memory.forward_step(observations, vertices)


def test_mixed_graph_preserves_geometric_neighbours_and_unique_observed_edges():
    vertices = torch.zeros(1, 8, 3)
    vertices[0, :, 0] = torch.arange(8) * .001
    observations = torch.zeros(1, 8, 6)
    observations[0, [0, 5, 6], 0] = 1
    indices, _, mask = build_event_guided_graph(
        vertices, observations, k=4, candidates=7, geometry_neighbors=2)
    # Query 0 keeps nearest 1 and 2, then reads the two observed distant sources.
    assert indices[0, 0].tolist() == [1, 2, 5, 6]
    geometry = build_event_guided_graph(
        vertices, observations, k=4, candidates=7, geometry_neighbors=2, event_weight=0)[0]
    assert torch.equal(indices[..., :2], geometry[..., :2])
    assert mask.all()
    for query in range(8):
        entries = indices[0, query].tolist()
        assert len(set(entries)) == 4 and query not in entries


def test_missing_query_does_not_prefer_zero_measurements_over_observed_sources():
    # This is the debug case where zero-observation similarity previously
    # isolated the only event source from all other vertices.
    vertices = torch.zeros(1, 34, 3)
    vertices[0, :, 0] = torch.arange(34) * .001
    observations = torch.zeros(1, 34, 6)
    observations[0, 0, 0] = 1
    indices = build_event_guided_graph(
        vertices, observations, k=8, candidates=32, geometry_neighbors=4)[0]
    assert 0 in indices[0, 1].tolist()
    assert 0 in indices[0, 8].tolist(), "an observed source fills a dynamic edge"
    # Missing-query feature garbage must not change affinity scoring; only its
    # zero count has semantics in this case.
    changed = observations.clone()
    changed[0, 8, 1:] = 100
    altered = build_event_guided_graph(
        vertices, changed, k=8, candidates=32, geometry_neighbors=4)[0]
    assert torch.equal(indices[0, 8], altered[0, 8])


@pytest.mark.parametrize("vertices_count", [1, 2, 3, 8])
def test_mixed_graph_small_sets_and_empty_observations_fall_back_to_geometry(vertices_count):
    generator = torch.Generator().manual_seed(13)
    vertices = torch.randn(2, vertices_count, 3, generator=generator)
    observations = torch.zeros(2, vertices_count, 6)
    mixed = build_event_guided_graph(
        vertices, observations, k=8, candidates=32, geometry_neighbors=4)
    geometry = build_event_guided_graph(vertices, observations, k=8, candidates=32, event_weight=0)
    for actual, expected in zip(mixed, geometry):
        torch.testing.assert_close(actual, expected)


def test_mixed_graph_zero_event_weight_is_exact_geometry_even_with_observations():
    generator = torch.Generator().manual_seed(15)
    vertices = torch.randn(2, 40, 3, generator=generator) * .01
    observations = torch.rand(2, 40, 6, generator=generator)
    mixed = build_event_guided_graph(
        vertices, observations, k=8, candidates=32, geometry_neighbors=4, event_weight=0)
    geometry = build_event_guided_graph(vertices, observations, k=8, candidates=32, event_weight=0)
    for actual, expected in zip(mixed, geometry):
        torch.testing.assert_close(actual, expected)


def test_dynamic_edges_still_respond_to_current_event_features():
    vertices = torch.zeros(1, 5, 3)
    vertices[0, :, 0] = torch.arange(5) * .001
    observations = torch.zeros(1, 5, 6)
    observations[..., 0] = 1
    first = build_event_guided_graph(
        vertices, observations, k=2, candidates=4, geometry_neighbors=1)[0]
    observations[0, 2, 4] = 10
    changed = build_event_guided_graph(
        vertices, observations, k=2, candidates=4, geometry_neighbors=1)[0]
    assert first[0, 0].tolist() == [1, 2]
    assert changed[0, 0].tolist() == [1, 3]
