"""Contracts for the event-guided, full-778-vertex recurrent MANO graph.

The tests cover event-dependent connectivity, evidence support, fixed LBS pooling,
and the raw-packet model boundary. They do not measure model accuracy or speed.
"""
from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))

from model import MNISTModel  # noqa: E402
from semkine.event_guided_mesh import (  # noqa: E402
    EventGuidedMeshEncoder,
    build_event_guided_graph,
)
from semkine.events import EV_BATCH, EV_P, EventPacketBatch  # noqa: E402
from semkine.mesh_graph import visible_vertices  # noqa: E402


HIDDEN = 16


def _cfg(**overrides):
    model = {
        "POSE_REPR": "mano_full_axis_angle",
        "OUTPUT_DIM": 51,
        "MANO_NCOMPS": 45,
        "PREDICT_DELTA": True,
        "PREVPOS_EMBED": False,
        "PREV_RENDER": False,
        "ZERO_EVENT_GATE": True,
        "RENDER_H": 180,
        "RENDER_W": 240,
        "RENDER_SCALE": 0.375,
        "ENCODER": "event_guided_mesh",
        "MESH_GRAPH_BAND_PX": 16.0,
        "MESH_GRAPH_FRONT_PX": 1.0,
        "MESH_GRAPH_Z_TOL": 0.01,
        "ENCODER_HIDDEN": HIDDEN,
        "ENCODER_LAYERS": 2,
        "ENCODER_K": 4,
        "EGM_CANDIDATES": 8,
        "EGM_EVENT_WEIGHT": 1.0,
        "EGM_GEOMETRY_SCALE": 0.02,
        "ACTIVE_HEAD": True,
        "ACTIVE_HIDDEN": 16,
    }
    model.update(overrides)
    return {
        "MODEL": model,
        "LOSS": {"LAMBDA_POSE": 450.0, "LAMBDA_T": 30000.0,
                 "LAMBDA_R": 60.0, "NORMALIZER": 51, "LOG10": True},
        "TRAIN": {"LR": 1e-3, "WARMUP_STEPS": 0},
        "DATA": {"HEIGHT": 180, "WIDTH": 240},
        "MANO": {"NPZ": str(ROOT / "assets/mano_right.npz")},
    }


@pytest.fixture(scope="module", autouse=True)
def _limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def model():
    with torch.random.fork_rng():
        torch.manual_seed(24)
        result = MNISTModel(_cfg())
    return result.eval()


def _state(batch_size=1):
    prev = torch.zeros(batch_size, 51)
    prev[:, 2] = 0.5
    return prev


def _packet(rows, prev=None):
    prev = _state() if prev is None else prev
    batch_size = prev.shape[0]
    events = torch.tensor(rows, dtype=torch.float32).reshape(-1, 5)
    counts = torch.bincount(events[:, EV_BATCH].long(), minlength=batch_size)
    ptr = torch.cat((torch.zeros(1, dtype=torch.long), counts.cumsum(0)))
    packet = EventPacketBatch(
        events=events,
        ptr=ptr,
        sequence_id=torch.arange(batch_size),
        t_start_us=torch.zeros(batch_size, dtype=torch.long),
        t_end_us=torch.full((batch_size,), 50_000, dtype=torch.long),
        delta_t_s=torch.full((batch_size,), 0.05),
        is_sequence_start=torch.zeros(batch_size, dtype=torch.bool),
        is_sequence_end=torch.zeros(batch_size, dtype=torch.bool),
        target=torch.zeros(batch_size, 51),
        prev_state=prev,
        betas=torch.zeros(batch_size, 10),
        camera_K=None,
    )
    packet.validate()
    return packet


def _events_on_hand(model, prev=None, packet_index=0):
    prev = _state() if prev is None else prev
    with torch.no_grad():
        betas, camera = model._resolve_betas_K(prev, None, None)
        vertices, _ = model._fk(prev, betas)
        uv, z = model._project_verts(vertices, camera)
        visible = visible_vertices(vertices, uv, z, model.mano.f, 180, 240, 1.0, 0.01)
    xy = uv[0]
    inside = ((xy[:, 0] >= 2) & (xy[:, 0] < 238)
              & (xy[:, 1] >= 2) & (xy[:, 1] < 178))
    ids = (visible[0] & inside).nonzero().flatten()
    assert ids.numel() >= 20, "the real MANO fixture must expose enough surface"
    return [
        (packet_index, float(xy[i, 0].round()), float(xy[i, 1].round()),
         0.045 * (n + 1) / (len(ids) + 1), n % 2)
        for n, i in enumerate(ids)
    ]


def test_events_change_neighbor_membership_with_identical_3d_vertices():
    vertices = torch.tensor([[[0.0, 0.0, 0.0], [0.010, 0.0, 0.0],
                              [0.011, 0.0, 0.0], [0.100, 0.0, 0.0]]])
    baseline = torch.zeros(1, 4, 6)
    baseline[..., 0] = 1.0
    changed = baseline.clone()
    changed[0, 1, 4] = 1.0  # Event-time disagreement outweighs the small distance gap.
    idx_a, _, mask_a = build_event_guided_graph(vertices, baseline, k=1, candidates=3)
    idx_b, _, mask_b = build_event_guided_graph(vertices, changed, k=1, candidates=3)
    assert mask_a.all() and mask_b.all()
    assert idx_a[0, 0].tolist() == [1]
    assert idx_b[0, 0].tolist() == [2]
    assert set(idx_a[0, 0].tolist()) != set(idx_b[0, 0].tolist())
    # Disabling event affinity is an actual geometry-only control.
    idx_geometry, _, _ = build_event_guided_graph(
        vertices, changed, k=1, candidates=3, event_weight=0.0)
    assert torch.equal(idx_geometry, idx_a)


def test_graph_keeps_all_778_vertices_without_self_loops_or_cross_batch_edges():
    generator = torch.Generator().manual_seed(6)
    vertices = torch.randn(2, 778, 3, generator=generator) * 0.04
    observations = torch.rand(2, 778, 6, generator=generator)
    vertices.requires_grad_()
    observations.requires_grad_()
    idx, dp, mask = build_event_guided_graph(vertices, observations, k=4, candidates=8)
    assert idx.shape == mask.shape == (2, 778, 4)
    assert dp.shape == (2, 778, 4, 3)
    assert idx.min() >= 0 and idx.max() < 778 and mask.all()
    centers = torch.arange(778).view(1, 778, 1)
    assert torch.all(idx != centers)
    assert not idx.requires_grad and not dp.requires_grad and not mask.requires_grad
    neighbour_vertices = vertices.detach().gather(
        1, idx.reshape(2, -1, 1).expand(-1, -1, 3)).reshape(2, 778, 4, 3)
    torch.testing.assert_close(dp, (neighbour_vertices - vertices.detach().unsqueeze(2)) / 0.02)
    for batch_index in range(2):
        single = build_event_guided_graph(
            vertices[batch_index:batch_index + 1],
            observations[batch_index:batch_index + 1], k=4, candidates=8)
        for combined, separate in zip((idx, dp, mask), single):
            torch.testing.assert_close(combined[batch_index:batch_index + 1], separate)
    # Chunk boundaries must not alter adjacency or relative geometry.
    differently_chunked = build_event_guided_graph(
        vertices, observations, k=4, candidates=8, chunk_size=137)
    for left, right in zip((idx, dp, mask), differently_chunked):
        torch.testing.assert_close(left, right)


@pytest.mark.parametrize("vertex_count", [1, 2, 3])
def test_graph_small_vertex_sets_have_only_valid_unpadded_edges(vertex_count):
    vertices = torch.arange(vertex_count * 3, dtype=torch.float32).view(1, vertex_count, 3)
    obs = torch.zeros(1, vertex_count, 6)
    idx, dp, mask = build_event_guided_graph(vertices, obs, k=8, candidates=32)
    expected_k = max(1, vertex_count - 1)
    assert idx.shape == mask.shape == (1, vertex_count, expected_k)
    assert dp.shape == (1, vertex_count, expected_k, 3)
    assert torch.isfinite(dp).all() and idx.min() >= 0 and idx.max() < vertex_count
    if vertex_count == 1:
        assert not mask.any() and torch.count_nonzero(dp) == 0
    else:
        assert mask.all()
        assert torch.all(idx != torch.arange(vertex_count).view(1, vertex_count, 1))
        for vertex in range(vertex_count):
            assert set(idx[0, vertex].tolist()) == set(range(vertex_count)) - {vertex}


def test_zero_event_observations_cannot_create_features_from_nonzero_geometry():
    torch.manual_seed(1)
    weights = torch.rand(9, 3)
    encoder = EventGuidedMeshEncoder(weights, hidden=8, n_layers=2, k=3, candidates=5)
    # Nonzero biases would manufacture evidence without explicit support gating.
    with torch.no_grad():
        for name, parameter in encoder.named_parameters():
            if name.endswith("bias"):
                parameter.fill_(0.5)
    observations = torch.zeros(2, 9, 6)
    vertices = torch.randn(2, 9, 3) + 2.0
    evidence, joint_support = encoder(observations, vertices)
    assert evidence.shape == (2, 3, 8)
    assert torch.count_nonzero(evidence) == 0
    assert torch.count_nonzero(encoder.last_node_features) == 0
    assert not encoder.last_support.any() and not joint_support.any()


def test_support_propagates_from_observed_senders_only():
    vertices = torch.tensor([[[0.0, 0.0, 0.5], [0.010, 0.0, 0.5],
                              [0.030, 0.0, 0.5], [0.031, 0.0, 0.5]]])
    observations = torch.zeros(1, 4, 6)
    observations[..., 1:] = 0.7  # Count zero still means no event support.
    observations[0, 0, 0] = 1.0
    encoder = EventGuidedMeshEncoder(torch.eye(4), hidden=8, n_layers=1,
                                    k=1, candidates=3, event_weight=0.0)
    evidence, joint_support = encoder(observations, vertices)
    assert encoder.last_support.dtype == torch.bool
    assert encoder.last_support.tolist() == [[True, True, False, False]]
    assert joint_support.bool().tolist() == [[True, True, False, False]]
    assert torch.count_nonzero(evidence[:, 2:]) == 0
    assert torch.count_nonzero(encoder.last_node_features[:, 2:]) == 0


def test_lbs_pool_is_the_fixed_normalized_weighted_mean_without_extra_channels():
    weights = torch.tensor([[1.0, 0.0, 0.0], [0.75, 0.25, 0.0],
                            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    encoder = EventGuidedMeshEncoder(weights, hidden=8, n_layers=0, k=1, candidates=3)
    observations = torch.zeros(1, 4, 6)
    observations[0, 0] = torch.tensor([1.0, 0.2, 0.3, 0.1, 0.2, 1.0])
    observations[0, 1] = torch.tensor([2.0, -0.2, 0.1, 0.2, 0.6, 0.0])
    vertices = torch.tensor([[[0.0, 0.0, 0.5], [0.01, 0.0, 0.5],
                              [0.02, 0.0, 0.5], [0.03, 0.0, 0.5]]])
    evidence, support = encoder(observations, vertices)
    node_features = encoder.last_node_features
    expected = torch.einsum("vj,bvc->bjc", weights, node_features)
    expected /= weights.sum(0).view(1, 3, 1)
    assert evidence.shape == (1, 3, 8)
    torch.testing.assert_close(evidence, expected)
    # Joint 1 includes an unsupported vertex in its fixed denominator.
    torch.testing.assert_close(evidence[0, 1], 0.25 * node_features[0, 1] / 1.25)
    assert support.bool().tolist() == [[True, True, False]]
    assert torch.count_nonzero(evidence[0, 2]) == 0


def test_prev_embedding_is_rejected():
    with pytest.raises(ValueError, match="PREVPOS_EMBED"):
        MNISTModel(_cfg(PREVPOS_EMBED=True))


def test_decoder_has_only_joint_features_and_support_as_inputs(model):
    parameters = inspect.signature(model._decode_event_guided_mesh).parameters
    assert len(parameters) == 2 and tuple(parameters)[1] == "joint_support"
    assert not any("prev" in name for name in parameters)
    assert not hasattr(model, "prev_mlp")
    assert model.conv1 is None and model.rn is None
    assert model.root_head[0].in_features == 16 * HIDDEN
    assert all(head[0].in_features == HIDDEN for head in model.joint_heads)
    evidence = torch.randn(1, 16, HIDDEN, requires_grad=True)
    support = torch.ones(1, 16, dtype=torch.bool)
    delta = model._decode_event_guided_mesh(evidence, support)
    assert delta.shape == (1, 51)
    for joint in range(15):
        gradient = torch.autograd.grad(delta[0, 6 + 3 * joint:9 + 3 * joint].sum(),
                                       evidence, retain_graph=True)[0]
        others = torch.arange(16) != joint + 1
        assert torch.count_nonzero(gradient[0, others]) == 0
        assert gradient[0, joint + 1].abs().sum() > 0
    root_gradient = torch.autograd.grad(delta[0, :6].sum(), evidence)[0]
    assert torch.all(root_gradient.abs().sum(-1) > 0)
    assert torch.count_nonzero(model._decode_event_guided_mesh(
        evidence, torch.zeros_like(support))) == 0


def test_real_mano_packet_backpropagates_through_all_trainable_parameters(model):
    packet = _packet(_events_on_hand(model))
    model.zero_grad(set_to_none=True)
    output = model.forward_packet(packet)
    assert output.shape == (1, 51) and torch.isfinite(output).all()
    assert not torch.equal(output, packet.prev_state)
    assert model.event_encoder.last_node_features.shape == (1, 778, HIDDEN)
    assert model.event_encoder.last_graph[0].shape == (1, 778, 4)
    loss = ((output - packet.prev_state) * torch.linspace(0.5, 1.5, 51)).sum()
    loss.backward()
    missing = [name for name, parameter in model.named_parameters()
               if parameter.requires_grad and parameter.grad is None]
    assert not missing, f"unused trainable parameters: {missing}"
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.requires_grad)
    important_modules = ["event_encoder.obs_embed", "root_head"]
    important_modules += [f"event_encoder.layers.{index}." for index in range(2)]
    important_modules += [f"joint_heads.{index}." for index in range(15)]
    for prefix in important_modules:
        gradients = [parameter.grad.abs().sum() for name, parameter in model.named_parameters()
                     if name.startswith(prefix) and parameter.requires_grad]
        assert gradients and sum(gradients) > 0, f"no learning signal in {prefix}"
    model.zero_grad(set_to_none=True)


def test_empty_packet_in_a_mixed_batch_returns_prev_bitwise(model):
    previous = _state(2)
    previous[1, :2] = torch.tensor([0.015, -0.010])
    previous[1, 6:] = torch.linspace(-0.05, 0.05, 45)
    packet = _packet(_events_on_hand(model, previous[:1]), prev=previous)
    with torch.no_grad():
        output = model.forward_packet(packet)
    assert not torch.equal(output[0], previous[0])
    assert torch.equal(output[1], previous[1])
    assert torch.count_nonzero(model.event_encoder.last_node_features[1]) == 0
    assert not model.event_encoder.last_support[1].any()


def test_autocast_preserves_full_precision_prev_for_empty_and_unrouted_packets(model):
    previous = _state(3)
    previous[1:, 0] = 0.01234567
    previous[1:, 2] = 0.50123456
    previous[1:, 6:] = torch.linspace(-0.037123, 0.043219, 45)
    rows = _events_on_hand(model, previous[:1])
    rows += [(2, 2.0, 2.0, 0.01, 1), (2, 237.0, 177.0, 0.02, 0)]
    packet = _packet(rows, previous)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.forward_packet(packet)
    assert torch.isfinite(output).all()
    assert output.dtype == previous.dtype
    assert torch.equal(output[1:], previous[1:])
    assert not model.event_encoder.last_support[1:].any()


def test_all_empty_and_events_missing_the_mesh_have_no_background_update(model):
    previous = _state(2)
    previous[1, 3:6] = torch.tensor([0.1, -0.1, 0.05])
    with torch.no_grad():
        assert torch.equal(model.forward_packet(_packet([], previous)), previous)
        far = _packet([(0, 2.0, 2.0, 0.01, 1), (0, 237.0, 177.0, 0.02, 0)])
        output = model.forward_packet(far)
    assert torch.equal(output, far.prev_state)
    assert not model.event_encoder.last_support.any()


def test_current_events_and_previous_state_are_used_without_target_leakage(model, monkeypatch):
    packet = _packet(_events_on_hand(model))
    used_states = []
    original_fk = model._fk

    def record_fk(previous, betas):
        used_states.append(previous.detach().clone())
        return original_fk(previous, betas)

    monkeypatch.setattr(model, "_fk", record_fk)
    altered_target = dataclasses.replace(packet, target=torch.full_like(packet.target, 1000.0))
    with torch.no_grad():
        prediction = model.forward_packet(packet)
        changed_target_prediction = model.forward_packet(altered_target)
    assert torch.equal(prediction, changed_target_prediction)
    assert used_states and all(torch.equal(state, packet.prev_state) for state in used_states)
    # Hold geometry fixed and alter only the current packet's polarity evidence.
    flipped = packet.events.clone()
    flipped[:, EV_P] = 1.0 - flipped[:, EV_P]
    with torch.no_grad():
        changed_events_prediction = model.forward_packet(dataclasses.replace(packet, events=flipped))
    assert not torch.equal(prediction, changed_events_prediction)


def test_next_packet_fk_receives_the_immediately_previous_prediction(model, monkeypatch):
    first = _packet(_events_on_hand(model))
    with torch.no_grad():
        prediction = model.forward_packet(first)
    # A second real packet consumes the first prediction through prev_state.
    second = dataclasses.replace(first, prev_state=prediction.detach(),
                                 target=torch.full_like(first.target, -500.0),
                                 t_start_us=first.t_end_us,
                                 t_end_us=first.t_end_us + 50_000)
    used_states = []
    original_fk = model._fk

    def record_fk(previous, betas):
        used_states.append(previous.detach().clone())
        return original_fk(previous, betas)

    monkeypatch.setattr(model, "_fk", record_fk)
    with torch.no_grad():
        output = model.forward_packet(second)
    assert output.shape == (1, 51) and torch.isfinite(output).all()
    assert used_states and all(torch.equal(state, prediction) for state in used_states)
