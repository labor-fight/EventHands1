"""Full model contracts for caller-owned mesh tracking and paired training."""
from __future__ import annotations

import dataclasses
import io

import pytest
import torch

from test_event_guided_mesh import _cfg, _events_on_hand, _packet, _state
from model import MNISTModel
from semkine.event_guided_mesh import MeshVertexState


@pytest.fixture
def tracker():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(81)
        model = MNISTModel(_cfg(EGM_MEMORY=True, EGM_GEOMETRY_NEIGHBORS=2))
    return model.eval()


def _pair(model):
    lead = _packet(_events_on_hand(model))
    main = dataclasses.replace(lead, prev_state=lead.prev_state + 0.01,
                               t_start_us=lead.t_end_us,
                               t_end_us=lead.t_end_us + 50_000,
                               target=lead.prev_state + 0.002)
    return lead, main


def test_track_holds_unobserved_memory_and_nonzero_history_on_empty_packets(tracker):
    lead, _ = _pair(tracker)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        pose, memory = tracker.track_packet(lead)
        observed = tracker.event_encoder.last_observed.clone()
        empty = _packet([], pose)
        held_pose, held_memory = tracker.track_packet(empty, memory)
    assert memory.hidden.shape == (1, 778, tracker.event_encoder.hidden)
    assert torch.count_nonzero(memory.hidden[~observed]) == 0
    assert torch.equal(memory.seen, observed)
    assert memory.hidden.dtype == torch.float32
    assert torch.equal(held_pose, pose)
    assert torch.equal(held_memory.hidden, memory.hidden)
    assert torch.equal(held_memory.seen, memory.seen)
    assert torch.count_nonzero(tracker.event_encoder.last_innovation) == 0


def test_history_cannot_drive_unrouted_or_ablated_packets(tracker):
    lead, _ = _pair(tracker)
    with torch.no_grad():
        pose, memory = tracker.track_packet(lead)
        background = _packet([(0, 2, 2, .01, 1), (0, 237, 177, .02, 0)], pose)
        for packet in (background, dataclasses.replace(lead, prev_state=pose)):
            if packet is not background:
                tracker.ablate_evidence = True
            predicted, state = tracker.track_packet(packet, memory)
            assert torch.equal(predicted, pose)
            assert torch.equal(state.hidden, memory.hidden)
            assert torch.equal(state.seen, memory.seen)


def test_cold_start_and_checkpoint_roundtrip_have_no_implicit_history(tracker):
    lead, main = _pair(tracker)
    with torch.no_grad():
        reference, memory = tracker.track_packet(lead)
        tracker.track_packet(main, memory)
        assert torch.equal(tracker.forward_packet(lead), reference)
    saved = io.BytesIO()
    torch.save(tracker.state_dict(), saved)
    saved.seek(0)
    clone = MNISTModel(tracker.cfg).eval()
    clone.load_state_dict(torch.load(saved), strict=True)
    with torch.no_grad():
        got_pose, got_memory = clone.track_packet(lead)
    assert torch.equal(got_pose, reference)
    assert torch.equal(got_memory.hidden, memory.hidden)
    assert torch.equal(got_memory.seen, memory.seen)


def test_paired_training_uses_predicted_pose_and_memory_without_main_prev_or_lead_target(tracker):
    lead, main = _pair(tracker)
    tracker.train()
    with torch.no_grad():
        first_pose, first_state = tracker.track_packet(lead)
        expected, _ = tracker.track_packet(dataclasses.replace(main, prev_state=first_pose), first_state)
        predicted, target, _, packed = tracker._predict_batch((lead, main))
        altered_lead = dataclasses.replace(lead, target=torch.full_like(lead.target, -100))
        altered_main = dataclasses.replace(main, prev_state=torch.full_like(main.prev_state, 100))
        repeated, _, _, _ = tracker._predict_batch((altered_lead, altered_main))
    assert torch.equal(predicted, expected)
    assert torch.equal(predicted, repeated)
    assert torch.equal(packed.prev_state, first_pose)
    assert target is main.target
    # No implicit sequence history survives a separate training sample pair.
    unrelated = dataclasses.replace(lead, events=lead.events[:0],
                                    ptr=torch.tensor([0, 0]))
    tracker.track_packet(unrelated, first_state)
    prediction, _, _, _ = tracker._predict_batch((lead, main))
    loss, _ = tracker._compute_loss(prediction, main.target, main.betas)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in tracker.parameters() if p.requires_grad)
    assert tracker.event_encoder.memory.cell.weight_hh.grad.abs().sum() > 0


def test_training_rejects_unpaired_or_discontinuous_streams(tracker):
    lead, main = _pair(tracker)
    tracker.train()
    with pytest.raises(ValueError, match="UNROLL_PAIR"):
        tracker._predict_batch(main)
    with pytest.raises(ValueError, match="consecutive"):
        tracker._predict_batch((lead, dataclasses.replace(main, sequence_id=main.sequence_id + 1)))
    with pytest.raises(ValueError, match="consecutive"):
        tracker._predict_batch((lead, dataclasses.replace(main, t_start_us=main.t_start_us + 1)))


def test_count_feature_ablation_cannot_silently_erase_observation_availability(tracker):
    packet = _packet(_events_on_hand(tracker))
    tracker.obs_feature_mask = torch.tensor([0., 1., 1., 1., 1., 1.])
    with pytest.raises(ValueError, match="observation availability"):
        tracker.track_packet(packet)


def test_unobserved_memory_can_hold_while_all_mesh_vertices_follow_root_motion(tracker):
    previous = _state()
    packet = _packet(_events_on_hand(tracker)[:2], previous)
    history = MeshVertexState(torch.randn(1, 778, tracker.event_encoder.hidden) * .01,
                              torch.ones(1, 778, dtype=torch.bool))
    with torch.no_grad():
        for head in [tracker.root_head, *tracker.joint_heads]:
            head[-1].weight.zero_()
            head[-1].bias.zero_()
        tracker.root_head[-1].bias[0] = .005
        old_mesh, _ = tracker.reconstruct_mesh(previous, packet.betas)
        pose, next_state = tracker.track_packet(packet, history)
        mesh, joints = tracker.reconstruct_mesh(pose, packet.betas)
    observed = tracker.event_encoder.last_observed
    assert observed.any() and (~observed).any()
    assert torch.equal(next_state.hidden[~observed], history.hidden[~observed])
    assert mesh.shape == (1, 778, 3) and joints.shape == (1, 21, 3)
    expected = old_mesh + torch.tensor([.005, 0, 0])
    torch.testing.assert_close(mesh, expected, atol=1e-7, rtol=1e-6)
