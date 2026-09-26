"""Full MANO output and causal state contracts; no checkpoint accuracy claims."""
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'model'))
from model import MNISTModel
from semkine.streaming_tracker import SparseS37Tracker
from semkine.routed_readout import pool_joint_evidence


@pytest.fixture
def tracker():
    torch.manual_seed(17)
    cfg = yaml.safe_load((ROOT/'configs/semkine/s37_routed_s3407.yaml').read_text())
    cfg['MODEL'].update(ENCODER_HIDDEN=8, ENCODER_FEAT=16, ENCODER_K=3,
                        ENCODER_WINDOW=5, ENCODER_MAX_NODES=32,
                        ENCODER_LAYERS=2, ACTIVE_HIDDEN=8)
    m = MNISTModel(cfg).eval()
    return SparseS37Tracker(m, frame_period_us=4000)


def initial(tracker, stream_id=1):
    K = torch.tensor([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]])
    return tracker.initialize(K, stream_id=stream_id, start_us=0)


def events():
    xy = torch.tensor([[120., 90., 0.], [121., 92., 1.], [118., 87., 0.]])
    return xy, torch.tensor([0, 300, 3999], dtype=torch.int64)


@pytest.mark.parametrize('autocast', [False, True])
def test_empty_output_keeps_float32_state_and_emits_full_mesh(tracker, autocast):
    s = initial(tracker)
    pose = s.pose.clone()
    pose[:, 2] = 0.50123
    s = replace(s, pose=pose)
    with torch.autocast('cpu', dtype=torch.bfloat16, enabled=autocast):
        out = tracker.query(4000, s)
    assert torch.equal(out.pose, pose)
    assert out.pose.dtype == torch.float32
    assert out.vertices.shape == (1, 778, 3)
    assert out.joints.shape == (1, 21, 3)
    assert torch.isfinite(out.vertices).all()
    assert s.last_query_us == 0  # The caller's previous state was not overwritten.


def test_no_new_events_does_not_integrate_old_evidence_again(tracker):
    xy, t = events()
    s = tracker.append(xy, t, initial(tracker))
    first = tracker.query(4000, s)
    second = tracker.query(8000, first.state)
    assert torch.equal(first.pose, second.pose)
    assert torch.equal(first.vertices, second.vertices)


def test_no_new_events_skips_readout_but_validates_configuration(tracker, monkeypatch):
    xy, t = events()
    s = tracker.query(4000, tracker.append(xy, t, initial(tracker))).state
    def forbidden_readout(*args):
        raise AssertionError('no unused feature recomputation in an empty update')
    monkeypatch.setattr(tracker.encoder, 'readout', forbidden_readout)
    out = tracker.query(8000, s)
    assert torch.equal(out.pose, s.pose)
    assert out.vertices.shape == (1, 778, 3)
    tracker.encoder.horizon_us += 1
    with pytest.raises(ValueError, match='configuration changed'):
        tracker.query(8000, s)


def test_same_model_supports_isolated_stream_states(tracker):
    xy, t = events()
    a, b = initial(tracker, 7), initial(tracker, 8)
    a = tracker.append(xy, t, a)
    tracker.query(4000, a)
    out_b = tracker.query(4000, b)
    assert torch.equal(out_b.pose, b.pose)
    assert b.graph is None


def test_watermark_future_and_cadence_are_enforced(tracker):
    xy, t = events()
    s = tracker.append(xy, t, initial(tracker))
    with pytest.raises(ValueError): tracker.query(3999, s)
    with pytest.raises(TypeError): tracker.query(4000.9, s)
    s = tracker.query(4000, s).state
    with pytest.raises(ValueError): tracker.query(4000, s)
    with pytest.raises(ValueError): tracker.append(xy[:1], t[:1], s)
    future = tracker.append(xy[:1], torch.tensor([8000]), s)
    with pytest.raises(ValueError): tracker.query(8000, future)


def test_period_shape_and_oracle_controls_fail_closed(tracker):
    with pytest.raises(TypeError): SparseS37Tracker(tracker.model, frame_period_us=0.5)
    with pytest.raises(ValueError): SparseS37Tracker(tracker.model, frame_period_us=50001)
    with pytest.raises(TypeError): tracker.initialize(torch.eye(3), stream_id=1, start_us=0.5)
    with pytest.raises(ValueError): tracker.initialize(torch.eye(3), stream_id=1, start_us=2**64)
    s = initial(tracker)
    with pytest.raises(ValueError): tracker.query(4000, replace(s, betas=torch.ones(1,10)))
    tracker.model.route_prev_override = torch.zeros(1,51)
    with pytest.raises(ValueError): tracker.query(4000, s)


def test_hidden_model_context_does_not_override_explicit_calibration_or_shape(tracker):
    xy, t = events()
    a = tracker.append(xy, t, initial(tracker))
    out_a = tracker.query(4000, a)
    tracker.model.set_hand_context(torch.full((1,10), float('nan')),
                                   torch.full((1,3,3), float('nan')))
    out_b = tracker.query(4000, a)
    assert torch.equal(out_a.pose, out_b.pose)
    assert torch.equal(out_a.vertices, out_b.vertices)


def test_event_evidence_and_heads_have_finite_gradient(tracker):
    xy, t = events()
    state = tracker.append(xy, t, initial(tracker))
    out = tracker.query(4000, state)
    (out.pose.square().mean()+out.vertices.square().mean()).backward()
    for name, param in tracker.model.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
    assert tracker.model.event_encoder.embed.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('module_name', ['root_head', 'joint_heads', 'prev_mlp'])
@pytest.mark.parametrize('with_events', [False, True])
def test_head_weight_change_requires_explicit_tracking_reset(tracker, module_name, with_events):
    xy, t = events()
    state = initial(tracker)
    if with_events:
        state = tracker.append(xy, t, state)
    # Normal optimizer-style in-place mutation, not the unsupported .data escape.
    param = next(p for name, p in tracker.model.named_parameters()
                 if name.startswith(module_name+'.'))
    with torch.no_grad():
        param.add_(0.001)
    with pytest.raises(ValueError, match='model parameters changed'):
        tracker.query(4000, state)
    with pytest.raises(ValueError, match='model parameters changed'):
        tracker.append(xy[:0], t[:0], state)
    fresh = initial(tracker)
    assert torch.equal(tracker.query(4000, fresh).pose, fresh.pose)


def test_full_prefix_and_cache_reconstruct_same_short_recursive_mesh_trajectory(tracker):
    s = initial(tracker)
    reference_pose = s.pose.clone()
    all_xy, all_t = [], []
    for step in range(8):
        xy, t = events()
        t = t+4000*step
        q = (step+1)*4000
        all_xy.append(xy)
        all_t.append(t)
        s = tracker.append(xy, t, s)
        out = tracker.query(q, s)
        feat,h,px,py,mask = tracker.encoder.full_reference(
            torch.cat(all_xy), torch.cat(all_t), q, stream_id=s.stream_id)
        a,_,_ = tracker.model._route_nodes(px,py,mask,reference_pose,s.betas,s.camera_K)
        evidence,_ = pool_joint_evidence(h,a,mask)
        delta = tracker.model._decode_active(feat,reference_pose,evidence)
        delta = delta+tracker.model.prev_mlp(reference_pose.to(delta.dtype))
        reference_pose = reference_pose+delta.float()
        v,j = tracker.model._fk(reference_pose,s.betas)
        torch.testing.assert_close(out.pose,reference_pose,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(out.vertices,v,atol=1e-5,rtol=1e-5)
        torch.testing.assert_close(out.joints,j,atol=1e-5,rtol=1e-5)
        s = out.state
