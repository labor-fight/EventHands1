"""Inference adapter contracts and actual independent-loop state propagation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from semkine.xkf_fusion import FusionPolicy, XKFFusionModel, correct_direct, validate_correction
from semkine.xkf_repair import XKFRepairModel
from tools.xkf_fusion import evaluate as Q


def test_disabled_adapter_delegates_exactly_to_frozen_path(monkeypatch):
    model = XKFFusionModel.__new__(XKFFusionModel)
    torch.nn.Module.__init__(model)
    model.fusion_policy = None
    token, packet = object(), object()
    monkeypatch.setattr(XKFRepairModel, 'forward_packet', lambda self, value: token if value is packet else None)
    assert model.forward_packet(packet) is token
    assert model.state_dict() == {}


@pytest.mark.parametrize('raises', [False, True])
def test_temperature_is_log_multiplier_and_restores_bitwise(monkeypatch, raises):
    model = XKFFusionModel.__new__(XKFFusionModel)
    torch.nn.Module.__init__(model)
    model.direct = torch.nn.Module()
    model.direct.log_beta = torch.nn.Parameter(torch.tensor(1.234567))
    original = model.direct.log_beta.detach().clone()
    observed = []
    def measurement(self, packet):
        observed.append(self.direct.log_beta.exp().item())
        if raises:
            raise RuntimeError('intentional inference error')
        return 'state', 'aux'
    monkeypatch.setattr(XKFRepairModel, 'direct_measurement', measurement)
    if raises:
        with pytest.raises(RuntimeError, match='intentional'):
            model.measure_direct(None, 1.5)
    else:
        assert model.measure_direct(None, 1.5) == ('state', 'aux')
    assert observed[0] == pytest.approx(original.exp().item() * 1.5)
    assert torch.equal(original, model.direct.log_beta)


def test_count_gate_uses_current_count_and_only_scales_d_gain():
    policy = FusionPolicy({'id': 'gated', 'gain_scale': [1., .25, .5], 'count_gate': 'ramp1000'})
    base = torch.ones(19) * .8
    gain = policy.gains(base, torch.tensor([0, 100, 1000, 2000]))
    assert gain[0].count_nonzero() == 0
    assert torch.allclose(gain[1, :4], torch.tensor([.08, .08, .08, .02]))
    assert gain[2, 4:].tolist() == pytest.approx([.4] * 15)
    assert torch.equal(gain[2], gain[3])


def test_empty_packet_holds_all_51_coordinates_despite_correction():
    policy = FusionPolicy({'id': 'corrected', 'direct_correction':
                           {'kind': 'translation_affine', 'matrix': torch.eye(3).tolist(), 'bias': [1., 2., 3.]}})
    prev = torch.randn(2, 51)
    result, _, _ = policy.fuse(torch.randn(2, 51), torch.randn(2, 51), {}, prev, torch.zeros(2), torch.ones(19))
    assert torch.equal(result, prev)


def test_affine_reads_only_direct_output_and_fixed_coefficients():
    state = torch.randn(2, 51)
    matrix = torch.tensor([[1., .1, 0.], [0., 2., 0.], [0., 0., 1.]])
    bias = torch.tensor([.01, -.02, .03])
    spec = validate_correction({'kind': 'translation_affine', 'matrix': matrix.tolist(), 'bias': bias.tolist()})
    actual = correct_direct(state, {}, spec)
    assert torch.allclose(actual[:, :3], state[:, :3] @ matrix.T + bias)
    assert torch.equal(actual[:, 3:], state[:, 3:])
    assert correct_direct(state, {}, None) is state
    with pytest.raises(ValueError, match='target-free'):
        validate_correction(dict(spec, target=[0] * 51))


def test_wrist_depth_correction_uses_physical_joint_offset():
    state = torch.zeros(1, 51); state[0, :3] = torch.tensor([.1, .2, .5])
    joints = torch.zeros(1, 21, 3); joints[0, 0] = torch.tensor([.01, -.02, .03])
    spec = validate_correction({'kind': 'wrist_depth_affine', 'scale': 2., 'offset': 0.})
    corrected = correct_direct(state, {'joints0': joints}, spec)
    assert torch.allclose(corrected[0, :3] + joints[0, 0], 2 * (state[0, :3] + joints[0, 0]))


def test_each_candidate_recomputes_i_from_its_own_previous_state(monkeypatch):
    class Model:
        def incremental_prediction(self, packet):
            value = packet.prev_state.clone()
            value[:, 0] = 2 * value[:, 0] + 1
            return value
        def gains(self, mode):
            return torch.full((19,), .5)
    class Cache:
        device = torch.device('cpu')
        beta = torch.zeros(1, 10)
        data = {'gt': np.zeros((3, 51), np.float32)}
        def packet(self, row, prev, temperatures):
            packet = SimpleNamespace(prev_state=prev, counts=torch.ones(len(prev), dtype=torch.long),
                                     lnes_window_ms=torch.full((len(prev),), 200))
            z = torch.zeros(1, 51)
            return packet, {1.: (z, {})}
    monkeypatch.setattr(Q.R, 'frame_metrics', lambda model, pred, truth, beta:
                        (pred.new_zeros(len(pred), len(Q.R.METRICS)), pred.new_zeros(len(pred), 3), pred.new_zeros(len(pred), 3)))
    clip = {'clip_id': 'test', 'row_indices': [0, 1], 'initial_gt_ms': 0}
    meta = {'rows': [{'end_ms': 1, 'packet_id': 'a'}, {'end_ms': 2, 'packet_id': 'b'}]}
    candidates = [Q.Candidate({'id': 'identity'}, 3407),
                  Q.Candidate({'id': 'no_d', 'gain_scale': [0., 0., 0.]}, 3407)]
    raw = Q.rollout(Model(), meta, clip, Cache(), candidates)
    modes = Q.BASE_MODES + ('identity', 'no_d')
    # Starting at x=0: no-D path has x1=1, x2=3. Fused has x1=.5,
    # x2=1. Reusing the fused incoming state would wrongly yield no-D x2=2.
    assert raw['predictions'][1, 0, modes.index('no_d'), 0] == 3.
    assert raw['predictions'][1, 0, modes.index('identity'), 0] == 1.
    assert np.array_equal(raw['predictions'][:, :, modes.index('identity')], raw['predictions'][:, :, modes.index('fused')])


def test_clip_calibration_is_selected_once_and_source_hash_required(tmp_path):
    document = {'clip_specs': {'one': {'kind': 'translation_affine', 'matrix': torch.eye(3).tolist(), 'bias': [1., 0., 0.]},
                               'two': {'kind': 'translation_affine', 'matrix': torch.eye(3).tolist(), 'bias': [2., 0., 0.]}}}
    path = tmp_path / 'calibration.json'
    Q.write_json(path, document)
    value = {'id': 'calibrated', 'direct_correction_file': {'path': str(path), 'sha256': Q.E.sha(path)}}
    candidate = Q.Candidate(value, 3407)
    first = candidate.for_clip('one')
    second = candidate.for_clip('two')
    assert first.correction['bias'][0] == 1. and second.correction['bias'][0] == 2.
    assert candidate.for_clip('one').correction == first.correction
    with pytest.raises(KeyError):
        candidate.for_clip('unseen')
    value['direct_correction_file']['sha256'] = 'wrong'
    with pytest.raises(ValueError, match='SHA'):
        Q.Candidate(value, 3407)
