"""Synthetic-only unit/clock checks for the new fixed zgz entry point."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tools.xvm_joint.zgz_eval import (AuditedPacketModel, initial_rng_for_sequence,
                                    metric_arrays, packet_input_token)


class ShapeBase:
    def _fk(self, states, beta):
        vertices = states[:, None, :3].expand(-1, 778, -1).clone()
        joints = states[:, None, :3].expand(-1, 21, -1).clone()
        vertices[:, 0, 1] += beta[:, 0]*.02
        joints[:, 0, 1] += beta[:, 0]*.01
        return vertices, joints


def test_metrics_use_prediction_beta0_actual_gt_beta_and_actual_fk_wrist():
    states = np.zeros((3, 51), np.float32)
    beta = np.zeros(10, np.float32);beta[0] = 1
    metrics = metric_arrays(ShapeBase(), states, states.copy(), beta, chunk=2)
    np.testing.assert_allclose(metrics['root_abs_mm'], 10., atol=1e-5)
    np.testing.assert_allclose(metrics['mpjpe_abs_mm'], 10./21, atol=1e-6)
    np.testing.assert_allclose(metrics['mpvpe_abs_mm'], 20./778, atol=1e-6)
    np.testing.assert_allclose(metrics['mpjpe_ra_mm'], 10.*20/21, atol=1e-5)
    np.testing.assert_allclose(metrics['mpvpe_ra_mm'], 20.*777/778, atol=1e-5)


def test_prefix_rng_advances_exact_original_run_noise_calls_without_prior_labels(tmp_path):
    folder = tmp_path/'test';folder.mkdir()
    np.savez(folder/'first_aux.npz', valid_runs_ms=np.array([[0, 49], [100, 150], [300, 450]]))
    inventory = [('first', 'test'), ('second', 'test')]
    def noise(cfg, rng, scale):
        return np.r_[rng.standard_normal(3), rng.standard_normal(3), rng.standard_normal(45)]
    rng, consumed = initial_rng_for_sequence({}, tmp_path, inventory, 'second', sample_noise=noise)
    reference = np.random.default_rng(0)
    noise({}, reference, 1.);noise({}, reference, 1.)
    assert consumed == 2
    np.testing.assert_array_equal(rng.standard_normal(51), reference.standard_normal(51))
    first_rng, count = initial_rng_for_sequence({}, tmp_path, inventory, 'first', sample_noise=noise)
    assert count == 0
    np.testing.assert_array_equal(first_rng.standard_normal(51), np.random.default_rng(0).standard_normal(51))


class PoisonPacket:
    def __init__(self):
        self.events = torch.tensor([[0., 1., 2., .01, 0.]])
        self.ptr = torch.tensor([0, 1])
        self.t_start_us, self.t_end_us = torch.tensor([0]), torch.tensor([50000])
        self.prev_state = torch.tensor([[0., 0., .6]+[0.]*48])
        self.camera_K = torch.eye(3)[None]
    @property
    def target(self):
        raise AssertionError('target input prohibited')
    @property
    def betas(self):
        raise AssertionError('actual beta input prohibited')


class NoLabelCore(nn.Module):
    def __init__(self):
        super().__init__();self.last_audit = {}
    def reset_state(self, previous):
        self.last_audit = {}
    def forward_packet(self, packet):
        self.last_audit = dict(target_accessed=False, actual_beta_accessed=False)
        return packet.prev_state+.001


def test_audit_token_and_guard_wrap_only_legal_forward_input(tmp_path):
    packet = PoisonPacket()
    token = packet_input_token(packet)
    assert len(token) == 64
    tracker = AuditedPacketModel(NoLabelCore(), tmp_path/'trace.jsonl', 'synthetic')
    tracker.reset_state(packet.prev_state)
    prediction = tracker.forward_packet(packet)
    torch.testing.assert_close(prediction, packet.prev_state+.001)
    row = json.loads((tmp_path/'trace.jsonl').read_text())
    assert row['input_token'] == token and row['run_id'] == 0 and row['frame'] == 0
    assert row['input_audit']['blocked_aux_beta_attempts'] == 0
    assert row['input_audit']['blocked_pose_meta_attempts'] == 0
    tracker.set_hand_context(object(), object())


def test_nonfinite_endpoint_is_not_silently_dropped():
    states = np.zeros((1, 51), np.float32);states[0, 0] = np.nan
    with pytest.raises(ArithmeticError, match='Nonfinite metric'):
        metric_arrays(ShapeBase(), states, np.zeros_like(states), np.zeros(10, np.float32))
