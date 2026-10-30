"""Repair contracts: unchanged inference without flags, FP32 state, pure GT motion."""
from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

from config import load_config
from semkine.events import EventPacketBatch
from semkine.xkf import XKFModel
from semkine.xkf_repair import XKFRepairModel
from semkine.xyz_recovery_study import RecoveryXYZModel


def config(**flags):
    cfg = load_config(str(REPO / 'configs/xkf/xkf_s3407.yaml'))
    cfg['MODEL'].update(flags)
    return cfg


def packet():
    generator = torch.Generator().manual_seed(771)
    count = 64
    events = torch.zeros(count, 5)
    events[:, 1] = torch.randint(80, 160, (count,), generator=generator)
    events[:, 2] = torch.randint(50, 130, (count,), generator=generator)
    events[:, 3] = torch.arange(count) * (.05 / count)
    events[:, 4] = torch.arange(count) % 2
    previous = torch.randn(2, 51, generator=generator) * .1
    previous[:, :3] = torch.tensor([[.012345, -.033234, .663333], [.014567, -.017894, .731234]])
    previous[:, 3:6] = torch.tensor([[.931111, -.473333, 1.132345], [1.23456, -.364567, .882345]])
    camera = torch.tensor([[603.45, 0., 325.09], [0., 602.96, 242.10], [0., 0., 1.]]).expand(2, 3, 3).clone()
    lnes = torch.zeros(2, 180, 240, 2)
    lnes[:, 55:125, 80:165] = torch.rand(2, 70, 85, 2, generator=generator)
    return EventPacketBatch(
        events=events, ptr=torch.tensor([0, count, count]), sequence_id=torch.zeros(2, dtype=torch.long),
        t_start_us=torch.zeros(2, dtype=torch.long), t_end_us=torch.full((2,), 50000, dtype=torch.long),
        delta_t_s=torch.full((2,), .05), is_sequence_start=torch.zeros(2, dtype=torch.bool),
        is_sequence_end=torch.zeros(2, dtype=torch.bool), target=previous + .01, prev_state=previous,
        betas=torch.zeros(2, 10), camera_K=camera, lnes=lnes, lnes_window_ms=torch.full((2,), 200))


@pytest.fixture(scope='module')
def original():
    torch.set_num_threads(4)
    torch.manual_seed(330)
    return XKFModel(config()).eval()


def repair(original, **flags):
    model = XKFRepairModel(config(**flags)).eval()
    model.load_state_dict(original.state_dict(), strict=True)
    return model


def test_no_new_parameter_or_buffer_and_original_checkpoint_loads_strictly(original):
    model = repair(original, XKF_REPAIR_FP32_ADD=True, XKF_REPAIR_MOTION_TARGET=True)
    assert list(model.state_dict()) == list(original.state_dict())
    assert [(n, tuple(p.shape)) for n, p in model.named_parameters()] == [(n, tuple(p.shape)) for n, p in original.named_parameters()]
    assert [(n, tuple(p.shape)) for n, p in model.named_buffers()] == [(n, tuple(p.shape)) for n, p in original.named_buffers()]


def test_incremental_path_survives_original_trainer_module_rebinding(original, monkeypatch):
    import semkine.xyz_recovery_study as module

    monkeypatch.setattr(module, 'RecoveryXYZModel', XKFModel)
    spec = importlib.util.spec_from_file_location('semkine._test_rebound_xkf_repair',
                                                 REPO / 'semkine/xkf_repair.py')
    rebound = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rebound)
    model = rebound.XKFRepairModel(config()).eval()
    model.load_state_dict(original.state_dict(), strict=True)
    packed = packet()
    with torch.no_grad():
        expected = RecoveryXYZModel.forward_packet(original, packed)
        assert torch.equal(model.incremental_prediction(packed), expected)


@pytest.mark.parametrize('amp', [False, True])
def test_disabled_flags_keep_original_predictions_and_losses_bitwise(original, amp):
    model, packed = repair(original), packet()
    with torch.no_grad(), torch.autocast('cpu', enabled=amp, dtype=torch.bfloat16):
        old = original.branches(packed)
        new = model.branches(packed)
        assert torch.equal(old[0], new[0]) and torch.equal(old[1], new[1])
        assert torch.equal(original.forward_packet(packed), model.forward_packet(packed))
        old_loss, old_parts = original._direct_loss(old[1], old[2], packed)
        new_loss, new_parts = model._direct_loss(new[1], new[2], packed)
        assert torch.equal(old_loss, new_loss)
        assert old_parts.keys() == new_parts.keys()
        assert all(torch.equal(old_parts[k], new_parts[k]) for k in old_parts)
        old_i, _ = original._compute_loss(old[0], packed.target, packed.betas)
        new_i, _ = model.incremental_loss(new[0], packed, {})
        assert torch.equal(old_i, new_i)


def test_actual_bf16_xyz_graph_zero_delta_holds_live_and_empty_exactly(original):
    model, packed = repair(original, XKF_REPAIR_FP32_ADD=True), packet()
    # XYZ's original zero initialization predicts zero before any update.
    assert not bool(model.xyz.fc2.weight.any()) and not bool(model.xyz.fc2.bias.any())
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        legacy = RecoveryXYZModel.forward_packet(original, packed)
        result = model.incremental_prediction(packed)
    assert result.dtype == torch.float32
    assert torch.equal(result, packed.prev_state)
    assert not torch.equal(legacy[0], packed.prev_state[0])  # reproduces the original live-row precision bug
    assert torch.equal(legacy[1], packed.prev_state[1])


def test_small_motion_survives_bf16_addition_and_empty_row_stays_exact(original, monkeypatch):
    model, packed = repair(original, XKF_REPAIR_FP32_ADD=True), packet()
    delta = torch.full_like(packed.prev_state, 0.0001, dtype=torch.bfloat16, requires_grad=True)
    monkeypatch.setattr(model, '_xyz_mesh_forward', lambda _batch, _prev: delta)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        result = model.incremental_prediction(packed)
    assert torch.equal(result[0], packed.prev_state[0] + delta[0].float())
    assert torch.equal(result[1], packed.prev_state[1])
    assert not torch.equal(result[0], packed.prev_state[0])
    result.sum().backward()
    assert torch.equal(delta.grad[0], torch.ones_like(delta.grad[0]))
    assert torch.equal(delta.grad[1], torch.zeros_like(delta.grad[1]))


def test_motion_target_does_not_denoise_incoming_state_error(original):
    model, packed = repair(original, XKF_REPAIR_MOTION_TARGET=True), packet()
    delta_gt = torch.full_like(packed.target, .002)
    wrong = dataclasses.replace(packed, prev_state=packed.prev_state + .125)
    good_prediction = packed.prev_state + delta_gt
    wrong_prediction = wrong.prev_state + delta_gt
    batch = {'motion_delta': delta_gt}
    loss_a, _ = model.incremental_loss(good_prediction, packed, batch)
    loss_b, _ = model.incremental_loss(wrong_prediction, wrong, batch)
    assert float(loss_a) < 1e-12 and float(loss_b) < 1e-12
    # Predicting the absolute GT is no longer the desired answer when the prior is wrong.
    loss_correct_to_gt, _ = model.incremental_loss(wrong.target, wrong, batch)
    assert float(loss_correct_to_gt) > .1
    with pytest.raises(ValueError, match='motion_delta'):
        model.incremental_loss(wrong_prediction, wrong, {})


def test_inference_never_reads_targets_or_requires_motion_labels(original):
    model = repair(original, XKF_REPAIR_FP32_ADD=True, XKF_REPAIR_MOTION_TARGET=True,
                   XKF_REPAIR_D_AUX_2D_WEIGHT=.5, XKF_REPAIR_D_AUX_3D_WEIGHT=.5)
    packed = packet()
    changed = dataclasses.replace(packed, target=torch.full_like(packed.target, float('nan')))
    with torch.no_grad():
        assert torch.equal(model.incremental_prediction(packed), model.incremental_prediction(changed))
        assert torch.equal(model.forward_packet(packed), model.forward_packet(changed))


def test_auxiliary_coefficients_are_relative_to_original_translation_weight(original, monkeypatch):
    import semkine.xkf_repair_losses as helpers

    model = repair(original, XKF_REPAIR_D_AUX_2D_WEIGHT=.5, XKF_REPAIR_D_AUX_3D_WEIGHT=.25)
    packed = packet()
    base = torch.tensor(2.)
    aux2, aux3 = torch.tensor(.02), torch.tensor(.03)
    monkeypatch.setattr(XKFModel, '_direct_loss', lambda self, z, aux, batch: (base, {'original': base}))
    monkeypatch.setattr(helpers, 'direct_aux_losses', lambda *args: {'d_aux_2d': aux2, 'd_aux_3d': aux3})
    loss, parts = model._direct_loss(packed.target, {}, packed)
    expected = base + model.lambda_t * (.5 * aux2 + .25 * aux3) / model.normalizer
    assert torch.equal(loss, expected)
    assert parts == {'original': base, 'd_aux_2d': aux2, 'd_aux_3d': aux3}


def test_training_moments_score_actual_gt_after_self_burn(original, monkeypatch):
    model, packed = repair(original, XKF_REPAIR_MOTION_TARGET=True), packet()
    burned = dataclasses.replace(packed, prev_state=packed.prev_state + .125)
    motion_delta = torch.full_like(packed.target, .002)
    xi = burned.prev_state + motion_delta
    direct = packed.target + .003
    observed = {}
    monkeypatch.setattr(model, '_burn_main', lambda batch: (burned, [0]))
    monkeypatch.setattr(model, 'branches', lambda main: (xi, direct, {}))
    monkeypatch.setattr(model, '_direct_loss', lambda z, aux, main: (torch.tensor(1.), {}))
    monkeypatch.setattr(model, '_moment_rows', lambda main, take: take)
    def capture(x_i, z_d, previous, truth, rows):
        observed.update(x_i=x_i, z_d=z_d, previous=previous, truth=truth, rows=rows)
    monkeypatch.setattr(model, '_update_moments', capture)
    monkeypatch.setattr(model, 'log', lambda *a, **kw: None)
    result = model.training_step({'main': packed, 'motion_delta': motion_delta}, 0)
    assert torch.isfinite(result)  # exact zero motion error has a finite log floor
    assert torch.equal(observed['truth'], packed.target)
    assert not torch.equal(observed['truth'], motion_delta)
    assert torch.equal(observed['previous'], burned.prev_state)
    assert observed['rows'] == [0]


@pytest.mark.parametrize('key,value', [('XKF_REPAIR_FP32_ADD', 'yes'),
                                      ('XKF_REPAIR_D_AUX_2D_WEIGHT', -1.),
                                      ('XKF_REPAIR_D_AUX_3D_WEIGHT', float('nan'))])
def test_repair_configuration_rejects_ambiguous_values(key, value):
    with pytest.raises(ValueError, match=key):
        XKFRepairModel(config(**{key: value}))
