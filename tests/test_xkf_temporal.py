"""True motion, causal labels, exact objective, and real CPU optimizer checks."""
from pathlib import Path
import copy
import dataclasses
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import pytest
import torch
from config import load_config
from mano_layer import ManoLayer
from tools.xdl.data import counter_rng, packet_hash
from tools.xkf_fusion.motion_data import MotionDataset, arm_config, collate_incremental
from tools.xkf_joint.train import parent_audit, all_buffer_hashes
from tools.xkf_temporal.model import TemporalDModel, physical_joint_motion_mse

BASE = ROOT / 'outputs/xkf/fusion_20261015_r3/direct_finetune/incremental_finetune'


def test_correct_nonzero_motion_is_zero_and_static_prediction_is_wrong():
    previous = torch.zeros(2, 21, 3)
    movement = torch.tensor([.02, -.01, .005]).expand_as(previous)
    assert physical_joint_motion_mse(previous, movement, previous, movement) == 0
    assert physical_joint_motion_mse(previous, previous, previous, movement) > 0
    # A constant position bias cancels in displacement, hence frame losses
    # remain necessary in both arms of the paired experiment.
    assert physical_joint_motion_mse(previous + .1, movement + .1, previous, movement) < 1e-16


def test_wrong_motion_has_both_endpoint_gradients_and_no_gt_gradients():
    previous = torch.zeros(2, 21, 3, requires_grad=True)
    main = torch.full((2, 21, 3), .01, requires_grad=True)
    truth_previous = torch.zeros(2, 21, 3, requires_grad=True)
    truth_main = torch.full((2, 21, 3), .02, requires_grad=True)
    physical_joint_motion_mse(previous, main, truth_previous, truth_main).backward()
    assert previous.grad is not None and main.grad is not None
    assert bool((previous.grad != 0).all()) and torch.equal(previous.grad, -main.grad)
    assert truth_previous.grad is None and truth_main.grad is None


@pytest.mark.parametrize('weight', [0., 1.])
def test_exact_equal_frame_and_fixed_half_temporal_weight(weight):
    @dataclasses.dataclass
    class LossPacket:
        target: torch.Tensor
        betas: torch.Tensor
        marker: float
    beta = torch.zeros(1, 10)
    main = LossPacket(target=torch.zeros(1, 51), betas=beta, marker=2.)
    previous = LossPacket(target=torch.zeros(1, 51), betas=beta, marker=4.)
    model = SimpleNamespace(temporal_d_weight=weight, lambda_t=30000., normalizer=51.,
        _direct_loss=lambda z, aux, packet: (torch.tensor(packet.marker), {'d_t':torch.tensor(packet.marker)}),
        _fk=lambda state, beta: (None, state[:, None, :3].expand(-1, 21, -1)))
    z = torch.zeros(1, 51)
    aux_main = {'joints0':torch.full((1, 21, 3), .01, requires_grad=True)}
    aux_previous = {'joints0':torch.zeros(1, 21, 3, requires_grad=True)}
    loss, parts = TemporalDModel.paired_direct_loss(model, z, aux_main, main, z, aux_previous, previous)
    expected = torch.tensor(3.) + .5 * weight * 30000. / 51. * torch.tensor(.01).square()
    torch.testing.assert_close(loss, expected, rtol=1e-7, atol=1e-7)
    assert parts['d_t'] == 3 and parts['d_main_normalized'] == 2 and parts['d_previous_normalized'] == 4


@pytest.mark.parametrize('weight', [True, -.1, .5, 2., float('nan')])
def test_undeclared_temporal_weights_rejected(weight):
    with pytest.raises(ValueError, match='control0 or temporal1'):
        TemporalDModel({'MODEL':{'XKF_TEMPORAL_D_WEIGHT':weight}})


@pytest.fixture(scope='module')
def real_inputs():
    torch.set_num_threads(4)
    cfg = arm_config(load_config(str(ROOT/'configs/xkf_repair/repair_s3407.yaml')), 'raw')
    path = BASE/'raw_s3407/xkf_ionly_raw_s3407-step=2600.ckpt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    dataset = MotionDataset(cfg, start_step=2600, max_steps=2601)
    index = next(i for i in range(20) if counter_rng(3407, 2600, i, 'self').random() < .5)
    item = dataset[index]
    return cfg, path, saved, item


def model_for(real_inputs, weight):
    cfg, path, saved, item = real_inputs
    cfg = copy.deepcopy(cfg)
    cfg['MODEL']['XKF_TEMPORAL_D_WEIGHT'] = weight
    model = TemporalDModel(cfg)
    audit = parent_audit(model, saved, path)
    assert len(audit['parameter_order']) == 96 and len(audit['all_buffers_sha256']) == 65
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    return model, collate_incremental([item])


def test_previous_loss_view_cannot_mutate_burn_targets_and_requires_same_clock(real_inputs):
    _, _, _, item = real_inputs
    batch = collate_incremental([item])
    packet = batch['prefix'][-1]
    before = packet_hash(packet)
    view = TemporalDModel.previous_loss_packet(batch)
    assert view is not packet and packet_hash(packet) == before
    assert torch.equal(view.target, batch['previous_gt']) and not bool(packet.target.any())
    assert view.events is packet.events and view.lnes is packet.lnes
    batch['prefix'][-1] = dataclasses.replace(packet, t_end_us=packet.t_end_us - 1000)
    with pytest.raises(ValueError, match='exactly50ms'):
        TemporalDModel.previous_loss_packet(batch)


def test_direct_previous_ignores_target_and_incoming_state(real_inputs):
    model, batch = model_for(real_inputs, 1.)
    model.eval()
    previous = batch['prefix'][-1]
    altered = dataclasses.replace(previous, target=torch.full_like(previous.target, float('nan')),
                                  prev_state=torch.full_like(previous.prev_state, float('nan')))
    with torch.no_grad():
        expected, auxiliary = model.direct_measurement(previous)
        actual, _ = model.direct_measurement(altered)
        _, joints = model._fk(expected, previous.betas)
    assert torch.equal(actual, expected) and torch.isfinite(actual).all()
    torch.testing.assert_close(auxiliary['joints0'] + expected[:, None, :3], joints, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('weight', [0., 1.])
def test_actual_bf16_joint_step_keeps_original_burn_96adam_and65buffers(real_inputs, weight):
    model, batch = model_for(real_inputs, weight)
    batch['main'].target.requires_grad_(True)
    batch['previous_gt'].requires_grad_(True)
    _, _, saved, _ = real_inputs
    optimizer = model.configure_optimizers()['optimizer']
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]))
    before = all_buffer_hashes(model)
    packet_before = [packet_hash(p) for p in batch['prefix']]
    lnes_before = [p.lnes.clone() for p in batch['prefix']]
    motion_before = batch['motion_delta'].clone()
    calls = []
    hook = model.direct.register_forward_pre_hook(lambda *args: calls.append(torch.is_grad_enabled()))
    model._trainer = SimpleNamespace(global_step=2600)
    model.log = lambda *args, **kwargs: None
    model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model.training_step(batch, 0)
    loss.backward()
    hook.remove()
    assert calls == [False, False, False, True, True]
    assert model.xdl_self_rows == 1 and model.xdl_burn_rows == 3
    assert batch['main'].target.grad is None and batch['previous_gt'].grad is None
    assert torch.equal(batch['motion_delta'], motion_before)
    assert [packet_hash(p) for p in batch['prefix']] == packet_before
    assert all(torch.equal(p.lnes, value) for p, value in zip(batch['prefix'], lnes_before))
    assert all(not bool(p.target.any()) for p in batch['prefix'])
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all()), name
    for direct in (False, True):
        assert any(bool((p.grad != 0).any()) for n, p in model.named_parameters() if n.startswith('direct.') == direct)
    optimizer.step()
    assert all(int(optimizer.state[p]['step']) == 2401 for p in model.parameters())
    assert all_buffer_hashes(model) == before
    with pytest.raises(RuntimeError, match='never update residual'):
        model._update_moments()
    model._trainer = SimpleNamespace(global_step=2800)
    marker = {}
    model.on_save_checkpoint(marker)
    assert marker['xkf_joint_continuation']['target_global_step'] == 2800
    assert marker['xkf_joint_continuation']['target_parameter_steps'] == 2600
    clock = marker['xkf_temporal_continuation']
    assert clock['global_step'] == 2800 and clock['i_adam_step'] == clock['d_adam_step'] == 2600
    assert clock['temporal_weight'] == weight and not clock['formal'] and not clock['automatic_full_training']
    assert not hasattr(model, 'fusion_policy')
    model._trainer = None
