"""CPU-only real joint gradient/Adam/frozen-buffer and release checks."""
from pathlib import Path
import copy
import json
import sys
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import pytest
import torch
from config import load_config
from mano_layer import ManoLayer
from tools.xdl.data import counter_rng
from tools.xkf_fusion.motion_data import MotionDataset, arm_config, collate_incremental
from tools.xkf_joint.model import JointFrozenStatisticsModel, ORIGIN_GLOBAL
from tools.xkf_joint.train import parent_audit, all_buffer_hashes
from tools.xkf_joint.release import validate_release, validate_evidence, REQUIRED_CHECKS, sha

BASE = ROOT / 'outputs/xkf/fusion_20261015_r3/direct_finetune/incremental_finetune'


def inputs(seed=3407):
    cfg = arm_config(load_config(str(ROOT / f'configs/xkf_repair/repair_s{seed}.yaml')), 'raw')
    path = BASE / f'raw_s{seed}/xkf_ionly_raw_s{seed}-step=2600.ckpt'
    return cfg, path, torch.load(path, map_location='cpu', weights_only=False)


@pytest.mark.parametrize('seed', [3407, 3408, 3409])
def test_joint_parent96_and_actual_clocks(seed):
    torch.set_num_threads(4)
    cfg, path, saved = inputs(seed)
    model = JointFrozenStatisticsModel(cfg)
    audit = parent_audit(model, saved, path)
    assert len(audit['parameter_order']) == 96
    assert len(audit['all_buffers_sha256']) == 65
    assert all(p.requires_grad for p in model.parameters())
    assert not hasattr(model, 'fusion_policy')  # original self-burn, no deployment adapter
    model._trainer = SimpleNamespace(global_step=6200)
    marker = {}
    model.on_save_checkpoint(marker)
    clock = marker['xkf_joint_continuation']
    assert clock['i_adam_step'] == clock['d_adam_step'] == 6000
    assert clock['global_step'] == 6200 and not clock['is_original_global6000_protocol']
    assert clock['parent_incremental_continuation'] == saved['xkf_incremental_continuation']
    model._trainer = None


def test_bf16_actual_joint_step_and_original_burn_freeze_every_buffer():
    torch.set_num_threads(4)
    cfg, path, saved = inputs()
    model = JointFrozenStatisticsModel(cfg)
    parent_audit(model, saved, path)
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    dataset = MotionDataset(cfg, start_step=2600, max_steps=2601)
    index = next(i for i in range(20) if counter_rng(3407, 2600, i, 'self').random() < .5)
    batch = collate_incremental([dataset[index]])
    optimizer = model.configure_optimizers()['optimizer']
    optimizer.load_state_dict(copy.deepcopy(saved['optimizer_states'][0]))
    before = all_buffer_hashes(model)
    calls = []
    hook = model.direct.register_forward_pre_hook(lambda *args: calls.append(torch.is_grad_enabled()))
    model._trainer = SimpleNamespace(global_step=ORIGIN_GLOBAL)
    model.log = lambda *args, **kwargs: None
    model.train()
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model.training_step(batch, 0)
    loss.backward()
    hook.remove()
    assert calls == [False, False, False, True]
    assert model.xdl_self_rows == 1 and model.xdl_burn_rows == 3
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for direct in (False, True):
        assert any(bool((p.grad != 0).any()) for name, p in model.named_parameters()
                   if name.startswith('direct.') == direct)
    optimizer.step()
    assert all_buffer_hashes(model) == before
    assert all(int(optimizer.state[p]['step']) == 2401 for p in model.parameters())
    with pytest.raises(RuntimeError, match='never update residual'):
        model._update_moments()
    model._trainer = None


def test_full_cannot_be_released_by_missing_or_old_single_readiness(tmp_path):
    with pytest.raises(ValueError, match='new three-seed'):
        validate_release(None, None, ROOT, tmp_path/'config', tmp_path/'parent', 3407)
    old = tmp_path / 'old.json'
    manifest = tmp_path / 'manifest.json'
    old.write_text(json.dumps({'kind':'XKF_IONLY_ORIGINAL_READINESS_ACTUAL_FUSION',
                               'readiness':{'status':'PASS','ready_for_full_training':True}}))
    manifest.write_text('{}')
    with pytest.raises(ValueError, match='complete three-seed'):
        validate_release(old, manifest, ROOT, tmp_path/'config', tmp_path/'parent', 3407)


@pytest.mark.parametrize('check', REQUIRED_CHECKS)
def test_empty_or_reused_single_readiness_is_not_typed_complete_evidence(tmp_path, check):
    empty = tmp_path / 'empty.json'
    empty.write_text('{}')
    count = 3 if check in ('original_readiness_all_three', 'full2590_all_three') else 1
    refs = [dict(path=str(empty), sha256=sha(empty))] * count
    with pytest.raises((ValueError, KeyError)):
        validate_evidence(check, refs, {})
    empty.write_text(json.dumps({'kind':'XKF_IONLY_ORIGINAL_READINESS_ACTUAL_FUSION',
                                'readiness':{'status':'PASS','ready_for_full_training':True}}))
    refs = [dict(path=str(empty), sha256=sha(empty))] * count
    with pytest.raises((ValueError, KeyError)):
        validate_evidence(check, refs, {})


def test_release_rechecks_actual_evidence_dependencies(tmp_path):
    dependency = tmp_path / 'actual_input.json'
    dependency.write_text('{}')
    value = tmp_path / 'assessment.json'
    value.write_text(json.dumps({'input_sha256':{str(dependency):'0'*64}}))
    with pytest.raises(ValueError, match='dependency changed'):
        validate_evidence('actual112_all_three', [dict(path=str(value), sha256=sha(value))], {})
