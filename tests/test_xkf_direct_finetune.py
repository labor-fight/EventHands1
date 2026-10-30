"""Real parent-state migration and real direct loss, not a mocked optimizer."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

from config import load_config
from tools.xkf.data import build_train_dataset
from tools.xkf_fusion.direct_data import DirectDataset, arm_config, collate_direct, sampling_identity
from tools.xkf_fusion.finetune_direct import (DirectFinetuneModel, START, RATE, optimizer_named_state,
                                             state_hashes, validate_parent)


@pytest.fixture(scope='module')
def actual():
    torch.set_num_threads(4)
    cfg = load_config(str(ROOT / 'configs/xkf_repair/repair_s3407.yaml'))
    parent = torch.load(ROOT / 'outputs/xkf/repair_20261015_r2/repair_s3407/xkf_repair_s3407-step=2200.ckpt',
                        map_location='cpu', weights_only=False)
    model = DirectFinetuneModel(cfg)
    audit = validate_parent(model, parent)
    return cfg, parent, model, audit


def test_all96_original_adam_slots_have_exact_named_migration(actual):
    _, parent, model, audit = actual
    assert audit['parameter_count'] == 96
    assert audit['direct_parameter_count'] == 60
    assert audit['all_named_moments_restored_exactly']
    assert len(parent['optimizer_states'][0]['param_groups'][0]['params']) == 96
    assert sum(p.requires_grad for p in model.parameters()) == 60
    assert all(p.requires_grad == n.startswith('direct.') for n, p in model.named_parameters())


def test_botharms_same_sample_ids_and_control_main_exact_old_producer(actual):
    cfg, _, _, _ = actual
    aug = DirectDataset(arm_config(cfg, 'aug'), start_step=START, max_steps=START + 2)
    raw = DirectDataset(arm_config(cfg, 'raw'), start_step=START, max_steps=START + 2)
    old = build_train_dataset(cfg, start_step=START, max_steps=START + 2)
    changed = 0
    for index in (0, 1, 2, 3, 511, 512, 1023, 1024):
        a, r, o = aug[index], raw[index], old[index]
        assert a['sampling_identity'] == r['sampling_identity'] == sampling_identity(o)
        for key in ('events', 'lnes', 'target', 'prev_state', 'camera_K', 'betas'):
            assert np.array_equal(getattr(a['main'], key), getattr(o['main'], key)), key
        changed += not np.array_equal(a['main'].lnes, r['main'].lnes)
        assert a['main'].lnes_window_ms == o['main'].lnes_window_ms
    assert changed > 0
    assert len(aug.sequences) == 72
    assert len({s.split('_')[0] for s, _ in aug.sequences}) == 9


def test_raw_config_only_declared_augmentation_flags_change(actual):
    cfg, _, _, _ = actual
    aug, raw = arm_config(cfg, 'aug'), arm_config(cfg, 'raw')
    raw['AUG']['DOMRAND']['ENABLED'] = aug['AUG']['DOMRAND']['ENABLED']
    raw['AUG']['POLARITY_FLIP'] = aug['AUG']['POLARITY_FLIP']
    raw['AUG']['PIXEL_POLARITY_SWAP'] = aug['AUG']['PIXEL_POLARITY_SWAP']
    assert raw == aug
    assert aug['XKF_DATA']['PREFIX_LNES'] is True


def test_real_bf16_d_only_update_preserves_i_bn_ema_and_adam(actual, monkeypatch):
    cfg, parent, model, _ = actual
    model.load_state_dict(parent['state_dict'], strict=True)
    configured = model.configure_optimizers()
    optimizer, scheduler = configured['optimizer'], configured['lr_scheduler']['scheduler']
    optimizer.load_state_dict(copy.deepcopy(parent['optimizer_states'][0]))
    scheduler.load_state_dict(copy.deepcopy(parent['lr_schedulers'][0]))
    frozen = state_hashes(model, frozen_only=True)
    moments = optimizer_named_state(model, optimizer)
    model.train(True)
    assert all(not m.training for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm))
    ds = DirectDataset(arm_config(cfg, 'raw'), start_step=START, max_steps=START + 2)
    batch = collate_direct([ds[0], ds[1]])
    monkeypatch.setattr(model, 'log', lambda *args, **kwargs: None)
    def forbidden(*args, **kwargs):
        raise AssertionError('D-only training must not use I, burn or update EMA')
    for method in ('incremental_prediction', '_burn_main', '_update_moments', '_xyz_mesh_forward'):
        monkeypatch.setattr(model, method, forbidden)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        prediction, aux = model.direct_measurement(batch['main'])
        expected = model._direct_loss(prediction, aux, batch['main'])[0].float().clamp_min(1e-12).log10()
        actual_loss = model.training_step(batch, 0)
    assert torch.equal(expected, actual_loss)
    altered = copy.copy(batch['main'])
    altered.prev_state = torch.full_like(altered.prev_state, float('nan'))
    with torch.no_grad(), torch.autocast('cpu', dtype=torch.bfloat16):
        prediction2, _ = model.direct_measurement(altered)
    assert torch.equal(prediction, prediction2)
    actual_loss.backward()
    for name, p in model.named_parameters():
        if name.startswith('direct.'):
            assert p.grad is not None and torch.isfinite(p.grad).all(), name
        else:
            assert p.grad is None, name
    optimizer.step()
    scheduler.step()
    assert scheduler.last_epoch == START + 1
    assert optimizer.param_groups[0]['lr'] == RATE
    assert state_hashes(model, frozen_only=True) == frozen
    after = optimizer_named_state(model, optimizer)
    changed = 0
    for name, p in model.named_parameters():
        assert int(optimizer.state[p]['step']) == (START + 1 if name.startswith('direct.') else START)
        if not name.startswith('direct.'):
            assert moments[name] == after[name]
        else:
            changed += not torch.equal(p.detach(), parent['state_dict'][name])
    assert changed > 0


def test_wrong_parent_clock_rejected(actual):
    _, parent, model, _ = actual
    bad = dict(parent, global_step=2000)
    with pytest.raises(ValueError, match='step2200'):
        validate_parent(model, bad)
