"""Engineering invariants of the opt-in full-S43 field frontend, no task scores."""
import copy
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tools.s43_revisit import s43_field_model as M
from tools.s43_revisit.field_query import FieldQueryBuilder, FieldQueryConfig
from tools.s43_revisit.field_query_probe import sample_patches


@pytest.fixture(scope='module')
def original():
    torch.set_num_threads(2)
    from tools.s43_integration.model import load_legacy
    return load_legacy()[0]


def inputs(B=1):
    field = torch.zeros(B, 40, 180, 240)
    field[:, 16:24] = 1
    field[:, :16, 50:90, 90:140] = .2
    field[:, 24:, 48:92, 88:144] = .1
    prev = torch.zeros(B, 51); prev[:, 2] = .6
    K = torch.tensor([[600., 0, 320], [0, 600, 240], [0, 0, 1.]])
    return field, prev, K


def test_public_stem_initialization_is_weighted_counts_not_LNES():
    torch.manual_seed(1); old = torch.nn.Conv2d(4, 3, 3, padding=1)
    new = M.initialize_field_stem(old)
    x = torch.rand(2, 42, 5, 6)
    weighted = (x[:, :16].reshape(2, 8, 2, 5, 6)*M.public_recency_weights()[None, :, None, None, None]).sum(1)
    expected = old(torch.cat((weighted, x[:, 40:]), 1))
    torch.testing.assert_close(new(x), expected, rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(new.weight[:, 40:], old.weight[:, 2:], rtol=0, atol=0)
    assert torch.count_nonzero(new.weight[:, 16:40]) == 0
    torch.testing.assert_close(new.bias, old.bias, rtol=0, atol=0)


def test_activity_ignores_coverage_and_learned_weights_but_includes_real_coarse_and_render():
    field = torch.zeros(1, 40, 5, 6); field[:, 16:24] = 1
    render = torch.zeros(1, 2, 5, 6)
    assert not M.observed_activity(field, render).any()
    field[0, 0, 1, 2] = 1; field[0, 24, 3, 4] = .1; render[0, 1, 4, 5] = .5
    m = M.observed_activity(field, render)
    assert m.sum() == 3 and m[0, 0, 1, 2] and m[0, 0, 3, 4] and m[0, 0, 4, 5]


def test_base_forward_is_never_called_and_no_empty_LNES_hold(original, monkeypatch):
    model = M.S43FieldModel(copy.deepcopy(original), 'global_only').eval()
    def poison(*a, **kw):
        raise AssertionError('old LNES forward called')
    monkeypatch.setattr(model.base, 'forward', poison)
    with torch.no_grad():
        model.base.rn.fc.weight.zero_(); model.base.rn.fc.bias.fill_(.001)
        for p in model.base.prev_mlp.parameters(): p.zero_()
    field, prev, K = inputs(); field.zero_()
    assert model.base.zero_event_gate
    out, aux = model(field, prev, K_source=K, ref_ms=0, cutoff_ms=50, return_aux=True)
    torch.testing.assert_close(out, prev+.001, rtol=0, atol=0)
    assert not aux['LNES_used'] and not aux['zero_event_gate']
    assert aux['local'] is None and aux['render'].any()


@pytest.mark.parametrize('arm', ['hand_box', 'static'])
def test_local_frozen_shared_sampler_keeps_time_slot_channels(original, arm):
    model = M.S43FieldModel(copy.deepcopy(original), arm).eval()
    field, prev, K = inputs()
    field[0, :16] *= torch.arange(1, 17)[:, None, None]
    patches, meta = model._local_patches(field, prev, torch.zeros(1, 10), K[None], [0], [50], [()])
    k = K.numpy().copy(); k[:2] *= model.base.render_scale
    query = FieldQueryBuilder(prev[0], np.zeros(10), k, 0, config=FieldQueryConfig()).build(50, mode=arm)
    q = {name: getattr(query, name) for name in ('grid', 'valid', 'in_view', 'full_interp_support')}
    expected = sample_patches(field[0], q, 2)
    assert patches.shape == (1, 8, 90, 5, 7, 7)
    torch.testing.assert_close(patches[0], expected, rtol=0, atol=0)
    assert meta[0]['K_space'] == 'event240x180; no internal scaling'
    assert not torch.count_nonzero(model.local_head[-1].weight)


def test_loss_shape_is_separate_from_public_geometry_input(original, monkeypatch):
    model = M.S43FieldModel(copy.deepcopy(original), 'static').eval()
    field, prev, K = inputs(); seen = {}
    old_render = model.base._render_prev
    def render(p, beta, k):
        seen['input_beta'] = beta.detach().clone()
        return old_render(p, beta, k)
    monkeypatch.setattr(model.base, '_render_prev', render)
    out, aux = model(field, prev, K_source=K, ref_ms=0, cutoff_ms=50, return_aux=True)
    assert torch.count_nonzero(seen['input_beta']) == 0
    def loss(p, target, beta):
        seen['target_beta'] = beta
        return (p-target).square().mean(), {}
    monkeypatch.setattr(model.base, '_compute_loss', loss)
    target_beta = torch.ones(1, 10)
    model.loss(out, prev, beta_target=target_beta)
    assert seen['target_beta'] is target_beta
    assert torch.count_nonzero(seen['input_beta']) == 0
    assert aux['query_metadata'][0]['shape_condition'] == 'beta0'
    with pytest.raises(ValueError, match='explicit beta_target'):
        model.loss(out, prev, beta_target=None)


def test_parameter_accounting_and_train_mode_are_explicit(original):
    models = [M.S43FieldModel(copy.deepcopy(original), arm) for arm in M.ARMS]
    assert len({m.counts()['total'] for m in models}) == 1
    assert models[0].counts()['structurally_active'] < models[1].counts()['structurally_active']
    assert models[1].counts()['structurally_active'] == models[2].counts()['structurally_active']
    assert all(p.requires_grad for m in models for p in m.parameters())
    for m in models:
        m.train(); assert m.base.rn.bn1.training
    disabled = M.S43FieldModel(copy.deepcopy(original), use_prev_mlp=False)
    assert not disabled.counts()['prev_mlp_active']
    assert disabled.counts()['structurally_active'] < models[2].counts()['structurally_active']


def test_original_update_conventions_are_reused(original, monkeypatch):
    model = M.S43FieldModel(copy.deepcopy(original)).eval()
    _, prev, _ = inputs(); delta = torch.ones_like(prev)*.001; feats = torch.zeros(1, 512)
    torch.testing.assert_close(model._compose(delta, prev, feats), prev+delta, rtol=0, atol=0)
    model.base.root_compose = 'so3'; seen = []
    method = model.base._compose_root_so3
    def spy(added, d, p):
        seen.append(True); return method(added, d, p)
    monkeypatch.setattr(model.base, '_compose_root_so3', spy)
    out = model._compose(delta, prev, feats)
    torch.testing.assert_close(out, method(prev+delta, delta, prev), rtol=0, atol=0)
    assert seen == [True]


def test_invalid_input_has_no_silent_hold_or_state_fallback(original):
    model = M.S43FieldModel(copy.deepcopy(original), 'global_only').eval()
    field, prev, K = inputs(); field[0, 0, 0, 0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        model(field, prev, K_source=K, ref_ms=0, cutoff_ms=50)
    field.zero_()
    with pytest.raises(ValueError, match='future'):
        model(field, prev, K_source=K, ref_ms=51, cutoff_ms=50)
