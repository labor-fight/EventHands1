"""Scientific-contract tests for frozen XAG interventions, not score gates."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from tools.xag.diagnose import (weighted_state_mse, optimize_h3, probe_initials,
    wrong_token_content, route_sham, validated_existing, save_npz, rollout_specs)
from tools.xag.probes import LayerProbe, pad_interface, normalize_inputs, validate_folds, state_metrics


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


def test_weighted_objective_is_original_group_mean_not_flat_or_log():
    prediction = torch.zeros(2, 51); target = torch.zeros_like(prediction)
    target[:, :3] = .01; target[:, 3:6] = .2; target[:, 6:] = .3
    actual = weighted_state_mse(prediction, target, {})
    expected = (30000 * .01**2 + 60 * .2**2 + 450 * .3**2) / 51
    assert torch.allclose(actual, torch.full((2,), expected))


def test_real_motion_target_differs_from_injected_correction():
    incoming = torch.zeros(1, 51); incoming[0, 2] = .05
    endpoint = torch.zeros(1, 51); endpoint[0, 0] = .02; endpoint[0, 6] = .1
    delta = endpoint - incoming
    assert delta[0, 0] == .02 and delta[0, 2] == -.05 and delta[0, 6] == .1
    assert weighted_state_mse(delta, endpoint - incoming, {}).item() == 0


def test_fstar_keeps_original_and_clamps_supported_channels():
    h = torch.zeros(2, 3, 2); support = torch.tensor([[1., 0., 1.], [1., 1., 0.]])
    weights = torch.arange(6., dtype=torch.float32).reshape(3, 2) / 10
    def readout(value):
        one = (value * weights).sum((1, 2))
        return one[:, None].expand(-1, 51)
    target = torch.ones(2, 51)
    best, pred, audit = optimize_h3(readout, h, support, torch.tensor([.1, .2]), target, {}, steps=12, lr=.05)
    assert (best.abs() <= torch.tensor([.3, .6]) + 1e-7).all()
    assert torch.equal(best[~support.bool()], h[~support.bool()])
    assert (audit['best_loss'] <= audit['initial_loss']).all()
    assert audit['max_bound_violation'] == 0
    assert torch.equal(pred, readout(best))


def test_fstar_zero_radius_bit_identity_and_no_updates():
    torch.manual_seed(2); h = torch.randn(1, 4, 2)
    readout = lambda x: x.sum((1, 2))[:, None].expand(-1, 51)
    best, prediction, audit = optimize_h3(readout, h, torch.ones(1, 4), torch.ones(2), torch.zeros(1, 51), {}, radius=0.)
    assert torch.equal(best, h) and torch.equal(prediction, readout(h))
    assert audit['steps'] == 0


def test_probe_physical_conditions_include_seven_and_preserve_translation():
    clean = np.linspace(-.1, .1, 51).astype(np.float32)
    values = probe_initials(clean, 'seq:1:100', np.zeros(45))
    assert len(values) == 7
    assert {spec['id'] for spec, _ in values} == {'clean', 'z-50mm', 'z+50mm', 'root-20deg', 'root+20deg', 'finger-20deg', 'finger+20deg'}
    for spec, state in values:
        if spec['block'] in ('root', 'finger'):
            np.testing.assert_array_equal(state[:3], clean[:3])
        if spec['block'] == 'root':
            np.testing.assert_array_equal(state[6:], clean[6:])
        if spec['block'] == 'finger':
            np.testing.assert_array_equal(state[:6], clean[:6])


def test_rollout_all_physical_signs_without_mixed():
    specs = rollout_specs('zgz_local', 1)
    assert len(specs) == 14
    for block in ('translation', 'root', 'finger'):
        for scale in ('small', 'large'):
            assert {s['sign'] for s in specs if s['block'] == block and s['scale'] == scale} == {-1, 1}


def test_probe_pairs_have_identical_shapes_and_initialization():
    for left, right in [('H0', 'H3'), ('slots', 'zs')]:
        torch.manual_seed(1701); a = LayerProbe(left)
        torch.manual_seed(1701); b = LayerProbe(right)
        assert a.state_dict().keys() == b.state_dict().keys()
        assert all(torch.equal(a.state_dict()[key], b.state_dict()[key]) for key in a.state_dict())
        assert a.readout[-1].out_features == 51


def test_zs_padding_controls_and_fit_only_normalization():
    rng = np.random.default_rng(3); real = rng.normal(size=(5, 544)).astype(np.float32)
    wrong = real + 3.; residual = rng.normal(size=(5, 258)).astype(np.float32)
    prev = rng.normal(size=(5, 48)).astype(np.float32); fit = np.array([1, 1, 1, 0, 0], bool)
    inputs, stats = normalize_inputs(real, wrong, residual, prev, fit, 'zs', 'real')
    changed = real.copy(); changed[~fit] += 1e5
    _, stats2 = normalize_inputs(changed, wrong, residual, prev, fit, 'zs', 'real')
    assert all(np.array_equal(stats[k], stats2[k]) for k in stats)
    assert np.count_nonzero(inputs[0][:, :, 32:]) == 0
    state, _ = normalize_inputs(real, wrong, residual, prev, fit, 'zs', 'state')
    assert not state[0].any() and not state[1].any()
    np.testing.assert_array_equal(state[2], inputs[2])
    w, _ = normalize_inputs(real, wrong, residual, prev, fit, 'zs', 'wrong')
    np.testing.assert_array_equal(w[1], inputs[1]); np.testing.assert_array_equal(w[2], inputs[2])


def test_reject_state99_and_q1():
    with pytest.raises(ValueError):
        pad_interface(np.zeros((1, 707), np.float32), 'zs')
    model = LayerProbe('zs')
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, 258), torch.zeros(1, 258), torch.zeros(1, 99))


def test_fold_validation_forbids_crossing_any_replica():
    rows = [{'split': 'fit', 'segment': 's:a', 'sequence': 's', 'index': 0},
            {'split': 'check', 'segment': 's:b', 'sequence': 's', 'index': 1}]
    assert validate_folds(rows).tolist() == [True, False]
    with pytest.raises(ValueError):
        validate_folds(rows + [{**rows[0], 'split': 'check'}])


def test_failed_metrics_keep_denominator_and_are_not_finite_mean():
    delta = np.zeros((2, 51)); delta[1, 0] = np.nan
    result = state_metrics(delta, np.zeros_like(delta), {})
    assert result['n'] == 2 and result['n_failed'] == 1 and result['translation_mm'] is None


def test_resume_verifies_content_and_identity(tmp_path):
    path = tmp_path / 'x.npz'; save_npz(path, {'x': np.arange(4)}, {'identity': {'a': 1}})
    assert validated_existing(path, {'a': 1})
    with pytest.raises(RuntimeError):
        validated_existing(path, {'a': 2})
    path.write_bytes(b'bad')
    with pytest.raises(RuntimeError):
        validated_existing(path, {'a': 1})


def test_wrong_tokens_change_only_xy_keep_association_geometry_objects():
    class Capture(nn.Module):
        def forward(self, *args, **kwargs):
            self.args = args; self.kwargs = kwargs; return args[0]
    model = nn.Module(); model.p = nn.Parameter(torch.zeros(())); model.xyz = Capture(); model.render_w = 240; model.render_h = 180
    tok = torch.zeros(1, 2, 7); tok[0, :, 2] = torch.tensor([-1., 1.]); tok[0, :, 3:] = .25
    ray = torch.randn(1, 2, 3); mask = torch.ones(1, 2, dtype=torch.bool); verts = torch.randn(1, 778, 3)
    assoc = {'vid': torch.zeros(1, 2, 4, dtype=torch.long), 'A': torch.ones(1, 2, 4) / 4}
    donor = np.array([[0, 60, 45, .01, 0], [0, 120, 90, .02, 1]], np.float32)
    with wrong_token_content(model, donor):
        changed = model.xyz(tok, ray, mask, verts, assoc, torch.zeros(1, 48))
    assert torch.equal(changed[0, :, :2], torch.tensor([[.25, .25], [.5, .5]]))
    assert torch.equal(changed[..., 2:], tok[..., 2:])
    assert model.xyz.args[1] is ray and model.xyz.args[3] is verts and model.xyz.args[4] is assoc
    assert torch.equal(model.xyz(tok, ray, mask, verts, assoc, torch.zeros(1, 48)), tok)


def test_sham_keeps_mass_and_exact_intervention_event_set(monkeypatch):
    import semkine.xag as xag
    base = {'vid': torch.tensor([[[0, 1, 2], [3, 4, 5]]]), 'A': torch.full((1, 2, 3), 1 / 3),
            'q': torch.tensor([[.4, .7]]), 'z': torch.tensor([[.8, .9]])}
    routed = {**base, 'vid': torch.tensor([[[4, 5, 6], [3, 4, 5]]])}
    original = lambda *a, **k: (routed, {'xag_replaced': torch.tensor([[True, False]])})
    monkeypatch.setattr(xag, 'route_association', original)
    with route_sham('sham_face'):
        result, audit = xag.route_association('face_route', base, None, None, torch.zeros(1, 10, 3))
    assert xag.route_association is original
    assert torch.equal(result['A'], base['A']) and result['q'] is base['q'] and result['z'] is base['z']
    assert torch.equal(result['vid'][:, 1], base['vid'][:, 1])
    assert (result['vid'][:, 0] != base['vid'][:, 0]).any()
    assert audit['sham_changed_events'] == 1


def test_slot_wrong_control_preserves_common_residual_inside_and_outside():
    rng = np.random.default_rng(2); real = rng.normal(size=(4, 17, 258)).astype(np.float32)
    wrong = real + 5; residual = real[:, -1]; prev = np.zeros((4, 48), np.float32); fit = np.array([1, 1, 0, 0], bool)
    r, _ = normalize_inputs(real, wrong, residual, prev, fit, 'slots', 'real')
    w, _ = normalize_inputs(real, wrong, residual, prev, fit, 'slots', 'wrong')
    np.testing.assert_array_equal(r[0][:, -1], w[0][:, -1])
    np.testing.assert_array_equal(r[1], w[1])
    assert not np.array_equal(r[0][:, :-1], w[0][:, :-1])


def test_probe_physical_wrist_callback_compensates_rotation_induced_shift():
    # Deliberately nonstandard wrist makes a missing wrist_fn observable.
    clean = np.zeros(51, np.float32); clean[:3] = [.1, -.1, .7]
    wrist = lambda state: np.asarray(state[:3]) + .02 * np.sin(np.asarray(state[3:6]))
    values = probe_initials(clean, 'packet', np.zeros(45), wrist_fn=wrist)
    for spec, state in values:
        if spec['block'] in ('root', 'finger'):
            np.testing.assert_allclose(wrist(state), wrist(clean), atol=1e-7)
    assert any(not np.array_equal(state[:3], clean[:3]) for spec, state in values if spec['block'] == 'root')


def test_probe_scores_so3_endpoint_and_hold_not_axis_angle_error_only():
    from tools.xag.probes import so3_endpoint_errors
    initial = np.zeros((2, 51)); target = np.zeros_like(initial)
    target[:, 3] = .2; target[:, 6] = .3
    scores = so3_endpoint_errors(target, target, initial, np.zeros(45))
    np.testing.assert_allclose(scores['root_endpoint_deg'], 0, atol=1e-6)
    np.testing.assert_allclose(scores['finger_endpoint_deg'], 0, atol=1e-6)
    np.testing.assert_allclose(scores['hold_root_endpoint_deg'], .2 * 180 / np.pi, atol=1e-6)
    assert (scores['hold_finger_endpoint_deg'] > 0).all()


def test_diagnostic_source_manifest_excludes_resource_tools():
    from tools.xag.diagnose import source_hashes
    sources = source_hashes()
    assert 'tools/xag/build_sources.py' in sources
    assert 'tools/xag/check_resources.py' not in sources
    assert 'tools/xag/diagnose.py' in sources
