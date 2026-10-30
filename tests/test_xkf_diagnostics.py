"""Formal XKF diagnostic clocks, target isolation, frozen gates, and failure retention."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

from tools.xkf import diagnostics as X


def tiny_data(n=1200):
    events = np.column_stack([np.arange(n) % 37, np.arange(n) % 29, np.arange(n) % 2]).astype(np.uint8)
    gt = np.zeros((n, 51), np.float32)
    gt[:, 2] = .5
    return dict(events=events, offsets=np.arange(n + 1), tsub=None, gt=gt,
                aux=dict(valid_runs_ms=np.array([[0, n]]), betas=np.zeros(10, np.float32),
                         camera_K=np.eye(3, dtype=np.float32)))


def test_zero_targets_and_causal_long_evidence_for_every_perturbation():
    data = tiny_data()
    states = X.initial_states(data['gt'][99])
    runtime = X.LongWindowEval([[0, 1200]])
    row = dict(end_ms=149, event_start_ms=100, previous_end_ms=99)
    packet = X.make_packet(data, row, states, torch.zeros(1, 10), torch.eye(3)[None],
                           torch.device('cpu'), runtime)
    assert not packet.target.any()
    assert packet.lnes_window_ms.tolist() == [150] * 7
    assert packet.t_start_us.tolist() == [100000] * 7
    assert packet.t_end_us.tolist() == [150000] * 7
    assert torch.equal(packet.lnes[0], packet.lnes[-1])
    assert torch.equal(packet.events[:50, 1:], packet.events[-50:, 1:])


def test_root_perturbation_is_camera_composition_and_exactly_twenty_degrees():
    state = np.zeros(51, np.float32)
    state[2], state[3:6] = .5, [.4, -.7, .3]
    out = X.initial_states(state)
    assert np.allclose(out[1:3, 2], [.45, .55])
    assert np.array_equal(out[3:, :3], np.repeat(state[None, :3], 4, axis=0))
    assert np.allclose(X.root_error_deg(out[3:], state), 20, atol=1e-5)
    expected = Rotation.from_rotvec([-np.deg2rad(20), 0, 0]) * Rotation.from_rotvec(state[3:6])
    assert np.allclose(Rotation.from_rotvec(out[3, 3:6]).as_matrix(), expected.as_matrix(), atol=1e-7)


def test_wrong_time_uses_same_sequence_indices_at_least_one_second_away():
    rows = [{'end_ms': n} for n in [49, 99, 1049, 1099, 2049]]
    got = X.wrong_time_indices(rows)
    assert got.tolist() == [2, 3, 4, 1, 2]
    assert all(abs(rows[i]['end_ms'] - rows[j]['end_ms']) >= 1000 for i, j in enumerate(got))
    with pytest.raises(ValueError, match='no same-sequence'):
        X.wrong_time_indices([{'end_ms': 49}, {'end_ms': 999}])


def test_pure_direct_prediction_bypasses_fusion_and_retains_failed_rows():
    data = tiny_data()
    row = dict(end_ms=49, event_start_ms=0, previous_end_ms=0)
    states = X.initial_states(data['gt'][0])
    packet = X.make_packet(data, row, states, torch.zeros(1, 10), torch.eye(3)[None],
                           torch.device('cpu'), X.LongWindowEval([[0, 1200]]))

    class Direct:
        def direct_measurement(self, p):
            assert not p.target.any()
            out = torch.ones_like(p.target)
            out[p.prev_state[:, 2] < .49] = torch.nan
            return out, {}

        def forward_packet(self, p):
            raise AssertionError('G1 must not run fused inference')

    values, failures = X.predict(Direct(), packet, direct=True, batch_size=2)
    assert np.isnan(values[1]).all()
    assert np.equal(values[[0, 2, 3, 4, 5, 6]], 1).all()
    assert failures == [{'row': 1, 'reason': 'nonfinite_prediction'}]
    assert X.finite_mean([1, float('nan'), 3]) is None


def test_recovery_requires_three_consecutive_and_accepts_tenth_step_with_confirmation():
    clean = np.zeros(12)
    e = np.full(12, 10.)
    e[9:] = 1.
    got = X.sustained_recovery(e, e, clean, clean)
    assert got['recovered'] and got['hit_step'] == 10 and got['confirmation_step'] == 12
    e[10] = np.nan
    got = X.sustained_recovery(e, e, clean, clean)
    assert not got['recovered'] and not got['trajectory_finite']
    short = X.sustained_recovery([0, 0], [0, 0], [0, 0], [0, 0])
    assert short['insufficient_for_any_confirmation'] and short['right_censored']
    assert not short['recovered']


def test_frozen_short_run_denominator_cannot_silently_be_relaxed():
    impossible = dict(spec='z+50mm', recovered=False, insufficient_for_any_confirmation=True, right_censored=True)
    okay = dict(spec='z+50mm', recovered=True, insufficient_for_any_confirmation=False, right_censored=False)
    result = X.summarize_recovery([copy.copy(impossible) for _ in range(102)] + [copy.copy(okay) for _ in range(288)])
    assert result['status'] == 'UNASSESSABLE_PROTOCOL'
    assert result['passed'] is False and result['protocol_feasibility'] is False
    assert result['trials'] == 390 and result['mathematical_ceiling'] == pytest.approx(288 / 390)
    assert result['eligible_only_descriptive_fraction'] == 1


def test_real_frozen_g3_metadata_has_65_starts_and_17_unconfirmable_local_runs():
    bank, _ = X.D.load_bank()
    clips = X.recovery_clips(bank)
    assert len(clips) == 65
    assert sum(seq == 'zgz_local' for seq, _, _ in clips) == 41
    assert sum(len(rows) < 3 for _, _, rows in clips) == 17
    assert all(len(rows) <= 12 for _, _, rows in clips)


def test_recovery_uses_own_states_and_never_exposes_truth(monkeypatch):
    data = tiny_data()
    rows = [dict(end_ms=49, event_start_ms=0, previous_end_ms=0),
            dict(end_ms=99, event_start_ms=50, previous_end_ms=49),
            dict(end_ms=149, event_start_ms=100, previous_end_ms=99)]
    bank = {'sequences': {'zgz_local': {'valid_runs_ms': [[0, 1200]]}}}
    monkeypatch.setattr(X, 'recovery_clips', lambda _: [('zgz_local', 'fixture', rows)])
    monkeypatch.setattr(X.D, 'load_data', lambda *_: data)
    monkeypatch.setattr(X, 'validate_bank_sequence', lambda *_: None)

    class OwnState:
        FK_ROOT_JOINT = 0

        def set_hand_context(self, *_):
            pass

        def forward_packet(self, p):
            assert not p.target.any()
            out = p.prev_state.clone()
            out[:, 0] += .001
            return out

        def _fk(self, states, beta):
            return None, states[:, None, :3]

    _, raw, failures = X.recovery(OwnState(), {}, bank, torch.device('cpu'), 2)
    pred = raw['recovery_000_prediction']
    assert np.allclose(pred[0, :, 0], [0, .001, .002, .003])
    assert not failures


def test_formal_main_rows_reject_wrong_checkpoint(tmp_path):
    row = dict(protocol=dict(exact_optimizer_step=6000, step_ms=50, window_ms=50, direct_window_ms=200),
               per_seed={'3407': dict(step=6000, checkpoint_sha256='wrong', n_frames=2590)},
               xkf=dict(kmode='fused'), finite_subset_reduction=False, numeric_status='PASS')
    (tmp_path / 'xkf_s3407_main_row.json').write_text(json.dumps(row))
    with pytest.raises(ValueError, match='identity/protocol'):
        X.read_main_rows('xkf_s3407', 3407, 'expected', tmp_path)


def test_checkpoint_requires_exact_step6000_before_model_loading(tmp_path):
    torch.save({'global_step': 200}, tmp_path / 'debug-step=200.ckpt')
    with pytest.raises(RuntimeError, match='exact unique step6000'):
        X.load_model(str(tmp_path), None, torch.device('cpu'))
    torch.save({'global_step': 200}, tmp_path / 'debug-step=6000.ckpt')
    with pytest.raises(RuntimeError, match='global_step'):
        X.load_model(str(tmp_path), None, torch.device('cpu'))


def test_g5_sign_and_missing_artifacts_are_explicit():
    raw = {}
    single = {}
    for s, offset in zip(X.NAMES, [0, .2]):
        truth = np.array([[0, 0, .4 + offset], [0, 0, .5 + offset], [0, 0, .6 + offset]])
        raw.update({f'{s}_true_wrist': truth, f'{s}_direct_wrist': truth.copy(),
                    f'{s}_wrong_time_indices': np.array([2, 2, 0])})
        single[s] = dict(n_packets=3, z_bias_mm=0, root_mean_deg=0)
    single['one_step'] = dict(depth_slope=.5, root_ratio_median=.4, clean_root_mean_deg=3., empty_packet_holds=True)
    rec = dict(summary=X.gate(True))
    rows = dict(values=dict(fused=8., i=10., d=10.1, d_zero=9.99,
                            jitter_all_mm_per_step=7., gt_move_mm_per_step=4.))
    gates = X.assess_gates(single, raw, rec, rows)
    assert gates['G1']['passed'] and gates['G2']['passed'] and gates['G4']['passed']
    assert gates['G5']['passed'] is False
    assert gates['G5']['improvement_mm']['d_zero'] == pytest.approx(1.99)
    single['one_step']['empty_packet_holds'] = False
    assert X.assess_gates(single, raw, rec, rows)['G2']['passed'] is False
    rows['values'].pop('d_zero')
    rows['values'].pop('jitter_all_mm_per_step')
    gates = X.assess_gates(single, raw, rec, rows)
    assert gates['G4']['status'] == gates['G5']['status'] == 'MISSING_ARTIFACT'


@pytest.mark.parametrize('change', ['finish_status', 'finish_preflight', 'finish_sha',
                                   'metadata_preflight', 'metadata_steps', 'metadata_seed', 'config_steps'])
def test_formal_identity_rejects_debug_artifacts_even_with_step6000_filename(tmp_path, change):
    checkpoint = tmp_path / 'xkf-step=6000.ckpt'
    torch.save({'global_step': 6000}, checkpoint)
    finish = dict(status='completed_exact_step6000', global_step=6000, preflight_not_formal=False,
                  checkpoint_sha256=X.ROW.sha(checkpoint))
    metadata = dict(preflight_not_formal=False, max_steps=6000, seed=3407)
    cfg = dict(SEED=3407, TRAIN=dict(MAX_STEPS=6000))
    if change == 'finish_status':
        finish['status'] = 'preflight_complete'
    elif change == 'finish_preflight':
        finish['preflight_not_formal'] = True
    elif change == 'finish_sha':
        finish['checkpoint_sha256'] = 'wrong'
    elif change == 'metadata_preflight':
        metadata['preflight_not_formal'] = True
    elif change == 'metadata_steps':
        metadata['max_steps'] = 200
    elif change == 'metadata_seed':
        metadata['seed'] = 3408
    elif change == 'config_steps':
        cfg['TRAIN']['MAX_STEPS'] = 200
    (tmp_path / 'train_finish.json').write_text(json.dumps(finish))
    (tmp_path / 'training_metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='formal'):
        X.formal_identity(tmp_path, checkpoint, cfg)


def test_single_step_mock_and_summary_json_serialization(monkeypatch, tmp_path):
    data = tiny_data(2200)
    rows = X.packet_rows(data['aux']['valid_runs_ms'])
    for i, row in enumerate(rows):
        row.update(index=i, packet_id=f'fixture:{i}', n_raw_events=50)
    meta = dict(valid_runs_ms=[[0, 2200]], rows=rows)
    bank = dict(sequences={name: meta for name in X.NAMES})
    monkeypatch.setattr(X.D, 'load_data', lambda *_: data)

    class Model:
        FK_ROOT_JOINT = 0

        def set_hand_context(self, *_):
            pass

        def direct_measurement(self, p):
            assert not p.target.any()
            out = torch.zeros_like(p.target)
            out[:, 2] = .5
            return out, {}

        def forward_packet(self, p):
            assert not p.target.any()
            return p.prev_state.clone()

        def _fk(self, states, beta):
            return None, states[:, None, :3]

    summary, raw, failures = X.single_step(Model(), {}, bank, torch.device('cpu'), 2)
    assert not failures
    assert summary['one_step']['empty_packet_holds']
    assert summary['one_step']['depth_slope'] == pytest.approx(1)
    assert raw['zgz_local_fused'].shape == (44, 7, 51)
    path = tmp_path / 'summary.json'
    X.ROW.write_json(path, dict(single_step=summary, failures=failures))
    result = json.loads(path.read_text())
    assert result['numeric_valid']
    assert result['single_step']['zgz_local']['n_packets'] == 44
