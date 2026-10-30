"""Failure-oriented fixtures for the r2 analysis reader, with no model/GPU work."""
import copy
import importlib.util
from pathlib import Path

import pytest
import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'tools/xkf_repair/assess.py'
spec = importlib.util.spec_from_file_location('repair_assess_under_test', PATH)
A = importlib.util.module_from_spec(spec)
spec.loader.exec_module(A)


def row(seed):
    accuracy = {group: {metric: float(seed - 3400) for metric in A.ACCURACY}
                for group in ('overall', 'local', 'global')}
    accuracy.update(n_frames=2590, step=2200, sequences=['zgz_local', 'zgz_global'])
    return {
        'protocol': {'formal': False, 'exact_optimizer_step': 2200, 'seeds': [seed], 'n_seeds': 1,
                     'split': 'val_core', 'step_ms': 50, 'window_ms': 50, 'direct_window_ms': 200,
                     'noise_rng_seed': 0, 'noise_scale': 1., 'tf32': False},
        'per_seed': {str(seed): accuracy},
        'two_seed_mean': {g: accuracy[g] for g in ('overall', 'local', 'global')},
        'finite_subset_reduction': False, 'numeric_status': 'PASS', 'numeric_valid': True,
        'latency_ms_scaled_full1p75': 4., 'macs_forward_packet': 1e9, 'params_total': 1e6,
        'latency_context_audit': {'actual_forward': True},
    }


def guards_fixture():
    rollouts = {arm: {} for arm in A.ARMS}
    readiness = {arm: {} for arm in A.ARMS}
    for arm in A.ARMS:
        for seed in A.SEEDS:
            readiness[arm][seed] = {'status': 'PASS'}
            rollouts[arm][seed] = {}
            for cohort, scope in [('local_bank', 'all_observed'), ('local_cold', 'full_horizon'), ('global_bank', 'all_observed')]:
                for mode in ('fused', 'hold'):
                    rollouts[arm][seed][(cohort, 'clean', mode, scope)] = {'pooled_frames': {'abs_mpjpe_mm': 10.}}
    return rollouts, readiness


def test_one_bad_seed_cannot_hide_in_an_improved_mean():
    rollouts, readiness = guards_fixture()
    for seed, value in zip(A.SEEDS, [1., 1., 10.1]):
        rollouts['repair'][seed][('local_bank', 'clean', 'fused', 'all_observed')]['pooled_frames']['abs_mpjpe_mm'] = value
    result = A.expanded_guards(rollouts, readiness)
    assert result['status'] == 'FAIL'
    assert result['checks']['eachseed_localbank_clean_fused_abs_not_worse_than_hold'] is False
    assert result['per_seed']['3409']['localbank_clean']['pass'] is False


def test_cold_guard_uses_uncensored_full_horizon_and_global_is_relative():
    rollouts, readiness = guards_fixture()
    rollouts['repair'][3407][('local_cold', 'clean', 'fused', 'all_observed')] = {'pooled_frames': {'abs_mpjpe_mm': 1.}}
    rollouts['repair'][3407][('local_cold', 'clean', 'fused', 'full_horizon')]['pooled_frames']['abs_mpjpe_mm'] = 10.01
    rollouts['control'][3408][('global_bank', 'clean', 'fused', 'all_observed')]['pooled_frames']['abs_mpjpe_mm'] = 5.
    result = A.expanded_guards(rollouts, readiness)
    assert result['per_seed']['3407']['localcold_fullhorizon_clean']['pass'] is False
    assert result['per_seed']['3408']['globalbank_clean']['limit'] == 5.5
    assert result['per_seed']['3408']['globalbank_clean']['pass'] is False


def test_debug_readiness_remains_required_even_if_expanded_metrics_pass():
    rollouts, readiness = guards_fixture()
    readiness['repair'][3408]['status'] = 'FAIL'
    result = A.expanded_guards(rollouts, readiness)
    assert result['status'] == 'FAIL'
    assert result['checks']['allthree_original_debug_readiness'] is False


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), True])
def test_nonfinite_guard_cannot_pass(value):
    rollouts, readiness = guards_fixture()
    rollouts['repair'][3407][('local_bank', 'clean', 'fused', 'all_observed')]['pooled_frames']['abs_mpjpe_mm'] = value
    with pytest.raises(ValueError, match='nonfinite'):
        A.expanded_guards(rollouts, readiness)


def test_merge_requires_three_exact_debug_seeds_and_keeps_scope_false():
    rows = {s: row(s) for s in A.SEEDS}
    with pytest.raises(ValueError, match='all three'):
        A.merge_main_rows({3407: rows[3407], 3408: rows[3408]}, 'repair', {})
    merged = A.merge_main_rows(rows, 'repair', {'unit_fixture': True})
    assert merged['protocol']['n_seeds'] == 3
    assert merged['protocol']['formal'] is False
    assert merged['automatic_full_training'] is False
    assert merged['two_seed_mean']['overall']['mpjpe_ra_mm'] == 8.


@pytest.mark.parametrize('mutation', ['censored', 'formal', 'wrong_step', 'null_accuracy', 'subset'])
def test_incomplete_or_invalid_row_never_merges(mutation):
    rows = {s: row(s) for s in A.SEEDS}
    bad = rows[3409]
    if mutation == 'censored':
        bad['per_seed']['3409']['n_frames'] = 2589
    elif mutation == 'formal':
        bad['protocol']['formal'] = True
    elif mutation == 'wrong_step':
        bad['per_seed']['3409']['step'] = 2199
    elif mutation == 'null_accuracy':
        bad['per_seed']['3409']['overall']['mpjpe_ra_mm'] = None
    else:
        bad['finite_subset_reduction'] = True
    with pytest.raises(ValueError):
        A.merge_main_rows(rows, 'repair', {})


def test_actual_lr_and_restore_clock_are_checked_not_only_config():
    record = {'global_step': 2000, 'rank': 0, 'phase': 'after_restore_before_first_update',
              'configured_lr': .0004, 'actual_after': [[.0004]], 'adam_state_and_scheduler_clock_preserved': True,
              'schedulers': [{'last_epoch': 2000, 'base_lrs': [.0004], 'last_lr': [.0004]}]}
    A.validate_lr(record, 0)
    bad = copy.deepcopy(record)
    bad['actual_after'] = [[.004]]
    with pytest.raises(ValueError, match='actual restored LR'):
        A.validate_lr(bad, 0)
    bad = copy.deepcopy(record)
    bad['schedulers'][0]['last_epoch'] = 0
    with pytest.raises(ValueError, match='scheduler clock'):
        A.validate_lr(bad, 0)


def test_missing_artifact_is_incomplete_and_never_authorizes_training(tmp_path):
    result = A.Assessment(tmp_path, tmp_path).run()
    assert result['status'] == 'INCOMPLETE'
    assert result['debug_readiness_safeguards']['eligible_for_decision'] is False
    assert result['automatic_full_training'] is False
    assert result['full_training_authorized_by_this_assessment'] is False
    assert result['merged_main_rows'] == {}


def test_inconsistent_manifest_is_invalid_not_a_missing_result(tmp_path):
    A.write_json(tmp_path / 'protocol.json', {'study': 'OTHER'})
    result = A.Assessment(tmp_path, tmp_path).run()
    assert result['status'] == 'INVALID'
    assert result['debug_readiness_safeguards']['eligible_for_decision'] is False


def test_raw_reduction_comparison_rejects_a_hidden_bad_frame():
    with pytest.raises(ValueError, match='numeric mismatch'):
        A.assert_close({'mean': 1.}, {'mean': 1.001}, 'all_raw_frames')
    with pytest.raises(ValueError, match='numeric mismatch'):
        A.assert_close({'mean': None}, {'mean': 1.}, 'all_raw_frames')


@pytest.mark.parametrize('defect', ['duplicate_sequence', 'changed_aggregate'])
def test_main_row_recomputes_complete_raw_sequence_aggregation(tmp_path, defect):
    value = row(3407)
    accuracy = value['per_seed']['3407']
    accuracy['per_seq'] = {name: {metric: 7. for metric in A.ACCURACY} for name in ('zgz_local', 'zgz_global')}
    value['provenance'] = {'source_sha256': {'stub': 'source'}}
    raw = {'complete': True, 'numeric_valid': True, 'finite_subset_reduction': False,
           'identity': {}, 'protocol': value['protocol'], 'raw_sequences': [
               {'seq': name, 'n_frames': count, **{m: 7. for m in A.ACCURACY}}
               for name, count in [('zgz_local', 1204), ('zgz_global', 1386)]]}
    if defect == 'duplicate_sequence':
        raw['raw_sequences'][1] = copy.deepcopy(raw['raw_sequences'][0])
    else:
        accuracy['local']['mpjpe_ra_mm'] = 2.
    class Reader(A.Assessment):
        def read(self, path):
            return raw if str(path).endswith('.raw_accuracy.json') else value
        def identity(self, *args, **kwargs):
            pass
        def sources(self, *args, **kwargs):
            pass
    reader = Reader(tmp_path, tmp_path)
    reader.accuracy_fn = A.extracted_functions(PATH.parents[1] / 'xdl/make_row.py', {'accuracy_aggregate'},
                                              {'math': A.math, 'METRICS': A.ACCURACY})['accuracy_aggregate']
    with pytest.raises(ValueError, match='duplicate/missing|recomputed'):
        reader.main_row('repair', 3407, {'checkpoint_sha256': 'stub'})


def test_direct_probe_rejects_truncated_array_despite_full_success_mask(tmp_path):
    first = tmp_path / 'zgz_local_W50.npz'
    np.savez(first, success=np.ones(1204, dtype=bool), direct_state=np.zeros((1204, 51)),
             truth=np.zeros((1204, 51)), learned__predpose_pred2d__t_error_mm=np.zeros((1, 3)))
    result = {'identity': {}, 'repair_adapter_sha256': {'stub': 'source'}, 'source_sha256': {'stub': 'source'},
              'complete': True, 'partial_smoke': False, 'formal_gate_assessed': False, 'finite_subset_reduction': False,
              'failures': [], 'bank_sha256': 'bank', 'preregistration_sha256': 'stub',
              'sequences': {seq: {str(w): {} for w in (50, 100, 150, 200)} for seq in A.NROWS},
              'raw_files': [{'path': str(tmp_path / f'{seq}_W{w}.npz'), 'sha256': 'stub'}
                            for seq in A.NROWS for w in (50, 100, 150, 200)]}
    class Reader(A.Assessment):
        def read(self, path):
            return result
        def identity(self, *args, **kwargs):
            pass
        def sources(self, *args, **kwargs):
            pass
        def digest(self, *args):
            return 'stub'
    reader = Reader(tmp_path, tmp_path)
    reader.old_schedule = {'bank_sha256': 'bank'}
    with pytest.raises(ValueError, match='truncated/mismatched'):
        reader.direct('repair', 3407, {'checkpoint_sha256': 'stub'})
