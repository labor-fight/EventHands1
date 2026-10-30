#!/usr/bin/env python3
"""Read-only raw-D mechanism audit for the preregistered paired200 experiment.

Metrics are recomputed from every raw prediction; an affine-corrected score can
never replace the uncalibrated training-effect test. Missing runs cannot pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
R2 = ROOT / 'outputs/xkf/repair_20261015_r2'
DEFAULT = ROOT / 'outputs/xkf/fusion_20261015_r3/direct_finetune'
SEEDS = (3407, 3408, 3409)
COUNTS = {'zgz_local': 1204, 'zgz_global': 1386}
INPUT_FIELDS = ('packet_id', 'selected_bank_index', 'end_ms', 'run_id', 'step_in_run',
                'effective_window_ms', 'short_event_count', 'long_event_count',
                'positive_lnes_fraction', 'cold_first3', 'original_debug_clip',
                'truth', 'previous_truth', 'camera_K', 'beta')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def metrics(pred, joints0, truth, truth_joints):
    """Physical wrist depth and state-translation norm, in millimetres."""
    for value in (pred, joints0, truth, truth_joints):
        require(np.isfinite(value).all(), 'nonfinite rows cannot be filtered or averaged')
    error = (np.asarray(pred, dtype=np.float64)[:, :3] - truth[:, :3]) * 1000.
    wrist = (joints0[:, 0] + pred[:, :3] - truth_joints[:, 0]) * 1000.
    return {'n': len(pred), 't_rmse_mm': float(np.sqrt(np.mean(np.sum(error ** 2, -1)))),
            'depth_rmse_mm': float(np.sqrt(np.mean(wrist[:, 2] ** 2))),
            'state_depth_rmse_mm': float(np.sqrt(np.mean(error[:, 2] ** 2))),
            'physical_wrist_rmse_mm': float(np.sqrt(np.mean(np.sum(wrist ** 2, -1)))),
            'physical_depth_bias_mm': float(wrist[:, 2].mean()),
            'finite_subset_reduction': False}


class Audit:
    def __init__(self, directory):
        self.directory = directory
        self.inputs = {}

    def digest(self, path):
        path = Path(path)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            raise FileNotFoundError(path)
        key = str(path.resolve())
        if key not in self.inputs:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b''):
                    h.update(chunk)
            self.inputs[key] = h.hexdigest()
        return self.inputs[key]

    def read(self, path):
        self.digest(path)
        return json.loads(Path(path).read_text())

    def raw(self, path, sequence):
        self.digest(path)
        with np.load(path, allow_pickle=False) as values:
            raw = {key: values[key].copy() for key in values.files}
        n = COUNTS[sequence]
        require(raw['success'].shape == (n,) and bool(raw['success'].all()), 'incomplete or failed full cohort')
        require(raw['direct_state'].shape == (n, 51) and raw['truth'].shape == (n, 51), 'state dimensions differ')
        require(raw['pred_joints0'].shape == raw['truth_joints'].shape == (n, 21, 3), 'joint dimensions differ')
        for key, value in raw.items():
            if value.dtype.kind in 'fc':
                require(np.isfinite(value).all(), f'nonfinite raw field {key}')
        require(np.array_equal(raw['selected_bank_index'], np.arange(n)), 'all chronological packets required')
        return raw

    def parent(self, seed):
        directory = R2 / f'evaluations/repair_s{seed}_step2200_direct'
        result = self.read(directory / 'result.json')
        identity = result['identity']
        require(result['complete'] and not result['partial_smoke'] and not result['failures'], 'parent direct evidence incomplete')
        require(identity['global_step'] == 2200 and identity['seed'] == seed, 'wrong parent step/seed')
        require(identity['checkpoint_sha256'] == self.digest(identity['checkpoint']), 'parent checkpoint SHA differs')
        values = {}
        for sequence in COUNTS:
            path = directory / f'{sequence}_W200.npz'
            raw = self.raw(path, sequence)
            values[sequence] = {'raw': raw, 'path': str(path), 'sha256': self.digest(path),
                                'metrics': metrics(raw['direct_state'], raw['pred_joints0'], raw['truth'], raw['truth_joints'])}
        return {'identity': identity, 'sequences': values}

    def arm(self, seed, arm, parent):
        directory = self.directory / f'calibration_{arm}_s{seed}'
        result = self.read(directory / 'result.json')
        protocol = self.read(directory / 'protocol.json')
        runtime = self.read(directory / 'translation_affine_runtime.json')
        identity = result['identity']
        require(identity == protocol['identity'] == runtime['identity'], 'producer/runtime identity disagreement')
        require(not result['partial_smoke'] and not protocol['partial_smoke'] and not runtime['partial_smoke'], 'smoke cannot pass full mechanism gate')
        require(result['BN_buffers_unchanged'] and result['all_external_rows_success'], 'inference mutation or failed rows')
        require(identity['global_step'] == 2400 and identity['seed'] == seed, 'wrong actual D step/seed')
        clocks = identity['training_clocks']
        require(clocks == {'I_parameter_Adam_step': 2200, 'D_parameter_Adam_step': 2400,
                          'scheduler_and_global_step': 2400, 'all_parameters_at_global_step': False}, 'mixed I/D clocks missing')
        require(identity['checkpoint_sha256'] == self.digest(identity['checkpoint']), 'actual checkpoint SHA differs')
        require(identity['config_sha256'] == self.digest(identity['config']), 'actual config SHA differs')
        finish = self.read(self.directory / f'{arm}_s{seed}' / 'train_finish.json')
        require(finish['status'] == 'DEBUG200_COMPLETE' and finish['global_step'] == 2400, 'run did not finish exact200')
        require(finish['checkpoint_sha256'] == identity['checkpoint_sha256'], 'result binds another arm/checkpoint')
        require(finish['parent_checkpoint_sha256'] == parent['identity']['checkpoint_sha256'], 'wrong paired parent checkpoint')
        require(runtime['checkpoint_sha256'] == identity['checkpoint_sha256'], 'affine checkpoint binding differs')
        require(runtime['fit_rows'] == protocol['fit_rows'] == 2304, 'training fit must retain all2304 rows')
        require(runtime['protocol_sha256'] == self.digest(directory / 'protocol.json'), 'runtime protocol binding differs')
        require(runtime['fit_source_sha256'] == self.digest(directory / 'train_predictions.npz'), 'training prediction SHA differs')
        spec = runtime['spec']
        require(spec['kind'] == 'translation_affine', 'unexpected corrected diagnostic')
        matrix, bias = np.asarray(spec['matrix']), np.asarray(spec['bias'])
        require(matrix.shape == (3, 3) and bias.shape == (3,) and np.isfinite(matrix).all() and np.isfinite(bias).all(), 'invalid affine')
        values = {}
        for sequence in COUNTS:
            path = directory / f'{sequence}_W200.npz'
            raw = self.raw(path, sequence)
            declared = result['external_scores'][sequence]
            require(declared['checkpoint_sha256'] == identity['checkpoint_sha256'] and declared['sha256'] == self.digest(path), 'external prediction/checkpoint binding differs')
            old = parent['sequences'][sequence]['raw']
            for key in INPUT_FIELDS:
                require(np.array_equal(raw[key], old[key]), f'paired evidence input differs {sequence}/{key}')
            require(np.allclose(raw['truth_joints'], old['truth_joints'], atol=2e-7, rtol=0), 'parent/new GT FK differs')
            before = metrics(raw['direct_state'], raw['pred_joints0'], raw['truth'], raw['truth_joints'])
            # Producer labels depth as state t depth; independently score physical
            # wrist here, and compare the producer to its exact stated t metric.
            producer = declared['cohorts']['all']['before']
            require(producer['valid'] and producer['n'] == COUNTS[sequence], 'producer before-D cohort invalid')
            require(np.isclose(producer['t_rmse_mm'], before['t_rmse_mm'], atol=1e-8, rtol=1e-8), 'producer t metric disagrees with raw')
            require(np.isclose(producer['depth_rmse_mm'], before['state_depth_rmse_mm'], atol=1e-8, rtol=1e-8), 'producer state-depth metric disagrees with raw')
            corrected = raw['direct_state'].copy()
            corrected[:, :3] = raw['direct_state'][:, :3] @ matrix.T + bias
            after = metrics(corrected, raw['pred_joints0'], raw['truth'], raw['truth_joints'])
            values[sequence] = {'metrics_uncalibrated': before, 'metrics_after_training_affine': after,
                                'raw_path': str(path), 'raw_sha256': self.digest(path),
                                'packet_id_sha256': hashlib.sha256(raw['packet_id'].tobytes()).hexdigest(),
                                'all_rows_identical_to_parent_inputs': True}
        return {'identity': identity, 'producer_result_sha256': self.digest(directory / 'result.json'), 'sequences': values}

    def run(self):
        protocol = self.read(self.directory / 'paired200_protocol.json')
        gate = protocol['mechanism_gate']
        require(protocol['seeds'] == list(SEEDS) and protocol['arms'] == ['raw', 'aug'], 'paired population changed')
        require(protocol['endpoint_step'] == 2400 and protocol['updates'] == 200, 'paired endpoint changed')
        require(gate['local'] == 'both metrics strictly smaller for raw2400 than BOTH parentrepair2200 and matched aug2400, separately for every seed', 'local threshold changed')
        require(gate['global'] == 'both raw2400 metrics <=1.1 times BOTH references, separately for every seed', 'global threshold changed')
        output = {'kind': 'XKF_DONLY_PAIRED200_MECHANISM_AUDIT', 'formal': False,
                  'automatic_full_training': False, 'protocol_sha256': self.digest(self.directory / 'paired200_protocol.json'),
                  'criteria': gate, 'seeds': {}, 'cells': [], 'missing': [], 'validation_errors': [],
                  'after_affine_scope': 'separate descriptive statistics only; never used in mechanism gate',
                  'producer_before_D_only': True, 'finite_subset_reduction': False}
        for seed in SEEDS:
            try:
                parent = self.parent(seed)
                raw = self.arm(seed, 'raw', parent)
                aug = self.arm(seed, 'aug', parent)
                output['seeds'][str(seed)] = {'parent_identity': parent['identity'], 'raw': raw, 'aug': aug,
                     'parent': {s: {k: v for k, v in value.items() if k != 'raw'} for s, value in parent['sequences'].items()}}
                for sequence in COUNTS:
                    for metric in ('t_rmse_mm', 'depth_rmse_mm'):
                        value = raw['sequences'][sequence]['metrics_uncalibrated'][metric]
                        old = parent['sequences'][sequence]['metrics'][metric]
                        ctl = aug['sequences'][sequence]['metrics_uncalibrated'][metric]
                        local = sequence == 'zgz_local'
                        comparisons = {}
                        for name, ref in (('parent2200', old), ('matched_aug2400', ctl)):
                            limit = ref if local else 1.1 * ref
                            comparisons[name] = {'value': ref, 'limit': limit, 'operator': '<' if local else '<=',
                                                 'ratio': value / ref, 'pass': value < limit if local else value <= limit}
                        output['cells'].append({'seed': seed, 'sequence': sequence, 'n': COUNTS[sequence],
                            'metric': metric, 'raw2400_uncalibrated': value, 'references': comparisons,
                            'pass': all(v['pass'] for v in comparisons.values()),
                            'raw_source_sha256': raw['sequences'][sequence]['raw_sha256'],
                            'aug_source_sha256': aug['sequences'][sequence]['raw_sha256'],
                            'parent_source_sha256': parent['sequences'][sequence]['sha256']})
            except FileNotFoundError as error:
                output['missing'].append({'seed': seed, 'path': str(error)})
            except (ValueError, KeyError, TypeError, OSError) as error:
                output['validation_errors'].append({'seed': seed, 'error': type(error).__name__, 'message': str(error)})
        complete = len(output['seeds']) == 3 and len(output['cells']) == 12 and not output['missing'] and not output['validation_errors']
        output['status'] = 'ASSESSED' if complete else ('INVALID' if output['validation_errors'] else 'INCOMPLETE')
        output['mechanism_gate_status'] = ('PASS' if all(v['pass'] for v in output['cells']) else 'FAIL') if complete else 'NOT_ASSESSED'
        output['failed_cells'] = [v for v in output['cells'] if not v['pass']]
        output['inputs_sha256'] = self.inputs
        output['assessor_sha256'] = self.digest(__file__)
        return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('audit cannot overwrite a historical result')
    result = Audit(args.study).run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'mechanism_gate_status': result['mechanism_gate_status'],
                      'missing': len(result['missing']), 'validation_errors': result['validation_errors']}))
    return 0 if result['status'] == 'ASSESSED' else 2


if __name__ == '__main__':
    raise SystemExit(main())
