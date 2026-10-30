#!/usr/bin/env python3
"""Supplementary unchanged-G1 formula replay on full D2400 measurement arrays.

This is a nonformal debug supplement. It neither replaces the paired mechanism
gate nor assesses G2--G4/readiness. Only the uncalibrated direct state is scored.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tools.xkf_fusion.paired_effect import Audit, DEFAULT, ROOT, SEEDS, COUNTS, require


def wrong_time(endpoints):
    endpoints = np.asarray(endpoints, np.int64)
    require(bool((np.diff(endpoints) > 0).all()), 'chronological unique endpoints required')
    indices = []
    for end in endpoints:
        later = int(np.searchsorted(endpoints, end + 1000))
        earlier = int(np.searchsorted(endpoints, end - 1000, side='right')) - 1
        selected = later if later < len(endpoints) else earlier
        require(selected >= 0 and abs(int(endpoints[selected]) - int(end)) >= 1000, 'no valid wrong-time packet')
        indices.append(selected)
    return np.asarray(indices, np.int64)


def compute(raws):
    pred_depth, true_depth, correct_errors, wrong_errors = [], [], [], []
    cohorts = {}
    for sequence in COUNTS:
        raw = raws[sequence]
        require(bool((raw['short_event_count'] > 0).all()), 'measurement arrays cannot emulate zero-event hold without explicit replay')
        pred = raw['pred_joints0'][:, 0] + raw['direct_state'][:, :3]
        truth = raw['truth_joints'][:, 0]
        indices = wrong_time(raw['end_ms'])
        correct = np.linalg.norm(pred - truth, axis=-1) * 1000.
        wrong = np.linalg.norm(pred[indices] - truth, axis=-1) * 1000.
        root = np.rad2deg((Rotation.from_rotvec(raw['truth'][:, 3:6]).inv()
                          * Rotation.from_rotvec(raw['direct_state'][:, 3:6])).magnitude())
        require(all(np.isfinite(v).all() for v in (pred, truth, correct, wrong, root)), 'nonfinite G1 row')
        cohorts[sequence] = {'n': len(pred), 'z_bias_mm': float(((pred[:, 2] - truth[:, 2]) * 1000).mean()),
            'root_mean_deg': float(root.mean()), 'wrist_mean_mm': float(correct.mean()),
            'wrong_time_wrist_mean_mm': float(wrong.mean()),
            'wrong_time_min_gap_ms': int(np.abs(raw['end_ms'][indices] - raw['end_ms']).min()),
            'wrong_time_indices': indices.tolist()}
        pred_depth.append(pred[:, 2]); true_depth.append(truth[:, 2])
        correct_errors.append(correct); wrong_errors.append(wrong)
    x, y = np.concatenate(true_depth), np.concatenate(pred_depth)
    x, y = x - x.mean(), y - y.mean()
    require(float(x @ x) > 1e-18, 'depth regression has zero support')
    slope = float(x @ y / (x @ x))
    correct = float(np.concatenate(correct_errors).mean())
    wrong = float(np.concatenate(wrong_errors).mean())
    require(correct > 0., 'wrong-time ratio has zero denominator')
    local = cohorts['zgz_local']
    checks = {'depth_slope': .8 <= slope <= 1.2, 'local_bias': abs(local['z_bias_mm']) <= 15.,
              'local_root': local['root_mean_deg'] <= 10., 'wrong_time': wrong >= 2. * correct}
    return {'formula_checks_status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks,
            'n_packets': len(np.concatenate(pred_depth)), 'pooled_depth_slope': slope,
            'wrist_mean_mm': correct, 'wrong_time_wrist_mean_mm': wrong, 'wrong_time_ratio': wrong / correct,
            'local_signed_depth_bias_mm': local['z_bias_mm'], 'local_root_mean_deg': local['root_mean_deg'],
            'cohorts': cohorts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    audit = Audit(args.study.resolve())
    paired = audit.read(args.study / 'paired_effect.json')
    require(paired['status'] == 'ASSESSED' and not paired['validation_errors'], 'full paired evidence must be verified first')
    formula = ROOT / 'tools/xkf/diagnostics.py'
    result = {'kind': 'XKF_D2400_G1_DEBUG_SUPPLEMENT', 'formal': False, 'automatic_full_training': False,
        'scope': 'uncalibrated D measurement formula replay only; does not establish G2-G4, formal success, or candidate readiness',
        'formula_source_sha256': audit.digest(formula), 'paired_effect_sha256': audit.digest(args.study / 'paired_effect.json'),
        'wrong_time_rule': 'nearest later packet >=1000ms, else nearest earlier >=1000ms, same sequence; frozen original rule',
        'input_equivalence': 'all packets nonempty, same sequence beta/camera, D independent of prev and target; wrong-time uses corresponding raw-event/causal-LNES direct prediction',
        'finite_subset_reduction': False, 'arms': {}}
    for arm in ('raw', 'aug'):
        result['arms'][arm] = {}
        for seed in SEEDS:
            directory = args.study / f'calibration_{arm}_s{seed}'
            identity = audit.read(directory / 'result.json')['identity']
            require(identity == paired['seeds'][str(seed)][arm]['identity'], 'paired identity differs')
            raws = {}
            for sequence in COUNTS:
                path = directory / f'{sequence}_W200.npz'
                expected = paired['seeds'][str(seed)][arm]['sequences'][sequence]['raw_sha256']
                require(audit.digest(path) == expected, 'paired raw SHA differs')
                raws[sequence] = audit.raw(path, sequence)
            result['arms'][arm][str(seed)] = {'identity': identity, **compute(raws)}
    result['input_sha256'] = audit.inputs
    result['assessor_sha256'] = audit.digest(__file__)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({arm: {seed: {key: value[key] for key in ('formula_checks_status', 'pooled_depth_slope',
        'local_signed_depth_bias_mm', 'local_root_mean_deg', 'wrong_time_ratio')} for seed, value in values.items()}
        for arm, values in result['arms'].items()}))


if __name__ == '__main__':
    main()
