#!/usr/bin/env python3
"""CPU-only arithmetic on immutable XDL predictions; no model import/inference.

Run from the repository root:
  python docs/assets/s37_diagnosis_20261014/recompute_clean_bias.py
The JSON is deterministic for identical input bytes and software versions. It
contains no wall-clock timestamp, elapsed time, absolute path or random sample.
Only the new JSON beside this script is written. Existing artifacts are read-only.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()
DEST = SCRIPT.with_name('clean_bias_recalculation.json')
RUNS = ('xdl_q1_s3407', 'xdl_q0_s3407', 'xdl_n1_s3407', 'xa_c0_s3407')
SEQUENCES = ('zgz_global', 'zgz_local')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identify(path, role):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path),
            'bytes': path.stat().st_size, 'role': role}


def meta51(path):
    # Exact storage/float32 conversion used by semkine/dataset.py:_read_meta51.
    with path.open('rb') as f:
        (n,) = struct.unpack('<i', f.read(4))
        assert n == 51
        dtype = np.dtype([('data', np.float64, n), ('m0', np.uint8), ('m1', np.uint8)])
        a = np.fromfile(f, dtype=dtype)
    assert np.all(a['m0'] == 4) and np.all(a['m1'] == 13)
    return a['data'].astype(np.float32)


def describe(a):
    a = np.asarray(a, np.float64)
    assert np.isfinite(a).all() and len(a)
    return {'n': len(a), 'mean': np.mean(a, axis=0).tolist(),
            'median': np.median(a, axis=0).tolist()}


def xyz(state):
    return np.asarray(state, np.float64)[:3]


def key(row):
    return row['seq'], row['packet_id'], row['spec']['id']


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    inputs = [identify(ROOT / 'outputs/xdl/packet_bank.json', 'packet selection and clock provenance')]
    bank_sha = inputs[0]['sha256']
    gt = {}
    for seq in SEQUENCES:
        meta = ROOT / f'data/hand_data51/val/{seq}.meta'
        aux = ROOT / f'data/hand_data51/val/{seq}_aux.npz'
        inputs.extend([identify(meta, 'GT state at actual end_ms / previous_end_ms'),
                       identify(aux, 'beta/camera provenance only; no FK or camera calculation')])
        gt[seq] = meta51(meta)

    results = {}
    for run in RUNS:
        raw = ROOT / f'outputs/xdl/diagnostics/{run}/bank.pt.gz'
        inputs.append(identify(raw, 'existing saved predictions; no new inference'))
        with gzip.open(raw, 'rb') as f:
            artifact = torch.load(f, map_location='cpu', weights_only=False)
        assert artifact['metadata']['bank_sha256'] == bank_sha
        data = artifact['data']
        rows = [r for r in data['rows'] if r['seq'] in SEQUENCES]
        lookup = {key(r): r for r in rows}
        assert len(lookup) == len(rows)
        results[run] = {}
        for seq in SEQUENCES:
            clean = sorted((r for r in rows if r['seq'] == seq and r['spec']['block'] == 'clean'),
                           key=lambda r: r['packet_id'])
            assert len(clean) == 24
            by_packet = {r['packet_id']: r for r in clean}
            errors, holds, clean_packets = [], [], []
            for r in clean:
                target = gt[seq][r['clock']['end_ms']]
                previous_gt = gt[seq][r['clock']['previous_end_ms']]
                assert np.array_equal(r['initial'], previous_gt)
                e = (xyz(r['prediction']) - xyz(target)) * 1000
                h = (xyz(r['initial']) - xyz(target)) * 1000
                errors.append(e)
                holds.append(h)
                clean_packets.append({'packet_id': r['packet_id'], 'clock': r['clock'],
                                      'signed_error_xyz_mm': e.tolist(), 'hold_error_xyz_mm': h.tolist(),
                                      'prediction_error_mm': float(np.linalg.norm(e)),
                                      'hold_error_mm': float(np.linalg.norm(h))})
            errors, holds = np.array(errors), np.array(holds)
            output_norm, hold_norm = np.linalg.norm(errors, axis=1), np.linalg.norm(holds, axis=1)
            result = {
                'clean': {'signed_error_xyz_mm': describe(errors), 'hold_signed_error_xyz_mm': describe(holds),
                          'prediction_error_mm': describe(output_norm), 'hold_error_mm': describe(hold_norm),
                          'improvement_over_hold_mm': describe(hold_norm - output_norm),
                          'n_strictly_better_than_hold': int(np.sum(output_norm < hold_norm)),
                          'z_fraction_of_total_squared_error': float(np.sum(errors[:, 2] ** 2) / np.sum(errors ** 2)),
                          'packets': clean_packets}, 'translation_perturbation': {}, 'wrong_events': {}}
            for scale in ('small', 'large'):
                selected = [r for r in rows if r['seq'] == seq and r['spec']['block'] == 'translation'
                            and r['spec']['scale'] == scale]
                assert len(selected) == 144
                b_rows, r_rows, e_rows, h_rows = [], [], [], []
                for r in selected:
                    c = by_packet[r['packet_id']]
                    assert r['clock'] == c['clock'] and r['event_sha256'] == c['event_sha256']
                    target = gt[seq][r['clock']['end_ms']]
                    b_rows.append((xyz(c['prediction']) - xyz(target)) * 1000)
                    r_rows.append((xyz(r['prediction']) - xyz(c['prediction'])) * 1000)
                    e_rows.append((xyz(r['prediction']) - xyz(target)) * 1000)
                    h_rows.append((xyz(r['initial']) - xyz(target)) * 1000)
                b, response, e, h = map(np.asarray, (b_rows, r_rows, e_rows, h_rows))
                improvement = np.linalg.norm(h, axis=1) - np.linalg.norm(e, axis=1)
                result['translation_perturbation'][scale] = {
                    'n': len(selected), 'axes': [0, 1, 2], 'signs': [-1, 1],
                    'amplitude_mm': 5 if scale == 'small' else 50,
                    'bias_squared_mm2': describe(np.sum(b * b, axis=1)),
                    'response_squared_mm2': describe(np.sum(response * response, axis=1)),
                    'twice_bias_response_inner_product_mm2': describe(2 * np.sum(b * response, axis=1)),
                    'error_squared_mm2': describe(np.sum(e * e, axis=1)),
                    'identity_max_abs_component_residual_mm': float(np.max(np.abs(e - b - response))),
                    'improvement_over_hold_mm': describe(improvement),
                    'n_strictly_better_than_hold': int(np.sum(improvement > 0))}
            for mode in ('swap', 'mirror'):
                controls = [r for r in data['controls'] if r['seq'] == seq and r['mode'] == mode]
                assert len(controls) == 240
                advantage = []
                for wrong in controls:
                    real = lookup[key(wrong)]
                    assert wrong['clock'] == real['clock'] and np.array_equal(wrong['initial'], real['initial'])
                    target = xyz(gt[seq][wrong['clock']['end_ms']])
                    advantage.append((np.linalg.norm(xyz(wrong['prediction']) - target)
                                      - np.linalg.norm(xyz(real['prediction']) - target)) * 1000)
                result['wrong_events'][mode] = {
                    'wrong_error_minus_real_error_mm': describe(advantage),
                    'n_real_strictly_better': int(np.sum(np.asarray(advantage) > 0)),
                    'scope': 'Existing 240 matched controls per sequence/mode, including multiple state blocks; not natural-rollout interventions.'}
            results[run][seq] = result

    out = {
        'study': 'S37 diagnosis: CPU recalculation of existing XDL bank outputs',
        'run_command_from_repository_root': 'python docs/assets/s37_diagnosis_20261014/recompute_clean_bias.py',
        'script': identify(SCRIPT, 'recalculation implementation'),
        'software': {'numpy': np.__version__, 'torch': torch.__version__},
        'reproducibility': 'No timestamp/random sampling/model import/FK/inference/training. JSON has sorted keys. Same input bytes and software produce identical output; software upgrades may change version strings or floating arithmetic. Existing inputs are read-only.',
        'selection': 'All zgz clean rows (24 per sequence/run), all pure translation probes (144 per sequence/run/scale), all existing swap/mirror controls (240 per sequence/run/mode). Matched on sequence/packet_id/spec.id, with clocks and states asserted.',
        'definitions': {
            'coordinates': 'Camera-frame MANO transl x,y,z; positive z is increasing camera depth. Source states are meters, converted to mm after float64 subtraction.',
            'clean': 'initial equals GT at previous_end_ms exactly; target is GT at end_ms, so hold includes actual inter-frame motion.',
            'wrist_equivalence': 'With fixed beta, wrist J0(beta)+transl implies wrist-error vector equals transl-error vector algebraically. No FK roundoff or vertex alignment is used here.',
            'e': 'e = 1000 * (pred_delta[:3] - GT_end[:3])',
            'b': 'b = 1000 * (pred_clean[:3] - GT_end[:3])',
            'r': 'r = 1000 * (pred_delta[:3] - pred_clean[:3]); e = b + r exactly up to floating arithmetic.',
            'squared': '||e||^2 = ||b||^2 + ||r||^2 + 2 <b,r>; terms measured in mm^2. b is repeated for each of 6 perturbations in a scale.',
            'hold': 'h = 1000 * (initial[:3] - GT_end[:3]); improvement_over_hold_mm = ||h|| - ||e||; positive is better.',
            'wrong_events': '||pred_wrong[:3]-GT_end[:3]|| - ||pred_real[:3]-GT_end[:3]||, in mm; positive means real events better.',
            'aggregation': 'Unweighted mean and NumPy median; strict improvement uses >0 with no tolerance. Samples share packets and are not independent sequence replicates.',
            'rho_limitation': 'Historical paired rho compares disturbed output with clean output, not with GT. These recalculations measure actual target error separately.'},
        'inputs': inputs, 'results': results,
        'limitations': ['Single seed and 24 fixed clean packets per sequence; not a new full-sequence accuracy result.',
                        'N1 is separately trained without observation features; it is not a same-weight test-time ablation.',
                        'Old C0 and XDL differ in multiple architecture/training choices; clean bias differences do not identify a unique cause.',
                        'Swap/mirror are the existing interventions; neither establishes event-guided recovery on the current model own natural errors.']}
    DEST.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(DEST.relative_to(ROOT).as_posix())
    for run, seqs in results.items():
        print(run, {s: {'n': x['clean']['signed_error_xyz_mm']['n'],
                       'mean_xyz_mm': x['clean']['signed_error_xyz_mm']['mean'],
                       'output_norm_mean_mm': x['clean']['prediction_error_mm']['mean'],
                       'hold_norm_mean_mm': x['clean']['hold_error_mm']['mean']}
                    for s, x in seqs.items()})


if __name__ == '__main__':
    main()
