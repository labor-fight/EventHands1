#!/usr/bin/env python3
"""CPU-only pooled D calibration with blocked cross-fitting and fixed clip specs.

GT is used only in fitting/scoring. Apply functions consume D outputs and visible
event statistics, never previous states, targets or sequence identity. The two
zgz sequences are development data; this is not new-identity generalization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (3407, 3408, 3409)
SEQUENCES = ('zgz_local', 'zgz_global')
KINDS = ('identity', 'translation_bias', 'translation_diagonal', 'translation_affine',
         'wrist_depth_affine', 'wrist_inverse_depth_affine', 'event_linear')
RIDGE = .001
FEATURES = ('t_x', 't_y', 't_z', 'log1p_count50', 'log_occupancy_plus_1e-5', 'log_window_ms')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def array_hash(value):
    x = np.ascontiguousarray(value)
    return hashlib.sha256(str((x.dtype, x.shape)).encode() + x.tobytes()).hexdigest()


def ridge_fit(x, y):
    mean, scale = x.mean(0), x.std(0)
    scale = np.maximum(scale, 1e-8)
    z = (x - mean) / scale
    bias = y.mean(0)
    weights = np.linalg.solve(z.T @ z + np.eye(z.shape[1]) * len(z) * RIDGE, z.T @ (y - bias))
    return mean, scale, weights, bias


def rotation_bias(pred, truth):
    residual = Rotation.from_rotvec(truth[:, 3:6]).as_matrix() @ Rotation.from_rotvec(pred[:, 3:6]).as_matrix().transpose(0, 2, 1)
    u, _, vt = np.linalg.svd(residual.mean(0))
    return u @ np.diag([1., 1., np.linalg.det(u @ vt)]) @ vt


def fit(kind, pred, joints0, truth, event_features, rotate=False):
    """Pooled fit; deliberately no sequence argument or per-sequence parameters."""
    t, target = pred[:, :3], truth[:, :3]
    residual = target - t
    spec = {'kind': kind, 'fit_rows': len(pred)}
    if kind == 'identity':
        pass
    elif kind == 'translation_bias':
        spec.update(kind='translation_affine', matrix=np.eye(3).tolist(), bias=residual.mean(0).tolist())
    elif kind in ('translation_diagonal', 'translation_affine'):
        matrix, intercept = np.eye(3), np.zeros(3)
        if kind == 'translation_diagonal':
            for k in range(3):
                mean, scale, weight, bias = ridge_fit(t[:, k:k + 1], residual[:, k:k + 1])
                slope = float((weight / scale[:, None])[0, 0])
                matrix[k, k] += slope
                intercept[k] = float(bias[0] - mean[0] * slope)
        else:
            mean, scale, weights, bias = ridge_fit(t, residual)
            affine = (weights / scale[:, None]).T
            matrix += affine
            intercept = bias - affine @ mean
        spec.update(kind='translation_affine', matrix=matrix.tolist(), bias=intercept.tolist())
    elif kind in ('wrist_depth_affine', 'wrist_inverse_depth_affine'):
        z = t[:, 2] + joints0[:, 0, 2]
        target_z = target[:, 2] + joints0[:, 0, 2]
        if kind == 'wrist_inverse_depth_affine':
            z, target_z = 1 / z, 1 / target_z
        mean, scale, weight, bias = ridge_fit(z[:, None], (target_z - z)[:, None])
        slope = float((weight / scale[:, None])[0, 0])
        spec.update(scale=1 + slope, offset=float(bias[0] - mean[0] * slope))
    elif kind == 'event_linear':
        mean, scale, weights, bias = ridge_fit(event_features, residual)
        spec.update(feature_names=FEATURES, mean=mean.tolist(), scale=scale.tolist(),
                    weights=weights.tolist(), bias=bias.tolist())
    else:
        raise ValueError(kind)
    if rotate:
        spec['root_left_rotation'] = rotation_bias(pred, truth).tolist()
    return spec


def apply(pred, joints0, event_features, spec):
    """Target/previous-state/sequence-free calibration application."""
    out, j0 = pred.copy(), joints0.copy()
    kind = spec['kind']
    if kind == 'translation_affine':
        out[:, :3] = pred[:, :3] @ np.asarray(spec['matrix']).T + np.asarray(spec['bias'])
    elif kind in ('wrist_depth_affine', 'wrist_inverse_depth_affine'):
        wrist = pred[:, :3] + joints0[:, 0]
        depth = wrist[:, 2]
        corrected = (spec['scale'] * depth + spec['offset'] if kind == 'wrist_depth_affine'
                     else 1 / (spec['scale'] / depth + spec['offset']))
        out[:, :3] = wrist * (corrected / depth)[:, None] - joints0[:, 0]
    elif kind == 'event_linear':
        x = (event_features - spec['mean']) / spec['scale']
        out[:, :3] = pred[:, :3] + x @ np.asarray(spec['weights']) + spec['bias']
    elif kind != 'identity':
        raise ValueError(kind)
    if 'root_left_rotation' in spec:
        bias = np.asarray(spec['root_left_rotation'])
        out[:, 3:6] = Rotation.from_matrix(bias @ Rotation.from_rotvec(pred[:, 3:6]).as_matrix()).as_rotvec()
        j0 = (joints0 - joints0[:, :1]) @ bias.T + joints0[:, :1]
    return out, j0


def score(pred, joints0, truth, truth_joints):
    joints = joints0 + pred[:, None, :3]
    valid = np.isfinite(pred).all() and np.isfinite(joints).all() and (joints[:, :, 2] > .001).all()
    if not valid:
        return {'n': len(pred), 'valid': False, 'finite_subset_reduction': False}
    error = (pred[:, :3] - truth[:, :3]) * 1000
    relative = (joints - joints[:, :1]) - (truth_joints - truth_joints[:, :1])
    root = Rotation.from_rotvec(truth[:, 3:6]).inv() * Rotation.from_rotvec(pred[:, 3:6])
    return {'n': len(pred), 'valid': True, 't_rmse_mm': float(np.sqrt(np.mean(np.sum(error ** 2, -1)))),
            'depth_rmse_mm': float(np.sqrt(np.mean(error[:, 2] ** 2))), 'depth_bias_mm': float(error[:, 2].mean()),
            'abs_mpjpe_mm': float(np.linalg.norm(joints - truth_joints, axis=-1).mean() * 1000),
            'ra_mpjpe_mm': float(np.linalg.norm(relative, axis=-1).mean() * 1000),
            'root_deg': float(np.degrees(root.magnitude()).mean())}


def embargo_mask(sequence, end_ms, train, excluded, embargo_ms):
    keep = train.copy()
    for seq in np.unique(sequence):
        test_times = end_ms[excluded & (sequence == seq)]
        candidate = np.flatnonzero(keep & (sequence == seq))
        if len(test_times) and len(candidate):
            distance = np.abs(end_ms[candidate, None] - test_times[None, :]).min(1)
            keep[candidate[distance < embargo_ms]] = False
    return keep


def load_seed(args, seed):
    fields = ('direct_state', 'pred_joints0', 'truth', 'truth_joints', 'end_ms', 'packet_id',
              'short_event_count', 'positive_lnes_fraction', 'effective_window_ms', 'kp_visible')
    joined = {k: [] for k in fields}
    paths, sequence = [], []
    for seq, count in zip(SEQUENCES, (1204, 1386)):
        path = args.input_root / f'repair_s{seed}_step2200_direct' / f'{seq}_W200.npz'
        with np.load(path, allow_pickle=False) as raw:
            assert raw['success'].shape == (count,) and raw['success'].all()
            for field in fields:
                joined[field].append(raw[field].copy())
            sequence.extend([seq] * count)
        paths.append({'path': str(path.resolve()), 'sha256': sha(path)})
    data = {k: np.concatenate(v) for k, v in joined.items()}
    data['sequence'] = np.asarray(sequence)
    data['features'] = np.column_stack([data['direct_state'][:, :3], np.log1p(data['short_event_count']),
                                       np.log(data['positive_lnes_fraction'] + 1e-5), np.log(data['effective_window_ms'])])
    for key in ('direct_state', 'pred_joints0', 'truth', 'truth_joints', 'features'):
        assert np.isfinite(data[key]).all(), key
    return data, paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=ROOT / 'outputs/xkf/repair_20261015_r2/evaluations')
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/xkf/fusion_20261015_r3/calibration')
    parser.add_argument('--block-ms', type=int, default=5000)
    parser.add_argument('--embargo-ms', type=int, default=200)
    args = parser.parse_args()
    if (args.output / 'result.json').exists():
        raise FileExistsError(args.output / 'result.json')
    schedule_path = ROOT / 'outputs/xkf/local_recovery_20261015/rollout/protocol.json'
    schedule = json.loads(schedule_path.read_text())['clips']
    protocol = {'kind': 'XKF_R3_D_CALIBRATION_CROSSFIT', 'formal': False, 'seeds': SEEDS,
                'input_identity': 'r2 repair step2200; W200 all zgz local/global packets',
                'development_scope': 'same zgz identity used for development and test; no new-identity generalization claim',
                'block_ms': args.block_ms, 'fold_rule': 'floor(end_ms / block_ms) % 2; both sequences pooled in each fit',
                'embargo_ms': args.embargo_ms, 'embargo_rule': 'exclude training rows within <200ms of held-out endpoints in same sequence',
                'candidate_kinds': KINDS, 'root_left_rotation': [False, True], 'ridge': RIDGE,
                'features': FEATURES, 'no_sequence_specific_parameters': True, 'no_previous_state_features': True,
                'no_target_at_apply': True, 'all_rows_preserved': True, 'clip_rule': 'fixed spec throughout clip; base opposite first-point fold plus exclude entire same-sequence blocks touched by clip, with W200 embargo',
                'source_sha256': sha(__file__), 'schedule_sha256': sha(schedule_path)}
    protocol['input_checkpoints'] = {}
    protocol['input_raw_sha256'] = {}
    for seed in SEEDS:
        directory = args.input_root / f'repair_s{seed}_step2200_direct'
        identity = json.loads((directory / 'result.json').read_text())['identity']
        assert identity['seed'] == seed and identity['global_step'] == 2200
        protocol['input_checkpoints'][str(seed)] = identity
        for sequence in SEQUENCES:
            path = directory / f'{sequence}_W200.npz'
            protocol['input_raw_sha256'][str(path.resolve())] = sha(path)
    write(args.output / 'preregistration.json', protocol)
    result = {'kind': protocol['kind'], 'protocol_sha256': sha(args.output / 'preregistration.json'),
              'formal': False, 'development_calibration': True, 'actual_rollout_required': True, 'seeds': {}}
    for seed in SEEDS:
        data, paths = load_seed(args, seed)
        assert all(protocol['input_raw_sha256'][item['path']] == item['sha256'] for item in paths)
        p, j, y, gt = (data[k] for k in ('direct_state', 'pred_joints0', 'truth', 'truth_joints'))
        feature, seq, end = (data[k] for k in ('features', 'sequence', 'end_ms'))
        fold = (end // args.block_ms) % 2
        block = end // args.block_ms
        masks = [embargo_mask(seq, end, fold != f, fold == f, args.embargo_ms) for f in (0, 1)]
        partitions = {}
        def partition(train):
            digest = array_hash(train)
            if digest not in partitions:
                partitions[digest] = {'train_rows': int(train.sum()), 'train_packet_ids': data['packet_id'][train].tolist(),
                    'training_time_blocks': {sequence: sorted(set(block[train & (seq == sequence)].tolist())) for sequence in SEQUENCES}}
            return digest
        for mask in masks:
            partition(mask)
        clip_masks = {}
        for clip in schedule:
            seqrows = np.flatnonzero(seq == clip['sequence'])
            rows = seqrows[np.asarray(clip['row_indices'])]
            basefold = int(fold[rows[0]])
            touched = sorted(set(block[rows].tolist()))
            excluded = (seq == clip['sequence']) & np.isin(block, touched)
            train = embargo_mask(seq, end, masks[basefold] & ~excluded, excluded, args.embargo_ms)
            assert not train[rows].any()
            partition(train)
            clip_masks[clip['clip_id']] = (rows, basefold, touched, train)
        entry = {'inputs': paths, 'n': len(p), 'fold_sizes': [int((fold == f).sum()) for f in (0, 1)],
                 'train_sizes_after_embargo': [int(m.sum()) for m in masks], 'candidates': {}, 'visibility': {}}
        for name in SEQUENCES:
            visible = data['kp_visible'][seq == name]
            entry['visibility'][name] = {'rows_any_outside': int((visible.mean(1) < 1).sum()), 'rows': len(visible)}
        saved = {'sequence': seq, 'packet_id': data['packet_id'], 'end_ms': end, 'fold': fold, 'truth': y,
                 'truth_joints': gt, 'input_direct_state': p}
        clip_output = {'seed': seed, 'protocol_sha256': result['protocol_sha256'], 'inputs': paths,
                       'scope': 'per-clip fixed held-out calibration; no scored clip GT used in corresponding fit', 'candidates': {}}
        for kind in KINDS:
            for rotate in (False, True):
                name = kind + ('_root' if rotate else '')
                out, oj = np.empty_like(p), np.empty_like(j)
                specs = []
                for f, train in enumerate(masks):
                    spec = fit(kind, p[train], j[train], y[train], feature[train], rotate)
                    test = fold == f
                    out[test], oj[test] = apply(p[test], j[test], feature[test], spec)
                    specs.append({'held_out_fold': f, 'train_mask_sha256': array_hash(train),
                                  'held_out_packet_sha256': array_hash(data['packet_id'][test]), 'spec': spec})
                metrics = {'overall': score(out, oj, y, gt)}
                for sequence in SEQUENCES:
                    m = seq == sequence
                    metrics[sequence] = score(out[m], oj[m], y[m], gt[m])
                    metrics[sequence]['folds'] = {str(f): score(out[m & (fold == f)], oj[m & (fold == f)],
                                                                y[m & (fold == f)], gt[m & (fold == f)]) for f in (0, 1)}
                entry['candidates'][name] = {'crossfit': metrics, 'fold_specs': specs,
                    'full_development_refit_not_crossfit': fit(kind, p, j, y, feature, rotate)}
                saved[name + '__state'] = out
                saved[name + '__joints0'] = oj
                perclip = {}
                for clip in schedule:
                    rows, basefold, touched, train = clip_masks[clip['clip_id']]
                    spec = (specs[basefold]['spec'] if np.array_equal(train, masks[basefold]) else
                            fit(kind, p[train], j[train], y[train], feature[train], rotate))
                    perclip[clip['clip_id']] = {'base_held_out_fold': basefold, 'scored_end_ms': end[rows].tolist(),
                        'scored_packet_ids': data['packet_id'][rows].tolist(), 'train_rows': int(train.sum()),
                        'additional_embargo_rows': int((masks[basefold] & ~train).sum()),
                        'excluded_time_blocks_in_clip_sequence': touched,
                        'train_mask_sha256': array_hash(train), 'spec': spec}
                clip_output['candidates'][name] = perclip
                write(args.output / f'seed{seed}_{name}_clip_specs.json',
                      {'seed': seed, 'candidate': name, 'protocol_sha256': result['protocol_sha256'], 'inputs': paths,
                       'scope': clip_output['scope'], 'clip_specs': {key: value['spec'] for key, value in perclip.items()},
                       'clip_fit_audit': {key: {k: v for k, v in value.items() if k != 'spec'} for key, value in perclip.items()},
                       'fit_partitions': partitions})
        np.savez_compressed(args.output / f'seed{seed}_crossfit.npz', **saved)
        clip_output['fit_partitions'] = partitions
        entry['fit_partitions'] = partitions
        write(args.output / f'seed{seed}_clip_specs.json', clip_output)
        entry['crossfit_npz_sha256'] = sha(args.output / f'seed{seed}_crossfit.npz')
        entry['clip_specs_sha256'] = sha(args.output / f'seed{seed}_clip_specs.json')
        result['seeds'][str(seed)] = entry
        write(args.output / f'seed{seed}_result.json', entry)
        print(json.dumps({'seed': seed, 'complete': True, 'candidates': len(entry['candidates'])}), flush=True)
    write(args.output / 'result.json', result)
    print(args.output / 'result.json', flush=True)


if __name__ == '__main__':
    main()
