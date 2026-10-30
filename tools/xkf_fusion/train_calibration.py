#!/usr/bin/env python3
"""Fit one D translation affine per seed using only canonical training labels."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from mano_layer import ManoLayer
from tools.xkf_fusion import bn_probe as B
from tools.xkf_fusion import calibration_probe as C
from tools.xkf_repair.evaluate import setup
from tools.xkf import debug_eval as E
from tools.xkf import direct_probe as P


def packet_for_measurement(lnes, camera, windows, betas):
    """D receives events/camera/beta only; target and incoming state are zero."""
    n, device = len(lnes), lnes.device
    zeros = torch.zeros(n, 51, device=device)
    return P.EventPacketBatch(
        events=torch.zeros(0, 5, device=device), ptr=torch.zeros(n + 1, dtype=torch.long, device=device),
        sequence_id=torch.zeros(n, dtype=torch.long, device=device),
        t_start_us=torch.zeros(n, dtype=torch.long, device=device),
        t_end_us=windows.long() * 1000, delta_t_s=torch.full((n,), .05, device=device),
        is_sequence_start=torch.zeros(n, dtype=torch.bool, device=device),
        is_sequence_end=torch.zeros(n, dtype=torch.bool, device=device),
        target=zeros, prev_state=zeros.clone(), betas=betas, camera_K=camera,
        lnes=lnes, lnes_window_ms=windows.long())


def score_external(seed, alpha, spec):
    """Only score after fixed training-only fit; all zgz rows stay outside fitting."""
    result = {}
    directory = ROOT / f'outputs/xkf/fusion_20261015_r3/bn/s{seed}/alpha_{alpha}'
    for sequence in ('zgz_local', 'zgz_global', 'lyq_local', 'ch_global'):
        path = directory / f'{sequence}_W200.npz'
        with np.load(path, allow_pickle=False) as raw:
            if not raw['success'].all():
                raise ValueError('external scoring input contains failed rows')
            pred, joints0 = C.apply(raw['direct_state'], raw['pred_joints0'], None, spec)
            masks = {'all': np.ones(len(pred), bool), 'cold_first3_per_run': raw['cold_first3'],
                     'mature_W200': raw['effective_window_ms'] == 200,
                     'original_bad_debug_clip_first10': raw['original_debug_clip']}
            result[sequence] = {'source': str(path), 'sha256': E.sha(path), 'cohorts': {}}
            for name, mask in masks.items():
                result[sequence]['cohorts'][name] = (C.score(pred[mask], joints0[mask], raw['truth'][mask],
                                                           raw['truth_joints'][mask]) if mask.any()
                                                    else {'n': 0, 'valid': None, 'reason': 'empty fixed cohort'})
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[3407, 3408, 3409], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-calibration-rows', type=int, default=0)
    args = parser.parse_args()
    if args.limit_calibration_rows < 0:
        parser.error('limit must be nonnegative')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    setup(); torch.set_num_threads(4)
    device = torch.device(args.device)
    cfgpath = ROOT / f'configs/xkf_repair/repair_s{args.seed}.yaml'
    checkpoint = ROOT / f'outputs/xkf/repair_20261015_r2/repair_s{args.seed}/xkf_repair_s{args.seed}-step=2200.ckpt'
    model, cfg, identity = E.load_model(cfgpath, checkpoint, device)
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    dataset = B.calibration_dataset(cfg)
    rows = B.calibration_rows(dataset)
    bnroot = ROOT / f'outputs/xkf/fusion_20261015_r3/bn/s{args.seed}'
    if rows != json.loads((bnroot / 'calibration_rows.json').read_text()):
        raise ValueError('canonical train calibration rows differ from frozen BN sampling')
    if args.limit_calibration_rows:
        rows = rows[:args.limit_calibration_rows]
    if any(row['sequence'].startswith('zgz_') for row in rows):
        raise ValueError('zgz must never enter training-only calibration')
    frozen = json.loads((ROOT / 'outputs/xkf/repair_20261015_r2/frozen_manifest.json').read_text())
    for name, digest in frozen['source_sha256'].items():
        if E.sha(ROOT / name) != digest:
            raise ValueError(f'frozen source changed: {name}')
    source = {**frozen['source_sha256'], **{name: E.sha(ROOT / name) for name in (
        'tools/xkf_fusion/train_calibration.py', 'tools/xkf_fusion/bn_probe.py',
        'tools/xkf_fusion/calibration_probe.py')}}
    declaration = {'kind': 'R3_TRAIN_ONLY_D_AFFINE', 'formal': False, 'identity': identity,
                   'source_sha256': source, 'alphas': [0., .5], 'ridge': C.RIDGE,
                   'fit': '12 parameters, one pooled translation affine per seed/BN variant; no sequence-dependent application',
                   'canonical_train_sequences': [name for name, _ in dataset.sequences],
                   'rows': rows, 'calibration_rows': len(rows), 'partial_smoke': bool(args.limit_calibration_rows),
                   'bn_calibration_protocol_sha256': E.sha(bnroot / 'protocol.json'),
                   'split_manifest_sha256': E.sha(Path(cfg['DATA']['ROOT']) / 'splits_semkine.json'),
                   'labels': 'raw training GT only for fitting/scoring after D forward; packet target and prev are zero',
                   'external_scoring': 'all zgz local/global and two training-identity controls, never used for fit or coefficients'}
    B.write(args.output / 'protocol.json', declaration)
    modules = B.bn_modules(model)
    parameter_hash = B.parameter_digest(model)
    result = {'kind': declaration['kind'], 'formal': False, 'identity': identity,
              'partial_smoke': declaration['partial_smoke'], 'variants': {}}
    started = time.time()
    for alpha in (0., .5):
        buffer_path = bnroot / f'alpha_{alpha}/bn_buffers.pt'
        saved = torch.load(buffer_path, map_location='cpu', weights_only=False)
        if saved['base_checkpoint_sha256'] != identity['checkpoint_sha256'] or saved['parameter_sha256'] != parameter_hash:
            raise ValueError('BN source/checkpoint mismatch')
        B.apply_buffers(modules, saved['bn_buffers']); model.eval()
        arrays = {key: [] for key in ('direct_state', 'pred_joints0', 'truth', 'truth_joints')}
        features = hashlib.sha256()
        for begin in range(0, len(rows), B.BATCH_SIZE):
            batch_rows = rows[begin:begin + B.BATCH_SIZE]
            lnes, camera, windows, images = B.calibration_batch(dataset, batch_rows, device)
            beta = torch.tensor(np.stack([dataset._handle(row['sequence_id']).betas for row in batch_rows]), device=device).float()
            packet = packet_for_measurement(lnes, camera, windows, beta)
            assert not bool(packet.target.any() or packet.prev_state.any())
            direct, aux = model.direct_measurement(packet)
            # This is the first pose-label access in the measurement/fit loop.
            truth = torch.tensor(np.stack([dataset._handle(row['sequence_id']).pos51[row['end_ms']]
                                           for row in batch_rows]), device=device).float()
            _, joints = model._fk(truth, beta)
            for key, tensor in zip(arrays, (direct, aux['joints0'], truth, joints)):
                arrays[key].append(tensor.cpu().numpy())
            for image in images:
                features.update(image.tobytes())
            features.update(camera.cpu().numpy().tobytes()); features.update(windows.cpu().numpy().tobytes())
        data = {key: np.concatenate(values) for key, values in arrays.items()}
        if not all(np.isfinite(value).all() for value in data.values()):
            raise FloatingPointError('training calibration predictions/labels nonfinite')
        if not args.limit_calibration_rows:
            expected = json.loads((bnroot / 'result.json').read_text())['calibration_feature_sha256']
            if features.hexdigest() != expected:
                raise ValueError('calibration event/camera features differ from frozen BN sample')
        spec = C.fit('translation_affine', data['direct_state'], data['pred_joints0'], data['truth'], None)
        runtime = {key: value for key, value in spec.items() if key != 'fit_rows'}
        corrected, joints0 = C.apply(data['direct_state'], data['pred_joints0'], None, runtime)
        predictions = args.output / f'train_predictions_alpha{alpha}.npz'
        np.savez_compressed(predictions, **data, sequence=np.array([r['sequence'] for r in rows]),
                            end_ms=np.array([r['end_ms'] for r in rows]))
        export = {'spec': runtime, 'seed': args.seed, 'alpha': alpha, 'fit_rows': len(rows),
                  'fit_scope': 'canonical9/72 training only, one shared affine across all sequences',
                  'fit_source_sha256': E.sha(predictions), 'checkpoint_sha256': identity['checkpoint_sha256'],
                  'bn_buffers_sha256': E.sha(buffer_path), 'protocol_sha256': E.sha(args.output / 'protocol.json')}
        B.write(args.output / f'translation_affine_alpha{alpha}_runtime.json', export)
        result['variants'][str(alpha)] = {'runtime': export, 'calibration_feature_sha256': features.hexdigest(),
            'training_score_before': C.score(data['direct_state'], data['pred_joints0'], data['truth'], data['truth_joints']),
            'training_score_after': C.score(corrected, joints0, data['truth'], data['truth_joints']),
            'training_depth_range_mm': np.quantile(data['truth_joints'][:, 0, 2] * 1000, [0., .05, .5, .95, 1.]).tolist(),
            'external_scores': score_external(args.seed, alpha, runtime)}
        print(json.dumps({'seed': args.seed, 'alpha': alpha, 'fit_rows': len(rows), 'phase': 'complete'}), flush=True)
    if B.parameter_digest(model) != parameter_hash:
        raise ValueError('training calibration changed parameters')
    for name, digest in source.items():
        if E.sha(ROOT / name) != digest:
            raise ValueError('producer changed during training calibration')
    result.update(elapsed_seconds=time.time() - started, parameter_sha256_before_and_after=parameter_hash)
    B.write(args.output / 'result.json', result)


if __name__ == '__main__':
    main()
