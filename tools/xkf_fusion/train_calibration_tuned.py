#!/usr/bin/env python3
"""Training-only affine and matched D measurements for a D-only continuation.

The I clock stays2200 while the direct/optimizer global clock is2400. No old D
predictions or affine coefficients are reused. BN buffers remain the parent's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

import numpy as np
import torch

from mano_layer import ManoLayer
from tools.xkf_fusion import bn_probe as B
from tools.xkf_fusion import calibration_probe as C
from tools.xkf_fusion.train_calibration import packet_for_measurement
from tools.xkf_repair.evaluate import setup
from tools.xkf import debug_eval as E
from tools.xkf import direct_probe as P


def source_hashes():
    root = ROOT / 'outputs/xkf/fusion_20261015_r3/direct_finetune'
    frozen = json.loads((root / 'frozen_manifest.json').read_text())['source_sha256']
    for name, digest in frozen.items():
        if E.sha(ROOT / name) != digest:
            raise ValueError(f'frozen training source changed: {name}')
    return {**frozen, **{name: E.sha(ROOT / name) for name in (
        'tools/xkf_fusion/train_calibration_tuned.py', 'tools/xkf_fusion/train_calibration.py',
        'tools/xkf_fusion/bn_probe.py', 'tools/xkf_fusion/calibration_probe.py')}}


def validate_clocks(model, saved, partial):
    step = int(saved.get('global_step', -1))
    clocks = saved.get('xkf_direct_continuation', {})
    if step != 2400 and not (partial and step == 2202):
        raise ValueError('full protocol requires D2400; D2202 is allowed only for declared partial smoke')
    if (clocks.get('start_step') != 2200 or clocks.get('i_adam_step') != 2200
            or clocks.get('d_adam_step') != step or not clocks.get('all_bn_and_xkf_statistics_frozen')):
        raise ValueError('missing/mismatched D-only clock metadata')
    optimizer = saved['optimizer_states'][0]
    parameters = list(model.named_parameters())
    ids = optimizer['param_groups'][0]['params']
    if len(ids) != len(parameters) or len(ids) != 96:
        raise ValueError('complete original96 Adam identity required')
    for (name, value), key in zip(parameters, ids):
        state = optimizer['state'][key]
        expected = step if name.startswith('direct.') else 2200
        if int(state['step']) != expected or state['exp_avg'].shape != value.shape:
            raise ValueError(f'Adam clock/shape differs: {name}')
    if any(s['last_epoch'] != step for s in saved['lr_schedulers']):
        raise ValueError('scheduler clock differs')
    return {'I_parameter_Adam_step': 2200, 'D_parameter_Adam_step': step,
            'scheduler_and_global_step': step, 'all_parameters_at_global_step': False}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-calibration-rows', type=int, default=0)
    parser.add_argument('--limit-eval-rows', type=int, default=0)
    args = parser.parse_args()
    if min(args.limit_calibration_rows, args.limit_eval_rows) < 0:
        parser.error('limits must be nonnegative')
    partial = bool(args.limit_calibration_rows or args.limit_eval_rows)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    setup()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, cfg, identity = E.load_model(args.config, args.checkpoint, device)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    identity['training_clocks'] = validate_clocks(model, saved, partial)
    seed = int(cfg['SEED'])
    if seed not in (3407, 3408, 3409):
        raise ValueError('requires canonical three-seed study')
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    bnroot = ROOT / f'outputs/xkf/fusion_20261015_r3/bn/s{seed}'
    original = torch.load(bnroot / 'alpha_0.0/bn_buffers.pt', map_location='cpu', weights_only=False)
    buffers = B.read_buffers(B.bn_modules(model))
    if set(buffers) != set(original['bn_buffers']):
        raise ValueError('BN module identity differs')
    for name, values in buffers.items():
        for field, value in values.items():
            if not torch.equal(value, original['bn_buffers'][name][field]):
                raise ValueError(f'BN differs from original2200 parent: {name}/{field}')
    del saved
    dataset = B.calibration_dataset(cfg)
    rows = B.calibration_rows(dataset)
    if rows != json.loads((bnroot / 'calibration_rows.json').read_text()):
        raise ValueError('training sample list differs from frozen canonical2304')
    if any(row['sequence'].startswith('zgz_') for row in rows):
        raise ValueError('zgz cannot enter training calibration')
    if args.limit_calibration_rows:
        rows = rows[:args.limit_calibration_rows]
    source = source_hashes()
    bank, banksha = P.D.load_bank()
    declaration = {'kind': 'R3_TRAIN_ONLY_D_TUNED_AFFINE', 'formal': False, 'automatic_full_training': False,
        'identity': identity, 'source_sha256': source, 'partial_smoke': partial,
        'BN': 'original2200 buffers bitwise retained; no calibration, no test-batch stats',
        'fit': 'one pooled12-parameter translation affine, same canonical9 identities/72 sequences raw training windows',
        'ridge': C.RIDGE, 'rows': rows, 'fit_rows': len(rows), 'bank_sha256': banksha,
        'canonical_train_sequences': [name for name, _ in dataset.sequences],
        'split_manifest_sha256': E.sha(Path(cfg['DATA']['ROOT']) / 'splits_semkine.json'),
        'labels': 'raw training GT read only after D forward; inference packet target and previous zero',
        'external': 'fresh same-checkpoint D W200 all2590 zgz packets plus128+128 controls; no reuse of old predictions',
        'external_limit_rows': args.limit_eval_rows,
        'oracle_surgeries': 'P.sequence also exports its labelled GTpose/GT2d diagnostics; they never enter affine fitting/inference'}
    B.write(args.output / 'protocol.json', declaration)
    started = time.time()
    parameters_before = B.parameter_digest(model)
    arrays = {key: [] for key in ('direct_state', 'pred_joints0', 'truth', 'truth_joints')}
    features = hashlib.sha256()
    for begin in range(0, len(rows), B.BATCH_SIZE):
        selected = rows[begin:begin + B.BATCH_SIZE]
        lnes, camera, windows, images = B.calibration_batch(dataset, selected, device)
        beta = torch.tensor(np.stack([dataset._handle(row['sequence_id']).betas for row in selected]), device=device).float()
        packet = packet_for_measurement(lnes, camera, windows, beta)
        if bool(packet.target.any() or packet.prev_state.any()):
            raise ValueError('training measurement packet must be target/previous free')
        pred, aux = model.direct_measurement(packet)
        truth = torch.tensor(np.stack([dataset._handle(row['sequence_id']).pos51[row['end_ms']]
                                      for row in selected]), device=device).float()
        _, joints = model._fk(truth, beta)
        for key, tensor in zip(arrays, (pred, aux['joints0'], truth, joints)):
            arrays[key].append(tensor.cpu().numpy())
        for image in images:
            features.update(image.tobytes())
        features.update(camera.cpu().numpy().tobytes())
        features.update(windows.cpu().numpy().tobytes())
    data = {key: np.concatenate(value) for key, value in arrays.items()}
    if not all(np.isfinite(value).all() for value in data.values()):
        raise FloatingPointError('nonfinite training fit rows; never filter')
    if not args.limit_calibration_rows:
        expected = json.loads((bnroot / 'result.json').read_text())['calibration_feature_sha256']
        if features.hexdigest() != expected:
            raise ValueError('calibration raw feature hash differs from frozen2304')
    spec = C.fit('translation_affine', data['direct_state'], data['pred_joints0'], data['truth'], None)
    runtime = {key: value for key, value in spec.items() if key != 'fit_rows'}
    corrected, joints0 = C.apply(data['direct_state'], data['pred_joints0'], None, runtime)
    prediction_path = args.output / 'train_predictions.npz'
    np.savez_compressed(prediction_path, **data, sequence=np.array([r['sequence'] for r in rows]),
                        end_ms=np.array([r['end_ms'] for r in rows]))
    export = {'spec': runtime, 'seed': seed, 'fit_rows': len(rows), 'identity': identity,
              'checkpoint_sha256': identity['checkpoint_sha256'], 'fit_source_sha256': E.sha(prediction_path),
              'fit_scope': 'training-only canonical9/72, one shared transform; no sequence/run switch',
              'protocol_sha256': E.sha(args.output / 'protocol.json'), 'partial_smoke': partial}
    B.write(args.output / 'translation_affine_runtime.json', export)
    result = {'kind': declaration['kind'], 'formal': False, 'identity': identity, 'partial_smoke': partial,
              'calibration_feature_sha256': features.hexdigest(), 'runtime': export,
              'training_score_before': C.score(data['direct_state'], data['pred_joints0'], data['truth'], data['truth_joints']),
              'training_score_after': C.score(corrected, joints0, data['truth'], data['truth_joints']),
              'external_scores': {}, 'external_d_probe': {}}
    P.WINDOWS = (200,)
    probe_args = SimpleNamespace(control_rows=128, limit_rows=args.limit_eval_rows, batch_size=32)
    for name in ('zgz_local', 'zgz_global', 'lyq_local', 'ch_global'):
        outputs, summaries, failures = P.sequence(model, cfg, name, bank['sequences'][name], device, probe_args)
        raw = outputs[f'{name}_W200']
        if failures or not raw['success'].all():
            B.write(args.output / 'failures.json', {'failures': failures, 'sequence': name})
            raise ValueError('fresh external D probe failed; no finite-subset scoring')
        path = args.output / f'{name}_W200.npz'
        np.savez_compressed(path, **raw)
        corrected, joints0 = C.apply(raw['direct_state'], raw['pred_joints0'], None, runtime)
        masks = {'all': np.ones(len(corrected), bool), 'cold_first3_per_run': raw['cold_first3'],
                 'mature_W200': raw['effective_window_ms'] == 200,
                 'original_bad_debug_clip_first10': raw['original_debug_clip']}
        scored = {}
        for label, mask in masks.items():
            scored[label] = ({'before': C.score(raw['direct_state'][mask], raw['pred_joints0'][mask], raw['truth'][mask], raw['truth_joints'][mask]),
                              'after': C.score(corrected[mask], joints0[mask], raw['truth'][mask], raw['truth_joints'][mask])}
                             if mask.any() else {'n': 0, 'valid': None, 'reason': 'empty fixed cohort'})
        result['external_scores'][name] = {'source': str(path), 'sha256': E.sha(path),
                                           'checkpoint_sha256': identity['checkpoint_sha256'], 'cohorts': scored}
        result['external_d_probe'][name] = summaries
        B.write(args.output / 'progress.json', {'seed': seed, 'completed_sequences': list(result['external_scores']),
                                               'elapsed_seconds': time.time() - started})
    if B.parameter_digest(model) != parameters_before or source_hashes() != source:
        raise ValueError('model parameters or source changed during fit/scoring')
    after = B.read_buffers(B.bn_modules(model))
    if any(not torch.equal(after[name][field], value) for name, values in buffers.items() for field, value in values.items()):
        raise ValueError('BN buffers changed during inference')
    if E.sha(args.checkpoint) != identity['checkpoint_sha256'] or E.sha(args.config) != identity['config_sha256']:
        raise ValueError('checkpoint/config changed during scoring')
    result.update(elapsed_seconds=time.time() - started, parameter_sha256_before_and_after=parameters_before,
                  BN_buffers_unchanged=True, all_external_rows_success=True)
    B.write(args.output / 'result.json', result)


if __name__ == '__main__':
    main()
