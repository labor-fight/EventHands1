#!/usr/bin/env python3
"""Training-event-only BN calibration; frozen-weight, nonformal D diagnostics."""
from __future__ import annotations

import argparse
import copy
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
from tools.xkf_repair.evaluate import setup
from tools.xkf import debug_eval as E
from tools.xkf import direct_probe as P
from tools.xkf.data import build_train_dataset, lnes_codes

ALPHAS = (0., .25, .5, 1.)
BATCH_SIZE = 32
ORDER_SEED = 20261015


def write(path, value):
    Path(path).write_text(json.dumps(E.safe_json(value), indent=2, allow_nan=False) + '\n')


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.named_parameters():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def calibration_dataset(cfg):
    """Use the canonical factory only for its readonly split, index and raw reader."""
    raw = copy.deepcopy(cfg)
    raw.setdefault('AUG', {}).setdefault('DOMRAND', {})['ENABLED'] = False
    raw['AUG']['POLARITY_FLIP'] = False
    raw['AUG']['PIXEL_POLARITY_SWAP'] = False
    raw['XKF_DATA'] = dict(raw.get('XKF_DATA', {}), WINDOW_MIN=200, WINDOW_MAX=200)
    dataset = build_train_dataset(raw, start_step=0, max_steps=1)
    assert not dataset.domrand.enabled
    assert not dataset.polarity_flip and not dataset.pixel_polarity_swap
    return dataset


def calibration_rows(dataset):
    """Fixed 4 truncated + 28 complete windows per each of the 72 train sequences."""
    rows = []
    for sequence_id, (name, _) in enumerate(dataset.sequences):
        indices = dataset.index[dataset.index[:, 0] == sequence_id]
        age = indices[:, 1] - indices[:, 2] + 1
        for label, mask, count in (('truncated', age < 200, 4), ('full200', age >= 200, 28)):
            eligible = indices[mask]
            # Retain real denominators; never duplicate rows to fill a quota.
            for index in np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=int):
                si, end, start, stop = map(int, eligible[index])
                rows.append(dict(sequence=name, sequence_id=si, end_ms=end,
                                 run_start_ms=start, run_stop_ms=stop,
                                 window_ms=min(200, end - start + 1), stratum=label))
    assert len(dataset.sequences) == 72 and len(rows) <= 2304
    order = np.random.default_rng(ORDER_SEED).permutation(len(rows))
    return [rows[int(i)] for i in order]


def calibration_batch(dataset, rows, device):
    """No pose labels, previous state or targets are read by this function."""
    images, cameras, spans = [], [], []
    for row in rows:
        handle = dataset._handle(row['sequence_id'])
        span = row['window_ms']
        xs, ys, ps, us = dataset._raw_events(handle, row['end_ms'], span)
        images.append(lnes_codes(xs, ys, np.minimum(ps, 1), us // 1000,
                                 dataset.height, dataset.width))
        cameras.append(handle.camera_K.copy())
        spans.append(span)
    codes = torch.from_numpy(np.stack(images)).to(device)
    windows = torch.tensor(spans, device=device, dtype=torch.float32)
    lnes = torch.where(codes > 0, (codes.float() - 1.) / windows[:, None, None, None],
                       torch.zeros_like(codes, dtype=torch.float32))
    return lnes, torch.from_numpy(np.stack(cameras)).to(device).float(), windows, images


def bn_modules(model):
    return {name: module for name, module in model.named_modules()
            if name.startswith('direct.') and isinstance(module, torch.nn.BatchNorm2d)}


def read_buffers(modules):
    return {name: {key: getattr(module, key).detach().cpu().clone()
                   for key in ('running_mean', 'running_var', 'num_batches_tracked')}
            for name, module in modules.items()}


def mixture_buffers(original, calibrated, alpha):
    """Moment mixture, including between-population mean variance."""
    result = {}
    for name, source in original.items():
        target = calibrated[name]
        mu0, mu1 = source['running_mean'], target['running_mean']
        result[name] = {
            'running_mean': (1. - alpha) * mu0 + alpha * mu1,
            'running_var': ((1. - alpha) * source['running_var'] + alpha * target['running_var']
                            + alpha * (1. - alpha) * (mu0 - mu1).square()),
            # Inference is frozen; preserve original training clock explicitly.
            'num_batches_tracked': source['num_batches_tracked'].clone(),
        }
    return result


def apply_buffers(modules, values):
    for name, module in modules.items():
        for key, value in values[name].items():
            getattr(module, key).copy_(value.to(getattr(module, key)))
        module.eval()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=[3407, 3408, 3409], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--declare-only', action='store_true')
    parser.add_argument('--calibration-batches', type=int, default=72,
                        help='72 is complete; smaller values are explicitly partial smoke tests')
    parser.add_argument('--limit-eval-rows', type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.calibration_batches <= 72 or args.limit_eval_rows < 0:
        parser.error('invalid calibration/evaluation limit')
    output = Path(args.output)
    if (output / 'result.json').exists():
        raise FileExistsError(output / 'result.json')
    output.mkdir(parents=True, exist_ok=True)
    setup()
    torch.set_num_threads(4)
    device = torch.device(args.device)
    cfgpath = ROOT / f'configs/xkf_repair/repair_s{args.seed}.yaml'
    checkpoint = ROOT / f'outputs/xkf/repair_20261015_r2/repair_s{args.seed}/xkf_repair_s{args.seed}-step=2200.ckpt'
    frozen = json.loads((ROOT / 'outputs/xkf/repair_20261015_r2/frozen_manifest.json').read_text())
    for name, digest in frozen['source_sha256'].items():
        if E.sha(ROOT / name) != digest:
            raise RuntimeError(f'frozen source changed: {name}')
    cfg = E.load_config(str(cfgpath))
    dataset = calibration_dataset(cfg)
    all_rows = calibration_rows(dataset)
    rows = all_rows[:args.calibration_batches * BATCH_SIZE]
    row_digest = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    bank, banksha = P.D.load_bank()
    declaration = {
        'kind': 'R3_TRAINING_ONLY_BN_CALIBRATION', 'formal': False,
        'checkpoint': str(checkpoint), 'checkpoint_sha256': E.sha(checkpoint),
        'config': str(cfgpath), 'config_sha256': E.sha(cfgpath), 'bank_sha256': banksha,
        'seed': args.seed, 'calibration_selection': 'canonical train only, 32/sequence: 4 W<200 + 28 W200; fixed ordering',
        'calibration_rows': len(rows), 'calibration_batch_size': BATCH_SIZE,
        'calibration_index_sha256': row_digest, 'order_seed': ORDER_SEED,
        'canonical_sequence_count': 72, 'actual_sequence_count': len({r['sequence'] for r in rows}),
        'available_full_calibration_rows': len(all_rows),
        'per_sequence_rows': {name: sum(row['sequence'] == name for row in rows)
                              for name, _ in dataset.sequences},
        'split_manifest_sha256': E.sha(Path(cfg['DATA']['ROOT']) / 'splits_semkine.json'),
        'source_sha256': {**frozen['source_sha256'], str(Path(__file__).relative_to(ROOT)): E.sha(__file__)},
        'alphas': list(ALPHAS), 'variance_blend': 'mixture of second moments, includes alpha*(1-alpha)*(mu0-mu1)^2',
        'calibration': 'direct BN only, reset and cumulative average; raw events+K+causal W<=200; no GT/prev/loss; no DR or polarity rewrites',
        'inference': 'BN eval frozen, targets zero, full local/global plus two training identity controls',
        'weights': 'all parameters unchanged; only independent BN buffer candidates saved',
        'partial_smoke': args.calibration_batches != 72 or args.limit_eval_rows != 0,
        'limit_eval_rows': args.limit_eval_rows,
    }
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != E.safe_json(declaration):
            raise RuntimeError('existing calibration declaration differs')
    else:
        write(protocol_path, declaration)
        write(output / 'calibration_rows.json', rows)
    if args.declare_only:
        print(json.dumps({'protocol': str(protocol_path), 'rows': len(rows), 'partial': declaration['partial_smoke']}))
        return
    model, _, identity = E.load_model(str(cfgpath), str(checkpoint), device)
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    modules = bn_modules(model)
    assert len(modules) == 15
    original = read_buffers(modules)
    parameter_hash = parameter_digest(model)
    for module in modules.values():
        module.reset_running_stats()
        module.momentum = None
        module.train()
    features = hashlib.sha256()
    started = time.time()
    for begin in range(0, len(rows), BATCH_SIZE):
        lnes, camera, windows, images = calibration_batch(dataset, rows[begin:begin + BATCH_SIZE], device)
        for image in images:
            features.update(image.tobytes())
        features.update(camera.cpu().numpy().tobytes())
        features.update(windows.cpu().numpy().tobytes())
        # Equal full batches give ordinary cumulative averaging. If a quota is
        # unavailable and the final batch is short, retain sample weighting.
        for module in modules.values():
            module.momentum = len(windows) / (begin + len(windows))
        model.direct(lnes, model._cam_planes(camera), windows)
        if begin % (BATCH_SIZE * 12) == 0:
            print(json.dumps({'phase': 'calibration', 'seed': args.seed, 'completed_rows': begin + BATCH_SIZE}), flush=True)
    calibrated = read_buffers(modules)
    actual_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
    assert all(int(v['num_batches_tracked']) == actual_batches for v in calibrated.values())
    assert parameter_digest(model) == parameter_hash
    model.eval()
    P.WINDOWS = (200,)
    options = argparse.Namespace(batch_size=32, control_rows=128, limit_rows=args.limit_eval_rows)
    candidates = {}
    for alpha in ALPHAS:
        directory = output / f'alpha_{alpha}'
        directory.mkdir()
        buffers = mixture_buffers(original, calibrated, alpha)
        apply_buffers(modules, buffers)
        buffer_path = directory / 'bn_buffers.pt'
        torch.save({'base_checkpoint_sha256': E.sha(checkpoint), 'alpha': alpha,
                    'parameter_sha256': parameter_hash, 'bn_buffers': buffers,
                    'calibration_batches': actual_batches}, buffer_path)
        value = {'alpha': alpha, 'sequences': {}, 'failures': [], 'raw_files': [],
                 'bn_buffers': str(buffer_path), 'bn_buffers_sha256': E.sha(buffer_path)}
        for name in ('zgz_local', 'zgz_global', 'lyq_local', 'ch_global'):
            arrays, summaries, failures = P.sequence(model, cfg, name, bank['sequences'][name], device, options)
            for label, raw in arrays.items():
                path = directory / f'{label}.npz'
                np.savez_compressed(path, **raw)
                value['raw_files'].append({'path': str(path), 'sha256': E.sha(path)})
            value['sequences'][name] = summaries
            value['failures'].extend(failures)
        value['complete'] = not value['failures'] and not declaration['partial_smoke']
        write(directory / 'result.json', value)
        candidates[str(alpha)] = value
        print(json.dumps({'phase': 'evaluated', 'seed': args.seed, 'alpha': alpha,
                          'failures': len(value['failures'])}), flush=True)
    if parameter_digest(model) != parameter_hash:
        raise RuntimeError('calibration changed model parameters')
    if E.sha(__file__) != declaration['source_sha256'][str(Path(__file__).relative_to(ROOT))]:
        raise RuntimeError('producer changed during probe')
    write(output / 'result.json', {'identity': identity, 'formal': False,
                                  'partial_smoke': declaration['partial_smoke'],
                                  'calibration_feature_sha256': features.hexdigest(),
                                  'calibration_batches': actual_batches,
                                  'parameter_sha256_before_and_after': parameter_hash,
                                  'elapsed_seconds': time.time() - started,
                                  'candidates': candidates})


if __name__ == '__main__':
    main()
