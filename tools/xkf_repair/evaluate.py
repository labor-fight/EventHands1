#!/usr/bin/env python3
"""Additive evaluation adapters for XKF repair candidates; never a formal gate override."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import os

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from mano_layer import ManoLayer
from semkine.xkf_repair import XKFRepairModel
from tools.xkf import debug_eval as E
from tools.xkf import recovery_probe as R
from tools.xkf import direct_probe as P
from tools.xkf import make_row as F
from tools.xdl import make_row as X
from tools.xkf_repair.data import build_train_dataset, collate_repair


def sources():
    names = ['tools/xkf_repair/evaluate.py', 'tools/xkf_repair/data.py',
             'semkine/xkf_repair.py', 'semkine/xkf_repair_losses.py',
             'tools/xkf/debug_eval.py', 'tools/xkf/recovery_probe.py',
             'tools/xkf/direct_probe.py', 'tools/xkf/make_row.py']
    return {p: E.sha(ROOT / p) for p in names}


def gradient_audit(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    main = batch['main']
    xi, zd, aux = model.branches(main)
    li, _ = model.incremental_loss(xi, main, batch)
    ld, _ = model._direct_loss(zd, aux, main)
    if model.repair_motion_target:
        loss = li.clamp_min(1e-12).log10() + ld.clamp_min(1e-12).log10()
    else:
        loss = li.log10() + ld.log10()
    loss.backward()
    groups = {}
    for label, direct in [('i', False), ('d', True)]:
        grads = [p.grad for name, p in model.named_parameters()
                 if name.startswith('direct.') == direct and p.requires_grad and p.grad is not None]
        groups[label] = {'gradient_tensors': len(grads),
                         'finite': all(bool(torch.isfinite(g).all()) for g in grads),
                         'nonzero': any(bool((g != 0).any()) for g in grads)}
    model.zero_grad(set_to_none=True)
    model.eval()
    return {'loss': float(loss.detach()), 'branches': groups,
            'pass': bool(torch.isfinite(loss)) and all(g['finite'] and g['nonzero'] for g in groups.values()),
            'scope': 'actual configured repair/control objective, main packet backward, no optimizer or burn'}


def setup():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    E.XKFModel = XKFRepairModel
    F.XKFModel = XKFRepairModel
    E.build_train_dataset = build_train_dataset
    E.collate_long = collate_repair
    E.gradient_audit = gradient_audit

    class MotionAdapter:
        @staticmethod
        def forward_packet(model, batch):
            return model.incremental_prediction(batch)
    R.RecoveryXYZModel = MotionAdapter


@torch.no_grad()
def rollout(args):
    output = Path(args.output)
    if (output / 'summary.json').exists():
        raise FileExistsError(output / 'summary.json')
    output.mkdir(parents=True, exist_ok=True)
    bank, banksha = R.D.load_bank()
    clips, excluded = R.schedule(bank, 10)
    before = sources()
    declaration = {'kind': 'XKF_REPAIR_EXPLORATORY_ROLLOUT', 'formal_gate_assessed': False,
                   'config_sha256': E.sha(args.config), 'checkpoint_sha256': E.sha(args.checkpoint),
                   'bank_sha256': banksha, 'source_sha256': before,
                   'conditions': R.CONDITIONS, 'modes': R.MODES, 'clips': clips, 'excluded': excluded,
                   'selection': 'exact previous112clip9condition6mode schedule; all short clips retained',
                   'tf32': False, 'inference_targets': 'zero', 'finite_subset_reduction': False}
    X.write_json(output / 'protocol.json', declaration)
    device = torch.device(args.device)
    model, cfg, identity = E.load_model(args.config, args.checkpoint, device)
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    started = time.perf_counter()
    records, caches = [], {}
    rawdir = output / 'raw'
    rawdir.mkdir(exist_ok=True)
    parity = {}
    with (output / 'per_clip.jsonl').open('w') as f:
        for index, clip in enumerate(clips):
            name = clip['sequence']
            meta = bank['sequences'][name]
            if name not in caches:
                data = R.D.load_data(cfg, name, meta)
                caches[name] = R.PacketCache(model, data, device)
            cache = caches[name]
            value = R.rollout(model, cache.data, meta, clip, cache)
            if clip['cohort'] in ('local_bank', 'global_bank') and name not in parity:
                old = E.snippets(model, cache.data, meta, device, 1, 10)['traces'][0]
                assert old['clip_id'] == clip['clip_id']
                # Compare the identical combined+ condition against the original serial mode path.
                ref = np.asarray([[s['predictions'][m] for m in E.MODES] for s in old['steps']])
                actual = value['predictions'][:, 7, :len(E.MODES), :]
                difference = np.abs(actual - ref)
                parity[name] = {'translation_max_mm': float(difference[..., :3].max() * 1000),
                                'pose_max_rad': float(difference[..., 3:].max())}
                if parity[name]['translation_max_mm'] > .2 or parity[name]['pose_max_rad'] > .002:
                    raise RuntimeError(f'rollout parity failed: {parity}')
            np.savez_compressed(rawdir / f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz', **value)
            new = R.make_rows(clip, value)
            for row in new:
                f.write(json.dumps(E.safe_json(row), allow_nan=False) + '\n')
            f.flush()
            records.extend(new)
            if index % 16 == 0:
                print(json.dumps({'completed_clips': index + 1, 'total': len(clips)}), flush=True)
    assert before == sources(), 'producer changed during evaluation'
    result = {'kind': declaration['kind'], 'formal_gate_assessed': False, 'identity': identity,
              'source_sha256': before, 'device': str(device), 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
              'protocol_sha256': E.sha(output / 'protocol.json'), 'metrics': R.METRICS,
              'partial': False, 'clip_count': len(clips), 'parity': parity,
              'gains_fused': model.gains('fused').cpu().tolist(),
              'gains_d_zero': model.gains('d_zero').cpu().tolist(),
              'elapsed_seconds': time.perf_counter() - started, 'aggregate': R.aggregate(records)}
    X.write_json(output / 'summary.json', result)
    print(json.dumps({'output': str(output / 'summary.json')}), flush=True)


@torch.no_grad()
def row(args):
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools import make_s36_row as C
    from tools.xkf.runtime import LongWindowEval
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    device = torch.device(args.device)
    m, cfg, identity = E.load_model(args.config, args.checkpoint, device)
    step = identity['global_step']
    if not 2000 <= step < 6000:
        raise ValueError('repair rows require explicit debug weights at2000..5999, never formal')
    before = sources()
    mano = ManoLayer(cfg['MANO']['NPZ'], add_mean=False).to(device).eval()
    root_data = Path(cfg['DATA']['ROOT'])
    rng = np.random.default_rng(0)
    raw = []
    protocol = {'formal': False, 'scope': 'paired repair continuation debug, full2590 own-loop',
                'seeds': [int(cfg['SEED'])], 'n_seeds': 1, 'exact_optimizer_step': step,
                'split': 'val_core', 'step_ms': 50, 'window_ms': 50, 'direct_window_ms': 200,
                'noise_rng_seed': 0, 'noise_scale': 1., 'tf32': False}
    for seq, directory in sequences_for_split(root_data, 'val_core', splits_manifest(cfg)):
        with LongWindowEval(F.runs_of(root_data, directory, seq), 200):
            result = ET.track_sequence(m, mano, cfg, root_data, directory, seq, 50, device, rng, 1., 1., None, window_ms=50)
        if result is None:
            raise RuntimeError(f'empty sequence {seq}')
        raw.append(result)
        X.write_json(str(output) + '.raw_accuracy.json', {'identity': identity, 'protocol': protocol,
                     'raw_sequences': raw, 'complete': len(raw) == 2, 'finite_subset_reduction': False})
    accuracy = X.accuracy_aggregate(raw)
    accuracy.update(step=step, ckpt=args.checkpoint, checkpoint_sha256=identity['checkpoint_sha256'])
    latency, audit = F.latency(m, cfg, device, 200)
    macs, params = F.macs(cfg, m)
    del m, mano
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    anchor = C.latency_anchor(device)
    assert before == sources()
    X.write_json(output, {'per_seed': {str(cfg['SEED']): accuracy},
                 'two_seed_mean': {k: accuracy[k] for k in ('overall', 'local', 'global')},
                 'latency_ms_scaled_full1p75': latency / anchor * C.FULL_PUBLISHED_MS,
                 'latency_ms_raw': latency, 'anchor_full_raw_ms': anchor, 'latency_context_audit': audit,
                 'macs_forward_packet': macs, 'params_total': params, 'protocol': protocol,
                 'numeric_status': 'PASS', 'finite_subset_reduction': False,
                 'provenance': dict(identity, source_sha256=before)})
    print(json.dumps({'output': str(output), 'formal': False}), flush=True)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--kind', required=True, choices=['debug', 'direct', 'rollout', 'row'])
    outer, rest = parser.parse_known_args()
    setup()
    sys.argv = [sys.argv[0]] + rest
    if outer.kind in ('debug', 'direct'):
        before = sources()
        if outer.kind == 'debug':
            E.main()
            output = Path(rest[rest.index('--output') + 1])
        else:
            P.main()
            output = Path(rest[rest.index('--output-dir') + 1]) / 'result.json'
        assert before == sources()
        result = json.loads(output.read_text())
        result['repair_adapter_sha256'] = before
        result['formal_gate_assessed'] = False
        if outer.kind == 'debug':
            result['reported_i_loss_scope'] = ('cohort loss.i remains the original absolute-state scoring loss; '
                                               'gradient audit uses the actual configured training objective')
        X.write_json(output, result)
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument('--config', required=True)
        ap.add_argument('--checkpoint', required=True)
        ap.add_argument('--output', required=True)
        ap.add_argument('--device', default='cuda:0')
        args = ap.parse_args()
        torch.set_num_threads(4)
        (rollout if outer.kind == 'rollout' else row)(args)


if __name__ == '__main__':
    main()
