#!/usr/bin/env python3
"""Original fixed-cohort readiness evidence plus explicit D-only lineage facts.

The frozen R2 gradient_audit remains exactly the original two-branch backward
capability probe. It is not evidence of optimizer updates. Current-stage D-only
gradients and zero I/burn/self exposure are separately recorded. Cumulative
ancestor exposure is verified from immutable snapshots, never silently inserted
into the original current-stage readiness result. No formal/full authorization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

import torch

from tools.xkf import debug_eval as E
from tools.xkf_repair import evaluate as RE
from tools.xkf_fusion.finetune_direct import DirectFinetuneModel, atomic_json, sha, state_hashes
from tools.xkf_fusion.train_calibration_tuned import source_hashes, validate_clocks

BASE = ROOT / 'outputs/xkf/fusion_20261015_r3/direct_finetune'
R2 = ROOT / 'outputs/xkf/repair_20261015_r2'
R1 = ROOT / 'outputs/xkf/debug_20261015_r1'


def lineage(checkpoint):
    """Verify each checkpoint link and each distinct stage's observed exposure."""
    checkpoint = Path(checkpoint).resolve()
    inputs = {}
    def read(path):
        path = Path(path).resolve()
        inputs[str(path)] = sha(path)
        return json.loads(path.read_text())
    metadata = read(checkpoint.parent / 'training_metadata.json')
    seed = int(metadata['seed'])
    parent = Path(metadata['resumed_from']).resolve()
    if sha(parent) != metadata['parent_checkpoint_sha256']:
        raise ValueError('current parent hash differs from training declaration')
    inputs[str(checkpoint)], inputs[str(parent)] = sha(checkpoint), sha(parent)
    finish = read(checkpoint.parent / 'train_finish.json')
    if finish['global_step'] != 2400 or finish['checkpoint_sha256'] != inputs[str(checkpoint)]:
        raise ValueError('current endpoint identity mismatch')
    declaration = read(R2 / 'protocol.json')['lineage'][str(seed)]
    original_initial = Path(declaration['actual_random_initial']['path'])
    initial_sha = declaration['actual_random_initial']['sha256']
    copied_initial = checkpoint.parent / 'initial_state.ckpt'
    if sha(original_initial) != initial_sha or sha(copied_initial) != initial_sha:
        raise ValueError('actual initial checkpoint lineage mismatch')
    initial = torch.load(copied_initial, map_location='cpu', weights_only=False)
    if initial.get('global_step') != 0 or initial['hyper_parameters']['SEED'] != seed:
        raise ValueError('the actual initial must really be step0 with matching seed')
    inputs[str(original_initial.resolve())] = initial_sha
    inputs[str(copied_initial.resolve())] = initial_sha
    stages, prior_step, prior_checkpoint = [], 0, None
    for step in (200, 800, 1600, 2000):
        snapshot = R1 / f'pipeline/snapshots/step{step}'
        manifest = read(snapshot / 'manifest.json')
        stage_dir = snapshot / f'xkf_s{seed}'
        for name, digest in manifest['files'].items():
            if name.startswith(f'xkf_s{seed}/') and sha(snapshot / name) != digest:
                raise ValueError(f'ancestor snapshot changed: {snapshot / name}')
        identity = manifest['checkpoints'][str(seed)]
        path = Path(identity['checkpoint']).resolve()
        if sha(path) != identity['sha256'] or identity['initial_sha256'] != initial_sha:
            raise ValueError('ancestor checkpoint/initial mismatch')
        inputs[str(path)] = identity['sha256']
        meta, end = read(stage_dir / 'training_metadata.json'), read(stage_dir / 'train_finish.json')
        if meta['start_step'] != prior_step or meta['max_steps'] != step or end['global_step'] != step or end['checkpoint_sha256'] != identity['sha256']:
            raise ValueError('ancestor stage clocks mismatch')
        if (meta['resumed_from'] is None) != (prior_checkpoint is None):
            raise ValueError('ancestor resume chain missing')
        if prior_checkpoint is not None and Path(meta['resumed_from']).resolve() != prior_checkpoint:
            raise ValueError('ancestor resume chain does not join')
        exposure = []
        for rank in (0, 1):
            value = read(stage_dir / f'actual_exposure_rank{rank}.json')
            if value['final_global_step'] != step or value['main_supervised_rows'] != (step-prior_step)*512:
                raise ValueError('ancestor exposure clock/count mismatch')
            exposure.append(dict(value, rank=rank))
        stages.append({'start_step': prior_step, 'end_step': step, 'checkpoint': str(path),
                       'checkpoint_sha256': identity['sha256'], 'exposure': exposure})
        prior_step, prior_checkpoint = step, path
    if inputs[str(prior_checkpoint)] != declaration['checkpoint']['sha256']:
        raise ValueError('R2 declared original parent differs from ancestor chain')
    meta, end = read(parent.parent / 'training_metadata.json'), read(parent.parent / 'train_finish.json')
    if (Path(meta['resumed_from']).resolve() != prior_checkpoint or meta['start_step'] != 2000
            or meta['max_steps'] != 2200 or end['global_step'] != 2200
            or end['checkpoint_sha256'] != inputs[str(parent)]):
        raise ValueError('R2 parent stage does not bind the ancestor chain')
    exposure = []
    for rank in (0, 1):
        value = read(parent.parent / f'actual_exposure_rank{rank}.json')
        if value['final_global_step'] != 2200 or value['main_supervised_rows'] != 200*512:
            raise ValueError('R2 parent exposure clock/count mismatch')
        exposure.append(dict(value, rank=rank))
    stages.append({'start_step': 2000, 'end_step': 2200, 'checkpoint': str(parent),
                   'checkpoint_sha256': inputs[str(parent)], 'exposure': exposure})
    current, cumulative = [], []
    for rank in (0, 1):
        value = read(checkpoint.parent / f'freeze_audit_rank{rank}.json')
        if (value['status'] != 'PASS' or value['global_step'] != 2400 or value['i_adam_step'] != 2200
                or value['d_adam_step'] != 2400 or value['main_supervised_rows'] != 200*512
                or value['burn_model_rows'] != 0 or value['i_forward_rows'] != 0):
            raise ValueError('current D-only exposure differs from protocol')
        current.append({'rank': rank, 'D_main_rows': value['main_supervised_rows'],
                        'I_main_rows': 0, 'self_main_rows': 0, 'burn_model_rows': 0,
                        'I_optimizer_updates': 0, 'D_optimizer_updates': 200})
        cumulative.append({'rank': rank,
            'I_main_rows': sum(s['exposure'][rank]['main_supervised_rows'] for s in stages),
            'D_main_rows': sum(s['exposure'][rank]['main_supervised_rows'] for s in stages)+200*512,
            'self_main_rows': sum(s['exposure'][rank]['self_main_rows'] for s in stages),
            'burn_model_rows': sum(s['exposure'][rank]['burn_model_rows'] for s in stages),
            'I_optimizer_updates': 2200, 'D_optimizer_updates': 2400})
    previous = read(R2 / f'evaluations/repair_s{seed}_step2200.json')
    if previous['identity']['checkpoint_sha256'] != inputs[str(parent)]:
        raise ValueError('parent debug gradient evidence is from another checkpoint')
    return {'seed': seed, 'actual_initial': {'path': str(copied_initial), 'sha256': initial_sha},
            'ancestor_stages': stages, 'current_stage_exposure': current,
            'cumulative_lineage_exposure': cumulative, 'parent_gradient_capability': previous['gradients'],
            'parent_original_readiness': previous['readiness'], 'verified_inputs': inputs,
            'interpretation': 'Current stage has zero I/self/burn exposure; ancestors have real observed exposure. Cumulative facts do not replace current-stage criteria automatically.'}


def actual_d_only_gradients(config, checkpoint):
    """Fresh disposable copy; the actual BF16 D-only objective, no optimizer."""
    from config import load_config
    cfg = load_config(str(config))
    model = DirectFinetuneModel(cfg)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(saved['state_dict'], strict=True)
    clocks = validate_clocks(model, saved, partial=False)
    dataset = RE.build_train_dataset(cfg, max_steps=1)
    batch = RE.collate_repair([dataset[i] for i in range(4)])
    before = state_hashes(model)
    model.train()
    model.log = lambda *args, **kwargs: None
    with torch.autocast('cpu', dtype=torch.bfloat16):
        loss = model.training_step(batch, 0)
    loss.backward()
    groups = {}
    for label, direct in [('i', False), ('d', True)]:
        params = [p for n, p in model.named_parameters() if n.startswith('direct.') == direct]
        gradients = [p.grad for p in params if p.grad is not None]
        groups[label] = {'parameters': len(params), 'trainable': sum(p.requires_grad for p in params),
                         'gradient_tensors': len(gradients),
                         'finite': all(bool(torch.isfinite(g).all()) for g in gradients),
                         'nonzero': any(bool((g != 0).any()) for g in gradients)}
    if state_hashes(model) != before:
        raise RuntimeError('actual-objective gradient probe changed weights or BN/EMA buffers')
    valid = (bool(torch.isfinite(loss)) and groups['d']['gradient_tensors'] == 60 and groups['d']['finite']
             and groups['d']['nonzero'] and groups['i']['gradient_tensors'] == groups['i']['trainable'] == 0)
    return {'loss': float(loss.detach()), 'groups': groups, 'pass': valid, 'training_clocks': clocks,
            'scope': 'actual D-only BF16 objective on fixed four main samples; fresh copy, zero optimizer updates; not observed training exposure',
            'all_saved_state_unchanged': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    torch.set_num_threads(4)
    started = time.monotonic()
    source = {**source_hashes(), 'tools/xkf_fusion/readiness_tuned.py': sha(__file__)}
    facts = lineage(args.checkpoint)
    atomic_json(args.output_dir / 'lineage.json', facts)
    RE.setup()
    # Keep the exact frozen R2 gradient function and original readiness formulas.
    assert E.gradient_audit is RE.gradient_audit
    initial_path, current_path = args.output_dir/'initial.json', args.output_dir/'current.json'
    saved_argv = sys.argv
    for checkpoint, output, comparison, gradients in (
        (Path(facts['actual_initial']['path']), initial_path, None, False),
        (args.checkpoint, current_path, initial_path, True),
    ):
        sys.argv = [saved_argv[0], '--config', str(args.config), '--checkpoint', str(checkpoint),
                    '--output', str(output), '--device', 'cpu', '--cpu-threads', '4',
                    '--train-batches', '2', '--batch-size', '4', '--train-start-index', '0',
                    '--check-rows', '8', '--clips', '1', '--steps', '10']
        if comparison is not None:
            sys.argv += ['--compare', str(comparison)]
        if gradients:
            sys.argv += ['--gradients']
        E.main()
    sys.argv = saved_argv
    current = json.loads(current_path.read_text())
    initial = json.loads(initial_path.read_text())
    if (current['identity']['config_sha256'] != initial['identity']['config_sha256']
            or initial['identity']['global_step'] != 0
            or initial['identity']['checkpoint_sha256'] != facts['actual_initial']['sha256']
            or any(current['cohorts'][k]['packet_sha256'] != initial['cohorts'][k]['packet_sha256'] for k in current['cohorts'])):
        raise ValueError('actual initial/current fixed input/config mismatch')
    direct = actual_d_only_gradients(args.config, args.checkpoint)
    if source != {name: sha(ROOT/name) for name in source}:
        raise ValueError('readiness sources changed')
    if any(sha(path) != digest for path,digest in facts['verified_inputs'].items()):
        raise ValueError('lineage evidence changed')
    atomic_json(args.output_dir/'result.json', {
        'kind': 'XKF_DONLY_ORIGINAL_READINESS_WITH_EXPLICIT_LINEAGE', 'formal': False,
        'automatic_full_training': False, 'identity': current['identity'], 'source_sha256': source,
        'original_readiness_current_stage': current['readiness'],
        'original_dual_branch_gradient_capability': dict(current['gradients'],
            interpretation='Original frozen R2 backward capability probe, NOT actual optimizer exposure in D-only stage'),
        'actual_d_only_objective_gradient_probe': direct,
        'lineage': facts, 'true_initial_and_fixed_packets_match': True,
        'artifacts': {name: {'path': str(path), 'sha256': sha(path)} for name,path in (
            ('initial',initial_path),('current',current_path),('lineage',args.output_dir/'lineage.json'))},
        'limitation': 'Original current-stage exposure requires self/burn but this stage intentionally trains D only. Verified cumulative ancestor exposure is separate; no criterion is silently redefined.',
        'elapsed_seconds': time.monotonic()-started})
    print(json.dumps({'status': 'COMPLETE', 'output': str(args.output_dir/'result.json'),
                      'original_readiness_current_stage': current['readiness']['status'],
                      'actual_d_only_gradient_pass': direct['pass']}), flush=True)


if __name__ == '__main__':
    main()
