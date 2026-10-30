#!/usr/bin/env python3
"""Small, repeatable XKF learning/debug probe; never produces a formal main row.

The canonical split is unchanged. Training samples use fixed counter-RNG indices;
check packets are two training identities, and zgz packets are explicitly labelled
development snippets. Inference targets are zero. Every failure remains in the
output (no finite-subset averaging). Compare initial and trained JSONs to obtain a
debug readiness decision; this does not assess the final G1--G6 gates.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

import numpy as np
import torch

from config import load_config
from semkine.lie import so3_exp, so3_log
from semkine.xkf import XKFModel, fuse_states
from tools.xdl import diagnostics as D
from tools.xkf.data import build_train_dataset, collate_long
from tools.xkf.runtime import LongWindowEval


DEBUG_CRITERIA = {
    'scope': 'short-debug readiness only; never formal G1-G6 or a main-table result',
    'check_direct_kp_mean_px_lt': 3.0,
    'check_direct_root_mean_deg_lt': 20.0,
    'check_direct_ra_mpjpe_mm_lt': 25.0,
    'check_direct_positive_depth_fraction_ge': 0.99,
    'check_metrics_must_improve_vs_initial': ['kp_mean_px', 'root_mean_deg', 'ra_mpjpe_mm', 'wrist_rmse_mm'],
    'fixed_train_direct_loss_must_improve_vs_initial': True,
    'comparison_requires_same_seed_and_config_sha256': True,
    'comparison_requires_actual_saved_initial_checkpoint_sha256': True,
    'minimum_optimizer_step': 600,
    'all_training_ranks_require_nonzero_self_and_burn_exposure': True,
    'both_branches_require_finite_nonzero_gradients': True,
    'loop_fused_mean_abs_mpjpe_must_not_exceed_hold': True,
    'loop_fused_endpoint_wrist_rmse_must_improve_vs_hold': True,
    'all_predictions_and_losses_must_be_finite': True,
}
MODES = ('i', 'd', 'fused', 'd_zero', 'hold')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def packet_hash(packet):
    h = hashlib.sha256()
    for field in dataclasses.fields(packet):
        value = getattr(packet, field.name)
        h.update(field.name.encode())
        if torch.is_tensor(value):
            value = value.detach().cpu().contiguous().numpy()
            h.update(str((value.dtype, value.shape)).encode())
            h.update(value.tobytes())
    return h.hexdigest()


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def load_model(config, checkpoint, device):
    cfg = load_config(str(config))
    torch.manual_seed(int(cfg['SEED']))
    model = XKFModel(cfg=cfg)
    identity = {'config': str(config), 'config_sha256': sha(config), 'global_step': 0,
                'checkpoint': None, 'checkpoint_sha256': None, 'seed': int(cfg['SEED'])}
    if checkpoint:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        model.load_state_dict(saved['state_dict'], strict=True)
        identity.update(checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                        global_step=int(saved.get('global_step', -1)))
        saved_seed = saved.get('hyper_parameters', {}).get('SEED')
        if saved_seed is not None and int(saved_seed) != int(cfg['SEED']):
            raise ValueError('checkpoint seed differs from evaluation config')
    model = model.to(device).float().eval()
    return model, cfg, identity


def training_identity(checkpoint, step, ranks):
    """Bind the initial checkpoint and burn audit to this training/resume lineage."""
    audit = {'initial_checkpoint': None, 'exposure': [], 'exposure_pass': False}
    if checkpoint is None:
        return audit
    directory = Path(checkpoint).resolve().parent
    seen = set()
    while directory not in seen:
        seen.add(directory)
        initial = directory / 'initial_state.ckpt'
        if initial.is_file():
            audit['initial_checkpoint'] = {'path': str(initial), 'sha256': sha(initial)}
            break
        metadata = directory / 'training_metadata.json'
        if not metadata.is_file():
            break
        resumed = json.loads(metadata.read_text()).get('resumed_from')
        if not resumed:
            break
        directory = Path(resumed).resolve().parent
    directory = Path(checkpoint).resolve().parent
    for rank in range(ranks):
        path = directory / f'actual_exposure_rank{rank}.json'
        if not path.is_file():
            audit['exposure'].append({'rank': rank, 'path': str(path), 'pass': False, 'reason': 'missing'})
            continue
        value = json.loads(path.read_text())
        passed = (int(value.get('final_global_step', -1)) == step
                  and int(value.get('self_main_rows', 0)) > 0
                  and int(value.get('burn_model_rows', 0)) > 0
                  and value.get('burn_count_attribute_present') is True)
        audit['exposure'].append({'rank': rank, 'path': str(path), 'sha256': sha(path),
                                 'pass': passed, 'record': value})
    audit['exposure_pass'] = bool(audit['exposure']) and all(v['pass'] for v in audit['exposure'])
    return audit


def move_batch(batch, device):
    return {k: [p.to(device) for p in v] if isinstance(v, list) else v.to(device)
            for k, v in batch.items()}


def outputs(model, packet, x_i, z_d):
    prev = packet.prev_state.float()
    result = {'i': x_i.float(), 'd': z_d.float(),
              'fused': fuse_states(x_i.float(), z_d.float(), model.gains('fused')),
              'd_zero': fuse_states(prev, z_d.float(), model.gains('d_zero')), 'hold': prev}
    return {k: torch.where((packet.counts <= 0)[:, None], prev, v) for k, v in result.items()}


@torch.no_grad()
def state_metrics(model, pred, truth, betas):
    finite = bool(torch.isfinite(pred).all() and torch.isfinite(truth).all())
    result = {'count': len(pred), 'finite': finite}
    if not finite:
        return result
    _, jp = model._fk(pred.float(), betas.float())
    _, jg = model._fk(truth.float(), betas.float())
    delta = (jp - jg) * 1000
    relative = ((jp - jp[:, :1]) - (jg - jg[:, :1])) * 1000
    root = so3_log(so3_exp(truth[:, 3:6]).transpose(-1, -2) @ so3_exp(pred[:, 3:6])).norm(dim=-1)
    dt = (pred[:, :3] - truth[:, :3]) * 1000
    result.update(abs_mpjpe_mm=float(delta.norm(dim=-1).mean()),
                  ra_mpjpe_mm=float(relative.norm(dim=-1).mean()),
                  wrist_rmse_mm=float(dt.square().sum(-1).mean().sqrt()),
                  depth_rmse_mm=float(dt[:, 2].square().mean().sqrt()),
                  depth_bias_mm=float(dt[:, 2].mean()),
                  root_mean_deg=float(root.mean() * (180 / np.pi)),
                  positive_depth_fraction=float((pred[:, 2] > 0).float().mean()),
                  depth_min_mm=float(pred[:, 2].min() * 1000),
                  depth_max_mm=float(pred[:, 2].max() * 1000))
    result['finite'] = all(np.isfinite(v) for v in result.values())
    return result


@torch.no_grad()
def keypoint_metrics(model, aux, truth, betas, camera):
    _, joints = model._fk(truth, betas)
    fx, fy, cx, cy = model._intrinsics(camera)
    z = joints[..., 2].clamp_min(1e-3)
    px = torch.stack([fx[:, None] * joints[..., 0] / z + cx[:, None],
                      fy[:, None] * joints[..., 1] / z + cy[:, None]], -1)
    valid = ((joints[..., 2] > 1e-3) & (px[..., 0] >= 0) & (px[..., 0] < model.render_w)
             & (px[..., 1] >= 0) & (px[..., 1] < model.render_h))
    err = (aux['kp_px'] - px).norm(dim=-1)
    return {'kp_mean_px': float(err[valid].mean()) if valid.any() else None,
            'kp_inside_count': int(valid.sum()), 'kp_total_count': valid.numel()}


@torch.no_grad()
def evaluate_packets(model, packets):
    predictions = {k: [] for k in (*MODES, 'raw_d')}
    truths, betas, live_rows, kp_parts, losses = [], [], [], [], []
    hashes = []
    for packet, truth in packets:
        hashes.append(packet_hash(packet))
        # Hand context is explicit; inference must never read the score labels.
        query = dataclasses.replace(packet, target=torch.zeros_like(truth))
        x_i, z_d, aux = model.branches(query)
        predicted = outputs(model, query, x_i, z_d)
        predicted['raw_d'] = z_d
        for k, value in predicted.items():
            predictions[k].append(value)
        truths.append(truth)
        betas.append(packet.betas)
        live_rows.append(packet.counts > 0)
        kp_parts.append(keypoint_metrics(model, aux, truth, packet.betas, packet.camera_K))
        loss_packet = dataclasses.replace(packet, target=truth)
        li, _ = model._compute_loss(x_i, truth, packet.betas)
        ld, parts = model._direct_loss(z_d, aux, loss_packet)
        losses.append({'count': len(truth), 'i': float(li), 'd': float(ld),
                       **{k: float(v) for k, v in parts.items()}})
    truth, beta, live = torch.cat(truths), torch.cat(betas), torch.cat(live_rows)
    summary = {}
    for k, parts in predictions.items():
        pred = torch.cat(parts)
        # No-event packets have no direct measurement; this is not a finite mask.
        mask = live if k == 'raw_d' else torch.ones_like(live)
        summary[k] = state_metrics(model, pred[mask], truth[mask], beta[mask]) if mask.any() else {'count': 0, 'finite': False}
    total_kp = sum(p['kp_inside_count'] for p in kp_parts)
    summary['raw_d']['kp_mean_px'] = (sum(p['kp_mean_px'] * p['kp_inside_count'] for p in kp_parts
                                                       if p['kp_inside_count']) / total_kp if total_kp else None)
    n = sum(p['count'] for p in losses)
    loss = {k: sum(p[k] * p['count'] for p in losses) / n for k in losses[0] if k != 'count'}
    return {'metrics': summary, 'loss': loss, 'count': len(truth), 'live_count': int(live.sum()),
            'finite': all(p['finite'] for p in summary.values()) and all(np.isfinite(v) for v in loss.values()),
            'packet_sha256': hashes, 'kp_inside_count': total_kp}


def gradient_audit(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    x_i, z_d, aux = model.branches(batch['main'])
    li, _ = model._compute_loss(x_i, batch['main'].target, batch['main'].betas)
    ld, _ = model._direct_loss(z_d, aux, batch['main'])
    loss = li.log10() + ld.log10()
    loss.backward()
    groups = {}
    for label, direct in [('i', False), ('d', True)]:
        gradients = [p.grad for name, p in model.named_parameters()
                     if name.startswith('direct.') == direct and p.requires_grad and p.grad is not None]
        groups[label] = {'gradient_tensors': len(gradients),
                         'finite': all(bool(torch.isfinite(g).all()) for g in gradients),
                         'nonzero': any(bool((g != 0).any()) for g in gradients)}
    model.zero_grad(set_to_none=True)
    model.eval()
    return {'loss': float(loss.detach()), 'branches': groups,
            'pass': bool(torch.isfinite(loss)) and all(g['finite'] and g['nonzero'] for g in groups.values()),
            'scope': 'one training-mode main-packet backward; no optimizer update; burn checked separately'}


def bank_packet(data, row, prev, device):
    beta, cam = D.context(data, device)
    packet = D.make_packet(D.raw_events(data, row), prev, beta, cam, device, row)
    runtime = LongWindowEval(data['aux']['valid_runs_ms'], 200)
    lnes, window = runtime.lnes(data['events'], data['offsets'], row['end_ms'])
    packet.lnes = torch.from_numpy(lnes).to(device).unsqueeze(0).expand(len(prev), -1, -1, -1)
    packet.lnes_window_ms = torch.full((len(prev),), window, dtype=torch.long, device=device)
    return packet


@torch.no_grad()
def snippets(model, data, meta, device, count, steps):
    selected = np.linspace(0, len(meta['clips']) - 1, min(count, len(meta['clips'])), dtype=int)
    per_mode = {k: [] for k in MODES}
    endpoints = {k: [] for k in MODES}
    truths, betas, endpoint_truth, endpoint_beta, traces = [], [], [], [], []
    for ci in selected:
        clip = meta['clips'][ci]
        rows = [meta['rows'][i] for i in clip['end_indices'][:steps]]
        first = torch.tensor(data['gt'][rows[0]['previous_end_ms']], device=device).float()
        initial = first.clone()
        initial[2] += .05
        initial[3:6] = so3_log(so3_exp(torch.tensor([np.pi / 9, 0., 0.], device=device)) @ so3_exp(first[3:6]))
        prev = initial.repeat(len(MODES), 1)
        trace = {'clip_id': clip['clip_id'], 'initial': initial.cpu().tolist(), 'steps': []}
        for row in rows:
            packet = bank_packet(data, row, prev, device)
            x_i, z_d, _ = model.branches(packet)
            values = outputs(model, packet, x_i, z_d)
            prev = torch.stack([values[k][i] for i, k in enumerate(MODES)])
            truth = torch.tensor(data['gt'][row['end_ms']], device=device).float().unsqueeze(0)
            truths.append(truth)
            betas.append(packet.betas[:1])
            for i, k in enumerate(MODES):
                per_mode[k].append(prev[i:i + 1])
            trace['steps'].append({'packet_id': row['packet_id'], 'truth': truth[0].cpu().tolist(),
                                   'predictions': {k: prev[i].cpu().tolist() for i, k in enumerate(MODES)}})
        traces.append(trace)
        endpoint_truth.append(truth)
        endpoint_beta.append(packet.betas[:1])
        for i, k in enumerate(MODES):
            endpoints[k].append(prev[i:i + 1])
    if not truths:
        return {'finite': False, 'reason': 'no eligible clips', 'metrics': {}, 'endpoint_metrics': {}}
    metrics = {k: state_metrics(model, torch.cat(per_mode[k]), torch.cat(truths), torch.cat(betas)) for k in MODES}
    endmetrics = {k: state_metrics(model, torch.cat(endpoints[k]), torch.cat(endpoint_truth), torch.cat(endpoint_beta)) for k in MODES}
    return {'finite': all(v['finite'] for v in [*metrics.values(), *endmetrics.values()]),
            'metrics': metrics, 'endpoint_metrics': endmetrics, 'traces': traces,
            'initialization': 'GT previous state +50 mm z and left-multiplied +20 degree camera-x root rotation'}


def readiness(result, baseline):
    checks = {'finite_predictions_and_losses': all(v['finite'] for v in result['cohorts'].values())
              and all(v['finite'] for v in result['snippets'].values())}
    check = result['cohorts']['check']['metrics']['raw_d']
    limits = [('kp_mean_px', 'check_direct_kp_mean_px_lt'), ('root_mean_deg', 'check_direct_root_mean_deg_lt'),
              ('ra_mpjpe_mm', 'check_direct_ra_mpjpe_mm_lt')]
    for metric, criterion in limits:
        checks[criterion] = check.get(metric) is not None and check[metric] < DEBUG_CRITERIA[criterion]
    checks['check_direct_positive_depth_fraction_ge'] = check.get('positive_depth_fraction', 0) >= DEBUG_CRITERIA['check_direct_positive_depth_fraction_ge']
    checks['finite_active_gradients_both_branches'] = result.get('gradients', {}).get('pass')
    checks['minimum_optimizer_step'] = result['identity']['global_step'] >= DEBUG_CRITERIA['minimum_optimizer_step']
    checks['observed_self_and_burn_exposure_all_ranks'] = result.get('training_identity', {}).get('exposure_pass', False)
    checks['matched_initial_comparison'] = None
    if baseline is not None:
        packet_match = all(result['cohorts'][k]['packet_sha256'] == baseline['cohorts'][k]['packet_sha256']
                           for k in result['cohorts'])
        identity_match = all(result['identity'].get(k) == baseline['identity'].get(k)
                             for k in ('seed', 'config_sha256'))
        actual_initial = result.get('training_identity', {}).get('initial_checkpoint')
        saved_initial_match = bool(actual_initial and baseline['identity'].get('checkpoint')
                                   and baseline['identity'].get('checkpoint_sha256') == actual_initial['sha256']
                                   and baseline['identity']['global_step'] == 0)
        matched = packet_match and identity_match and saved_initial_match
        checks['matched_initial_comparison'] = matched
        checks['same_seed_and_config_as_initial'] = identity_match
        checks['actual_saved_initial_checkpoint_identity'] = saved_initial_match
        old = baseline['cohorts']['check']['metrics']['raw_d']
        for key in DEBUG_CRITERIA['check_metrics_must_improve_vs_initial']:
            checks[f'check_{key}_improves'] = bool(matched and check.get(key) is not None and old.get(key) is not None
                                                 and check[key] < old[key])
        checks['train_direct_loss_improves'] = bool(matched and result['cohorts']['train']['loss']['d'] < baseline['cohorts']['train']['loss']['d'])
    for name, snippet in result['snippets'].items():
        m, e = snippet['metrics'], snippet['endpoint_metrics']
        checks[f'{name}_loop_fused_not_worse_than_hold'] = bool(snippet['finite'] and m['fused']['abs_mpjpe_mm'] <= m['hold']['abs_mpjpe_mm'])
        checks[f'{name}_endpoint_wrist_recovers'] = bool(snippet['finite'] and e['fused']['wrist_rmse_mm'] < e['hold']['wrist_rmse_mm'])
    complete = all(v is not None for v in checks.values())
    return {'status': ('PASS' if all(checks.values()) else 'FAIL') if complete else 'INCOMPLETE',
            'ready_for_full_training': complete and all(checks.values()), 'criteria': DEBUG_CRITERIA, 'checks': checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', help='any explicitly named debug checkpoint; omitted means deterministic random init')
    ap.add_argument('--output', required=True)
    ap.add_argument('--compare', help='initial-step0 debug JSON from the same fixed inputs')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--cpu-threads', type=int, default=4)
    ap.add_argument('--train-batches', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--train-start-index', type=int, default=0)
    ap.add_argument('--check-rows', type=int, default=8, help='fixed evenly spaced packets per sequence')
    ap.add_argument('--clips', type=int, default=1, help='frozen-bank snippets per zgz sequence')
    ap.add_argument('--steps', type=int, default=10)
    ap.add_argument('--gradients', action='store_true')
    args = ap.parse_args()
    if min(args.train_batches, args.batch_size, args.check_rows, args.clips, args.steps) < 1:
        ap.error('all probe sizes must be positive')
    if args.device == 'cuda':
        ap.error('use an explicit CUDA index and an explicitly restricted CUDA_VISIBLE_DEVICES')
    if args.gradients and args.batch_size < 2:
        ap.error('training-mode gradient audit requires --batch-size >=2')
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    device = torch.device(args.device)
    model, cfg, identity = load_model(args.config, args.checkpoint, device)
    final_index = args.train_start_index + args.train_batches * args.batch_size
    dataset = build_train_dataset(cfg, max_steps=max(1, (final_index + 1023) // 1024))
    train_batches = []
    for i in range(args.train_batches):
        start = args.train_start_index + i * args.batch_size
        train_batches.append(move_batch(collate_long([dataset[j] for j in range(start, start + args.batch_size)]), device))
    result = {'kind': 'XKF_DEBUG_ONLY', 'partial_smoke': True, 'formal_gate_assessed': False,
              'finite_subset_reduction': False, 'identity': identity, 'device': str(device),
              'cohorts': {'train': evaluate_packets(model, [(b['main'], b['main'].target) for b in train_batches])},
              'snippets': {}, 'sampling': vars(args), 'gains': model.gains().detach().cpu().tolist(),
              'xkf_updates': int(model.xkf_updates), 'producer_sha256': sha(__file__)}
    result['training_identity'] = training_identity(args.checkpoint, identity['global_step'], int(cfg['TRAIN']['DEVICES']))
    bank, banksha = D.load_bank()
    result['bank_sha256'] = banksha
    for cohort, names in [('check', ('lyq_local', 'ch_global')), ('zgz', ('zgz_local', 'zgz_global'))]:
        packets = []
        for name in names:
            meta = bank['sequences'][name]
            data = D.load_data(cfg, name, meta)
            selected = np.linspace(0, len(meta['rows']) - 1, min(args.check_rows, len(meta['rows'])), dtype=int)
            for ri in selected:
                row = meta['rows'][ri]
                prev = torch.tensor(data['gt'][row['previous_end_ms']], device=device).float().unsqueeze(0)
                packet = bank_packet(data, row, prev, device)
                truth = torch.tensor(data['gt'][row['end_ms']], device=device).float().unsqueeze(0)
                packets.append((packet, truth))
            if cohort == 'zgz':
                result['snippets'][name] = snippets(model, data, meta, device, args.clips, args.steps)
        result['cohorts'][cohort] = evaluate_packets(model, packets)
        print(json.dumps({'phase': 'debug_eval', 'cohort_complete': cohort}), flush=True)
    if args.gradients:
        # Run last: batchnorm buffers must not alter any of the checkpoint evaluations.
        result['gradients'] = gradient_audit(model, train_batches[0])
    baseline = json.loads(Path(args.compare).read_text()) if args.compare else None
    result['readiness'] = readiness(result, baseline)
    result['elapsed_seconds'] = time.perf_counter() - started
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(safe_json(result), indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'kind': result['kind'], 'debug_readiness': result['readiness']['status']}), flush=True)


if __name__ == '__main__':
    main()
