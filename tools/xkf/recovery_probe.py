#!/usr/bin/env python3
"""Frozen-checkpoint exploratory own-loop recovery probe (no optimizer or formal gates).

Run --prepare first to freeze packet selection, initial conditions and interventions.
Each --seed run writes every trajectory and all frames, including censored runs.
The historical debug metric called wrist_rmse_mm measures MANO parameter t; this
probe deliberately records parameter t and the physical MANO wrist separately.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

import numpy as np
import torch

from semkine.lie import so3_exp, so3_log
from semkine.xkf import fuse_states
from semkine.xyz_recovery_study import RecoveryXYZModel
from tools.xdl import diagnostics as D
from tools.xkf.debug_eval import load_model, safe_json, sha, snippets
from tools.xkf.runtime import LongWindowEval

OUT = REPO / 'outputs/xkf/local_recovery_20261015/rollout'
DEBUG = REPO / 'outputs/xkf/debug_20261015_r1'
MODES = ('i', 'd', 'fused', 'd_zero', 'hold', 'fused_keep_t')
I_MODES = ('i', 'fused', 'fused_keep_t')
CONDITIONS = (
    ('clean', 0., 0., 0.), ('z_plus50', .05, 0., 0.), ('z_minus50', -.05, 0., 0.),
    ('rx_plus20', 0., 20., 0.), ('rx_minus20', 0., -20., 0.),
    ('ry_plus20', 0., 0., 20.), ('ry_minus20', 0., 0., -20.),
    ('z_plus50_rx_plus20', .05, 20., 0.), ('z_minus50_rx_minus20', -.05, -20., 0.),
)
METRICS = (
    'abs_mpjpe_mm', 'ra_mpjpe_mm', 'abs_mpvpe_mm', 'ra_mpvpe_mm',
    'state_t_error_mm', 'state_t_squared_mm2', 'state_depth_error_mm', 'state_depth_squared_mm2',
    'physical_wrist_error_mm', 'physical_wrist_squared_mm2',
    'physical_wrist_depth_error_mm', 'physical_wrist_depth_squared_mm2',
    'root_error_deg', 'fingers_rmse_rad', 'absolute_joint_mse_mm2',
    'relative_joint_mse_mm2', 'wrist_relative_cross_mm2',
)


def write_json(path, value):
    Path(path).write_text(json.dumps(safe_json(value), indent=2, allow_nan=False) + '\n')


def schedule(bank, steps=10):
    clips, excluded = [], []
    for name in ('zgz_local', 'zgz_global'):
        meta = bank['sequences'][name]
        if name == 'zgz_local':
            for rid, run in enumerate(meta['valid_runs_ms']):
                rows = [r for r in meta['rows'] if r['run_id'] == rid]
                if not rows:
                    excluded.append({'sequence': name, 'run_id': rid, 'run_ms': run,
                                     'reason': 'no legal W50 bank packet'})
                    continue
                for start, label in ((0, 'cold'), (3, 'warm3')):
                    selected = rows[start:start + steps]
                    if not selected:
                        excluded.append({'sequence': name, 'run_id': rid, 'start_kind': label,
                                         'reason': 'no packet after three-packet warm offset'})
                        continue
                    clips.append({'sequence': name, 'cohort': f'local_{label}',
                                  'clip_id': f'{name}:run{rid}:{label}', 'run_id': rid,
                                  'row_indices': [r['index'] for r in selected],
                                  'available_steps': len(rows) - start, 'observed_steps': len(selected),
                                  'requested_steps': steps, 'censored': len(selected) < steps,
                                  'initial_gt_ms': selected[0]['previous_end_ms']})
        for clip in meta['clips']:
            indices = clip['end_indices'][:steps]
            clips.append({'sequence': name, 'cohort': 'local_bank' if name == 'zgz_local' else 'global_bank',
                          'clip_id': clip['clip_id'], 'run_id': clip['run_id'], 'row_indices': indices,
                          'available_steps': len(clip['end_indices']), 'observed_steps': len(indices),
                          'requested_steps': steps, 'censored': len(indices) < steps,
                          'initial_gt_ms': meta['rows'][indices[0]]['previous_end_ms']})
    return clips, excluded


def prepare(out, steps):
    bank, banksha = D.load_bank()
    clips, excluded = schedule(bank, steps)
    identities = {}
    for seed in (3407, 3408, 3409):
        cfg = REPO / f'configs/xkf/xkf_s{seed}.yaml'
        checkpoint = DEBUG / f'xkf_s{seed}/xkf_s{seed}-step=2000.ckpt'
        identities[str(seed)] = {'config': str(cfg), 'config_sha256': sha(cfg),
                                 'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint)}
    frozen = json.loads((DEBUG / 'frozen_manifest.json').read_text())['source_sha256']
    source_hashes = {name: sha(REPO / name) for name in frozen}
    if source_hashes != frozen:
        raise RuntimeError('debug frozen source manifest mismatch')
    for name in ('tools/xdl/diagnostics.py', 'semkine/eval_track.py'):
        source_hashes[name] = sha(REPO / name)
    protocol = {
        'kind': 'XKF_EXPLORATORY_RECOVERY_NO_TRAINING', 'formal_gate_assessed': False,
        'main_row': False, 'producer_sha256': sha(__file__), 'bank_sha256': banksha,
        'dependency_sha256': source_hashes,
        'horizon_steps': steps, 'conditions': CONDITIONS, 'modes': MODES,
        'condition_definition': 'GT previous state; z additive in metres; root left-multiplied in camera axes',
        'fused_keep_t_definition': 'Own-loop fused root/fingers; copy incoming parameter t exactly each step. Causal diagnostic intervention only.',
        'warm3_definition': 'Reset to declared GT+perturbation at packet index3, NOT a GT reset during rollout; LNES contains causal history from run start.',
        'inference': 'FP32 eval/no_grad; matmul/cuDNN TF32 disabled, float32 highest precision; all targets zero; only initial uses GT; each mode owns its previous state',
        'sampling': 'all nonempty local run starts, warm3 if any remaining packet, all24 bank local and all24 bank global clips; no response-dependent selection',
        'censoring': 'No padding or extrapolation. Keep short runs with observed horizon; summarize all and full-horizon separately; list empty runs.',
        'aggregation': 'Per cohort/condition/mode: pooled observed frames and equal-trajectory means; endpoint at last observed step, separately full-horizon.',
        'd_cache': 'One direct_measurement per unique packet per seed; independent of prior; old-parity reference evaluated separately.',
        'state_t_vs_wrist': 'Historical wrist_rmse is parameter t. Physical wrist is FK OpenPose joint0; both scored explicitly.',
        'error_decomposition': 'mean_j ||delta_j||² = ||delta_wrist||² + mean_j ||delta_j-delta_wrist||² + 2 delta_wrist dot mean_j(delta_j-delta_wrist); mm².',
        'parity': {'reference': 'original debug_eval.snippets first local/global clip vs saved CPU FP32 and optimized GPU rollout',
                   'required': 'CPU replay matches saved reference; optimized matches sameGPU original; crossbackend mean abs error within0.2mm. Crossbackend state differences retained separately.',
                   'state_translation_atol_m': .0002, 'state_pose_atol_rad': .002, 'abs_mpjpe_atol_mm': .2},
        'preflight_amendment': 'Before expanded inference: v1 failed strict crossbackend state equality although exact CPU replay is bitwise identical and disabling TF32 gives sameGPU batching differences below0.0006mm. Archive parity_preflight_v1 and seed parity_* files; sample/conditions/modes unchanged. This is a backend-sensitive recurrence, not permission to relax effectiveness gates.',
        'gpu_assignment': {'3407': 2, '3408': 4, '3409': 6},
        'identities': identities, 'clips': clips, 'excluded': excluded,
    }
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != json.loads(json.dumps(protocol)):
            raise RuntimeError('existing protocol differs; do not overwrite after observing results')
    else:
        write_json(path, protocol)
    print(json.dumps({'protocol': str(path), 'clips': dict(__import__('collections').Counter(c['cohort'] for c in clips)),
                      'excluded': len(excluded)}), flush=True)


def initial_states(clean):
    result = []
    for _, z, rx, ry in CONDITIONS:
        state = clean.clone()
        state[2] += z
        if rx or ry:
            perturb = clean.new_tensor([math.radians(rx), math.radians(ry), 0.])
            state[3:6] = so3_log(so3_exp(perturb) @ so3_exp(clean[3:6]))
        result.append(state)
    return torch.stack(result)


@torch.no_grad()
def frame_metrics(model, pred, truth, beta):
    """One metric vector per state, with an exact signed geometric cross term."""
    n = len(pred)
    truth = truth.expand(n, -1)
    beta = beta.expand(n, -1)
    vp, jp = model._fk(pred, beta)
    vg, jg = model._fk(truth, beta)
    dj, dv = (jp - jg) * 1000, (vp - vg) * 1000
    dw = dj[:, 0]
    rj = dj - dw[:, None]
    rv = dv - dw[:, None]
    dt = (pred[:, :3] - truth[:, :3]) * 1000
    root = so3_log(so3_exp(truth[:, 3:6]).transpose(-1, -2) @ so3_exp(pred[:, 3:6])).norm(dim=-1)
    values = (
        dj.norm(dim=-1).mean(-1), rj.norm(dim=-1).mean(-1), dv.norm(dim=-1).mean(-1), rv.norm(dim=-1).mean(-1),
        dt.norm(dim=-1), dt.square().sum(-1), dt[:, 2], dt[:, 2].square(),
        dw.norm(dim=-1), dw.square().sum(-1), dw[:, 2], dw[:, 2].square(),
        root * 180 / math.pi, (pred[:, 6:] - truth[:, 6:]).square().mean(-1).sqrt(),
        dj.square().sum(-1).mean(-1), rj.square().sum(-1).mean(-1), 2 * (dw * rj.mean(1)).sum(-1),
    )
    return torch.stack(values, -1), dw, dt


class PacketCache:
    def __init__(self, model, data, device):
        self.model, self.data, self.device = model, data, device
        self.beta, self.camera = D.context(data, device)
        self.runtime = LongWindowEval(data['aux']['valid_runs_ms'], 200)
        self.cache = {}

    @torch.no_grad()
    def packet(self, row, prev):
        key = row['packet_id']
        if key not in self.cache:
            raw = D.raw_events(self.data, row)
            image, window = self.runtime.lnes(self.data['events'], self.data['offsets'], row['end_ms'])
            lnes = torch.from_numpy(image).to(self.device).unsqueeze(0)
            packet = D.make_packet(raw, prev[:1], self.beta, self.camera, self.device, row)
            packet.lnes = lnes
            packet.lnes_window_ms = torch.tensor([window], device=self.device, dtype=torch.long)
            assert not bool(packet.target.any())
            z, _ = self.model.direct_measurement(packet)
            self.cache[key] = (raw, lnes, window, z.detach())
        raw, lnes, window, z = self.cache[key]
        packet = D.make_packet(raw, prev, self.beta, self.camera, self.device, row)
        packet.lnes = lnes.expand(len(prev), -1, -1, -1)
        packet.lnes_window_ms = torch.full((len(prev),), window, device=self.device, dtype=torch.long)
        assert not bool(packet.target.any())
        return packet, z


@torch.no_grad()
def rollout(model, data, meta, clip, cache):
    device = cache.device
    rows = [meta['rows'][i] for i in clip['row_indices']]
    clean = torch.tensor(data['gt'][clip['initial_gt_ms']], device=device).float()
    initial = initial_states(clean)
    prev = initial[:, None, :].expand(-1, len(MODES), -1).clone()
    imode = torch.tensor([MODES.index(k) for k in I_MODES], device=device)
    predictions, truths, metrics, wrist_errors, state_errors, windows, counts = [], [], [], [], [], [], []
    D.reset(model)
    for row in rows:
        incoming = prev.index_select(1, imode).reshape(-1, 51)
        packet, direct = cache.packet(row, incoming)
        motion = RecoveryXYZModel.forward_packet(model, packet).view(len(CONDITIONS), len(I_MODES), 51)
        z = direct.expand(len(CONDITIONS), -1)
        nxt = prev.clone()
        nxt[:, MODES.index('i')] = motion[:, 0]
        nxt[:, MODES.index('d')] = z
        nxt[:, MODES.index('fused')] = fuse_states(motion[:, 1], z, model.gains('fused'))
        nxt[:, MODES.index('d_zero')] = fuse_states(prev[:, MODES.index('d_zero')], z, model.gains('d_zero'))
        intervention = fuse_states(motion[:, 2], z, model.gains('fused'))
        intervention[:, :3] = prev[:, MODES.index('fused_keep_t'), :3]
        nxt[:, MODES.index('fused_keep_t')] = intervention
        if int(packet.counts[0]) == 0:
            nxt = prev
        truth = torch.tensor(data['gt'][row['end_ms']], device=device).float().unsqueeze(0)
        score, dw, dt = frame_metrics(model, nxt.reshape(-1, 51), truth, cache.beta)
        if not torch.isfinite(nxt).all() or not torch.isfinite(score).all():
            raise FloatingPointError(f'nonfinite trajectory {clip["clip_id"]} {row["packet_id"]}')
        prev = nxt
        predictions.append(nxt.cpu().numpy())
        truths.append(truth[0].cpu().numpy())
        metrics.append(score.view(len(CONDITIONS), len(MODES), -1).cpu().numpy())
        wrist_errors.append(dw.view(len(CONDITIONS), len(MODES), 3).cpu().numpy())
        state_errors.append(dt.view(len(CONDITIONS), len(MODES), 3).cpu().numpy())
        windows.append(int(packet.lnes_window_ms[0])); counts.append(int(packet.counts[0]))
    D.reset(model)
    return {'initial': initial.cpu().numpy(), 'clean': clean.cpu().numpy(),
            'predictions': np.stack(predictions), 'truth': np.stack(truths), 'metrics': np.stack(metrics),
            'physical_wrist_error_xyz_mm': np.stack(wrist_errors), 'state_t_error_xyz_mm': np.stack(state_errors),
            'lnes_window_ms': np.array(windows), 'n_raw_events': np.array(counts),
            'end_ms': np.array([r['end_ms'] for r in rows]), 'packet_ids': np.array([r['packet_id'] for r in rows])}


def summarize_values(values):
    means = np.asarray(values, dtype=np.float64).mean(0)
    result = dict(zip(METRICS, means.tolist()))
    for out, source in [('state_t_rmse_mm', 'state_t_squared_mm2'),
                        ('state_depth_rmse_mm', 'state_depth_squared_mm2'),
                        ('physical_wrist_rmse_mm', 'physical_wrist_squared_mm2'),
                        ('physical_wrist_depth_rmse_mm', 'physical_wrist_depth_squared_mm2')]:
        result[out] = math.sqrt(max(0., result[source]))
    return result


def make_rows(clip, result):
    rows = []
    for ci, (condition, *_rest) in enumerate(CONDITIONS):
        for mi, mode in enumerate(MODES):
            values = result['metrics'][:, ci, mi]
            rows.append({**clip, 'condition': condition, 'mode': mode,
                         'mean': summarize_values(values), 'first': summarize_values(values[:1]),
                         'endpoint': summarize_values(values[-1:]),
                         'lnes_first_ms': int(result['lnes_window_ms'][0])})
    return rows


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['cohort'], row['condition'], row['mode'], 'all_observed')].append(row)
        if not row['censored']:
            groups[(row['cohort'], row['condition'], row['mode'], 'full_horizon')].append(row)
    result = []
    for (cohort, condition, mode, scope), group in sorted(groups.items()):
        weights = np.asarray([r['observed_steps'] for r in group], dtype=np.float64)
        values = np.asarray([[r['mean'][k] for k in METRICS] for r in group])
        pooled = (values * weights[:, None]).sum(0) / weights.sum()
        result.append({'cohort': cohort, 'condition': condition, 'mode': mode, 'scope': scope,
                       'n_trajectories': len(group), 'n_censored': sum(r['censored'] for r in group),
                       'n_frames': int(weights.sum()), 'pooled_frames': summarize_values(pooled[None]),
                       'equal_trajectory': summarize_values(values),
                       'first': summarize_values([[r['first'][k] for k in METRICS] for r in group]),
                       'endpoint': summarize_values([[r['endpoint'][k] for k in METRICS] for r in group])})
    return result


def parity_difference(a, b):
    delta = np.asarray(a) - np.asarray(b)
    return {'max_translation_difference_m': float(np.max(np.abs(delta[..., :3]))),
            'max_pose_difference_rad': float(np.max(np.abs(delta[..., 3:])))}


@torch.no_grad()
def parity(model, cfg, seed, device, protocol):
    stored_path = DEBUG / f'pipeline/evaluations/xkf_s{seed}_step2000.json'
    stored = json.loads(stored_path.read_text())
    bank, _ = D.load_bank()
    results = {}
    for name in ('zgz_local', 'zgz_global'):
        meta = bank['sequences'][name]
        data = D.load_data(cfg, name, meta)
        reference = snippets(model, data, meta, device, 1, 10)
        previous = stored['snippets'][name]
        original_modes = MODES[:5]
        def array(value):
            return np.array([[step['predictions'][mode] for mode in original_modes]
                             for step in value['traces'][0]['steps']])
        original, old = array(reference), array(previous)
        clip = next(c for c in protocol['clips'] if c['sequence'] == name and c['cohort'].endswith('_bank'))
        optimized = rollout(model, data, meta, clip, PacketCache(model, data, device))
        optimized_array = optimized['predictions'][:10, 7, :5]
        result = {'stored_cpu_fp32': parity_difference(original, old),
                  'cached_batched_vs_original_same_gpu': parity_difference(optimized_array, original),
                  'max_abs_mpjpe_difference_mm': max(abs(reference['metrics'][k]['abs_mpjpe_mm'] - previous['metrics'][k]['abs_mpjpe_mm']) for k in original_modes),
                  'stored_sha256': sha(stored_path)}
        tolerances = protocol['parity']
        def within(value):
            return (value['max_translation_difference_m'] <= tolerances['state_translation_atol_m']
                    and value['max_pose_difference_rad'] <= tolerances['state_pose_atol_rad'])
        result['crossbackend_state_within_original_tolerance'] = within(result['stored_cpu_fp32'])
        cpu_path = OUT / f's{seed}/parity_cpu_reproduction.json'
        cpu_replay = json.loads(cpu_path.read_text())[name]
        result['cpu_replay_sha256'] = sha(cpu_path)
        result['cpu_replay_bitwise_match'] = (cpu_replay['stored_cpu_fp32']['max_translation_difference_m'] == 0
                                             and cpu_replay['stored_cpu_fp32']['max_pose_difference_rad'] == 0)
        result['pass'] = (result['cpu_replay_bitwise_match']
                          and within(result['cached_batched_vs_original_same_gpu'])
                          and result['max_abs_mpjpe_difference_mm'] <= tolerances['abs_mpjpe_atol_mm'])
        results[name] = result
    return results


def run(out, seed, device, parity_only=False, limit=None):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision('highest')
    protocol = json.loads((out / 'protocol.json').read_text())
    if protocol['producer_sha256'] != sha(__file__):
        raise RuntimeError('producer changed after protocol freeze')
    bank, banksha = D.load_bank()
    if banksha != protocol['bank_sha256']:
        raise RuntimeError('packet bank changed after freeze')
    def verify_dependencies():
        for name, expected in protocol['dependency_sha256'].items():
            if sha(REPO / name) != expected:
                raise RuntimeError(f'dependency changed after freeze: {name}')
    verify_dependencies()
    identity = protocol['identities'][str(seed)]
    for key in ('config', 'checkpoint'):
        if sha(identity[key]) != identity[key + '_sha256']:
            raise RuntimeError(f'{key} changed after freeze')
    model, cfg, actual = load_model(identity['config'], identity['checkpoint'], device)
    target = out / f's{seed}'
    target.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    checked = parity(model, cfg, seed, device, protocol)
    write_json(target / 'parity.json', checked)
    print(json.dumps({'phase': 'parity', 'seed': seed, 'results': checked}), flush=True)
    if not all(r['pass'] for r in checked.values()):
        raise RuntimeError('parity failed; stop without expanded rollout')
    if parity_only:
        return
    selected = protocol['clips'][:limit] if limit else protocol['clips']
    rows, caches = [], {}
    rawdir = target / 'raw'
    rawdir.mkdir(exist_ok=True)
    with (target / 'per_clip.jsonl').open('w') as file:
        for index, clip in enumerate(selected):
            name = clip['sequence']
            meta = bank['sequences'][name]
            if name not in caches:
                data = D.load_data(cfg, name, meta)
                caches[name] = PacketCache(model, data, device)
            cache = caches[name]
            value = rollout(model, cache.data, meta, clip, cache)
            np.savez_compressed(rawdir / f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz', **value)
            newrows = make_rows(clip, value)
            for row in newrows:
                file.write(json.dumps(safe_json(row), allow_nan=False) + '\n')
            file.flush(); rows.extend(newrows)
            progress = {'phase': 'rollout', 'seed': seed, 'completed_clips': index + 1, 'total_clips': len(selected),
                        'last_clip': clip['clip_id'], 'elapsed_seconds': time.perf_counter() - started,
                        'cached_direct_packets': {n: len(c.cache) for n, c in caches.items()}}
            write_json(target / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    verify_dependencies()
    result = {'kind': protocol['kind'], 'formal_gate_assessed': False, 'identity': actual,
              'protocol_sha256': sha(out / 'protocol.json'), 'metrics': METRICS,
              'device': str(device), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'gains_fused': model.gains('fused').cpu().tolist(), 'gains_d_zero': model.gains('d_zero').cpu().tolist(),
              'parity': checked, 'partial': limit is not None, 'clip_count': len(selected),
              'elapsed_seconds': time.perf_counter() - started, 'aggregate': aggregate(rows)}
    write_json(target / 'summary.json', result)
    print(json.dumps({'phase': 'complete', 'seed': seed, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--seed', type=int, choices=(3407, 3408, 3409))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--limit', type=int, help='explicit partial smoke; never reported as full')
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    if args.prepare:
        prepare(args.out, args.steps)
    else:
        if args.seed is None:
            parser.error('--seed is required unless --prepare')
        run(args.out, args.seed, torch.device(args.device), args.parity_only, args.limit)


if __name__ == '__main__':
    main()
