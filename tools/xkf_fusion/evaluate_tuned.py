#!/usr/bin/env python3
"""Own-loop fusion evaluation after isolated D-only continuation, on the unchanged112 clips.

The six historical modes remain controls. Each added candidate owns its incoming
state and runs I again on that state. D is cached once per packet and temperature.
All inference targets are zero; GT enters only initialization and scoring.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

import numpy as np
import torch

from mano_layer import ManoLayer
from semkine.xkf import fuse_states
from semkine.xkf_fusion import FusionPolicy, XKFFusionModel
from tools.xkf import recovery_probe as R
from tools.xkf import debug_eval as E
from tools.xkf_repair import evaluate as RE
from tools.xdl.make_row import write_json

R2 = ROOT / 'outputs/xkf/repair_20261015_r2'
BASE_MODES = tuple(R.MODES)


def sources():
    frozen = json.loads((R2 / 'frozen_manifest.json').read_text())['source_sha256']
    for name, expected in frozen.items():
        if E.sha(ROOT / name) != expected:
            raise ValueError(f'r2 frozen source changed: {name}')
    return {**frozen, **{name: E.sha(ROOT / name) for name in (
        'semkine/xkf_fusion.py', 'tools/xkf_fusion/evaluate.py', 'tools/xkf_fusion/evaluate_tuned.py')}}


class Candidate:
    """Resolve coefficients once at clip entry; never switch calibration mid-loop."""
    def __init__(self, value, seed):
        spec = copy.deepcopy(value)
        self.source = None
        descriptors = spec.pop('direct_correction_by_seed', None)
        descriptor = spec.pop('direct_correction_file', None)
        if descriptors is not None:
            if descriptor is not None:
                raise ValueError('provide one calibration source mechanism')
            descriptor = descriptors[str(seed)]
        self.document = None
        if descriptor is not None:
            if 'direct_correction' in spec:
                raise ValueError('provide inline or file calibration, not both')
            path = Path(descriptor['path'])
            if not path.is_absolute():
                path = ROOT / path
            digest = E.sha(path)
            if descriptor['sha256'] != digest:
                raise ValueError('external calibration SHA mismatch')
            self.source = {'path': str(path), 'sha256': digest}
            self.document = json.loads(path.read_text())
        self.spec = spec
        FusionPolicy(spec)

    def for_clip(self, clip_id):
        spec = copy.deepcopy(self.spec)
        if self.document is not None:
            if 'clip_specs' in self.document:
                spec['direct_correction'] = self.document['clip_specs'][clip_id]
            elif 'spec' in self.document:
                spec['direct_correction'] = self.document['spec']
            else:
                spec['direct_correction'] = self.document
        return FusionPolicy(spec)

    def unchanged(self):
        return self.source is None or E.sha(self.source['path']) == self.source['sha256']


class PacketCache:
    def __init__(self, model, data, device):
        self.model, self.data, self.device = model, data, device
        self.beta, self.camera = R.D.context(data, device)
        self.runtime = R.LongWindowEval(data['aux']['valid_runs_ms'], 200)
        self.inputs, self.measurements = {}, {}

    @torch.no_grad()
    def packet(self, row, prev, temperatures):
        key = row['packet_id']
        if key not in self.inputs:
            raw = R.D.raw_events(self.data, row)
            image, window = self.runtime.lnes(self.data['events'], self.data['offsets'], row['end_ms'])
            self.inputs[key] = (raw, torch.from_numpy(image).to(self.device).unsqueeze(0), window)
        raw, lnes, window = self.inputs[key]
        packet = R.D.make_packet(raw, prev, self.beta, self.camera, self.device, row)
        packet.lnes = lnes.expand(len(prev), -1, -1, -1)
        packet.lnes_window_ms = torch.full((len(prev),), window, device=self.device, dtype=torch.long)
        if bool(packet.target.any()):
            raise ValueError('inference packet has a nonzero target')
        for temperature in temperatures:
            if (key, temperature) not in self.measurements:
                one = R.D.make_packet(raw, prev[:1], self.beta, self.camera, self.device, row)
                one.lnes = lnes
                one.lnes_window_ms = packet.lnes_window_ms[:1]
                z, aux = self.model.measure_direct(one, temperature)
                keep = {k: v.detach() for k, v in aux.items() if k in ('joints0', 'kp_px', 'w', 'R', 'dgp_covariance')}
                self.measurements[(key, temperature)] = (z.detach(), keep)
        return packet, {t: self.measurements[(key, t)] for t in temperatures}


@torch.no_grad()
def rollout(model, meta, clip, cache, candidates):
    policies = [candidate.for_clip(clip['clip_id']) for candidate in candidates]
    modes = BASE_MODES + tuple(policy.name for policy in policies)
    motion_modes = ('i', 'fused', 'fused_keep_t') + tuple(policy.name for policy in policies)
    temperatures = sorted({1., *(policy.temperature_multiplier for policy in policies)})
    rows = [meta['rows'][i] for i in clip['row_indices']]
    clean = torch.tensor(cache.data['gt'][clip['initial_gt_ms']], device=cache.device).float()
    initial = R.initial_states(clean)
    prev = initial[:, None, :].expand(-1, len(modes), -1).clone()
    collected = {k: [] for k in ('predictions', 'truth', 'metrics', 'physical_wrist_error_xyz_mm',
                               'state_t_error_xyz_mm', 'lnes_window_ms', 'n_raw_events',
                               'motion_predictions', 'corrected_direct', 'effective_gains')}
    R.D.reset(model)
    for row in rows:
        # Preserve the original 27-row batch for the three historical I paths.
        # Additional candidates are independent batches of at most three modes.
        motion = {}
        for begin in range(0, len(motion_modes), 3):
            labels = motion_modes[begin:begin + 3]
            selected = torch.tensor([modes.index(name) for name in labels], device=cache.device)
            incoming = prev.index_select(1, selected).reshape(-1, 51)
            packet, measurements = cache.packet(row, incoming, temperatures)
            states = model.incremental_prediction(packet).view(len(R.CONDITIONS), len(labels), 51)
            motion.update({name: states[:, i] for i, name in enumerate(labels)})
        z = measurements[1.][0].expand(len(R.CONDITIONS), -1)
        nxt = prev.clone()
        nxt[:, modes.index('i')] = motion['i']
        nxt[:, modes.index('d')] = z
        nxt[:, modes.index('fused')] = fuse_states(motion['fused'], z, model.gains('fused'))
        nxt[:, modes.index('d_zero')] = fuse_states(prev[:, modes.index('d_zero')], z, model.gains('d_zero'))
        keep_t = fuse_states(motion['fused_keep_t'], z, model.gains('fused'))
        keep_t[:, :3] = prev[:, modes.index('fused_keep_t'), :3]
        nxt[:, modes.index('fused_keep_t')] = keep_t
        count = packet.counts[:1].expand(len(R.CONDITIONS))
        corrected, gains = [], []
        for policy in policies:
            direct, aux = measurements[policy.temperature_multiplier]
            state, zd, k = policy.fuse(motion[policy.name], direct.expand(len(R.CONDITIONS), -1),
                                      aux, prev[:, modes.index(policy.name)], count, model.gains('fused'))
            nxt[:, modes.index(policy.name)] = state
            corrected.append(zd[0]); gains.append(k[0])
        if int(count[0]) == 0:
            nxt = prev
        truth = torch.tensor(cache.data['gt'][row['end_ms']], device=cache.device).float().unsqueeze(0)
        score, wrist, state_t = R.frame_metrics(model, nxt.reshape(-1, 51), truth, cache.beta)
        if not bool(torch.isfinite(nxt).all() and torch.isfinite(score).all()):
            raise FloatingPointError(f'nonfinite trajectory: {clip["clip_id"]} {row["packet_id"]}')
        collected['predictions'].append(nxt.cpu().numpy())
        collected['truth'].append(truth[0].cpu().numpy())
        collected['metrics'].append(score.view(9, len(modes), -1).cpu().numpy())
        collected['physical_wrist_error_xyz_mm'].append(wrist.view(9, len(modes), 3).cpu().numpy())
        collected['state_t_error_xyz_mm'].append(state_t.view(9, len(modes), 3).cpu().numpy())
        collected['motion_predictions'].append(torch.stack([motion[k] for k in motion_modes], 1).cpu().numpy())
        collected['corrected_direct'].append(torch.stack(corrected).cpu().numpy())
        collected['effective_gains'].append(torch.stack(gains).cpu().numpy())
        collected['lnes_window_ms'].append(int(packet.lnes_window_ms[0]))
        collected['n_raw_events'].append(int(count[0]))
        prev = nxt
    R.D.reset(model)
    return {**{key: np.stack(values) for key, values in collected.items()},
            'initial': initial.cpu().numpy(), 'clean': clean.cpu().numpy(),
            'end_ms': np.array([r['end_ms'] for r in rows]), 'packet_ids': np.array([r['packet_id'] for r in rows])}


def per_clip_rows(clip, raw, modes):
    result = []
    for ci, condition in enumerate(R.CONDITIONS):
        for mi, mode in enumerate(modes):
            v = raw['metrics'][:, ci, mi]
            result.append(dict(clip, condition=condition[0], mode=mode, mean=R.summarize_values(v),
                               first=R.summarize_values(v[:1]), endpoint=R.summarize_values(v[-1:]),
                               lnes_first_ms=int(raw['lnes_window_ms'][0])))
    return result


def parity(raw, old, candidates, modes):
    kept = [BASE_MODES.index(name) for name in ('i', 'hold')]
    difference = np.abs(raw['predictions'][:, :, kept] - old['predictions'][:, :, kept])
    result = {'original_i_hold_translation_max_mm': float(difference[..., :3].max() * 1000),
              'original_i_hold_pose_max_rad': float(difference[..., 3:].max())}
    for candidate in candidates:
        policy = FusionPolicy(candidate.spec)
        if policy.is_identity() and candidate.document is None:
            diff = np.abs(raw['predictions'][:, :, modes.index(policy.name)] - raw['predictions'][:, :, modes.index('fused')])
            result[f'{policy.name}_translation_max_mm'] = float(diff[..., :3].max() * 1000)
            result[f'{policy.name}_pose_max_rad'] = float(diff[..., 3:].max())
    for name, value in result.items():
        if value > (.2 if name.endswith('_mm') else .002):
            raise ValueError(f'same-GPU/identity trajectory parity failed: {name}={value}')
    for key in ('truth', 'clean', 'end_ms', 'packet_ids', 'lnes_window_ms', 'n_raw_events'):
        if not np.array_equal(raw[key], old[key]):
            raise ValueError(f'original raw packet identity changed: {key}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, type=int, choices=[3407, 3408, 3409])
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-clips', type=int, default=0, help='explicit incomplete smoke only')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    if args.limit_clips < 0:
        parser.error('limit-clips must be nonnegative')
    RE.setup()
    E.XKFModel = XKFFusionModel
    torch.set_num_threads(4)
    specification = json.loads(args.spec.read_text())
    if specification.get('formal') is not False or not 2200 <= specification.get('checkpoint_step', -1) < 6000:
        raise ValueError('spec must declare formal=false and exact debug checkpoint_step in2200..5999')
    candidates = [Candidate(value, args.seed) for value in specification['candidates']]
    if not 1 <= len(candidates) <= 8:
        raise ValueError('one to eight explicitly declared candidates required; no automatic search')
    names = tuple(candidate.spec['id'] for candidate in candidates)
    if len(set(names)) != len(names) or set(names) & set(BASE_MODES):
        raise ValueError('candidate names must be unique and distinct from original modes')
    modes = BASE_MODES + names
    config, checkpoint = args.config, args.checkpoint
    bank, banksha = R.D.load_bank()
    clips, excluded = R.schedule(bank, 10)
    before = sources()
    resolved = {clip['clip_id']: [candidate.for_clip(clip['clip_id']).spec for candidate in candidates] for clip in clips}
    declared = {'kind': 'XKF_D_ONLY_CONTINUATION_FUSION_EXPLORATORY', 'formal_gate_assessed': False, 'automatic_full_training': False,
                'seed': args.seed, 'checkpoint_step': specification['checkpoint_step'], 'config_sha256': E.sha(config), 'checkpoint_sha256': E.sha(checkpoint),
                'spec': specification, 'spec_path': str(args.spec.resolve()), 'spec_sha256': E.sha(args.spec),
                'source_sha256': before, 'bank_sha256': banksha, 'clips': clips, 'excluded': excluded,
                'conditions': R.CONDITIONS, 'modes': modes, 'metrics': R.METRICS,
                'motion_modes': ('i', 'fused', 'fused_keep_t') + names,
                'calibration_sources': [candidate.source for candidate in candidates], 'resolved_clip_policies': resolved,
                'inference': 'FP32, TF32 off, target zero, each mode owns state; D cached by packet and inverse-temperature multiplier',
                'calibration': 'one fixed spec per complete clip; provenance retained; no automatic crossfit claim',
                'count_gate': 'candidate ramp1000 multiplies D gains by min(1,current raw W50 count/1000); new placement, historical S39 conclusions do not transfer',
                'partial': bool(args.limit_clips), 'limit_clips': args.limit_clips,
                'finite_subset_reduction': False, 'censoring': 'retain short clips; all-observed and full-horizon summaries both required'}
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'summary.json').exists():
        raise FileExistsError('completed result exists; never overwrite a candidate run')
    protocol_path = output / 'protocol.json'
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text())
        if previous != json.loads(json.dumps(declared)):
            # write_json adds these numeric-validation fields.
            previous.pop('numeric_valid', None); previous.pop('invalid_numeric_paths', None)
            if previous != json.loads(json.dumps(declared)):
                raise ValueError('existing candidate declaration differs')
    else:
        write_json(protocol_path, declared)
    if args.prepare_only:
        print(json.dumps({'protocol': str(protocol_path), 'candidate_count': len(candidates)}), flush=True)
        return
    device = torch.device(args.device)
    model, cfg, identity = E.load_model(config, checkpoint, device)
    if identity['global_step'] != specification['checkpoint_step'] or identity['seed'] != args.seed:
        raise ValueError('candidate checkpoint step or seed differs from explicit specification')
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    log_beta_before = model.direct.log_beta.detach().clone()
    rawdir = output / 'raw'; rawdir.mkdir(exist_ok=True)
    started = time.perf_counter()
    caches, records, parity_records = {}, [], []
    selected = clips[:args.limit_clips] if args.limit_clips else clips
    with (output / 'per_clip.jsonl').open('w') as stream:
        for index, clip in enumerate(selected):
            sequence = clip['sequence']
            if sequence not in caches:
                data = R.D.load_data(cfg, sequence, bank['sequences'][sequence])
                caches[sequence] = PacketCache(model, data, device)
            value = rollout(model, bank['sequences'][sequence], clip, caches[sequence], candidates)
            filename = f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz'
            oldpath = R2 / f'evaluations/repair_s{args.seed}_step2200_rollout/raw' / filename
            with np.load(oldpath, allow_pickle=False) as old:
                parity_records.append({'clip_id': clip['clip_id'], **parity(value, old, candidates, modes)})
            np.savez_compressed(rawdir / filename, **value)
            rows = per_clip_rows(clip, value, modes)
            for row in rows:
                stream.write(json.dumps(E.safe_json(row), allow_nan=False) + '\n')
            stream.flush(); records.extend(rows)
            progress = {'completed_clips': index + 1, 'total_clips': len(selected), 'seed': args.seed,
                        'elapsed_seconds': time.perf_counter() - started}
            write_json(output / 'progress.json', progress)
            if index % 8 == 0:
                print(json.dumps(progress), flush=True)
    if sources() != before or E.sha(args.spec) != declared['spec_sha256'] or not all(c.unchanged() for c in candidates):
        raise ValueError('candidate source/spec changed during evaluation')
    if not torch.equal(log_beta_before, model.direct.log_beta.detach()):
        raise ValueError('temperature adapter changed saved model parameter')
    write_json(output / 'summary.json', {'kind': declared['kind'], 'formal_gate_assessed': False,
        'automatic_full_training': False, 'finite_subset_reduction': False, 'identity': identity,
        'protocol_sha256': E.sha(protocol_path), 'source_sha256': before, 'spec_sha256': declared['spec_sha256'],
        'device': str(device), 'cuda_visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'), 'partial': bool(args.limit_clips),
        'clip_count': len(selected), 'modes': modes, 'metrics': R.METRICS, 'parity': parity_records,
        'gains_fused': model.gains('fused').cpu().tolist(), 'temperature_parameter_restored': True,
        'elapsed_seconds': time.perf_counter() - started, 'aggregate': R.aggregate(records)})
    print(json.dumps({'output': str(output / 'summary.json'), 'clips': len(selected), 'formal': False}), flush=True)


if __name__ == '__main__':
    main()
