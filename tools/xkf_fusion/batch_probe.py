#!/usr/bin/env python3
"""Audit actual batch1/batch9 deployment against saved 27-row candidate loops.

This is a numerical/runtime diagnostic. No parity threshold is relaxed and no
candidate is selected by it. Targets are zero during every inference call.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

import numpy as np
import torch

from tools.xkf_fusion import evaluate as Q
from tools.xkf_fusion.bn_probe import apply_buffers, bn_modules, parameter_digest


def write(path, value):
    Q.write_json(path, value)


def configure(seed, spec):
    Q.RE.setup()
    Q.E.XKFModel = Q.XKFFusionModel
    torch.set_num_threads(4)
    config = ROOT / f'configs/xkf_repair/repair_s{seed}.yaml'
    checkpoint = Q.R2 / f'repair_s{seed}/xkf_repair_s{seed}-step=2200.ckpt'
    return config, checkpoint


def load(seed, specification, device):
    config, checkpoint = configure(seed, specification)
    model, cfg, identity = Q.E.load_model(config, checkpoint, device)
    descriptor = specification.get('bn_calibration_by_seed', {}).get(str(seed))
    if descriptor:
        path = ROOT / descriptor['path']
        if Q.E.sha(path) != descriptor['sha256']:
            raise ValueError('BN SHA changed')
        artifact = torch.load(path, map_location='cpu', weights_only=False)
        if (artifact['base_checkpoint_sha256'] != identity['checkpoint_sha256']
                or artifact['parameter_sha256'] != parameter_digest(model)):
            raise ValueError('BN model identity differs')
        apply_buffers(bn_modules(model), artifact['bn_buffers'])
        identity['bn_buffers'] = descriptor
    model.eval()
    for module in model.modules():
        if isinstance(module, Q.ManoLayer):
            module.fast_chain = True
    return model, cfg, identity


def difference(a, b):
    delta = np.abs(a - b)
    return {'translation_max_mm': float(delta[..., :3].max() * 1000),
            'pose_max_rad': float(delta[..., 3:].max())}


@torch.no_grad()
def locate_failure(args, specification, output):
    """Save a formerly rejected raw clip, then isolate batch shape at equal input."""
    device = torch.device(args.device)
    model, cfg, identity = load(args.seed, specification, device)
    candidates = [Q.Candidate(v, args.seed) for v in specification['candidates']]
    if len(candidates) != 2 or not candidates[0].for_clip('').is_identity():
        raise ValueError('failure locator requires original two-candidate declaration with identity first')
    bank, banksha = Q.R.D.load_bank()
    clips, _ = Q.R.schedule(bank, 10)
    clip = clips[args.clip_index]
    meta = bank['sequences'][clip['sequence']]
    data = Q.R.D.load_data(cfg, clip['sequence'], meta)
    cache = Q.PacketCache(model, data, device)
    model.xyz_keep_aux = True
    original = model.incremental_prediction
    captures = []

    def capture(packet):
        result = original(packet)
        aux = model.xyz_last
        captures.append({'previous': packet.prev_state.cpu().numpy(), 'prediction': result.cpu().numpy(),
                         'vertices': aux['verts'].cpu().numpy(),
                         'association_vid': aux['assoc']['vid'].cpu().numpy(),
                         'association_hit': aux['assoc']['hit'].cpu().numpy(),
                         'association_valid': aux['assoc']['valid'].cpu().numpy()})
        return result

    model.incremental_prediction = capture
    raw = Q.rollout(model, meta, clip, cache, candidates)
    model.incremental_prediction = original
    modes = Q.BASE_MODES + tuple(c.spec['id'] for c in candidates)
    fused, same = modes.index('fused'), modes.index(candidates[0].spec['id'])
    rows = [meta['rows'][i] for i in clip['row_indices']]
    report = []
    for step, row in enumerate(rows):
        baseline, added = captures[step * 2:step * 2 + 2]
        bi, ai = np.arange(9) * 3 + 1, np.arange(9) * 2
        values = []
        # Re-evaluate exactly the SAME incoming states at each batch size.
        incoming = torch.from_numpy(baseline['previous'][bi]).to(device)
        for copies in (3, 2, 1):
            prev = incoming[:, None].expand(-1, copies, -1).reshape(-1, 51)
            packet, _ = cache.packet(row, prev, [1.])
            state = original(packet).reshape(9, copies, 51)[:, 0]
            aux = model.xyz_last
            take = np.arange(9) * copies
            values.append({'prediction': state.cpu().numpy(),
                           'vertices': aux['verts'].cpu().numpy()[take],
                           'association_vid': aux['assoc']['vid'].cpu().numpy()[take],
                           'association_hit': aux['assoc']['hit'].cpu().numpy()[take]})
        entry = {'step': step, 'end_ms': int(row['end_ms']),
                 'incoming': difference(baseline['previous'][bi], added['previous'][ai]),
                 'incremental_prediction': difference(baseline['prediction'][bi], added['prediction'][ai]),
                 'fused_vs_identity': difference(raw['predictions'][step, :, fused], raw['predictions'][step, :, same]),
                 'own_history_vertex_max_mm': float(np.abs(baseline['vertices'][bi] - added['vertices'][ai]).max() * 1000),
                 'own_history_association_record_disagreements': int(np.any(baseline['association_vid'][bi] != added['association_vid'][ai], -1).sum())}
        entry['equal_input_batch_shape'] = {}
        for other, label in ((1, '27_vs18'), (2, '27_vs9')):
            a, b = values[0], values[other]
            entry['equal_input_batch_shape'][label] = {
                **difference(a['prediction'], b['prediction']),
                'vertex_max_mm': float(np.abs(a['vertices'] - b['vertices']).max() * 1000),
                'association_record_disagreements': int(np.any(a['association_vid'] != b['association_vid'], -1).sum()),
                'association_hit_disagreements': int((a['association_hit'] != b['association_hit']).sum())}
        delta = np.abs(raw['predictions'][step, :, fused] - raw['predictions'][step, :, same])
        ci, state_index = np.unravel_index(delta.argmax(), delta.shape)
        entry['maximum_difference_condition'] = Q.R.CONDITIONS[ci][0]
        entry['maximum_difference_state_index'] = int(state_index)
        report.append(entry)
    np.savez_compressed(output / 'rejected_clip_raw.npz', **raw)
    write(output / 'failure_location.json', {'formal': False, 'identity': identity, 'clip': clip,
          'bank_sha256': banksha, 'spec_sha256': Q.E.sha(args.spec), 'steps': report,
          'first_pose_parity_failure_step': next((r['step'] for r in report if r['fused_vs_identity']['pose_max_rad'] > .002), None),
          'threshold_unchanged': {'translation_mm': .2, 'pose_rad': .002},
          'source_sha256': Q.sources() | {'tools/xkf_fusion/batch_probe.py': Q.E.sha(Path(__file__))}})


@torch.no_grad()
def deployment(args, specification, output):
    reference = args.reference_dir
    protocol = json.loads((reference / 'protocol.json').read_text())
    summary = json.loads((reference / 'summary.json').read_text())
    if summary['partial'] or summary['clip_count'] != 112 or summary['spec_sha256'] != Q.E.sha(args.spec):
        raise ValueError('requires complete112 reference from identical spec')
    value = next(v for v in specification['candidates'] if v['id'] == args.candidate_id)
    candidate = Q.Candidate(value, args.seed)
    if candidate.document is not None and 'clip_specs' in candidate.document:
        raise ValueError('deployment audit requires one fixed train-only/global affine, no per-clip fitting')
    device = torch.device(args.device)
    model, cfg, identity = load(args.seed, specification, device)
    if identity['checkpoint_sha256'] != summary['identity']['checkpoint_sha256']:
        raise ValueError('reference checkpoint differs')
    bank, banksha = Q.R.D.load_bank()
    clips, excluded = Q.R.schedule(bank, 10)
    if clips != protocol['clips'] or banksha != protocol['bank_sha256']:
        raise ValueError('reference schedule or bank differs')
    selected = clips[:args.limit_clips] if args.limit_clips else clips
    modes = ('reference27', 'cached9', 'forward9', 'forward1')
    before = Q.sources() | {'tools/xkf_fusion/batch_probe.py': Q.E.sha(Path(__file__)),
                           'tools/xkf_fusion/bn_probe.py': Q.E.sha(ROOT / 'tools/xkf_fusion/bn_probe.py')}
    write(output / 'protocol.json', {'formal': False, 'automatic_full_training': False,
          'seed': args.seed, 'identity': identity, 'spec_sha256': Q.E.sha(args.spec),
          'candidate_id': args.candidate_id, 'source_sha256': before,
          'reference_summary_sha256': Q.E.sha(reference / 'summary.json'), 'clips': selected,
          'partial': bool(args.limit_clips), 'modes': modes, 'conditions': Q.R.CONDITIONS,
          'inference': 'FP32,TF32off; each path owns state; target zeros; cached9 uses cached D1; forward9/1 call real forward_packet',
          'no_gate_relaxation': True})
    (output / 'raw').mkdir(exist_ok=True)
    caches, records, parity = {}, [], []
    started = time.monotonic()
    for index, clip in enumerate(selected):
        sequence = clip['sequence']
        if sequence not in caches:
            data = Q.R.D.load_data(cfg, sequence, bank['sequences'][sequence])
            caches[sequence] = Q.PacketCache(model, data, device)
        cache = caches[sequence]
        policy = candidate.for_clip(clip['clip_id'])
        model.configure_fusion(policy.spec)
        if policy.temperature_multiplier != 1.:
            raise ValueError('deployment batch audit fixed temperature1')
        model.kmode = 'fused'
        filename = f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz'
        with np.load(reference / 'raw' / filename, allow_pickle=False) as loaded:
            old = {key: loaded[key] for key in loaded.files}
        ref_mode = summary['modes'].index(args.candidate_id)
        prev = torch.from_numpy(old['initial']).to(device)[:, None].expand(-1, 3, -1).clone()
        collected = {key: [] for key in ('predictions', 'metrics', 'physical_wrist_error_xyz_mm', 'state_t_error_xyz_mm')}
        Q.R.D.reset(model)
        for step, row_index in enumerate(clip['row_indices']):
            row = bank['sequences'][sequence]['rows'][row_index]
            packet, measured = cache.packet(row, prev[:, 0], [1.])
            xi = model.incremental_prediction(packet)
            direct, aux = measured[1.]
            cached, _, _ = policy.fuse(xi, direct.expand(9, -1), aux, prev[:, 0], packet.counts, model.gains('fused'))
            packet9, _ = cache.packet(row, prev[:, 1], [1.])
            actual9 = model.forward_packet(packet9)
            actual1 = []
            for ci in range(9):
                packet1, _ = cache.packet(row, prev[ci:ci + 1, 2], [1.])
                actual1.append(model.forward_packet(packet1))
            actual1 = torch.cat(actual1)
            prev = torch.stack([cached, actual9, actual1], 1)
            reference_state = torch.from_numpy(old['predictions'][step, :, ref_mode]).to(device)
            nxt = torch.cat([reference_state[:, None], prev], 1)
            truth = torch.from_numpy(old['truth'][step:step + 1]).to(device)
            score, wrist, state_t = Q.R.frame_metrics(model, nxt.reshape(-1, 51), truth, cache.beta)
            if not bool(torch.isfinite(nxt).all() and torch.isfinite(score).all()):
                raise FloatingPointError('nonfinite deployment path')
            collected['predictions'].append(nxt.cpu().numpy())
            collected['metrics'].append(score.reshape(9, 4, -1).cpu().numpy())
            collected['physical_wrist_error_xyz_mm'].append(wrist.reshape(9, 4, 3).cpu().numpy())
            collected['state_t_error_xyz_mm'].append(state_t.reshape(9, 4, 3).cpu().numpy())
        Q.R.D.reset(model)
        raw = {key: np.stack(vals) for key, vals in collected.items()}
        for key in ('initial', 'clean', 'truth', 'end_ms', 'packet_ids', 'lnes_window_ms', 'n_raw_events'):
            raw[key] = old[key]
        np.savez_compressed(output / 'raw' / filename, **raw)
        records.extend(Q.per_clip_rows(clip, raw, modes))
        parity.append({'clip_id': clip['clip_id'], **{name: difference(raw['predictions'][:, :, mi], raw['predictions'][:, :, 0])
                      for mi, name in enumerate(modes) if mi}})
        write(output / 'progress.json', {'completed_clips': index + 1, 'total_clips': len(selected),
                                        'elapsed_seconds': time.monotonic() - started})
    if (Q.sources() | {'tools/xkf_fusion/batch_probe.py': Q.E.sha(Path(__file__)),
                      'tools/xkf_fusion/bn_probe.py': Q.E.sha(ROOT / 'tools/xkf_fusion/bn_probe.py')}) != before or not candidate.unchanged():
        raise ValueError('sources changed')
    write(output / 'summary.json', {'formal': False, 'automatic_full_training': False,
          'partial': bool(args.limit_clips), 'clip_count': len(selected), 'identity': identity,
          'candidate_id': args.candidate_id, 'modes': modes, 'metrics': Q.R.METRICS,
          'aggregate': Q.R.aggregate(records), 'parity': parity,
          'elapsed_seconds': time.monotonic() - started, 'no_gate_relaxation': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', required=True, type=int)
    parser.add_argument('--spec', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--locate-failure', action='store_true')
    parser.add_argument('--clip-index', type=int, default=0)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--candidate-id')
    parser.add_argument('--limit-clips', type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError('batch diagnostic must use a new output directory')
    args.output_dir.mkdir(parents=True)
    specification = json.loads(args.spec.read_text())
    if args.locate_failure:
        locate_failure(args, specification, args.output_dir)
    else:
        if args.reference_dir is None or args.candidate_id is None:
            parser.error('deployment audit requires reference-dir and candidate-id')
        deployment(args, specification, args.output_dir)


if __name__ == '__main__':
    main()
