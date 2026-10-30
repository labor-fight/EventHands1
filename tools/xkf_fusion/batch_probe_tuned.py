#!/usr/bin/env python3
"""Full actual-forward deployment audit for one frozen D2400 fusion candidate.

Compare unmodified batch1, unmodified batch9, and canonical per-sample full
forward at batch9. D is recomputed by the actual model for every invocation.
The saved27-row research loop remains a descriptive reference, never selected
because its batch size gives a better score.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch

from semkine.xkf_centroid import CentroidFusionPolicy
from semkine.xkf_stable_geometry import XKFStableGeometryModel
from tools.xkf_fusion import evaluate_tuned as Q
from tools.xkf_fusion.batch_probe import difference
from tools.xkf_fusion.bn_probe import parameter_digest


def source_hashes():
    return Q.sources() | {name: Q.E.sha(ROOT / name) for name in (
        'semkine/xkf_centroid.py', 'semkine/xkf_stable_geometry.py',
        'tools/xkf_fusion/batch_probe_tuned.py', 'tools/xkf_fusion/batch_probe.py',
        'tools/xkf_fusion/bn_probe.py')}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--reference-dir', required=True, type=Path)
    parser.add_argument('--assessment', type=Path,
                        help='completed independent assessment binding the reference input hashes; required for full112')
    parser.add_argument('--candidate-id', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-clips', type=int, default=0)
    args = parser.parse_args()
    if args.limit_clips < 0:
        parser.error('limit-clips must be nonnegative')
    if not args.limit_clips and args.assessment is None:
        parser.error('full112 requires --assessment')
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    protocol = json.loads((args.reference_dir / 'protocol.json').read_text())
    reference = json.loads((args.reference_dir / 'summary.json').read_text())
    if (reference['partial'] or reference['clip_count'] != 112
            or Q.E.sha(args.reference_dir / 'protocol.json') != reference['protocol_sha256']):
        raise ValueError('reference must be complete unchanged112')
    assessment_sha, reference_inputs = None, {}
    if args.assessment is not None:
        assessment_sha = Q.E.sha(args.assessment)
        assessment = json.loads(args.assessment.read_text())
        if (assessment['status'] != 'ASSESSED' or assessment['validation_errors'] or assessment['missing']):
            raise ValueError('independent reference assessment must be complete and valid')
        assessed = assessment['seeds'][str(reference['identity']['seed'])]
        if (assessed['identity'] != reference['identity'] or args.candidate_id not in assessed['candidates']):
            raise ValueError('reference identity/candidate differs from independent assessment')
        paths = [args.reference_dir / name for name in ('summary.json', 'protocol.json')]
        paths.extend(sorted((args.reference_dir / 'raw').glob('*.npz')))
        if len(paths) != 114:
            raise ValueError('reference requires exactly112 raw files')
        for path in paths:
            key = str(path.resolve())
            digest = Q.E.sha(path)
            if assessment['input_sha256'].get(key) != digest:
                raise ValueError('reference input changed after independent assessment: ' + key)
            reference_inputs[key] = digest
    Q.RE.setup()
    Q.E.XKFModel = XKFStableGeometryModel
    Q.FusionPolicy = CentroidFusionPolicy
    torch.set_num_threads(4)
    device = torch.device(args.device)
    model, cfg, identity = Q.E.load_model(args.config, args.checkpoint, device)
    if identity['global_step'] != 2400 or identity['checkpoint_sha256'] != reference['identity']['checkpoint_sha256']:
        raise ValueError('exact evaluated D2400 checkpoint required')
    if identity['config_sha256'] != reference['identity']['config_sha256']:
        raise ValueError('reference config differs')
    identity['clock_scope'] = {'I': 2200, 'D_and_global': 2400}
    for module in model.modules():
        if isinstance(module, Q.ManoLayer):
            module.fast_chain = True
    spec = protocol['spec']
    if 'bn_calibration_by_seed' in spec:
        raise ValueError('tuned deployment keeps the checkpoint BN buffers')
    value = next(v for v in spec['candidates'] if v['id'] == args.candidate_id)
    candidate = Q.Candidate(value, identity['seed'])
    if candidate.document is not None:
        if ('clip_specs' in candidate.document or candidate.document.get('partial_smoke') is not False
                or candidate.document['checkpoint_sha256'] != identity['checkpoint_sha256']):
            raise ValueError('requires fixed full training-only same-checkpoint affine')
    bank, banksha = Q.R.D.load_bank()
    clips, excluded = Q.R.schedule(bank, 10)
    if banksha != protocol['bank_sha256'] or clips != protocol['clips'] or excluded != protocol['excluded']:
        raise ValueError('same112 schedule required')
    selected = clips[:args.limit_clips] if args.limit_clips else clips
    modes = ('reference27', 'off_forward1', 'off_forward9', 'stable_forward9', 'hold')
    source = source_hashes()
    parameters_before = parameter_digest(model)
    buffers_before = {name: value.cpu().clone() for name, value in model.named_buffers()}
    declaration = {'kind': 'XKF_D2400_ACTUAL_FORWARD_BATCH_AUDIT', 'formal': False,
        'automatic_full_training': False, 'identity': identity, 'source_sha256': source,
        'candidate_id': args.candidate_id, 'reference_summary_sha256': Q.E.sha(args.reference_dir / 'summary.json'),
        'independent_assessment': str(args.assessment.resolve()) if args.assessment else None,
        'independent_assessment_sha256': assessment_sha,
        'assessed_reference_input_sha256': reference_inputs,
        'reference_protocol_sha256': reference['protocol_sha256'], 'clips': selected, 'modes': modes,
        'conditions': Q.R.CONDITIONS, 'metrics': Q.R.METRICS, 'partial_smoke': bool(args.limit_clips),
        'GT': 'only fixed initialization and metric scoring; every inference packet target zero',
        'D': 'actual forward_packet computes D every call; packet cache stores only events/LNES, no D predictions',
        'stability': 'per_sample_model serializes the complete original batch1 I+D+fusion for each row, with no rounding',
        'comparison': 'original actual batch1 anchors the deployment; compare off9 and stable9 to it; saved27 is descriptive',
        'original_thresholds_unchanged': True}
    Q.write_json(args.output / 'protocol.json', declaration)
    rawdir = args.output / 'raw'; rawdir.mkdir()
    caches, records, parity, raw_hashes = {}, [], [], {}
    started = time.perf_counter()
    forward_time = {name: 0. for name in modes[1:4]}
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    for index, clip in enumerate(selected):
        sequence = clip['sequence']
        if sequence not in caches:
            data = Q.R.D.load_data(cfg, sequence, bank['sequences'][sequence])
            caches[sequence] = Q.PacketCache(model, data, device)
        cache = caches[sequence]
        filename = f'{index:03d}_{clip["cohort"]}_run{clip["run_id"]}.npz'
        raw_path = args.reference_dir / 'raw' / filename
        raw_hashes[str(raw_path)] = Q.E.sha(raw_path)
        if reference_inputs and raw_hashes[str(raw_path)] != reference_inputs[str(raw_path.resolve())]:
            raise ValueError('reference raw changed before use')
        with np.load(raw_path, allow_pickle=False) as values:
            old = {key: values[key].copy() for key in values.files}
        reference_index = reference['modes'].index(args.candidate_id)
        hold_index = reference['modes'].index('hold')
        policy = candidate.for_clip(clip['clip_id'])
        model.configure_fusion(policy.spec)
        model.kmode = 'fused'
        initial = torch.from_numpy(old['initial']).to(device)
        previous = initial[:, None].expand(-1, 3, -1).clone()
        collected = {key: [] for key in ('predictions', 'metrics', 'physical_wrist_error_xyz_mm', 'state_t_error_xyz_mm')}
        Q.R.D.reset(model)
        for step, row_index in enumerate(clip['row_indices']):
            row = bank['sequences'][sequence]['rows'][row_index]
            next_states = []
            for pi, name in enumerate(modes[1:4]):
                model.configure_stability('per_sample_model' if name == 'stable_forward9' else 'off')
                if name == 'off_forward1':
                    values = []
                    for ci in range(9):
                        packet, measurements = cache.packet(row, previous[ci:ci + 1, pi], [])
                        if measurements:
                            raise ValueError('actual deployment audit must not cache D')
                        sync(); start = time.perf_counter()
                        values.append(model.forward_packet(packet))
                        sync(); forward_time[name] += time.perf_counter() - start
                    next_states.append(torch.cat(values))
                else:
                    packet, measurements = cache.packet(row, previous[:, pi], [])
                    if measurements:
                        raise ValueError('actual deployment audit must not cache D')
                    sync(); start = time.perf_counter()
                    next_states.append(model.forward_packet(packet))
                    sync(); forward_time[name] += time.perf_counter() - start
            previous = torch.stack(next_states, 1)
            old_candidate = torch.from_numpy(old['predictions'][step, :, reference_index]).to(device)
            old_hold = torch.from_numpy(old['predictions'][step, :, hold_index]).to(device)
            states = torch.cat([old_candidate[:, None], previous, old_hold[:, None]], 1)
            truth = torch.from_numpy(old['truth'][step:step + 1]).to(device)
            metrics, wrist, state_t = Q.R.frame_metrics(model, states.reshape(-1, 51), truth, cache.beta)
            if not bool(torch.isfinite(states).all() and torch.isfinite(metrics).all()):
                raise FloatingPointError('nonfinite actual forward')
            collected['predictions'].append(states.cpu().numpy())
            collected['metrics'].append(metrics.reshape(9, 5, -1).cpu().numpy())
            collected['physical_wrist_error_xyz_mm'].append(wrist.reshape(9, 5, 3).cpu().numpy())
            collected['state_t_error_xyz_mm'].append(state_t.reshape(9, 5, 3).cpu().numpy())
        if cache.measurements:
            raise ValueError('D prediction cache unexpectedly populated')
        Q.R.D.reset(model)
        raw = {key: np.stack(values) for key, values in collected.items()}
        for key in ('initial', 'truth', 'clean', 'packet_ids', 'end_ms', 'lnes_window_ms', 'n_raw_events'):
            raw[key] = old[key]
        np.savez_compressed(rawdir / filename, **raw)
        rows = Q.per_clip_rows(clip, raw, modes)
        records.extend(rows)
        anchor = raw['predictions'][:, :, 1]
        parity.append({'clip_id': clip['clip_id'], 'same_own_initial_and_raw_inputs': True,
            **{modes[mi]: {**difference(raw['predictions'][:, :, mi], anchor),
                 'bitwise_equal': bool(np.array_equal(raw['predictions'][:, :, mi], anchor))}
               for mi in (0, 2, 3)}})
        Q.write_json(args.output / 'progress.json', {'completed_clips': index + 1, 'total_clips': len(selected),
            'elapsed_seconds': time.perf_counter() - started})
        if index % 8 == 0:
            print(json.dumps({'completed_clips': index + 1, 'seed': identity['seed']}), flush=True)
    if source_hashes() != source or not candidate.unchanged() or parameter_digest(model) != parameters_before:
        raise ValueError('source, calibration or parameters changed')
    for name, value in model.named_buffers():
        if not torch.equal(value.cpu(), buffers_before[name]):
            raise ValueError('model buffer changed during deployment audit')
    if Q.E.sha(args.checkpoint) != identity['checkpoint_sha256']:
        raise ValueError('checkpoint changed')
    if args.assessment is not None:
        if Q.E.sha(args.assessment) != assessment_sha:
            raise ValueError('independent assessment changed during deployment audit')
        for name, digest in reference_inputs.items():
            if Q.E.sha(Path(name)) != digest:
                raise ValueError('assessed reference input changed during deployment audit: ' + name)
    Q.write_json(args.output / 'summary.json', {'kind': declaration['kind'], 'formal': False,
        'automatic_full_training': False, 'identity': identity, 'candidate_id': args.candidate_id,
        'partial_smoke': bool(args.limit_clips), 'clip_count': len(selected), 'modes': modes,
        'metrics': Q.R.METRICS, 'aggregate': Q.R.aggregate(records), 'parity': parity,
        'all_stable9_bitwise_equal_actual_batch1': all(row['stable_forward9']['bitwise_equal'] for row in parity),
        'actual_forward_seconds': forward_time, 'elapsed_seconds': time.perf_counter() - started,
        'unchanged_parameters_sha256': parameters_before, 'all_buffers_unchanged': True,
        'input_raw_sha256': raw_hashes, 'protocol_sha256': Q.E.sha(args.output / 'protocol.json')})


if __name__ == '__main__':
    main()
