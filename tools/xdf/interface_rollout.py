#!/usr/bin/env python3
"""XDF-1 paired free feedback, common-Q1-history pointwise replay and GT wrist clamp."""
from __future__ import annotations
import argparse
import datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'model'))
from tools.xdf.interface_probe import sha, save_json, make_head, gpu_contract
OUT = ROOT / 'outputs/xdf/interface/rollout'
BASE = ROOT / 'outputs/xdf/interface'
SEQUENCES = ('zgz_local', 'zgz_global')
RUN = ROOT / 'outputs/semkine/xdl_q1_s3407'


def prepare():
    e1path = ROOT / 'outputs/xde/e1/protocol.json'
    e1 = json.loads(e1path.read_text())
    modes = ['q1'] + [f'{interface}_s{seed}' for interface in ('F', 'B', 'state') for seed in (1701, 1702)] + ['hold', 'gt_wrist_clamp']
    head_hash = {mode: sha(BASE / 'heads' / f'{mode}.pt') for mode in modes if '_s' in mode}
    sources = [Path(__file__), e1path, BASE / 'protocol.json', ROOT / 'tools/xdf/interface_probe.py',
               ROOT / 'tools/xdl/diagnostics.py', ROOT / 'tools/xdl/geometry.py',
               ROOT / 'semkine/xyz_recovery_study.py', ROOT / 'outputs/xdl/packet_bank.json']
    sources.extend(ROOT / f'outputs/xde/e1/q1_own_{name}.pt.gz' for name in SEQUENCES)
    p = {'study': 'XDF-1-ROLLING-COMPLETE707-20261014-V1', 'created_before_rollout': True,
         'selection': e1['own_selection']['selected'], 'e1_protocol_sha256': sha(e1path),
         'head_sha256': head_hash, 'source_sha256': {str(path): sha(path) for path in sources},
         'steps': 20, 'initializations': ['own', 'clean'], 'modes': modes,
         'own_initial': 'exact fixed E1 Q1-own initial; no additional noise',
         'clean_initial': 'GT only at t0, then free feedback',
         'free_update': 'only translation replaced for F/B/state; root/finger use Q1 output at the same branch state, then all51 feed back',
         'teacher': 'pointwise on common Q1 incoming history: own exact E1 real saved trajectory, clean newly replayed Q1 clean-start trajectory; no teacher output feedback',
         'teacher_score': '20 paired endpoint errors, same-Q1-incoming hold; never scored as sustained recovery or an autonomous trajectory',
         'q1_replay': 'own free trajectory and teacher Q1 output vs E1 real; float32 max absolute51D difference <=1e-6',
         'gt_wrist_clamp': 'mechanism oracle only: initial wrist and each output wrist FK-compensated to GT actual wrist; root/fingers continue own feedback; post-clamp FK wrist residual <=0.01mm; separate from deployable recovery',
         'metrics': 'existing D.score_trajectories for actual21states, real timestamps, time AUC absolute/RA MPJPE MPVPE wrist root/finger; existing200ms motion/lag, raw50 sensitivity; initially-good separate; translation+SO3 angular finite-difference roughness',
         'recovery_rule': 'unchanged XDL absolute MPJPE<=20mm AND RA<=10mm sustained3queries; initially satisfied separated; oracle never counts as deployable recovery',
         'roughness': '20 output states only; exact50ms; wrist translation velocity/acceleration/jerk and SO3 angular velocity/acceleration error; GT raw finite differences remain noisy',
         'numeric': 'source batch1 float32, cudnn.allow_tf32=True cuda.matmul.allow_tf32=False, four CPU threads per process',
         'physical_gpus': [2, 3], 'max_worker_seconds': 1200, 'maximum_rollout_gpu_hours': .75,
         'expected_primary_source_forwards': 15360, 'max_additional_teacher_forwards_if_cache_not_exact': 960,
         'full_xdf1_budget_gpu_hours_including_parent_angle_work': 2.,
         'limits': ['not new clean acceptance gate', '24 stratified own-state clips per sequence, not full-sequence accuracy',
                    'head seeds are not complete-model independent training seeds', 'teacher pointwise output is not a closed-loop trajectory',
                    'GT wrist clamp changes inputs using truth and is solely an angular mechanism upper bound'],
    }
    dest = OUT / 'protocol.json'
    if dest.exists():
        if json.loads(dest.read_text()) != p: raise RuntimeError('rollout protocol exists and differs')
    else: save_json(dest, p)
    (OUT / 'protocol.sha256').write_text(sha(dest) + '\n')
    print(json.dumps({'protocol_sha256': sha(dest), 'heads': list(head_hash), 'source_forwards': 15360}), flush=True)


def stat(values):
    x = np.asarray([np.nan if x is None else x for x in values], np.float64)
    good = x[np.isfinite(x)]
    return {'mean': float(good.mean()) if len(good) else None, 'p95': float(np.quantile(good, .95)) if len(good) else None,
            'defined': int(len(good)), 'undefined': int(len(x) - len(good))}


def sanitize(value):
    if isinstance(value, dict): return {str(k): sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [sanitize(v) for v in value]
    if isinstance(value, np.ndarray): return sanitize(value.tolist())
    if isinstance(value, np.generic): return sanitize(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def roughness(predictions, truth, times, mean, D):
    from scipy.spatial.transform import Rotation
    if not np.all(np.diff(times[1:]) == 50): raise RuntimeError('roughness requires declared50ms outputs')
    result = {}
    pred = predictions[1:, :3].astype(np.float64) * 1000.
    gt = truth[1:, :3].astype(np.float64) * 1000.
    for order, name in ((1, 'velocity'), (2, 'acceleration'), (3, 'jerk')):
        a = np.diff(pred, n=order, axis=0) / (.05 ** order)
        b = np.diff(gt, n=order, axis=0) / (.05 ** order)
        for label, array in [('pred', a), ('gt', b), ('error', a - b)]:
            result[f'wrist_{name}_{label}_mm_s{order}'] = float(np.linalg.norm(array, axis=-1).mean())
    pr, pf = D.state_rotations_safe(predictions[1:], mean)
    gr, gf = D.state_rotations_safe(truth[1:], mean)
    for block, pm, gm in [('root', pr, gr), ('finger', pf, gf)]:
        def velocities(mats):
            rel = mats[1:] @ np.swapaxes(mats[:-1], -1, -2)
            return Rotation.from_matrix(rel.reshape(-1, 3, 3)).as_rotvec().reshape(rel.shape[:-2] + (3,)) / .05 * (180. / np.pi)
        a = velocities(pm); b = velocities(gm)
        for order, name in ((1, 'velocity'), (2, 'acceleration')):
            av = a if order == 1 else np.diff(a, axis=0) / .05
            bv = b if order == 1 else np.diff(b, axis=0) / .05
            for label, array in [('pred', av), ('gt', bv), ('error', av - bv)]:
                result[f'{block}_{name}_{label}_deg_s{order}'] = float(np.linalg.norm(array, axis=-1).mean())
    return result


def run(sequence, start):
    import torch
    from tools.xdl import diagnostics as D
    device = gpu_contract(); torch.backends.cudnn.allow_tf32 = True; torch.backends.cuda.matmul.allow_tf32 = False
    p = json.loads((OUT / 'protocol.json').read_text()); ph = sha(OUT / 'protocol.json')
    model, cfg, identity = D.load_model(str(RUN), device); model.xyz_keep_aux = True
    for parameter in model.parameters(): parameter.requires_grad_(False)
    mano = D.ManoLayer(cfg['MANO']['NPZ'], add_mean=False).to(device).eval(); mano.fast_chain = True
    mean = mano.hands_mean.cpu().numpy(); bank = json.loads((ROOT / 'outputs/xdl/packet_bank.json').read_text())
    data = D.load_data(cfg, sequence, bank['sequences'][sequence]); beta, camera = D.context(data, device)
    model.set_hand_context(beta, camera); captured = {}
    handle = model.xyz.fc1.register_forward_pre_hook(lambda _m, inputs: captured.update(full=inputs[0].detach().clone()))
    heads = {}
    for mode, digest in p['head_sha256'].items():
        path = BASE / 'heads' / f'{mode}.pt'
        if sha(path) != digest: raise RuntimeError('head changed after freeze')
        saved = torch.load(path, map_location='cpu'); head = make_head().to(device); head.load_state_dict(saved['state_dict']); head.eval()
        heads[mode] = (head, torch.as_tensor(saved['mean'], device=device), torch.as_tensor(saved['std'], device=device), saved['interface'])
    with gzip.open(ROOT / f'outputs/xde/e1/q1_own_{sequence}.pt.gz', 'rb') as stream:
        e1 = torch.load(stream, map_location='cpu')
    originals = {r['clip_id']: r['predictions'] for r in e1['trajectories'] if r['spec']['block'] == 'real'}
    calls = 0; teacher_extra = 0; rows_out = []; teachers = []; replays = []; clamp_max = 0.; clamp_fk_calls = 0

    def predict(events, incoming, row):
        nonlocal calls
        if time.perf_counter() - start > p['max_worker_seconds']: raise RuntimeError('rollout worker budget exhausted')
        packet = D.make_packet(events, incoming[None], beta, camera, device, row)
        if packet.target.any(): raise RuntimeError('GT target leaked into network')
        captured.clear()
        with torch.no_grad(): output = model.forward_packet(packet)
        calls += 1
        if not len(events): return output.float().cpu().numpy()[0], None
        if captured['full'].shape != (1, 707): raise RuntimeError('not complete707')
        return output.float().cpu().numpy()[0], captured['full']

    def replace(mode, output, full, incoming):
        if mode == 'q1' or full is None: return output.copy()
        head, center, scale, interface = heads[mode]
        with torch.no_grad():
            x = (full - center) / scale
            if interface == 'B': x[:, :544] = 0.
            elif interface == 'state': x[:, :608] = 0.
            elif interface != 'F': raise RuntimeError('undeclared rollout interface')
            delta = head(x).float().cpu().numpy()[0] * .05
        result = output.copy(); result[:3] = incoming[:3] + delta
        return result

    def clamp(state, true_wrist):
        nonlocal clamp_max, clamp_fk_calls
        joints, _ = D.fk(mano, state[None], beta, device); clamp_fk_calls += 1
        value = state.copy(); value[:3] += true_wrist - joints[0, 0]
        after, _ = D.fk(mano, value[None], beta, device); clamp_fk_calls += 1
        err = float(np.linalg.norm(after[0, 0] - true_wrist) * 1000.); clamp_max = max(clamp_max, err)
        if err > .01: raise RuntimeError(f'GT wrist clamp residual {err}mm')
        if not np.array_equal(value[3:], state[3:]): raise RuntimeError('clamp changed angle coordinates')
        return value

    for clip in p['selection'][sequence]:
        packet_rows = [bank['sequences'][sequence]['rows'][i] for i in clip['end_indices']]
        events = [D.raw_events(data, row) for row in packet_rows]
        times = np.array([clip['initial_time_ms']] + [r['end_ms'] for r in packet_rows])
        truth = data['gt'][times]; truth_joints, _ = D.fk(mano, truth, beta, device)
        for initialization in p['initializations']:
            initial = np.array(clip['initial'] if initialization == 'own' else truth[0], np.float32)
            free = {}; q1_features = []; D.reset(model); states = [initial.copy()]
            for ev, row in zip(events, packet_rows):
                output, features = predict(ev, states[-1], row); states.append(output); q1_features.append(features)
            free['q1'] = np.stack(states)
            if initialization == 'own':
                err = float(np.max(np.abs(free['q1'] - originals[clip['clip_id']])))
                replays.append(err)
                if err > 1e-6: raise RuntimeError(f'Q1 E1 real mismatch {err}')
            common = originals[clip['clip_id']] if initialization == 'own' else free['q1']
            conditional = {mode: [] for mode in p['modes'] if mode != 'gt_wrist_clamp'}
            for step, (ev, row) in enumerate(zip(events, packet_rows)):
                incoming = common[step]
                if np.array_equal(incoming, free['q1'][step]):
                    output, features = free['q1'][step + 1], q1_features[step]
                else:
                    output, features = predict(ev, incoming, row); teacher_extra += 1
                err = float(np.max(np.abs(output - common[step + 1])))
                if err > 1e-6: raise RuntimeError(f'common Q1 teacher replay mismatch {err}')
                for mode in conditional:
                    conditional[mode].append(incoming.copy() if mode == 'hold' else replace(mode, output, features, incoming))
            for mode in heads:
                D.reset(model); states = [initial.copy()]
                for ev, row in zip(events, packet_rows):
                    output, features = predict(ev, states[-1], row)
                    states.append(replace(mode, output, features, states[-1]))
                free[mode] = np.stack(states)
            free['hold'] = np.repeat(initial[None], len(times), axis=0)
            D.reset(model); states = [clamp(initial, truth_joints[0, 0])]
            for step, (ev, row) in enumerate(zip(events, packet_rows)):
                output, _ = predict(ev, states[-1], row)
                states.append(clamp(output, truth_joints[step + 1, 0]))
            free['gt_wrist_clamp'] = np.stack(states)
            predictions = np.stack([free[mode] for mode in p['modes']])
            if not np.isfinite(predictions).all(): raise FloatingPointError('nonfinite free trajectory retained as command failure')
            specs = [{'id': mode, 'block': mode, 'deployable': mode != 'gt_wrist_clamp'} for mode in p['modes']]
            with torch.no_grad():
                scored = D.score_trajectories(mano, predictions, truth, times, specs, predictions[:, 0], beta, device, mean)
            for row in scored:
                row.update(sequence=sequence, clip_id=clip['clip_id'], initialization=initialization, truth=truth,
                           transl_z_error_mm=(row['predictions'][:, 2] - truth[:, 2]) * 1000.,
                           roughness=roughness(row['predictions'], truth, times, mean, D))
                rows_out.append(row)
            for mode, values in conditional.items():
                values = np.stack(values)
                with torch.no_grad(): errors, _, _ = D.endpoint(mano, values, truth[1:], beta, device, mean)
                teachers.append({'sequence': sequence, 'clip_id': clip['clip_id'], 'initialization': initialization,
                                 'mode': mode, 'incoming': common[:-1], 'predictions': values, 'truth': truth[1:],
                                 'times_ms': times[1:], 'errors': errors,
                                 'transl_z_error_mm': (values[:, 2] - truth[1:, 2]) * 1000.,
                                 'actual_feedback': False, 'recovery': 'not defined for pointwise teacher outputs'})
        print(json.dumps({'sequence': sequence, 'clip': clip['clip_id'], 'calls': calls, 'wall_seconds': time.perf_counter() - start}), flush=True)
    handle.remove()
    dest = OUT / f'{sequence}.pt.gz'
    payload = {'protocol_sha256': ph, 'identity': identity, 'records': rows_out, 'pointwise_teacher': teachers,
               'calls': calls, 'teacher_extra_calls': teacher_extra, 'clamp_fk_calls': clamp_fk_calls,
               'q1_e1_replay_max_state_difference': max(replays), 'gt_wrist_clamp_max_mm': clamp_max,
               'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'wall_seconds': time.perf_counter() - start,
               'executed_script_sha256': sha(__file__)}
    with gzip.open(dest, 'wb', compresslevel=2) as stream: torch.save(payload, stream)
    compact = {key: value for key, value in payload.items() if key not in ('records', 'pointwise_teacher')}
    compact['raw_sha256'] = sha(dest)
    save_json(dest.with_suffix('.json'), compact)
    print(json.dumps({'done': sequence, **compact}), flush=True)


def summarize():
    import torch
    torch.set_num_threads(4)
    p = json.loads((OUT / 'protocol.json').read_text()); result = {'protocol_sha256': sha(OUT / 'protocol.json'), 'sequences': {}, 'audit': {}}
    all_calls = 0; seconds = 0.; max_replay = 0.; max_clamp = 0.; first_angles_equal = True
    for sequence in SEQUENCES:
        with gzip.open(OUT / f'{sequence}.pt.gz', 'rb') as stream: saved = torch.load(stream, map_location='cpu')
        records = saved['records']; teachers = saved['pointwise_teacher']; all_calls += saved['calls']; seconds += saved['wall_seconds']
        max_replay = max(max_replay, saved['q1_e1_replay_max_state_difference']); max_clamp = max(max_clamp, saved['gt_wrist_clamp_max_mm'])
        cohorts = {}
        for initialization in p['initializations']:
            q1 = {r['clip_id']: r for r in records if r['initialization'] == initialization and r['spec']['id'] == 'q1'}
            for mode in p['modes']:
                chosen = [r for r in records if r['initialization'] == initialization and r['spec']['id'] == mode]
                good = [r for r in chosen if r['finite']]
                out = {'n_clips': len(chosen), 'n_finite': len(good), 'n_failed': len(chosen) - len(good),
                       'actual_feedback': True, 'mechanism_oracle_only': mode == 'gt_wrist_clamp',
                       'auc21_actual_time': {key: stat([r['auc'][key] for r in chosen]) for key in chosen[0]['auc']},
                       'endpoint20': {key: stat(np.concatenate([r['errors'][key][1:] for r in chosen])) for key in chosen[0]['errors']},
                       'first': {key: stat([r['errors'][key][1] for r in chosen]) for key in chosen[0]['errors']},
                       'final': {key: stat([r['errors'][key][-1] for r in chosen]) for key in chosen[0]['errors']},
                       'z_signed_bias_mm': stat(np.concatenate([r['transl_z_error_mm'][1:] for r in chosen])),
                       'motion': {space: {metric: stat([r['motion'][space][metric] for r in chosen])
                                 for metric in ('gain', 'gain_bias', 'cosine', 'lag_ms')} for space in chosen[0]['motion']},
                       'roughness': {key: stat([r['roughness'][key] for r in chosen]) for key in chosen[0]['roughness']}}
                initial_good = [r for r in chosen if r['recovery']['initially_satisfied']]
                eligible = [r for r in chosen if not r['recovery']['initially_satisfied']]
                out['recovery'] = {'initially_satisfied': len(initial_good), 'eligible_new_recovery': len(eligible),
                    'new_sustained_recovery': sum(r['recovery']['hit_step'] is not None for r in eligible),
                    'initially_satisfied_final_good': sum(r['errors']['mpjpe_abs_mm'][-1] <= 20. and r['errors']['mpjpe_ra_mm'][-1] <= 10. for r in initial_good),
                    'final_good': sum(r['errors']['mpjpe_abs_mm'][-1] <= 20. and r['errors']['mpjpe_ra_mm'][-1] <= 10. for r in chosen),
                    'criterion': p['recovery_rule'], 'deployable_interpretation': mode != 'gt_wrist_clamp'}
                if mode not in ('hold', 'gt_wrist_clamp'):
                    same = all(np.array_equal(r['predictions'][1, 3:], q1[r['clip_id']]['predictions'][1, 3:]) for r in chosen)
                    first_angles_equal &= same; out['first_angles_exactly_q1'] = same
                if mode != 'gt_wrist_clamp':
                    point = [r for r in teachers if r['initialization'] == initialization and r['mode'] == mode]
                    out['pointwise_teacher'] = {'n_queries': sum(len(r['predictions']) for r in point), 'actual_feedback': False,
                        'recovery': 'not scored',
                        'endpoint': {key: stat(np.concatenate([r['errors'][key] for r in point])) for key in point[0]['errors']},
                        'z_signed_bias_mm': stat(np.concatenate([r['transl_z_error_mm'] for r in point]))}
                cohorts[f'{initialization}/{mode}'] = sanitize(out)
        result['sequences'][sequence] = cohorts
    receipts = [json.loads(path.read_text()) for path in (OUT / 'commands').glob('run_*.json')]
    charged = sum(r['wall_seconds'] for r in receipts)
    result['audit'] = {'actual_model_calls': all_calls, 'expected_primary_calls': 15360,
                       'q1_e1_max_state_difference': max_replay, 'gt_wrist_clamp_max_mm': max_clamp,
                       'first_free_head_angles_exactly_q1': first_angles_equal,
                       'worker_seconds_before_serialization': seconds, 'GPU_wall_seconds_including_serialization': charged,
                       'rollout_GPU_hours': charged / 3600.,
                       'interface_main_GPU_hours': json.loads((BASE / 'summary.json').read_text())['GPU_hours_charged']}
    result['limits'] = p['limits']; save_json(OUT / 'summary.json', sanitize(result))
    print(json.dumps(result['audit']), flush=True)


def main():
    start = time.perf_counter(); parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('prepare', 'run', 'summarize'))
    parser.add_argument('--sequence', choices=SEQUENCES); args = parser.parse_args(); status = 'complete'; error = None
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    try:
        if args.phase == 'prepare': prepare()
        elif args.phase == 'run':
            if not args.sequence: raise ValueError('--sequence required')
            run(args.sequence, start)
        else: summarize()
    except BaseException as exc:
        status = 'failed'; error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}; raise
    finally:
        save_json(OUT / 'commands' / f'{args.phase}_{args.sequence or "all"}_{time.time_ns()}.json',
                  {'argv': sys.argv, 'phase': args.phase, 'sequence': args.sequence, 'started_utc': started,
                   'wall_seconds': time.perf_counter() - start, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                   'status': status, 'error': error, 'script_sha256': sha(__file__)})


if __name__ == '__main__': main()
