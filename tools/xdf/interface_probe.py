#!/usr/bin/env python3
"""XDF-1 frozen complete-707 interface audit; physical CUDA devices 2/3 only.

Historical E2 files and production code are read-only. CLI: prepare, extract,
fit, summarize, state-cross. All reported heads use fixed epoch 300.
"""
from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model'))
OUT = ROOT / 'outputs/xdf/interface'
OLD = ROOT / 'outputs/xde/e2'
RUN = ROOT / 'outputs/semkine/xdl_q1_s3407'
INTERFACES = ('F', 'B', 'Wlegacy', 'state', 'wrongFull')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare():
    old = json.loads((OLD / 'protocol.json').read_text())
    names = sorted({r['sequence'] for r in old['selected_packets']})
    inputs = [OLD / 'protocol.json', ROOT / 'outputs/xdf/preregistration_v1.md',
              Path(old['checkpoint']), Path(old['config']), Path(old['official_split']),
              ROOT / 'semkine/xyz_recovery_study.py', ROOT / 'semkine/xyz_mesh.py',
              ROOT / 'semkine/xyz_observation.py', ROOT / 'tools/xde/e2_probes.py',
              ROOT / 'tools/xdl/diagnostics.py']
    for name in names:
        inputs.extend([OLD / 'cache' / f'{name}.npz', OLD / 'cache' / f'{name}.json'])
    p = {
        'study': 'XDF-1-COMPLETE707-20261014-V1', 'created_before_extraction': True,
        'source_hashes': {str(x): sha(x) for x in inputs}, 'script_at_freeze_sha256': sha(__file__),
        'source_protocol': str(OLD / 'protocol.json'), 'source_protocol_sha256': sha(OLD / 'protocol.json'),
        'selected_packets': old['selected_packets'], 'names': names,
        'packets': 2560, 'rows': 7680, 'source_forwards': 15360,
        'layout': {'legacy': [0, 544], 'bank': [544, 608], 'state': [608, 707]},
        'perturb_z_mm': [0., -50., 50.], 'source_batch': 1,
        'wrong_content': old['wrong_content'], 'fold_and_row_metadata': 'exactly reused E2 rows',
        'target': 'GT endpoint translation minus incoming, mm / 50; actual motion retained',
        'head': {'interfaces': list(INTERFACES), 'seeds': [1701, 1702], 'input': 707,
                 'width': 256, 'output': 3, 'architecture': 'Linear707,256;ReLU;Linear256,3',
                 'optimizer': 'fresh AdamW', 'lr': .001, 'weight_decay': .0001,
                 'batch_size': 256, 'epochs': 300, 'selection': 'fixed final epoch; no check/zgz selection',
                 'normalization': 'common real full707 fit mean/std floor .001; masks after normalization',
                 'paired_randomness': 'identical initial weights and explicit minibatch orders per seed for every arm'},
        'gate': {'metric': 'check clean wrist Euclidean endpoint mm (equals transl norm at fixed beta)',
                 'primary_references': ['B', 'Wlegacy'], 'event_references': ['state', 'wrongFull'],
                 'absolute_mm': 2., 'relative': .1, 'both_seeds_required': True},
        'numeric_checks': {'old_cache_max_abs': 1e-6, 'original_head_delta_replay_max_abs': 1e-6,
                           'finite_required': True, 'no_target_in_forward': True},
        'physical_gpus': [2, 3], 'threads_per_process': 4,
        'stage_max_gpu_hours': .5, 'per_process_seconds': {'extract': 450, 'fit': 360},
        'state_cross': 'CPU cached original fc1/fc2; obs(s0/sdelta) x state(s0/sdelta), all added to sdelta; explicit normalized t_z +/- .05 = +/-5mm',
        'limitations': ['head seeds are not full-model seeds', 'frozen backbone saw training identities/segments',
                        'wrong stream is an OOD content counterfactual', '707 controls do not isolate earlier candidate/aggregation losses'],
    }
    assert len(p['selected_packets']) == p['packets'] and len(names) == 74
    path = OUT / 'protocol.json'
    if path.exists():
        previous = json.loads(path.read_text())
        p['script_at_freeze_sha256'] = previous['script_at_freeze_sha256']
        if previous != p:
            raise RuntimeError('frozen protocol differs')
    else:
        save_json(path, p)
    (OUT / 'protocol.sha256').write_text(sha(path) + '\n')
    print(json.dumps({'phase': 'prepare', 'protocol_sha256': sha(path), 'packets': 2560, 'rows': 7680}), flush=True)


@functools.lru_cache(maxsize=1)
def protocol():
    p = json.loads((OUT / 'protocol.json').read_text())
    if sha(OLD / 'protocol.json') != p['source_protocol_sha256']:
        raise RuntimeError('historical protocol changed')
    return p


def gpu_contract():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('2', '3'):
        raise RuntimeError('CUDA_VISIBLE_DEVICES must be exactly physical 2 or 3')
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device('cuda:0')


def check_budget(start, phase):
    limit = protocol()['per_process_seconds'][phase]
    if time.perf_counter() - start > limit:
        raise RuntimeError(f'{phase} process budget {limit}s exhausted; retain partial outputs')


def exact_diff(a, b):
    a = np.asarray(a); b = np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError(f'invalid arrays: {a.shape}, {b.shape}')
    return float(np.max(np.abs(a - b), initial=0.))


def extract(shard, start):
    import torch
    import torch.nn.functional as functional
    from tools.xdl import diagnostics as D
    from tools.xde.e2_probes import wrong_events
    device = gpu_contract(); p = protocol(); ph = sha(OUT / 'protocol.json')
    oldp = json.loads((OLD / 'protocol.json').read_text())
    model, cfg, identity = D.load_model(str(RUN), device)
    model.xyz_keep_aux = True
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    captured = {}

    def full_hook(module, inputs):
        if inputs[0].shape[-1] != 707:
            raise RuntimeError('fc1 hook ran before _recovery_readout')
        captured['full'] = inputs[0].detach().clone()

    # Registration order guarantees this runs after production's pre-hook.
    handle = model.xyz.fc1.register_forward_pre_hook(full_hook)
    delta_handle = model.xyz.fc2.register_forward_hook(lambda _m, _i, out: captured.update(delta=out.detach().clone()))
    pre_handle = model.observation_bank.compress.register_forward_pre_hook(lambda _m, inputs: captured.update(pre=inputs[0].detach().clone()))
    bank = json.loads(Path(oldp['source_bank']).read_text())
    completed = 0; calls = 0; maxima = {}; pilot = None
    names = p['names'][shard::2]
    for name in names:
        dest = OUT / 'cache' / f'{name}.npz'
        if dest.exists():
            audit = json.loads(dest.with_suffix('.json').read_text())
            if audit['protocol_sha256'] != ph or audit['cache_sha256'] != sha(dest):
                raise RuntimeError('existing XDF cache identity mismatch')
            continue
        oldpath = OLD / 'cache' / f'{name}.npz'
        oldaudit = json.loads(oldpath.with_suffix('.json').read_text())
        if oldaudit['cache_sha256'] != sha(oldpath) or oldaudit['protocol_sha256'] != p['source_protocol_sha256']:
            raise RuntimeError('old cache identity mismatch')
        with np.load(oldpath) as loaded:
            old = {key: loaded[key] for key in loaded.files}
        data = D.load_data(cfg, name, bank['sequences'][name]); beta, camera = D.context(data, device)
        model.set_hand_context(beta, camera); D.reset(model)
        values = {key: [] for key in ('full707', 'wrong_full707', 'incoming51', 'target51', 'prediction51',
                                      'wrong_prediction51', 'target_xyz_mm', 'replay_delta51')}
        local_max = {}; idx = 0; query_seconds = 0.
        for item in [r for r in p['selected_packets'] if r['sequence'] == name]:
            row = item['row']; events = D.raw_events(data, row)
            wrong = wrong_events(events, D.raw_events(data, item['donor_row']))
            for dz in p['perturb_z_mm']:
                incoming = data['gt'][row['previous_end_ms']].copy(); incoming[2] += dz / 1000.
                target = data['gt'][row['end_ms']].copy()
                meta = oldaudit['rows'][idx]
                if any(meta[key] != val for key, val in [('index', row['index']), ('dz_mm', dz), ('split', item['split'])]):
                    raise RuntimeError('old row order changed')
                if hashlib.sha256(events.tobytes()).hexdigest() != meta['event_sha256']:
                    raise RuntimeError('event bytes changed')
                for label, raw in (('', events), ('wrong_', wrong)):
                    check_budget(start, 'extract')
                    packet = D.make_packet(raw, incoming[None], beta, camera, device, row)
                    if packet.target.any() or not len(raw):
                        raise RuntimeError('target leak or undeclared empty packet')
                    captured.clear(); torch.cuda.synchronize(); query_start = time.perf_counter()
                    with torch.no_grad():
                        prediction = model.forward_packet(packet)
                        full = captured['full']
                        replay = functional.linear(torch.relu(functional.linear(full, model.xyz.fc1.weight, model.xyz.fc1.bias)),
                                                   model.xyz.fc2.weight, model.xyz.fc2.bias)
                    torch.cuda.synchronize(); query_seconds += time.perf_counter() - query_start; calls += 1
                    feature = full.cpu().numpy()[0]; prediction = prediction.cpu().numpy()[0]
                    comparisons = {
                        'head_delta_replay': (replay.cpu().numpy()[0], captured['delta'].cpu().numpy()[0]),
                        'endpoint_replay': (replay.cpu().numpy()[0] + incoming, prediction),
                        label + 'old_bank': (feature[544:608], old[label + 'post'][idx, :, :4].reshape(-1)),
                        label + 'old_pre': (captured['pre'].cpu().numpy()[0], old[label + 'pre'][idx]),
                        label + 'old_state': (feature[608:], old['state'][idx]),
                        'old_incoming': (incoming, old['incoming51'][idx]),
                        'old_target': (target, old['target51'][idx]),
                    }
                    if not label:
                        comparisons['old_prediction'] = (prediction, old['prediction51'][idx])
                    for key, pair in comparisons.items():
                        diff = exact_diff(*pair); local_max[key] = max(diff, local_max.get(key, 0.)); maxima[key] = max(diff, maxima.get(key, 0.))
                        if diff > 1e-6:
                            raise RuntimeError(f'identity check {name}:{idx}:{key} differs {diff}')
                    values[label + 'full707'].append(feature)
                    values[label + 'prediction51'].append(prediction)
                    if not label:
                        values['replay_delta51'].append(replay.cpu().numpy()[0])
                values['incoming51'].append(incoming); values['target51'].append(target)
                values['target_xyz_mm'].append((target[:3] - incoming[:3]) * 1000.)
                idx += 1
                if calls >= 600 and pilot is None:
                    elapsed = time.perf_counter() - start
                    pilot = {'source_packets': 100, 'calls': calls, 'elapsed_seconds_including_startup': elapsed,
                             'estimated_shard_seconds': elapsed * 7680 / calls}
                    print(json.dumps({'phase': 'pilot', 'shard': shard, **pilot}), flush=True)
                    if pilot['estimated_shard_seconds'] > p['per_process_seconds']['extract']:
                        raise RuntimeError('pilot exceeds preregistered process budget')
        assert idx == len(oldaudit['rows'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest, **{key: np.stack(value) for key, value in values.items()})
        save_json(dest.with_suffix('.json'), {'protocol_sha256': ph, 'executed_script_sha256': sha(__file__),
                  'source_cache': str(oldpath), 'source_cache_sha256': sha(oldpath), 'identity': identity,
                  'rows': oldaudit['rows'], 'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'],
                  'numeric_max_abs': local_max, 'query_seconds': query_seconds, 'cache_sha256': sha(dest)})
        completed += idx
        print(json.dumps({'phase': 'sequence_done', 'shard': shard, 'sequence': name, 'rows': idx,
                          'calls': calls, 'wall_seconds': time.perf_counter() - start}), flush=True)
    handle.remove(); delta_handle.remove(); pre_handle.remove()
    result = {'protocol_sha256': ph, 'calls': calls, 'new_rows': completed, 'numeric_max_abs': maxima,
              'pilot': pilot, 'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'], 'wall_seconds': time.perf_counter() - start}
    save_json(OUT / f'extract_shard{shard}.json', result)
    return result


def load_cache():
    p = protocol(); ph = sha(OUT / 'protocol.json'); arrays = {}; metadata = []
    for name in p['names']:
        path = OUT / 'cache' / f'{name}.npz'; audit = json.loads(path.with_suffix('.json').read_text())
        if audit['protocol_sha256'] != ph or audit['cache_sha256'] != sha(path):
            raise RuntimeError('complete cache identity mismatch')
        with np.load(path) as item:
            for key in item.files:
                arrays.setdefault(key, []).append(item[key])
        metadata.extend(audit['rows'])
    arrays = {key: np.concatenate(value) for key, value in arrays.items()}
    if len(metadata) != p['rows'] or not all(np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('incomplete/nonfinite cache')
    fit = {r['segment'] for r in metadata if r['split'] == 'fit'}
    check = {r['segment'] for r in metadata if r['split'] == 'check'}
    if fit & check:
        raise RuntimeError('fit/check segment overlap')
    return arrays, metadata, p


def normalize_interface(full, wrong, mean, std, interface):
    """Shared fit normalization then exact ablation masking; NumPy float32."""
    x = np.array((full - mean) / std, dtype=np.float32, copy=True)
    if interface == 'B':
        x[:, :544] = 0.
    elif interface == 'Wlegacy':
        x[:, :544] = (wrong[:, :544] - mean[:544]) / std[:544]
    elif interface == 'state':
        x[:, :608] = 0.
    elif interface == 'wrongFull':
        x[:, :608] = (wrong[:, :608] - mean[:608]) / std[:608]
    elif interface != 'F':
        raise ValueError(interface)
    return x


def make_head():
    from torch import nn
    return nn.Sequential(nn.Linear(707, 256), nn.ReLU(), nn.Linear(256, 3))


def metrics(prediction, target):
    if not len(prediction):
        return {'n': 0}
    error = np.asarray(prediction, np.float64) - np.asarray(target, np.float64)
    norm = np.linalg.norm(error, axis=-1); hold = np.linalg.norm(target, axis=-1)
    active = np.abs(target[:, 2]) >= 1.
    return {'n': len(prediction), 'wrist_mm': float(norm.mean()), 'wrist_p95_mm': float(np.quantile(norm, .95)),
            'xyz_mae_mm': np.abs(error).mean(0).tolist(), 'xyz_signed_mm': error.mean(0).tolist(),
            'z_mae_mm': float(np.abs(error[:, 2]).mean()), 'z_signed_bias_mm': float(error[:, 2].mean()),
            'hold_wrist_mm': float(hold.mean()), 'hold_z_mae_mm': float(np.abs(target[:, 2]).mean()),
            'better_than_hold_fraction': float((norm < hold).mean()), 'z_active_n': int(active.sum()),
            'z_direction_accuracy': float((prediction[active, 2] * target[active, 2] > 0).mean()) if active.any() else None}


def metric_groups(prediction, target, metadata, subjects=True):
    split = np.array([r['split'] for r in metadata]); dz = np.array([r['dz_mm'] for r in metadata])
    seq = np.array([r['sequence'] for r in metadata]); groups = {}
    for key in ('fit', 'check', 'zgz'):
        groups[key] = split == key
        for mode in ('local', 'global'):
            groups[f'{key}_{mode}'] = (split == key) & np.array([f'_{mode}' in x for x in seq])
    if subjects:
        for subject in sorted({x.split('_')[0] for x in seq}):
            for key in ('fit', 'check', 'zgz'):
                mask = (split == key) & np.array([x.startswith(subject + '_') for x in seq])
                if mask.any():
                    groups[f'{key}_subject_{subject}'] = mask
    result = {}
    for name, mask in groups.items():
        result[name] = metrics(prediction[mask], target[mask])
        for value in (0., -50., 50.):
            take = mask & (dz == value)
            result[f'{name}_dz{int(value)}'] = metrics(prediction[take], target[take])
    return result


def fit(seed, start):
    import torch
    device = gpu_contract(); data, metadata, p = load_cache()
    fit_mask = np.array([row['split'] == 'fit' for row in metadata]); fi = np.flatnonzero(fit_mask)
    mean = data['full707'][fit_mask].mean(0); std = np.maximum(data['full707'][fit_mask].std(0), .001)
    y = data['target_xyz_mm'].astype(np.float32); yt = torch.from_numpy(y / 50.).to(device)
    # CPU generator makes all epoch/batch indices explicit and bit-identical across arms.
    generator = torch.Generator(device='cpu').manual_seed(seed)
    orders = [fi[torch.randperm(len(fi), generator=generator).numpy()] for _ in range(p['head']['epochs'])]
    order_sha = hashlib.sha256(np.stack(orders).tobytes()).hexdigest()
    results = []; initial_sha = None
    for interface in p['head']['interfaces']:
        check_budget(start, 'fit'); torch.manual_seed(seed)
        head = make_head().to(device)
        current_sha = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in head.state_dict().values())).hexdigest()
        initial_sha = initial_sha or current_sha
        if initial_sha != current_sha:
            raise RuntimeError('paired initialization mismatch')
        x = normalize_interface(data['full707'], data['wrong_full707'], mean, std, interface)
        xt = torch.from_numpy(x).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        arm_start = time.perf_counter(); curve = []
        for epoch, indices in enumerate(orders, 1):
            check_budget(start, 'fit'); head.train(); total_loss = 0.; count = 0
            for begin in range(0, len(indices), 256):
                batch = torch.as_tensor(indices[begin:begin + 256], device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = (head(xt[batch]) - yt[batch]).square().mean()
                loss.backward(); optimizer.step()
                total_loss += float(loss.detach()) * len(batch); count += len(batch)
            if epoch % 25 == 0:
                curve.append({'epoch': epoch, 'fit_mse_normalized': total_loss / count})
        head.eval()
        with torch.no_grad():
            predictions = torch.cat([head(part) for part in xt.split(1024)]).cpu().numpy() * 50.
        active = int((np.ptp(x[fit_mask], axis=0) > 0).sum())
        report = {'interface': interface, 'seed': seed, 'epochs': 300, 'selected_epoch': 300,
                  'epoch_selection': 'fixed', 'nominal_parameters': sum(q.numel() for q in head.parameters()),
                  'variable_inputs': active, 'active_first_layer_parameters': active * 256,
                  'initial_weights_sha256': initial_sha, 'minibatch_order_sha256': order_sha,
                  'normalization_mean_sha256': hashlib.sha256(mean.tobytes()).hexdigest(),
                  'normalization_std_sha256': hashlib.sha256(std.tobytes()).hexdigest(),
                  'train_seconds': time.perf_counter() - arm_start, 'curve': curve,
                  'metrics': metric_groups(predictions, y, metadata)}
        path = OUT / 'heads' / f'{interface}_s{seed}.pt'; path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    'mean': mean, 'std': std, 'interface': interface, 'seed': seed, 'width': 256,
                    'input_layout': p['layout'], 'protocol_sha256': sha(OUT / 'protocol.json'),
                    'epochs': 300, 'target_scale_mm': 50.}, path)
        np.savez_compressed(path.with_suffix('.predictions.npz'), delta_xyz_mm=predictions, target_xyz_mm=y)
        report['checkpoint_sha256'] = sha(path); report['prediction_sha256'] = sha(path.with_suffix('.predictions.npz'))
        save_json(path.with_suffix('.json'), report); results.append(report)
        print(json.dumps({'phase': 'fit_done', 'seed': seed, 'interface': interface,
                          'train_seconds': report['train_seconds'], 'check_clean': report['metrics']['check_dz0'],
                          'local_clean': report['metrics']['zgz_local_dz0'], 'global_clean': report['metrics']['zgz_global_dz0']}), flush=True)
    result = {'protocol_sha256': sha(OUT / 'protocol.json'), 'physical_gpu': os.environ['CUDA_VISIBLE_DEVICES'],
              'wall_seconds': time.perf_counter() - start, 'results': results}
    save_json(OUT / f'fit_s{seed}.json', result)
    return result


def summarize():
    data, metadata, p = load_cache(); reports = {}
    for seed in p['head']['seeds']:
        for result in json.loads((OUT / f'fit_s{seed}.json').read_text())['results']:
            reports[(seed, result['interface'])] = result
    gates = []
    for seed in p['head']['seeds']:
        for reference in ('B', 'Wlegacy', 'state', 'wrongFull'):
            candidate = reports[(seed, 'F')]['metrics']['check_dz0']['wrist_mm']
            base = reports[(seed, reference)]['metrics']['check_dz0']['wrist_mm']
            gain = base - candidate
            gates.append({'seed': seed, 'reference': reference, 'F_mm': candidate, 'reference_mm': base,
                          'improvement_mm': gain, 'relative_improvement': gain / base,
                          'pass': gain >= 2. and gain >= .1 * base})
    baseline = (data['prediction51'][:, :3] - data['incoming51'][:, :3]) * 1000.
    maxima = {}
    for name in p['names']:
        a = json.loads((OUT / 'cache' / f'{name}.json').read_text())
        for key, value in a['numeric_max_abs'].items():
            maxima[key] = max(value, maxima.get(key, 0.))
    cost = []
    for path in sorted((OUT / 'commands').glob('*.json')):
        item = json.loads(path.read_text())
        if item.get('physical_gpu') in ('2', '3'):
            cost.append({'path': str(path), 'phase': item['phase'], 'wall_seconds': item['wall_seconds'], 'status': item['status']})
    result = {'protocol_sha256': sha(OUT / 'protocol.json'), 'rows': len(metadata), 'numeric_max_abs': maxima,
              'primary_legacy_gate_pass': all(x['pass'] for x in gates if x['reference'] in ('B', 'Wlegacy')),
              'event_control_gate_pass': all(x['pass'] for x in gates if x['reference'] in ('state', 'wrongFull')),
              'gates': gates, 'original_Q1': metric_groups(baseline, data['target_xyz_mm'], metadata),
              'GPU_commands': cost, 'GPU_hours_charged': sum(x['wall_seconds'] for x in cost) / 3600.,
              'results': list(reports.values()), 'interpretation': 'frozen feature readout screening; not full-model seed replication, not recovery proof'}
    save_json(OUT / 'summary.json', result)
    print(json.dumps({key: result[key] for key in ('rows', 'numeric_max_abs', 'primary_legacy_gate_pass', 'event_control_gate_pass', 'GPU_hours_charged', 'gates')}), flush=True)


def state_cross():
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        raise RuntimeError('state-cross is CPU only: set CUDA_VISIBLE_DEVICES empty')
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    data, metadata, p = load_cache(); oldp = json.loads((OLD / 'protocol.json').read_text())
    checkpoint = torch.load(oldp['checkpoint'], map_location='cpu')
    sd = checkpoint['state_dict']; keys = ('xyz.fc1.weight', 'xyz.fc1.bias', 'xyz.fc2.weight', 'xyz.fc2.bias')
    weights = [sd[key].float() for key in keys]

    def replay(values):
        with torch.no_grad():
            x = torch.from_numpy(np.asarray(values, np.float32))
            return F.linear(torch.relu(F.linear(x, weights[0], weights[1])), weights[2], weights[3]).numpy()

    full = data['full707']; delta = replay(full)
    cpu_replay_max = exact_diff(delta, data['replay_delta51'])
    clean_map = {(r['sequence'], r['index']): i for i, r in enumerate(metadata) if r['dz_mm'] == 0}
    ci = np.array([clean_map[(r['sequence'], r['index'])] for r in metadata]); clean = full[ci]
    results = {}; arrays = {}
    for obs in ('s0', 'sdelta'):
        for state in ('s0', 'sdelta'):
            x = full.copy(); x[:, :608] = clean[:, :608] if obs == 's0' else full[:, :608]
            x[:, 608:] = clean[:, 608:] if state == 's0' else full[:, 608:]
            key = f'obs_{obs}_state_{state}'
            prediction = replay(x)[:, :3] * 1000.
            arrays[key] = prediction
            results[key] = metric_groups(prediction, data['target_xyz_mm'], metadata)
    tz = {}
    for sign in (-1, 1):
        x = full.copy(); x[:, 610] += sign * .05  # state translation normalized by .1m
        changed = replay(x)[:, :3] * 1000.
        key = f'explicit_tz_{sign * 5:+d}mm'
        arrays[key] = changed; results[key] = metric_groups(changed, data['target_xyz_mm'], metadata)
        tz[key] = {'delta_change_xyz_mm_mean': (changed - delta[:, :3] * 1000.).mean(0).tolist(),
                   'z_update_gain_per_injected_mm': float((changed[:, 2] - delta[:, 2] * 1000.).mean() / (sign * 5))}
    np.savez_compressed(OUT / 'state_cross.predictions.npz', **arrays)
    result = {'protocol_sha256': sha(OUT / 'protocol.json'), 'CPU_original_delta_replay_max_abs': cpu_replay_max,
              'all_outputs_added_to': 'same actual sdelta; target actual GTnext', 'rows': len(metadata),
              'counterfactual_limit': 'mixed state/observation is off-manifold; explicit tz change leaves UV/valid/obs fixed',
              'tz_sensitivity': tz, 'metrics': results,
              'prediction_sha256': sha(OUT / 'state_cross.predictions.npz')}
    save_json(OUT / 'state_cross.json', result)
    print(json.dumps({'phase': 'state_cross', 'CPU_replay_max': cpu_replay_max, 'tz_sensitivity': tz}), flush=True)


def main():
    start = time.perf_counter(); started = utc()
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=('prepare', 'extract', 'fit', 'summarize', 'state-cross'))
    parser.add_argument('--shard', type=int, choices=(0, 1)); parser.add_argument('--seed', type=int, choices=(1701, 1702))
    args = parser.parse_args(); status = 'complete'; error = None
    try:
        if args.phase == 'prepare': prepare()
        elif args.phase == 'extract':
            if args.shard is None: raise ValueError('--shard required')
            extract(args.shard, start)
        elif args.phase == 'fit':
            if args.seed is None: raise ValueError('--seed required')
            fit(args.seed, start)
        elif args.phase == 'summarize': summarize()
        else: state_cross()
    except BaseException as exc:
        status = 'failed'; error = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()}
        raise
    finally:
        suffix = str(args.shard) if args.shard is not None else str(args.seed) if args.seed is not None else 'all'
        path = OUT / 'commands' / f'{args.phase}_{suffix}_{time.time_ns()}.json'
        save_json(path, {'phase': args.phase, 'argv': sys.argv, 'started_utc': started, 'finished_utc': utc(),
                  'wall_seconds': time.perf_counter() - start, 'physical_gpu': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
                  'status': status, 'error': error, 'script_sha256': sha(__file__)})


if __name__ == '__main__':
    main()
