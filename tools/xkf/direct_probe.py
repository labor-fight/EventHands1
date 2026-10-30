#!/usr/bin/env python3
"""Exploratory D measurement/DGP decomposition of frozen XKF debug checkpoints.

No optimizer, gain changes, or formal result rows. Every network target is zero.
GT enters only incoming-state context (unused by D), scoring and labelled oracle
surgeries after the network call. All rows, including failures, remain in NPZ.
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

import numpy as np
import torch

from mano_layer import ManoLayer
from semkine.events import EventPacketBatch
from semkine.lie import so3_exp, so3_log
from semkine.xkf_direct import dgp_translation
from tools.xdl import diagnostics as D
from tools.xkf.debug_eval import load_model, safe_json, sha
from tools.xkf.diagnostics import validate_bank_sequence
from tools.xkf.runtime import LongWindowEval

WINDOWS = (50, 100, 150, 200)
VARIANTS = ('predpose_pred2d', 'GTpose_pred2d', 'predpose_GT2d', 'GTpose_GT2d')
WEIGHTS = ('learned', 'equal')
SOURCES = ('tools/xkf/direct_probe.py', 'tools/xkf/runtime.py', 'tools/xkf/debug_eval.py',
           'tools/xkf/diagnostics.py', 'tools/xdl/diagnostics.py', 'semkine/xkf.py',
           'semkine/xkf_direct.py', 'model/model.py', 'model/mano_layer.py',
           'model/backbones.py', 'semkine/eval_track.py', 'semkine/events.py')


def finite_mean(a):
    a = np.asarray(a)
    return float(a.mean()) if a.size and np.isfinite(a).all() else None


def stats(a):
    a = np.asarray(a)
    good = bool(a.size and np.isfinite(a).all())
    return {'count': len(a), 'nonfinite_count': int((~np.isfinite(a)).sum()),
            'mean': float(a.mean()) if good else None,
            'p50': float(np.median(a)) if good else None,
            'p90': float(np.percentile(a, 90)) if good else None}


def correlation(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 2 or not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.std(), b.std()) < 1e-10:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def source_hashes():
    return {p: sha(REPO / p) for p in SOURCES}


def declaration(args, bank, banksha):
    return {
        'kind': 'XKF_DIRECT_DIAGNOSTIC_PREREG', 'exploratory': True,
        'declared_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'formal_gate_assessed': False, 'training_authorized_by_this_probe': False,
        'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_sha256': sha(args.checkpoint),
        'config': str(Path(args.config).resolve()), 'config_sha256': sha(args.config),
        'source_sha256': source_hashes(), 'bank_sha256': banksha,
        'cohorts': {n: {'available': len(bank['sequences'][n]['rows']),
                         'selected': (len(bank['sequences'][n]['rows']) if n.startswith('zgz_') else args.control_rows),
                         'rule': 'all chronological packets' if n.startswith('zgz_') else '128 evenly spaced integer linspace indices'}
                    for n in ('zgz_local', 'zgz_global', 'lyq_local', 'ch_global')},
        'windows_ms': WINDOWS, 'window_rule': 'min(requested W, endpoint-run_start+1), causal LNES',
        'variants': VARIANTS, 'weights': WEIGHTS,
        'oracle_warning': 'GTpose and GT2d are diagnostic interventions only; unavailable at deployment',
        'network_inputs': 'real raw W50 events, GT previous state, zero target, sequence beta and camera, causal LNES; direct_measurement never reads prev or target',
        'subsets': ['all', 'cold_first3_per_run', 'mature_W200', 'original_bad_debug_clip_first10', 'per_run'],
        'metrics': ['state transl vs physical MANO wrist, separately', 'root angle, RA/absolute MPJPE',
                    '2D visible error, centered radial scale and radius ratio, visibility',
                    'raw event counts, positive LNES occupancy, GT motion over evidence window',
                    'DGP normal matrix conditioning, learned confidence and equal-confidence comparison'],
        'failure_rule': 'full row denominator retained; NaNs retained in NPZ; nonfinite aggregate is null, never finite-only mean',
        'selection_rule': 'no weight, gain, checkpoint or model selection, no optimization against GT',
        'args': vars(args),
    }


def packet_batch(data, rows, prev, beta, camera, lnes, spans, device):
    chunks = []
    counts = []
    for i, row in enumerate(rows):
        ev = D.raw_events(data, row).copy()
        ev[:, 0] = i
        chunks.append(ev)
        counts.append(len(ev))
    n = len(rows)
    return EventPacketBatch(
        events=torch.from_numpy(np.concatenate(chunks)).to(device),
        ptr=torch.tensor(np.r_[0, np.cumsum(counts)], dtype=torch.long, device=device),
        sequence_id=torch.zeros(n, dtype=torch.long, device=device),
        t_start_us=torch.tensor([r['event_start_ms'] * 1000 for r in rows], dtype=torch.long, device=device),
        t_end_us=torch.tensor([(r['end_ms'] + 1) * 1000 for r in rows], dtype=torch.long, device=device),
        delta_t_s=torch.full((n,), .05, device=device),
        is_sequence_start=torch.tensor([r['state_interval_ms'] == 49 for r in rows], device=device),
        is_sequence_end=torch.zeros(n, dtype=torch.bool, device=device),
        target=torch.zeros_like(prev), prev_state=prev, betas=beta.expand(n, -1),
        camera_K=camera.expand(n, -1, -1), lnes=torch.from_numpy(np.stack(lnes)).to(device),
        lnes_window_ms=torch.tensor(spans, dtype=torch.long, device=device))


def project(model, joints, camera):
    fx, fy, cx, cy = model._intrinsics(camera)
    z = joints[..., 2]
    return torch.stack([fx[:, None] * joints[..., 0] / z + cx[:, None],
                        fy[:, None] * joints[..., 1] / z + cy[:, None]], -1)


def normalized(model, px, camera):
    fx, fy, cx, cy = model._intrinsics(camera)
    return torch.stack([(px[..., 0] - cx[:, None]) / fx[:, None],
                        (px[..., 1] - cy[:, None]) / fy[:, None]], -1)


def condition(uv, w):
    u, v = uv.unbind(-1)
    z, o = torch.zeros_like(u), torch.ones_like(u)
    A = torch.cat([torch.stack([-o, z, u], -1), torch.stack([z, -o, v], -1)], 1)
    ww = torch.cat([w, w], 1)
    normal = torch.einsum('bni,bn,bnj->bij', A, ww, A)
    eig = torch.linalg.eigvalsh(normal)
    return eig, eig[:, -1] / eig[:, 0]


@torch.no_grad()
def analyze_batch(model, packet, truth):
    if bool(packet.target.any()):
        raise AssertionError('network target must remain zero')
    direct, aux = model.direct_measurement(packet)
    beta, camera = packet.betas, packet.camera_K
    _, jg = model._fk(truth, beta)
    gt0 = truth.clone()
    gt0[:, :3] = 0
    _, jg0 = model._fk(gt0, beta)
    pxgt = project(model, jg, camera)
    pxpred = aux['kp_px']
    uvgt, uvpred = normalized(model, pxgt, camera), normalized(model, pxpred, camera)
    visible = (jg[..., 2] > 1e-3) & (pxgt[..., 0] >= 0) & (pxgt[..., 0] < model.render_w) & (pxgt[..., 1] >= 0) & (pxgt[..., 1] < model.render_h)
    gt_centered, pred_centered = pxgt - pxgt.mean(1, keepdim=True), pxpred - pxpred.mean(1, keepdim=True)
    energy = gt_centered.square().sum((1, 2))
    centered_scale = (gt_centered * pred_centered).sum((1, 2)) / energy
    radius_ratio = (pred_centered.square().sum((1, 2)) / energy).sqrt()
    kp_error = (pxpred - pxgt).norm(dim=-1)
    kp_visible_error = (kp_error * visible).sum(1) / visible.sum(1)
    result = {'direct_state': direct, 'truth_joints': jg, 'pred_joints0': aux['joints0'],
              'gt_joints0': jg0, 'kp_pred_px': pxpred, 'kp_gt_px': pxgt, 'kp_visible': visible,
              'kp_error_px': kp_error, 'kp_visible_mean_px': kp_visible_error,
              'kp_centered_scale': centered_scale, 'kp_radius_ratio': radius_ratio,
              'kp_centroid_error_px': (pxpred.mean(1) - pxgt.mean(1)).norm(dim=-1),
              'confidence': aux['w'], 'dgp_covariance': aux['dgp_covariance'],
              'gt_pose_zero_wrist': jg0[:, 0], 'pred_pose_zero_wrist': aux['joints0'][:, 0]}
    # Exact 2x4 factorial; each predicted pose is held fixed throughout its DGP solve.
    for wi, weight_name in enumerate(WEIGHTS):
        w = aux['w'] if weight_name == 'learned' else torch.ones_like(aux['w'])
        for vi, variant in enumerate(VARIANTS):
            gtpose, gt2d = variant.startswith('GTpose'), variant.endswith('GT2d')
            j0, uv = (jg0 if gtpose else aux['joints0']), (uvgt if gt2d else uvpred)
            t = dgp_translation(j0, uv, w)
            state = truth.clone() if gtpose else direct.clone()
            state[:, :3] = t
            joints = j0 + t[:, None, :]
            t_error = (t - truth[:, :3]) * 1000
            wrist_error = (joints[:, 0] - jg[:, 0]) * 1000
            rel = (joints - joints[:, :1]) - (jg - jg[:, :1])
            root = so3_log(so3_exp(truth[:, 3:6]).transpose(-1, -2) @ so3_exp(state[:, 3:6])).norm(dim=-1) * (180 / np.pi)
            eig, cond = condition(uv, w)
            prefix = f'{weight_name}__{variant}__'
            result.update({prefix + 'state': state, prefix + 'wrist': joints[:, 0],
                           prefix + 't_error_mm': t_error, prefix + 'wrist_error_mm': wrist_error,
                           prefix + 'abs_mpjpe_mm': (joints - jg).norm(dim=-1).mean(1) * 1000,
                           prefix + 'ra_mpjpe_mm': rel.norm(dim=-1).mean(1) * 1000,
                           prefix + 'root_deg': root, prefix + 'normal_eigenvalues': eig,
                           prefix + 'normal_condition': cond})
    return {k: v.detach().cpu().numpy() for k, v in result.items()}


def summarize(raw, mask):
    indices = np.flatnonzero(mask)
    out = {'n': len(indices), 'failed_rows': int((~raw['success'][indices]).sum()),
           'kp_visible_mean_px': stats(raw['kp_visible_mean_px'][indices]),
           'kp_centered_scale': stats(raw['kp_centered_scale'][indices]),
           'kp_radius_ratio': stats(raw['kp_radius_ratio'][indices]),
           'kp_centroid_error_px': stats(raw['kp_centroid_error_px'][indices]),
           'visible_fraction': finite_mean(raw['kp_visible'][indices]),
           'positive_lnes_fraction': stats(raw['positive_lnes_fraction'][indices]),
           'long_event_count': stats(raw['long_event_count'][indices]),
           'evidence_wrist_motion_mm': stats(raw['evidence_wrist_motion_mm'][indices]),
           'variants': {}}
    for w in WEIGHTS:
        for v in VARIANTS:
            p = f'{w}__{v}__'
            te, we = raw[p + 't_error_mm'][indices], raw[p + 'wrist_error_mm'][indices]
            out['variants'][p[:-2]] = {
                't_rmse_mm': float(np.sqrt(np.mean(np.sum(te ** 2, axis=-1)))) if len(te) and np.isfinite(te).all() else None,
                'physical_wrist_rmse_mm': float(np.sqrt(np.mean(np.sum(we ** 2, axis=-1)))) if len(we) and np.isfinite(we).all() else None,
                't_depth_bias_mm': finite_mean(te[:, 2]), 'physical_wrist_depth_bias_mm': finite_mean(we[:, 2]),
                't_depth_error_mm': stats(te[:, 2]),
                **{k: stats(raw[p + k][indices]) for k in ('abs_mpjpe_mm', 'ra_mpjpe_mm', 'root_deg', 'normal_condition')}}
    base = raw['learned__predpose_pred2d__t_error_mm'][indices, 2]
    out['descriptive_correlations_with_t_depth_error'] = {
        k: correlation(base, raw[k][indices]) for k in
        ('kp_centered_scale', 'kp_radius_ratio', 'kp_visible_mean_px', 'positive_lnes_fraction',
         'long_event_count', 'evidence_wrist_motion_mm', 'evidence_root_motion_deg')}
    return out


@torch.no_grad()
def sequence(model, cfg, name, meta, device, args):
    data = D.load_data(cfg, name, meta)
    validate_bank_sequence(data, meta)
    beta, camera = D.context(data, device)
    selection = np.arange(len(meta['rows'])) if name.startswith('zgz_') else np.linspace(0, len(meta['rows']) - 1, min(args.control_rows, len(meta['rows'])), dtype=int)
    if args.limit_rows:
        selection = selection[:args.limit_rows]
    rows = [meta['rows'][i] for i in selection]
    n = len(rows)
    run_steps = {}
    row_step = {}
    for i, row in enumerate(meta['rows']):
        rid = row['run_id']
        run_steps[rid] = run_steps.get(rid, 0) + 1
        row_step[i] = run_steps[rid]
    cold = np.array([row_step[i] <= 3 for i in selection])
    bad = np.isin(selection, meta['clips'][0]['end_indices'][:10]) if meta['clips'] else np.zeros(n, bool)
    base = {'selected_bank_index': selection, 'run_id': np.array([r['run_id'] for r in rows]),
            'end_ms': np.array([r['end_ms'] for r in rows]),
            'packet_id': np.array([r['packet_id'] for r in rows]),
            'step_in_run': np.array([row_step[i] for i in selection]),
            'truth': np.stack([data['gt'][r['end_ms']] for r in rows]),
            'previous_truth': np.stack([data['gt'][r['previous_end_ms']] for r in rows]),
            'short_event_count': np.array([r['n_raw_events'] for r in rows]),
            'cold_first3': cold, 'original_debug_clip': bad,
            'camera_K': camera.cpu().numpy(), 'beta': beta.cpu().numpy()}
    outputs, summaries, failures = {}, {}, []
    for window in WINDOWS:
        runtime = LongWindowEval(meta['valid_runs_ms'], window)
        raw = {k: v.copy() for k, v in base.items()}
        raw.update(success=np.zeros(n, bool), effective_window_ms=np.zeros(n, np.int64),
                   positive_lnes_fraction=np.full(n, np.nan), long_event_count=np.zeros(n, np.int64),
                   evidence_wrist_motion_mm=np.full(n, np.nan), evidence_root_motion_deg=np.full(n, np.nan))
        for begin in range(0, n, args.batch_size):
            stop = min(begin + args.batch_size, n)
            chunk = rows[begin:stop]
            lnes, spans = zip(*(runtime.lnes(data['events'], data['offsets'], r['end_ms']) for r in chunk))
            raw['effective_window_ms'][begin:stop] = spans
            raw['positive_lnes_fraction'][begin:stop] = [np.count_nonzero(a) / a.size for a in lnes]
            starts = [r['end_ms'] - w + 1 for r, w in zip(chunk, spans)]
            raw['long_event_count'][begin:stop] = [int(data['offsets'][r['end_ms'] + 1] - data['offsets'][s]) for r, s in zip(chunk, starts)]
            prev = torch.tensor(base['previous_truth'][begin:stop], device=device).float()
            truth = torch.tensor(base['truth'][begin:stop], device=device).float()
            packet = packet_batch(data, chunk, prev, beta, camera, lnes, spans, device)
            try:
                values = analyze_batch(model, packet, truth)
                for k, value in values.items():
                    if k not in raw:
                        raw[k] = np.full((n, *value.shape[1:]), np.nan, dtype=np.float64)
                    raw[k][begin:stop] = value
                raw['success'][begin:stop] = np.stack([np.isfinite(v).reshape(len(chunk), -1).all(-1) for v in values.values()]).all(0)
                start_state = torch.tensor(np.stack([data['gt'][s] for s in starts]), device=device).float()
                _, js = model._fk(start_state, packet.betas)
                raw['evidence_wrist_motion_mm'][begin:stop] = np.linalg.norm(values['truth_joints'][:, 0] - js[:, 0].cpu().numpy(), axis=-1) * 1000
                raw['evidence_root_motion_deg'][begin:stop] = (so3_log(so3_exp(start_state[:, 3:6]).transpose(-1, -2) @ so3_exp(truth[:, 3:6])).norm(dim=-1) * (180 / np.pi)).cpu().numpy()
                for k in np.flatnonzero(~raw['success'][begin:stop]):
                    failures.append({'sequence': name, 'window': window, 'row': begin + int(k), 'packet_id': chunk[k]['packet_id'], 'reason': 'nonfinite'})
            except Exception as e:
                failures.extend({'sequence': name, 'window': window, 'row': i, 'packet_id': rows[i]['packet_id'], 'reason': type(e).__name__, 'message': str(e)} for i in range(begin, stop))
            if (begin // args.batch_size) % 10 == 0:
                print(f'{name} W{window} {stop}/{n}', flush=True)
        if 'direct_state' not in raw:
            raise RuntimeError(f'All direct batches failed: {failures[:2]}')
        outputs[f'{name}_W{window}'] = raw
        masks = {'all': np.ones(n, bool), 'cold_first3_per_run': cold,
                 'mature_W200': raw['effective_window_ms'] == 200,
                 'original_bad_debug_clip_first10': bad}
        summaries[str(window)] = {k: summarize(raw, m) for k, m in masks.items()}
        summaries[str(window)]['per_run'] = {str(rid): summarize(raw, raw['run_id'] == rid) for rid in np.unique(raw['run_id'])}
    return outputs, summaries, failures


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batch-size', type=int, default=32)
    ap.add_argument('--cpu-threads', type=int, default=4)
    ap.add_argument('--control-rows', type=int, default=128)
    ap.add_argument('--limit-rows', type=int, default=0, help='smoke test only; never relabel as complete')
    ap.add_argument('--declare-only', action='store_true')
    args = ap.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bank, banksha = D.load_bank()
    declared = declaration(args, bank, banksha)
    path = output / 'preregistration.json'
    if path.exists():
        prior = json.loads(path.read_text())
        for key in ('checkpoint_sha256', 'config_sha256', 'source_sha256', 'bank_sha256'):
            if prior[key] != declared[key]:
                raise ValueError(f'preregistration identity changed: {key}')
        declared = prior
    else:
        path.write_text(json.dumps(declared, indent=2) + '\n')
    if args.declare_only:
        print(path)
        return
    if (output / 'result.json').exists():
        raise ValueError('refusing to overwrite completed probe')
    torch.set_num_threads(args.cpu_threads)
    started = time.perf_counter()
    device = torch.device(args.device)
    model, cfg, identity = load_model(args.config, args.checkpoint, device)
    for module in model.modules():
        if isinstance(module, ManoLayer):
            module.fast_chain = True
    result = {'kind': 'XKF_DIRECT_DIAGNOSTIC_EXPLORATORY', 'formal_gate_assessed': False,
              'finite_subset_reduction': False, 'partial_smoke': bool(args.limit_rows),
              'identity': identity, 'preregistration_sha256': sha(path),
              'source_sha256': declared['source_sha256'], 'bank_sha256': banksha,
              'state_translation_convention': 'state[:3] is MANO transl; physical wrist=FK(state)[0], store both explicitly',
              'sequences': {}, 'failures': [], 'raw_files': []}
    for name in ('zgz_local', 'zgz_global', 'lyq_local', 'ch_global'):
        arrays, summary, failures = sequence(model, cfg, name, bank['sequences'][name], device, args)
        for label, raw in arrays.items():
            destination = output / f'{label}.npz'
            np.savez_compressed(destination, **raw)
            result['raw_files'].append({'path': str(destination), 'sha256': sha(destination)})
        result['sequences'][name] = summary
        result['failures'].extend(failures)
        result['elapsed_seconds'] = time.perf_counter() - started
        (output / 'progress.json').write_text(json.dumps(safe_json(result), indent=2, allow_nan=False) + '\n')
    if source_hashes() != declared['source_sha256']:
        raise ValueError('producer or dependency changed while probe was running')
    result['complete'] = not args.limit_rows
    (output / 'result.json').write_text(json.dumps(safe_json(result), indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output / 'result.json'), 'failures': len(result['failures']), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
