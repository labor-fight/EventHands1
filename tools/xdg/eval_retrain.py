#!/usr/bin/env python3
"""XDG-1: pre-registration, launch freeze and mechanism evaluation of the three fresh Q1-derived runs.

  prepare     freeze outputs/xdg/retrain/protocol.json (gates) and the train.py source/config manifest
  evaluate    one run x one domain (zgz_local, zgz_global, check): GT previous state, previous t_z + {-50, 0, +50} mm,
              batch-1 forward, CPU or one GPU; raw outputs kept
  summarize   pre-registered gates over all evaluated runs

Runs: the original xdl_q1_s3407 plus xdg_q1r / xdg_q1t0 / xdg_q1dz (configs/xdg, tools/xdg/make_configs.py).
Writes outputs/xdg/retrain only; checkpoints and data are read only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
OUT = ROOT / 'outputs/xdg/retrain'
RUNS = ['xdl_q1_s3407', 'xdg_q1r_s3407', 'xdg_q1t0_s3407', 'xdg_q1dz_s3407']
NEW = {'xdg_q1t0_s3407': (2, 3), 'xdg_q1dz_s3407': (4, 5), 'xdg_q1r_s3407': (6, 7)}
DOMAINS = ['zgz_local', 'zgz_global', 'check']
DZ_MM = (0., -50., 50.)
NATIVE_MAX_MM = 718.4612365954911
TRAIN_SOURCES = ('tools/xdl/data.py', 'tools/xdl/train.py', 'semkine/xyz_observation.py',
                 'semkine/xyz_recovery_study.py', 'model/model.py', 'semkine/xyz_mesh.py', 'semkine/event_gnn.py',
                 'semkine/dataset.py', 'semkine/events.py', 'semkine/domrand.py', 'semkine/encoder.py',
                 'semkine/train.py', 'model/mano_layer.py', 'model/config.py', 'tools/xdg/make_configs.py',
                 'configs/xdl/xdl_q1_s3407.yaml', 'data/hand_data51/splits_semkine.json')


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def rel(p):
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(p).resolve())


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def save(p, v):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(clean(v), ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def read(p):
    return json.loads(Path(p).read_text())


def ident(p):
    return {'path': rel(p), 'sha256': sha(p)}


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        raise RuntimeError('protocol already frozen; do not overwrite')
    configs = [ROOT / f'configs/xdg/{r}.yaml' for r in NEW]
    for r in NEW:
        d = ROOT / f'outputs/semkine/{r}'
        if d.exists() and any(d.iterdir()):
            raise RuntimeError(f'fresh run directory not empty: {d}')
    p = {
        'study': 'XDG-1-fresh-retrain-explicit-translation-and-depth-support-20261014',
        'frozen_before_any_training': True, 'created_unix': time.time(),
        'authorization': 'user 2026-10-14: GPU2-7 six cards for training, all CPU',
        'prior_evidence': 'XDG-0 head refit (outputs/xdg/head/result.json): removing t columns cut local clean |z bias| '
                          '90.9/90.6 -> 7.5/3.7 mm and output p95 depth 657 -> 755 mm; check clean wrist 10.6 -> 14.8 mm',
        'base': 'configs/xdl/xdl_q1_s3407.yaml (XDL Q1): fresh init, seed3407, exact step6000, 2 GPU x 512, bf16, '
                'rawW50, self K4 burn, mixed previous noise; no intermediate checkpoint selection',
        'arms': {
            'xdg_q1t0_s3407': {'change': 'MODEL.XDL_STATE_T=false: state99 -> state96 (absolute t/.1 removed); fc1 704 inputs',
                               'gpus': [2, 3], 'master_port': 29720},
            'xdg_q1dz_s3407': {'change': 'AUG.DOMRAND.SCALE_MODE=depth: the scale moves the labelled root depth (same K) '
                                         'instead of the focal length; also changes size/depth semantics, so this is the '
                                         'semantic total effect, not pure coverage',
                               'gpus': [4, 5], 'master_port': 29721},
            'xdg_q1r_s3407': {'change': 'none (fresh same-seed repeat of Q1: engineering noise floor and clamp reproduction)',
                              'gpus': [6, 7], 'master_port': 29722}},
        'environment': {'NCCL_P2P_DISABLE': '1', 'NCCL_ASYNC_ERROR_HANDLING': '1', 'NCCL_DEBUG': 'INFO',
                        'python': '/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python'},
        'budget_gpu_hours': {'training_cap': 36.0, 'expected_per_arm': 9.3, 'formal_rows': 3.0},
        'smoke': 'CPU real-data batch16 forward + forced self burn + backward finite for all three configs; default path '
                 'bitwise equal to HEAD on 20 Q1 packets',
        'evaluation': {
            'mechanism': 'tools/xdg/eval_retrain.py on CPU (fp32, batch 1): every zgz_local (1204) and zgz_global (1386) '
                         'packet and the 576 E2 check packets (training identities, held-out blocks), GT previous state '
                         'with previous t_z + {0, -50, +50} mm; same for original Q1',
            'formal_rows': 'tools/xdl/make_row.py --run <run> --device cuda:0, then tools/report_table.py',
            'depth': 'actual wrist depth = t_z + J0_z(beta); wrist error = translation error (fixed beta)'},
        'primary_gate': {
            'validity': 'xdg_q1r must reproduce the clamp: local clean |mean signed z| >= 30 mm and output wrist depth '
                        'p95 <= 718.46 mm; otherwise the mechanism test is reported as not interpretable',
            'G1_bias': 'per arm (q1t0, q1dz): local clean |mean signed z| <= 0.5 x min(|Q1|, |Q1r|)',
            'G2_unclamped': 'per arm: local clean output wrist depth p95 > 718.46 mm (native training maximum)',
            'arm_pass': 'G1 and G2'},
        'interpretation': {
            'q1t0 pass': 'explicit absolute t is necessary for the clamp in an end-to-end trained network',
            'q1dz pass': 'depth-support exposure (with its scale semantics) removes the clamp while t stays',
            'q1t0 fail (XDG-0 passed)': 'the trained network rebuilds the depth prior through other state paths',
            'q1dz fail': 'thin exposure (4-5% of targets in 705-749 mm) is not enough, or semantics differ'},
        'descriptive_only': ['nine-column formal rows and Q1r-Q1 per-metric difference as the engineering noise floor',
                             'global clean and check clean wrist / signed z / root / finger', 'output-vs-input depth slope '
                             'at +/-50 mm (1 = hold, 0 = full correction; not a success criterion)'],
        'not_claimed': 'no S39 replacement; one seed; zgz is one subject; no recovery claim from single-step metrics',
        'configs': [ident(c) for c in configs],
        'inputs': [ident(ROOT / s) for s in TRAIN_SOURCES] + [ident(next((ROOT / 'outputs/semkine/xdl_q1_s3407').glob('*-step=6000.ckpt')))],
        'executed_source': ident(__file__)}
    save(path, p)
    manifest = {'status': 'frozen_for_xdg1_three_fresh_runs', 'protocol': ident(path),
                'source_sha256': {s: sha(ROOT / s) for s in TRAIN_SOURCES + tuple(rel(c) for c in configs)}}
    save(OUT / 'frozen_manifest.json', manifest)
    print(json.dumps({'protocol': rel(path), 'manifest': rel(OUT / 'frozen_manifest.json')}), flush=True)


def packets(domain, bank):
    if domain in ('zgz_local', 'zgz_global'):
        return [(domain, row) for row in bank['sequences'][domain]['rows']]
    e2 = read(ROOT / 'outputs/xde/e2/protocol.json')['selected_packets']
    return [(r['sequence'], r['row']) for r in e2 if r['split'] == 'check']


def evaluate(run, domain, device_name):
    import torch
    from tools.xdl import diagnostics as D
    from tools.xdl import geometry as G
    if device_name == 'cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
            raise RuntimeError('CPU evaluation requires CUDA_VISIBLE_DEVICES=""')
        torch.set_num_threads(int(os.environ.get('XDG_THREADS', '16')))
    dev = torch.device(device_name)
    start = time.perf_counter()
    model, cfg, identity = D.load_model(run, dev)
    mean = model.mano.hands_mean.detach().cpu().numpy()
    bank, bank_sha = D.load_bank()
    items = packets(domain, bank)
    j0 = read(ROOT / 'docs/assets/s37_next_controls_20261014/e2_domain_clean_audit.json')['sequence_J0_xyz_mm']
    cache = {}
    initial, truth, pred, seqs, failures = [], [], [], [], 0
    for seq, row in items:
        if seq not in cache:
            data = D.load_data(cfg, seq, bank['sequences'][seq])
            cache = {seq: (data,) + tuple(D.context(data, dev))}
        data, beta, camera = cache[seq]
        model.set_hand_context(beta, camera)
        ev = D.raw_events(data, row)
        gt_prev = np.asarray(data['gt'][row['previous_end_ms']], np.float32)
        outs = []
        for dz in DZ_MM:
            state = gt_prev.copy()
            state[2] += dz / 1000.
            D.reset(model)
            with torch.no_grad():
                value = model.forward_packet(D.make_packet(ev, state[None], beta, camera, dev, row))[0].float().cpu().numpy()
            if not np.isfinite(value).all():
                failures += 1
            outs.append(value)
        initial.append(gt_prev)
        truth.append(np.asarray(data['gt'][row['end_ms']], np.float32))
        pred.append(np.stack(outs))
        seqs.append(seq)
    initial, truth, pred = np.stack(initial), np.stack(truth), np.stack(pred)
    j0z = np.array([j0[s][2] for s in seqs])
    path = OUT / 'eval' / f'{run}__{domain}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, initial=initial, truth=truth, prediction=pred, dz_mm=np.array(DZ_MM), j0z=j0z,
                        sequence=np.array(seqs))
    gr, gf = G.rotations(truth, mean)
    summary = {'run': run, 'domain': domain, 'n_packets': len(items), 'nonfinite_predictions': failures,
               'identity': identity, 'packet_bank_sha256': bank_sha, 'raw': ident(path), 'by_dz': {}}
    hold = np.linalg.norm((initial[:, :3] - truth[:, :3]) * 1000., axis=1)
    for k, dz in enumerate(DZ_MM):
        p = pred[:, k].astype(np.float64)
        err = (p[:, :3] - truth[:, :3]) * 1000.
        wrist = np.linalg.norm(err, axis=1)
        depth = p[:, 2] * 1000. + j0z
        pr, pf = G.rotations(p.astype(np.float32), mean)
        summary['by_dz'][f'{dz:+.0f}'] = {
            'signed_xyz_mm': err.mean(0).tolist(), 'wrist_mm': float(wrist.mean()),
            'better_than_hold': int((wrist < hold).sum()) if dz == 0 else None, 'hold_wrist_mm': float(hold.mean()),
            'input_wrist_depth_mean_mm': float((initial[:, 2] * 1000. + dz + j0z).mean()),
            'output_wrist_depth_mm': {q: float(np.percentile(depth, int(q[1:]))) for q in ('p05', 'p50', 'p95')} | {'max': float(depth.max())},
            'output_above_native_max': float(np.mean(depth > NATIVE_MAX_MM)),
            'root_deg': float(G.geodesic_deg(pr, gr).mean()), 'finger_deg': float(G.geodesic_deg(pf, gf).mean(-1).mean())}
    din = np.concatenate([(initial[:, 2] * 1000. + dz + j0z) for dz in DZ_MM])
    dout = np.concatenate([pred[:, k, 2].astype(np.float64) * 1000. + j0z for k in range(len(DZ_MM))])
    summary['output_vs_input_depth_slope'] = float(np.polyfit(din, dout, 1)[0])
    summary['wall_seconds'] = time.perf_counter() - start
    summary['device'] = device_name
    summary['source_sha256'] = sha(__file__)
    summary['protocol_sha256'] = sha(OUT / 'protocol.json')
    save(path.with_suffix('.json'), summary)
    print(json.dumps({'run': run, 'domain': domain, 'n': len(items), 'seconds': summary['wall_seconds'],
                      'local_bias_z': summary['by_dz']['+0']['signed_xyz_mm'][2]}), flush=True)


def summarize():
    e = {(r, d): read(OUT / 'eval' / f'{r}__{d}.json') for r in RUNS for d in DOMAINS
         if (OUT / 'eval' / f'{r}__{d}.json').exists()}

    def bias(r):
        return abs(e[(r, 'zgz_local')]['by_dz']['+0']['signed_xyz_mm'][2])

    def p95(r):
        return e[(r, 'zgz_local')]['by_dz']['+0']['output_wrist_depth_mm']['p95']

    q1, q1r = 'xdl_q1_s3407', 'xdg_q1r_s3407'
    validity = bias(q1r) >= 30. and p95(q1r) <= NATIVE_MAX_MM
    arms = {}
    for arm in ('xdg_q1t0_s3407', 'xdg_q1dz_s3407'):
        g1 = bias(arm) <= .5 * min(bias(q1), bias(q1r))
        g2 = p95(arm) > NATIVE_MAX_MM
        arms[arm] = {'abs_bias_mm': bias(arm), 'output_depth_p95_mm': p95(arm), 'G1_bias': g1, 'G2_unclamped': g2,
                     'pass': g1 and g2}
    table = {f'{r}/{d}': {'wrist_mm': v['by_dz']['+0']['wrist_mm'], 'signed_z_mm': v['by_dz']['+0']['signed_xyz_mm'][2],
                          'root_deg': v['by_dz']['+0']['root_deg'], 'finger_deg': v['by_dz']['+0']['finger_deg'],
                          'output_depth_p50_mm': v['by_dz']['+0']['output_wrist_depth_mm']['p50'],
                          'slope': v['output_vs_input_depth_slope'], 'nonfinite': v['nonfinite_predictions']}
             for (r, d), v in e.items()}
    result = {'validity_q1r_reproduces_clamp': validity, 'q1_abs_bias_mm': bias(q1), 'q1r_abs_bias_mm': bias(q1r),
              'q1_p95_mm': p95(q1), 'q1r_p95_mm': p95(q1r), 'arms': arms, 'descriptive': table,
              'protocol': ident(OUT / 'protocol.json'), 'source': ident(__file__)}
    save(OUT / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'evaluate', 'summarize'])
    ap.add_argument('--run', choices=RUNS)
    ap.add_argument('--domain', choices=DOMAINS)
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    try:
        if args.command == 'prepare':
            prepare()
        elif args.command == 'evaluate':
            evaluate(args.run, args.domain, args.device)
        else:
            summarize()
    except Exception as error:
        save(OUT / 'failures' / f'{args.command}_{args.run}_{args.domain}.json',
             {'error': repr(error), 'traceback': traceback.format_exc(), 'arguments': vars(args)})
        raise
