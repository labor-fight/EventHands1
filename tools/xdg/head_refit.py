#!/usr/bin/env python3
"""XDG-0: frozen Q1 full707 final head refit, only the explicit state translation columns removed.

Mechanism under test (docs/S37_XYZ_ASSOC_DIAGNOSIS.md, XDG): Q1 reads the absolute previous translation
`prev[:, :3] / .1` (full707 columns 608:611), is trained with mixed previous-state noise (30% sigma 50 mm)
on native wrist depths <= 718.5 mm, and outputs local wrist depths capped near the training upper tail.

Arms (all from the same Q1 fc1/fc2, same 14400-row XDF-3 cache, same loss, order, epochs, CPU fp32):
  L0      the XDF-3 z-weight-1 head, re-fitted on CPU as the matched control
  noT     L0 with full707 columns 608:611 (t / .1) set to zero in training and evaluation
  noTUV   secondary: also the 32 projected-joint UV columns 659:691 set to zero

CPU only. Writes outputs/xdg/head only; the production model, checkpoints and XDF caches are read only.
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
OUT = ROOT / 'outputs/xdg/head'
RUN = ROOT / 'outputs/semkine/xdl_q1_s3407'
XDF = ROOT / 'outputs/xdf'
#: full707 = legacy544 | bank64 | state99 = [t/.1 (3), AA48, UV32, valid16]
T_COLS = list(range(608, 611))
UV_COLS = list(range(659, 691))
ARMS = {'L0': [], 'noT': T_COLS, 'noTUV': T_COLS + UV_COLS}
JOBS = [('L0', 1701, 0), ('L0', 1702, 0), ('noT', 1701, 0), ('noT', 1702, 0), ('noT', 1701, 1),
        ('noTUV', 1701, 0), ('noTUV', 1702, 0)]
#: native training target wrist depth maximum, outputs/xdf/depth/native_population.json
NATIVE_MAX_MM = 718.4612365954911


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


def tag(job):
    arm, seed, repeat = JOBS[job]
    return f'{arm}_s{seed}_r{repeat}'


def cpu():
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        raise RuntimeError('XDG-0 is CPU only: set CUDA_VISIBLE_DEVICES=""')
    import torch
    torch.set_num_threads(int(os.environ.get('XDG_THREADS', '8')))
    torch.use_deterministic_algorithms(True)
    return torch.device('cpu')


def checkpoint():
    return next(RUN.glob('*-step=6000.ckpt'))


def sources():
    paths = [checkpoint(), RUN / 'config_resolved.yaml', XDF / 'objective/protocol.json',
             XDF / 'objective/eval_clean.json', XDF / 'depth/native_population.json',
             ROOT / 'docs/assets/s37_next_controls_20261014/e2_domain_clean_audit.json',
             ROOT / 'tools/xdf/objective_probe.py', ROOT / 'semkine/xyz_recovery_study.py']
    for d in ('interface/cache', 'angles/cache', 'objective/own_cache', 'objective/clean_cache'):
        paths += sorted((XDF / d).glob('*.npz'))
    return paths


def prepare():
    path = OUT / 'protocol.json'
    if path.exists():
        raise RuntimeError('protocol already frozen; do not overwrite')
    p = {
        'study': 'XDG-0-head-refit-without-explicit-translation-20261014',
        'frozen_before_any_fit_or_evaluation': True,
        'created_unix': time.time(),
        'mechanism': 'explicit absolute previous translation (state99 t/.1) x mixed previous noise x native depth '
                     'support <= 718.5 mm -> learned shrinkage to the training depth tail; local zgz wrist depth '
                     '705.5-749.4 mm lies wholly above the training p99 671.9 mm',
        'device': 'CPU fp32, torch deterministic algorithms, no autocast',
        'model': 'original Q1 seed3407 step6000; heads initialised from its fc1/fc2; all features frozen (XDF caches)',
        'columns': {'full707': 'legacy544 | bank64 | state99[t/.1 3, AA48, UV32, valid16]',
                    't_columns': T_COLS, 'uv_columns': UV_COLS},
        'arms': {'L0': 'XDF-3 z-weight-1 head refitted on CPU (matched control)',
                 'noT': 'only change: full707 columns 608:611 zero in training and evaluation (primary)',
                 'noTUV': 'secondary diagnostic: also UV columns 659:691 zero; not part of the primary gate'},
        'jobs': [{'job': i, 'tag': tag(i), 'arm': a, 'seed': s, 'repeat': r} for i, (a, s, r) in enumerate(JOBS)],
        'training': 'identical to tools/xdf/objective_probe.py fit with z_weight 1: XDF-3 14400 fit rows (interface 5184, '
                    'angles 3456, own 5760), loss log10((450 mse finger AA + 60 mse root AA + 30000 mse t)/51), '
                    'pred = incoming + head(x), fresh Adam lr 1e-4, 100 epochs, batch 512, CPU torch.randperm order '
                    'seeded by the head seed, fixed final epoch, no selection',
        'evaluation': {
            'local_clean': 'objective/clean_cache/zgz_local.npz, all 1204 GT-previous packets',
            'global_clean': 'objective/clean_cache/zgz_global.npz, all 1386 packets',
            'perturbed': 'interface/cache zgz rows, 128 packets x dz {-50, 0, +50} per sequence',
            'check': 'interface/cache split=check (training identities, held-out blocks), clean and +/-50',
            'depth': 'actual wrist depth = t_z + J0_z(beta) per sequence; wrist error = translation error (fixed beta)',
            'references': 'original Q1 head; XDF-3 GPU L0 heads (outputs/xdf/objective/heads/z1_*)'},
        'primary_gate': {
            'G1_bias': 'per seed 1701 and 1702: |mean signed z error| of noT on local clean 1204 <= 0.5 x that of the '
                       'same-seed CPU L0',
            'G2_unclamped': 'per seed: p95 of noT output actual wrist depth on local clean 1204 > 718.46 mm '
                            '(native training maximum)',
            'pass': 'G1 and G2 for both seeds',
            'determinism': 'noT s1701 r1 must equal r0 bit for bit; otherwise report the difference as the noise floor'},
        'descriptive_only': ['local/global +/-50 output-vs-input depth slope (slope 1 = hold, 0 = full correction; '
                             'not a success criterion)', 'global clean wrist and signed z', 'check clean wrist',
                             'root / finger geodesic error', 'noTUV arm', 'CPU L0 versus GPU L0 agreement'],
        'interpretation': {
            'pass': 'explicit t columns are necessary for the clamp at the frozen-feature head level; supports the '
                    'mechanism; full matched retrains still required (features come from a t-trained network)',
            'noT fail, noTUV pass': 'projected-joint UV scale substitutes for t as a depth-prior path',
            'both fail': 'the clamp is carried by legacy544/bank64 features or by fit-target support itself; '
                         'explicit t is not necessary at the head level'},
        'not_claimed': 'no deployment candidate, no main-table row, no recovery claim; zgz is one subject',
        'inputs': [ident(p) for p in sources()],
        'executed_source': ident(__file__)}
    save(path, p)
    print(json.dumps({'protocol': rel(path), 'jobs': len(JOBS)}), flush=True)


def protocol_sha():
    return sha(OUT / 'protocol.json')


def read_cache_dir(directory, fit_only):
    keys = ('full707', 'incoming51', 'target51', 'prediction51')
    arrays = {k: [] for k in keys}
    meta = []
    for path in sorted(Path(directory).glob('*.npz')):
        audit = read(path.with_suffix('.json'))
        rows = audit['rows']
        if audit.get('cache_sha256') != sha(path):
            raise RuntimeError(f'cache SHA changed: {path}')
        with np.load(path) as z:
            assert z['full707'].shape == (len(rows), 707)
            keep = np.array([r.get('split') == 'fit' or not fit_only for r in rows], bool)
            for k in keys:
                arrays[k].append(z[k][keep])
        for r in np.array(rows, dtype=object)[keep]:
            r = dict(r)
            r.setdefault('sequence', path.stem)
            meta.append(r)
    return {k: np.concatenate(v) for k, v in arrays.items()}, meta


def fit_data():
    expected = read(XDF / 'objective/protocol.json')['expected_own_rows']
    parts = [('interface', XDF / 'interface/cache', 5184), ('angles', XDF / 'angles/cache', 3456),
             ('own', XDF / 'objective/own_cache', expected)]
    arrays = {k: [] for k in ('full707', 'incoming51', 'target51')}
    counts = {}
    for name, path, n in parts:
        a, m = read_cache_dir(path, True)
        if len(m) != n:
            raise RuntimeError(f'{name} incomplete: {len(m)} != {n}')
        for k in arrays:
            arrays[k].append(a[k])
        counts[name] = len(m)
    a = {k: np.concatenate(v) for k, v in arrays.items()}
    if not all(np.isfinite(v).all() for v in a.values()):
        raise FloatingPointError('nonfinite train cache')
    return a, counts


def make_head(state=None):
    import torch
    from torch import nn
    sd = torch.load(checkpoint(), map_location='cpu', weights_only=False)['state_dict'] if state is None else state
    w = sd['xyz.fc1.weight'].shape[0] if 'xyz.fc1.weight' in sd else sd['0.weight'].shape[0]
    head = nn.Sequential(nn.Linear(707, w), nn.ReLU(), nn.Linear(w, 51))
    if 'xyz.fc1.weight' in sd:
        head.load_state_dict({f'{i}.{k}': sd[f'xyz.fc{j}.{k}'] for i, j in [(0, 1), (2, 2)] for k in ['weight', 'bias']})
    else:
        head.load_state_dict(sd)
    return head


def masked(x, arm):
    cols = ARMS[arm]
    if cols:
        x = x.clone()
        x[:, cols] = 0.
    return x


def loss(pred, target):
    err = (pred - target).float()
    return ((450 * err[:, 6:].square().mean() + 60 * err[:, 3:6].square().mean()
             + 30000 * err[:, :3].square().mean()) / 51).log10()


def fit(job):
    import torch
    cpu()
    start = time.perf_counter()
    arm, seed, repeat = JOBS[job]
    a, counts = fit_data()
    torch.manual_seed(seed)
    head = make_head()
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-4)
    x = masked(torch.as_tensor(a['full707']), arm)
    initial = torch.as_tensor(a['incoming51'])
    target = torch.as_tensor(a['target51'])
    rng = torch.Generator(device='cpu')
    rng.manual_seed(seed)
    orders = hashlib.sha256()
    curve = []
    steps = 0
    for epoch in range(100):
        order = torch.randperm(len(x), generator=rng)
        orders.update(order.numpy().tobytes())
        losses = []
        for idx in order.split(512):
            optimizer.zero_grad(set_to_none=True)
            value = loss(initial[idx] + head(x[idx]), target[idx])
            if not torch.isfinite(value):
                raise FloatingPointError('nonfinite train loss')
            value.backward()
            optimizer.step()
            steps += 1
            losses.append(float(value.detach()))
        curve.append({'epoch': epoch + 1, 'mean_batch_log10_loss': float(np.mean(losses))})
        if (epoch + 1) % 20 == 0:
            print(json.dumps({'phase': 'fit', 'head': tag(job), 'epoch': epoch + 1, 'loss': curve[-1]['mean_batch_log10_loss'],
                              'seconds': time.perf_counter() - start}), flush=True)
    if not all(torch.isfinite(v).all() for v in head.state_dict().values()):
        raise FloatingPointError('nonfinite fitted head')
    path = OUT / 'heads' / f'{tag(job)}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'state_dict': {k: v.detach().clone() for k, v in head.state_dict().items()}, 'job': job, 'arm': arm,
                'seed': seed, 'repeat': repeat, 'protocol_sha256': protocol_sha()}, path)
    save(path.with_suffix('.json'), {'head': tag(job), 'arm': arm, 'masked_columns': ARMS[arm], 'seed': seed,
                                     'repeat': repeat, 'cache_counts': counts, 'training_rows': len(x), 'steps': steps,
                                     'order_sha256': orders.hexdigest(), 'curve': curve, 'head_sha256': sha(path),
                                     'protocol_sha256': protocol_sha(), 'source_sha256': sha(__file__),
                                     'wall_seconds': time.perf_counter() - start, 'threads': torch.get_num_threads()})


def j0_table():
    return read(ROOT / 'docs/assets/s37_next_controls_20261014/e2_domain_clean_audit.json')['sequence_J0_xyz_mm']


def heads():
    import torch
    result = {'q1': ('L0', make_head())}
    for job in range(len(JOBS)):
        saved = torch.load(OUT / 'heads' / f'{tag(job)}.pt', map_location='cpu', weights_only=False)
        assert saved['protocol_sha256'] == protocol_sha()
        result[tag(job)] = (saved['arm'], make_head(saved['state_dict']))
    for path in sorted((XDF / 'objective/heads').glob('z1_*.pt')):
        saved = torch.load(path, map_location='cpu', weights_only=False)
        result['xdf_gpu_' + path.stem] = ('L0', make_head(saved['state_dict']))
    for _, h in result.values():
        h.eval()
    return result


def predict(head, arm, x):
    import torch
    with torch.no_grad():
        return torch.cat([head(masked(xx, arm)) for xx in torch.as_tensor(x).split(512)]).numpy().astype(np.float64)


def stats(v):
    v = np.asarray(v, float)
    return {'mean': float(v.mean()), 'p05': float(np.percentile(v, 5)), 'p50': float(np.percentile(v, 50)),
            'p95': float(np.percentile(v, 95)), 'max': float(v.max()), 'min': float(v.min())}


def score(pred, initial, target, j0z, rot=None):
    out_t = initial[:, :3] + pred[:, :3]
    err = (out_t - target[:, :3]) * 1000.
    hold = np.linalg.norm((initial[:, :3] - target[:, :3]) * 1000., axis=1)
    wrist = np.linalg.norm(err, axis=1)
    depth = out_t[:, 2] * 1000. + j0z
    r = {'n': len(pred), 'finite': bool(np.isfinite(pred).all()), 'signed_xyz_mm': err.mean(0).tolist(),
         'wrist_mm': float(wrist.mean()), 'hold_wrist_mm': float(hold.mean()), 'better_than_hold': int((wrist < hold).sum()),
         'output_wrist_depth_mm': stats(depth), 'input_wrist_depth_mm': stats(initial[:, 2] * 1000. + j0z),
         'target_wrist_depth_mm': stats(target[:, 2] * 1000. + j0z),
         'output_above_native_max': float(np.mean(depth > NATIVE_MAX_MM))}
    if rot is not None:
        from tools.xdl import geometry as G
        mean, gr, gf = rot
        pr, pf = G.rotations((initial + pred).astype(np.float32), mean)
        r['root_deg'] = float(G.geodesic_deg(pr, gr).mean())
        r['finger_deg'] = float(G.geodesic_deg(pf, gf).mean(-1).mean())
    return r


def evaluate():
    from tools.xdl import geometry as G
    cpu()
    start = time.perf_counter()
    hs = heads()
    j0 = j0_table()
    result = {'protocol_sha256': protocol_sha(), 'source_sha256': sha(__file__), 'clean': {}, 'perturbed': {}, 'check': {}}
    raw = {}
    for seq in ('zgz_local', 'zgz_global'):
        z = np.load(XDF / 'objective/clean_cache' / f'{seq}.npz')
        initial, target = z['incoming51'].astype(np.float64), z['target51'].astype(np.float64)
        mean = z['hands_mean']
        gr, gf = G.rotations(z['target51'], mean)
        result['clean'][seq] = {}
        for name, (arm, h) in hs.items():
            pred = predict(h, arm, z['full707'])
            if name == 'q1':
                result['clean'][seq]['q1_replay_maxdiff'] = float(np.abs(initial + pred - z['prediction51']).max())
            result['clean'][seq][name] = score(pred, initial, target, j0[seq][2], (mean, gr, gf))
            raw[f'clean/{seq}/{name}'] = initial + pred
    a, meta = read_cache_dir(XDF / 'interface/cache', False)
    seqs = np.array([m['sequence'] for m in meta])
    split = np.array([m['split'] for m in meta])
    dz = np.array([float(m['dz_mm']) for m in meta])
    j0z = np.array([j0[s][2] for s in seqs])
    for name, (arm, h) in hs.items():
        pred = predict(h, arm, a['full707'])
        initial, target = a['incoming51'].astype(np.float64), a['target51'].astype(np.float64)
        depth_in = initial[:, 2] * 1000. + j0z
        depth_out = (initial[:, 2] + pred[:, 2]) * 1000. + j0z
        for seq in ('zgz_local', 'zgz_global'):
            m = seqs == seq
            slope = float(np.polyfit(depth_in[m], depth_out[m], 1)[0])
            per = {}
            for d in (-50., 0., 50.):
                k = m & (dz == d)
                per[f'{d:+.0f}'] = {'n': int(k.sum()), 'input_mean_mm': float(depth_in[k].mean()),
                                    'output_mean_mm': float(depth_out[k].mean()),
                                    'target_mean_mm': float((target[k, 2] * 1000. + j0z[k]).mean()),
                                    'wrist_mm': float(np.linalg.norm((initial[k, :3] + pred[k, :3] - target[k, :3]) * 1000., axis=1).mean())}
            result['perturbed'].setdefault(seq, {})[name] = {'output_vs_input_slope': slope, 'by_dz': per}
        for d in (0., -50., 50.):
            k = (split == 'check') & (dz == d)
            result['check'].setdefault(f'dz{d:+.0f}', {})[name] = score(pred[k], initial[k], target[k], j0z[k])
    np.savez_compressed(OUT / 'eval_clean_predictions.npz', **{k.replace('/', '__'): v for k, v in raw.items()})
    result['raw'] = ident(OUT / 'eval_clean_predictions.npz')
    result['wall_seconds'] = time.perf_counter() - start
    save(OUT / 'eval.json', result)
    print(json.dumps({'phase': 'evaluate', 'seconds': result['wall_seconds']}), flush=True)


def summarize():
    import torch
    e = read(OUT / 'eval.json')
    local = e['clean']['zgz_local']
    gates = []
    for seed in (1701, 1702):
        base, cand = local[f'L0_s{seed}_r0'], local[f'noT_s{seed}_r0']
        bias_b, bias_c = abs(base['signed_xyz_mm'][2]), abs(cand['signed_xyz_mm'][2])
        g1 = bias_c <= .5 * bias_b
        g2 = cand['output_wrist_depth_mm']['p95'] > NATIVE_MAX_MM
        gates.append({'seed': seed, 'L0_abs_bias_mm': bias_b, 'noT_abs_bias_mm': bias_c, 'ratio': bias_c / bias_b,
                      'noT_output_depth_p95_mm': cand['output_wrist_depth_mm']['p95'], 'G1_bias': g1, 'G2_unclamped': g2})
    a = torch.load(OUT / 'heads/noT_s1701_r0.pt', weights_only=False)['state_dict']
    b = torch.load(OUT / 'heads/noT_s1701_r1.pt', weights_only=False)['state_dict']
    det = max(float((a[k] - b[k]).abs().max()) for k in a)
    secondary = {}
    for seed in (1701, 1702):
        c = local[f'noTUV_s{seed}_r0']
        secondary[seed] = {'abs_bias_mm': abs(c['signed_xyz_mm'][2]), 'output_depth_p95_mm': c['output_wrist_depth_mm']['p95']}
    result = {'gate_by_seed': gates, 'primary_pass': all(g['G1_bias'] and g['G2_unclamped'] for g in gates),
              'determinism_max_abs_param_diff_noT_s1701': det, 'secondary_noTUV': secondary,
              'cpu_vs_gpu_L0_local_signed_z': {s: [local[f'L0_s{s}_r0']['signed_xyz_mm'][2],
                                                   local[f'xdf_gpu_z1_s{s}_r0']['signed_xyz_mm'][2]] for s in (1701, 1702)},
              'scope': 'frozen-feature final head refit only; not a deployment candidate; zgz is one subject',
              'protocol': ident(OUT / 'protocol.json'), 'source': ident(__file__)}
    receipts = [{'path': rel(p), 'seconds': read(p)['wall_seconds']} for p in sorted((OUT / 'heads').glob('*.json'))]
    result['cpu_seconds'] = {'fits': receipts, 'evaluate': e['wall_seconds']}
    save(OUT / 'result.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'fit', 'evaluate', 'summarize'])
    ap.add_argument('--job', type=int)
    args = ap.parse_args()
    try:
        if args.command == 'prepare':
            prepare()
        elif args.command == 'fit':
            fit(args.job)
        elif args.command == 'evaluate':
            evaluate()
        else:
            summarize()
    except Exception as error:
        save(OUT / 'failures' / f'{args.command}_j{args.job}.json',
             {'error': repr(error), 'traceback': traceback.format_exc(), 'arguments': vars(args)})
        raise
