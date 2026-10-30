#!/usr/bin/env python3
"""XDI Phase 0: zero-training feasibility of the event-to-contour measurement innovation (semkine/xyz_innovation.py).

From the GT previous state of every zgz_local / zgz_global packet (and the 576 E2 check packets) the state is
perturbed by known amounts; one pure damped Gauss-Newton innovation step (gain 1, no learning, no network) is applied
with the packet's real events, sampled exactly as the S37 association samples them (`stride_sample`, 2048). Recovery is
scored against the window-end GT. A wrong-content control uses the events of a far packet of the same sequence.

  prepare     freeze outputs/xdi/feasibility/protocol.json (configuration and gates) before any real-data result
  run         one domain x shard (CPU), raw outputs kept
  summarize   pre-registered gates
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
OUT = ROOT / 'outputs/xdi/feasibility'
RUN = 'xdl_q1_s3407'
DOMAINS = ['zgz_local', 'zgz_global', 'check']
CONFIG = {'schedule_px': [16.0, 12.0, 8.0, 6.0, 4.0, 3.0, 3.0], 'sigma_c': 0.15, 'band_px': 24.0, 'lam': 0.1,
          'nu': 0.1, 'k': 8, 'orient_k': 8}
#: name -> (kind, value); translation in m, rotations in degrees (camera axes for the root, local x for a finger)
PERTURB = {'clean': ('none', 0)}
for _s in (1, -1):
    PERTURB[f'z{50 * _s:+d}'] = ('t', (2, .05 * _s))
    PERTURB[f'x{20 * _s:+d}'] = ('t', (0, .02 * _s))
    PERTURB[f'y{20 * _s:+d}'] = ('t', (1, .02 * _s))
    for _a in range(3):
        PERTURB[f'r{"xyz"[_a]}{10 * _s:+d}'] = ('root', (_a, 10 * _s))
        PERTURB[f'r{"xyz"[_a]}{20 * _s:+d}'] = ('root', (_a, 20 * _s))
    PERTURB[f'f{20 * _s:+d}'] = ('finger', 20 * _s)
WRONG = ['z+50', 'z-50']


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
    p = {'study': 'XDI-phase0-contour-innovation-feasibility-20261014', 'created_unix': time.time(),
         'frozen_before_any_real_data_result': True,
         'configuration': CONFIG, 'dof_primary': 51, 'dof_secondary': 6,
         'configuration_selection': 'chosen on synthetic events only (GT visible-contour samples, 0.7 px noise, 0/7%/20% '
                                    'uniform hand-box outliers); no real-data outcome was looked at',
         'real_event_contour_share_seen_before_freeze': 'share of real events within 2/4 px of the GT contour '
                                                       '(local 63%/90%, global 35%/67%); a data property, not a gate outcome',
         'disclosure': 'before this freeze a 3-packet-per-sequence debugging run (rows 200-202) of this tool was seen: '
                       'local clean drift about 50 mm and a robust cost lower at a state 25-47 mm farther than at the '
                       'GT; the configuration was NOT changed after it, the gates below are unchanged from the plan',
         'perturbations': {k: list(v) if isinstance(v, tuple) else v for k, v in PERTURB.items()},
         'root_rotation': 'left-multiplied in the camera frame, translation compensated so the FK wrist stays put',
         'finger': 'joint (packet index mod 15), local x axis, right-multiplied on the full local rotation',
         'wrong_content': 'events of row (i + n/2) mod n of the same sequence, perturbations ' + ', '.join(WRONG),
         'events': 'stride_sample(ptr, 2048) of the raw W50 packet, render-space rays as in _xyz_mesh_forward',
         'scoring': 'against the window-end GT; translation recovery = 1 - |t_out - t_gt| / |t_in - t_gt| (wrist '
                    'vector); root recovery = 1 - geo(R_out, R_gt) / geo(R_in, R_gt); medians over packets, empty '
                    'packets included (they hold)',
         'gates_dof51': {
             'P0a_clean_safety': 'per zgz sequence: mean wrist error after the step minus the hold error <= 3 mm and '
                                 'median |delta t| <= 5 mm',
             'P0b_depth': 'per zgz sequence and sign: median recovery of z +/-50 mm >= 0.5',
             'P0c_root': 'per zgz sequence: median over the six +/-10 deg root perturbations of the per-packet '
                         'recovery >= 0.5',
             'P0d_events_needed': 'per zgz sequence: median recovery of z +/-50 mm with wrong-content events < 0.2',
             'stop_rule': 'P0b or P0c failing stops XDI before any training'},
         'descriptive': ['x/y +/-20 mm', 'root +/-20 deg', 'finger +/-20 deg', 'check domain', 'dof 6', 'inlier share',
                         'CPU seconds'],
         'model_use': f'{RUN} only for MANO FK, intrinsics and hand context; no network output is used',
         'inputs': [ident(ROOT / 'semkine/xyz_innovation.py'), ident(ROOT / 'semkine/xyz_mesh.py'),
                    ident(ROOT / 'outputs/xdl/packet_bank.json'), ident(ROOT / 'outputs/xde/e2/protocol.json')],
         'executed_source': ident(__file__)}
    save(path, p)
    print(json.dumps({'protocol': rel(path)}), flush=True)


def packets(domain, bank):
    if domain in ('zgz_local', 'zgz_global'):
        rows = bank['sequences'][domain]['rows']
        return [(domain, i, row) for i, row in enumerate(rows)]
    e2 = read(ROOT / 'outputs/xde/e2/protocol.json')['selected_packets']
    return [(r['sequence'], int(r['row']['index']), r['row']) for r in e2 if r['split'] == 'check']


def run(domain, shard, shards):
    import torch
    from tools.xdl import diagnostics as D
    from semkine.events import EV_X, EV_Y
    from semkine.lie import so3_exp, so3_log
    from semkine.xyz_innovation import innovation_step
    from semkine.xyz_mesh import stride_sample
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        raise RuntimeError('Phase 0 is CPU only')
    torch.set_num_threads(int(os.environ.get('XDI_THREADS', '8')))
    start = time.perf_counter()
    dev = torch.device('cpu')
    model, cfg, identity = D.load_model(RUN, dev)
    fk = lambda p, b: model._fk(p, b)[0]
    wrist = lambda p, b: model._fk(p, b)[1][:, 0]
    faces = model.mano.f
    mean = model.mano.hands_mean.detach().float().reshape(15, 3)
    bank, bank_sha = D.load_bank()
    items = packets(domain, bank)[shard::shards]
    names = list(PERTURB)
    cache = {}
    rec = {k: [] for k in ('x_in51', 'x_out51', 'x_out6_51', 'gt_prev51', 'gt_next51', 'inlier', 'wrong_in51',
                           'wrong_out51', 'n_events')}
    meta = []
    for seq, index, row in items:
        if seq not in cache:
            data = D.load_data(cfg, seq, bank['sequences'][seq])
            beta, cam = D.context(data, dev)
            cache = {seq: (data, beta, cam)}
        data, beta, cam = cache[seq]
        model.set_hand_context(beta, cam)
        fx, fy, cx, cy = model._intrinsics(cam)
        fpx = (fx * fy).sqrt()
        gt_prev = torch.tensor(np.asarray(data['gt'][row['previous_end_ms']], np.float32))[None]
        gt_next = torch.tensor(np.asarray(data['gt'][row['end_ms']], np.float32))[None]

        def rays(r):
            ev = D.raw_events(data, r)
            pk = D.make_packet(ev, gt_prev.numpy(), beta, cam, dev, r)
            if pk.events.shape[0] == 0:
                return torch.zeros(1, 1, 2), torch.zeros(1, 1, dtype=torch.bool), 0
            src, mask = stride_sample(pk.ptr, model.xyz_max_nodes)
            e = pk.events[src.reshape(-1)].view(1, -1, pk.events.shape[1])
            p = torch.stack([(e[..., EV_X].float() - cx[:, None]) / fx[:, None],
                             (e[..., EV_Y].float() - cy[:, None]) / fy[:, None]], -1)
            return p, mask, len(ev)

        states = []
        for name in names:
            kind, val = PERTURB[name]
            s = gt_prev.clone()
            if kind == 't':
                s[0, val[0]] += val[1]
            elif kind == 'root':
                w = torch.zeros(3)
                w[val[0]] = float(np.deg2rad(val[1]))
                w0 = wrist(s, beta)
                s[0, 3:6] = so3_log(so3_exp(w)[None] @ so3_exp(s[:, 3:6]))[0]
                s[0, :3] += (w0 - wrist(s, beta))[0]
            elif kind == 'finger':
                j = index % 15
                full = s[0, 6 + 3 * j:9 + 3 * j] + mean[j]
                w = torch.tensor([np.deg2rad(val), 0., 0.], dtype=torch.float32)
                s[0, 6 + 3 * j:9 + 3 * j] = so3_log(so3_exp(full)[None] @ so3_exp(w)[None])[0] - mean[j]
            states.append(s)
        X = torch.cat(states)
        B = X.shape[0]
        p, mask, n_ev = rays(row)
        P, M = p.expand(B, -1, -1).contiguous(), mask.expand(B, -1).contiguous()
        out51 = innovation_step(fk, X, beta.expand(B, -1), faces, P, M, fpx.expand(B), dof=51, **CONFIG)
        out6 = innovation_step(fk, X, beta.expand(B, -1), faces, P, M, fpx.expand(B), dof=6, **CONFIG)
        rows_all = bank['sequences'][seq]['rows']
        donor = rows_all[(index + len(rows_all) // 2) % len(rows_all)]
        pw, mw, _ = rays(donor)
        Xw = torch.cat([states[names.index(n)] for n in WRONG])
        ow = innovation_step(fk, Xw, beta.expand(len(WRONG), -1), faces, pw.expand(len(WRONG), -1, -1).contiguous(),
                             mw.expand(len(WRONG), -1).contiguous(), fpx.expand(len(WRONG)), dof=51, **CONFIG)
        rec['x_in51'].append(X.numpy())
        rec['x_out51'].append((X + out51['delta']).numpy())
        rec['x_out6_51'].append((X + out6['delta']).numpy())
        rec['gt_prev51'].append(gt_prev.numpy()[0])
        rec['gt_next51'].append(gt_next.numpy()[0])
        rec['inlier'].append(out51['inlier'].numpy())
        rec['wrong_in51'].append(Xw.numpy())
        rec['wrong_out51'].append((Xw + ow['delta']).numpy())
        rec['n_events'].append(n_ev)
        meta.append({'sequence': seq, 'index': index, 'end_ms': row['end_ms'], 'donor_end_ms': donor['end_ms']})
        if len(meta) % 25 == 0:
            print(json.dumps({'domain': domain, 'shard': shard, 'done': len(meta), 'of': len(items),
                              'seconds': time.perf_counter() - start}), flush=True)
    path = OUT / 'raw' / f'{domain}__{shard:02d}of{shards:02d}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: np.stack(v) for k, v in rec.items()}, names=np.array(names))
    save(path.with_suffix('.json'), {'rows': meta, 'raw': ident(path), 'identity': identity, 'packet_bank_sha256': bank_sha,
                                     'wall_seconds': time.perf_counter() - start, 'threads': torch.get_num_threads(),
                                     'protocol_sha256': sha(OUT / 'protocol.json'), 'source_sha256': sha(__file__)})


def summarize():
    import torch
    from semkine.lie import so3_exp, so3_log
    names = None
    res = {'protocol': ident(OUT / 'protocol.json'), 'source': ident(__file__), 'domains': {}, 'gates': {}}
    for domain in DOMAINS:
        files = sorted((OUT / 'raw').glob(f'{domain}__*.npz'))
        if not files:
            continue
        z = [np.load(f) for f in files]
        names = list(z[0]['names'])
        cat = {k: np.concatenate([q[k] for q in z]) for k in ('x_in51', 'x_out51', 'x_out6_51', 'gt_prev51', 'gt_next51',
                                                               'inlier', 'wrong_in51', 'wrong_out51', 'n_events')}
        gt = cat['gt_next51'][:, None, :]

        def tr(x):
            return np.linalg.norm((x[..., :3] - gt[..., :3]) * 1000., axis=-1)

        def geo(x):
            a = torch.tensor(x[..., 3:6].reshape(-1, 3), dtype=torch.float64)
            b = torch.tensor(np.broadcast_to(gt[..., 3:6], x[..., 3:6].shape).reshape(-1, 3), dtype=torch.float64)
            return (so3_log(so3_exp(a) @ so3_exp(b).transpose(1, 2)).norm(dim=-1).numpy() * 180 / np.pi).reshape(x.shape[:-1])

        d = {'n_packets': int(len(cat['n_events'])), 'empty_packets': int((cat['n_events'] == 0).sum()),
             'inlier_median': float(np.median(cat['inlier'][:, 0])), 'dof': {}}
        hold = np.linalg.norm((cat['gt_prev51'][:, :3] - cat['gt_next51'][:, :3]) * 1000., axis=-1)
        for dof, key in ((51, 'x_out51'), (6, 'x_out6_51')):
            xin, xout = cat['x_in51'], cat[key]
            tin, tout, gin, gout = tr(xin), tr(xout), geo(xin), geo(xout)
            r = {}
            c = names.index('clean')
            r['clean'] = {'hold_wrist_mm': float(hold.mean()), 'after_wrist_mm': float(tout[:, c].mean()),
                          'increase_mm': float(tout[:, c].mean() - hold.mean()),
                          'median_abs_dt_mm': float(np.median(np.linalg.norm((xout[:, c, :3] - xin[:, c, :3]) * 1000., axis=-1))),
                          'root_deg_before': float(gin[:, c].mean()), 'root_deg_after': float(gout[:, c].mean())}
            for i, n in enumerate(names):
                if n == 'clean':
                    continue
                kind = n[0]
                if kind in 'xyz':
                    rec = 1 - tout[:, i] / np.maximum(tin[:, i], 1e-9)
                elif kind == 'r':
                    rec = 1 - gout[:, i] / np.maximum(gin[:, i], 1e-9)
                else:
                    rec = None
                entry = {'wrist_mm_before': float(tin[:, i].mean()), 'wrist_mm_after': float(tout[:, i].mean()),
                         'root_deg_before': float(gin[:, i].mean()), 'root_deg_after': float(gout[:, i].mean())}
                if rec is not None:
                    entry['median_recovery'] = float(np.median(rec))
                    entry['per_packet_recovery'] = rec
                r[n] = entry
            r10 = np.median(np.stack([r[n]['per_packet_recovery'] for n in names if n[0] == 'r' and n.endswith('10')]), axis=0)
            r['root10_median_recovery'] = float(np.median(r10))
            for n in names:
                if n in r and isinstance(r[n], dict):
                    r[n].pop('per_packet_recovery', None)
            d['dof'][str(dof)] = r
        tw_in, tw_out = tr(cat['wrong_in51']), tr(cat['wrong_out51'])
        d['wrong_content'] = {n: float(np.median(1 - tw_out[:, i] / np.maximum(tw_in[:, i], 1e-9))) for i, n in enumerate(WRONG)}
        res['domains'][domain] = d
    for seq in ('zgz_local', 'zgz_global'):
        if seq not in res['domains']:
            continue
        r = res['domains'][seq]['dof']['51']
        g = {'P0a': r['clean']['increase_mm'] <= 3.0 and r['clean']['median_abs_dt_mm'] <= 5.0,
             'P0b': r['z+50']['median_recovery'] >= 0.5 and r['z-50']['median_recovery'] >= 0.5,
             'P0c': r['root10_median_recovery'] >= 0.5,
             'P0d': all(v < 0.2 for v in res['domains'][seq]['wrong_content'].values())}
        res['gates'][seq] = g
    res['stop'] = any(not (g['P0b'] and g['P0c']) for g in res['gates'].values())
    save(OUT / 'result.json', res)
    print(json.dumps({'gates': res['gates'], 'stop': res['stop']}, indent=1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['prepare', 'run', 'summarize'])
    ap.add_argument('--domain', choices=DOMAINS)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    args = ap.parse_args()
    try:
        if args.command == 'prepare':
            prepare()
        elif args.command == 'run':
            run(args.domain, args.shard, args.shards)
        else:
            summarize()
    except Exception as error:
        save(OUT / 'failures' / f'{args.command}_{args.domain}_{args.shard}.json',
             {'error': repr(error), 'traceback': traceback.format_exc(), 'arguments': vars(args)})
        raise
