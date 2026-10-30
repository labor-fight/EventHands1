#!/usr/bin/env python3
"""XRF debug evaluation of one checkpoint (CPU): prev-free single-step measurement quality, gain calibration on held-out
training-identity blocks, zgz closed loop, and recovery from perturbed starts.

  python tools/xrf/debug_eval.py --run xrf_dbg_a            (latest *-step=N.ckpt in outputs/semkine/<run>)

Writes outputs/xrf/debug/<run>/{single_step,calibration,closed_loop,recovery,summary}.json. The calibration uses only
the 576 E2 `check` packets (training identities, held-out time blocks); zgz is never used to set a gain.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]
OUT = REPO / 'outputs/xrf/debug'


def latest_ckpt(run):
    d = REPO / 'outputs/semkine' / run
    c = [(int(re.search(r'step=(\d+)', p.name).group(1)), p) for p in d.glob('*-step=*.ckpt')]
    if not c:
        raise FileNotFoundError(f'no checkpoint in {d}')
    return max(c)[1]


def load(run, ckpt=None):
    from config import load_config
    from semkine.xrf import XRFModel
    ckpt = Path(ckpt) if ckpt else latest_ckpt(run)
    cfg = load_config(str(REPO / 'outputs/semkine' / run / 'config_resolved.yaml'))
    m = XRFModel.load_from_checkpoint(str(ckpt), cfg=cfg, map_location='cpu', strict=True).float().eval()
    from mano_layer import ManoLayer
    for mod in m.modules():
        if isinstance(mod, ManoLayer):
            mod.fast_chain = True
    return m, cfg, ckpt


def rot_err_deg(a, b):
    from semkine.lie import so3_exp, so3_log
    return (so3_log(so3_exp(torch.as_tensor(b, dtype=torch.float64)).transpose(-1, -2)
                    @ so3_exp(torch.as_tensor(a, dtype=torch.float64))).norm(dim=-1).numpy() * 180 / np.pi)


def packets(domain, bank):
    if domain in ('zgz_local', 'zgz_global'):
        return [(domain, row) for row in bank['sequences'][domain]['rows']]
    e2 = json.loads((REPO / 'outputs/xde/e2/protocol.json').read_text())['selected_packets']
    return [(r['sequence'], r['row']) for r in e2 if r['split'] == 'check']


class Seq:
    def __init__(self, m, cfg, bank, seq):
        from tools.xdl import diagnostics as D
        self.D = D
        self.data = D.load_data(cfg, seq, bank['sequences'][seq])
        self.beta, self.cam = D.context(self.data, torch.device('cpu'))
        self.runs = np.asarray(bank['sequences'][seq]['valid_runs_ms'], np.int64).reshape(-1, 2)
        self.seq = seq

    def packet(self, m, row, state, window_ms=200):
        from semkine import eval_track as ET
        end = int(row['end_ms'])
        a = [int(r0) for r0, r1 in self.runs if r0 <= end < r1][0]
        w = min(window_ms, end - a + 1)
        ev = self.D.raw_events(self.data, row)
        pk = self.D.make_packet(ev, np.asarray(state, np.float32)[None], self.beta, self.cam, torch.device('cpu'), row)
        img = ET.build_lnes(self.data['events'], self.data['offsets'], end, w, ('last',))
        pk.lnes = torch.from_numpy(img)[None]
        pk.lnes_window_ms = torch.tensor([w])
        return pk


@torch.no_grad()
def single_step(m, cfg, bank):
    res, raw = {}, {}
    for domain in ('check', 'zgz_local', 'zgz_global'):
        items = packets(domain, bank)
        cache = {}
        zd, zr, gt, kp_err = [], [], [], []
        for seq, row in items:
            if seq not in cache:
                cache = {seq: Seq(m, cfg, bank, seq)}
            S = cache[seq]
            m.set_hand_context(S.beta, S.cam)
            y = np.asarray(S.data['gt'][row['end_ms']], np.float32)
            pk = S.packet(m, row, y)                       # prev is irrelevant to the measurement chain
            d, aux = m.direct_measurement(pk)
            r = m.refine(pk, d)
            _, jg = m._fk(torch.from_numpy(y)[None], S.beta)
            fx, fy, cx, cy = m._intrinsics(S.cam)
            g = torch.stack([fx * jg[..., 0] / jg[..., 2] + cx, fy * jg[..., 1] / jg[..., 2] + cy], -1)[0]
            inside = (g[:, 0] >= 0) & (g[:, 0] < 240) & (g[:, 1] >= 0) & (g[:, 1] < 180)
            e = (aux['kp_px'][0] - g).norm(dim=-1)
            kp_err.append(float(e[inside].mean()) if bool(inside.any()) else np.nan)
            zd.append(d[0].numpy()); zr.append(r[0].numpy()); gt.append(y)
        zd, zr, gt = np.stack(zd).astype(np.float64), np.stack(zr).astype(np.float64), np.stack(gt).astype(np.float64)
        raw[domain] = {'zd': zd, 'zr': zr, 'gt': gt}

        def stats(x):
            dt = (x[:, :3] - gt[:, :3]) * 1000
            fin = np.degrees(np.linalg.norm((x[:, 6:] - gt[:, 6:]).reshape(-1, 15, 3), axis=-1).mean(-1))
            slope = float(np.polyfit(gt[:, 2], x[:, 2], 1)[0]) if np.ptp(gt[:, 2]) > 0.02 else None
            return {'wrist_mm': float(np.linalg.norm(dt, axis=1).mean()), 'signed_z_mm': float(dt[:, 2].mean()),
                    'xy_mm': float(np.linalg.norm(dt[:, :2], axis=1).mean()), 'depth_slope': slope,
                    'root_deg': float(rot_err_deg(x[:, 3:6], gt[:, 3:6]).mean()), 'finger_deg': float(fin.mean())}
        res[domain] = {'n': len(items), 'D': stats(zd), 'R': stats(zr), 'D_kp_px': float(np.nanmean(kp_err))}
    allz = np.concatenate([raw[d]['gt'][:, 2] for d in ('zgz_local', 'zgz_global')])
    for key in ('zd', 'zr'):
        allp = np.concatenate([raw[d][key][:, 2] for d in ('zgz_local', 'zgz_global')])
        res[f'zgz_depth_slope_{key}'] = float(np.polyfit(allz, allp, 1)[0])
    return res, raw


def calibrate(m, cfg, bank, raw):
    """Q: per-group variance of the 50 ms GT step on the check packets; R: per-group residual variance of z_R / z_D on
    the same packets. Steady-state random-walk Kalman gain per group."""
    from semkine.xkf import group_moments, group_residuals
    from semkine.xrf import steady_state_gain
    items = packets('check', bank)
    prevs = []
    cache = {}
    for seq, row in items:
        if seq not in cache:
            from tools.xdl import diagnostics as D
            cache = {seq: D.load_data(cfg, seq, bank['sequences'][seq])}
        prevs.append(np.asarray(cache[seq]['gt'][row['end_ms'] - 50], np.float32))
    gt = torch.from_numpy(raw['check']['gt']).float()
    prev = torch.from_numpy(np.stack(prevs)).float()
    eq = group_residuals(gt, prev)
    Q = group_moments(eq, eq)
    out = {}
    for key, name in (('zr', 'R'), ('zd', 'D')):
        e = group_residuals(torch.from_numpy(raw['check'][key]).float(), gt)
        Rv = group_moments(e, e)
        K = steady_state_gain(Q, Rv)
        out[name] = {'R': Rv.tolist(), 'K': K.tolist()}
    out['Q'] = Q.tolist()
    m.xrf_gains.copy_(torch.tensor(out['R']['K']))
    m.xrf_gains_d.copy_(torch.tensor(out['D']['K']))
    return out


@torch.no_grad()
def closed_loop(m, cfg, modes=('fused', 'r', 'd_fused')):
    from mano_layer import ManoLayer
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.xkf.runtime import LongWindowEval
    from tools.xkf.make_row import runs_of
    root = Path(cfg['DATA']['ROOT'])
    mano = ManoLayer(cfg['MANO']['NPZ'], add_mean=False).eval()
    res = {}
    for mode in modes:
        m.xrf_kmode = mode
        rng = np.random.default_rng(0)
        res[mode] = {}
        for seq, d in sequences_for_split(root, 'val_core', splits_manifest(cfg)):
            with LongWindowEval(runs_of(root, d, seq), 200):
                r = ET.track_sequence(m, mano, cfg, root, d, seq, 50, torch.device('cpu'), rng, 1., 1., None, window_ms=50)
            res[mode][seq] = {k: r[k] for k in ('mpjpe_ra_mm', 'mpjpe_abs_mm', 'mpvpe_ra_mm', 'jitter_all_mm_per_step',
                                                'gt_move_mm_per_step', 'n_frames')}
        n = {s: v['n_frames'] for s, v in res[mode].items()}
        res[mode]['overall_ra_mm'] = sum(res[mode][s]['mpjpe_ra_mm'] * n[s] for s in n) / sum(n.values())
    m.xrf_kmode = 'fused'
    return res


@torch.no_grad()
def recovery(m, cfg, bank, steps=10):
    """Starts: every zgz_local run start and 24 evenly spaced zgz_global starts; GT start and 6 perturbations
    (z +/-50 mm, root +/-20 deg about camera x and y). Recovered: within `steps`, wrist <= 2x + 1 mm and root <= 2x +
    1 deg of the same clip from GT, for 3 consecutive steps."""
    from semkine.lie import so3_exp, so3_log
    m.xrf_kmode = 'fused'
    perts = {'z+50': ('t', 2, .05), 'z-50': ('t', 2, -.05)}
    for ax in (0, 1):
        for sg in (1, -1):
            perts[f'r{"xy"[ax]}{20 * sg:+d}'] = ('r', ax, np.radians(20 * sg))
    out = {'trials': 0, 'recovered': 0, 'by_pert': {}}
    for seq in ('zgz_local', 'zgz_global'):
        S = Seq(m, cfg, bank, seq)
        m.set_hand_context(S.beta, S.cam)
        rows = bank['sequences'][seq]['rows']
        by_run = {}
        for i, row in enumerate(rows):
            by_run.setdefault(row['run_id'], []).append(i)
        starts = [v[0] for v in by_run.values() if len(v) > steps] if seq == 'zgz_local' else \
            list(range(0, len(rows) - steps - 1, max(1, (len(rows) - steps - 1) // 24)))[:24]
        for s in starts:
            clip = rows[s:s + steps]
            x0 = np.asarray(S.data['gt'][clip[0]['previous_end_ms']], np.float32)

            def roll(x):
                errs = []
                for row in clip:
                    pk = S.packet(m, row, x)
                    x = m.forward_packet(pk)[0].numpy()
                    y = np.asarray(S.data['gt'][row['end_ms']], np.float32)
                    errs.append((np.linalg.norm((x[:3] - y[:3]) * 1000), float(rot_err_deg(x[None, 3:6], y[None, 3:6])[0])))
                return np.array(errs)
            ref = roll(x0.copy())
            for name, (kind, ax, val) in perts.items():
                x = x0.copy()
                if kind == 't':
                    x[ax] += val
                else:
                    w = torch.zeros(3); w[ax] = float(val)
                    x[3:6] = so3_log(so3_exp(w) @ so3_exp(torch.from_numpy(x[3:6]).float())).numpy()
                e = roll(x)
                ok = (e[:, 0] <= 2 * ref[:, 0] + 1) & (e[:, 1] <= 2 * ref[:, 1] + 1)
                rec = any(ok[i:i + 3].all() for i in range(len(ok) - 2))
                out['trials'] += 1
                out['recovered'] += int(rec)
                b = out['by_pert'].setdefault(f'{seq}/{name}', [0, 0])
                b[0] += int(rec); b[1] += 1
    out['rate'] = out['recovered'] / max(1, out['trials'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--ckpt')
    ap.add_argument('--skip', default='')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES', '') != '':
        raise SystemExit('CPU only: set CUDA_VISIBLE_DEVICES=""')
    torch.set_num_threads(int(os.environ.get('XRF_THREADS', '32')))
    from tools.xdl import diagnostics as D
    t0 = time.perf_counter()
    m, cfg, ckpt = load(args.run, args.ckpt)
    bank, _ = D.load_bank()
    out = OUT / args.run
    out.mkdir(parents=True, exist_ok=True)
    summary = {'run': args.run, 'checkpoint': str(ckpt), 'step': int(re.search(r'step=(\d+)', ckpt.name).group(1))}
    ss, raw = single_step(m, cfg, bank)
    (out / 'single_step.json').write_text(json.dumps(ss, indent=1))
    summary['single_step'] = ss
    cal = calibrate(m, cfg, bank, raw)
    (out / 'calibration.json').write_text(json.dumps(cal, indent=1))
    summary['gains_R'] = {'t': cal['R']['K'][:3], 'root': cal['R']['K'][3], 'finger_mean': float(np.mean(cal['R']['K'][4:]))}
    if 'loop' not in args.skip:
        cl = closed_loop(m, cfg)
        (out / 'closed_loop.json').write_text(json.dumps(cl, indent=1))
        summary['closed_loop'] = cl
    if 'recovery' not in args.skip:
        rc = recovery(m, cfg, bank)
        (out / 'recovery.json').write_text(json.dumps(rc, indent=1))
        summary['recovery'] = rc
    summary['wall_seconds'] = time.perf_counter() - t0
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
