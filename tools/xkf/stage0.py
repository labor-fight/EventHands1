#!/usr/bin/env python3
"""XKF Stage 0 (CPU, before training): F1 DGP depth from noisy 2D keypoints at real hand states; F2 the existing XYZ
motion branches against the zero-motion predictor on training-identity check packets; F3 the implied fusion."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]
OUT = REPO / 'outputs/xkf/stage0'


def f1():
    from tools.xdl import diagnostics as D
    from semkine.xkf_direct import dgp_translation
    torch.set_num_threads(32)
    dev = torch.device('cpu')
    model, cfg, _ = D.load_model('xdl_q1_s3407', dev)
    bank, _ = D.load_bank()
    res = {}
    for seq in ('zgz_local', 'zgz_global', 'lyq_local', 'ch_global'):
        data = D.load_data(cfg, seq, bank['sequences'][seq])
        beta, cam = D.context(data, dev)
        rows = bank['sequences'][seq]['rows'][::7][:150]
        y = torch.tensor(np.stack([data['gt'][r['end_ms']] for r in rows]), dtype=torch.float32)
        B = len(rows)
        _, J = model._fk(y, beta.expand(B, -1))
        y0 = y.clone(); y0[:, :3] = 0
        _, J0 = model._fk(y0, beta.expand(B, -1))
        fx, fy, cx, cy = model._intrinsics(cam)
        px = torch.stack([fx * J[..., 0] / J[..., 2] + cx, fy * J[..., 1] / J[..., 2] + cy], -1)
        g = torch.Generator().manual_seed(0)
        out = {'wrist_depth_mm': [float((y[:, 2] * 1000).min()), float((y[:, 2] * 1000).max())]}
        for noise in (0.0, 1.0, 2.0, 3.0):
            for shrink in (0.0, 0.03):
                c = px.mean(1, keepdim=True)
                q = c + (px - c) * (1 - shrink) + noise * torch.randn(px.shape, generator=g)
                uv = torch.stack([(q[..., 0] - cx) / fx, (q[..., 1] - cy) / fy], -1)
                t = dgp_translation(J0, uv, torch.ones(B, 21))
                dz = (t[:, 2] - y[:, 2]) * 1000
                out[f'noise{noise:g}px_shrink{shrink:g}'] = {'z_rmse_mm': float(dz.square().mean().sqrt()),
                                                          'z_bias_mm': float(dz.mean()),
                                                          'xy_rmse_mm': float(((t[:, :2] - y[:, :2]) * 1000).norm(dim=-1).square().mean().sqrt())}
        res[seq] = out
    return res


def f2():
    from semkine.lie import so3_exp, so3_log
    res = {}
    for run in ('xdg_q1t0_s3407', 'xdg_q1dz_s3407', 'xdl_q1_s3407'):
        z = np.load(REPO / f'outputs/xdg/retrain/eval/{run}__check.npz')
        init, truth, pred = z['initial'].astype(np.float64), z['truth'].astype(np.float64), z['prediction'][:, 0].astype(np.float64)

        def blocks(x):
            t = np.linalg.norm((x[:, :3] - truth[:, :3]) * 1000, axis=1)
            r = so3_log(so3_exp(torch.tensor(truth[:, 3:6])).transpose(1, 2) @ so3_exp(torch.tensor(x[:, 3:6]))).norm(dim=-1).numpy()
            f = np.linalg.norm((x[:, 6:] - truth[:, 6:]).reshape(-1, 15, 3), axis=-1).mean(-1)
            return {'wrist_rms_mm': float(np.sqrt((t ** 2).mean())), 'root_rms_deg': float(np.degrees(np.sqrt((r ** 2).mean()))),
                    'finger_rms_rad': float(np.sqrt((f ** 2).mean()))}
        res[run] = {'motion_branch_from_gt_prev': blocks(pred), 'zero_motion_hold': blocks(init)}
    return res


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    r = {'F1_dgp_geometry': f1(), 'F2_motion_vs_hold_check': f2(),
         'F3_note': 'with a direct measurement of root error sigma_D and a motion branch sigma_I, the minimum-variance '
                    'fusion error is sigma_I sigma_D / sqrt(sigma_I^2 + sigma_D^2) for uncorrelated errors; if the motion '
                    'branch is worse than holding, the fusion reduces to the direct measurement plus a zero-motion smoother'}
    (OUT / 'result.json').write_text(json.dumps(r, indent=1) + '\n')
    print(json.dumps(r, indent=1))
