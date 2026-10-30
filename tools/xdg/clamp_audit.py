#!/usr/bin/env python3
"""XDG zero-training audit: local output depth clamp versus the state-only MMSE denoiser prediction.

Reads only existing artifacts:
  outputs/xdf/depth/native_histogram.npz      native training target wrist depth (5 mm bins)
  outputs/xde/e1/<run>_clean_local.pt.gz      E1 GT-previous single-step outputs, all 1204 zgz_local packets
The denoiser is E[t | t_in] under that prior and the z marginal of the training previous-state noise
(semkine/dataset.py mixed mode: 50% N(0, 5 mm), 30% N(0, 50 mm), 20% directed |n| ~ U(0, 50 mm)).
It is a one-dimensional approximation: it ignores every other input and the self-burn states.
"""
from __future__ import annotations
import gzip
import io
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/xdg/clamp_audit.json'
RUNS = ['xdl_q1_s3407', 'xdl_q0_s3407', 'xdl_n1_s3407', 'xa_c0_s3407']
J0Z_ZGZ = 6.197364040361121


def denoiser():
    h = np.load(ROOT / 'outputs/xdf/depth/native_histogram.npz')
    e = h['edges_mm'].astype(float)
    c = (e[:-1] + e[1:]) / 2
    prior = h['target_counts'] / h['target_counts'].sum()
    rng = np.random.default_rng(0)
    n = 2_000_000
    u = rng.random(n)
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    noise = np.where(u < .5, rng.standard_normal(n) * 5,
                     np.where(u < .8, rng.standard_normal(n) * 50, v[:, 2] * rng.uniform(0, 50, n)))
    b = np.arange(-80, 80.5, 1.0)
    pdf, _ = np.histogram(noise, bins=b, density=True)
    bc = (b[:-1] + b[1:]) / 2

    def bias(t):
        w = prior * np.interp(t - c, bc, pdf, left=0, right=0)
        return (w * c).sum() / w.sum() - t
    return bias, {'prior_mean_mm': float((prior * c).sum())}


def main():
    import torch
    bias, prior = denoiser()
    result = {'prior': prior, 'denoiser_bias_mm': {str(t): bias(t) for t in range(560, 760, 10)}, 'runs': {}}
    for run in RUNS:
        path = ROOT / f'outputs/xde/e1/{run}_clean_local.pt.gz'
        with gzip.open(path, 'rb') as g:
            o = torch.load(io.BytesIO(g.read()), map_location='cpu', weights_only=False)
        tin = np.asarray(o['initial'], float)[:, 2] * 1000 + J0Z_ZGZ
        out = np.asarray(o['predictions'], float)[:, 2] * 1000 + J0Z_ZGZ
        gt = np.asarray(o['truth'], float)[:, 2] * 1000 + J0Z_ZGZ
        pred = np.array([bias(t) for t in tin])
        result['runs'][run] = {
            'n': len(tin), 'mean_signed_z_mm': float((out - gt).mean()), 'mean_update_z_mm': float((out - tin).mean()),
            'denoiser_predicted_update_mm': float(pred.mean()),
            'corr_update_vs_denoiser': float(np.corrcoef(out - tin, pred)[0, 1]),
            'output_depth_mm': {'p05': float(np.percentile(out, 5)), 'p50': float(np.percentile(out, 50)),
                                'p95': float(np.percentile(out, 95)), 'max': float(out.max())},
            'output_vs_input_slope': float(np.polyfit(tin, out, 1)[0])}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['runs'], indent=1))


if __name__ == '__main__':
    main()
