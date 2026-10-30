#!/usr/bin/env python3
"""Render retained local recovery diagnostics; no network calls."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'outputs/xkf/local_recovery_20261015'
OUT = ROOT / 'docs/assets/xkf_local_recovery_20261015'
SEEDS = (3407, 3408, 3409)
COLORS = ('#2479b2', '#e48825', '#30926d')


def main():
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    variants = ('predpose_pred2d', 'GTpose_pred2d', 'predpose_GT2d', 'GTpose_GT2d')
    for i, (seed, color) in enumerate(zip(SEEDS, COLORS)):
        data = np.load(DATA / f'direct_s{seed}/zgz_local_W200.npz')
        rmse = [np.sqrt(np.mean(np.sum(data[f'learned__{v}__t_error_mm'] ** 2, axis=1))) for v in variants]
        axs[0, 0].bar(np.arange(4) + (i - 1) * .24, rmse, width=.23, color=color, label=str(seed))
        vals = []
        for win in (50, 100, 150, 200):
            a = np.load(DATA / f'direct_s{seed}/zgz_local_W{win}.npz')
            vals.append(np.sqrt(np.mean(np.sum(a['learned__predpose_pred2d__t_error_mm'] ** 2, axis=1))))
        axs[0, 1].plot((50, 100, 150, 200), vals, 'o-', color=color, label=str(seed))
    axs[0, 0].set(xticks=np.arange(4), xticklabels=['D prediction', 'GT pose only', 'GT 2D only', 'GT both'],
                  ylabel='Translation RMSE (mm)', title='A  DGP interventions: all 1,204 local packets, W200')
    axs[0, 0].legend(title='Debug seed', frameon=False)
    axs[0, 1].set(xlabel='Causal LNES window cap (ms)', ylabel='Translation RMSE (mm)',
                  title='B  Same packet endpoints, naturally clipped at run start')
    scored = json.loads((DATA / 'scoring/score_audit.json').read_text())['saved_trace_replay']['results']
    raw = np.load(DATA / 'scoring/scored_states.npz')
    x = np.arange(1, 11)
    for seed, color in zip(SEEDS, COLORS):
        modes = scored[str(seed)]['zgz_local']['modes']
        axs[1, 0].plot(x, modes['fused']['per_packet']['abs_mpjpe_mm'], 'o-', color=color, label=f'Fused {seed}')
        err = (raw[f's{seed}_zgz_local_fused'][:, 2] - raw[f's{seed}_zgz_local_truth'][:, 2]) * 1000
        axs[1, 1].plot(x, err, 'o-', color=color, label=f'Fused {seed}')
    axs[1, 0].plot(x, scored['3407']['zgz_local']['modes']['hold']['per_packet']['abs_mpjpe_mm'], '--', color='#444', label='Hold')
    err = (raw['s3407_zgz_local_hold'][:, 2] - raw['s3407_zgz_local_truth'][:, 2]) * 1000
    axs[1, 1].plot(x, err, '--', color='#444', label='Hold')
    axs[1, 1].axhline(0, color='#999', lw=.8)
    axs[1, 0].set(xlabel='Update step (50 ms each)', ylabel='Absolute MPJPE (mm)',
                  title='C  Original failing clip: combined +50 mm / +20 deg')
    axs[1, 1].set(xlabel='Update step (50 ms each)', ylabel='Signed depth error (mm)',
                  title='D  Same clip: depth overshoots in two seeds')
    axs[1, 0].legend(frameon=False)
    for ax in axs.flat:
        ax.grid(axis='y', alpha=.2)
    fig.suptitle('XKF frozen debug step 2000: diagnosis only, no full-training claim', fontsize=14)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'diagnosis.{ext}', dpi=180)
    print(OUT / 'diagnosis.png')


if __name__ == '__main__':
    main()
