#!/usr/bin/env python3
"""Plot registered diagnostic outputs, not a new main accuracy table."""
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / 'docs/assets/xde_20261014'
SOURCES = {}


def read(relative):
    p = ROOT / relative
    SOURCES[relative] = hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    runs = [('C0', 'xa_c0_s3407'), ('Q0', 'xdl_q0_s3407'),
            ('Q1', 'xdl_q1_s3407'), ('N1', 'xdl_n1_s3407')]
    values = [read(f'outputs/xde/e1/{r}_clean_local.json')['summary'] for _, r in runs]
    heads = {(s, k): read(f'outputs/xde/e2/heads/{k}_w128_s{s}.json')
             for s in (1701, 1702) for k in ('state', 'pre', 'post', 'wrong_pre', 'wrong_post')}
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.0))
    colors = ['#678f6c', '#93b5d6', '#225da0', '#9ca0a6']
    x = np.arange(4)
    height = [v['wrist_mm']['mean'] for v in values]
    axes[0].bar(x, height, color=colors, width=.65)
    axes[0].axhline(values[0]['hold_wrist_mm']['mean'], color='#a24137', linestyle='--',
                    label=f"Hold {values[0]['hold_wrist_mm']['mean']:.2f} mm")
    for i, y in enumerate(height): axes[0].text(i, y+1.0, f'{y:.1f}', ha='center', fontsize=9)
    axes[0].set(xticks=x, xticklabels=[r[0] for r in runs], ylabel='Wrist error (mm)',
                title='A. Correct previous state\nAll 1,204 local packets', ylim=(0, 78))
    axes[0].legend(frameon=False, loc='upper left')
    interfaces = ['state', 'pre', 'post', 'wrong_pre', 'wrong_post']
    labels = ['State', 'Pre-34', 'Post-4', 'Wrong\npre', 'Wrong\npost']
    for ax, group, title in zip(axes[1:], ['check', 'zgz_local'],
                               ['B. Selected training-check blocks', 'C. Local sequence: no head fitting']):
        for s, marker, offset in [(1701, 'o', -.10), (1702, 's', .10)]:
            ys = [heads[(s, k)]['metrics'][group]['z_mae_mm'] for k in interfaces]
            ax.plot(np.arange(5)+offset, ys, marker, ms=6, label=f'Head seed {s}',
                    color='#225da0' if s==1701 else '#db923b')
        ax.set(xticks=np.arange(5), xticklabels=labels, ylabel='z correction MAE (mm)', title=title)
        ax.grid(axis='y', alpha=.2)
        ax.set_ylim(bottom=0)
    axes[1].legend(frameon=False, loc='upper left')
    fig.text(.5, .01, 'Frozen Q1 encoder; width 128; clean / -50 / +50 mm z inputs. '
             'Check is used for epoch selection; two head seeds are not two encoder seeds.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0,.055,1,1))
    for ext in ('png', 'svg'):
        fig.savefig(DEST / f'fast_evidence.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    manifest = {'source_sha256': SOURCES, 'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'scope': 'diagnostic plot, no new official main row; all head seeds retained',
                'figure_sha256': {ext: hashlib.sha256((DEST/f'fast_evidence.{ext}').read_bytes()).hexdigest()
                                  for ext in ('png','svg')}}
    (DEST/'figure_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(DEST/'fast_evidence.png')


if __name__ == '__main__':
    main()
