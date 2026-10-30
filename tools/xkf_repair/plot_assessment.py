#!/usr/bin/env python3
"""Plot a complete r2 assessment; never infer values from missing evaluations.

Defaults to the retained step2000 versus paired step2200 debug comparison.
An ASSESSED result may have failed debug safeguards and is still plotted as FAIL;
INCOMPLETE/INVALID evidence is rejected before creating output files.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (3407, 3408, 3409)
ARMS = ('old2000', 'control', 'repair')
COLORS = ('#2479b2', '#e48825', '#30926d')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def paired_values(value, label):
    """Check all three seeds and reducer consistency, returning seed-by-arm data."""
    require(set(value['per_seed']) == set(map(str, SEEDS)), f'{label}: expected all three fixed seeds')
    require(set(value['equal_seed_mean']) == set(ARMS), f'{label}: incomplete arm means')
    rows = []
    for seed in SEEDS:
        row = value['per_seed'][str(seed)]
        require(all(number(row.get(arm)) and row[arm] >= 0 for arm in ARMS),
                f'{label}: nonfinite, negative or missing metric for seed {seed}')
        for delta, left, right in (
            ('repair_minus_control', 'repair', 'control'),
            ('repair_minus_old2000', 'repair', 'old2000'),
            ('control_minus_old2000', 'control', 'old2000'),
        ):
            require(number(row.get(delta)) and math.isclose(row[delta], row[left] - row[right], rel_tol=1e-9, abs_tol=1e-8),
                    f'{label}: inconsistent paired delta for seed {seed}')
        rows.append([row[arm] for arm in ARMS])
    values = np.asarray(rows, dtype=float)
    for i, arm in enumerate(ARMS):
        mean = value['equal_seed_mean'][arm]
        require(number(mean) and math.isclose(mean, float(values[:, i].mean()), rel_tol=1e-9, abs_tol=1e-8),
                f'{label}: inconsistent equal-seed mean for {arm}')
    return values


def indexed(groups, keys, count, label):
    require(isinstance(groups, list) and len(groups) == count, f'{label}: incomplete comparison groups')
    result = {}
    for group in groups:
        identity = tuple(group[key] for key in keys)
        require(identity not in result, f'{label}: duplicate comparison group {identity}')
        result[identity] = group
    return result


def extract(assessment):
    require(assessment.get('kind') == 'XKF_R2_PAIRED_DEBUG_ASSESSMENT', 'not an r2 assessment')
    require(assessment.get('status') == 'ASSESSED',
            f"refusing assessment status {assessment.get('status')!r}; complete valid evidence is required")
    require(assessment.get('validation_errors') == [] and assessment.get('missing') == [], 'assessment has unresolved evidence')
    require(assessment.get('exact_optimizer_step') == 2200, 'expected fixed step2200 endpoint')
    require(all(assessment.get(key) is False for key in
                ('formal_gate_assessed', 'automatic_full_training', 'full_training_authorized_by_this_assessment')),
            'assessment scope must remain nonformal and must not authorize training')
    guard = assessment['debug_readiness_safeguards']
    require(guard.get('eligible_for_decision') is True and guard.get('status') in ('PASS', 'FAIL'), 'unassessed debug safeguards')
    require(set(assessment['arms']) == {'control', 'repair'}, 'missing paired arm')
    require(set(assessment['old2000']) == set(map(str, SEEDS)), 'missing old2000 seed')
    for arm in ('control', 'repair'):
        require(set(assessment['arms'][arm]) == set(map(str, SEEDS)), f'{arm}: missing seed')
        for seed in SEEDS:
            evidence = assessment['arms'][arm][str(seed)]
            require(evidence.get('direct_complete_all_rows') is True and evidence.get('main_row_complete_2590') is True
                    and evidence.get('rollout', {}).get('complete') is True, f'{arm}/{seed}: incomplete evaluation')
    mechanisms = assessment['mechanisms']
    rollout = indexed(mechanisms['rollout'], ('cohort', 'condition', 'mode', 'scope'), 432, 'rollout')
    direct = indexed(mechanisms['direct'], ('sequence', 'window_ms', 'variant'), 128, 'direct')
    main = indexed(mechanisms['main_row'], ('cohort', 'metric'), 12, 'main_row')
    data = {}
    for mode in ('i', 'fused'):
        entry = rollout[('local_bank', 'clean', mode, 'all_observed')]
        data[mode] = paired_values(entry['metrics']['abs_mpjpe_mm'], f'local-bank {mode}')
    entry = direct[('zgz_local', 200, 'learned__predpose_pred2d')]
    require(entry['n_rows'] == 1204, 'D W200 local denominator must be all 1204 packets')
    data['d'] = paired_values(entry['metrics']['t_rmse_mm'], 'D W200 local')
    data['ra'] = paired_values(main[('overall', 'mpjpe_ra_mm')], 'main-row RA overall')
    data['local'] = paired_values(main[('local', 'mpjpe_ra_mm')], 'main-row MPJPE-local')
    return data, guard['status']


def draw(data, guard_status, output, provenance):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.lines import Line2D

    font = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    chinese = font.is_file()
    if chinese:
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'svg.fonttype': 'path'})
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 5.2))
    fig.subplots_adjust(left=.055, right=.985, bottom=.26, top=.73, wspace=.27)
    x = np.arange(3)
    styles = (('i', '-', 'o'), ('fused', '--', 's'))
    for seed_index, color in enumerate(COLORS):
        for key, line, marker in styles:
            axes[0].plot(x, data[key][seed_index], color=color, linestyle=line, marker=marker, lw=1.7, ms=6)
        axes[1].plot(x, data['d'][seed_index], color=color, marker='o', lw=1.7, ms=6)
        for key, line, marker in (('ra', '-', 'o'), ('local', '--', 's')):
            axes[2].plot(x, data[key][seed_index], color=color, linestyle=line, marker=marker, lw=1.7, ms=6)
    for ax in axes:
        ax.set_xticks(x, ['old2000\nstep 2000', 'control\nstep 2200', 'repair\nstep 2200'])
        ax.set_xlim(-.17, 2.17)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.22)
        ax.tick_params(axis='x', length=0, pad=8)
        ax.margins(y=.14)
    axes[0].set_title('A  local-bank / clean / all_observed', loc='left', pad=39, fontsize=11)
    axes[0].set_ylabel('绝对 MPJPE (mm)' if chinese else 'Absolute MPJPE (mm)')
    axes[1].set_title('B  D：W200，全部 1,204 个 local 包' if chinese else 'B  D: W200, all 1,204 local packets', loc='left', pad=39, fontsize=11)
    axes[1].set_ylabel('状态平移 t 的 RMSE (mm)' if chinese else 'State translation t RMSE (mm)')
    axes[1].text(0, 1.035, 'learned confidence · predicted pose + 2D', transform=axes[1].transAxes, fontsize=9, color='#555')
    axes[2].set_title('C  主表：完整 2,590 包递推' if chinese else 'C  Main row: all 2,590 tracking packets', loc='left', pad=39, fontsize=11)
    axes[2].set_ylabel('RA-MPJPE (mm)')
    gray = '#555555'
    for ax, labels in ((axes[0], ('I', 'Fused')), (axes[2], ('RA overall', 'MPJPE-local'))):
        ax.legend([Line2D([], [], color=gray, linestyle='-', marker='o'),
                   Line2D([], [], color=gray, linestyle='--', marker='s')], labels,
                  loc='lower left', bbox_to_anchor=(0, 1.005), ncol=2, frameon=False,
                  borderaxespad=0, handlelength=2.0, columnspacing=1.4, fontsize=9)
    fig.legend([Line2D([], [], color=color, marker='o', lw=1.7) for color in COLORS],
               [f'Seed {seed}' for seed in SEEDS], loc='lower center', bbox_to_anchor=(.5, .10),
               ncol=3, frameon=False, columnspacing=3.0)
    fig.suptitle('XKF：step2000 → 配对 step2200，非正式调试' if chinese else
                 'XKF: step2000 to paired step2200 — nonformal debug', y=.975, fontsize=15)
    fig.text(.5, .895, f'完整证据；调试保障门 {guard_status}；不代表全量训练获准' if chinese else
             f'Complete evidence; debug safeguards {guard_status}; no full-training authorization',
             ha='center', fontsize=10, color='#555')
    fig.text(.5, .035, '同色为同一 seed；逐 seed 配对，越低越好。I/Fused 与主表采用不同评分范围。' if chinese else
             'Same color = same seed; paired values, lower is better. Rollout and main-row scoring scopes differ.',
             ha='center', fontsize=9, color='#555')
    output.mkdir(parents=True, exist_ok=True)
    description = json.dumps(provenance, sort_keys=True)
    for ext in ('png', 'svg'):
        fig.savefig(output / f'assessment.{ext}', dpi=180,
                    metadata={'Description': description, 'Title': 'XKF r2 paired step2200 nonformal debug'})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment', type=Path, default=ROOT / 'outputs/xkf/repair_20261015_r2/assessment.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'docs/assets/xkf_repair_20261015_r2')
    args = parser.parse_args()
    try:
        raw = args.assessment.read_bytes()
        data, guard_status = extract(json.loads(raw))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(2, f'Refusing to plot: {exc}\n')
    provenance = {'assessment_path': str(args.assessment.resolve()), 'assessment_sha256': hashlib.sha256(raw).hexdigest(),
                  'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'seeds': SEEDS, 'arms': ARMS, 'steps': [2000, 2200, 2200],
                  'formal_gate_assessed': False, 'debug_safeguards': guard_status}
    draw(data, guard_status, args.output_dir, provenance)
    print(json.dumps({'png': str(args.output_dir / 'assessment.png'), 'svg': str(args.output_dir / 'assessment.svg'),
                      **provenance}))


if __name__ == '__main__':
    main()
