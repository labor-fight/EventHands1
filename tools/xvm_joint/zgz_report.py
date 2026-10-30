"""Generate the fixed XJ zgz rows through the original main-row tools."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from .train import dump, sha

ROOT = Path(__file__).resolve().parents[2]
METRICS = ('mpjpe_ra_mm', 'mpvpe_ra_mm', 'mpjpe_abs_mm', 'mpvpe_abs_mm')


def main_rows(protocol_path, freeze_path, output):
    from .zgz_eval import verify_identity
    protocol = verify_identity(protocol_path, freeze_path)[0]
    output = Path(output).resolve()
    cfg = json.loads(Path(protocol['config']['path']).read_text())
    import yaml
    base_cfg = yaml.safe_load(Path(protocol['base_config']['path']).read_text())
    manifest = json.loads(Path(protocol['split']['path']).read_text())
    seqs = [(name, manifest['val_core']['legacy_dir'][name]) for name in protocol['sequences']]
    paths, labels, evidence = [], {}, {}
    for mode in protocol['modes']:
        costs_path = output / 'costs' / (mode + '.json')
        costs = json.loads(costs_path.read_text())
        if (costs['status'] != 'COMPLETE' or costs['mode'] != mode
                or costs['protocol_sha256'] != sha(protocol_path)
                or costs['sourcefreeze_sha256'] != sha(freeze_path)
                or costs['checkpoint_sha256'] != protocol['checkpoint']['sha256']):
            raise RuntimeError('Complete measured packet costs required')
        for arm in protocol['arms']:
            rows = []
            for seq in protocol['sequences']:
                folder = output / 'accuracy' / mode / arm / seq
                summary = json.loads((folder / 'summary.json').read_text())
                if (summary['status'] != 'COMPLETE' or summary['arm'] != arm
                        or summary['mode'] != mode or summary['sequence'] != seq
                        or summary['n_frames'] != protocol['expected_frames'][seq]
                        or summary['eval_protocol_sha256'] != sha(protocol_path)
                        or summary['eval_sourcefreeze_sha256'] != sha(freeze_path)):
                    raise RuntimeError('Incomplete or mismatched accuracy worker')
                if (sha(folder/'frames.npz') != summary['frames_sha256']
                        or sha(folder/'trace.jsonl') != summary['trace_sha256']):
                    raise RuntimeError('Changed raw prediction/metric evidence')
                with np.load(folder/'frames.npz', allow_pickle=False) as arrays:
                    for metric in METRICS:
                        values = arrays['metric_' + metric]
                        if len(values) != summary['n_frames'] or not np.isfinite(values).all():
                            raise RuntimeError('Missing/nonfinite scored frames')
                        if not np.isclose(values.mean(), summary['metrics'][metric], atol=1e-6, rtol=0):
                            raise RuntimeError('Raw metrics and summary disagree')
                rows.append(summary)
            total = sum(row['n_frames'] for row in rows)
            if total != protocol['expected_total_frames']:
                raise RuntimeError('The complete2590-frame protocol is required')
            metrics = {}
            for block in ('overall', 'local', 'global'):
                selected = rows if block == 'overall' else [r for r in rows if r['sequence'].endswith('_'+block)]
                denominator = sum(r['n_frames'] for r in selected)
                metrics[block] = {key: sum(r['n_frames']*r['metrics'][key] for r in selected)/denominator
                                  for key in METRICS}
            step = protocol['checkpoint']['updates'] if arm == 'primary' else 6000
            checkpoint = protocol['checkpoint']['path'] if arm == 'primary' else protocol['base_checkpoint']['path']
            agg = dict(step=step, ckpt=checkpoint, sequences=protocol['sequences'], n_frames=total,
                       per_seq={r['sequence']:r['metrics'] for r in rows}, **metrics)
            name = 'xj_' + output.name + '_' + mode + '_' + arm
            step_path = ROOT/'outputs/semkine'/f'{name}_main_row_step{step}.json'
            alias = ROOT/'outputs/semkine'/f'{name}_main_row.json'
            if step_path.exists() or alias.exists():
                raise FileExistsError('Never overwrite a previously generated main row')
            measured = costs['arms'][arm]
            anchor = measured['anchor_full_raw_ms']
            canonical_macs = costs['arms']['global']['macs_forward_packet']
            canonical_params = measured['params_canonical']
            # This adapter only supplies already measured evidence to the
            # original writer. Its aggregation and field layout stay intact.
            spec = importlib.util.spec_from_file_location('_xj_original_main_row', ROOT/'tools/make_s36_row.py')
            writer = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(writer)
            writer.eval_seed = lambda *args, _agg=agg, **kwargs: (_agg, copy.deepcopy(base_cfg), seqs, Path(protocol['data_root']))
            writer.latency_anchor = lambda *args, **kwargs: anchor
            writer.latency_s36 = lambda *args, **kwargs: (measured['latency_ms_raw'],
                    measured['latency_mean_events_per_packet'], measured['latency_packets'])
            writer.macs_s36 = lambda *args, **kwargs: (canonical_macs,
                    measured['macs_forward_packet']-canonical_macs, measured['params_total'], canonical_params)
            writer.macs_forward_packet = lambda *args, **kwargs: (measured['macs_forward_packet'], measured['macs_forward_packet_note'])
            previous_argv = sys.argv
            sys.argv = ['make_s36_row.py', '--run', name, '--seeds', str(protocol['seed']), '--step', str(step)]
            try:
                writer.main()
            finally:
                sys.argv = previous_argv
            # Add identity/scope metadata without changing any measured cell.
            written = json.loads(step_path.read_text())
            written['xj_zgz_binding'] = dict(protocol_sha256=sha(protocol_path), sourcefreeze_sha256=sha(freeze_path),
                    costs_sha256=sha(costs_path), accuracy_summary_sha256=[sha(output/'accuracy'/mode/arm/r['sequence']/'summary.json') for r in rows],
                    beta_input='zeros10', prediction_FK_beta='zeros10', GT_FK_beta='stored actual beta',
                    mode=mode, arm=arm, fixed_local_checkpoint=protocol['checkpoint'], frozen_base_checkpoint=protocol['base_checkpoint'],
                    forward_implementation='original masked-dense S43 image forward plus declared XJ packet modules',
                    cost_scope='standard-layer MAC lower bound; full packet latency excluding original packet staging',
                    training_scope='local-only Stage1; not a matched complete-budget end-to-end training run',
                    frontend_cost_field='canonical standard-layer MACs, including its head; local standard layers in macs_heads')
            dump(step_path, written)
            os.link(step_path, alias)
            paths.append(name)
            display = ('XJ 阶段一' if arm == 'primary' else 'S43 同口径控制')
            display += '，β0，' + ('W50原始' if mode == 'w50' else 'S39栈') + '，seed3407'
            labels[name] = display
            evidence[mode+'_'+arm] = dict(main_row_path=str(alias), main_row_sha256=sha(alias),
                                        canonical_writer_path=str(step_path), accuracy=agg,
                                        resources=measured)
    references = ['dt3_dz_l3_w200cfg', 's43n_sp_tf_pert']
    labels.update(dt3_dz_l3_w200cfg='S39：W200＋DT2＋ramp1000（历史参照）',
                  s43n_sp_tf_pert='S43：W50原始，已知β／稀疏执行（历史参照）')
    command = [sys.executable, str(ROOT/'tools/report_table.py'), *references, *paths]
    for name, display in labels.items():
        command += ['--label', name+'='+display]
    table = subprocess.check_output(command, cwd=ROOT, text=True)
    note = ('\n\n本轮新行均为单seed3407、固定末权重；XJ只完成局部模块训练。新行保持β0输入与预测FK，GT用真实β；历史行的手形／执行条件不同，仅作参照。'
            '\nLatency包含完整packet内的F16、几何、局部CNN、求解与过滤；沿原表排除磁盘及packet预制。FLOPs列为实际执行标准层的MAC下界，未计functional MANO/FK、有限差分、SDF与求解器。\n')
    (output/'main_table.md').write_text(table+note)
    result = dict(status='COMPLETE', protocol_sha256=sha(protocol_path), sourcefreeze_sha256=sha(freeze_path),
                  rows=evidence, table_sha256=sha(output/'main_table.md'), table_command=command,
                  scope='Actual zgz fixed-checkpoint evaluation; single-seed local-only trained XJ; no adoption claim')
    dump(output/'report.json', result)
    print(table)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--source-freeze', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main_rows(args.protocol, args.source_freeze, args.out)


if __name__ == '__main__':
    main()
