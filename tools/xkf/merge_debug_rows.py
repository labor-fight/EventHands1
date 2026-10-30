#!/usr/bin/env python3
"""Combine the three explicitly nonformal step2000 whole-sequence rows."""
from pathlib import Path
import json
import hashlib
import statistics

ROOT = Path(__file__).resolve().parents[2]


def main():
    rows, paths = [], []
    for seed in (3407, 3408, 3409):
        path = ROOT / f'outputs/semkine/xkf_debug2000_s{seed}_main_row.json'
        row = json.loads(path.read_text())
        assert row['protocol']['formal'] is False
        assert row['protocol']['exact_optimizer_step'] == 2000
        assert row['numeric_status'] == 'PASS' and row['numeric_valid'] is True
        assert row['finite_subset_reduction'] is False
        assert row['per_seed'][str(seed)]['n_frames'] == 2590
        rows.append(row)
        paths.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    result = {k: rows[0][k] for k in ('macs_forward_packet', 'macs_forward_packet_note', 'params_total')}
    for key in result:
        assert all(row[key] == result[key] for row in rows)
    result['per_seed'] = {seed: val for row in rows for seed, val in row['per_seed'].items()}
    result['two_seed_mean'] = {
        group: {metric: statistics.mean(row['two_seed_mean'][group][metric] for row in rows)
                for metric in rows[0]['two_seed_mean'][group]}
        for group in ('overall', 'local', 'global')}
    result['latency_ms_scaled_full1p75'] = statistics.mean(row['latency_ms_scaled_full1p75'] for row in rows)
    result['latency_context_audit'] = {'aggregation': 'arithmetic mean of per-seed full-model-scaled measurements',
                                      'sources': [row['latency_context_audit'] for row in rows]}
    result['protocol'] = dict(rows[0]['protocol'], seeds=[3407, 3408, 3409], n_seeds=3)
    result.update(finite_subset_reduction=False, numeric_status='PASS', numeric_valid=True,
                  provenance={'source_rows': paths, 'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    output = ROOT / 'outputs/semkine/xkf_debug2000_main_row.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(output)


if __name__ == '__main__':
    main()
