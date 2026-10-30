#!/usr/bin/env python3
"""Merge real full rows, retaining each seed's train-fitted fixed policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from tools.xkf_fusion import finalize_rows as F
from semkine.xkf_paired_statistics import PairedStatisticsPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', required=True, type=Path)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    policies = {}
    for seed in F.SEEDS:
        row = json.loads((args.input_dir / f's{seed}_main_row.json').read_text())
        policy = row['protocol']['policy']
        PairedStatisticsPolicy(policy)
        if row['protocol']['exact_optimizer_step'] != 2600:
            raise ValueError('wrong paired-statistics endpoint')
        policies[str(seed)] = policy
    F.main()
    path = ROOT / f'outputs/semkine/{args.run}_main_row.json'
    value = json.loads(path.read_text())
    value['protocol'].pop('policy')
    value['protocol']['policy_by_seed'] = policies
    value['protocol']['calibration_scope'] = 'same training-only paired19 estimator per model; actual identity/family held-out validation before development'
    value['base_finalizer_sha256'] = value.pop('finalizer_sha256')
    value['finalizer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
