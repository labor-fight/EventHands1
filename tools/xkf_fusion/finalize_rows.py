#!/usr/bin/env python3
"""Validate complete three-seed debug rows, archive producers, and merge a row."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (3407, 3408, 3409)
KEYS = ('mpjpe_abs_mm', 'mpvpe_abs_mm', 'mpjpe_ra_mm', 'mpvpe_ra_mm')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, list):
        for item in value:
            finite(item)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--run', required=True)
    args = ap.parse_args()
    target = ROOT / f'outputs/semkine/{args.run}_main_row.json'
    if target.exists():
        raise FileExistsError(target)
    rows, provenance = {}, {}
    for seed in SEEDS:
        path = args.input_dir / f's{seed}_main_row.json'
        row = json.loads(path.read_text())
        finite(row)
        assert row['protocol']['formal'] is False and row['finite_subset_reduction'] is False
        assert row['provenance']['seed'] == seed and row['numeric_status'] == 'PASS'
        step = row['protocol']['exact_optimizer_step']
        assert 2200 <= step < 6000
        rawpath = Path(str(path) + '.raw_accuracy.json')
        raw = json.loads(rawpath.read_text())
        assert raw['complete'] and len(raw['raw_sequences']) == 2
        byname = {item['seq']: item for item in raw['raw_sequences']}
        assert set(byname) == {'zgz_local', 'zgz_global'}
        accuracy = row['per_seed'][str(seed)]
        for seq, count, group in [('zgz_local', 1204, 'local'), ('zgz_global', 1386, 'global')]:
            assert byname[seq]['n_frames'] == count
            with np.load(str(path) + f'.{seq}.npz', allow_pickle=False) as predictions:
                assert predictions['predictions'].shape == (count, 51)
                assert np.isfinite(predictions['predictions']).all()
                assert predictions['run_ids'].shape == (count,)
            for key in KEYS:
                assert abs(accuracy[group][key] - byname[seq][key]) < 1e-8
        for key in KEYS:
            expected = sum(item['n_frames'] * item[key] for item in byname.values()) / 2590
            assert abs(accuracy['overall'][key] - expected) < 1e-8
            for group in ('overall', 'local', 'global'):
                assert abs(accuracy[group][key] - row['two_seed_mean'][group][key]) < 1e-8
        for name, digest in row['provenance']['source_sha256'].items():
            original = ROOT / name
            archived = args.input_dir / 'source_snapshot' / name
            if sha(original) == digest:
                archived.parent.mkdir(parents=True, exist_ok=True)
                if not archived.exists():
                    shutil.copyfile(original, archived)
            assert archived.exists() and sha(archived) == digest, name
        for name, digest in row['protocol']['input_sha256'].items():
            assert sha(name) == digest
        rows[seed] = row
        provenance[str(seed)] = {'path': str(path.resolve()), 'sha256': sha(path),
                                'raw_accuracy_sha256': sha(rawpath), 'identity': row['provenance']}
    first = rows[SEEDS[0]]
    for row in rows.values():
        assert row['params_total'] == first['params_total']
        assert row['macs_forward_packet'] == first['macs_forward_packet']
    merged = {key: first[key] for key in ('params_total', 'macs_forward_packet')}
    merged.update(per_seed={str(s): rows[s]['per_seed'][str(s)] for s in SEEDS},
        two_seed_mean={g: {k: float(np.mean([rows[s]['per_seed'][str(s)][g][k] for s in SEEDS]))
                          for k in KEYS} for g in ('overall', 'local', 'global')},
        latency_ms_scaled_full1p75=float(np.mean([rows[s]['latency_ms_scaled_full1p75'] for s in SEEDS])),
        latency_context_audit={str(s): rows[s]['latency_context_audit'] for s in SEEDS},
        protocol=dict(first['protocol'], seeds=list(SEEDS), n_seeds=3, arm=args.run,
                      averaging='equal three-seed mean in legacy two_seed_mean field'),
        provenance=provenance, numeric_status='PASS', finite_subset_reduction=False,
        automatic_full_training=False, finalizer_sha256=sha(__file__))
    target.write_text(json.dumps(merged, indent=2, allow_nan=False) + '\n')
    print(target)


if __name__ == '__main__':
    main()
