#!/usr/bin/env python3
"""Read-only CPU diagnosis of frozen G2 OLS and train motion-family overlap.

No gate is changed, no coefficient is fitted, and no network is forwarded.
"""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
from config import load_config
from semkine.dataset import sequences_for_split, splits_manifest, _read_meta51


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def slope(x, y):
    dx, dy = x-x.mean(), y-y.mean()
    require(float((dx*dx).mean()) > 0, 'degenerate OLS')
    return float((dx*dy).mean() / (dx*dx).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gates-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), 'refusing to overwrite diagnosis')
    args.output_dir.mkdir(parents=True)
    inputs, records, missing = {}, {}, []
    for arm in ('raw', 'aug'):
        for seed in (3407, 3408, 3409):
            label = f'{arm}_s{seed}'
            path = args.gates_dir / f'{label}.json'
            rawpath = path.with_suffix('.npz')
            if not path.exists() or not rawpath.exists():
                missing.append(label)
                continue
            value = json.loads(path.read_text())
            inputs[str(path.resolve())], inputs[str(rawpath.resolve())] = sha(path), sha(rawpath)
            require(value['raw']['sha256'] == sha(rawpath), 'original raw SHA differs')
            with np.load(rawpath, allow_pickle=False) as archive:
                x = archive['zgz_local_incoming_wrist'][:, :3, 2].astype(np.float64)
                y = archive['zgz_local_fused_wrist'][:, :3, 2].astype(np.float64)
                truth = archive['zgz_local_true_wrist'][:, 2].astype(np.float64)
            require(x.shape == y.shape == (1204, 3) and truth.shape == (1204,), 'incomplete literal G2 inputs')
            require(all(np.isfinite(v).all() for v in (x, y, truth)), 'nonfinite input; no finite-subset reduction')
            mx, my = x.mean(1), y.mean(1)
            dx, dy = x-mx[:, None], y-my[:, None]
            between_variance = float(np.mean((mx-mx.mean())**2))
            within_variance = float(np.mean(dx**2))
            total_variance = float(np.mean((x-x.mean())**2))
            between_covariance = float(np.mean((mx-mx.mean())*(my-my.mean())))
            within_covariance = float(np.mean(dx*dy))
            original = value['gates']['G2']
            pooled = slope(x, y)
            within = within_covariance / within_variance
            individual = (dx*dy).sum(1)/(dx*dx).sum(1)
            require(abs(pooled-original['depth_slope']) < 1e-12, 'original pooled formula mismatch')
            require(abs(float(individual.mean())-original['within_packet_depth_slope_mean']) < 1e-12,
                    'original saved within-packet mean differs')
            require(abs(total_variance-between_variance-within_variance) < 1e-15,
                    'ANOVA variance decomposition failed')
            require(abs(pooled-(between_covariance+within_covariance)/total_variance) < 1e-12,
                    'ANOVA OLS decomposition failed')
            records[label] = dict(original_G2=original, n_packets=1204, conditions=['clean', 'z_minus50', 'z_plus50'],
                pooled_OLS=pooled, within_packet_pooled_OLS=within,
                within_packet_mean_OLS=float(individual.mean()), between_packet_OLS=between_covariance/between_variance,
                within_packet_variance_weight=within_variance/total_variance,
                between_packet_variance_weight=between_variance/total_variance,
                weighted_between_contribution=between_covariance/total_variance,
                weighted_within_contribution=within_covariance/total_variance,
                oracle_GTcurrent_output_pooled_OLS=slope(x, np.repeat(truth[:, None], 3, axis=1)),
                within_packet_slope_percentiles={str(p): float(np.percentile(individual, p)) for p in (0, 5, 25, 50, 75, 95, 100)},
                same_packet_slope_gt_0_6_fraction=float(np.mean(individual > .6)),
                input_true_depth_std_mm=float(np.std(mx)*1000),
                numeric_decomposition_verified=True,
                scope='Within-packet centering and perfect-GT output are diagnostic only; the original G2 verdict remains unchanged.')
    config = ROOT / 'configs/xkf_repair/repair_s3407.yaml'
    cfg = load_config(str(config)); split = Path(splits_manifest(cfg))
    inputs[str(config)], inputs[str(split.resolve())] = sha(config), sha(split)
    sequences = sequences_for_split(Path(cfg['DATA']['ROOT']), 'train', split)
    require(len(sequences) == 72 and len({name.split('_')[0] for name, _ in sequences}) == 9,
            'not canonical nine identities / 72 sequences')
    families = {}
    for name, directory in sequences:
        family = re.sub(r'_v[234]$', '', name)
        path = Path(cfg['DATA']['ROOT']) / directory / (name + '.meta')
        inputs[str(path.resolve())] = sha(path)
        truth = _read_meta51(str(path))
        families.setdefault(family, []).append(dict(sequence=name, n_ms=len(truth),
            full_float32_GT_sha256=hashlib.sha256(np.ascontiguousarray(truth).tobytes()).hexdigest()))
    require(len(families) == 18 and all(len(v) == 4 for v in families.values()), 'unexpected family structure')
    duplicates = {name: dict(versions=versions, all_four_complete_GT_identical=len({v['full_float32_GT_sha256'] for v in versions}) == 1)
                  for name, versions in families.items()}
    require(all(sha(path) == digest for path, digest in inputs.items()), 'diagnostic input changed')
    result = dict(kind='INDEPENDENT_G2_AND_TRAIN_STATISTICS_DESIGN_AUDIT',
        status='COMPLETE_AVAILABLE_G2_AND_CANONICAL_DATA', formal=False, changed_original_gates=False,
        no_neural_forward=True, no_fitting=True, no_training=True, all_six_G2_complete=not missing,
        missing_G2=missing, G2_decomposition=records,
        decomposition='beta_pooled = w_between * beta_between + (1-w_between) * beta_within. '
                      'Each original packet has the same three +/-50mm perturbations; all 1204 packets retained.',
        motion_families=duplicates,
        identical_motion_family_count=sum(row['all_four_complete_GT_identical'] for row in duplicates.values()),
        recommendation='Keep nine-identity primary CV. Secondary leave-one-sequence-out across 72 versions does not exclude '
                       'the same motion trajectory in its other three versions; use 18 identity+local/global family folds '
                       'or label 72 folds only as version-transfer diagnostics. Canonical training split is unchanged.',
        source_sha256={str(Path(__file__).relative_to(ROOT)):sha(__file__), 'tools/xkf/diagnostics.py':sha(ROOT/'tools/xkf/diagnostics.py'),
                       'semkine/dataset.py':sha(ROOT/'semkine/dataset.py')}, input_sha256=inputs)
    destination = args.output_dir / 'result.json'
    destination.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(destination), complete_G2=len(records), missing_G2=missing,
                          identical_motion_families=result['identical_motion_family_count'])), flush=True)


if __name__ == '__main__':
    main()
