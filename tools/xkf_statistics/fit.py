#!/usr/bin/env python3
"""Fit paired gains on canonical training data, excluding whole motion families.

This creates cross-validation candidates, never an accepted deployment sidecar.
Actual independent trajectories and the predeclared acceptance remain required.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
from semkine.xkf_paired_statistics import fit_paired_gains


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_hashes(values):
    for name, digest in values.items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        if sha(path) != digest:
            raise ValueError('source/input changed: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', required=True, type=Path)
    parser.add_argument('--collection', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    declaration = json.loads(args.declaration.read_text())
    collection = json.loads(args.collection.read_text())
    if (declaration['status'] != 'ACTIVE_TRAIN_ONLY_POLICY_TRIAL'
            or declaration['read_development_data'] is not False
            or declaration['crossfit']['secondary'] != 'motion_family'):
        raise ValueError('active training-only family-held-out declaration required')
    if (collection['status'] != 'COMPLETE' or collection['partial_smoke']
            or collection['rows'] != 2016 or collection['actual_forward_calls'] != 8064
            or collection['declaration_sha256'] != sha(args.declaration)
            or collection['all_buffers_unchanged'] is not True):
        raise ValueError('incomplete or mismatched actual collection')
    for obj in (declaration, collection):
        verify_hashes(obj['source_sha256'])
        verify_hashes(obj['input_sha256'])
    raw_path = Path(collection['raw_path'])
    if sha(raw_path) != collection['raw_sha256']:
        raise ValueError('collection raw changed')
    seed = collection['seed']
    if seed not in declaration['seeds']:
        raise ValueError('unknown seed')
    model = declaration['models'][str(seed)]
    identity = collection['identity']
    if (identity['seed'] != seed or identity['global_step'] != 2600
            or identity['checkpoint_sha256'] != model['checkpoint_sha256']
            or identity['config_sha256'] != model['config_sha256']):
        raise ValueError('collection is bound to a different model/seed/endpoint')
    for key in ('config', 'checkpoint'):
        if sha(model[key]) != model[key + '_sha256']:
            raise ValueError('bound model changed')
    with np.load(raw_path, allow_pickle=False) as loaded:
        raw = {k: loaded[k] for k in loaded.files}
    indices = np.asarray(declaration['fit_row_indices'])
    if not np.array_equal(raw['row_indices'], indices):
        raise ValueError('collector changed canonical rows or order')
    canonical = json.loads(Path(declaration['rows_path']).read_text())
    sequences = np.array([canonical[i]['sequence'] for i in indices])
    if not np.array_equal(raw['sequence'], sequences):
        raise ValueError('sequence identities differ')
    identities = np.array([s.split('_')[0] for s in sequences])
    families = np.array([s.split('_v')[0] for s in sequences])
    if len(set(identities)) != 9 or len(set(families)) != 18 or len(set(sequences)) != 72:
        raise ValueError('canonical nine identities/18 motion families/72 sequences required')
    if any(s.startswith('zgz') for s in sequences):
        raise ValueError('development data in training fit')
    ei, ed, gate = raw['e_i'], raw['e_d'], raw['gate']
    speed = raw['gt_joint_speed_mm_s']
    if ei.shape != (2016, 51) or ed.shape != ei.shape or gate.shape != (2016,) or speed.shape != (2016,):
        raise ValueError('paired arrays have unexpected shape')
    if not np.isfinite(speed).all() or (speed < 0).any():
        raise ValueError('invalid physical GT joint speed')
    fallback = collection['baseline_effective_gains']
    options = declaration['fitting']

    def fit(mask, family=None):
        result = fit_paired_gains(ei[mask], ed[mask], gate[mask], fallback, **options)
        result.update(index_space='canonical2304', train_indices=indices[mask].tolist(),
                      excluded_indices=indices[~mask].tolist(), fold_family=family,
                      high_motion_threshold_mm_s=float(np.quantile(speed[mask], .75)))
        return result

    all_train = fit(np.ones(len(indices), bool))
    identity = {name: fit(identities != name, name) for name in sorted(set(identities))}
    family_fits = {name: fit(families != name, name) for name in sorted(set(families))}
    # Four versions share the same held-out motion family. These aliases are
    # only for lookup, never 72 independent validation folds.
    trial = {name: family_fits[name.split('_v')[0]] for name in sorted(set(sequences))}
    sources = {str(ROOT / name): sha(ROOT / name) for name in (
        'tools/xkf_statistics/fit.py', 'semkine/xkf_paired_statistics.py')}
    inputs = {str(args.declaration.resolve()): sha(args.declaration),
              str(args.collection.resolve()): sha(args.collection), str(raw_path): sha(raw_path)}
    result = dict(kind='PAIRED_STATISTICS_CROSSFIT_CANDIDATES', status='FITTED_NOT_ACCEPTED',
                  accepted_for_dev=False, automatic_full_training=False, seed=seed,
                  declaration_path=str(args.declaration.resolve()), declaration_sha256=sha(args.declaration),
                  collection_summary_path=str(args.collection.resolve()), collection_summary_sha256=sha(args.collection),
                  collection_raw_path=str(raw_path), collection_raw_sha256=sha(raw_path),
                  baseline_effective_gains=fallback, all_train=all_train,
                  identity=identity, motion_family=family_fits, trial=trial,
                  identity_folds=9, motion_family_folds=18, sequence_lookup_aliases=72,
                  source_sha256=sources, input_sha256=inputs,
                  note='all_train coefficients are inert candidates until actual held-out own-loop validation passes')
    verify_hashes(inputs)
    verify_hashes(sources)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(seed=seed, output=str(args.output), status=result['status'])), flush=True)


if __name__ == '__main__':
    main()
