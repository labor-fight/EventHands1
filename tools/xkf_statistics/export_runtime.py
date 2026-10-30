#!/usr/bin/env python3
"""Export all three fixed policies only after complete training-CV acceptance."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from tools.xkf_statistics.runtime import sha, load_runtime
from semkine.xkf_paired_statistics import PairedStatisticsPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assessment', required=True, type=Path)
    parser.add_argument('--declaration', required=True, type=Path)
    parser.add_argument('--fit-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assessment = json.loads(args.assessment.read_text())
    declaration = json.loads(args.declaration.read_text())
    if (assessment['status'] != 'ACCEPTED_FOR_DEV_EVALUATION'
            or assessment['accepted_for_dev'] is not True or assessment['failures']
            or assessment['declaration_sha256'] != sha(args.declaration)):
        raise ValueError('complete actual train-only CV has not accepted the candidate')
    for key in ('input_sha256', 'source_sha256'):
        for name, digest in assessment[key].items():
            file = Path(name)
            file = file if file.is_absolute() else ROOT / file
            if sha(file) != digest:
                raise ValueError('validation evidence changed: ' + name)
    if declaration['seeds'] != [3407, 3408, 3409] or declaration['read_development_data'] is not False:
        raise ValueError('wrong training-only population')
    records = []
    for seed in declaration['seeds']:
        fitpath = (args.fit_dir / f'folds_s{seed}.json').resolve()
        fit = json.loads(fitpath.read_text())
        if (fit['seed'] != seed or fit['declaration_sha256'] != sha(args.declaration)
                or assessment['input_sha256'].get(str(fitpath)) != sha(fitpath)):
            raise ValueError('fit is not the one assessed for this seed')
        policy = dict(id='paired_statistics', gain_scale=[1., 1., 1.], count_gate='ramp1000',
                      motion_count_gate='ramp1000', translation_reference='joint_centroid',
                      fixed_effective_gains=fit['all_train']['coefficients'])
        PairedStatisticsPolicy(policy)
        records.append((seed, fitpath, policy))
    args.output.mkdir(parents=True)
    for seed, fitpath, policy in records:
        policypath = (args.output / f'policy_s{seed}.json').resolve()
        policypath.write_text(json.dumps(policy, indent=2) + '\n')
        value = dict(kind='TRAIN_ACCEPTED_PAIRED_STATISTICS_RUNTIME', accepted_for_dev=True,
                     seed=seed, checkpoint_sha256=declaration['models'][str(seed)]['checkpoint_sha256'],
                     automatic_full_training=False, formal=False)
        for key, path in (('assessment', args.assessment), ('declaration', args.declaration),
                          ('fold_specs', fitpath), ('policy', policypath)):
            value[key + '_path'] = str(path.resolve())
            value[key + '_sha256'] = sha(path)
        path = args.output / f'runtime_s{seed}.json'
        path.write_text(json.dumps(value, indent=2) + '\n')
        load_runtime(path, seed, declaration['models'][str(seed)]['checkpoint'],
                     declaration['models'][str(seed)]['config'])
    print(json.dumps(dict(output=str(args.output), seeds=declaration['seeds'],
                          status='EXPORTED_FOR_DEV_EVALUATION', full_training=False)))


if __name__ == '__main__':
    main()
