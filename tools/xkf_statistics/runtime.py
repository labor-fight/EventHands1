"""Load only a declared paired-statistics policy accepted by actual training CV."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_runtime(path, seed, checkpoint, config):
    from semkine.xkf_paired_statistics import PairedStatisticsPolicy
    path = Path(path).resolve()
    value = json.loads(path.read_text())
    if (value['kind'] != 'TRAIN_ACCEPTED_PAIRED_STATISTICS_RUNTIME'
            or value['seed'] != seed or value['accepted_for_dev'] is not True
            or value['checkpoint_sha256'] != sha(checkpoint)):
        raise ValueError('wrong or unaccepted paired-statistics runtime')
    inputs = {str(path): sha(path)}
    for kind in ('assessment', 'declaration', 'fold_specs', 'policy'):
        file = Path(value[kind + '_path']).resolve()
        if sha(file) != value[kind + '_sha256']:
            raise ValueError('runtime evidence changed: ' + kind)
        inputs[str(file)] = sha(file)
    assessment = json.loads(Path(value['assessment_path']).read_text())
    if (assessment['status'] != 'ACCEPTED_FOR_DEV_EVALUATION'
            or assessment['accepted_for_dev'] is not True or assessment['failures']
            or assessment['declaration_sha256'] != value['declaration_sha256']):
        raise ValueError('actual training validation did not accept this candidate')
    for group in ('source_sha256', 'input_sha256'):
        if not assessment[group]:
            raise ValueError('missing actual validation provenance')
        for name, digest in assessment[group].items():
            file = Path(name)
            file = file if file.is_absolute() else ROOT / file
            if sha(file) != digest:
                raise ValueError('training validation source/input changed: ' + name)
            inputs[str(file.resolve())] = digest
    declaration = json.loads(Path(value['declaration_path']).read_text())
    bound = declaration['models'][str(seed)]
    if (declaration['seeds'] != [3407, 3408, 3409]
            or declaration['read_development_data'] is not False
            or bound['checkpoint_sha256'] != value['checkpoint_sha256']
            or bound['config_sha256'] != sha(config)):
        raise ValueError('runtime checkpoint disagrees with training declaration')
    fit = json.loads(Path(value['fold_specs_path']).read_text())
    if (fit['seed'] != seed or fit['declaration_sha256'] != value['declaration_sha256']
            or str(Path(value['fold_specs_path']).resolve()) not in inputs):
        raise ValueError('runtime coefficients are not from accepted fit evidence')
    # The fit must be in the assessor's actual input closure, not merely named
    # by an independently supplied runtime descriptor.
    assessed = {str((Path(n) if Path(n).is_absolute() else ROOT / n).resolve()): d
                for n, d in assessment['input_sha256'].items()}
    if assessed.get(str(Path(value['fold_specs_path']).resolve())) != value['fold_specs_sha256']:
        raise ValueError('assessor did not assess this fitted model')
    policy = json.loads(Path(value['policy_path']).read_text())
    if policy['fixed_effective_gains'] != fit['all_train']['coefficients']:
        raise ValueError('runtime coefficients differ from all-training refit')
    PairedStatisticsPolicy(policy)
    return policy, inputs
