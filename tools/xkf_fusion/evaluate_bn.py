#!/usr/bin/env python3
"""Independent BN-buffer adapter for the unchanged r3 own-loop evaluator."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
import numpy as np
import torch
from tools.xkf_fusion import evaluate as Q
from tools.xkf_fusion.bn_probe import apply_buffers, bn_modules, parameter_digest


def parity(raw, old, candidates, modes):
    result = {'scope': 'BN changes D intentionally; old D/fused differences descriptive only',
              'raw_inputs_identical': True, 'unchanged_i_and_hold_checked': True}
    for key in ('truth', 'clean', 'initial', 'end_ms', 'packet_ids', 'lnes_window_ms', 'n_raw_events'):
        if not np.array_equal(raw[key], old[key]):
            raise ValueError(f'original raw packet identity changed: {key}')
    for name in ('i', 'hold'):
        index = Q.BASE_MODES.index(name)
        difference = np.abs(raw['predictions'][:, :, index] - old['predictions'][:, :, index])
        t, pose = float(difference[..., :3].max() * 1000), float(difference[..., 3:].max())
        result[f'{name}_translation_max_mm'], result[f'{name}_pose_max_rad'] = t, pose
        if t > .2 or pose > .002:
            raise ValueError(f'unchanged {name} trajectory differs')
    for candidate in candidates:
        policy = Q.FusionPolicy(candidate.spec)
        if policy.is_identity() and candidate.document is None:
            delta = np.abs(raw['predictions'][:, :, modes.index(policy.name)]
                           - raw['predictions'][:, :, modes.index('fused')])
            t, pose = float(delta[..., :3].max() * 1000), float(delta[..., 3:].max())
            result[f'{policy.name}_vs_current_bn_fused_translation_max_mm'] = t
            result[f'{policy.name}_vs_current_bn_fused_pose_max_rad'] = pose
            if t > .2 or pose > .002:
                raise ValueError('identity candidate differs from current BN fused')
    delta = np.abs(raw['predictions'][:, :, :len(Q.BASE_MODES)] - old['predictions'])
    result['descriptive_all_old_modes_translation_change_max_mm'] = float(delta[..., :3].max() * 1000)
    result['descriptive_all_old_modes_pose_change_max_rad'] = float(delta[..., 3:].max())
    return result


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--seed', required=True, type=int)
    pre.add_argument('--spec', required=True, type=Path)
    args, _ = pre.parse_known_args()
    spec = json.loads(args.spec.read_text())
    descriptor = spec['bn_calibration_by_seed'][str(args.seed)]
    path = ROOT / descriptor['path']
    if Q.E.sha(path) != descriptor['sha256']:
        raise ValueError('BN buffer SHA differs from frozen specification')
    artifact = torch.load(path, map_location='cpu', weights_only=False)
    if artifact['alpha'] != .5 or artifact['calibration_batches'] != 72:
        raise ValueError('this wrapper requires the declared complete BN alpha .5 candidate')
    original_load, original_sources = Q.E.load_model, Q.sources
    loaded = []

    def load_model(*values, **kwargs):
        model, cfg, identity = original_load(*values, **kwargs)
        if identity['checkpoint_sha256'] != artifact['base_checkpoint_sha256']:
            raise ValueError('BN calibration checkpoint mismatch')
        if parameter_digest(model) != artifact['parameter_sha256']:
            raise ValueError('BN calibration parameter identity mismatch')
        modules = bn_modules(model)
        if set(modules) != set(artifact['bn_buffers']):
            raise ValueError('BN module identity mismatch')
        apply_buffers(modules, artifact['bn_buffers'])
        model.eval()
        loaded.append(model)
        identity.update(bn_calibration_alpha=.5, bn_buffers=str(path),
                        bn_buffers_sha256=descriptor['sha256'])
        return model, cfg, identity

    def sources():
        result = original_sources()
        for name in ('tools/xkf_fusion/evaluate_bn.py', 'tools/xkf_fusion/bn_probe.py',
                     descriptor['path']):
            result[name] = Q.E.sha(ROOT / name)
        return result

    Q.E.load_model = load_model
    Q.sources = sources
    Q.parity = parity
    Q.main()
    for model in loaded:
        if parameter_digest(model) != artifact['parameter_sha256']:
            raise ValueError('own-loop evaluation changed parameters')
        for name, module in bn_modules(model).items():
            for key, expected in artifact['bn_buffers'][name].items():
                if not torch.equal(getattr(module, key).cpu(), expected):
                    raise ValueError('own-loop evaluation updated BN buffers')


if __name__ == '__main__':
    main()
