#!/usr/bin/env python3
"""Centroid-policy wrapper over the frozen 112-clip target-free own-loop evaluator.

Only explicit candidate fusion policies use centroid coupling. The historical
six modes remain original state-t fusion controls. Optional BN .5 uses the same
independently declared training-only buffers as the previous BN wrapper.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]

from semkine.xkf_centroid import CentroidFusionPolicy, XKFCentroidModel
from tools.xkf_fusion import evaluate as Q


class BoundCandidate:
    def __init__(self, candidate, model, betas):
        self.candidate, self.model, self.betas = candidate, model, betas

    def for_clip(self, clip_id):
        return self.candidate.for_clip(clip_id).bind_geometry(self.model._fk, self.betas)


def rollout_with_geometry(original_rollout, model, meta, clip, cache, candidates):
    bound = [BoundCandidate(candidate, model, cache.beta) for candidate in candidates]
    return original_rollout(model, meta, clip, cache, bound)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--spec', required=True, type=Path)
    args, _ = parser.parse_known_args()
    spec = json.loads(args.spec.read_text())
    if spec.get('geometry') != 'all21_joint_centroid_relative_to_state_translation':
        raise ValueError('centroid geometry must be explicitly preregistered')
    for candidate in spec['candidates']:
        if candidate.get('translation_reference') not in ('state_t', 'joint_centroid'):
            raise ValueError('every candidate must explicitly name its translation reference')
    original_sources, original_rollout = Q.sources, Q.rollout

    def sources():
        result = original_sources()
        for name in ('semkine/xkf_centroid.py', 'tools/xkf_fusion/evaluate_centroid.py'):
            result[name] = Q.E.sha(ROOT / name)
        return result

    def rollout(model, meta, clip, cache, candidates):
        return rollout_with_geometry(original_rollout, model, meta, clip, cache, candidates)

    Q.FusionPolicy = CentroidFusionPolicy
    Q.XKFFusionModel = XKFCentroidModel
    Q.sources, Q.rollout = sources, rollout
    if 'bn_calibration_by_seed' in spec:
        from tools.xkf_fusion import evaluate_bn
        evaluate_bn.main()
    else:
        Q.main()


if __name__ == '__main__':
    main()
