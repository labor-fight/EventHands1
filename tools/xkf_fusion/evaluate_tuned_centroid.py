#!/usr/bin/env python3
"""Exact-step2400 D-only paired evaluation with explicit centroid candidates.

Uses the new tuned evaluator's checkpoint/config identity and I/hold parity.
No BN adaptation is applied: the D-only checkpoint already contains the frozen
parent BN buffers. Every candidate/seed/arm remains in the declared 112x9 plan.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from semkine.xkf_centroid import CentroidFusionPolicy, XKFCentroidModel
from tools.xkf_fusion import evaluate_tuned as Q
from tools.xkf_fusion.evaluate_centroid import rollout_with_geometry

DECLARATION = ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/evaluation_policy_declaration.json'


def validate_spec(spec, declaration):
    if spec.get('checkpoint_step') != 2400 or spec.get('formal') is not False:
        raise ValueError('paired D-only centroid evaluation requires declared nonformal exact step2400')
    if 'bn_calibration_by_seed' in spec:
        raise ValueError('new D evaluation must retain checkpoint BN; no evaluate_bn adapter')
    if spec.get('geometry') != 'all21_joint_centroid_relative_to_state_translation':
        raise ValueError('centroid geometry must match the explicit all21-joint declaration')
    candidates = spec['candidates']
    if len(candidates) != len(declaration['candidates']) or len(candidates) != 6:
        raise ValueError('all six predeclared candidates are required')
    for actual, expected in zip(candidates,declaration['candidates']):
        for key in ('id','gain_scale','count_gate','translation_reference'):
            if actual.get(key) != expected[key]:
                raise ValueError(f'predeclared candidate {expected["id"]} changed {key}')
        sources = [key for key in ('direct_correction_by_seed','direct_correction_file','direct_correction') if key in actual]
        calibrated = expected['calibration'] != 'none'
        if calibrated and (len(sources) != 1 or sources[0] == 'direct_correction'):
            raise ValueError('training affine requires one hashed external exact2400 fit artifact')
        if not calibrated and sources:
            raise ValueError('uncalibrated candidate unexpectedly has a D correction')


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--spec',required=True,type=Path)
    args,_=parser.parse_known_args()
    spec=json.loads(args.spec.read_text());declaration=json.loads(DECLARATION.read_text())
    validate_spec(spec,declaration)
    original_sources,original_rollout=Q.sources,Q.rollout
    def sources():
        result=original_sources()
        for name in ('semkine/xkf_centroid.py','tools/xkf_fusion/evaluate_centroid.py',
                     'tools/xkf_fusion/evaluate_tuned_centroid.py',str(DECLARATION.relative_to(ROOT))):
            result[name]=Q.E.sha(ROOT/name)
        return result
    def rollout(model,meta,clip,cache,candidates):
        return rollout_with_geometry(original_rollout,model,meta,clip,cache,candidates)
    Q.FusionPolicy=CentroidFusionPolicy
    Q.XKFFusionModel=XKFCentroidModel
    Q.sources,Q.rollout=sources,rollout
    Q.main()


if __name__=='__main__':
    main()
