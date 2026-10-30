#!/usr/bin/env python3
"""Fresh own-loops for declared W50 motion/update gates; frozen evaluator unchanged."""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'model')]
from semkine.xkf_motion_gate import MotionCountFusionPolicy,XKFMotionGateModel
from tools.xkf_fusion import evaluate_tuned as Q
from tools.xkf_fusion.evaluate_centroid import rollout_with_geometry

def main():
    parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--spec',required=True,type=Path)
    args,_=parser.parse_known_args();spec=json.loads(args.spec.read_text())
    if spec.get('geometry')!='all21_joint_centroid_relative_to_state_translation':
        raise ValueError('explicit centroid geometry required')
    original_sources,original_rollout=Q.sources,Q.rollout
    def sources():
        result=original_sources()
        for name in ('semkine/xkf_centroid.py','semkine/xkf_motion_gate.py','tools/xkf_fusion/evaluate_centroid.py','tools/xkf_fusion/evaluate_motion_gate.py'):
            result[name]=Q.E.sha(ROOT/name)
        return result
    def rollout(model,meta,clip,cache,candidates):
        return rollout_with_geometry(original_rollout,model,meta,clip,cache,candidates)
    Q.FusionPolicy=MotionCountFusionPolicy;Q.XKFFusionModel=XKFMotionGateModel
    Q.sources,Q.rollout=sources,rollout
    Q.main()
if __name__=='__main__':main()
