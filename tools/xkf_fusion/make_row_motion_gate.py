#!/usr/bin/env python3
"""Full2590 debug row with actual count gating and batch-stable inference.

Uses the existing centroid row producer and actual-policy latency/MAC audits.
All source adapters are explicit and hashed; historical producers are untouched.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from semkine.xkf_motion_gate import MotionCountFusionPolicy
from semkine.xkf_motion_gate_runtime import XKFStableMotionGateModel
from tools.xkf_fusion import make_row_centroid as C


def main():
    original_sources = C.M.sources
    def sources():
        result = original_sources()
        for name in ('semkine/xkf_motion_gate.py', 'semkine/xkf_stable_geometry.py',
                     'semkine/xkf_motion_gate_runtime.py', 'tools/xkf_fusion/make_row_motion_gate.py'):
            result[name] = C.M.E.sha(ROOT / name)
        return result
    original_write = C.M.X.write_json
    def write_json(path, value):
        if isinstance(value, dict) and 'mac_profile_audit' in value:
            value = dict(value, stability_mode='per_sample_model',
                         stability_scope='actual uncached forward per sample; batch1 deployment arithmetic retained')
        return original_write(path, value)
    C.M.sources = sources
    C.M.X.write_json = write_json
    C.XKFCentroidModel = XKFStableMotionGateModel
    C.CentroidFusionPolicy = MotionCountFusionPolicy
    C.main()


if __name__ == '__main__':
    main()
