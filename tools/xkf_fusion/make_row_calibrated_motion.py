#!/usr/bin/env python3
"""Complete debug row for a training-validated scalar I evidence gate."""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from semkine.xkf_calibrated_motion_gate import (
    CalibratedMotionCountPolicy, XKFCalibratedMotionModel,
)
from tools.xkf_fusion.evaluate_calibrated_motion import load_runtime
from tools.xkf_fusion import make_row_centroid as C


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--motion-runtime', type=Path, required=True)
    own, remaining = parser.parse_known_args()
    identity_parser = argparse.ArgumentParser(add_help=False)
    identity_parser.add_argument('--seed', type=int, required=True)
    identity_parser.add_argument('--checkpoint', type=Path, required=True)
    identity_parser.add_argument('--policy', type=Path, required=True)
    args, _ = identity_parser.parse_known_args(remaining)
    runtime, inputs = load_runtime(own.motion_runtime, args.seed, C.M.E.sha(args.checkpoint))
    policy = json.loads(args.policy.read_text())
    CalibratedMotionCountPolicy(policy)
    if policy.get('motion_count_threshold') != runtime['threshold']:
        raise ValueError('policy does not use the accepted training-fitted threshold')
    original_sources, original_write = C.M.sources, C.M.X.write_json

    def sources():
        result = original_sources()
        for name in ('semkine/xkf_motion_gate.py', 'semkine/xkf_stable_geometry.py',
                     'semkine/xkf_motion_gate_runtime.py', 'semkine/xkf_calibrated_motion_gate.py',
                     'tools/xkf_fusion/evaluate_calibrated_motion.py',
                     'tools/xkf_fusion/make_row_calibrated_motion.py'):
            result[name] = C.M.E.sha(ROOT / name)
        return result

    def write_json(path, value):
        if isinstance(value, dict) and 'policy' in value and 'input_sha256' in value:
            # Retain the same protocol object used by the producer's raw/final rows.
            value['motion_calibration'] = runtime
            value['motion_calibration_input_sha256'] = inputs
            value['stability_mode'] = 'per_sample_model'
        if isinstance(value, dict) and 'mac_profile_audit' in value:
            for name, digest in inputs.items():
                if C.M.E.sha(name) != digest:
                    raise ValueError('motion calibration input changed: ' + name)
            value['stability_mode'] = 'per_sample_model'
        return original_write(path, value)

    C.M.sources, C.M.X.write_json = sources, write_json
    C.XKFCentroidModel, C.CentroidFusionPolicy = XKFCalibratedMotionModel, CalibratedMotionCountPolicy
    sys.argv = [sys.argv[0], *remaining]
    C.main()


if __name__ == '__main__':
    main()
