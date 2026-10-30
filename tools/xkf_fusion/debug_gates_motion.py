#!/usr/bin/env python3
"""Unchanged original debug gate formulas for stable, motion-gated fusion."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from semkine.xkf_motion_gate import MotionCountFusionPolicy
from semkine.xkf_motion_gate_runtime import XKFStableMotionGateModel
from tools.xkf_fusion import debug_gates as G


def main():
    original_sources = G.sources

    def sources():
        result = original_sources()
        for name in ('semkine/xkf_motion_gate.py', 'semkine/xkf_stable_geometry.py',
                     'semkine/xkf_motion_gate_runtime.py',
                     'tools/xkf_fusion/debug_gates_motion.py'):
            result[name] = G.E.sha(ROOT / name)
        return result

    G.sources = sources
    G.XKFCentroidModel = XKFStableMotionGateModel
    G.CentroidFusionPolicy = MotionCountFusionPolicy
    G.main()


if __name__ == '__main__':
    main()
