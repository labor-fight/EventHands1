#!/usr/bin/env python3
"""Unchanged debug G1--G4 formulas with an accepted paired-statistics policy."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from semkine.xkf_paired_statistics import XKFPairedStatisticsModel, PairedStatisticsPolicy
from tools.xkf_statistics.runtime import load_runtime
from tools.xkf_fusion import debug_gates_incremental as G


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args, unused = parser.parse_known_args()
    policy, evidence = load_runtime(args.runtime, args.seed, args.checkpoint, args.config)
    import json
    descriptor = json.loads(args.runtime.read_text())
    G.POLICY = policy
    G.XKFCentroidModel = XKFPairedStatisticsModel
    G.CentroidFusionPolicy = PairedStatisticsPolicy
    G.SOURCE_FILES += ('semkine/xkf_paired_statistics.py', 'tools/xkf_statistics/runtime.py',
                      'tools/xkf_statistics/gates.py')
    old_sources = G.sources
    G.sources = lambda: old_sources() | evidence
    sys.argv = [sys.argv[0], *unused, '--checkpoint', str(args.checkpoint),
                '--config', str(args.config), '--policy', descriptor['policy_path']]
    G.main()


if __name__ == '__main__':
    main()
