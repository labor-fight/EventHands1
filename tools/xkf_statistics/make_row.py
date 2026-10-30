#!/usr/bin/env python3
"""Full2590 stable row using accepted train-only paired statistics."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from semkine.xkf_paired_statistics import XKFPairedStatisticsModel, PairedStatisticsPolicy
from tools.xkf_statistics.runtime import load_runtime
from tools.xkf_fusion import make_row_incremental as M


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args, unused = parser.parse_known_args()
    policy, evidence = load_runtime(args.runtime, args.seed, args.checkpoint, args.config)
    M.POLICY = policy
    M.XKFStableMotionGateModel = XKFPairedStatisticsModel
    M.MotionCountFusionPolicy = PairedStatisticsPolicy
    M.SOURCE_FILES += ('semkine/xkf_paired_statistics.py', 'tools/xkf_statistics/runtime.py',
                      'tools/xkf_statistics/make_row.py')
    # Treat runtime/evidence as producer dependencies; old row's normal closure
    # guards then also recheck these at completion and retain them in the row.
    old_sources = M.sources
    M.sources = lambda: old_sources() | evidence
    sys.argv = [sys.argv[0], *unused, '--seed', str(args.seed), '--checkpoint', str(args.checkpoint), '--config', str(args.config)]
    M.main()


if __name__ == '__main__':
    main()
