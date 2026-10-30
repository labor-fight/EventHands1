#!/usr/bin/env python3
"""Freeze an additive debug revision without rewriting the original XKF protocol."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.xkf.freeze import TRAIN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'protocol.json'
    if path.exists():
        raise SystemExit(f'Refusing to replace frozen protocol: {path}')
    sources = sorted(set(TRAIN) | {
        'configs/xkf/xkf_s3409.yaml', 'tools/xkf/launch.sh', 'tools/xkf/freeze_debug.py',
        'tools/xkf/runtime.py', 'tools/xkf/make_row.py', 'tools/xkf/debug_eval.py',
        'tests/test_xkf_data_debug.py'})
    hashes = {p: hashlib.sha256((REPO / p).read_bytes()).hexdigest() for p in sources}
    original = REPO / 'outputs/xkf/protocol.json'
    from tools.xkf.debug_eval import DEBUG_CRITERIA
    protocol = {
        'study': 'XKF-20261015-debug-before-full',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'authorization': 'User 2026-10-15: execute on GPU2-7, six cards; debug until effect is good before full training.',
        'supersedes_execution_only': str(original.relative_to(REPO)),
        'original_protocol_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'formal_mechanism_gates': json.loads(original.read_text())['gates'],
        'allocation': {'3407': [2, 3], '3408': [4, 5], '3409': [6, 7]},
        'training_contract': 'Each arm uses two ranks x 512, no accumulation, original nine identities/72 sequences. Debug endpoints are nonformal; full runs restart fresh and stop exactly at step6000.',
        'debug_plan': 'First 200 steps per seed; assess matched initial/final checkpoints, then extend debug if needed. No automatic full run on mere process success.',
        'debug_readiness_criteria': DEBUG_CRITERIA,
        'formal_start_rule': 'All three seeds must pass the learning gate, and the implementation hashes must match. Formal G1-G6 remain unassessed until full evaluation.',
        'repairs_before_debug': [
            'Implement planned heatmap Gaussian KL (weight30, sigma6px) and root-relative FK joint MSE (weight30000); retain keypoint L1 weight300.',
            'Pure D evaluation returns D without the fused 60-degree innovation clip.',
            'After self curriculum starts, moments only use nonempty self rows, with zero-weight distributed participation.',
            'DGP exposes conditional solve covariance; fusion still uses constant empirical residual moments.',
            'Long-window D reuses augmented overlapping K4 packets; only older history uses new RNG streams.',
            'Incremental evaluation timestamp surface resets on run/sequence changes and matches build_lnes.',
        ],
        'stage0_warning_retained': 'Original F1 local 2px geometric depth RMSE30.13mm failed <=20mm. Do not relabel it PASS; learned keypoint accuracy and actual measurement efficacy must be checked.',
        'source_sha256': hashes,
    }
    path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + '\n')
    manifest = {'status': 'debug_revision_frozen', 'protocol_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_sha256': hashes}
    (out / 'frozen_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for source in sources:
        destination = out / 'source_snapshot' / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / source, destination)
    print(json.dumps({'protocol': str(path), 'files': len(sources)}))


if __name__ == '__main__':
    main()
