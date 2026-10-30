#!/usr/bin/env python3
"""Whole-sequence diagnostic row for the frozen step2000 XKF debug weights.

This never writes a formal xkf_s<seed> row or changes any readiness gate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

import numpy as np
import torch
from config import load_config
from mano_layer import ManoLayer
from semkine import eval_track as ET
from semkine.dataset import sequences_for_split, splits_manifest
from tools import make_s36_row as C
from tools.xdl import make_row as X
from tools.xkf import make_row as F
from tools.xkf.runtime import LongWindowEval


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', required=True, type=int, choices=[3407, 3408, 3409])
    ap.add_argument('--debug-root', default='outputs/xkf/debug_20261015_r1')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root_debug = REPO / args.debug_root
    frozen = root_debug / 'frozen_manifest.json'
    manifest = json.loads(frozen.read_text())
    mismatches = [p for p, h in manifest['source_sha256'].items() if X.sha(REPO / p) != h]
    if mismatches:
        raise RuntimeError(f'frozen source mismatch: {mismatches}')
    run = f'xkf_debug2000_s{args.seed}'
    directory = root_debug / f'xkf_s{args.seed}'
    config = directory / 'config_resolved.yaml'
    ckpt = directory / f'xkf_s{args.seed}-step=2000.ckpt'
    saved = torch.load(ckpt, map_location='cpu', weights_only=False)
    if int(saved.get('global_step', -1)) != 2000:
        raise RuntimeError('exact step2000 checkpoint required')
    del saved
    cfg = load_config(config)
    if int(cfg['SEED']) != args.seed:
        raise RuntimeError('seed identity mismatch')
    output = REPO / f'outputs/semkine/{run}_main_row.json'
    if output.exists():
        raise FileExistsError(output)
    scope = 'nonformal frozen debug step2000, full2590 own-loop evaluation; no checkpoint selection, no gate revision, no full-training readiness claim'
    protocol = {'scope': scope, 'split': 'val_core', 'seeds': [args.seed], 'n_seeds': 1,
                'step_ms': 50, 'window_ms': 50, 'direct_window_ms': 200,
                'exact_optimizer_step': 2000, 'model_class': 'XKFModel',
                'formal': False, 'noise_rng_seed': 0, 'noise_scale': 1.0,
                'gain_mode': 'fused', 'inference_targets': 'zero', 'tf32': False}
    provenance = {'checkpoint_sha256': X.sha(ckpt), 'config_sha256': X.sha(config),
                  'frozen_manifest_sha256': X.sha(frozen), 'producer_sha256': X.sha(__file__)}
    X.write_json(str(output) + '.protocol.json', {'protocol': protocol, 'provenance': provenance})
    device = torch.device(args.device)
    m = F.load_model(cfg, ckpt, device, 'fused')
    mano = ManoLayer(cfg['MANO']['NPZ'], add_mean=False).to(device).eval()
    root_data = Path(cfg['DATA']['ROOT'])
    rng = np.random.default_rng(0)
    raw = []
    for seq, d in sequences_for_split(root_data, 'val_core', splits_manifest(cfg)):
        with LongWindowEval(F.runs_of(root_data, d, seq), 200):
            result = ET.track_sequence(m, mano, cfg, root_data, d, seq, 50, device,
                                       rng, 1., 1., None, window_ms=50)
        if result is None:
            raise RuntimeError(f'empty sequence: {seq}')
        raw.append(result)
        X.write_json(str(output) + '.raw_accuracy.json', {
            'run': run, 'protocol': protocol, 'provenance': provenance,
            'raw_sequences': raw, 'complete': len(raw) == 2, 'finite_subset_reduction': False})
        print(json.dumps({'sequence': seq, 'frames': result['n_frames']}), flush=True)
    aggregate = X.accuracy_aggregate(raw)
    aggregate.update(step=2000, ckpt=str(ckpt), checkpoint_sha256=X.sha(ckpt))
    latency, audit = F.latency(m, cfg, device, 200)
    gains = m.gains('fused').tolist()
    mac, params = F.macs(cfg, m)
    del m, mano
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    anchor = C.latency_anchor(device)
    row = {'per_seed': {str(args.seed): aggregate},
           'two_seed_mean': {k: aggregate[k] for k in ['overall', 'local', 'global']},
           'latency_ms_scaled_full1p75': latency / anchor * C.FULL_PUBLISHED_MS,
           'latency_ms_raw': latency, 'anchor_full_raw_ms': anchor,
           'macs_forward_packet': mac, 'params_total': params,
           'macs_forward_packet_note': 'thop standard-layer MAC lower bound; native FK/DGP/scatter arithmetic excluded',
           'latency_context_audit': audit, 'xkf': {'kmode': 'fused', 'gains': gains, 'eval_window_ms': 200},
           'protocol': protocol, 'provenance': provenance,
           'finite_subset_reduction': False, 'numeric_status': 'PASS'}
    X.write_json(output, row)
    print(json.dumps({'output': str(output), 'formal': False, 'step': 2000}), flush=True)


if __name__ == '__main__':
    main()
