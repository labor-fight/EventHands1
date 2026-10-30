#!/usr/bin/env python3
"""XKF formal row: `tools/xdl/make_row.py` (zgz val_core, rawW50 packets, rng0 noisy initialisation, own-loop latency)
with the long-window LNES attached by `tools/xkf/runtime.LongWindowEval` and `semkine.xkf.XKFModel`.

`--kmode` evaluates one checkpoint with fused / i / d / d_zero gains; the default output name gets a suffix for the
ablations so the fused row stays `outputs/semkine/<run>_main_row.json`."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / 'model')]

import numpy as np  # noqa: E402
import torch  # noqa: E402
from config import load_config  # noqa: E402
from mano_layer import ManoLayer  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine.dataset import sequences_for_split, splits_manifest  # noqa: E402
from semkine.xkf import KMODES, XKFModel  # noqa: E402
from tools import make_s36_row as C  # noqa: E402
from tools.xdl import make_row as X  # noqa: E402
from tools.xkf.runtime import LongWindowEval  # noqa: E402


def load_model(cfg, checkpoint, device, kmode):
    cfg['MODEL']['XKF_KMODE'] = kmode
    m = XKFModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location='cpu', strict=True)
    m.kmode = kmode
    return m.to(device).eval()


def runs_of(root, directory, seq):
    _, _, aux, _ = ET.load_sequence(root, directory, seq)
    return np.asarray(aux['valid_runs_ms'], np.int64).reshape(-1, 2)


@torch.no_grad()
def latency(m, cfg, device, window_ms):
    root = Path(cfg['DATA']['ROOT'])
    seq, d = next((s, d) for s, d in sequences_for_split(root, 'train', splits_manifest(cfg)) if s == 'lyq_local')
    with LongWindowEval(runs_of(root, d, seq), window_ms):
        packets = X.latency_packets(m, cfg, device)
    for p in packets[:100]:
        m.forward_packet(p)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    import time
    means = []
    for _ in range(3):
        start = time.perf_counter()
        for p in packets:
            m.forward_packet(p)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        means.append((time.perf_counter() - start) * 1000 / len(packets))
    return min(means), {'packets': 600, 'warm': 100, 'passes': 3, 'pass_mean_ms': means,
                        'context': 'own-loop staged packets with long-window LNES; forward_packet includes both '
                                   'branches, FK, DGP and fusion; LNES rasterisation and sensor pipeline excluded'}


def macs(cfg, m):
    from thop import profile
    from tools.xc.make_row import fixed_mac_fixture
    p, _ = fixed_mac_fixture()
    p.lnes = torch.zeros(1, m.render_h, m.render_w, 2)
    p.lnes_window_ms = torch.tensor([200])
    mm = XKFModel(cfg=cfg).float().eval()

    class Wrap(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, packet):
            return self.inner.forward_packet(packet)
    value, _ = profile(Wrap(mm), inputs=(p,), verbose=False)
    return float(value), sum(v.numel() for v in mm.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--config')
    ap.add_argument('--output')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--kmode', default='fused', choices=KMODES)
    ap.add_argument('--skip-latency', action='store_true')
    args = ap.parse_args()
    directory = REPO / 'outputs/semkine' / args.run
    cfg = load_config(args.config or directory / 'config_resolved.yaml')
    window_ms = int(cfg.get('XKF_DATA', {}).get('EVAL_WINDOW_MS', 200))
    checkpoint = X.exact_checkpoint(directory)
    device = torch.device(args.device)
    suffix = '' if args.kmode == 'fused' else f'_k{args.kmode}'
    output = Path(args.output or REPO / f'outputs/semkine/{args.run}{suffix}_main_row.json')
    mano = ManoLayer(cfg['MANO']['NPZ'], add_mean=False).to(device).eval()
    m = load_model(cfg, checkpoint, device, args.kmode)
    root = Path(cfg['DATA']['ROOT'])
    rng = np.random.default_rng(0)
    raw = []
    for seq, d in sequences_for_split(root, 'val_core', splits_manifest(cfg)):
        with LongWindowEval(runs_of(root, d, seq), window_ms):
            result = ET.track_sequence(m, mano, cfg, root, d, seq, 50, device, rng, 1., 1., None, window_ms=50)
        if result is None:
            raise RuntimeError(f'empty whole sequence result: {seq}')
        raw.append(result)
        X.write_json(str(output) + '.raw_accuracy.json', {'run': args.run, 'kmode': args.kmode,
                     'checkpoint': str(checkpoint), 'checkpoint_sha256': X.sha(checkpoint), 'raw_sequences': raw,
                     'whole_sequence_protocol': f'original rawW50/rng0 noisy initialization; direct branch W{window_ms} '
                                                'cut at run start', 'complete': len(raw) == 2,
                     'finite_subset_reduction': False})
    aggregate = X.accuracy_aggregate(raw)
    aggregate.update(step=6000, ckpt=str(checkpoint), checkpoint_sha256=X.sha(checkpoint))
    gains = m.gains(args.kmode).tolist()
    if args.skip_latency:
        lat, audit, anchor = float('nan'), {'skipped': True}, float('nan')
    else:
        lat, audit = latency(m, cfg, device, window_ms)
        anchor = C.latency_anchor(device)
    del m, mano
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    mac, params = macs(cfg, XKFModel(cfg=cfg))
    seed = int(cfg['SEED'])
    row = {'per_seed': {str(seed): aggregate}, 'two_seed_mean': {k: aggregate[k] for k in ['overall', 'local', 'global']},
           'latency_ms_scaled_full1p75': lat / anchor * C.FULL_PUBLISHED_MS if not args.skip_latency else None,
           'latency_ms_raw': lat if not args.skip_latency else None, 'anchor_full_raw_ms': anchor if not args.skip_latency else None,
           'macs_forward_packet': mac, 'params_total': params, 'latency_context_audit': audit,
           'xkf': {'kmode': args.kmode, 'gains': gains, 'eval_window_ms': window_ms},
           'protocol': {'split': 'val_core', 'seeds': [seed], 'n_seeds': 1, 'step_ms': 50, 'window_ms': 50,
                        'direct_window_ms': window_ms, 'exact_optimizer_step': 6000, 'model_class': 'XKFModel'},
           'finite_subset_reduction': False, 'numeric_status': 'PASS',
           'provenance': {'config_sha256': X.sha(args.config or directory / 'config_resolved.yaml'),
                          'checkpoint_sha256': X.sha(checkpoint), 'producer_sha256': X.sha(__file__)}}
    X.write_json(output, row)
    print(json.dumps({'output': str(output), 'kmode': args.kmode, 'numeric_status': 'PASS'}), flush=True)


if __name__ == '__main__':
    main()
