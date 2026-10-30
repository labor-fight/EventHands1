#!/usr/bin/env python3
"""CPU-only event-age accounting paired with the measured software latency.

This is a timestamp-derived offline estimate at scheduled window closure, not a
measurement of hardware transport, real-time backlog, or display latency.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from config import load_config
from tools.xdl import diagnostics as D


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only postprocessing requires CUDA_VISIBLE_DEVICES empty')
    out = ROOT / 'outputs/xde/pipeline'
    raw = torch.load(out / 'raw.pt', map_location='cpu', weights_only=False)
    bank, banksha = D.load_bank()
    cfg = load_config(raw['identity']['config'])
    results = {}
    resolution = .25
    edges = np.arange(0, 200 + resolution, resolution)
    start = time.perf_counter()
    for seq in raw['sequences']:
        name = seq['sequence']
        data = D.load_data(cfg, name, bank['sequences'][name])
        hist = np.zeros(len(edges)-1, np.int64)
        age_hist = hist.copy()
        total_age = total_with_compute = 0.
        counts = 0
        for row, record in zip(bank['sequences'][name]['rows'], seq['records']):
            assert row['end_ms'] == record['end_ms']
            events = D.raw_events(data, row)
            assert len(events) == record['n_events']
            age = 50. - events[:, 3].astype(np.float64) * 1000.
            end_to_host = age + record['total_ms']
            assert np.all((age > -.001) & (age <= 50.001))
            assert np.all((end_to_host >= 0) & (end_to_host < edges[-1]))
            hist += np.histogram(end_to_host, edges)[0]
            age_hist += np.histogram(age, edges)[0]
            counts += len(age)
            total_age += age.sum()
            total_with_compute += end_to_host.sum()
        def quantiles(h):
            cs = np.cumsum(h)
            return {f'p{p}_ms': float(edges[min(np.searchsorted(cs, counts*p/100), len(h)-1)] + resolution/2)
                    for p in (50, 95, 99)}
        assert counts == int(hist.sum()) == int(age_hist.sum())
        results[name] = {'events': counts, 'queries': len(seq['records']),
            'at_window_close': {'mean_ms': total_age/counts, **quantiles(age_hist)},
            'at_host_output_offline_estimate': {'mean_ms': total_with_compute/counts, **quantiles(hist)},
            'event_rate_per_valid_window_second': counts / (len(seq['records'])*.05),
            'clock_assumption': 'Window closes at end_ms+1ms; no acquisition or queue delay; each event weighted once'}
    payload = {'status': 'complete', 'results': results,
        'quantile_histogram_resolution_ms': resolution, 'quantile_max_rounding_error_ms': resolution/2,
        'scope': 'offline timestamp estimate; does not measure sensor-to-output latency',
        'inputs_sha256': {'raw.pt': sha(out/'raw.pt'), 'packet_bank.json': banksha,
                          'result.json': sha(out/'result.json'), 'script': sha(__file__)},
        'cpu_seconds': time.perf_counter()-start, 'gpu_hours': 0.}
    (out/'event_age.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps(results, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
