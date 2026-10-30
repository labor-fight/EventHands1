"""Actual FIT events: offline count constructor vs online source-ID ingest.

This is a contract check, not a scientific readout or accuracy evaluation.
Target fields are never accessed. Source events remain on the native grid until
the shared pixel-centre reduction. Complete-bin prefixes contain no future data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .cache_data import load_cache
from .carrier import Event, EvidenceMemory, GridKernel, encode_batch
from .data import counts_from_events
from .geometry import candidate_paths, map_pixels, render_paths
from .representations import SENSOR


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / 'outputs/s43_br_20261027/data/existing_fit_select'
DEFAULT_OUT = ROOT / 'outputs/s43_br_20261027/existing_contracts.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(cache=DEFAULT_CACHE, output=DEFAULT_OUT):
    output = Path(output)
    if output.exists():
        raise FileExistsError('preserve existing contract receipt: ' + str(output))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    samples = [s for s in load_cache(cache) if s['partition'] == 'FIT7'][:2]
    if len(samples) != 2:
        raise ValueError('need the first two FIT samples from the fixed cache')
    files = [Path(__file__), ROOT / 'tools/s43_br/carrier.py',
             ROOT / 'tools/s43_br/geometry.py', ROOT / 'tools/s43_br/data.py',
             ROOT / 'tools/s43_br/cache_data.py', ROOT / 'tools/s43_br/representations.py',
             Path(cache).with_suffix('.npz'), Path(cache).with_suffix('.json')]
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in files}
    before = time.perf_counter()
    reports = []
    passed = True
    for sample in samples:
        start = time.perf_counter()
        native = np.asarray(sample['events'])
        ids = np.asarray(sample['source_ids'])
        if len(ids) != len(native) or len(np.unique(ids)) != len(ids):
            raise ValueError('source IDs do not uniquely identify original event rows')
        stamps = native[:, 0].astype(np.int64)
        if np.any(stamps < 0) or np.any(stamps >= 50_000):
            raise ValueError('native event outside declared causal window')
        x, y = map_pixels(native[:, 1], native[:, 2], (24, 32))
        px = y * 32 + x
        low = counts_from_events(native, (24, 32), cap=SENSOR['cap'])
        # Only historical state, external beta/K and fixed public candidate rules.
        states = candidate_paths(sample['prev_state'], amplitudes=(-.015, 0., .015), axis='tx')
        camera = sample['K_cam'].copy()
        camera[:2] *= .375
        paths = render_paths(states, sample['betas'], camera,
                             image_size=(24, 32), texture_seed=None)
        kernel = GridKernel(paths, polarity_convention='stored_mapping_unverified_assumed_0_ON_1_OFF',
                            context_id=sample['recording'] + '/contract', **SENSOR)
        z = EvidenceMemory(kernel, max_sources=100_000)
        row = {'recording': sample['recording'], 'start_us': sample['start_us'],
               'native_events': len(native), 'unique_sources': len(np.unique(ids)),
               'source_cap': z.max_sources, 'target_used': False,
               'candidate_source': 'fixed root-x offsets around frozen causal S43 history',
               'physical_polarity_calibrated': False, 'prefixes': []}
        if len(ids) > z.max_sources:
            # This is an explicit supported-domain boundary, never silent drop.
            row.update(status='unsupported_source_cap', elapsed_seconds=time.perf_counter()-start)
            reports.append(row)
            passed = False
            continue
        previous = 0
        for bins in (1, 10, 25, 50):
            mask = (stamps >= previous * 1000) & (stamps < bins * 1000)
            events = [Event(int(uid), int(t), int(p), int(pol))
                      for uid, t, p, pol in zip(ids[mask], stamps[mask], px[mask], native[mask, 3])]
            # Use two packet sizes, keeping the watermark behind staged events.
            for offset in range(0, len(events), 137):
                z.ingest(events[offset:offset + 137])
            z.advance(bins * 1000, watermark_us=bins * 1000)
            digest = z.digest()
            online = z.query()
            if z.digest() != digest:
                raise AssertionError('query mutated an actual-data prefix')
            offline = encode_batch(paths[..., :bins + 1], low[:bins],
                polarity_convention=kernel.polarity_convention,
                context_id=kernel.context_id, **SENSOR)
            if not np.array_equal(z.counts[:bins], low[:bins]):
                raise AssertionError('native source-ID ingestion changed pixel/bin multiplicity')
            errors = {}
            for field in ('ell', 'log_likelihood', 'phi', 'R', 'posterior'):
                a, b = online[field], offline[field]
                if not (np.isfinite(a).all() and np.isfinite(b).all()):
                    raise AssertionError('nonfinite contract state: ' + field)
                errors[field] = float(np.max(np.abs(a-b)))
                np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
            expected_sources = int(np.count_nonzero(stamps < bins * 1000))
            if len(z.seen) != expected_sources:
                raise AssertionError('real repeated events were deleted')
            row['prefixes'].append({'bins': bins, 'cutoff_us': bins*1000,
                'count_exact': True, 'source_count': len(z.seen),
                'query_readonly': True, 'errors': errors})
            previous = bins
        row.update(status='passed', elapsed_seconds=time.perf_counter()-start,
                   memory=z.memory_bytes())
        reports.append(row)
    result = {'kind': 'actual FIT source/time/constructor contract, no task readout',
              'passed': passed, 'samples': reports, 'source_hashes_before_check': source_hashes,
              'source_hashes_unchanged': all(sha(ROOT/p) == h for p,h in source_hashes.items()),
              'GT_target_used': False, 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'elapsed_seconds': time.perf_counter()-before}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if not result['passed']:
        raise RuntimeError('actual-data constructor not supported for all fixed samples; see ' + str(output))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = verify(args.cache, args.out)
    print(json.dumps({'passed': result['passed'], 'samples': len(result['samples']),
                      'elapsed_seconds': result['elapsed_seconds'], 'out': str(args.out)}))


if __name__ == '__main__':
    main()
