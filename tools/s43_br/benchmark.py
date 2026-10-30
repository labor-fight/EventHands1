"""Fixed-window CPU component costs, independent of scientific scores.

The first two FIT windows are repeated three times. Targets are never accessed.
These are replay/component timings, NOT end-to-end deployment latency: cached
frozen-S43 history, shared readout context and trained readout are excluded.
Old feature timing arrays remain untouched.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from .cache_data import load_cache
from .carrier import Event, EvidenceMemory, GridKernel, _cached_count_pmf
from .data import counts_from_events
from .geometry import candidate_paths, fk_geometry, map_pixels, render_paths
from .representations import SENSOR, pixel_features, probabilities


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/s43_br_20261027'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timed(function):
    start = time.perf_counter()
    result = function()
    return result, time.perf_counter() - start


def lnes_only(counts):
    a = np.minimum(counts, SENSOR['cap'])
    ticks = np.arange(len(a))[:, None, None]
    return (np.where(a > 0, ticks, 0).max(0) / float(len(a))).astype(np.float32)


def native_adapter(sample):
    native = np.asarray(sample['events'])
    x, y = map_pixels(native[:, 1], native[:, 2], (24, 32))
    return [Event(int(uid), int(row[0]), int(py * 32 + px), int(row[3]))
            for uid, row, px, py in zip(sample['source_ids'], native, x, y)]


def query_arrays_bytes(result):
    return int(sum(v.nbytes for v in result.values() if isinstance(v, np.ndarray)))


def measure_bank(paths, events, counts, *, copies, flip_prior, temperature):
    clock_start = time.perf_counter()
    if copies == 1:
        bank = paths
        omega = np.ones((3, 3)) / 3.
    elif copies == 2:
        bank = np.concatenate([paths, -paths], axis=1)
        omega = np.tile(np.r_[np.full(3, (1. - flip_prior) / 3.),
                             np.full(3, flip_prior / 3.)], (3, 1))
    else:
        raise ValueError('only actual M3/M6 implementations are benchmarked')
    bank_prepare_seconds = time.perf_counter() - clock_start
    start = time.perf_counter()
    kernel = GridKernel(bank,
                        polarity_convention='stored polarity unverified; assumed model only',
                        context_id='fixed-FIT-cost-check', **SENSOR)
    memory = EvidenceMemory(kernel, omega=omega, max_sources=100_000)
    init_seconds = time.perf_counter() - start
    if len(events) > memory.max_sources:
        raise ValueError('fixed window exceeds explicit source cap')
    start = time.perf_counter()
    for offset in range(0, len(events), 137):
        memory.ingest(events[offset:offset + 137])
    ingest_seconds = time.perf_counter() - start
    start = time.perf_counter()
    memory.advance(50_000, watermark_us=50_000)
    advance_seconds = time.perf_counter() - start
    start = time.perf_counter()
    query = memory.query()
    query_seconds = time.perf_counter() - start
    start = time.perf_counter()
    # This is exactly the locked confidence transformation. No label or target.
    calibrated = probabilities(query['log_likelihood'] / temperature)
    calibration_seconds = time.perf_counter() - start
    if not np.isfinite(calibrated).all():
        raise AssertionError('cost measurement yielded nonfinite support')
    if not np.array_equal(memory.counts, counts):
        raise AssertionError('native ingestion disagrees with the shared count constructor')
    if len(memory.seen) != len(events):
        raise AssertionError('native simultaneous source multiplicity was deleted')
    resident = memory.memory_bytes()
    return {'M': kernel.M, 'zero_weight_branches_still_computed': copies == 2 and flip_prior in (0., 1.),
            'bank_prepare_seconds': bank_prepare_seconds,
            'kernel_memory_init_seconds': init_seconds,
            'native_source_ingest_seconds': ingest_seconds,
            'advance_50_bins_seconds': advance_seconds,
            'owned_query_seconds': query_seconds,
            'locked_calibration_readout_seconds': calibration_seconds,
            'resident': resident,
            'owned_query_array_bytes': query_arrays_bytes(query),
            'external_bank_input_bytes': int(bank.nbytes),
            'external_original_paths_also_live_bytes': int(paths.nbytes) if copies == 2 else 0,
            'source_packet_size': 137, 'unique_sources': len(memory.seen),
            'source_cap': memory.max_sources,
            'phase_and_measurement_scores_not_exported': True}


def summarize(records, key):
    values = np.asarray([record[key] for record in records], float)
    return {'median': float(np.median(values)), 'min': float(values.min()),
            'max': float(values.max()), 'repetitions': len(values)}


def run(cache, output, repeats=3):
    if repeats < 3:
        raise ValueError('at least three repetitions required')
    cache, output = Path(cache), Path(output)
    if output.exists():
        raise FileExistsError('preserve existing benchmark: ' + str(output))
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(1)
    samples = [s for s in load_cache(cache) if s['partition'] == 'FIT7'][:2]
    if len(samples) != 2:
        raise ValueError('need the first two fixed FIT windows')
    lock_path, calibration_path = OUT / 'final_lock.json', OUT / 'calibration_v2.json'
    lock = json.loads(lock_path.read_text())
    if lock['calibration_sha256'] != sha(calibration_path):
        raise RuntimeError('calibration differs from final lock')
    weight = float(lock['selected']['flip_prior'])
    temperature = float(lock['selected']['temperature'])
    files = list((ROOT / 'tools/s43_br').glob('*.py')) + [lock_path, calibration_path,
              cache.with_suffix('.npz'), cache.with_suffix('.json'), ROOT / 'assets/mano_right.npz']
    sources = {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}
    # Explicit one-time module/asset warmup, held out from steady-state medians.
    _, warmup_seconds = timed(lambda: fk_geometry(samples[0]['prev_state'], samples[0]['betas']))
    pmf_before = _cached_count_pmf.cache_info()._asdict()
    all_records = []
    wall_start = time.perf_counter()
    for sample_index, sample in enumerate(samples):
        for repetition in range(repeats):
            repetition_start = time.perf_counter()
            counts, count_seconds = timed(lambda: counts_from_events(sample['events'], (24, 32), cap=8))
            lnes, lnes_seconds = timed(lambda: lnes_only(counts))
            enhanced, enhanced_seconds = timed(lambda: pixel_features(counts)[1])
            events, adapter_seconds = timed(lambda: native_adapter(sample))
            states, candidate_seconds = timed(lambda: candidate_paths(sample['prev_state'],
                                                        amplitudes=(-.015, 0., .015), axis='tx'))
            camera = sample['K_cam'].copy()
            camera[:2] *= .375
            paths, render_seconds = timed(lambda: render_paths(states, sample['betas'], camera,
                                                       image_size=(24, 32), texture_seed=None))
            banks = {}
            for name, copies in [('M3', 1), ('M6', 2)]:
                bank = measure_bank(paths, events, counts, copies=copies,
                                    flip_prior=weight, temperature=temperature)
                compute = sum(bank[k] for k in ('bank_prepare_seconds', 'kernel_memory_init_seconds',
                              'advance_50_bins_seconds', 'owned_query_seconds', 'locked_calibration_readout_seconds'))
                bank['offline_constructor_components_total_seconds'] = candidate_seconds + render_seconds + count_seconds + compute
                bank['online_source_components_total_seconds'] = (candidate_seconds + render_seconds + adapter_seconds +
                                               bank['native_source_ingest_seconds'] + compute)
                banks[name] = bank
            actual_wall = time.perf_counter() - repetition_start
            # Validation outside the recorded component times.
            np.testing.assert_array_equal(lnes, pixel_features(counts)[0])
            all_records.append({'sample_index': sample_index, 'recording': sample['recording'],
                'start_us': int(sample['start_us']), 'repetition': repetition + 1,
                'native_events': len(sample['events']), 'target_accessed': False,
                'count_aggregation_seconds': count_seconds,
                'native_Event_adapter_seconds': adapter_seconds,
                'lnes_only_seconds': lnes_seconds, 'enhanced_constructor_seconds': enhanced_seconds,
                'lnes_pixel_representation_total_seconds': count_seconds + lnes_seconds,
                'enhanced_pixel_representation_total_seconds': count_seconds + enhanced_seconds,
                'candidate_generation_seconds': candidate_seconds,
                'MANO_radiance_render_seconds': render_seconds,
                'actual_wall_all_measured_components_seconds': actual_wall,
                'external_shared_array_bytes': {'native_events': int(sample['events'].nbytes),
                    'native_source_ids': int(sample['source_ids'].nbytes), 'counts': int(counts.nbytes),
                    'lnes': int(lnes.nbytes), 'enhanced': int(enhanced.nbytes),
                    'candidate_states': int(states.nbytes)}, 'banks': banks})
    pixel_keys = ('count_aggregation_seconds', 'native_Event_adapter_seconds', 'lnes_only_seconds',
                  'enhanced_constructor_seconds', 'lnes_pixel_representation_total_seconds',
                  'enhanced_pixel_representation_total_seconds', 'candidate_generation_seconds',
                  'MANO_radiance_render_seconds', 'actual_wall_all_measured_components_seconds')
    bank_keys = ('bank_prepare_seconds', 'kernel_memory_init_seconds', 'native_source_ingest_seconds',
                 'advance_50_bins_seconds', 'owned_query_seconds', 'locked_calibration_readout_seconds',
                 'offline_constructor_components_total_seconds', 'online_source_components_total_seconds')
    result = {'kind': 'fixed FIT single-thread CPU component replay benchmark',
        'at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        'samples': len(samples), 'repeats_per_sample': repeats, 'records': all_records,
        'summary_all_fixed_repetitions': {k: summarize(all_records, k) for k in pixel_keys},
        'bank_summary': {name: {k: summarize([r['banks'][name] for r in all_records], k) for k in bank_keys}
                         for name in ('M3', 'M6')},
        'module_asset_warmup_seconds_excluded_from_medians': warmup_seconds,
        'pmf_cache_before': pmf_before, 'pmf_cache_after': _cached_count_pmf.cache_info()._asdict(),
        'locked_flip_prior': weight, 'locked_temperature': temperature,
        'torch_threads': torch.get_num_threads(), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'scientific_scores_hyperparameters_or_old_arrays_modified': False,
        'target_accessed': False, 'source_sha256_before': sources,
        'source_sha256_unchanged': all(sha(ROOT / p) == v for p,v in sources.items()),
        'total_benchmark_wall_seconds': time.perf_counter() - wall_start,
        'scope': [
            'No frozen-S43 history warmup inference, common readout context construction or trained readout prediction is measured.',
            'Pixel-arm totals are representation-only. Bank totals include public candidate/radiance work and are component sums, not end-to-end sensor-to-mesh latency.',
            'Both actual M6 orientations are computed even when locked flip prior is zero; no hypothetical M3 speedup substituted.',
            'Query returns owned full phase diagnostics; its measured cost includes their copying.',
            'Resident memory reports kernel/source ownership plus separately listed arrays; not complete Python-process RSS or peak memory.',
            'The external Event adapter object-list Python overhead and shared MANO module storage are not included in resident-array totals.',
            'One-time MANO asset warmup is separate; all three repeated timings, including first-cache effects, are retained.',
            'Correction feature query_seconds remains the old M3 reference; this artifact is the separate fresh M3/M6 measurement.'
        ]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, default=OUT/'data/existing_fit_select')
    p.add_argument('--out', type=Path, default=OUT/'benchmark.json')
    p.add_argument('--repeats', type=int, default=3)
    args = p.parse_args()
    result = run(args.cache, args.out, args.repeats)
    print(json.dumps({'out': str(args.out), 'samples': result['samples'],
        'repeats': result['repeats_per_sample'],
        'sources_unchanged': result['source_sha256_unchanged'],
        'M6_query_median_seconds': result['bank_summary']['M6']['owned_query_seconds']['median']}))


if __name__ == '__main__':
    main()
