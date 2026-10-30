"""Read-only two-window clips for S43 representation integration.

Selection is entirely inherited cache order plus a run-length guard. Initial
history is the old causal frozen-S43 prediction from a public fixed pose.
Targets are stored separately from every input window; a training loop injects
its own current (detached) hand prediction into each input window dynamically.
The old unique split manifest, source streams, labels and camera are untouched.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from tools.s43_br.cache_data import load_cache
from tools.s43_br.data import DATA_ROOT, SequenceReader


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/s43_integration_20261027'
DEFAULT_DIR = OUT / 'data'
DEFAULT_CLIPS = DEFAULT_DIR / 'clips.npz'
DEFAULT_SOURCE = ROOT / 'outputs/s43_br_20261027/data/existing_fit_select'
FIT = ('ch', 'lfz', 'lpc', 'lr')
SELECT = ('ycy', 'ylf')


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _source_paths(reader):
    return [Path(str(reader.base) + suffix) for suffix in
            ('_events.npy', '_offsets.npy', '_tsub.npy', '_aux.npz', '.meta')]


def prepare(out_dir=DEFAULT_DIR, source=DEFAULT_SOURCE, per_recording=4,
            root=DATA_ROOT):
    """Create fixed current+next W50 clips; refuse to overwrite artifacts."""
    out_dir, source, root = Path(out_dir), Path(source), Path(root)
    if int(per_recording) != per_recording or per_recording < 1:
        raise ValueError('positive integer per_recording required')
    arrays_path, metadata_path = out_dir / 'clips.npz', out_dir / 'metadata.json'
    if arrays_path.exists() or metadata_path.exists():
        raise FileExistsError(f'preserve existing clip cache: {out_dir}')
    source_meta = json.loads(source.with_suffix('.json').read_text())
    # Check scope before opening a cache which could contain report-only data.
    recordings = {s['recording'] for s in source_meta['samples']}
    if any(seq.startswith('zgz_') for seq in recordings):
        raise ValueError('zgz is outside integration train/SELECT clip preparation')
    if any(seq.split('_')[0] not in FIT + SELECT for seq in recordings):
        raise ValueError('source cache contains subjects outside the registered subset')
    protocol = OUT / 'protocol.json'
    if not protocol.exists():
        raise RuntimeError('freeze the integration protocol before preparing clips')
    started = time.perf_counter()
    samples = load_cache(source)
    grouped = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[sample['recording']].append((index, sample))
    readers = {}
    chosen, skipped, clips, windows = [], [], [], []
    source_files = {source.with_suffix('.npz'), source.with_suffix('.json'),
                    root / 'splits_semkine.json', Path(__file__).resolve(), protocol}
    input_checks = []
    for seq in sorted(grouped):
        reader = readers[seq] = SequenceReader(seq, root)
        source_files.update(_source_paths(reader))
        picked = 0
        for cache_index, sample in grouped[seq]:
            if picked >= per_recording:
                break
            end_ms, run_end = int(sample['end_ms']), int(sample['run_end_ms'])
            if end_ms + 50 >= run_end:
                skipped.append({'recording': seq, 'cache_index': cache_index,
                                'end_ms': end_ms, 'run_end_ms': run_end,
                                'reason': 'next end=end+50 is outside half-open valid run'})
                continue
            start_us = int(sample['start_us'])
            if start_us % 1000:
                raise ValueError('registered cache start is not an integer ms')
            start_ms = start_us // 1000
            current_check = reader.window(start_ms)
            if not (np.array_equal(current_check['events'], sample['events']) and
                    np.array_equal(current_check['source_ids'], sample['source_ids'])):
                raise ValueError('old current-window cache no longer matches its read-only source')
            if not np.array_equal(reader.betas, sample['betas']) or not np.array_equal(reader.K_cam, sample['K_cam']):
                raise ValueError('external beta/K changed since source cache')
            nxt = reader.window(start_ms + 50)
            clip_id = f'{seq}:run{int(sample["run_id"])}:start{start_ms}'
            partition = 'train' if seq.split('_')[0] in FIT else 'validation'
            if (partition == 'train') != (sample['partition'] == 'FIT7'):
                raise ValueError('inherited cache partition differs from registered FIT/SELECT subjects')
            metadata = {'clip_id': clip_id, 'recording': seq,
                        'subject': seq.split('_')[0], 'domain': sample['domain'],
                        'partition': partition, 'cache_index': cache_index,
                        'run_id': int(sample['run_id']),
                        'run_start_ms': int(sample['run_start_ms']), 'run_end_ms': run_end,
                        'end_ms': end_ms, 'next_end_ms': end_ms + 50,
                        'history_source': sample['history_source'],
                        'history_time_us': int(sample['prev_state_time_us']),
                        'camera_input_grid': [480, 640],
                        'polarity_contract': sample['polarity_contract'],
                        'time_contract': 'absolute input us; label target=cutoff-1000us; held history starts 1ms old',
                        'windows': []}
            for step, item in enumerate((sample, nxt)):
                wstart = start_us + step * 50000
                events = np.asarray(item['events'], np.int64).copy()
                events[:, 0] += wstart
                ids = np.asarray(item['source_ids'], np.int64).copy()
                if len(np.unique(ids)) != len(ids):
                    raise ValueError('source row IDs unexpectedly duplicate inside an unaugmented raw window')
                if len(events) and not np.all((events[:, 0] >= wstart) & (events[:, 0] < wstart + 50000)):
                    raise ValueError('event outside declared half-open window')
                windows.append({'events': events, 'source_ids': ids,
                                'start_us': wstart, 'cutoff_us': wstart + 50000,
                                'target_time_us': wstart + 49000})
                metadata['windows'].append({'seq_id': clip_id,
                        'segment_id': f'{seq}:run{int(sample["run_id"])}',
                        'step': step, 'start_us': wstart, 'cutoff_us': wstart + 50000,
                        'history_time_us': wstart - 1000,
                        'target_time_us': wstart + 49000,
                        'time_is_relative': False, 'valid': True})
            first_ids, next_ids = windows[-2]['source_ids'], windows[-1]['source_ids']
            if np.intersect1d(first_ids, next_ids).size:
                raise ValueError('adjacent clip windows overlap source rows')
            targets = np.stack((np.asarray(sample['target'], np.float32), reader.target(end_ms + 50)))
            clips.append({'metadata': metadata, 'history': np.asarray(sample['prev_state'], np.float32).copy(),
                          'betas': np.asarray(sample['betas'], np.float32).copy(),
                          'K_cam': np.asarray(sample['K_cam'], np.float32).copy(), 'targets': targets})
            chosen.append({'recording': seq, 'cache_index': cache_index,
                           'clip_id': clip_id, 'end_ms': end_ms,
                           'next_end_ms': end_ms + 50, 'run_end_ms': run_end,
                           'selection': 'first eligible inherited-cache order, no target criterion'})
            input_checks.append({'clip_id': clip_id, 'source_current_array_equal': True,
                                 'adjacent_source_disjoint': True, 'input_window_target_keys': [],
                                 'history_public_causal': 'public' in sample['history_source']})
            picked += 1
    if not clips:
        raise ValueError('no eligible two-window clips')
    lengths = np.asarray([len(w['events']) for w in windows], np.int64)
    ptr = np.r_[0, np.cumsum(lengths)]
    arrays = {'event_ptr': ptr,
              'events': np.concatenate([w['events'] for w in windows]),
              'source_ids': np.concatenate([w['source_ids'] for w in windows]),
              'histories': np.stack([c['history'] for c in clips]),
              'betas': np.stack([c['betas'] for c in clips]),
              'cameras': np.stack([c['K_cam'] for c in clips]),
              'targets': np.stack([c['targets'] for c in clips])}
    metadata = {'schema': 's43_integration_two_window_clips_v1',
                'created_at': datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                'array_file': arrays_path.name, 'input_source_cache': str(source),
                'window_bins': 50, 'clip_steps': 2,
                'per_recording_limit': int(per_recording),
                'n_clips': len(clips), 'n_windows': len(windows),
                'partitions': dict(Counter(c['metadata']['partition'] for c in clips)),
                'recordings': dict(Counter(c['metadata']['recording'] for c in clips)),
                'clips': [c['metadata'] for c in clips],
                'selected_ids': chosen, 'skipped': skipped,
                'input_checks': input_checks,
                'target_role': 'separate supervision/scoring arrays, never input-window construction',
                'history_role': 'initial public-cold causal frozen S43 history; next-window history supplied dynamically',
                'zgz_read': False, 'raw_data_split_labels_K_modified': False,
                'arrays': {k: {'shape': list(v.shape), 'dtype': str(v.dtype)} for k, v in arrays.items()},
                'source_files': {str(p): {'sha256': _sha(p), 'bytes': p.stat().st_size}
                                 for p in sorted(source_files)},
                'prepare_seconds_before_serialization': time.perf_counter() - started,
                'command': list(sys.argv)}
    out_dir.mkdir(parents=True, exist_ok=True)
    temporary = out_dir / '.clips.tmp.npz'
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, arrays_path)
    metadata['npz_sha256'] = _sha(arrays_path)
    metadata['total_seconds'] = time.perf_counter() - started
    temporary_meta = out_dir / '.metadata.tmp.json'
    temporary_meta.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=_json) + '\n')
    os.replace(temporary_meta, metadata_path)
    # Serialization and interface audit; no loss/error or class selection reads.
    loaded = load_clips(arrays_path)
    if len(loaded) != len(clips):
        raise AssertionError('clip cache changed row count')
    for index, (a, b) in enumerate(zip(clips, loaded)):
        for key in ('history', 'betas', 'K_cam', 'targets'):
            if not np.array_equal(a[key], b[key]):
                raise AssertionError(f'clip {index} {key} roundtrip mismatch')
        for step, w in enumerate(b['windows']):
            original = windows[index * 2 + step]
            if not np.array_equal(w['events'], original['events']) or not np.array_equal(w['source_ids'], original['source_ids']):
                raise AssertionError('ragged observation roundtrip mismatch')
            if 'target' in w or 'targets' in w or 'history' in w:
                raise AssertionError('input window contains target/fixed history')
    receipt = {'n_clips': len(clips), 'partitions': metadata['partitions'],
               'n_windows': len(windows), 'all_current_cache_source_equal': True,
               'all_adjacent_source_disjoint': True, 'roundtrip_exact': True,
               'targets_separate_from_inputs': True, 'history_dynamic_after_clip_start': True,
               'zgz_read': False, 'npz_sha256': metadata['npz_sha256'],
               'total_seconds': time.perf_counter() - started,
               'metadata': str(metadata_path), 'clips': str(arrays_path)}
    (out_dir / 'integrity.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def load_clips(path=DEFAULT_CLIPS, verify=True):
    """Return list of clips with targets separate from input-window dictionaries.

    ``path`` accepts either clips.npz or its containing directory. Inputs have
    no fixed history key: at every step the caller supplies its current causal
    detached hand estimate, enabling the identical online state transition.
    """
    path = Path(path)
    if path.is_dir():
        path = path / 'clips.npz'
    metadata_path = path.with_name('metadata.json')
    metadata = json.loads(metadata_path.read_text())
    if metadata.get('schema') != 's43_integration_two_window_clips_v1':
        raise ValueError('unsupported clip schema')
    if verify and _sha(path) != metadata['npz_sha256']:
        raise ValueError('clip array identity changed')
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    if any(a.dtype.hasobject for a in data.values()):
        raise ValueError('pickle/object arrays forbidden')
    n = len(metadata['clips'])
    ptr = data['event_ptr']
    if ptr.shape != (2 * n + 1,) or ptr[0] != 0 or np.any(np.diff(ptr) < 0) or ptr[-1] != len(data['events']):
        raise ValueError('invalid ragged window pointer')
    if len(data['source_ids']) != len(data['events']) or data['targets'].shape != (n, 2, 51):
        raise ValueError('observation/target shape mismatch')
    results = []
    for index, meta in enumerate(metadata['clips']):
        clip = {'clip_id': meta['clip_id'], 'recording': meta['recording'],
                'partition': meta['partition'], 'history': data['histories'][index].copy(),
                'betas': data['betas'][index].copy(), 'K_cam': data['cameras'][index].copy(),
                'targets': data['targets'][index].copy(), 'windows': [],
                'history_time_us': int(meta['history_time_us']),
                'domain': meta['domain'], 'metadata': dict(meta)}
        for step, wmeta in enumerate(meta['windows']):
            window_index = index * 2 + step
            lo, hi = map(int, ptr[window_index:window_index + 2])
            window = dict(wmeta)
            window.update(events=data['events'][lo:hi].copy(),
                          source_ids=data['source_ids'][lo:hi].copy(),
                          betas=clip['betas'].copy(), K_cam=clip['K_cam'].copy())
            clip['windows'].append(window)
        results.append(clip)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out-dir', type=Path, default=DEFAULT_DIR)
    p.add_argument('--source-cache', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--per-recording', type=int, default=4)
    p.add_argument('--root', type=Path, default=DATA_ROOT)
    args = parser.parse_args()
    print(json.dumps(prepare(args.out_dir, args.source_cache, args.per_recording, args.root),
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
