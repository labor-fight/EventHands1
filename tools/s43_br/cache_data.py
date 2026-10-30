"""Generate/load a pickle-free small existing-data snapshot for S43 B-R.

This command computes only the shared causal history and caches inputs/targets;
it does not evaluate a representation or inspect target errors. Source data,
the frozen checkpoint and the unique split manifest are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.s43_br.data import (CHECKPOINT, CONFIG, DATA_ROOT, FrozenS43History,
                               load_existing_samples, recording_dirs, sha256)

ARRAY_FIELDS = ('prev_state', 'prev_older_state', 'history_predictions', 'target',
                'betas', 'K_cam', 'camera_K', 'history_initial_state')


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def save_cache(prefix: Path, samples: list, provenance: dict) -> dict:
    """Write arrays and metadata separately. Existing artifacts are preserved."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = prefix.with_suffix('.npz'), prefix.with_suffix('.json')
    if npz_path.exists() or json_path.exists():
        raise FileExistsError(f'preserve existing snapshot: {prefix}')
    lengths = np.asarray([len(s['events']) for s in samples], np.int64)
    ptr = np.r_[0, np.cumsum(lengths)]
    arrays = {
        'event_ptr': ptr,
        'events': np.concatenate([s['events'] for s in samples], axis=0),
        'source_ids': np.concatenate([s['source_ids'] for s in samples], axis=0),
    }
    arrays.update({key: np.stack([s[key] for s in samples]) for key in ARRAY_FIELDS})
    metadata = [{k: v for k, v in s.items()
                 if k not in ARRAY_FIELDS and k not in ('events', 'source_ids')}
                for s in samples]
    temp_npz = npz_path.with_name(npz_path.name + '.tmp.npz')
    np.savez_compressed(temp_npz, **arrays)
    os.replace(temp_npz, npz_path)
    receipt = {'schema': 's43_br_existing_cache_v1', 'arrays_file': npz_path.name,
               'array_fields': list(ARRAY_FIELDS), 'samples': metadata,
               'n_samples': len(samples), 'npz_sha256': sha256(npz_path),
               'partitions': dict(Counter(s['partition'] for s in samples)),
               'provenance': provenance}
    temp_json = json_path.with_name(json_path.name + '.tmp')
    temp_json.write_text(json.dumps(receipt, indent=2, ensure_ascii=False,
                                    default=_json_value) + '\n')
    os.replace(temp_json, json_path)
    return {'npz': str(npz_path), 'json': str(json_path),
            'n_samples': len(samples), 'partitions': receipt['partitions'],
            'npz_sha256': receipt['npz_sha256']}


def load_cache(prefix: Path, verify: bool = True) -> list:
    """Return the same sample dictionaries as data.load_existing_samples()."""
    prefix = Path(prefix)
    meta = json.loads(prefix.with_suffix('.json').read_text())
    path = prefix.with_suffix('.npz')
    if verify and sha256(path) != meta['npz_sha256']:
        raise ValueError('cache content changed')
    with np.load(path, allow_pickle=False) as z:
        # Compressed NPZ arrays must be decoded once. Repeated z['events']
        # inside the sample loop would decompress the complete ragged stream
        # for every sample, obscuring prototype construction costs.
        arrays = {key: z[key] for key in z.files}
        if any(array.dtype.hasobject for array in arrays.values()):
            raise ValueError('object/pickle arrays forbidden')
        ptr = arrays['event_ptr']
        result = []
        for i, entry in enumerate(meta['samples']):
            s = dict(entry)
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            s['events'] = arrays['events'][lo:hi].copy()
            s['source_ids'] = arrays['source_ids'][lo:hi].copy()
            for key in meta['array_fields']:
                s[key] = arrays[key][i].copy()
            result.append(s)
    return result


def source_identities(samples: list, root: Path, checkpoint: Path,
                      config: Path) -> dict:
    paths = {Path(root) / 'splits_semkine.json', Path(checkpoint), Path(config),
             REPO / 'tools/s43_br/data.py', Path(__file__).resolve()}
    dirs = recording_dirs(root)
    for seq in sorted({s['recording'] for s in samples}):
        base = Path(root) / dirs[seq] / seq
        paths.update(Path(str(base) + suffix) for suffix in
                     ('_events.npy', '_offsets.npy', '_tsub.npy', '_aux.npz', '.meta'))
    return {str(path): {'sha256': sha256(path), 'bytes': path.stat().st_size}
            for path in sorted(paths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-prefix', required=True, type=Path)
    parser.add_argument('--subjects', nargs='+',
                        default=['ch', 'lfz', 'lpc', 'lr', 'ycy', 'ylf'])
    parser.add_argument('--domains', nargs='+', default=['local', 'global'])
    parser.add_argument('--samples-per-recording', type=int, default=8)
    parser.add_argument('--root', type=Path, default=DATA_ROOT)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--allow-zgz-report-only', action='store_true')
    args = parser.parse_args()
    if 'zgz' in args.subjects and not args.allow_zgz_report_only:
        raise SystemExit('zgz is last/report-only: explicitly allow after correction is fixed')
    for path in (args.output_prefix.with_suffix('.npz'), args.output_prefix.with_suffix('.json')):
        if path.exists():
            raise SystemExit(f'preserve existing artifact {path}')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    started = time.perf_counter()
    history = FrozenS43History(args.checkpoint, args.config, args.threads)
    samples = load_existing_samples(root=args.root, subjects=args.subjects,
                                   domains=args.domains,
                                   samples_per_recording=args.samples_per_recording,
                                   history_provider=history,
                                   checkpoint=args.checkpoint, config=args.config)
    generated = time.perf_counter()
    identities = source_identities(samples, args.root, args.checkpoint, args.config)
    protocol = REPO / 'outputs/s43_br_20261027/protocol.json'
    provenance = {'command': sys.argv, 'python': sys.executable,
                  'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                  'threads': args.threads, 'generation_seconds': generated - started,
                  'source_hashing_seconds': time.perf_counter() - generated,
                  'source_files': identities,
                  'protocol_sha256': sha256(protocol) if protocol.exists() else None,
                  'GT_input': False, 'GT_use': 'target supervision/scoring only',
                  'source_data_unchanged': True, 'real_sensor_test': False,
                  'history': 'two frozen causal W50 steps from public constant initial state'}
    result = save_cache(args.output_prefix, samples, provenance)
    # Integrity/serialization check only, never computes target differences.
    reloaded = load_cache(args.output_prefix)
    for old, new in zip(samples, reloaded):
        for key in ARRAY_FIELDS + ('events', 'source_ids'):
            if not np.array_equal(old[key], new[key]):
                raise RuntimeError(f'cache roundtrip mismatch: {key}')
    result['roundtrip_exact'] = True
    result['total_seconds'] = time.perf_counter() - started
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
