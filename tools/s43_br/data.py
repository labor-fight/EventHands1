"""Read-only existing-data adapter for the opt-in S43 B-R prototype.

No target pose enters a representation, candidate, crop or history constructor.
The default history is two causal frozen-S43 steps from a public fixed pose,
not the old evaluation trace (whose run initialization used a label).

Use the EventHandsTrain Python environment for the optional frozen model.
All labels are pseudo-GT and all streams here are the existing hand_data51
dataset; this adapter does not imply that these are real sensor recordings.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = Path('/data1/lyq/code/mesh/EventHands/data/hand_data51')
CHECKPOINT = REPO / 'outputs/semkine/s43n_sp_s3407/s43n_sp_s3407-step=6000.ckpt'
CONFIG = REPO / 'outputs/semkine/s43n_sp_s3407/config_resolved.yaml'
DEFAULT_SUBJECTS = ('ch', 'lfz', 'lpc', 'lr', 'ycy', 'ylf', 'zgz')
FIT7 = ('ch', 'lfz', 'lpc', 'lr', 'ly', 'lyh', 'lyq')
SELECT = ('ycy', 'ylf')
PUBLIC_INITIAL_STATE = np.array([0., 0., .6] + [0.] * 48, dtype=np.float32)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_manifest(root: Path = DATA_ROOT) -> dict:
    """Use the existing unique manifest without creating a new split."""
    return json.loads((Path(root) / 'splits_semkine.json').read_text())


def recording_dirs(root: Path = DATA_ROOT) -> Dict[str, str]:
    manifest = read_manifest(root)
    out = {}
    for split in ('train', 'val_core'):
        info = manifest[split]
        for seq in info['trials']:
            out[seq] = info['legacy_dir'][seq]
    return out


def meta51_mmap(path: Path) -> np.ndarray:
    """Read the established 51D binary format without materializing all labels."""
    path = Path(path)
    with path.open('rb') as f:
        n, = struct.unpack('<i', f.read(4))
    if n != 51:
        raise ValueError('expected 51D MANO metadata')
    dtype = np.dtype([('data', '<f8', 51), ('m0', 'u1'), ('m1', 'u1')])
    if (path.stat().st_size - 4) % dtype.itemsize:
        raise ValueError('truncated pose metadata')
    return np.memmap(path, dtype=dtype, mode='r', offset=4)


def sample_endpoints(valid_runs_ms: np.ndarray, limit: int = 8,
                     window_ms: int = 50, stride_ms: int = 100,
                     start_margin_ms: int = 150) -> np.ndarray:
    """Fixed evenly spaced choices; selection never depends on labels/events.

    Current window is [end-window+1, end+1). Its start is at least 150 ms
    after a valid run starts. Main windows are separated by at least 100 ms.
    """
    if limit < 1 or window_ms < 1 or stride_ms < window_ms:
        raise ValueError('invalid endpoint budget')
    if start_margin_ms < 2 * window_ms:
        raise ValueError('not enough causal warmup')
    candidates = []
    for run_id, (a, b) in enumerate(np.asarray(valid_runs_ms, np.int64).reshape(-1, 2)):
        first = int(a) + start_margin_ms + window_ms - 1
        candidates.extend((int(end), int(a), int(b), int(run_id))
                          for end in range(first, int(b), stride_ms))
    if not candidates:
        return np.empty((0, 4), np.int64)
    idx = np.unique(np.linspace(0, len(candidates) - 1,
                                min(limit, len(candidates))).astype(np.int64))
    return np.asarray([candidates[int(i)] for i in idx], np.int64)


class SequenceReader:
    """Read-only mappings of one existing recording; no augmentation."""
    def __init__(self, recording: str, root: Path = DATA_ROOT):
        self.root = Path(root)
        self.recording = recording
        self.legacy_dir = recording_dirs(root)[recording]
        self.base = self.root / self.legacy_dir / recording
        self.events = np.load(str(self.base) + '_events.npy', mmap_mode='r')
        self.offsets = np.load(str(self.base) + '_offsets.npy', mmap_mode='r')
        self.tsub = np.load(str(self.base) + '_tsub.npy', mmap_mode='r')
        with np.load(str(self.base) + '_aux.npz', allow_pickle=False) as aux:
            self.betas = np.asarray(aux['betas'], np.float32).reshape(10).copy()
            self.K_cam = np.asarray(aux['camera_K'], np.float32).reshape(3, 3).copy()
            self.runs = np.asarray(aux['valid_runs_ms'], np.int64).copy()
        self.labels = meta51_mmap(Path(str(self.base) + '.meta'))
        if self.events.ndim != 2 or self.events.shape[1] != 3:
            raise ValueError('expected events[x,y,stored-polarity]')
        if int(self.offsets[-1]) != len(self.events) or len(self.tsub) != len(self.events):
            raise ValueError('event/offset/sub-ms row mismatch')

    def window(self, start_ms: int, window_ms: int = 50) -> dict:
        """Native event grid, times relative to start, half-open interval.

        events[:, :] = [t_us, x, y, stored_polarity]. Stable source IDs are
        absolute event-row indices in this recording, never x/y/p/t hashes.
        Physical ON/OFF is NOT asserted for stored 0/1.
        """
        stop = int(start_ms) + int(window_ms)
        if start_ms < 0 or stop >= len(self.offsets):
            raise ValueError('window outside recording')
        lo, hi = int(self.offsets[start_ms]), int(self.offsets[stop])
        ev = np.asarray(self.events[lo:hi], np.int64)
        ms = np.repeat(np.arange(window_ms, dtype=np.int64),
                       np.diff(np.asarray(self.offsets[start_ms:stop + 1], np.int64)))
        sub = np.asarray(self.tsub[lo:hi], np.int64)
        if len(sub) and (np.any(sub < 0) or np.any(sub >= 1000)):
            raise ValueError('sub-ms timestamp outside declared bin')
        out = np.empty((len(ev), 4), np.int64)
        out[:, 0] = ms * 1000 + sub
        out[:, 1:] = ev
        if len(ev) and (np.any(ev[:, :2] < 0) or np.any(ev[:, 0] >= 240)
                        or np.any(ev[:, 1] >= 180) or np.any((ev[:, 2] < 0) | (ev[:, 2] > 1))):
            raise ValueError('invalid native event coordinate/polarity')
        return {'events': out, 'source_ids': np.arange(lo, hi, dtype=np.int64),
                'start_us': int(start_ms) * 1000, 'cutoff_us': stop * 1000,
                'polarity_contract': 'stored 0/1; physical mapping unverified'}

    def target(self, end_ms: int) -> np.ndarray:
        row = self.labels[int(end_ms)]
        if int(row['m0']) != 4 or int(row['m1']) != 13:
            raise ValueError('pose metadata marker mismatch')
        return np.asarray(row['data'], np.float32).copy()


def native_lnes(events: np.ndarray, window_ms: int = 50) -> np.ndarray:
    """Original S43 last-LNES convention including its zero first-ms code."""
    ev = np.asarray(events)
    out = np.zeros((180, 240, 2), np.float32)
    if len(ev):
        ms = ev[:, 0].astype(np.int64) // 1000
        # np.maximum.at also gives deterministic same-slot duplicates, including
        # unsorted packet arrival. Original sorted last assignment has same values.
        np.maximum.at(out, (ev[:, 2].astype(np.intp), ev[:, 1].astype(np.intp),
                            ev[:, 3].astype(np.intp)),
                      ms.astype(np.float32) / np.float32(window_ms))
    return out


class FrozenS43History:
    """Two causal W50 steps from a public pose; no label argument exists."""
    def __init__(self, checkpoint: Path = CHECKPOINT, config: Path = CONFIG,
                 threads: int = 4):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for path in (REPO, REPO / 'model'):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import torch
        from config import load_config
        from model import MNISTModel
        torch.set_num_threads(threads)
        self.torch = torch
        self.checkpoint = Path(checkpoint)
        self.config = Path(config)
        cfg = load_config(self.config)
        self.model = MNISTModel.load_from_checkpoint(
            str(self.checkpoint), cfg=cfg, map_location='cpu').to('cpu').eval()

    def predict(self, first_images: np.ndarray, second_images: np.ndarray,
                betas: np.ndarray, K_cam: np.ndarray,
                batch_size: int = 8) -> Tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        older, previous = [], []
        with torch.inference_mode():
            for i in range(0, len(first_images), batch_size):
                n = min(batch_size, len(first_images) - i)
                init = torch.from_numpy(np.repeat(PUBLIC_INITIAL_STATE[None], n, axis=0))
                b = torch.from_numpy(np.repeat(np.asarray(betas, np.float32)[None], n, axis=0))
                K = torch.from_numpy(np.repeat(np.asarray(K_cam, np.float32)[None], n, axis=0))
                x1 = torch.from_numpy(np.asarray(first_images[i:i + n], np.float32))
                x2 = torch.from_numpy(np.asarray(second_images[i:i + n], np.float32))
                p1 = self.model(x1, init, betas=b, camera_K=K)
                p2 = self.model(x2, p1, betas=b, camera_K=K)
                older.append(p1.float().cpu().numpy().copy())
                previous.append(p2.float().cpu().numpy().copy())
        return np.concatenate(older), np.concatenate(previous)


def load_existing_samples(limit_per_recording: Optional[int] = None,
                          root: Path = DATA_ROOT,
                          subjects: Sequence[str] = DEFAULT_SUBJECTS,
                          domains: Sequence[str] = ('local', 'global'),
                          samples_per_recording: int = 8,
                          window_ms: int = 50,
                          stride_ms: int = 100,
                          history_provider: Optional[FrozenS43History] = None,
                          history_mode: str = 'frozen_cold',
                          checkpoint: Path = CHECKPOINT,
                          config: Path = CONFIG) -> List[dict]:
    """Return small nonoverlapping windows with strictly causal history.

    Only `target` reads labels. Its use is restricted to supervision/scoring;
    callers must not use it to propose candidates or render an input.
    `public_constant` is a no-learning/cold-start control, not a GT upper bound.
    Data selection and all warmup images are label independent.
    """
    if history_mode not in ('frozen_cold', 'public_constant'):
        raise ValueError('only label-free frozen_cold/public_constant histories supported')
    if window_ms != 50:
        raise ValueError('this prototype protocol fixes W50')
    limit = int(limit_per_recording or samples_per_recording)
    dirs = recording_dirs(root)
    recordings = [f'{s}_{d}' for s in subjects for d in domains]
    missing = [seq for seq in recordings if seq not in dirs]
    if missing:
        raise ValueError(f'recordings outside unchanged manifest: {missing}')
    if history_mode == 'frozen_cold' and history_provider is None:
        history_provider = FrozenS43History(checkpoint, config)
    samples = []
    for seq in recordings:
        reader = SequenceReader(seq, root)
        endpoints = sample_endpoints(reader.runs, limit, window_ms, stride_ms)
        if len(endpoints) == 0:
            continue
        first, second = [], []
        for end, _, _, _ in endpoints:
            start = int(end) - window_ms + 1
            first.append(native_lnes(reader.window(start - 2 * window_ms)['events']))
            second.append(native_lnes(reader.window(start - window_ms)['events']))
        if history_mode == 'frozen_cold':
            older, prev = history_provider.predict(np.stack(first), np.stack(second),
                                                   reader.betas, reader.K_cam)
        else:
            prev = np.repeat(PUBLIC_INITIAL_STATE[None], len(endpoints), axis=0)
            older = prev.copy()
        for i, (end, run_a, run_b, run_id) in enumerate(endpoints):
            start = int(end) - window_ms + 1
            current = reader.window(start)
            sample = dict(current)
            sample.update({
                'prev_state': np.asarray(prev[i], np.float32).copy(),
                'prev_older_state': np.asarray(older[i], np.float32).copy(),
                'history_predictions': np.stack((older[i], prev[i])).astype(np.float32),
                'target': reader.target(int(end)),
                'betas': reader.betas.copy(), 'K_cam': reader.K_cam.copy(),
                'camera_K': reader.K_cam.copy(),
                'recording': seq, 'subject': seq.split('_')[0],
                'domain': seq.split('_')[1],
                'partition': ('zgz_dev_report_only' if seq.startswith('zgz_') else
                              'SELECT' if seq.split('_')[0] in SELECT else 'FIT7'),
                'data_kind': 'existing_hand_data51_v2e_provenance_not_real_sensor_test',
                'end_ms': int(end), 'run_start_ms': int(run_a),
                'run_end_ms': int(run_b), 'run_id': int(run_id),
                'target_time_us': int(end) * 1000,
                'prev_state_time_us': (start - 1) * 1000,
                'prev_older_state_time_us': (start - window_ms - 1) * 1000,
                'cutoff': (int(end) + 1) * 1000,
                'targettime': int(end) * 1000,
                'history_source': ('frozen_s43_two_causal_W50_public_initial_pose' if history_mode == 'frozen_cold'
                                   else 'public_constant_pose_no_GT'),
                'history_checkpoint': str(checkpoint) if history_mode == 'frozen_cold' else None,
                'history_initial_state': PUBLIC_INITIAL_STATE.copy(),
                'external_conditions': 'sequence beta/K per original S43; not estimated by events',
                'native_event_shape': (180, 240),
                'camera_source_shape': (480, 640),
                'camera_to_event_scale': .375,
                'label_role': 'target_only_not_constructor_input',
                'timing_contract': 'event_cutoff=target_time+1ms; explicitly retrospective 1ms',
            })
            samples.append(sample)
    return samples


def counts_from_events(events: np.ndarray, image_size: Tuple[int, int] = (18, 24),
                       bins: int = 50, bin_us: int = 1000,
                       cap: Optional[int] = None) -> np.ndarray:
    """Common integer count observation, preserving true simultaneous multiplicity.

    Spatial reduction is fixed for every arm and does not use a target crop.
    Cap is applied to observations after aggregation, never to latent triggers.
    """
    h, w = map(int, image_size)
    if h < 1 or w < 1 or bins < 1 or bin_us < 1:
        raise ValueError('positive observation dimensions required')
    ev = np.asarray(events)
    if ev.ndim != 2 or ev.shape[1] != 4:
        raise ValueError('events must have columns [t_us,x,y,stored_p]')
    result = np.zeros((bins, h * w, 2), np.int64)
    if len(ev):
        b = ev[:, 0].astype(np.int64) // bin_us
        # Pixel-centre reduction, identical to geometry.map_pixels and the
        # intrinsics map (cx + .5) * scale - .5. Integer arithmetic avoids
        # packet-dependent floating-point rounding at reduced-cell edges.
        x = (2 * ev[:, 1].astype(np.int64) + 1) * w // (2 * 240)
        y = (2 * ev[:, 2].astype(np.int64) + 1) * h // (2 * 180)
        p = ev[:, 3].astype(np.int64)
        if (np.any(b < 0) or np.any(b >= bins) or np.any(x < 0) or np.any(x >= w)
                or np.any(y < 0) or np.any(y >= h) or np.any(p < 0) or np.any(p > 1)):
            raise ValueError('event outside count observation')
        np.add.at(result, (b, y * w + x, p), 1)
    return np.minimum(result, cap) if cap is not None else result


def inspect_sources(root: Path = DATA_ROOT) -> dict:
    """Metadata-only inventory suitable before experiment readings."""
    return {'root': str(root), 'manifest': str(Path(root) / 'splits_semkine.json'),
            'manifest_sha256': sha256(Path(root) / 'splits_semkine.json'),
            'checkpoint': str(CHECKPOINT), 'checkpoint_exists': CHECKPOINT.exists(),
            'config': str(CONFIG), 'config_exists': CONFIG.exists(),
            'history_modes': ['frozen_cold', 'public_constant'],
            'default_recordings': [f'{s}_{d}' for s in DEFAULT_SUBJECTS for d in ('local', 'global')],
            'GT_input': False, 'real_sensor_test': False,
            'camera_source_shape': [480, 640],
            'camera_to_event_scale': .375}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata-only', action='store_true', required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_sources(), indent=2, ensure_ascii=False))
