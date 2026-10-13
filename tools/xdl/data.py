"""Paired raw W50 K4 training stream for the opt-in XDL study.

Each sample is a pure function of seed, optimizer step, global row and named
counter stream. Indices are sequential before DDP's rank-strided sampling.
Only the main packet carries a target. The prefix carries an initial noisy
state, then placeholders which the model replaces with its own predictions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from semkine import domrand as DR
from semkine.dataset import SemKineDataset, sequences_for_split, splits_manifest
from semkine.events import collate_packets

EFFECTIVE_BATCH = 1024
CLIP_K = 4
WINDOW_MS = 50


def sample_coordinates(index: int) -> tuple[int, int]:
    if index < 0:
        raise IndexError(index)
    return divmod(int(index), EFFECTIVE_BATCH)


def rank_sampler(dataset, *, replicas, rank):
    # Lightning 1.9 always chooses shuffle=True for an injected training sampler,
    # even when DataLoader(shuffle=False). An explicit sampler preserves clocks.
    return torch.utils.data.DistributedSampler(dataset, num_replicas=int(replicas),
                                               rank=int(rank), shuffle=False, drop_last=True)


def counter_rng(seed: int, step: int, row: int, stream: str) -> np.random.Generator:
    payload = f'XDL-v2|{int(seed)}|{int(step)}|{int(row)}|{stream}'.encode('utf8')
    key = int.from_bytes(hashlib.sha256(payload).digest()[:16], 'little')
    return np.random.default_rng(key)


def packet_hash(batch) -> str:
    h = hashlib.sha256()
    for key in ('sequence_id', 't_start_us', 't_end_us', 'target', 'prev_state',
                'events', 'ptr', 'delta_t_s', 'betas', 'camera_K',
                'is_sequence_start', 'is_sequence_end'):
        tensor = getattr(batch, key).detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


class RawRecoveryDataset(SemKineDataset):
    """K4 legal clip sampler retaining the canonical augmentation/noise laws.

    GT timestamps are labelled millisecond indices: the first incoming state
    is at its evidence start; subsequent incoming states are at the previous
    labelled endpoint, 1ms before their evidence starts. Thus the first target
    interval is 49ms and later prediction intervals are 50ms.
    """

    def __init__(self, *, root, sequences: Sequence[tuple[str, str]], components,
                 mano_npz, seed=3407, max_steps=6000, start_step=0,
                 width=240, height=180, domrand=None, polarity_flip=False,
                 pixel_polarity_swap=False, prev_noise=(0., 0., 0.),
                 prev_noise_mode='gaussian', prev_noise_large=(.05, .3, .3),
                 prev_noise_mix=(.5, .3, .2), render_scale=.375):
        super().__init__(root=root, sequences=sequences, components=components,
                         mano_npz=mano_npz, seed=seed, input_mode='raw_packed',
                         window_min=WINDOW_MS, window_max=WINDOW_MS,
                         fixed_window=WINDOW_MS, width=width, height=height,
                         train=True, speed_aug=False, polarity_flip=polarity_flip,
                         pixel_polarity_swap=pixel_polarity_swap, domrand=domrand,
                         prev_noise=prev_noise, prev_noise_mode=prev_noise_mode,
                         prev_noise_large=prev_noise_large, prev_noise_mix=prev_noise_mix,
                         render_scale=render_scale, fast_pipe=False)
        self.max_steps = int(max_steps)
        self.start_step = int(start_step)
        if not 0 <= self.start_step < self.max_steps:
            raise ValueError('start_step must be below max_steps')
        self.legal_index = self.index[self.index[:, 1] >= self.index[:, 2] + CLIP_K * WINDOW_MS - 1]
        if not len(self.legal_index):
            raise ValueError('no legal continuous raw K4 W50 training clips')
        counts = np.bincount(self.legal_index[:, 0], minlength=len(self.sequences))
        self.coverage = {
            'legal_main_windows': int(len(self.legal_index)),
            'sequences_with_legal_clips': int((counts > 0).sum()),
            'sequences_without_legal_clips': [self.sequences[i][0] for i in np.flatnonzero(counts == 0)],
            'legal_windows_per_sequence': {s[0]: int(n) for s, n in zip(self.sequences, counts)},
            'eligible_main_index_sha256': hashlib.sha256(self.legal_index.tobytes()).hexdigest(),
            'window_ms': WINDOW_MS, 'clip_k': CLIP_K,
            'selection': 'uniform over all legal main endpoint millisecond indices',
        }

    def __len__(self):
        return (self.max_steps - self.start_step) * EFFECTIVE_BATCH

    def __getitem__(self, local_index):
        local_index = int(local_index)
        if not 0 <= local_index < len(self):
            raise IndexError(local_index)
        index = local_index + self.start_step * EFFECTIVE_BATCH
        step, row = sample_coordinates(index)
        rng = lambda stream: counter_rng(self.seed, step, row, stream)
        legal_row = int(rng('clip_selection').integers(len(self.legal_index)))
        si, end, run_a, run_b = map(int, self.legal_index[legal_row])
        h = self._handle(si)
        starts = [end - CLIP_K * WINDOW_MS + 1 + k * WINDOW_MS for k in range(CLIP_K)]
        dr = DR.sample_params(self.domrand, rng('shared_domrand'),
                              self.height * self.width * 2 * WINDOW_MS)
        flip = self.polarity_flip and int(rng('shared_polarity_flip').integers(2)) == 1
        swap = (rng('shared_polarity_swap').integers(0, 2, (self.height, self.width)).astype(bool)
                if self.pixel_polarity_swap else None)
        packets = []
        # Construct main first so prefix reads cannot perturb its named RNG stream.
        for k in (3, 0, 1, 2):
            start = starts[k]
            packet_end = start + WINDOW_MS - 1
            incoming = start if k == 0 else start - 1
            xs, ys, ps, us = self._raw_events(h, packet_end, WINDOW_MS)
            if self.domrand.enabled:
                xs, ys, ps, us = self._domrand_events(
                    xs, ys, ps, us, WINDOW_MS, dr, h.camera_K, rng(f'events_{k}'))
            if k in (0, 3):
                prev = h.pos51[incoming].copy()
            else:
                prev = np.zeros(51, np.float32)
            # Only main labels exist at the model interface. Geometry transform of
            # zero prefix targets would create nonzero translations, so zero after it.
            target = h.pos51[packet_end].copy() if k == 3 else np.zeros(51, np.float32)
            camera_K = h.camera_K.copy()
            if self.domrand.enabled:
                labels, camera_K = DR.transform_labels(np.stack([prev, target]), dr,
                                                        camera_K, h.j0, self.render_scale)
                prev, target = labels
            if k != 3:
                target = np.zeros(51, np.float32)
            if k in (1, 2):
                prev = np.zeros(51, np.float32)
            else:
                noise = self._sample_prev_noise(rng('main_noise' if k == 3 else 'initial_noise'))
                if noise is not None:
                    prev += noise
            polarity = self._polarity(xs, ys, ps, flip, swap)
            packet = self._pack(
                si, h, xs, ys, polarity, us, WINDOW_MS,
                t_start_us=start * 1000, t_end_us=(packet_end + 1) * 1000,
                target=target, prev=prev, camera_K=camera_K,
                is_start=start == run_a, is_end=packet_end == run_b - 1,
                meta={'window_ms': WINDOW_MS, 'end_idx': packet_end, 'incoming_ms': incoming,
                      'target_timestamp_us': packet_end * 1000, 'seq': h.seq,
                      'category': h.category, 'domrand': dr, 'run_start_ms': run_a,
                      'run_end_ms': run_b, 'sample_index': index, 'prefix_index': k,
                      'optimizer_step': step, 'global_row': row})
            packets.append((k, packet))
        by_k = dict(packets)
        return {'main': by_k[3], 'prefix': [by_k[k] for k in range(3)],
                'global_row': row, 'optimizer_step': step, 'sample_index': index}


def collate_recovery(items):
    return {'main': collate_packets([it['main'] for it in items]),
            'prefix': [collate_packets([it['prefix'][k] for it in items]) for k in range(3)],
            **{key: torch.tensor([it[key] for it in items], dtype=torch.long)
               for key in ('global_row', 'optimizer_step', 'sample_index')}}


def build_train_dataset(cfg, *, start_step=0, max_steps=None):
    data, aug, track = cfg['DATA'], cfg.get('AUG', {}), cfg.get('TRACK', {})
    root = Path(data['ROOT'])
    manifest = splits_manifest(cfg) or root / 'splits_semkine.json'
    if manifest.resolve() != (root / 'splits_semkine.json').resolve():
        raise ValueError('XDL requires the canonical readonly splits_semkine.json')
    sequences = sequences_for_split(root, 'train', manifest)
    subjects = sorted({s.split('_')[0] for s, _ in sequences})
    required = ['ch', 'lfz', 'lpc', 'lr', 'ly', 'lyh', 'lyq', 'ycy', 'ylf']
    if len(sequences) != 72 or subjects != required:
        raise ValueError(f'XDL requires canonical 9 subjects/72 sequences: {subjects}/{len(sequences)}')
    if bool(aug.get('SPEED_AUG', False)):
        raise ValueError('XDL raw W50 forbids speed augmentation')
    components = np.load(cfg['MANO']['NPZ'])['hands_components'].astype(np.float32)
    return RawRecoveryDataset(
        root=root, sequences=sequences, components=components, mano_npz=cfg['MANO']['NPZ'],
        width=data['WIDTH'], height=data['HEIGHT'], seed=cfg['SEED'],
        max_steps=int(max_steps if max_steps is not None else cfg['TRAIN']['MAX_STEPS']),
        start_step=start_step, domrand=DR.DomRandConfig.from_cfg(cfg),
        polarity_flip=aug.get('POLARITY_FLIP', True),
        pixel_polarity_swap=aug.get('PIXEL_POLARITY_SWAP', True),
        prev_noise=tuple(float(track.get(key, 0.)) for key in ('PREV_NOISE_T', 'PREV_NOISE_R', 'PREV_NOISE_POSE')),
        prev_noise_mode=track.get('PREV_NOISE_MODE', 'gaussian'),
        prev_noise_large=tuple(float(track.get(key, default)) for key, default in (
            ('PREV_NOISE_LARGE_T', .05), ('PREV_NOISE_LARGE_R', .3), ('PREV_NOISE_LARGE_POSE', .3))),
        prev_noise_mix=tuple(float(track.get(key, default)) for key, default in (
            ('PREV_NOISE_P_SMALL', .5), ('PREV_NOISE_P_LARGE', .3), ('PREV_NOISE_P_CORR', .2))),
        render_scale=cfg['MODEL'].get('RENDER_SCALE', .375))
