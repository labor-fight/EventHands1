"""Add training-only physical motion labels to the unchanged XKF event stream.

The extra dictionary values never enter EventPacketBatch. Main and prefix packet
bytes, counter streams, augmentation and the canonical split remain unchanged.
"""
from pathlib import Path

import numpy as np
import torch

from semkine import domrand as DR
from semkine.dataset import sequences_for_split, splits_manifest
from tools.xkf.data import LongWindowRecoveryDataset, WINDOW_MIN, WINDOW_MAX, collate_long


class MotionTargetDataset(LongWindowRecoveryDataset):
    def __getitem__(self, index):
        item = super().__getitem__(index)
        main = item['main']
        handle = self._handle(main.sequence_id)
        previous_gt = handle.pos51[int(main.meta['incoming_ms'])].copy()
        if self.domrand.enabled:
            previous_gt, camera = DR.transform_labels(
                previous_gt, main.meta['domrand'], handle.camera_K,
                handle.j0, self.render_scale)
            if not np.array_equal(camera, main.camera_K):
                raise RuntimeError('motion target camera differs from main packet')
        previous_gt = np.asarray(previous_gt, np.float32)
        item['previous_gt'] = previous_gt
        # Preserve the existing additive 51D state convention. In particular,
        # root/finger coordinates are additive axis angles, not SO(3) tangents.
        item['motion_delta'] = np.asarray(main.target - previous_gt, np.float32)
        return item


def collate_motion(items):
    batch = collate_long(items)
    batch['previous_gt'] = torch.from_numpy(np.stack([item['previous_gt'] for item in items]))
    batch['motion_delta'] = torch.from_numpy(np.stack([item['motion_delta'] for item in items]))
    return batch


# Evaluation/trainer wrappers use this explicit repair name; both names denote
# the identical collator, and neither adds any field to the inference packets.
collate_repair = collate_motion


def build_train_dataset(cfg, *, start_step=0, max_steps=None):
    """Same factory contract as XKF, changing only the dataset subclass."""
    data, aug, track = cfg['DATA'], cfg.get('AUG', {}), cfg.get('TRACK', {})
    root = Path(data['ROOT'])
    manifest = splits_manifest(cfg) or root / 'splits_semkine.json'
    if manifest.resolve() != (root / 'splits_semkine.json').resolve():
        raise ValueError('XKF repair requires canonical readonly splits_semkine.json')
    sequences = sequences_for_split(root, 'train', manifest)
    subjects = sorted({s.split('_')[0] for s, _ in sequences})
    if len(sequences) != 72 or subjects != ['ch', 'lfz', 'lpc', 'lr', 'ly', 'lyh', 'lyq', 'ycy', 'ylf']:
        raise ValueError(f'XKF repair requires canonical 9 subjects/72 sequences: {subjects}/{len(sequences)}')
    if bool(aug.get('SPEED_AUG', False)):
        raise ValueError('XKF repair raw W50 forbids speed augmentation')
    components = np.load(cfg['MANO']['NPZ'])['hands_components'].astype(np.float32)
    xkf = cfg.get('XKF_DATA', {})
    return MotionTargetDataset(
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
        render_scale=cfg['MODEL'].get('RENDER_SCALE', .375),
        window_min=int(xkf.get('WINDOW_MIN', WINDOW_MIN)),
        window_max=int(xkf.get('WINDOW_MAX', WINDOW_MAX)),
        prefix_lnes=bool(xkf.get('PREFIX_LNES', True)))


__all__ = ['MotionTargetDataset', 'build_train_dataset', 'collate_motion', 'collate_repair']
