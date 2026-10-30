"""D-only paired continuation data, preserving the canonical main evidence.

All K4 packets and their LNES are constructed by the frozen original producer.
Only afterwards are the unused prefix packets discarded. Disabling prefix LNES
earlier changes the span of augmented old history and can change main evidence.
"""
from __future__ import annotations

import copy

import torch
from torch.utils.data import Dataset

from semkine.events import collate_packets
from tools.xkf.data import build_train_dataset as original_dataset


def arm_config(cfg, arm):
    if arm not in ('aug', 'raw'):
        raise ValueError('D-only arm must be aug or raw')
    cfg = copy.deepcopy(cfg)
    cfg.setdefault('XKF_DATA', {})['PREFIX_LNES'] = True
    if arm == 'raw':
        aug = cfg.setdefault('AUG', {})
        aug.setdefault('DOMRAND', {})['ENABLED'] = False
        aug['POLARITY_FLIP'] = False
        aug['PIXEL_POLARITY_SWAP'] = False
    return cfg


def sampling_identity(item):
    main = item['main']
    meta = main.meta
    return {
        **{key: int(item[key]) for key in ('sample_index', 'optimizer_step', 'global_row')},
        'sequence_id': int(main.sequence_id), 'sequence': str(meta['seq']),
        'end_ms': int(meta['end_idx']), 'incoming_ms': int(meta['incoming_ms']),
        'run_start_ms': int(meta['run_start_ms']), 'run_end_ms': int(meta['run_end_ms']),
        'lnes_window_ms': int(main.lnes_window_ms),
    }


class DirectDataset(Dataset):
    def __init__(self, cfg, *, start_step=2200, max_steps=2400):
        self.base = original_dataset(cfg, start_step=start_step, max_steps=max_steps)
        if not self.base.prefix_lnes:
            raise ValueError('preserve original augmented main LNES: PREFIX_LNES must remain true')
        self.sequences, self.coverage = self.base.sequences, self.base.coverage

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        identity = sampling_identity(item)
        return {'main': item['main'], 'sampling_identity': identity,
                **{key: item[key] for key in ('sample_index', 'optimizer_step', 'global_row')}}


def collate_direct(items):
    return {'main': collate_packets([item['main'] for item in items]),
            'sampling_identity': [item['sampling_identity'] for item in items],
            **{key: torch.tensor([item[key] for item in items], dtype=torch.long)
               for key in ('sample_index', 'optimizer_step', 'global_row')}}
