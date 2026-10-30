"""XKF training stream: the paired XDL K4 clips plus a long causal event window per packet for the direct branch.

`LongWindowRecoveryDataset` returns exactly the XDL sample (main + three prefix packets, bitwise: `packet_hash` is
unchanged) and attaches to every packet `lnes` = int16 last-timestamp codes over its own window
`[end_k - W_k + 1, end_k]`, `W_k = min(W, end_k - run_start + 1)`, and `lnes_window_ms = W_k`. Code `c > 0` decodes to
`(c - 1) / W_k`, which is `splat_event_image`'s LNES bit for bit; 0 is an empty slot.

W is drawn log-uniformly in [WINDOW_MIN, WINDOW_MAX] per sample (S39's training law). Events inside the four W50
packets are reused after augmentation, including dropout, hot pixels and polarity. Only older history uses new
counter streams (`xkf_events`, `xkf_hot`), with the same camera/polarity transform and Poisson hot pixels scaled to
its own span. Thus D and I see the same overlapping evidence and the XDL packet streams remain untouched.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from semkine import domrand as DR
from tools.xdl.data import (EFFECTIVE_BATCH, WINDOW_MS, RawRecoveryDataset, collate_recovery,
                            counter_rng, sample_coordinates)
from semkine.dataset import sequences_for_split, splits_manifest

WINDOW_MIN, WINDOW_MAX = 30, 300


def lnes_codes(xs, ys, ps, ms_rel, height, width):
    """int16 `(H, W, 2)`: ms_rel + 1 of the last event per (pixel, polarity), 0 where empty (events time-sorted)."""
    out = np.zeros((height, width, 2), np.int16)
    if len(xs):
        out[ys.astype(np.intp), xs.astype(np.intp), ps.astype(np.intp)] = (ms_rel + 1).astype(np.int16)
    return out


class LongWindowRecoveryDataset(RawRecoveryDataset):
    def __init__(self, *args, window_min=WINDOW_MIN, window_max=WINDOW_MAX, prefix_lnes=True, **kw):
        super().__init__(*args, **kw)
        self.xkf_window = (int(window_min), int(window_max))
        if not 1 <= self.xkf_window[0] <= self.xkf_window[1] <= np.iinfo(np.int16).max:
            raise ValueError('XKF window bounds must be ordered positive int16 millisecond spans')
        #: False: only the main packet gets a long window (XRF trains per packet, the prefix is never forwarded)
        self.prefix_lnes = bool(prefix_lnes)

    def __getitem__(self, local_index):
        item = super().__getitem__(local_index)
        index = int(local_index) + self.start_step * EFFECTIVE_BATCH
        step, row = sample_coordinates(index)
        rng = lambda stream: counter_rng(self.seed, step, row, stream)
        all_packets = item['prefix'] + [item['main']]
        main = item['main']
        run_a = int(main.meta['run_start_ms'])
        dr = main.meta['domrand']
        lo, hi = np.log(self.xkf_window[0]), np.log(self.xkf_window[1])
        W = int(round(float(np.exp(rng('xkf_window').uniform(lo, hi)))))
        packets = all_packets if self.prefix_lnes else [main]
        ends = [int(p.meta['end_idx']) for p in packets]
        spans = [min(W, e - run_a + 1) for e in ends]
        first = min(e - s + 1 for e, s in zip(ends, spans))
        clip_start = all_packets[0].t_start_us // 1000
        chunks = []
        if first < clip_start:
            h = self._handle(main.sequence_id)
            span = clip_start - first
            xs, ys, ps, us = self._raw_events(h, clip_start - 1, span)
            if self.domrand.enabled:
                n_hot = int(rng('xkf_hot').poisson(dr.n_hot * span / float(WINDOW_MS))) if dr.n_hot > 0 else 0
                xs, ys, ps, us = self._domrand_events(
                    xs, ys, ps, us, span, dataclasses.replace(dr, n_hot=n_hot), h.camera_K, rng('xkf_events'))
            flip = self.polarity_flip and int(rng('shared_polarity_flip').integers(2)) == 1
            swap = (rng('shared_polarity_swap').integers(0, 2, (self.height, self.width)).astype(bool)
                    if self.pixel_polarity_swap else None)
            ps = self._polarity(xs, ys, ps, flip, swap)
            chunks.append((xs, ys, ps, first + us // 1000))
        for p in all_packets:
            ev = p.events
            # _pack stores integer microseconds as float32 seconds. Round back to microseconds
            # before binning: directly flooring seconds * 1000 moves exact-ms events backwards.
            us = np.rint(ev[:, 2].astype(np.float64) * 1e6).astype(np.int64)
            chunks.append((ev[:, 0], ev[:, 1], ev[:, 3], p.t_start_us // 1000 + us // 1000))
        xs, ys, ps, ms_abs = [np.concatenate(parts) for parts in zip(*chunks)]
        for k, (e, w) in enumerate(zip(ends, spans)):
            s = e - w + 1
            sel = (ms_abs >= s) & (ms_abs <= e)
            packets[k].lnes = lnes_codes(xs[sel], ys[sel], ps[sel], ms_abs[sel] - s, self.height, self.width)
            packets[k].lnes_window_ms = int(w)
        return item


def collate_long(items):
    return collate_recovery(items)


def build_train_dataset(cfg, *, start_step=0, max_steps=None):
    """`tools/xdl/data.build_train_dataset` with the long-window subclass (same arguments and checks)."""
    data, aug, track = cfg['DATA'], cfg.get('AUG', {}), cfg.get('TRACK', {})
    root = Path(data['ROOT'])
    manifest = splits_manifest(cfg) or root / 'splits_semkine.json'
    if manifest.resolve() != (root / 'splits_semkine.json').resolve():
        raise ValueError('XKF requires the canonical readonly splits_semkine.json')
    sequences = sequences_for_split(root, 'train', manifest)
    subjects = sorted({s.split('_')[0] for s, _ in sequences})
    if len(sequences) != 72 or subjects != ['ch', 'lfz', 'lpc', 'lr', 'ly', 'lyh', 'lyq', 'ycy', 'ylf']:
        raise ValueError(f'XKF requires canonical 9 subjects/72 sequences: {subjects}/{len(sequences)}')
    if bool(aug.get('SPEED_AUG', False)):
        raise ValueError('XKF raw W50 forbids speed augmentation')
    components = np.load(cfg['MANO']['NPZ'])['hands_components'].astype(np.float32)
    xkf = cfg.get('XKF_DATA', {})
    return LongWindowRecoveryDataset(
        root=root, sequences=sequences, components=components, mano_npz=cfg['MANO']['NPZ'],
        width=data['WIDTH'], height=data['HEIGHT'], seed=cfg['SEED'],
        max_steps=int(max_steps if max_steps is not None else cfg['TRAIN']['MAX_STEPS']),
        start_step=start_step, domrand=DR.DomRandConfig.from_cfg(cfg),
        polarity_flip=aug.get('POLARITY_FLIP', True), pixel_polarity_swap=aug.get('PIXEL_POLARITY_SWAP', True),
        prev_noise=tuple(float(track.get(key, 0.)) for key in ('PREV_NOISE_T', 'PREV_NOISE_R', 'PREV_NOISE_POSE')),
        prev_noise_mode=track.get('PREV_NOISE_MODE', 'gaussian'),
        prev_noise_large=tuple(float(track.get(key, default)) for key, default in (
            ('PREV_NOISE_LARGE_T', .05), ('PREV_NOISE_LARGE_R', .3), ('PREV_NOISE_LARGE_POSE', .3))),
        prev_noise_mix=tuple(float(track.get(key, default)) for key, default in (
            ('PREV_NOISE_P_SMALL', .5), ('PREV_NOISE_P_LARGE', .3), ('PREV_NOISE_P_CORR', .2))),
        render_scale=cfg['MODEL'].get('RENDER_SCALE', .375),
        window_min=int(xkf.get('WINDOW_MIN', WINDOW_MIN)), window_max=int(xkf.get('WINDOW_MAX', WINDOW_MAX)),
        prefix_lnes=bool(xkf.get('PREFIX_LNES', True)))


__all__ = ['LongWindowRecoveryDataset', 'WINDOW_MAX', 'WINDOW_MIN', 'build_train_dataset', 'collate_long', 'lnes_codes']
