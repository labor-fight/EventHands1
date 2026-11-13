r"""CMP sequence samples with EventHPE's event frames (docs/EXPERIMENT_RECORDS.md#cmp-amend7-20261110).

`semkine/cmp_seq.py`'s K-packet samples (same index, same random draws in the same order), with every packet also given
as EventHPE's event frame (official flow_net/prepare_data.py `generate_events_frame` and event_pose_estimation/
dataloader.py: the events of one step in equal temporal partitions, one binary image per partition -- 1 where the pixel
fired in that partition, polarity ignored; a step is 2 gray-frame intervals x 4 partitions = 8 channels). Here a step is
one PACKET_MS packet cut into EVENT_BINS (8) equal parts.

A sample is `semkine/cmp_seq.py`'s dict plus `ev8` (K, EVENT_BINS, H, W) uint8.
"""
from __future__ import annotations

import numpy as np
import torch

from . import domrand as DR
from .cmp_seq import CMP_SALT, PACKET_MS, SeqDataset, voxel_grid
from .dataset import SemKineDataset

EVENT_BINS = 8


def event_frames(xs, ys, us, window_ms, bins, height, width):
    """EventHPE's event frame of one step: (bins, H, W) uint8, 1 where the pixel fired in the temporal partition."""
    out = np.zeros((bins, height, width), np.uint8)
    if len(xs):
        b = np.clip(np.asarray(us, np.int64) * bins // (int(window_ms) * 1000), 0, bins - 1)
        out[b, np.asarray(ys, np.int64), np.asarray(xs, np.int64)] = 1
    return out


class SeqEhpeDataset(SeqDataset):
    """`SeqDataset` whose packets also carry EventHPE's event frames (built by `upgrade`)."""

    def _setup_ehpe(self, event_bins=EVENT_BINS):
        if event_bins < 1:
            raise ValueError("EVENT_BINS >= 1")
        self.event_bins = int(event_bins)
        return self

    def __getitem__(self, idx: int):
        rng = np.random.default_rng([self.seed, int(idx), self.salt])
        si, ends, _, _ = self.ends(idx)
        P, H, W = self.packet_ms, self.height, self.width
        h = self._handle(si)
        starts = [e - P + 1 for e in ends]
        camera_K = h.camera_K.copy()
        states = np.stack([self._to_target(h.pos51[starts[0]])] + [self._to_target(h.pos51[e]) for e in ends])
        dr = DR.sample_params(self.domrand, rng, H * W * 2 * P)
        raw = []
        for e in ends:
            xs, ys, ps, us = self._raw_events(h, e, P)
            if self.domrand.enabled:
                xs, ys, ps, us = self._domrand_events(xs, ys, ps, us, P, dr, camera_K, rng)
            raw.append((xs, ys, ps, us))
        if self.domrand.enabled:
            states, camera_K = DR.transform_labels(states, dr, camera_K, h.j0, self.render_scale)
        prev0 = states[0].copy()
        noise = self._sample_prev_noise(rng)
        if noise is not None:
            prev0 = prev0 + noise
        flip = self.polarity_flip and int(rng.integers(0, 2)) == 1
        swap_mask = rng.integers(0, 2, size=(H, W)).astype(bool) if self.pixel_polarity_swap else None
        lnes = np.empty((self.k, H, W, 2 * len(self.event_channels)), np.float32)
        counts = np.empty(self.k, np.int64)
        voxel = np.empty((self.k, self.voxel_bins, H, W), np.float32)
        ev8 = np.empty((self.k, self.event_bins, H, W), np.uint8)
        for i, (xs, ys, ps, us) in enumerate(raw):
            p_eff = self._polarity(xs, ys, ps, flip, swap_mask)
            lnes[i] = self._splat_lnes(xs, ys, p_eff, us // 1000, P, H, W)
            counts[i] = len(xs)
            voxel[i] = voxel_grid(xs, ys, p_eff, us, P, self.voxel_bins, H, W)
            ev8[i] = event_frames(xs, ys, us, P, self.event_bins, H, W)
        return {"lnes": torch.from_numpy(lnes), "voxel": torch.from_numpy(voxel), "ev8": torch.from_numpy(ev8),
                "prev0": torch.from_numpy(np.asarray(prev0, np.float32)),
                "states": torch.from_numpy(np.asarray(states, np.float32)),
                "betas": torch.from_numpy(h.betas.copy()), "camera_K": torch.from_numpy(np.asarray(camera_K, np.float32)),
                "counts": torch.from_numpy(counts)}


def upgrade(ds: SemKineDataset, k: int, packet_ms: int = PACKET_MS, voxel_bins: int = 5,
            event_bins: int = EVENT_BINS) -> SeqEhpeDataset:
    if type(ds) is not SemKineDataset:
        raise TypeError("upgrade expects semkine.dataset.SemKineDataset")
    if voxel_bins < 1:
        raise ValueError("the EventHPE samples carry the voxel grid of the flow network")
    ds.__class__ = SeqEhpeDataset
    ds._setup_seq(k, packet_ms, voxel_bins, CMP_SALT)
    return ds._setup_ehpe(event_bins)


__all__ = ["EVENT_BINS", "SeqEhpeDataset", "event_frames", "upgrade"]
