r"""CMP sequence samples with EvHandPose's hand crop around a tracker-like state (docs/EXPERIMENT_RECORDS.md
#cmp-amend6-20261110).

`semkine/cmp_seq_crop.py` cuts packet k around the GT state at the packet's start, so the crop scale s = SIZE / box side
always equals the one the true depth gives, the hand always fills the same fraction of the crop, and the network never
has to read depth from the crop: its crop-frame depth t'_z = Z / s is a constant, the decoded depth Z = t'_z s only
copies the box's, and online (box from the tracker's own previous state) the depth is a multiplicative random walk
(#cmp-results-20261110: predicted depth 3x the truth within 200 ms, 60 m after one second).

Here the box of packet k comes from what a deployed tracker has, a wrong state: the GT state at the packet's start plus
one draw of B2's prior noise (the protocol's model of a tracker's state error, drawn per packet) and the box side times a
log-uniform zoom in [1 / ZOOM, ZOOM]; the official per-packet shift is kept. Everything else is `SeqCropDataset`: the same
index, the same events and augmentation draws for the same sample (the box draws use their own stream), the same
targets, and `crop` = [scale, top-left x, top-left y, fx, fy, cx, cy] of the box actually cut, so the translation target
in the crop frame now varies with the box and the depth has to be read from the hand's size in the crop.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import domrand as DR
from .cmp_seq import CMP_SALT, PACKET_MS, voxel_grid
from .cmp_seq_crop import RATE, SHIFT, SIZE, SeqCropDataset, crop_events, crop_params
from .dataset import SemKineDataset

ZOOM = 1.5
#: stream of the box draws (prior noise per packet, zoom), apart from the sample's own stream
BOX_STREAM = 1


class SeqCropXDataset(SeqCropDataset):
    """`SeqCropDataset` with every packet's box built from GT + prior noise and zoomed (built by `upgrade`)."""

    def _setup_track(self, zoom=ZOOM):
        if zoom < 1.0:
            raise ValueError("ZOOM >= 1")
        self.track_zoom = float(zoom)
        return self

    def box_states(self, states, rng):
        """(K, 51) tracker-like states of the K packet starts: GT + one prior-noise draw each."""
        out = np.asarray(states[:self.k], np.float32).copy()
        for i in range(self.k):
            noise = self._sample_prev_noise(rng)
            if noise is not None:
                out[i] += noise
        return out

    def __getitem__(self, idx: int):
        rng = np.random.default_rng([self.seed, int(idx), self.salt])
        rng_box = np.random.default_rng([self.seed, int(idx), self.salt, BOX_STREAM])
        si, ends, _, _ = self.ends(idx)
        P, H, W, S = self.packet_ms, self.height, self.width, self.crop_size
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
        rs = self.render_scale
        fx, fy, cx, cy = (float(camera_K[0, 0] * rs), float(camera_K[1, 1] * rs), float(camera_K[0, 2] * rs),
                          float(camera_K[1, 2] * rs))
        cxy = np.array([cx, cy])
        j2 = self._projector()(self.box_states(states, rng_box), h.betas, fx, fy, cx, cy)   # each packet's box state
        lnes = np.empty((self.k, S, S, 2 * len(self.event_channels)), np.float32)
        counts = np.empty(self.k, np.int64)
        voxel = np.empty((self.k, self.voxel_bins, S, S), np.float32)
        crop = np.empty((self.k, 7), np.float32)
        lz = math.log(self.track_zoom)
        for i, (xs, ys, ps, us) in enumerate(raw):
            p_eff = self._polarity(xs, ys, ps, flip, swap_mask)
            zoom = math.exp(float(rng_box.uniform(-lz, lz)))
            scale, tl = crop_params(j2[i], cxy, S, self.crop_rate * zoom)       # the box side x zoom about its centre
            if self.train:
                tl = tl + (rng.random(2) - 0.5) * 2.0 * S * self.crop_shift
            u, v, keep = crop_events(xs, ys, scale, tl, cxy, S)
            lnes[i] = self._splat_lnes(u[keep], v[keep], p_eff[keep], us[keep] // 1000, P, S, S)
            voxel[i] = voxel_grid(u[keep], v[keep], p_eff[keep], us[keep], P, self.voxel_bins, S, S)
            counts[i] = int(keep.sum())
            crop[i] = (scale, tl[0], tl[1], fx, fy, cx, cy)
        return {"lnes": torch.from_numpy(lnes), "voxel": torch.from_numpy(voxel), "crop": torch.from_numpy(crop),
                "prev0": torch.from_numpy(np.asarray(prev0, np.float32)),
                "states": torch.from_numpy(np.asarray(states, np.float32)),
                "betas": torch.from_numpy(h.betas.copy()), "camera_K": torch.from_numpy(np.asarray(camera_K, np.float32)),
                "counts": torch.from_numpy(counts)}


def upgrade(ds: SemKineDataset, k: int, packet_ms: int = PACKET_MS, voxel_bins: int = 5, size=SIZE, rate=RATE,
            shift=SHIFT, zoom=ZOOM) -> SeqCropXDataset:
    if type(ds) is not SemKineDataset:
        raise TypeError("upgrade expects semkine.dataset.SemKineDataset")
    if voxel_bins < 1:
        raise ValueError("the crop samples carry the voxel grid")
    ds.__class__ = SeqCropXDataset
    ds._setup_seq(k, packet_ms, voxel_bins, CMP_SALT)
    return ds._setup_crop(size, rate, shift)._setup_track(zoom)


__all__ = ["BOX_STREAM", "SeqCropXDataset", "ZOOM", "upgrade"]
