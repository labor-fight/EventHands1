r"""CMP sequence samples (docs/EXPERIMENT_RECORDS.md#cmp-prereg-20261110): K consecutive 50 ms packets of one valid run.

The training dataset of the frame stages (`semkine.dataset.build_dataset(cfg, "train")`, legacy LNES face, B2's
augmentation contract) re-indexed by the last packet's end ms, as `semkine/tpd_rol.py` re-indexes B2's raw packets: one
sample = K packets of PACKET_MS sharing one domain-randomisation realisation (roll / scale / shift / keep fraction), one
polarity flip and one per-pixel swap mask; B2's prior noise is drawn once, for the first packet's start. Every packet is
cut and splatted with the frame stages' own calls (`_raw_events`, `_domrand_events`, `_polarity`, `_splat_lnes`), so a
packet equals the legacy LNES of the same 50 ms window under the same augmentation draw. Optional per packet: the
EvHandPose voxel grid (`voxel_grid`, the official `event_to_voxel`: polarity +-1, time normalised over the packet,
linear weights onto VOXEL_BINS bins).

A sample is a dict of stacked tensors (default_collate stacks them, few file descriptors per batch):
  lnes (K, H, W, C)  prev0 (51,)  states (K + 1, 51)  betas (10,)  camera_K (3, 3)  counts (K,)  [voxel (K, bins, H, W)]
"""
from __future__ import annotations

import numpy as np
import torch

from . import domrand as DR
from .dataset import SemKineDataset

#: per-sample stream salt of the CMP sequence samples
CMP_SALT = 20261110
PACKET_MS = 50


def voxel_grid(xs, ys, ps, us, window_ms, bins, height, width):
    """EvHandPose `event_to_voxel` on integer pixels: (bins, H, W) float32."""
    out = np.zeros(bins * height * width, np.float64)
    if len(xs):
        t = np.asarray(us, np.float64) / (float(window_ms) * 1000.0) * (bins - 1)
        b0 = np.clip(np.floor(t).astype(np.int64), 0, bins - 1)
        frac = t - b0
        pol = np.asarray(ps, np.float64) * 2.0 - 1.0
        pix = np.asarray(ys, np.int64) * width + np.asarray(xs, np.int64)
        out += np.bincount(b0 * height * width + pix, weights=pol * np.clip(1.0 - frac, 0.0, 1.0),
                           minlength=out.size)
        b1 = b0 + 1
        ok = b1 < bins
        out += np.bincount(b1[ok] * height * width + pix[ok], weights=(pol * frac)[ok], minlength=out.size)
    return out.astype(np.float32).reshape(bins, height, width)


class SeqDataset(SemKineDataset):
    """The frame stages' training dataset re-indexed for K consecutive packets (built by `upgrade`)."""

    def _setup_seq(self, k: int, packet_ms: int = PACKET_MS, voxel_bins: int = 0, salt: int = CMP_SALT):
        if k < 1 or packet_ms < 1 or voxel_bins < 0:
            raise ValueError("K >= 1, PACKET_MS >= 1, VOXEL_BINS >= 0")
        if not self.train or self.input_mode != "legacy_lnes" or self.xag_source_root or self.prev_lag_ms is not None:
            raise ValueError("CMP sequences are built from the legacy-LNES training dataset only")
        self.k, self.packet_ms, self.voxel_bins, self.salt = int(k), int(packet_ms), int(voxel_bins), int(salt)
        self.set_fast_pipe(False)
        span = self.k * self.packet_ms
        runs = np.unique(self.index[:, [0, 2, 3]], axis=0)                     # (sequence, run start, run end)
        rows = [np.stack([np.full(b - (a + span - 1), si), np.arange(a + span - 1, b), np.full(b - (a + span - 1), a),
                          np.full(b - (a + span - 1), b)], 1) for si, a, b in runs.tolist() if b - (a + span - 1) > 0]
        self.base_len = int(len(self.index))
        self.index = np.concatenate(rows).astype(np.int64) if rows else np.zeros((0, 4), np.int64)
        return self

    def ends(self, idx: int):
        si, last, run_a, run_b = (int(v) for v in self.index[idx])
        P, K = self.packet_ms, self.k
        return si, [last - P * (K - 1 - i) for i in range(K)], run_a, run_b

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
        voxel = np.empty((self.k, self.voxel_bins, H, W), np.float32) if self.voxel_bins else None
        for i, (xs, ys, ps, us) in enumerate(raw):
            p_eff = self._polarity(xs, ys, ps, flip, swap_mask)
            lnes[i] = self._splat_lnes(xs, ys, p_eff, us // 1000, P, H, W)
            counts[i] = len(xs)
            if voxel is not None:
                voxel[i] = voxel_grid(xs, ys, p_eff, us, P, self.voxel_bins, H, W)
        item = {"lnes": torch.from_numpy(lnes), "prev0": torch.from_numpy(np.asarray(prev0, np.float32)),
                "states": torch.from_numpy(np.asarray(states, np.float32)),
                "betas": torch.from_numpy(h.betas.copy()), "camera_K": torch.from_numpy(np.asarray(camera_K, np.float32)),
                "counts": torch.from_numpy(counts)}
        if voxel is not None:
            item["voxel"] = torch.from_numpy(voxel)
        return item


def upgrade(ds: SemKineDataset, k: int, packet_ms: int = PACKET_MS, voxel_bins: int = 0) -> SeqDataset:
    """Turn the legacy-LNES training dataset into its K-packet sequence form, keeping every augmentation attribute."""
    if type(ds) is not SemKineDataset:
        raise TypeError("upgrade expects semkine.dataset.SemKineDataset")
    ds.__class__ = SeqDataset
    return ds._setup_seq(k, packet_ms, voxel_bins)


__all__ = ["CMP_SALT", "PACKET_MS", "SeqDataset", "upgrade", "voxel_grid"]
