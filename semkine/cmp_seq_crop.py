r"""CMP sequence samples with EvHandPose's hand crop (docs/EXPERIMENT_RECORDS.md#cmp-amend1-20261110).

`semkine/cmp_seq.py`'s K-packet samples (same index, same augmentation draws in the same order), with every packet cut
to EvHandPose's crop (official core/dataset/dataset.py: box = joints_to_bbox_rate x the larger side of the 2D joint
extent, centred on it; scale = SIZE / box; events mapped x' = (x - c) scale + c - top_left with c the principal point,
kept inside SIZE x SIZE; training top-left shifted by U(-1, 1) x SIZE x SHIFT per packet) and the LNES / voxel grid built
on the crop (nearest pixel; the official code splats bilinearly). The box of packet k comes from the GT state at the
packet's start (its 21 MANO joints projected with the event-frame intrinsics), the online analogue of what a deployed
tracker has: its own state at the packet's start (tools/cmp/nets.py, kind epc). Per packet the sample also carries
`crop` = [scale, top-left x, top-left y, fx, fy, cx, cy] (event-frame intrinsics after domrand), which the model needs
to decode the translation and to render the flow pseudo ground truth in the crop.
"""
from __future__ import annotations

import numpy as np
import torch

from . import domrand as DR
from .cmp_seq import CMP_SALT, PACKET_MS, SeqDataset, voxel_grid
from .dataset import SemKineDataset

SIZE, RATE, SHIFT = 128, 2.0, 0.15


def crop_params(joints2d, cxy, size=SIZE, rate=RATE):
    """(21, 2) projected joints -> (scale, top-left (2,)) of EvHandPose's crop."""
    lo, hi = joints2d.min(0), joints2d.max(0)
    side = max(rate * float((hi - lo).max()), 1.0)
    scale = size / side
    tl = ((lo + hi) / 2.0 - cxy) * scale + cxy - size / 2.0
    return float(scale), tl.astype(np.float64)


def crop_events(xs, ys, scale, tl, cxy, size=SIZE):
    """Integer crop pixels of the events and the mask of those inside the crop."""
    u = np.floor((np.asarray(xs, np.float64) - cxy[0]) * scale + cxy[0] - tl[0] + 0.5).astype(np.int64)
    v = np.floor((np.asarray(ys, np.float64) - cxy[1]) * scale + cxy[1] - tl[1] + 0.5).astype(np.int64)
    keep = (u >= 0) & (u < size) & (v >= 0) & (v < size)
    return u, v, keep


class JointProjector:
    """The 21 MANO joints of 51-D states, projected with event-frame intrinsics (CPU, one per process)."""

    def __init__(self, mano_npz):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model"))
        from mano_layer import ManoLayer
        self.mano = ManoLayer(mano_npz, add_mean=False).eval()

    @torch.no_grad()
    def __call__(self, states, betas, fx, fy, cx, cy):
        from pose_repr import decode_to_mano_inputs
        p = torch.as_tensor(np.asarray(states, np.float32)).view(-1, 51)
        dec = decode_to_mano_inputs(p, "mano_full_axis_angle", self.mano.hands_components, self.mano.hands_mean)
        b = torch.as_tensor(np.asarray(betas, np.float32)).view(1, -1).expand(p.shape[0], -1)
        _, joints = self.mano(b, dec["global_orient"], dec["local_full_aa"], dec["transl"])
        J = joints.numpy().astype(np.float64)
        z = np.clip(J[..., 2], 1e-6, None)
        return np.stack([fx * J[..., 0] / z + cx, fy * J[..., 1] / z + cy], -1)          # (N, 21, 2)


class SeqCropDataset(SeqDataset):
    """`SeqDataset` with every packet cut to EvHandPose's crop (built by `upgrade`)."""

    def _setup_crop(self, size=SIZE, rate=RATE, shift=SHIFT):
        self.crop_size, self.crop_rate, self.crop_shift = int(size), float(rate), float(shift)
        self._proj = None
        return self

    def _projector(self):
        if self._proj is None:
            torch.set_num_threads(1)
            self._proj = JointProjector(self.mano_npz)
        return self._proj

    def __getstate__(self):
        state = super().__getstate__()
        state["_proj"] = None
        return state

    def __getitem__(self, idx: int):
        rng = np.random.default_rng([self.seed, int(idx), self.salt])
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
        j2 = self._projector()(states[:self.k], h.betas, fx, fy, cx, cy)              # each packet's start state
        lnes = np.empty((self.k, S, S, 2 * len(self.event_channels)), np.float32)
        counts = np.empty(self.k, np.int64)
        voxel = np.empty((self.k, self.voxel_bins, S, S), np.float32)
        crop = np.empty((self.k, 7), np.float32)
        for i, (xs, ys, ps, us) in enumerate(raw):
            p_eff = self._polarity(xs, ys, ps, flip, swap_mask)
            scale, tl = crop_params(j2[i], cxy, S, self.crop_rate)
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
            shift=SHIFT) -> SeqCropDataset:
    if type(ds) is not SemKineDataset:
        raise TypeError("upgrade expects semkine.dataset.SemKineDataset")
    if voxel_bins < 1:
        raise ValueError("the crop samples carry the voxel grid")
    ds.__class__ = SeqCropDataset
    ds._setup_seq(k, packet_ms, voxel_bins, CMP_SALT)
    return ds._setup_crop(size, rate, shift)


__all__ = ["JointProjector", "RATE", "SHIFT", "SIZE", "SeqCropDataset", "crop_events", "crop_params", "upgrade"]
