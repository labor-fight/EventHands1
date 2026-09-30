#!/usr/bin/env python3
"""S1 domain randomisation, re-landed from the §11 recipe after that write-up was retired.

Why this exists at all. Four consecutive network-side changes (EDD, J3D, CMN, KSGN) failed to
move the closed-loop metric, and the one change that passed its pre-registered gate touched only
the dataloader: recursive RA-MPJPE 19.26 -> 12.77 mm (worse of two replicates, -34%) with the
network unchanged to the parameter. The cause it addressed is measured: `zgz_local` fires 1490
events per 50 ms, roughly 1/20 of the same
subject's global sequence, and the training distribution never contained such sparse evidence.
That code was lost in a rollback; this module restores it.

Two families, both applied to the *events* rather than to a rasterised image:

1. Camera-consistent geometry: roll about the principal point, isotropic scale, pixel shift.
   Applied to integer event coordinates before any surface is built, so no resampling happens.
2. Event statistics: random keep fraction and hot pixels, which is what actually covers the
   sparse-evidence domain.

Label mirroring is exact rather than approximate, and the two families need different treatment:

*Scale and shift are pure intrinsics changes.* Writing the event-space pixel map as
`q -> a(q - c_s) + c_s + dp` with `c_s = s K[:2,2]` the principal point in event pixels and
`s = RENDER_SCALE`, the same image is produced by leaving the 3D labels alone and using

    K' = A K,   A = [[a, 0, (1-a) c_x + dp_x / s],
                     [0, a, (1-a) c_y + dp_y / s],
                     [0, 0, 1]]

because the renderer forms its intrinsics as `s K` (`MNISTModel._intrinsics`), and per-sample K
already reaches that branch. No 3D quantity moves.

*Roll is a 3D rotation.* MANO composes as `V = R_g (V_rest - j0) + j0 + t` with
`j0 = J(betas)[0]` the LBS pivot (`ManoLayer.forward`: the root transform contributes
`j0 - R_g j0`). Requiring `Q V` for `Q = R_z(theta)` gives the exact label transform

    R_g <- Q R_g,     t <- Q (j0 + t) - j0

and *not* `t <- Q t`, which is the tempting but wrong version. A z-rotation of 3D points equals
a rotation of pixels about the principal point iff `fx == fy`; this camera has
`fx/fy = 1.0008`, leaving a sub-pixel residual that :func:`roll_pixel_residual_px` measures.

Everything is off by default. With `AUG.DOMRAND.ENABLED: false` the dataset path is bitwise
identical to the legacy loader.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class DomRandConfig:
    """Randomisation ranges. Defaults reproduce the §11 recipe."""

    enabled: bool = False
    #: roll about the principal point, degrees, uniform in [-roll_deg, +roll_deg]
    roll_deg: float = 15.0
    #: isotropic scale, log-uniform in [scale_min, scale_max]
    scale_min: float = 0.8
    scale_max: float = 1.25
    #: pixel shift in event coordinates, uniform in [-shift_px, +shift_px] per axis
    shift_px: float = 14.0
    #: random keep fraction, uniform in [keep_min, keep_max]; 1.0 keeps everything
    keep_min: float = 0.25
    keep_max: float = 1.0
    #: expected hot-pixel events per (pixel, polarity, ms), i.e. per LNES slot per frame
    hot_pixel_rate: float = 2.0e-4
    #: probability of applying the geometric family at all (1.0 = always)
    p_geometric: float = 1.0
    #: probability of applying the event-statistics family at all
    p_event_stats: float = 1.0

    @classmethod
    def from_cfg(cls, cfg: dict | None) -> "DomRandConfig":
        d = ((cfg or {}).get("AUG", {}) or {}).get("DOMRAND", {}) or {}
        known = {f: getattr(cls, f) for f in cls.__dataclass_fields__}
        unknown = sorted(set(k.lower() for k in d) - set(known))
        if unknown:
            # A silently ignored key would be a config-level ablation that never reached the
            # data, which is exactly the failure mode `MODEL_KEYS` was added to prevent.
            raise ValueError(f"unknown AUG.DOMRAND keys: {unknown}")
        vals = {}
        for f, default in known.items():
            v = d.get(f.upper(), d.get(f, default))
            vals[f] = bool(v) if isinstance(default, bool) else type(default)(v)
        return cls(**vals)


@dataclass
class DomRandSample:
    """One realisation, kept explicit so tests can assert the label mirroring."""

    roll_rad: float
    scale: float
    shift_px: Tuple[float, float]
    keep: float
    n_hot: int

    @property
    def geometric_identity(self) -> bool:
        return (
            self.roll_rad == 0.0
            and self.scale == 1.0
            and self.shift_px == (0.0, 0.0)
        )


def sample_params(cfg: DomRandConfig, rng: np.random.Generator,
                  n_slots: int) -> DomRandSample:
    """Draw one realisation. `n_slots = H*W*2*window_ms` sets the hot-pixel count."""
    if not cfg.enabled:
        return DomRandSample(0.0, 1.0, (0.0, 0.0), 1.0, 0)
    if rng.random() < cfg.p_geometric:
        roll = float(np.deg2rad(rng.uniform(-cfg.roll_deg, cfg.roll_deg)))
        scale = float(np.exp(rng.uniform(np.log(cfg.scale_min), np.log(cfg.scale_max))))
        shift = (float(rng.uniform(-cfg.shift_px, cfg.shift_px)),
                 float(rng.uniform(-cfg.shift_px, cfg.shift_px)))
    else:
        roll, scale, shift = 0.0, 1.0, (0.0, 0.0)
    if rng.random() < cfg.p_event_stats:
        keep = float(rng.uniform(cfg.keep_min, cfg.keep_max))
        n_hot = int(rng.poisson(cfg.hot_pixel_rate * n_slots))
    else:
        keep, n_hot = 1.0, 0
    return DomRandSample(roll, scale, shift, keep, n_hot)


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rodrigues(rv: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(rv))
    if th < 1e-12:
        return np.eye(3)
    k = rv / th
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(th) * K + (1.0 - np.cos(th)) * (K @ K)


def _log_so3(R: np.ndarray) -> np.ndarray:
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    th = float(np.arccos(tr))
    if th < 1e-12:
        return np.zeros(3)
    if np.pi - th < 1e-6:
        # Near pi the skew part vanishes; take the axis from the symmetric part.
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, None))
        i = int(np.argmax(axis))
        if axis[i] > 0:
            axis = A[:, i] / axis[i]
        n = np.linalg.norm(axis)
        return (axis / n * th) if n > 0 else np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * (th / (2.0 * np.sin(th)))


def intrinsics_matrix(sample: DomRandSample, camera_K: np.ndarray,
                      render_scale: float) -> np.ndarray:
    """`A` such that `K' = A K` realises the scale+shift part in event pixels."""
    a = sample.scale
    dpx, dpy = sample.shift_px
    cx, cy = float(camera_K[0, 2]), float(camera_K[1, 2])
    A = np.eye(3, dtype=np.float64)
    A[0, 0] = A[1, 1] = a
    A[0, 2] = (1.0 - a) * cx + dpx / render_scale
    A[1, 2] = (1.0 - a) * cy + dpy / render_scale
    return A


def transform_labels(pos51: np.ndarray, sample: DomRandSample, camera_K: np.ndarray,
                     j0: np.ndarray, render_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mirror one realisation onto (51D state, camera_K).

    `pos51` may be a single row or a stack; the local 45 residuals are untouched because a
    global rotation does not change any local joint rotation.
    """
    out = np.array(pos51, dtype=np.float32, copy=True)
    K_new = np.asarray(camera_K, dtype=np.float32).copy()
    if sample.roll_rad != 0.0:
        Q = rot_z(sample.roll_rad)
        flat = out.reshape(-1, out.shape[-1])
        for i in range(flat.shape[0]):
            t = flat[i, 0:3].astype(np.float64)
            Rg = _rodrigues(flat[i, 3:6].astype(np.float64))
            flat[i, 0:3] = (Q @ (j0 + t) - j0).astype(np.float32)
            flat[i, 3:6] = _log_so3(Q @ Rg).astype(np.float32)
        out = flat.reshape(out.shape)
    if sample.scale != 1.0 or sample.shift_px != (0.0, 0.0):
        A = intrinsics_matrix(sample, K_new.astype(np.float64), render_scale)
        K_new = (A @ K_new.astype(np.float64)).astype(np.float32)
    return out, K_new


def transform_events(xs: np.ndarray, ys: np.ndarray, sample: DomRandSample,
                     camera_K: np.ndarray, render_scale: float, width: int,
                     height: int) -> np.ndarray:
    """Map event pixels; returns the in-bounds mask and edits `xs`/`ys` in place.

    The map is `q -> a R(theta) (q - c_s) + c_s + dp` with `c_s` the principal point in event
    pixels. Rounding happens once, here, so no intermediate surface is resampled.
    """
    if sample.geometric_identity:
        return np.ones(len(xs), dtype=bool)
    cxs = float(camera_K[0, 2]) * render_scale
    cys = float(camera_K[1, 2]) * render_scale
    c, s = np.cos(sample.roll_rad), np.sin(sample.roll_rad)
    a = sample.scale
    dx, dy = sample.shift_px
    u = xs.astype(np.float64) - cxs
    v = ys.astype(np.float64) - cys
    nu = a * (c * u - s * v) + cxs + dx
    nv = a * (s * u + c * v) + cys + dy
    xi = np.rint(nu)
    yi = np.rint(nv)
    keep = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    xs[:] = np.where(keep, xi, 0).astype(xs.dtype)
    ys[:] = np.where(keep, yi, 0).astype(ys.dtype)
    return keep


def roll_pixel_residual_px(camera_K: np.ndarray, roll_rad: float, width: int,
                           height: int, render_scale: float) -> float:
    """Max pixel disagreement between the 3D z-rotation and an exact pixel rotation.

    Zero when `fx == fy`. Reported so the `fx/fy = 1.0008` approximation is a measured
    sub-pixel quantity rather than an assumption.
    """
    fx, fy = float(camera_K[0, 0]), float(camera_K[1, 1])
    cxs, cys = float(camera_K[0, 2]) * render_scale, float(camera_K[1, 2]) * render_scale
    gy, gx = np.meshgrid(np.arange(height, dtype=np.float64),
                         np.arange(width, dtype=np.float64), indexing="ij")
    u, v = gx - cxs, gy - cys
    c, s = np.cos(roll_rad), np.sin(roll_rad)
    # Exact pixel rotation
    ru, rv = c * u - s * v, s * u + c * v
    # What a 3D z-rotation produces: x = u z /(s fx), y = v z /(s fy); rotate, reproject.
    ax, ay = u / fx, v / fy
    bx, by = c * ax - s * ay, s * ax + c * ay
    pu, pv = bx * fx, by * fy
    return float(np.max(np.hypot(ru - pu, rv - pv)))


def sample_hot_pixels(n_hot: int, window_ms: int, width: int, height: int,
                      rng: np.random.Generator) -> Optional[np.ndarray]:
    """`(n_hot, 4)` int64 array of `[ms_offset, x, y, p]` noise events, or None."""
    if n_hot <= 0:
        return None
    return np.stack([
        rng.integers(0, window_ms, n_hot),
        rng.integers(0, width, n_hot),
        rng.integers(0, height, n_hot),
        rng.integers(0, 2, n_hot),
    ], axis=1).astype(np.int64)


def keep_mask(n: int, keep: float, rng: np.random.Generator) -> Optional[np.ndarray]:
    """Independent Bernoulli keep mask, or None when nothing is dropped."""
    if keep >= 1.0 or n == 0:
        return None
    return rng.random(n) < keep
