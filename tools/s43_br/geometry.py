"""CPU MANO geometry and a small, explicit triangle radiance provider.

This module never reads events, labels, RGB, or recorded predictions.  Geometry is
read-only import of the repository MANO layer.  The public constant/vertex texture
is an assumed imaging model, not a recovered texture of any recorded hand.  Its
likelihood must therefore be identified as a model-conditional score on recordings.

All states use the S43 storage convention: translation in m, root axis-angle in
rad, and 45 residual local axis-angles (the MANO mean is added exactly once).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from model.mano_layer import ManoLayer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET = ROOT / "assets" / "mano_right.npz"


@lru_cache(maxsize=2)
def _mano(asset: str = str(DEFAULT_ASSET)) -> ManoLayer:
    model = ManoLayer(asset, add_mean=True).cpu().eval()
    model.fast_chain = True
    return model


def _states_array(states: np.ndarray) -> np.ndarray:
    x = np.asarray(states, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != 51 or not np.isfinite(x).all():
        raise ValueError("states must be finite (...,51), in metres/radians")
    return x


def fk_geometry(states: np.ndarray, betas: np.ndarray,
                asset: str | Path = DEFAULT_ASSET) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices (...,778,3) and OpenPose joints (...,21,3), both in m."""
    x = _states_array(states)
    lead = x.shape[:-1]
    beta = np.asarray(betas, dtype=np.float32)
    if beta.shape == (10,):
        beta = np.broadcast_to(beta, lead + (10,))
    elif beta.shape != lead + (10,):
        raise ValueError("betas must have shape (10,) or states.shape[:-1]+(10,)")
    if not np.isfinite(beta).all():
        raise ValueError("betas must be finite")
    flat = torch.from_numpy(np.array(x.reshape(-1, 51), dtype=np.float32))
    bt = torch.from_numpy(np.array(beta.reshape(-1, 10), dtype=np.float32))
    model = _mano(str(Path(asset).resolve()))
    with torch.no_grad():
        verts, joints = model(bt, flat[:, 3:6], flat[:, 6:], flat[:, :3])
    return (verts.numpy().reshape(lead + (778, 3)),
            joints.numpy().reshape(lead + (21, 3)))


def fk_joints(states: np.ndarray, betas: np.ndarray,
              asset: str | Path = DEFAULT_ASSET) -> np.ndarray:
    return fk_geometry(states, betas, asset)[1]


def scale_intrinsics(K_cam: np.ndarray, image_size: tuple[int, int],
                     source_size: tuple[int, int] = (180, 240)) -> np.ndarray:
    """Scale the given camera once; sizes are (height,width), no centre recrop.

    Pixel-centre convention is x_low=(x_source+.5)*scale-.5.  The same map must
    be used by an event adapter; the helper :func:`map_pixels` implements it.
    """
    k = np.asarray(K_cam, dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("K_cam must be finite (3,3)")
    h, w = map(int, image_size)
    hs, ws = map(int, source_size)
    if min(h, w, hs, ws) < 1:
        raise ValueError("positive image/source dimensions required")
    out = k.copy()
    sx, sy = w / ws, h / hs
    out[0] *= sx
    out[1] *= sy
    out[0, 2] += .5 * sx - .5
    out[1, 2] += .5 * sy - .5
    return out


def map_pixels(x: np.ndarray, y: np.ndarray,
               image_size: tuple[int, int],
               source_size: tuple[int, int] = (180, 240)) -> tuple[np.ndarray, np.ndarray]:
    """Map source pixel *cells* to low-resolution integer pixel cells.

    Pixel-centre coordinates map by scale_intrinsics; rounding their mapped
    centre gives floor((x+.5)*scale), clamped only at the upper valid boundary.
    Invalid source coordinates raise instead of silently creating valid events.
    """
    x, y = np.asarray(x), np.asarray(y)
    h, w = image_size
    hs, ws = source_size
    if np.any(x < 0) or np.any(x >= ws) or np.any(y < 0) or np.any(y >= hs):
        raise ValueError("event pixel outside source image")
    return (np.minimum(np.floor((x + .5) * w / ws).astype(np.int64), w - 1),
            np.minimum(np.floor((y + .5) * h / hs).astype(np.int64), h - 1))


def _vertex_albedo(model: ManoLayer, texture_seed: int | None,
                   foreground: float) -> np.ndarray:
    if not 0 < foreground <= 1:
        raise ValueError("foreground must lie in (0,1]")
    if texture_seed is None:
        return np.full(778, foreground, dtype=np.float64)
    # Fixed public canonical texture, generated without state, labels or RGB.
    # Smooth template-coordinate texture avoids using a face's arbitrary index
    # as a privileged segmentation label.
    rng = np.random.default_rng(int(texture_seed))
    canonical = model.v_template.cpu().numpy().astype(np.float64)
    direction = rng.normal(size=(3, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    freq = np.array([125., 205., 315.])
    phase = rng.uniform(-np.pi, np.pi, size=3)
    pattern = np.sin(canonical @ (direction * freq[:, None]).T + phase).mean(1)
    return np.clip(foreground + .16 * pattern, .04, .98)


def rasterize(vertices: np.ndarray, faces: np.ndarray, K_low: np.ndarray,
              image_size: tuple[int, int], vertex_albedo: np.ndarray,
              background: float = .15, near: float = .03,
              supersample: int = 1) -> np.ndarray:
    """Two-sided perspective-correct triangle z-buffer in linear intensity.

    No labels or historical point-splat image enter this renderer.  Occluding
    faces win by nearest positive depth.  Subpixel anti-aliasing, when enabled,
    averages *linear intensity*, and log is applied only by render_paths.
    Triangles crossing the near plane are skipped (the prototype's supported
    domain requires all visible vertices in front of this plane).
    """
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    k = np.asarray(K_low, dtype=np.float64)
    albedo = np.asarray(vertex_albedo, dtype=np.float64)
    h, w = map(int, image_size)
    if v.shape != (778, 3) or f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("expected MANO vertices and triangular faces")
    if k.shape != (3, 3) or albedo.shape != (778,):
        raise ValueError("camera/albedo shape")
    if not np.isfinite(v).all() or not np.isfinite(k).all() or not np.isfinite(albedo).all():
        raise ValueError("finite rendering inputs required")
    if supersample not in (1, 2, 3, 4) or not 0 < background <= 1:
        raise ValueError("supported supersample=1..4 and background in (0,1]")
    if np.min(v[:, 2]) <= near:
        raise ValueError("MANO intersects near plane; unsupported prototype geometry")
    hh, ww = h * supersample, w * supersample
    if supersample > 1:
        k = k.copy()
        k[:2] *= supersample
        k[:2, 2] += .5 * supersample - .5
    xyh = v @ k.T
    xy = xyh[:, :2] / xyh[:, 2:3]
    image = np.full((hh, ww), background, dtype=np.float64)
    depth = np.full((hh, ww), np.inf, dtype=np.float64)
    # Tiny faces dominate at this resolution.  Bounding boxes are checked before
    # allocating local arrays so most off-grid/subpixel faces are inexpensive.
    for tri_ids in f:
        tri = xy[tri_ids]
        left = max(0, int(np.ceil(tri[:, 0].min())))
        right = min(ww - 1, int(np.floor(tri[:, 0].max())))
        top = max(0, int(np.ceil(tri[:, 1].min())))
        bottom = min(hh - 1, int(np.floor(tri[:, 1].max())))
        if left > right or top > bottom:
            continue
        x0, y0 = tri[0]
        x1, y1 = tri[1]
        x2, y2 = tri[2]
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-14:
            continue
        yy, xx = np.mgrid[top:bottom + 1, left:right + 1]
        w0 = ((y1 - y2) * (xx - x2) + (x2 - x1) * (yy - y2)) / den
        w1 = ((y2 - y0) * (xx - x2) + (x0 - x2) * (yy - y2)) / den
        w2 = 1. - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        iz = 1. / v[tri_ids, 2]
        inv_z = w0 * iz[0] + w1 * iz[1] + w2 * iz[2]
        z = np.divide(1., inv_z, out=np.full_like(inv_z, np.inf), where=inv_z > 0)
        local_depth = depth[top:bottom + 1, left:right + 1]
        update = inside & (z < local_depth)
        if not np.any(update):
            continue
        color = ((w0 * iz[0] * albedo[tri_ids[0]] +
                  w1 * iz[1] * albedo[tri_ids[1]] +
                  w2 * iz[2] * albedo[tri_ids[2]]) / inv_z)
        local_depth[update] = z[update]
        image[top:bottom + 1, left:right + 1][update] = color[update]
    if supersample > 1:
        image = image.reshape(h, supersample, w, supersample).mean(axis=(1, 3))
    return image


def render_intensity(states: np.ndarray, betas: np.ndarray, K_cam: np.ndarray,
                     image_size: tuple[int, int] = (24, 32),
                     source_size: tuple[int, int] = (180, 240),
                     texture_seed: int | None = None, foreground: float = .7,
                     background: float = .15, supersample: int = 1,
                     asset: str | Path = DEFAULT_ASSET) -> np.ndarray:
    """Render (...,H,W) public assumed radiance images, without any η scaling."""
    x = _states_array(states)
    model = _mano(str(Path(asset).resolve()))
    verts, _ = fk_geometry(x, betas, asset)
    faces = model.f.cpu().numpy()
    colors = _vertex_albedo(model, texture_seed, foreground)
    low_k = scale_intrinsics(K_cam, image_size, source_size)
    result = np.stack([rasterize(v, faces, low_k, image_size, colors,
                                 background, supersample=supersample)
                       for v in verts.reshape(-1, 778, 3)])
    return result.reshape(x.shape[:-1] + tuple(image_size))


def render_paths(states: np.ndarray, betas: np.ndarray, K_cam: np.ndarray,
                 image_size: tuple[int, int] = (24, 32),
                 scales: Sequence[float] = (.8, 1., 1.2),
                 texture_seed: int | None = None, delta: float = .12,
                 source_size: tuple[int, int] = (180, 240),
                 foreground: float = .7, background: float = .15,
                 supersample: int = 1,
                 asset: str | Path = DEFAULT_ASSET) -> np.ndarray:
    """Return int32 [K,M,P,B+1] absolute log-brightness grid paths.

    η is a shared contrast/threshold-scale branch for the complete epoch.
    Geometry is computed once per candidate/time, shared across all η branches.
    Quantization is of absolute log intensity, never rounded per-bin deltas.
    Source frames have no relation to test RGB.  `texture_seed=None` is the
    deployable, deliberately restrictive constant-foreground assumption.
    """
    x = _states_array(states)
    if x.ndim != 3 or not 2 <= x.shape[0] <= 8 or x.shape[1] < 2:
        raise ValueError("paths require states[K=2..8,B+1>=2,51]")
    eta = np.asarray(tuple(scales), dtype=np.float64)
    if eta.ndim != 1 or not len(eta) or not np.isfinite(eta).all() or np.any(eta <= 0):
        raise ValueError("positive finite contrast scales required")
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("positive finite log-intensity grid pitch required")
    images = render_intensity(x, betas, K_cam, image_size, source_size,
                              texture_seed, foreground, background, supersample, asset)
    # Fixed background is common to the η branches; scale contrast in log space.
    log_bg = np.log(background)
    levels = log_bg + eta[None, :, None, None, None] * (np.log(images[:, None]) - log_bg)
    levels = np.rint(levels / delta).astype(np.int32)
    return levels.transpose(0, 1, 3, 4, 2).reshape(x.shape[0], len(eta), -1, x.shape[1])


def _rot_exp(v: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(v))
    x, y, z = v
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    a = 1. - theta * theta / 6. if theta < 1e-7 else np.sin(theta) / theta
    b = .5 - theta * theta / 24. if theta < 1e-7 else (1. - np.cos(theta)) / theta ** 2
    return np.eye(3) + a * skew + b * (skew @ skew)


def _rot_log(R: np.ndarray) -> np.ndarray:
    theta = float(np.arccos(np.clip((np.trace(R) - 1.) / 2., -1., 1.)))
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-7:
        return .5 * skew
    if np.pi - theta < 1e-5:
        raise ValueError("rotation too close to pi for the prototype's fixed axis-angle chart")
    return theta / (2. * np.sin(theta)) * skew


def retract(state: np.ndarray, tangent: np.ndarray,
            asset: str | Path = DEFAULT_ASSET) -> np.ndarray:
    """MANO left-composition chart; local mean is restored before composition."""
    s, d = _states_array(state), _states_array(tangent)
    s, d = np.broadcast_arrays(s, d)
    out = s.copy()
    out[..., :3] += d[..., :3]
    mean = _mano(str(Path(asset).resolve())).hands_mean.cpu().numpy().reshape(15, 3)
    for i, (row, delta_row) in enumerate(zip(s.reshape(-1, 51), d.reshape(-1, 51))):
        dst = out.reshape(-1, 51)[i]
        dst[3:6] = _rot_log(_rot_exp(delta_row[3:6]) @ _rot_exp(row[3:6]))
        for j in range(15):
            sl = slice(6 + 3 * j, 9 + 3 * j)
            dst[sl] = _rot_log(_rot_exp(delta_row[sl]) @ _rot_exp(row[sl] + mean[j])) - mean[j]
    return out


def candidate_paths(history: np.ndarray, bins: int = 50,
                    amplitudes: Sequence[float] = (-.008, 0., .008),
                    axis: str | np.ndarray = "tx",
                    base_increment: np.ndarray | None = None,
                    asset: str | Path = DEFAULT_ASSET) -> np.ndarray:
    """Fixed public candidate bank generated from a history state, never GT.

    `tx` amplitudes are metres. `index_y` amplitudes are radians around the
    first MANO index joint's local y-axis; this is a coordinate intervention,
    not an unconditional claim of anatomical pure flexion.  An explicit 51D
    tangent direction can be supplied.  The bank is frozen for the epoch.
    """
    h = _states_array(history)
    if h.shape != (51,) or type(bins) is not int or bins < 1:
        raise ValueError("history[51] and positive integer bins required")
    amp = np.asarray(tuple(amplitudes), dtype=np.float64)
    if amp.ndim != 1 or not 2 <= len(amp) <= 8 or not np.isfinite(amp).all():
        raise ValueError("2..8 finite candidate amplitudes required")
    direction = np.zeros(51, dtype=np.float64)
    if isinstance(axis, str):
        if axis == "tx":
            direction[0] = 1.
        elif axis == "ty":
            direction[1] = 1.
        elif axis == "tz":
            direction[2] = 1.
        elif axis == "index_y":
            direction[7] = 1.
        else:
            raise ValueError("axis must be tx/ty/tz/index_y or an explicit tangent")
    else:
        direction = _states_array(axis)
        if direction.shape != (51,):
            raise ValueError("explicit tangent direction must be [51]")
    base = np.zeros(51) if base_increment is None else _states_array(base_increment)
    if base.shape != (51,):
        raise ValueError("base_increment must be [51]")
    fraction = np.linspace(0., 1., bins + 1)
    delta = fraction[None, :, None] * (base[None, :] + amp[:, None] * direction)[..., None, :]
    return retract(np.broadcast_to(h, delta.shape), delta, asset)


def provider_metadata(texture_seed: int | None = None, delta: float = .12,
                      image_size: tuple[int, int] = (24, 32),
                      scales: Sequence[float] = (.8, 1., 1.2)) -> dict:
    """Provenance included by experiments in their feature cache manifests."""
    return {"geometry": "read-only model.mano_layer.ManoLayer(add_mean=True)",
            "units": "translation_m_rotation_rad", "renderer": "triangle_zbuffer_linear_intensity",
            "pixel_map": "centre_scaled_then_nearest_cell", "image_size": list(image_size),
            "texture": "constant_public_assumption" if texture_seed is None else "public_synthetic_texture",
            "texture_seed": texture_seed, "uses_test_RGB_or_GT_texture": False,
            "eta_scales": list(scales), "delta_log": float(delta),
            "recording_interpretation": "assumed-model score; not calibrated real-camera likelihood"}
