"""XAG diagnostic pseudo-sources, indexed by the original immutable event row.

These are GT-mesh geometric labels, not measured physical event identities.  Labels
are defined before augmentation and travel with their event; injected hot pixels
have unknown sources.  Importing this module has no CUDA or RNG side effects.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

SOURCE_VERSION = "xag-source-v2-fg2px-depth10mm-face-majority-fk1"
SOURCE_FIELDS = ("xag_source_index", "xag_source_ms", "xag_part", "xag_face",
                 "xag_bary", "xag_valid")
DEPTH_JUMP, CONTOUR_R = 0.010, 2.0
_OFFS = sorted([(dx, dy) for dx in range(-2, 3) for dy in range(-2, 3)
                if dx * dx + dy * dy <= CONTOUR_R ** 2],
               key=lambda o: (o[0] ** 2 + o[1] ** 2, o))


def _shift(a, dx, dy, fill):
    out = np.full_like(a, fill)
    h, w = a.shape
    y0, y1, x0, x1 = max(0, -dy), min(h, h - dy), max(0, -dx), min(w, w - dx)
    out[y0:y1, x0:x1] = a[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
    return out


def source_maps(Z, P, FID):
    """Extend XA ``label_maps`` with the exact foreground source pixel and face.

    The part/kind/depth operations and tie order are the XA rules verbatim.  A
    depth discontinuity's far pixel first inherits the nearer 4-neighbour; then
    events inherit the nearest contour within 2 px (ties: foreground depth).
    """
    h, w = Z.shape
    hand = np.isfinite(Z)
    pixels = np.arange(h * w, dtype=np.int64).reshape(h, w)
    fgZ, fgP, fgFID, fgPIX = Z.copy(), P.copy(), FID.copy(), pixels.copy()
    Zp, Pp = np.pad(Z, 1, constant_values=np.nan), np.pad(P, 1, constant_values=-1)
    Fp, Ip = np.pad(FID, 1, constant_values=-1), np.pad(pixels, 1, constant_values=-1)
    sil, disc = np.zeros((h, w), bool), np.zeros((h, w), bool)
    with np.errstate(invalid="ignore"):
        for sl in ((slice(0, -2), slice(1, -1)), (slice(2, None), slice(1, -1)),
                   (slice(1, -1), slice(0, -2)), (slice(1, -1), slice(2, None))):
            Zn, Pn = Zp[sl], Pp[sl]
            inimg, nhand = ~np.isnan(Zn), np.isfinite(Zn)
            sil |= hand & inimg & ~nhand
            far, near = hand & nhand & ((Z - Zn) > DEPTH_JUMP), hand & nhand & ((Zn - Z) > DEPTH_JUMP)
            disc |= far | near
            upd = far & (Zn < fgZ)
            fgZ, fgP = np.where(upd, Zn, fgZ), np.where(upd, Pn, fgP)
            fgFID, fgPIX = np.where(upd, Fp[sl], fgFID), np.where(upd, Ip[sl], fgPIX)
    contour = sil | disc
    ckind = np.where(sil, 1, 2).astype(np.int8)
    bestd, bestz = np.full((h, w), np.inf), np.full((h, w), np.inf)
    bpart, bkind = np.full((h, w), -1, np.int8), np.full((h, w), -1, np.int8)
    bface, bpix = np.full((h, w), -1, np.int32), np.full((h, w), -1, np.int64)
    for dx, dy in _OFFS:
        distance = float(np.hypot(dx, dy))
        c = _shift(contour, dx, dy, False)
        zz = _shift(np.where(contour, fgZ, np.inf), dx, dy, np.inf)
        better = c & ((distance < bestd) | ((distance == bestd) & (zz < bestz)))
        bestd, bestz = np.where(better, distance, bestd), np.where(better, zz, bestz)
        bpart = np.where(better, _shift(fgP, dx, dy, -1), bpart)
        bkind = np.where(better, _shift(ckind, dx, dy, -1), bkind)
        bface = np.where(better, _shift(fgFID, dx, dy, -1), bface)
        bpix = np.where(better, _shift(fgPIX, dx, dy, -1), bpix)
    has = np.isfinite(bestd)
    return {
        "part": np.where(has, bpart, np.where(hand, P, -1)).astype(np.int8),
        "kind": np.where(has, bkind, np.where(hand, 0, -1)).astype(np.int8),
        "depth": np.where(has, bestz, np.where(hand, Z, np.nan)).astype(np.float32),
        "face": np.where(has, bface, np.where(hand, FID, -1)).astype(np.int16),
        "pixel": np.where(has, bpix, np.where(hand, pixels, -1)),
    }


def source_barycentrics(verts, faces, fxyc, face, pixel, width=240):
    """Perspective-correct weights at the *source* pixel, never the event pixel.

    Uses the rasterizer's projected triangle convention, clamps only its 1e-5
    shared-edge slack, and pads unknown sources with zero weights.
    """
    out = np.zeros((len(face), 3), np.float32)
    good = (face >= 0) & (pixel >= 0)
    if not good.any():
        return out
    tri = np.asarray(verts, np.float64)[np.asarray(faces)[face[good].astype(np.int64)]]
    fx, fy, cx, cy = map(float, fxyc)
    u, v = fx * tri[..., 0] / tri[..., 2] + cx, fy * tri[..., 1] / tri[..., 2] + cy
    x, y = pixel[good] % width, pixel[good] // width
    den = (u[:, 1] - u[:, 0]) * (v[:, 2] - v[:, 0]) - (u[:, 2] - u[:, 0]) * (v[:, 1] - v[:, 0])
    b1 = ((x - u[:, 0]) * (v[:, 2] - v[:, 0]) - (u[:, 2] - u[:, 0]) * (y - v[:, 0])) / den
    b2 = ((u[:, 1] - u[:, 0]) * (y - v[:, 0]) - (x - u[:, 0]) * (v[:, 1] - v[:, 0])) / den
    affine = np.stack([1.0 - b1 - b2, b1, b2], -1)
    if not np.isfinite(affine).all() or (affine < -1.1e-5).any():
        raise ValueError("source pixel is outside its rasterized source face")
    weights = np.maximum(affine, 0.0) / tri[..., 2]
    weights /= weights.sum(-1, keepdims=True)
    out[good] = weights.astype(np.float32)
    return out


@lru_cache(maxsize=96)
def _open_sources(root: str, seq: str, pid: int):
    del pid  # part of the cache key: workers never reuse a parent's open maps
    base = Path(root) / seq
    meta = json.loads((base / "metadata.json").read_text())
    if meta.get("version") != SOURCE_VERSION or meta.get("seq") != seq:
        raise ValueError(f"incompatible XAG source cache: {base}")
    arrays = {key: np.load(base / (key + ".npy"), mmap_mode="r", allow_pickle=False)
              for key in ("part", "face", "bary", "offsets", "done_ms")}
    return meta, arrays


def load_sources(root, seq, row_ids, event_ms):
    """Return the six numpy XAG fields in caller order, including repeated rows.

    ``row_ids=-1`` is reserved for synthetic hot pixels.  Every other row must
    agree with its exact original millisecond and have a completed cache record;
    stale/incomplete caches fail instead of silently assigning unknown labels.
    """
    rows = np.asarray(row_ids, dtype=np.int64)
    ms = np.asarray(event_ms, dtype=np.int64)
    if rows.ndim != 1 or rows.shape != ms.shape or (rows < -1).any():
        raise ValueError("XAG row_ids/event_ms must be matching vectors with row >= -1")
    out = {"xag_source_index": rows.copy(), "xag_source_ms": ms.copy(),
           "xag_part": np.full(len(rows), -1, np.int64), "xag_face": np.full(len(rows), -1, np.int64),
           "xag_bary": np.zeros((len(rows), 3), np.float32), "xag_valid": np.zeros(len(rows), bool)}
    real = rows >= 0
    if not real.any():
        return out
    meta, a = _open_sources(str(Path(root).resolve()), str(seq), os.getpid())
    rr, mm = rows[real], ms[real]
    if (rr >= meta["n_events"]).any() or (mm < 0).any() or (mm >= len(a["done_ms"])).any():
        raise IndexError(f"XAG source row/time out of range: {seq}")
    if not np.all((rr >= a["offsets"][mm]) & (rr < a["offsets"][mm + 1])):
        raise ValueError(f"XAG original event row/millisecond mismatch: {seq}")
    if not np.all(a["done_ms"][mm]):
        raise RuntimeError(f"XAG source cache incomplete at requested milliseconds: {seq}")
    out["xag_part"][real] = a["part"][rr]
    out["xag_face"][real] = a["face"][rr]
    out["xag_bary"][real] = a["bary"][rr]
    out["xag_valid"][real] = (a["part"][rr] >= 0) & (a["face"][rr] >= 0)
    return out
