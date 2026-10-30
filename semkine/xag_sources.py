"""XAG diagnostic pseudo-sources, indexed by the original immutable event row.

These are GT-mesh geometric labels, not measured physical event identities.  Labels
are defined before augmentation and travel with their event; injected hot pixels
have unknown sources.  Importing this module has no CUDA or RNG side effects.
"""
from __future__ import annotations

import json
import os
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

SOURCE_VERSION = "xag-source-v2-fg2px-depth10mm-face-majority-fk1"
SOURCE_FIELDS = ("xag_source_index", "xag_source_ms", "xag_part", "xag_face",
                 "xag_bary", "xag_valid")
DEPTH_VERSION = "xcl-source-depth-v1-source-ms-foreground-fk1"
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


@lru_cache(maxsize=96)
def _open_depth(root: str, seq: str, pid: int):
    """Open only a completely published depth sidecar bound to its legacy cache."""
    del pid
    base = Path(root) / seq
    source_bytes = (base / "metadata.json").read_bytes()
    source = json.loads(source_bytes)
    meta = json.loads((base / "depth_metadata.json").read_text())
    if (meta.get("version") != DEPTH_VERSION or meta.get("seq") != seq
            or meta.get("source_version") != SOURCE_VERSION
            or meta.get("source_metadata_sha256") != hashlib.sha256(source_bytes).hexdigest()
            or meta.get("n_events") != source.get("n_events")
            or meta.get("n_ms") != source.get("n_ms")):
        raise ValueError(f"incompatible XCL depth sidecar: {base}")
    if not source.get("complete") or not meta.get("complete"):
        raise RuntimeError(f"XCL depth sidecar incomplete: {base}")
    # Verify complete files once per process/sequence, before any oracle label is
    # consumed. Numeric range checks alone would accept a corrupted positive z.
    for name in ("depth.npy", "depth_done_ms.npy"):
        expected = meta.get("outputs_sha256", {}).get(name)
        digest = hashlib.sha256()
        with (base / name).open("rb") as stream:
            for block in iter(lambda: stream.read(8 << 20), b""):
                digest.update(block)
        if expected != digest.hexdigest():
            raise ValueError(f"corrupt XCL depth sidecar checksum: {base / name}")
    arrays = {key: np.load(base / (key + ".npy"), mmap_mode="r", allow_pickle=False)
              for key in ("depth", "depth_done_ms")}
    if (arrays["depth"].shape != (source["n_events"],) or arrays["depth"].dtype != np.float32
            or arrays["depth_done_ms"].shape != (source["n_ms"],)
            or arrays["depth_done_ms"].dtype != np.bool_):
        raise ValueError(f"corrupt XCL depth sidecar shape/dtype: {base}")
    return arrays


def load_sources(root, seq, row_ids, event_ms, *, require_depth=False):
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
    if require_depth:
        out["xag_depth"] = np.full(len(rows), np.nan, np.float32)
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
    if require_depth:
        d = _open_depth(str(Path(root).resolve()), str(seq), os.getpid())
        if not np.all(d["depth_done_ms"][mm]):
            raise RuntimeError(f"XCL depth sidecar incomplete at requested milliseconds: {seq}")
        values = np.asarray(d["depth"][rr])
        known = out["xag_valid"][real]
        if (not np.all(np.isfinite(values[known]) & (values[known] > 0))
                or not np.all(np.isnan(values[~known]))):
            raise ValueError(f"corrupt XCL source depth/validity: {seq}")
        out["xag_depth"][real] = values
    return out


def depth_manifest(root, sequences):
    """Validate all requested sidecars and bind their complete physical files.

    This deliberately rehashes each file once at acceptance. Runtime supervisors
    may then call :func:`verify_depth_manifest` for cheap metadata/stat checks.
    Callers supply the fixed split's sequence names; no directory discovery can
    accidentally accept a subset as an all-sequence release.
    """
    root = Path(root).resolve()
    sequences = list(sequences)
    if not sequences or len(sequences) != len(set(sequences)):
        raise ValueError("depth manifest requires nonempty unique fixed-split sequences")
    _open_depth.cache_clear()
    entries = {}
    for seq in sequences:
        _open_depth(str(root), str(seq), os.getpid())
        base = root / seq
        meta_path = base / "depth_metadata.json"
        meta = json.loads(meta_path.read_text())
        files = {}
        for name in ("depth.npy", "depth_done_ms.npy"):
            stat = (base / name).stat()
            files[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                           "sha256": meta["outputs_sha256"][name]}
        entries[seq] = {"metadata_sha256": hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                        "source_metadata_sha256": meta["source_metadata_sha256"],
                        "n_events": meta["n_events"], "n_ms": meta["n_ms"], "files": files}
    return {"version": DEPTH_VERSION, "source_root": str(root), "complete": True,
            "sequences": sequences, "entries": entries}


def verify_depth_manifest(manifest, sequences=None):
    """Fail closed on a changed/missing sidecar after full SHA acceptance."""
    if manifest.get("version") != DEPTH_VERSION or not manifest.get("complete"):
        raise ValueError("incomplete or incompatible XCL depth manifest")
    listed = manifest.get("sequences", [])
    if sequences is not None and set(listed) != set(sequences):
        raise ValueError("XCL depth manifest differs from fixed split")
    if not listed or set(listed) != set(manifest.get("entries", {})):
        raise ValueError("XCL depth manifest sequence entries differ")
    root = Path(manifest["source_root"])
    for seq in listed:
        entry, base = manifest["entries"][seq], root / seq
        for name, key in (("depth_metadata.json", "metadata_sha256"),
                          ("metadata.json", "source_metadata_sha256")):
            if hashlib.sha256((base / name).read_bytes()).hexdigest() != entry[key]:
                raise ValueError(f"XCL depth manifest metadata changed: {base / name}")
        for name, expected in entry["files"].items():
            stat = (base / name).stat()
            if stat.st_size != expected["bytes"] or stat.st_mtime_ns != expected["mtime_ns"]:
                raise ValueError(f"XCL depth manifest array changed: {base / name}")
    return True
