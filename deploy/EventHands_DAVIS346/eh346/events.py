"""Sensor events -> the network's 240 x 180 LNES.

SensorMap      sensor pixel (x, y) of any resolution (DAVIS346: 346 x 260) -> LNES pixel, the same integer mapping
               the training data used for its 640 x 480 sensor (`x * 240 // W`, `y * 180 // H`), optional 180-degree
               rotation, optional lens undistortion (per-pixel look-up table); and the matching intrinsics.
EventBuffer    time-ordered events in 1 ms bins relative to a session origin t0 (`ms = (t_us - t0_us) // 1000`).
build_lnes     LNES of the window [end - w + 1, end] ms: per (pixel, polarity) the latest event's (ms - start) / w.
               Bit-identical to the training repository's `semkine.events.splat_event_image` ("last" channel).
"""
from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np

H, W = 180, 240


class SensorMap:
    def __init__(self, width: int, height: int, K: np.ndarray, dist: Optional[np.ndarray] = None,
                 rot180: bool = False, hotpix: Optional[np.ndarray] = None):
        """K: 3 x 3 intrinsics of the sensor in its own pixels (OpenCV convention). dist: OpenCV distortion
        coefficients or None. hotpix: (height, width) bool, True = drop the pixel's events."""
        self.w, self.h, self.rot180 = int(width), int(height), bool(rot180)
        K = np.asarray(K, np.float64).reshape(3, 3).copy()
        if self.rot180:
            K[0, 2], K[1, 2] = (self.w - 1) - K[0, 2], (self.h - 1) - K[1, 2]
            if hotpix is not None:
                hotpix = hotpix[::-1, ::-1]
        self.K_sensor = K
        self.keep = None if hotpix is None else ~np.asarray(hotpix, bool).reshape(self.h, self.w)
        self.lut = None
        if dist is not None and np.any(np.abs(np.asarray(dist, np.float64)) > 0):
            import cv2
            d = np.asarray(dist, np.float64).reshape(-1)
            if self.rot180:            # distortion is defined on the unrotated sensor: undistort there, then rotate
                K0 = np.asarray(K, np.float64).copy()
                K0[0, 2], K0[1, 2] = (self.w - 1) - K[0, 2], (self.h - 1) - K[1, 2]
            else:
                K0 = K
            ys, xs = np.mgrid[0:self.h, 0:self.w]
            pts = np.stack([xs.ravel(), ys.ravel()], -1).astype(np.float64)[:, None, :]
            und = cv2.undistortPoints(pts, K0, d, P=K0).reshape(self.h, self.w, 2)
            if self.rot180:
                und = np.stack([(self.w - 1) - und[..., 0], (self.h - 1) - und[..., 1]], -1)[::-1, ::-1]
            lx = np.floor(und[..., 0] * W / self.w).astype(np.int32)
            ly = np.floor(und[..., 1] * H / self.h).astype(np.int32)
            ok = (lx >= 0) & (lx < W) & (ly >= 0) & (ly < H)
            self.lut = (np.where(ok, lx, 0).astype(np.uint8), np.where(ok, ly, 0).astype(np.uint8), ok)

    def K_model(self) -> np.ndarray:
        """The intrinsics the network expects: the 640 x 480 convention of its training camera, i.e. K such that
        K * 0.375 is the LNES image's (240 x 180) projection. Each sensor axis is scaled on its own."""
        S = np.diag([640.0 / self.w, 480.0 / self.h, 1.0])
        return (S @ self.K_sensor).astype(np.float32)

    def map(self, x: np.ndarray, y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sensor events -> (x240 uint8, y180 uint8, polarity uint8 in {0, 1}, kept mask)."""
        x = np.asarray(x, np.int32)
        y = np.asarray(y, np.int32)
        if self.rot180:
            x, y = (self.w - 1) - x, (self.h - 1) - y
        ok = (x >= 0) & (x < self.w) & (y >= 0) & (y < self.h)
        if self.keep is not None:
            ok &= self.keep[np.clip(y, 0, self.h - 1), np.clip(x, 0, self.w - 1)]
        if self.lut is not None:
            xc, yc = np.clip(x, 0, self.w - 1), np.clip(y, 0, self.h - 1)
            ok &= self.lut[2][yc, xc]
            xo, yo = self.lut[0][yc, xc], self.lut[1][yc, xc]
        else:
            xo = (x * W // self.w).astype(np.uint8)
            yo = (y * H // self.h).astype(np.uint8)
        po = (np.asarray(p).astype(np.int64) > 0).astype(np.uint8)
        return xo[ok], yo[ok], po[ok], ok


class EventBuffer:
    """Mapped events, appended in time order, binned to integer ms relative to `t0_us`. Thread-safe."""

    def __init__(self, t0_us: int, capacity: int = 1 << 22):
        self.t0 = int(t0_us)
        self.ms = np.empty(capacity, np.int64)
        self.xyp = np.empty((capacity, 3), np.uint8)
        self.lo, self.hi = 0, 0
        self.latest_ms = -1
        self.lock = threading.Lock()

    def push(self, t_us: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray):
        if not len(t_us):
            return
        ms = (np.asarray(t_us, np.int64) - self.t0) // 1000
        with self.lock:
            # the stream is time ordered; anything older than the last bin is put into the last bin
            ms = np.maximum.accumulate(np.maximum(ms, self.latest_ms))
            n = len(ms)
            if self.hi + n > len(self.ms):
                self._compact(n)
            self.ms[self.hi:self.hi + n] = ms
            self.xyp[self.hi:self.hi + n, 0] = x
            self.xyp[self.hi:self.hi + n, 1] = y
            self.xyp[self.hi:self.hi + n, 2] = p
            self.hi += n
            self.latest_ms = int(ms[-1])

    def _compact(self, need: int):
        live = self.hi - self.lo
        cap = len(self.ms)
        while live + need > cap // 2:
            cap *= 2
        ms, xyp = np.empty(cap, np.int64), np.empty((cap, 3), np.uint8)
        ms[:live], xyp[:live] = self.ms[self.lo:self.hi], self.xyp[self.lo:self.hi]
        self.ms, self.xyp, self.lo, self.hi = ms, xyp, 0, live

    def prune(self, keep_from_ms: int):
        with self.lock:
            self.lo = self.lo + int(np.searchsorted(self.ms[self.lo:self.hi], keep_from_ms, side="left"))

    def _range(self, lo_ms: int, hi_ms: int):
        a = self.lo + int(np.searchsorted(self.ms[self.lo:self.hi], lo_ms, side="left"))
        b = self.lo + int(np.searchsorted(self.ms[self.lo:self.hi], hi_ms, side="right"))
        return a, b

    def count(self, lo_ms: int, hi_ms: int) -> int:
        with self.lock:
            a, b = self._range(lo_ms, hi_ms)
            return b - a

    def window(self, end_ms: int, w: int):
        """Events of [end - w + 1, end]: (xyp (N, 3) uint8, ms_rel (N,) int64 in [0, w))."""
        with self.lock:
            a, b = self._range(end_ms - w + 1, end_ms)
            return self.xyp[a:b].copy(), self.ms[a:b] - (end_ms - w + 1)


def build_lnes(xyp: np.ndarray, ms_rel: np.ndarray, w: int) -> np.ndarray:
    """(180, 240, 2) float32 LNES: the latest event's (ms - start) / w per (pixel, polarity), 0 elsewhere."""
    out = np.zeros((H, W, 2), np.float32)
    if not len(xyp):
        return out
    tval = ms_rel.astype(np.float32) / float(w)
    out[xyp[:, 1].astype(np.intp), xyp[:, 0].astype(np.intp), np.clip(xyp[:, 2], 0, 1).astype(np.intp)] = tval
    return out
