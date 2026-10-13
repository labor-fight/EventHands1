"""Event sources: a live DAVIS346 (dv-processing), an AEDAT4 recording, or an .npz event file.

Every source yields batches `(t_us int64, x int, y int, p int/bool)` in time order through `read()`
(None when nothing new is available yet, StopIteration at the end of a file) and exposes `resolution` (w, h).
"""
from __future__ import annotations

import datetime
import time
from typing import Optional

import numpy as np


def _dv():
    try:
        import dv_processing as dv
        return dv
    except ImportError as e:
        raise ImportError("dv-processing is needed for the camera and AEDAT4 files: pip install dv-processing") from e


def _unpack(store):
    """dv.EventStore -> (t_us, x, y, p) numpy arrays."""
    if store is None or store.size() == 0:
        return None
    a = store.numpy()
    return (a["timestamp"].astype(np.int64), a["x"].astype(np.int32), a["y"].astype(np.int32),
            a["polarity"].astype(np.uint8))


class DavisCamera:
    """A live DAVIS346 (any iniVation camera dv-processing can open)."""

    def __init__(self, serial: str = "", ba_filter_us: int = 2000, refractory_us: int = 0, frames: bool = True,
                 sw_ba_filter_us: int = 0):
        dv = _dv()
        if hasattr(dv.io, "camera"):                                     # dv-processing >= 2.0
            self.cam = dv.io.camera.open(serial) if serial else dv.io.camera.open()
        else:                                                            # dv-processing 1.x
            self.cam = dv.io.CameraCapture(serial) if serial else dv.io.CameraCapture()
        if hasattr(self.cam, "setTimeInterval"):
            self.cam.setTimeInterval(datetime.timedelta(milliseconds=1))   # commit data every 1 ms: low latency
        if ba_filter_us and hasattr(self.cam, "setBackgroundActivityFilter"):
            self.cam.setBackgroundActivityFilter(max(1, int(round(ba_filter_us / 250))))   # on-camera BA filter
        if refractory_us and hasattr(self.cam, "setRefractoryPeriodFilter"):
            self.cam.setRefractoryPeriodFilter(max(1, int(round(refractory_us / 250))))
        if hasattr(self.cam, "setFramesRunning"):
            self.cam.setFramesRunning(bool(frames))
        r = self.cam.getEventResolution()
        self.resolution = (int(r[0]), int(r[1]))
        self.name = self.cam.getCameraName()
        self.sw_filter = None
        if sw_ba_filter_us:
            self.sw_filter = dv.noise.BackgroundActivityNoiseFilter(
                r, backgroundActivityDuration=datetime.timedelta(microseconds=int(sw_ba_filter_us)))
        self.last_frame = None
        self.live = True

    def read(self):
        if not self.cam.isRunning():
            raise StopIteration
        if hasattr(self.cam, "isFrameStreamAvailable") and self.cam.isFrameStreamAvailable():
            f = self.cam.getNextFrame()
            if f is not None:
                self.last_frame = f.image
        store = self.cam.getNextEventBatch()
        if store is not None and self.sw_filter is not None:
            self.sw_filter.accept(store)
            store = self.sw_filter.generateEvents()
        return _unpack(store)


class Aedat4File:
    """An AEDAT4 recording (e.g. recorded with DV from a DAVIS346), read as fast as the consumer asks."""

    def __init__(self, path: str, sw_ba_filter_us: int = 0):
        dv = _dv()
        self.rec = dv.io.MonoCameraRecording(str(path))
        r = self.rec.getEventResolution()
        self.resolution = (int(r[0]), int(r[1]))
        self.name = self.rec.getCameraName()
        self.sw_filter = None
        if sw_ba_filter_us:
            self.sw_filter = dv.noise.BackgroundActivityNoiseFilter(
                r, backgroundActivityDuration=datetime.timedelta(microseconds=int(sw_ba_filter_us)))
        self.last_frame = None
        self.live = False

    def read(self):
        store = self.rec.getNextEventBatch()
        if store is None:
            if not self.rec.isRunning():
                raise StopIteration
            return None
        if self.sw_filter is not None:
            self.sw_filter.accept(store)
            store = self.sw_filter.generateEvents()
        return _unpack(store)


class NpzEvents:
    """Events from an .npz with arrays t_us, x, y, p (and width, height), handed out in chunks of `chunk_ms`."""

    def __init__(self, path: str, chunk_ms: int = 1):
        z = np.load(path)
        self.t, self.x, self.y, self.p = z["t_us"].astype(np.int64), z["x"], z["y"], z["p"]
        self.resolution = (int(z["width"]), int(z["height"]))
        self.name = str(z["name"]) if "name" in z.files else "npz"
        self.chunk = int(chunk_ms) * 1000
        self.i = 0
        self.last_frame = None
        self.live = False

    def read(self):
        if self.i >= len(self.t):
            raise StopIteration
        t_end = self.t[self.i] - self.t[self.i] % self.chunk + self.chunk
        j = int(np.searchsorted(self.t, t_end, side="left"))
        j = max(j, self.i + 1)
        sl = slice(self.i, j)
        self.i = j
        return self.t[sl], self.x[sl], self.y[sl], self.p[sl]


class SimulatedLive:
    """A recording replayed as if it came from a live camera: a batch is released only once the wall clock has passed
    its last timestamp (1 ms batches), so the real-time loop (clock, step scheduling, skip-ahead, display) runs exactly
    as with the camera. Use it to test the deployment without a DAVIS346: --source sim:recording.aedat4"""

    def __init__(self, inner, speed: float = 1.0):
        self.inner, self.speed = inner, float(speed)
        self.resolution, self.name = inner.resolution, "simulated live: " + str(getattr(inner, "name", ""))
        self.last_frame = None
        self.live = True
        self.pending = None
        self.t_first = None
        self.wall0 = None
        self.eof = False

    def read(self):
        if self.pending is None:
            if self.eof:
                raise StopIteration
            try:
                b = self.inner.read()
            except StopIteration:
                self.eof = True
                raise
            if b is None:
                return None
            self.pending = b
            if self.t_first is None:
                self.t_first, self.wall0 = int(b[0][0]), time.perf_counter()
        t = self.pending[0]
        if (int(t[-1]) - self.t_first) / 1e6 > (time.perf_counter() - self.wall0) * self.speed:
            return None
        b, self.pending = self.pending, None
        return b


def open_source(spec: str, **kw):
    """'camera' / 'camera:<serial>' / a path to .aedat4 or .npz / 'sim:<path>' (a recording replayed in real time)."""
    if spec == "camera" or spec.startswith("camera:"):
        return DavisCamera(serial=spec.split(":", 1)[1] if ":" in spec else "", **kw)
    if spec.startswith("sim:"):
        inner = open_source(spec[4:], **{k: v for k, v in kw.items() if k == "sw_ba_filter_us"})
        if isinstance(inner, Aedat4File) or isinstance(inner, NpzEvents):
            if isinstance(inner, NpzEvents):
                inner.chunk = 1000
            return SimulatedLive(inner)
    if spec.endswith(".npz"):
        return NpzEvents(spec)
    return Aedat4File(spec, sw_ba_filter_us=kw.get("sw_ba_filter_us", 0))


class Clock:
    """Session clock in ms of sensor time. `now_ms()` is the latest event bin when events flow; when no event arrives
    (static scene + noise filter) it advances with the wall clock from the last event (10 ms behind it, so a step is
    only computed once its events can have arrived), so the tracker keeps stepping (empty windows hold the state)."""

    def __init__(self):
        self.last_ms = None
        self.last_wall = None

    def update(self, latest_ms: int):
        if self.last_ms is None or latest_ms > self.last_ms:
            self.last_ms, self.last_wall = int(latest_ms), time.perf_counter()

    def now_ms(self, live: bool) -> Optional[int]:
        if self.last_ms is None:
            return None
        if not live:
            return self.last_ms
        return max(self.last_ms, int(self.last_ms + (time.perf_counter() - self.last_wall) * 1000.0) - 10)
