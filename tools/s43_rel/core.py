"""Causal sensor-only directed pair memory from the S43 redesign, equations 1--4.

Timestamps in an event packet are explicitly integer ms or integer us. A group
at t is complete only once the watermark reaches t + one native timestamp bin.
Queries use t < cutoff and never recover an already-consumed past prefix.

The persistent state has fixed size. A packet may end with one incomplete time
group, held as cell/polarity counts; arbitrary future packets are not buffered.
Packet ordinals and optional disjoint source ranges enforce upstream uniqueness
without an unbounded UID set. Equal payloads from distinct sources retain their
true multiplicity. All kernels are fixed; this module has no torch dependency.
"""
from __future__ import annotations

import ctypes
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class RelConfig:
    width: int = 240
    height: int = 180
    cell_size: int = 2
    tau_ms: Tuple[float, ...] = (2.0, 8.0, 32.0)
    gamma_ms: Tuple[float, ...] = (16.0, 64.0)
    dtype: str = "float32"
    time_unit: str = "ms"
    backend: str = "auto"
    # Eight immediate cell offsets and their doubled versions, fixed by design.
    offsets: Tuple[Tuple[int, int], ...] = tuple(
        (s * x, s * y) for s in (1, 2)
        for y in (-1, 0, 1) for x in (-1, 0, 1) if x or y
    )

    def __post_init__(self):
        for name in ("width", "height", "cell_size"):
            v = getattr(self, name)
            if isinstance(v, bool) or int(v) != v or v <= 0:
                raise ValueError("%s must be a positive integer" % name)
        if self.width % self.cell_size or self.height % self.cell_size:
            raise ValueError("cell_size must divide sensor width and height")
        for name in ("tau_ms", "gamma_ms"):
            v = tuple(float(x) for x in getattr(self, name))
            if not v or not all(math.isfinite(x) and x > 0 for x in v):
                raise ValueError("%s must contain finite positive fixed poles" % name)
            object.__setattr__(self, name, v)
        dt = np.dtype(self.dtype)
        if dt not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        object.__setattr__(self, "dtype", dt.name)
        if self.time_unit not in ("ms", "us") or self.backend not in ("auto", "cpp", "numpy"):
            raise ValueError("unsupported time_unit or backend")
        if any(any(isinstance(x, bool) or int(x) != x for x in xy) for xy in self.offsets):
            raise ValueError("offset coordinates must be integers")
        off = tuple(tuple(int(x) for x in xy) for xy in self.offsets)
        if not off or any(len(xy) != 2 or xy == (0, 0) for xy in off):
            raise ValueError("offsets must be nonzero directed (dx,dy) pairs")
        if len(set(off)) != len(off) or any((-x, -y) not in off for x, y in off):
            raise ValueError("offsets must be unique and contain every reverse")
        object.__setattr__(self, "offsets", off)

    @property
    def grid_width(self):
        return self.width // self.cell_size

    @property
    def grid_height(self):
        return self.height // self.cell_size

    @property
    def n_cells(self):
        return self.grid_width * self.grid_height


@dataclass(frozen=True)
class RelationTopology:
    src: np.ndarray
    dst: np.ndarray
    reverse: np.ndarray
    incoming: np.ndarray
    centers_px: np.ndarray
    midpoint_px: np.ndarray
    delta_px: np.ndarray
    grid_shape: Tuple[int, int]

    @property
    def n_edges(self):
        return len(self.src)

    @property
    def nbytes(self):
        return sum(x.nbytes for x in (self.src, self.dst, self.reverse, self.incoming,
                                     self.centers_px, self.midpoint_px, self.delta_px))


@lru_cache(maxsize=16)
def _topology(width, height, cell_size, offsets):
    W, H = width // cell_size, height // cell_size
    cell = np.arange(W * H, dtype=np.int32)
    x, y = cell % W, cell // W
    centers = np.stack(((x + .5) * cell_size - .5,
                        (y + .5) * cell_size - .5), -1).astype(np.float32)
    ss, dd = [], []
    for dx, dy in offsets:
        good = (x + dx >= 0) & (x + dx < W) & (y + dy >= 0) & (y + dy < H)
        s = cell[good]
        ss.append(s)
        dd.append((s + dy * W + dx).astype(np.int32))
    src, dst = np.concatenate(ss), np.concatenate(dd)
    # Sealing touches all incoming edges of an active target. Store those edges
    # together so the 24-channel Z updates do not stride across the whole field.
    edge_order = np.lexsort((src, dst))
    src, dst = src[edge_order], dst[edge_order]
    keys = src.astype(np.int64) * (W * H) + dst
    order = np.argsort(keys)
    reverse = order[np.searchsorted(keys[order], dst.astype(np.int64) * (W * H) + src)].astype(np.int32)
    incoming = np.full((W * H, len(offsets)), -1, np.int32)
    starts = np.r_[0, np.flatnonzero(dst[1:] != dst[:-1]) + 1]
    rank = np.arange(len(dst)) - np.repeat(starts, np.diff(np.r_[starts, len(dst)]))
    incoming[dst, rank] = np.arange(len(dst), dtype=np.int32)
    return RelationTopology(*[_readonly(a) for a in (src, dst, reverse, incoming, centers,
                             (centers[src] + centers[dst]) * .5, centers[dst] - centers[src])],
                            grid_shape=(H, W))


@dataclass(frozen=True)
class RelationSnapshot:
    A: Optional[np.ndarray]
    Z: np.ndarray
    edge_ids: np.ndarray
    cutoff_ms: float
    watermark_ms: float
    version: int
    seq_id: Any
    missing: bool
    last_event_ms: Optional[float]
    delta_since_commit_ms: float
    fixed_output_lag_ms: float = 1.0


@dataclass(frozen=True)
class ControlSnapshot:
    """Sensor controls. -inf marks absent first/last times, never a zero event.

    count is the exact prefix count. EMA equals A at fixed tau, last/first are
    prefix timestamps, coarse is a coarser spatial sum of the SAME EMA state.
    These fields define bounded controls, not a claim to exact W50 raw history.
    """
    count: np.ndarray
    first_ms: np.ndarray
    last_ms: np.ndarray
    ema: np.ndarray
    coarse_ema: np.ndarray
    cutoff_ms: float
    version: int


@lru_cache(maxsize=1)
def _cpp_library():
    source = Path(__file__).with_name("core_cpu.cpp")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    target_dir = Path(tempfile.gettempdir()) / ("s43_rel_" + str(os.getuid()))
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / ("core_" + digest + ".so")
    if not target.exists():
        temp = target.with_name(target.name + "." + str(os.getpid()) + ".tmp")
        result = subprocess.run(["g++", "-O3", "-std=c++17", "-shared", "-fPIC",
                                 str(source), "-o", str(temp)], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("relation core C++ build failed: " + result.stderr[-2000:])
        os.replace(str(temp), str(target))
    lib = ctypes.CDLL(str(target))
    ptr = ctypes.c_void_p
    args = [ptr] * 6 + [ctypes.c_int] * 4 + [ptr] * 4 + [ctypes.c_int] + [ptr] * 5
    for name in ("rel_update_f32", "rel_update_f64"):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = args, ctypes.c_int
    return lib


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + " must be an integer")
    try:
        i = int(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(name + " must be a finite integer")
    if i != value or not -(2**63) < i < 2**63:
        raise ValueError(name + " is not an int64 integer")
    return i


def _ms_tick(value):
    value = float(value)
    if not math.isfinite(value) or value < 0 or value * 1000 >= 2**62:
        raise ValueError("sensor time must be finite, nonnegative and within int64 us")
    v = value * 1000
    if abs(v - round(v)) > 1e-5:
        raise ValueError("cutoff/watermark resolution is one microsecond")
    return int(round(v))


class RelationMemory:
    """One sequence/valid segment, with independent watermark and source ledger.

    readout makes an immutable copy only of selected edges (all if omitted).
    edge_ids=[] omits Z cheaply. include_A=False omits full-grid A. Query never
    writes decay into memory. Repeated queries and hand-state rebinding therefore
    cannot modify observations. To query an earlier consumed prefix save a typed
    snapshot or export_state at that prefix; current compressed state cannot do it.
    """
    STATE_FORMAT = 1
    _METADATA_NAMES = ("seq_id", "_pending_t", "_watermark", "_boundary",
                       "_last_sealed", "_last_input", "_ordinal", "_range_end",
                       "_source_mode", "_unit", "_version", "_last_commit",
                       "_commit_count", "_missing")

    def __init__(self, config: Optional[RelConfig] = None, seq_id=None):
        self.config = config or RelConfig()
        c = self.config
        self.topology = _topology(c.width, c.height, c.cell_size, c.offsets)
        self.tau_us = np.asarray(c.tau_ms, np.float64) * 1000
        self.gamma_us = np.asarray(c.gamma_ms, np.float64) * 1000
        self.backend = "numpy"
        self._lib = None
        if c.backend != "numpy":
            try:
                self._lib = _cpp_library()
                self.backend = "cpp"
            except (OSError, RuntimeError):
                if c.backend == "cpp":
                    raise
        self._released = False
        self.reset(seq_id)

    def reset(self, seq_id=None, boundary_ms=0.0):
        boundary = _ms_tick(boundary_ms)
        c, E = self.config, self.topology.n_edges
        self.seq_id = seq_id
        self._A = np.zeros((c.n_cells, 2, len(c.tau_ms)), c.dtype)
        self._Z = np.zeros((E, 2, 2, len(c.tau_ms), len(c.gamma_ms)), c.dtype)
        self._ta = np.full(c.n_cells, boundary, np.int64)
        self._tz = np.full(E, boundary, np.int64)
        self._count = np.zeros((c.n_cells, 2), np.int64)
        self._first = np.full((c.n_cells, 2), -1, np.int64)
        self._last = np.full((c.n_cells, 2), -1, np.int64)
        self._pending = np.zeros((c.n_cells, 2), np.int64)
        self._pending_t = None
        self._watermark = self._boundary = boundary
        self._last_sealed = self._last_input = None
        self._ordinal = -1
        self._range_end = None
        self._source_mode = None
        self._unit = None
        self._version = 0
        self._last_commit = boundary
        self._commit_count = 0
        self._missing = False
        self._released = False
        self._poisoned = False
        self._stats = dict(packets=0, events=0, sealed_groups=0, ingest_seconds=0.0,
                           seal_seconds=0.0, readout_seconds=0.0, readouts=0)
        return self

    @property
    def version(self):
        return self._version

    @property
    def watermark_ms(self):
        return self._watermark / 1000.0

    @property
    def commit_count(self):
        return self._commit_count

    def _check(self, seq_id=None):
        if self._released:
            raise RuntimeError("sequence memory has been released")
        if self._poisoned:
            raise RuntimeError("numeric failure invalidated memory; reset or restore checkpoint")
        if seq_id is not None and seq_id != self.seq_id:
            raise ValueError("mixed sequence: reset or use a separate RelationMemory")

    def _parse(self, events, unit):
        if unit not in ("ms", "us"):
            raise ValueError("explicit time_unit must be ms or us")
        if isinstance(events, dict):
            cols = [np.asarray(events[n]) for n in ("x", "y", "t", "p")]
            if any(a.ndim != 1 or len(a) != len(cols[0]) for a in cols):
                raise ValueError("event columns must be equally sized 1D arrays")
        else:
            a = np.asarray(events)
            if a.dtype.names:
                if any(n not in a.dtype.names for n in ("x", "y", "t", "p")):
                    raise ValueError("structured events require x,y,t,p fields")
                cols = [np.asarray(a[n]) for n in ("x", "y", "t", "p")]
            else:
                if a.size == 0:
                    a = np.empty((0, 4))
                if a.ndim != 2 or a.shape[1] != 4:
                    raise ValueError("events must be Nx4 [x,y,t,p]")
                cols = [a[:, j] for j in range(4)]
        x, y, t, p = cols
        if any(a.ndim != 1 for a in cols) or any(not np.all(np.isfinite(a)) for a in cols):
            raise ValueError("event fields must be finite one-dimensional numeric arrays")
        if any(np.any(a != np.floor(a)) for a in cols):
            raise ValueError("event coordinates, native timestamps and polarities must be integers")
        c = self.config
        if np.any(x < 0) or np.any(x >= c.width) or np.any(y < 0) or np.any(y >= c.height):
            raise ValueError("event lies outside fixed sensor dictionary")
        if np.any(t < 0) or np.any(t >= 2**62 // (1000 if unit == "ms" else 1)):
            raise ValueError("event timestamp overflows safe int64-us range")
        if not np.all(np.isin(p, [-1, 0, 1])):
            raise ValueError("polarity must use 0/1 or -1/+1")
        if np.any(p == -1) and np.any(p == 0):
            raise ValueError("cannot mix 0/1 and -1/+1 polarity encodings in one packet")
        t = t.astype(np.int64) * (1000 if unit == "ms" else 1)
        if len(t) > 1 and np.any(t[1:] < t[:-1]):
            raise ValueError("source packet timestamps must be nondecreasing")
        cp = (y.astype(np.int64) // c.cell_size * c.grid_width
              + x.astype(np.int64) // c.cell_size) * 2 + (p > 0).astype(np.int64)
        return t, cp

    def ingest(self, events, watermark_ms=None, packet_ordinal=None, *, seq_id=None,
               time_unit=None, source_range=None, valid=True):
        """Consume a unique ordered source packet; watermark proves completeness.

        ``source_range=(start,end)`` is an optional half-open native event ordinal
        range, length equal to the packet. Ranges may have gaps but never overlap.
        ``valid=False`` is padding and reads neither payload nor sequence ID.
        With no watermark, only one incomplete timestamp may be staged and the
        stream is marked missing; a later explicit watermark clears this flag.
        """
        if not valid:
            return self
        started = time.perf_counter()
        self._check(seq_id)
        ordinal = _integer(packet_ordinal, "packet_ordinal")
        if ordinal <= self._ordinal:
            raise ValueError("duplicate or reordered source packet ordinal")
        unit = time_unit or self.config.time_unit
        if self._unit is not None and unit != self._unit:
            raise ValueError("timestamp units cannot change within one sequence")
        t, cp = self._parse(events, unit)
        quantum = 1000 if unit == "ms" else 1
        watermark = self._watermark if watermark_ms is None else _ms_tick(watermark_ms)
        if watermark < self._watermark:
            raise ValueError("complete watermark cannot move backwards")
        if len(t) and (t[0] + quantum <= self._watermark or
                       (self._last_input is not None and t[0] < self._last_input)):
            raise ValueError("late event violates sealed watermark or source ordering")
        has_range = source_range is not None
        if self._source_mode is not None and has_range != self._source_mode:
            raise ValueError("source-range contract cannot change within one sequence")
        range_end = self._range_end
        if has_range:
            if len(source_range) != 2:
                raise ValueError("source_range requires (start,end)")
            a, b = [_integer(v, "source range") for v in source_range]
            if a < 0 or b < a or b - a != len(t) or (range_end is not None and a < range_end):
                raise ValueError("overlapping, reversed or wrong-length source range")
            range_end = b
        if len(t) and watermark_ms is not None and t[-1] > watermark:
            raise ValueError("future events beyond watermark cannot be buffered")
        # Only a single terminal timestamp may remain unsealed. Parse/contract
        # failures above have changed no state; aggregation also precedes mutation.
        group_t, begin = np.unique(t, return_index=True)
        end = np.r_[begin[1:], len(t)]
        grouped = []
        for stamp, a, b in zip(group_t, begin, end):
            counts = np.bincount(cp[a:b], minlength=self.config.n_cells * 2).reshape(-1, 2)
            grouped.append((int(stamp), counts))
        if self._pending_t is not None:
            if grouped and grouped[0][0] == self._pending_t:
                if np.any(grouped[0][1] > np.iinfo(np.int64).max - self._pending):
                    raise OverflowError("timestamp multiplicity overflow")
                grouped[0] = (self._pending_t, grouped[0][1] + self._pending)
            else:
                grouped.insert(0, (self._pending_t, self._pending.copy()))
        incomplete = [(tt, n) for tt, n in grouped if tt + quantum > watermark]
        if len(incomplete) > 1:
            raise ValueError("bounded staging permits one incomplete timestamp only")
        sealed = [(tt, n) for tt, n in grouped if tt + quantum <= watermark]
        if sealed:
            self._seal(sealed)
            self._last_sealed = sealed[-1][0]
        self._pending.fill(0)
        self._pending_t = None
        if incomplete:
            self._pending_t, n = incomplete[0]
            self._pending[:] = n
        self._watermark = watermark
        self._last_input = int(t[-1]) if len(t) else self._last_input
        self._ordinal, self._range_end, self._source_mode = ordinal, range_end, has_range
        self._unit = unit
        self._missing = watermark_ms is None
        self._version += 1
        self._stats["packets"] += 1
        self._stats["events"] += len(t)
        self._stats["ingest_seconds"] += time.perf_counter() - started
        return self

    def _seal(self, groups):
        start = time.perf_counter()
        times, cells, counts, offsets = [], [], [], [0]
        for t, n in groups:
            active = np.flatnonzero(n.any(axis=1)).astype(np.int32)
            times.append(t)
            cells.append(active)
            counts.append(n[active])
            offsets.append(offsets[-1] + len(active))
        tt, oo = np.asarray(times, np.int64), np.asarray(offsets, np.int64)
        cc, nn = np.concatenate(cells), np.concatenate(counts)
        if self.backend == "cpp":
            fn = getattr(self._lib, "rel_update_f32" if self.config.dtype == "float32" else "rel_update_f64")
            ptr = lambda a: a.ctypes.data_as(ctypes.c_void_p)
            c = self.config
            result = fn(*[ptr(a) for a in (self._A, self._Z, self._ta, self._tz,
                                           self.topology.src, self.topology.incoming)],
                        c.n_cells, len(c.offsets), len(c.tau_ms), len(c.gamma_ms),
                        *[ptr(a) for a in (self.tau_us, self.gamma_us, tt, oo)], len(tt),
                        *[ptr(a) for a in (cc, nn, self._count, self._first, self._last)])
            if result:
                self._poisoned = True
                raise OverflowError("nonfinite relation state or int64 count overflow")
        else:
            self._seal_numpy(tt, oo, cc, nn)
        self._stats["sealed_groups"] += len(groups)
        self._stats["seal_seconds"] += time.perf_counter() - start

    def _seal_numpy(self, tt, offsets, cells, counts):
        for g, t in enumerate(tt):
            b = cells[offsets[g]:offsets[g + 1]]
            n = counts[offsets[g]:offsets[g + 1]]
            edges = self.topology.incoming[b].ravel()
            edges = edges[edges >= 0]
            src, dst = self.topology.src[edges], self.topology.dst[edges]
            # Use unmodified old A for the WHOLE timestamp group.
            old = self._A[src] * np.exp(-(t - self._ta[src])[:, None, None] / self.tau_us)
            target_n = np.zeros((self.config.n_cells, 2), np.int64)
            target_n[b] = n
            z = (self._Z[edges] * np.exp(-(t - self._tz[edges])[:, None, None, None, None]
                                       / self.gamma_us)
                 + old[:, :, None, :, None] * target_n[dst, None, :, None, None])
            a = self._A[b] * np.exp(-(t - self._ta[b])[:, None, None] / self.tau_us) + n[:, :, None]
            if not np.all(np.isfinite(z)) or not np.all(np.isfinite(a)) or np.any(z > np.finfo(self._Z.dtype).max):
                self._poisoned = True
                raise OverflowError("nonfinite relation state")
            if np.any(n > np.iinfo(np.int64).max - self._count[b]):
                self._poisoned = True
                raise OverflowError("int64 count overflow")
            self._Z[edges], self._tz[edges] = z, t
            self._A[b], self._ta[b] = a, t
            self._count[b] += n
            self._first[b] = np.where((self._first[b] < 0) & (n > 0), t, self._first[b])
            self._last[b] = np.where(n > 0, t, self._last[b])

    def _cutoff(self, cutoff_ms):
        self._check()
        t = _ms_tick(cutoff_ms)
        if t < self._boundary or t > self._watermark:
            raise ValueError("cutoff requires a complete known prefix within the current watermark")
        quantum = 1000 if (self._unit or self.config.time_unit) == "ms" else 1
        if (t + quantum - 1) // quantum * quantum > self._watermark:
            raise ValueError("cutoff includes a native timestamp bin not yet sealed by watermark")
        if self._last_sealed is not None and t <= self._last_sealed:
            raise ValueError("cannot read a past half-open prefix after consuming a future group")
        return t

    def readout(self, cutoff_ms, edge_ids=None, include_A=True, *, seq_id=None):
        started = time.perf_counter()
        self._check(seq_id)
        t = self._cutoff(cutoff_ms)
        if edge_ids is None:
            ids = np.arange(self.topology.n_edges, dtype=np.int32)
        else:
            raw = np.asarray(edge_ids)
            if raw.ndim != 1 or np.any(~np.isfinite(raw)) or np.any(raw != np.floor(raw)):
                raise ValueError("edge_ids must be a one-dimensional integer array")
            if np.any(raw < 0) or np.any(raw >= self.topology.n_edges):
                raise ValueError("edge index out of range")
            ids = raw.astype(np.int32, copy=True)
            if len(np.unique(ids)) != len(ids):
                raise ValueError("query edge_ids must be unique")
        A = None
        if include_A:
            A = (self._A * np.exp(-(t - self._ta)[:, None, None] / self.tau_us)).astype(self.config.dtype)
            _readonly(A)
        Z = (self._Z[ids] * np.exp(-(t - self._tz[ids])[:, None, None, None, None]
                                  / self.gamma_us)).astype(self.config.dtype)
        snap = RelationSnapshot(A, _readonly(Z), _readonly(ids), t / 1000., self.watermark_ms,
                                self._version, self.seq_id, self._missing,
                                None if self._last_sealed is None else self._last_sealed / 1000.,
                                (t - self._last_commit) / 1000.)
        self._stats["readout_seconds"] += time.perf_counter() - started
        self._stats["readouts"] += 1
        return snap

    def controls(self, cutoff_ms, coarse_factor=2):
        t = self._cutoff(cutoff_ms)
        factor = _integer(coarse_factor, "coarse_factor")
        H, W = self.topology.grid_shape
        if factor <= 0 or H % factor or W % factor:
            raise ValueError("coarse factor must divide grid dimensions")
        ema = (self._A * np.exp(-(t - self._ta)[:, None, None] / self.tau_us)).astype(self.config.dtype)
        coarse = ema.reshape(H // factor, factor, W // factor, factor, 2, -1).sum(axis=(1, 3))
        first = np.where(self._first >= 0, self._first / 1000., -np.inf)
        last = np.where(self._last >= 0, self._last / 1000., -np.inf)
        return ControlSnapshot(*[_readonly(a) for a in (self._count.copy(), first, last, ema, coarse)],
                               cutoff_ms=t / 1000., version=self._version)

    def commit(self, snapshot=None, *, cutoff_ms=None, version=None, seq_id=None):
        """Validate the sensor side of an explicit external hand-state commit.

        Prediction/context lives with the caller. Committing never changes A/Z.
        Ingest after snapshot creation invalidates that snapshot for commit.
        """
        self._check(seq_id)
        if snapshot is not None:
            if snapshot.seq_id != self.seq_id:
                raise ValueError("snapshot belongs to a different sequence")
            cutoff_ms, version = snapshot.cutoff_ms, snapshot.version
        t = self._cutoff(cutoff_ms)
        if version != self._version:
            raise ValueError("stale observation snapshot; recompute before commit")
        if t <= self._last_commit:
            raise ValueError("one commit per strictly increasing sensor cutoff")
        self._last_commit = t
        self._commit_count += 1

    def stats(self):
        self._check()
        result = dict(self._stats)
        result.update(backend=self.backend, watermark_ms=self.watermark_ms,
                      pending_events=int(self._pending.sum()), commits=self._commit_count,
                      resident_bytes=sum(getattr(self, name).nbytes for name in self._array_names()),
                      shared_topology_bytes=self.topology.nbytes,
                      n_cells=self.config.n_cells, n_edges=self.topology.n_edges,
                      source_ledger_entries=1)
        return result

    @staticmethod
    def _array_names():
        return ("_A", "_Z", "_ta", "_tz", "_count", "_first", "_last", "_pending")

    def export_state(self):
        """Bounded checkpoint, detached by construction; no raw event history."""
        self._check()
        metadata = {k: getattr(self, k) for k in self._METADATA_NAMES}
        return dict(format=self.STATE_FORMAT, config=asdict(self.config), metadata=metadata,
                    arrays={name: getattr(self, name).copy() for name in self._array_names()},
                    stats=dict(self._stats))

    state_dict = export_state

    def load_state_dict(self, state):
        if state.get("format") != self.STATE_FORMAT or RelConfig(**state["config"]) != self.config:
            raise ValueError("checkpoint kernel/sensor/config identity mismatch")
        arrays = {}
        for name in self._array_names():
            a, own = np.asarray(state["arrays"][name]), getattr(self, name)
            if a.shape != own.shape or a.dtype != own.dtype or not np.all(np.isfinite(a)):
                raise ValueError("invalid checkpoint array " + name)
            arrays[name] = a.copy()
        if set(state["metadata"]) != set(self._METADATA_NAMES):
            raise ValueError("checkpoint metadata schema mismatch")
        m = state["metadata"]
        if not (0 <= m["_boundary"] <= m["_last_commit"] <= m["_watermark"]):
            raise ValueError("checkpoint sensor clock or commit clock is inconsistent")
        if m["_last_sealed"] is not None and not (m["_boundary"] <= m["_last_sealed"] < m["_watermark"]):
            raise ValueError("checkpoint consumed prefix exceeds complete watermark")
        if m["_unit"] not in (None, "ms", "us") or m["_ordinal"] < -1 or m["_version"] < 0:
            raise ValueError("checkpoint source metadata is inconsistent")
        if any(np.any(arrays[k] < 0) for k in ("_A", "_Z", "_count", "_pending")):
            raise ValueError("checkpoint has negative counts or exponential statistics")
        if m["_pending_t"] is None and np.any(arrays["_pending"]):
            raise ValueError("checkpoint has staged counts without a timestamp")
        for k, v in state["metadata"].items():
            setattr(self, k, v)
        for k, v in arrays.items():
            setattr(self, k, v)
        self._stats = dict(state["stats"])
        self._released = self._poisoned = False
        return self

    @classmethod
    def from_state(cls, state):
        return cls(RelConfig(**state["config"])).load_state_dict(state)

    def release(self, seq_id=None):
        self._check(seq_id)
        for name in self._array_names():
            setattr(self, name, None)
        self._released = True


def all_pair_reference(events, config: RelConfig, cutoff_ms, time_unit=None):
    """Independent O(N^2) equation evaluator, only for tiny arithmetic audits.

    This deliberately does not invoke ingest/grouping/recurrence. It returns the
    full sensor A/Z at a strict half-open cutoff, including real multiplicity.
    """
    helper = RelationMemory(RelConfig(**dict(asdict(config), backend="numpy")))
    t, cp = helper._parse(events, time_unit or config.time_unit)
    T = _ms_tick(cutoff_ms)
    good = t < T
    t, cp = t[good], cp[good]
    cells, p = cp // 2, cp % 2
    R, L = len(config.tau_ms), len(config.gamma_ms)
    A = np.zeros((config.n_cells, 2, R), np.float64)
    Z = np.zeros((helper.topology.n_edges, 2, 2, R, L), np.float64)
    edge_map = {(int(a), int(b)): e for e, (a, b) in enumerate(zip(helper.topology.src, helper.topology.dst))}
    for i in range(len(t)):
        A[cells[i], p[i]] += np.exp(-(T - t[i]) / helper.tau_us)
        for j in range(len(t)):
            e = edge_map.get((int(cells[i]), int(cells[j])))
            if t[i] < t[j] and e is not None:
                Z[e, p[i], p[j]] += (np.exp(-(t[j] - t[i]) / helper.tau_us)[:, None]
                                          * np.exp(-(T - t[j]) / helper.gamma_us)[None, :])
    return A, Z
