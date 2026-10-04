#!/usr/bin/env python3
"""Deployment form of the dense render-and-compare tracker (DT2, package D): zero training, every default unchanged.

The pieces, each opt-in and each bit-identical to the code it stands in for (tests/test_dt_render.py,
tests/test_dt_deploy.py pin that against the live repo):

  enable_fast_render(model, cache=True)
        Swaps the model INSTANCE's `_render_prev` for the host-sync-free `render_fast.RenderFast` (no boolean indexing, per
        sequence MANO shape terms cached while the betas / K are the sequence context) and turns on
        `ManoLayer.fast_chain` (Python-int parents: the 15 `.item()` syncs of the kinematic chain go). model/model.py is
        not touched; `disable_fast_render` undoes everything. With it, `model.forward` is host-sync free, which is what
        lets a CUDA graph record it (`render_fast.GraphedForward` records `model.forward` itself, so every arm is covered).

  GpuLNES
        The event image (`semkine.events.splat_event_image`, the planes of DATA.EVENT_CHANNELS) built on the GPU from the
        packet's raw events (x, y, p as uint8, one integer millisecond per event, time ordered), bit for bit the host
        image. `last` / `first` are integer scatter-reductions (amax / amin of the millisecond index per (pixel,
        polarity), then the same float32 lookup table `tab[ms] = f32(ms) / f32(window)` the host splat uses); `count` is an
        integer scatter-add followed by the host's own table of `log1p(n) / log1p(COUNT_REF)`. Fixed-capacity pinned
        staging buffers, one non-blocking H2D per packet; a packet over the capacity falls back to the host splat.

  DeployTracker
        One 50 ms step end to end with the state resident on the GPU: raw events H2D -> GPU LNES -> forward -> the
        `semkine.anchored.AdaptiveFilter` device path (the very module code, run on the model's output) -> state update; the
        only device-to-host transfer of a step is the 51 numbers read at its end. The CUDA graph can record the forward
        alone (`scope="model"`), the forward and the filter (`"filter"`) or LNES, forward and filter (`"all"`); the same
        stage functions also run eagerly (`graph=False`), and the two are bitwise equal.

Why a graph at all: at batch 1 the forward is bound by the host issuing ~230 small kernels; the graph replays the same
kernels in the same order, so the step costs the GPU's own time (outputs/dt2/reports/D.md has the measured table).
"""
from __future__ import annotations

import bisect
import math
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "model"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from render_fast import RenderFast, _cache_hits, capture_checked  # noqa: E402
from semkine import events as EV                                 # noqa: E402
from semkine.anchored import AdaptiveFilter, anchor_blend        # noqa: E402

STEP = 50
#: the filter the DT2 package B recommends (two-fold selected on zgz, three-seed RA 9.761 vs 10.211 raw)
RECOMMENDED_SPEC = {"a_root": [[300, 0.3], [3000, 0.5], [30000, 0.8]], "a_rest": 0.8, "a_trans": 0.5}


# ------------------------------------------------------------------------------------------------ fast render
class _ContextHook:
    """Stands in for `model.set_hand_context`: the original, then the render cache learns the new sequence."""

    def __init__(self, model, render: RenderFast):
        self.model, self.render = model, render

    def __call__(self, betas, camera_K):
        type(self.model).set_hand_context(self.model, betas, camera_K)
        self.render.set_context(self.model._ctx_betas, self.model._ctx_K)


def _dense_models(obj):
    """The dense PREV_RENDER models inside `obj` (itself, or a `trk_model` / `abs_model` of a filter / anchor wrapper)."""
    if hasattr(obj, "prev_render") and hasattr(obj, "_render_prev"):
        return [obj]
    out = []
    for name in ("trk_model", "abs_model"):
        inner = getattr(obj, name, None)
        if inner is not None:
            out += _dense_models(inner)
    return out


def is_fast_render(model) -> bool:
    return isinstance(model.__dict__.get("_render_prev"), RenderFast)


def _enable_one(m, cache: bool) -> None:
    if not m.prev_render or getattr(m, "encoder_name", "") or getattr(m, "active_head", False):
        raise NotImplementedError("enable_fast_render covers the dense PREV_RENDER arm only (no ENCODER, no ACTIVE_HEAD)")
    if is_fast_render(m):
        disable_fast_render(m)
    mano = m.mano
    saved = {"fast_chain": bool(getattr(mano, "fast_chain", False)), "had_py": hasattr(mano, "_parents_py")}
    if not saved["had_py"]:                              # a layer from before ManoLayer.fast_chain: one sync, here
        mano._parents_py = [int(p) for p in mano.parents.tolist()]
    mano.fast_chain = True
    rf = RenderFast(m, raster=True, fk=True, cache=bool(cache))
    m._render_prev = rf
    m.set_hand_context = _ContextHook(m, rf)
    m._fast_render_saved = saved
    rf.set_context(m._ctx_betas, m._ctx_K)               # a context set before enabling is picked up


def enable_fast_render(model, cache: bool = True):
    """Opt-in, default off: the bit-identical, host-synchronisation-free render of the dense PREV_RENDER arm, on this model
    INSTANCE (model/model.py is untouched):

      * `model._render_prev` -> `render_fast.RenderFast` (raster without boolean indexing, MANO FK without hidden syncs);
      * `model.mano.fast_chain = True` (the kinematic chain of every `ManoLayer.forward` call of this model, also the
        root head's FK, runs on Python-int parents);
      * `cache`: while the betas / K are the `set_hand_context` ones (batch 1), reuse the per-sequence MANO terms (the
        cache is keyed on the context tensors' storage and version, so an edited / foreign tensor takes the uncached
        path). `model.set_hand_context` is wrapped to keep it current.

    `model` may be a `FilteredTracker` / `AdaptiveFilter` / `AnchoredTracker`: the models inside are patched. Outputs
    equal the unpatched model's bit for bit (tests/test_dt_render.py, tests/test_dt_deploy.py). Not to be combined with
    `copy.deepcopy` of a patched model that you then use independently without `disable_fast_render` first. Returns
    `model`."""
    targets = _dense_models(model)
    if not targets:
        raise NotImplementedError("enable_fast_render: no dense PREV_RENDER model found in the given object")
    for m in targets:
        _enable_one(m, cache)
    return model


def disable_fast_render(model):
    for m in _dense_models(model):
        saved = m.__dict__.pop("_fast_render_saved", None)
        m.__dict__.pop("_render_prev", None)
        m.__dict__.pop("set_hand_context", None)
        if saved is not None:
            m.mano.fast_chain = saved["fast_chain"]
            if not saved["had_py"]:
                del m.mano._parents_py
    return model


# ------------------------------------------------------------------------------------------------ GPU event image
def host_lnes_from(ev: np.ndarray, bounds: np.ndarray, window: int, height: int, width: int,
                   channels: Sequence[str] = ("last",)) -> np.ndarray:
    """The host image of a packet given as `GpuLNES` takes it: `ev` (N, 3) uint8 [x, y, p], `bounds` (window,) the
    cumulative event count at the end of each millisecond. Exactly `semkine.eval_track.build_lnes` (same splat call)."""
    n = int(bounds[-1]) if len(bounds) else 0
    if n == 0:
        return np.zeros((height, width, 2 * len(channels)), np.float32)
    counts = np.diff(np.asarray(bounds, dtype=np.int64), prepend=0)
    ms_rel = np.repeat(np.arange(window, dtype=np.float32), counts)
    return EV.splat_event_image(ev[:, 0], ev[:, 1], np.clip(ev[:, 2], 0, 1), ms_rel, window, height, width,
                                tuple(channels))


def packet_bounds(offsets: np.ndarray, end: int, window: int = STEP) -> np.ndarray:
    """`(window,)` int32: the cumulative event count at the end of each millisecond of the window `[end-window+1, end]`
    (`offsets` is the per-millisecond event index of the sequence, as `semkine.eval_track.load_sequence` returns it)."""
    start = end - window + 1
    return (np.asarray(offsets[start + 1:end + 2]) - int(offsets[start])).astype(np.int32)


def bounds_from_ms(ms_rel: np.ndarray, window: int = STEP) -> np.ndarray:
    """`(window,)` int32 `bounds` of a live packet from its events' integer millisecond within the window (`ms_rel` (N,)
    non-decreasing ints in `[0, window)`, the events' time order): the cumulative event count at the end of each ms."""
    return np.searchsorted(np.asarray(ms_rel), np.arange(1, window + 1), side="left").astype(np.int32)


class GpuLNES:
    """`semkine.events.splat_event_image` on the GPU, bit for bit, from raw events (see the module docstring).

    Staging: one pinned uint8 buffer `[bounds: window x int32][events: capacity x 3 uint8]` and its device twin; `stage`
    copies a packet into the host side and issues ONE non-blocking H2D of the bytes in use. The device side only reads
    that buffer, so `build` can be recorded in a CUDA graph (static shapes, no host synchronisation, no data dependent
    branch: events past the packet's end and events outside the frame write into one extra "dump" slot that is dropped).
    The host must not `stage` again before the previous packet's copy has been consumed (a step ends with a host sync).

    Each event's millisecond is recovered on the device from `bounds` (a `searchsorted` of the event index), the same
    number `np.repeat(arange(window), counts)` gives the host. `last` is the largest millisecond per slot (events are
    time ordered, so it is the host's "later write wins"); `first` the smallest (the host's reversed splat); an empty slot is
    0.0 in both."""

    def __init__(self, device, window: int = STEP, height: int = 180, width: int = 240,
                 channels: Sequence[str] = ("last",), capacity: int = 262144):
        ch = tuple(channels)
        bad = [c for c in ch if c not in EV.EVENT_CHANNELS]
        if bad or not ch or ch[0] != "last" or len(set(ch)) != len(ch):
            raise ValueError(f"channels must be a duplicate-free subset of {list(EV.EVENT_CHANNELS)} beginning with 'last', got {list(ch)}")
        self.device = torch.device(device)
        self.window, self.height, self.width, self.channels = int(window), int(height), int(width), ch
        self.capacity = int(capacity)
        self.n_slots = self.height * self.width * 2
        # the table the host splat computes with: tval = f32(ms) / python float(window), in float32; slot `window` is 0.0
        # (the "no event" value of the amin scatter)
        tab = np.arange(self.window, dtype=np.float32).astype(np.float32) / float(self.window)
        self._tab = torch.from_numpy(np.concatenate([tab, np.zeros(1, np.float32)])).to(self.device)
        # `count`: the host builds `(log1p(n) / log1p(COUNT_REF)).clip(max=1)` in float32 from the integer count n; every
        # n >= COUNT_REF + 1 is clipped to 1.0, so a table of n = 0 .. COUNT_REF + 1 with a clamped index is exact
        self._count_top = int(math.ceil(EV.COUNT_REF)) + 1
        n = np.arange(self._count_top + 1).astype(np.float32)
        self._ctab = torch.from_numpy((np.log1p(n) / np.log1p(EV.COUNT_REF)).clip(max=1.0).astype(np.float32)).to(self.device)
        self._ar = torch.arange(self.capacity, dtype=torch.int32, device=self.device)
        hb = 4 * self.window
        self._hdr_bytes = hb
        self._pin = torch.empty(hb + 3 * self.capacity, dtype=torch.uint8).pin_memory()
        self._pin_np = self._pin.numpy()
        self._hdr_np = self._pin_np[:hb].view(np.int32)
        self._ev_np = self._pin_np[hb:].reshape(self.capacity, 3)
        self.dev = torch.zeros(hb + 3 * self.capacity, dtype=torch.uint8, device=self.device)
        #: `(window,)` int32 cumulative counts and `(capacity, 3)` uint8 events, the device views `build` reads
        self.hdr_dev = self.dev[:hb].view(torch.int32)
        self.ev_dev = self.dev[hb:].view(self.capacity, 3)

    @property
    def n_planes(self) -> int:
        return 2 * len(self.channels)

    def fits(self, n_events: int) -> bool:
        return int(n_events) <= self.capacity

    def stage(self, ev: Optional[np.ndarray], bounds: np.ndarray) -> None:
        """Host -> device. `ev` (N, 3) uint8 [x, y, p] (None: header only, the events are not used), `bounds` (window,)
        the cumulative per-millisecond counts, `bounds[-1] == N`. Raises if N exceeds the capacity."""
        if len(bounds) != self.window:
            raise ValueError(f"bounds must have {self.window} entries, got {len(bounds)}")
        self._hdr_np[:] = bounds
        nbytes = self._hdr_bytes
        if ev is not None:
            n = int(bounds[-1])
            if len(ev) != n:
                raise ValueError(f"bounds[-1] = {n} but {len(ev)} events were given")
            if n > self.capacity:
                raise ValueError(f"{n} events exceed the staging capacity {self.capacity}; use the host splat")
            self._ev_np[:n] = ev
            nbytes += 3 * n
        self.dev[:nbytes].copy_(self._pin[:nbytes], non_blocking=True)

    def build(self) -> torch.Tensor:
        """`(1, H, W, 2 * len(channels))` float32 from the staged packet; no host synchronisation (graph-capturable)."""
        W, Hh, Wd, ns = self.window, self.height, self.width, self.n_slots
        dev = self.device
        n = self.hdr_dev[W - 1]                                              # 0-d device int32: the packet's event count
        ar = self._ar
        ms = torch.searchsorted(self.hdr_dev, ar, right=True)                # (cap,) int64; == W past the packet's end
        ev = self.ev_dev
        xi, yi, pi = ev[:, 0].long(), ev[:, 1].long(), ev[:, 2].long().clamp_(0, 1)
        ok = (ar < n) & (xi < Wd) & (yi < Hh)
        slot = torch.where(ok, (yi * Wd + xi) * 2 + pi, ns)                  # invalid events -> the dump slot `ns`
        planes = []
        for name in self.channels:
            if name == "last":
                red = torch.zeros(ns + 1, dtype=torch.int64, device=dev).scatter_reduce_(0, slot, ms, reduce="amax", include_self=True)
                planes.append(self._tab[red[:ns]])
            elif name == "first":
                red = torch.full((ns + 1,), W, dtype=torch.int64, device=dev).scatter_reduce_(0, slot, ms, reduce="amin", include_self=True)
                planes.append(self._tab[red[:ns]])
            else:                                                            # "count"
                cnt = torch.zeros(ns + 1, dtype=torch.int64, device=dev).scatter_add_(0, slot, torch.ones_like(slot))
                planes.append(self._ctab[cnt[:ns].clamp_(max=self._count_top)])
        return torch.cat([p.view(Hh, Wd, 2) for p in planes], dim=-1).unsqueeze(0)

    def lnes(self, ev: np.ndarray, bounds: np.ndarray) -> torch.Tensor:
        self.stage(ev, bounds)
        return self.build()


# ------------------------------------------------------------------------------------------------ host filter
class HostGains:
    """The gains of an `AdaptiveFilter` spec evaluated on the host (python floats): piecewise linear in log10(n), constant
    beyond the first and the last node, as `semkine.anchored.GainSchedule` (float64 arithmetic). Used by the legacy
    host-side filter forms (`FilteredTracker`); the device filter needs none of this."""

    def __init__(self, spec: Dict):
        canon = AdaptiveFilter.canonical_spec(spec)
        if canon["beta_root"] > 0 or canon["beta_trans"] > 0:
            raise ValueError("the host filter is the constant-gain-per-packet, no-velocity form")
        self.canon = canon
        self._sched = {k: self._schedule(canon[k]) for k in ("a_root", "a_rest", "a_trans")}

    @staticmethod
    def _schedule(g):
        if isinstance(g, (int, float)):
            return float(g)
        nodes = sorted((float(n), float(v)) for n, v in g)
        return [math.log10(n) for n, _ in nodes], [v for _, v in nodes]

    @staticmethod
    def _eval(s, n: float) -> float:
        if isinstance(s, float):
            return s
        lx, ly = s
        v = min(max(math.log10(max(float(n), 1.0)), lx[0]), lx[-1])
        i = min(max(bisect.bisect_right(lx, v), 1), len(lx) - 1)
        return ly[i - 1] + (v - lx[i - 1]) / (lx[i] - lx[i - 1]) * (ly[i] - ly[i - 1])

    def __call__(self, n_events: float) -> Tuple[float, float, float]:
        return tuple(self._eval(self._sched[k], n_events) for k in ("a_root", "a_rest", "a_trans"))


# ------------------------------------------------------------------------------------------------ the tracker
class _ZStub(nn.Module):
    """The `trk_model` of the device `AdaptiveFilter`: it returns the tracker output handed to it (`z`), so the filter
    module's own code runs on the output of the graph / forward stage."""

    encoder_name = ""

    def __init__(self):
        super().__init__()
        self.z = None

    def forward(self, x, prevpos, betas=None, camera_K=None):
        return self.z


class DeployTracker:
    """The deployment loop of one tracker: raw events in, the filtered 51-D state out, the state resident on the GPU.

        tr = DeployTracker(model, RECOMMENDED_SPEC)            # model: an eval()'d dense PREV_RENDER MNISTModel on CUDA
        tr.begin_sequence(betas, camera_K)                     # per sequence
        tr.begin_segment(prev0)                                # per tracked segment (the filter / state restart)
        state = tr.step_sequence(events, offsets, end)         # per 50 ms packet (or `step_events(ev, bounds)`)

    Per step: pinned events -> one H2D -> GPU LNES -> forward -> filter -> `prev` updated in place -> 51 floats read back
    (the step's only device-to-host copy and its only host synchronisation).

    lnes   "gpu" (default; `GpuLNES`, bit-identical to the host image) | "host" (`semkine.events.splat_event_image`)
    filt   "gpu" (default when a spec is given; `semkine.anchored.AdaptiveFilter`'s device path -- its own code, float64) |
           "host" (the pre-DT2 form: `anchor_blend` in numpy with the gains evaluated on the host; the tracker's output
           goes down, the new state back up; an empty packet is `n_events == 0`) | "none" (the bare tracker: state = output)
    graph  record the step as a CUDA graph (default; needs CUDA); `scope` is what the graph holds:
           "model"   the forward (LNES and filter run as eager kernels around it)
           "filter"  the forward and the filter
           "all"     GPU LNES, forward and filter -- one replay per packet (default)
           A packet that does not fit the staging buffers (`capacity` events) goes through the host splat and the
           "filter" graph; `n_fallback` counts them. `graph=False` runs the same stage functions eagerly: the two agree bit
           for bit (tests/test_dt_deploy.py). The model's betas / K are graph inputs, a new sequence needs no re-capture.
    fast_render  call `enable_fast_render(model)` (the graph cannot record a model that synchronises with the host)

    Bit-exactness: LNES, forward, graph replay and the device filter are each bitwise equal to their eager / host
    counterparts; the device filter is `AdaptiveFilter.forward` itself, so it equals `evalx.run_sequence`'s filtered
    closed loop (host LNES, eager forward, the module) bit for bit on the same GPU.
    """

    def __init__(self, model, spec: Optional[Dict] = None, *, window: int = STEP, channels: Sequence[str] = ("last",),
                 height: int = 180, width: int = 240, capacity: int = 262144, lnes: str = "gpu",
                 filt: Optional[str] = None, graph: bool = True, scope: str = "all", device=None,
                 fast_render: bool = True):
        if lnes not in ("gpu", "host"):
            raise ValueError(f"lnes must be 'gpu' or 'host', got {lnes!r}")
        filt = ("gpu" if spec is not None else "none") if filt is None else filt
        if filt not in ("gpu", "host", "none"):
            raise ValueError(f"filt must be 'gpu', 'host' or 'none', got {filt!r}")
        if filt == "none" and spec is not None:
            raise ValueError("filt='none' takes no spec")
        if filt in ("gpu", "host") and spec is None:
            raise ValueError(f"filt={filt!r} needs a filter spec")
        if scope not in ("model", "filter", "all"):
            raise ValueError(f"scope must be 'model', 'filter' or 'all', got {scope!r}")
        if model.training:
            raise ValueError("DeployTracker needs the model in eval() mode")
        self.model = model
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        if graph and self.device.type != "cuda":
            raise ValueError("graph=True needs the model on a CUDA device")
        if graph and filt == "host" and scope != "model":
            raise ValueError("a host filter cannot be inside the graph: scope must be 'model'")
        if graph and scope == "all" and lnes != "gpu":
            raise ValueError("scope='all' records the GPU LNES: lnes must be 'gpu'")
        self.lnes_mode, self.filt, self.use_graph, self.scope = lnes, filt, bool(graph), scope
        if fast_render and not is_fast_render(model):
            enable_fast_render(model)
        self.window, self.height, self.width = int(window), int(height), int(width)
        self.channels = tuple(channels)
        self.glnes = GpuLNES(self.device, window, height, width, self.channels, capacity)
        dev = self.device
        self.x_buf = torch.zeros(1, height, width, self.glnes.n_planes, dtype=torch.float32, device=dev)
        self.prev_buf = torch.zeros(1, 51, dtype=torch.float32, device=dev)
        self.betas_buf = torch.zeros(1, 10, dtype=torch.float32, device=dev)
        self.K_buf = torch.eye(3, dtype=torch.float32, device=dev).unsqueeze(0).clone()
        self._have_context = False
        self._out_pin = torch.zeros(51, dtype=torch.float32).pin_memory()
        self._raw_pin = torch.zeros(51, dtype=torch.float32).pin_memory()
        self._prev_pin = torch.zeros(1, 51, dtype=torch.float32).pin_memory()
        self._prev_host = np.zeros(51, np.float32)
        self.spec = None if spec is None else AdaptiveFilter.canonical_spec(spec)
        self._flt = self._hgains = None
        if filt == "gpu":
            if self.spec["host"]:
                raise ValueError("filt='gpu' is the device path: the spec must not set host=True")
            self._flt = AdaptiveFilter.from_spec(_ZStub(), self.spec).to(dev).eval()
            self._flt.reset_state(self.prev_buf)                     # the persistent (graph-visible) velocity buffers
        elif filt == "host":
            self._hgains = HostGains(self.spec)
        self._graphs: Dict[str, tuple] = {}
        self.n_fallback = 0
        self.n_steps = 0

    # ---- sequence / segment
    def begin_sequence(self, betas, camera_K) -> None:
        """Per sequence: the betas (10,) / (1, 10) and K (3, 3) / (1, 3, 3) (tensors or arrays) the render uses."""
        b = torch.as_tensor(betas, dtype=torch.float32).to(self.device).reshape(1, -1)
        k = torch.as_tensor(camera_K, dtype=torch.float32).to(self.device).reshape(1, 3, 3)
        self.model.set_hand_context(b, k)
        self.betas_buf.copy_(b)
        self.K_buf.copy_(k)
        self._have_context = True

    def begin_segment(self, prev0) -> None:
        """Per tracked segment: the initial state (51,) and a restarted filter (its velocity is zeroed in place: the graph
        reads those very buffers)."""
        p = torch.as_tensor(prev0, dtype=torch.float32).reshape(1, 51).cpu()
        self.prev_buf.copy_(p)
        self._prev_host = p.numpy().reshape(51).copy()
        self._zero_filter_state()

    def _zero_filter_state(self) -> None:
        if self._flt is not None:
            self._flt._v_root.zero_()
            self._flt._v_trans.zero_()

    # ---- stages (the same functions run eagerly and under capture)
    def _stage_lnes(self) -> None:
        self.x_buf.copy_(self.glnes.build())

    def _stage_model(self, explicit: bool) -> torch.Tensor:
        # graph: betas / K are the static buffers (so a new sequence needs no re-capture and the per-sequence render cache,
        # which is keyed on the model's context tensors, can never be frozen into the graph); eager: the model's context
        # (so the cache is used). Both are bitwise the same render.
        with torch.no_grad():
            if explicit:
                return self.model(self.x_buf, self.prev_buf, self.betas_buf, self.K_buf)
            return self.model(self.x_buf, self.prev_buf)

    def _stage_filter(self, raw: torch.Tensor) -> None:
        """prev_buf <- the filtered state, in place (every read of prev_buf happens before the final copy)."""
        with torch.no_grad():
            flt = self._flt
            if flt is None:
                new = raw
            else:
                n = self.glnes.hdr_dev[self.window - 1:].to(torch.float64)         # (1,) the packet's raw event count
                v0 = (flt._v_root, flt._v_trans)
                flt.trk_model.z = raw
                new = flt(self.x_buf, self.prev_buf, n_events=n)
                if flt._v_root is not v0[0]:                      # the velocity was re-assigned: keep the persistent buffers
                    v0[0].copy_(flt._v_root)
                    v0[1].copy_(flt._v_trans)
                    flt._v_root, flt._v_trans = v0
                flt.trk_model.z = None
            self.prev_buf.copy_(new)

    # ---- graphs
    def _snapshot(self):
        flt = self._flt
        return (self.prev_buf.clone(), None if flt is None else (flt._v_root.clone(), flt._v_trans.clone()))

    def _restore(self, snap) -> None:
        self.prev_buf.copy_(snap[0])
        if snap[1] is not None:
            self._flt._v_root.copy_(snap[1][0])
            self._flt._v_trans.copy_(snap[1][1])

    def _graph(self, key: str):
        """(graph, static output) of `key` in model | filter | all, recorded on first use (the capture executes the stage
        functions a few times eagerly: the state they mutate is saved and restored around it)."""
        if key not in self._graphs:
            if not self._have_context:
                raise RuntimeError("call begin_sequence(betas, camera_K) before the first step")
            if key == "model":
                fn = lambda: self._stage_model(True)                                           # noqa: E731
            elif key == "filter":
                def fn():
                    self._stage_filter(self._stage_model(True))
                    return self.prev_buf
            else:
                def fn():
                    self._stage_lnes()
                    self._stage_filter(self._stage_model(True))
                    return self.prev_buf
            snap = self._snapshot()
            hits0 = _cache_hits(self.model)
            try:
                self._graphs[key] = capture_checked(fn)
                if hits0 is not None and _cache_hits(self.model) != hits0:
                    del self._graphs[key]
                    raise RuntimeError("the per-sequence render cache was used while recording the graph (it would be frozen "
                                       "into it)")
            except RuntimeError as e:
                if "synchroniz" in str(e):
                    raise RuntimeError(str(e) + "\n[DeployTracker] the forward synchronises with the host and cannot be "
                                       "recorded: deploy_fast.enable_fast_render(model) removes the syncs of the dense "
                                       "render arm") from e
                raise
            finally:
                self._restore(snap)
        return self._graphs[key]

    # ---- the step
    def step_sequence(self, events, offsets, end: int) -> np.ndarray:
        """One packet of a recorded sequence: the events of ms `[end - window + 1, end]` (`events` the (N, 3) uint8 array of
        `semkine.eval_track.load_sequence`, `offsets` its per-ms index). Returns the new state (51,) float32."""
        start = end - self.window + 1
        a0, a1 = int(offsets[start]), int(offsets[end + 1])
        return self.step_events(np.asarray(events[a0:a1]), packet_bounds(offsets, end, self.window))

    def step_events(self, ev: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        """One packet from its raw events `ev` (N, 3) uint8 [x, y, p] (time ordered) and `bounds` (window,) the cumulative
        counts at the end of each millisecond (`bounds[-1] == N`). Returns the new state (51,) float32 (a copy)."""
        n = int(bounds[-1])
        gpu_lnes = self.lnes_mode == "gpu" and self.glnes.fits(n)
        if gpu_lnes:
            self.glnes.stage(ev, bounds)
        else:
            if self.lnes_mode == "gpu":
                self.n_fallback += 1
            self.glnes.stage(None, bounds)                                  # the header carries the event count
            lnes = host_lnes_from(ev, bounds, self.window, self.height, self.width, self.channels)
            self.x_buf.copy_(torch.from_numpy(lnes).unsqueeze(0))
        filt_in_graph = self.use_graph and self.scope in ("filter", "all") and self.filt != "host"
        raw = None
        if self.use_graph:
            if self.scope == "all" and gpu_lnes:
                graph, _ = self._graph("all")
            else:
                if gpu_lnes:
                    self._stage_lnes()
                graph, raw = self._graph("filter" if filt_in_graph else "model")
            graph.replay()
        else:
            if gpu_lnes:
                self._stage_lnes()
            raw = self._stage_model(False)
        if self.filt == "host":
            out = self._host_filter(raw, n)
        else:
            if not filt_in_graph:
                self._stage_filter(raw)
            self._out_pin.copy_(self.prev_buf[0], non_blocking=True)
            self._sync()
            out = self._out_pin.numpy().copy()
        self.n_steps += 1
        return out

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()

    def _host_filter(self, raw: torch.Tensor, n_events: int) -> np.ndarray:
        self._raw_pin.copy_(raw[0], non_blocking=True)
        self._sync()
        prev, z = self._prev_host, self._raw_pin.numpy()
        if n_events <= 0:
            new = prev.copy()
        else:
            a_r, a_f, a_t = self._hgains(n_events)
            new = anchor_blend(torch.from_numpy(prev[None]), torch.from_numpy(z.copy()[None]), a_r, a_f, a_t).numpy()[0]
        self._prev_host = new.astype(np.float32)
        self._prev_pin.numpy()[0] = self._prev_host
        self.prev_buf.copy_(self._prev_pin, non_blocking=True)
        return self._prev_host.copy()


# ------------------------------------------------------------------------------------------------ protocol loop
def run_closed_loop(tracker: DeployTracker, cfg, root, d: str, seq: str, rng, record_times: Optional[list] = None) -> dict:
    """`tools/tracking/evalx.run_sequence` (mode "model") with a `DeployTracker`: the protocol -- init state = ground truth
    plus the protocol's noise from `rng` (one rng seeded 0 across the split's sequences in order), 50 ms steps, the
    previous output fed back -- but every step is `tracker.step_sequence`. Returns evalx's dict (`pred`, `gt`, `end`, `run`,
    `elapsed`, `count`, `betas`, `prev` omitted) for `evalx.per_step_metrics`. `record_times`: append each step's seconds."""
    import time
    from semkine import eval_track as ET
    events, offsets, aux, pos51 = ET.load_sequence(Path(root), d, seq)
    tracker.begin_sequence(aux["betas"], aux["camera_K"])
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    preds, gts, ends_all, runs_all, elapsed, counts = [], [], [], [], [], []
    for run_id, (a, b) in enumerate(runs):
        ends = np.arange(a + STEP - 1, b, STEP, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        tracker.begin_segment(prev)
        for end in ends:
            t0 = time.perf_counter()
            pred = tracker.step_sequence(events, offsets, int(end))
            if record_times is not None:
                record_times.append(time.perf_counter() - t0)
            preds.append(pred)
            gts.append(pos51[end])
            ends_all.append(int(end))
            runs_all.append(run_id)
            elapsed.append(int(end - a))
            counts.append(int(offsets[int(end) + 1] - offsets[int(end) - STEP + 1]))
    return {"pred": np.stack(preds).astype(np.float32), "gt": np.stack(gts).astype(np.float32),
            "end": np.asarray(ends_all), "run": np.asarray(runs_all), "elapsed": np.asarray(elapsed),
            "count": np.asarray(counts), "betas": aux["betas"]}
