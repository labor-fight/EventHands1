"""Shared causal event representation lifecycle for opt-in S43 integration.

The bank owns observation memory and an immutable epoch-start condition. It
does not own, infer, or commit a hand state. A tracker explicitly starts the
next epoch after committing its (possibly candidate-external) 51D output.
Such a new epoch is a documented conditional restart with a uniform phase
prior, not a fictitious exact Bayesian handoff.

``ingest`` touches only event slots before a certified watermark seals bins.
BR phase advancement is still global over K*M*P*Q; native image readout is a
global reduction. The full candidate radiance bank is built at reset. No claim
of fully local neural inference is made by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import sys
import time
from typing import Callable, Hashable

import numpy as np

from tools.s43_br.carrier import EvidenceMemory, GridKernel


NATIVE_SIZE = (180, 240)
LOW_SIZE = (24, 32)
MAX_SOURCES_PER_EPOCH = 262144
SENSOR = dict(ap=2, am=2, alpha=.9, nu=.02, cap=8,
              polarity_convention="stored 0/1 assumed ON/OFF; recorded mapping unverified")
COORDINATE_SCALE = np.asarray([.05] * 3 + [.3] * 48, dtype=np.float64)


def _array(value, dtype=None):
    """Detach a possible Tensor; no sequence condition retains an autograd graph."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.array(value, dtype=dtype, copy=True)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _provider(history, betas, K_cam, *, bins):
    from .fast_geometry import render_paths
    return render_paths(history, betas, K_cam, bins=bins)


def _template_prefixes(kernel):
    """Candidate-predicted LNES, averaged over initial phase and shared eta.

    These templates see model paths only. Each prefix uses only its own
    predicted events, not future observations and not the observed native last
    timestamps. Uniform initial phase is propagated deterministically along
    each candidate/eta; averaging does not reinitialize phase in every bin.
    Returned values are timestamp numerators; readout divides by prefix length.
    """
    shape = (kernel.K, kernel.M, kernel.P, kernel.Q)
    phase = np.broadcast_to(np.arange(kernel.Q), shape).copy()
    last_plus = np.zeros(shape, dtype=np.float32)
    last_minus = np.zeros(shape, dtype=np.float32)
    result = np.zeros((kernel.B + 1, kernel.K, 2, kernel.P), dtype=np.float32)
    for b in range(kernel.B):
        destination, plus, minus = kernel.predict(b)
        np.copyto(last_plus, b, where=np.take_along_axis(plus, phase, -1) > 0)
        np.copyto(last_minus, b, where=np.take_along_axis(minus, phase, -1) > 0)
        phase = np.take_along_axis(destination, phase, -1)
        result[b + 1, :, 0] = last_plus.mean(axis=(1, 3))
        result[b + 1, :, 1] = last_minus.mean(axis=(1, 3))
    return result


@dataclass
class _Epoch:
    seq_id: Hashable
    segment_id: object
    start_us: int
    history_time_us: int
    history: np.ndarray
    betas: np.ndarray
    K_cam: np.ndarray
    native_counts: np.ndarray
    seen: dict
    watermark_us: int
    end: int
    live: np.ndarray
    states: np.ndarray | None
    carrier: EvidenceMemory | None
    template_numerators: np.ndarray | None
    geometry_seconds: float
    template_seconds: float
    reset_seconds: float = 0.
    local_ingest_seconds: float = 0.
    phase_advance_seconds: float = 0.


class SequenceBank:
    """Bounded, explicitly keyed observation memory for independent sequences.

    Padding uses ``True = ignore`` and is filtered before payload validation.
    Watermark certifies all observations before that physical timestamp; only
    complete 1ms groups are consumed. Newly staged future groups are allowed,
    but a query can read only the current consumed prefix, never a future or
    historical state without a separately constructed prefix snapshot.
    """

    def __init__(self, mode="br", provider: Callable | None = None,
                 max_sequences=16, max_sources=MAX_SOURCES_PER_EPOCH, bins=50,
                 image_size=LOW_SIZE):
        if mode not in ("legacy", "enhanced", "br"):
            raise ValueError("mode must be legacy, enhanced, or br")
        self.mode, self.provider = mode, provider or _provider
        self.max_sequences = _integer(max_sequences, "max_sequences")
        self.max_sources = _integer(max_sources, "max_sources")
        self.bins = _integer(bins, "bins")
        if min(self.max_sequences, self.max_sources, self.bins) < 1 or self.bins > 50:
            raise ValueError("positive bounds and 1..50 bins required")
        if tuple(image_size) != LOW_SIZE:
            raise ValueError("the registered integration provider fixes 24x32")
        self.image_size = LOW_SIZE
        self.states: dict[Hashable, _Epoch] = {}

    def reset(self, seq_id, start_us, history, betas, K_cam,
              segment_id=None, history_time_us=None):
        reset_start = time.perf_counter()
        hash(seq_id)
        start_us = _integer(start_us, "start_us")
        if seq_id not in self.states and len(self.states) >= self.max_sequences:
            raise OverflowError("sequence bound exceeded; release old sequence explicitly")
        history = _array(history, np.float64).reshape(-1)
        betas = _array(betas, np.float64).reshape(-1)
        camera = _array(K_cam, np.float64)
        if history.shape != (51,) or betas.shape != (10,) or camera.shape != (3, 3):
            raise ValueError("history[51], betas[10], K_cam[3,3] required")
        if not all(np.isfinite(x).all() for x in (history, betas, camera)):
            raise ValueError("finite epoch conditions required")
        stamp = start_us - 1000 if history_time_us is None else _integer(history_time_us, "history_time_us")
        if stamp > start_us:
            raise ValueError("future hand condition is not a causal epoch boundary")
        states = carrier = templates = None
        geometry_seconds = template_seconds = 0.
        if self.mode == "br":
            before = time.perf_counter()
            states, paths = self.provider(history.copy(), betas.copy(), camera.copy(), bins=self.bins)
            states = _array(states, np.float64)
            paths = _array(paths)
            if states.shape != (3, self.bins + 1, 51) or paths.shape != (3, 3, np.prod(LOW_SIZE), self.bins + 1):
                raise ValueError("provider must return states[3,B+1,51], paths[3,3,768,B+1]")
            if not np.isfinite(states).all() or not np.issubdtype(paths.dtype, np.integer):
                raise ValueError("finite candidate states and integer absolute radiance paths required")
            kernel = GridKernel(paths, context_id=f"{seq_id!r}:{segment_id!r}:{start_us}", **SENSOR)
            carrier = EvidenceMemory(kernel, start_us=start_us, bin_us=1000,
                                     max_sources=self.max_sources, max_refresh=0)
            geometry_seconds = time.perf_counter() - before
            before = time.perf_counter()
            templates = _template_prefixes(kernel)
            template_seconds = time.perf_counter() - before
        native = np.zeros((self.bins, np.prod(NATIVE_SIZE), 2), dtype=np.uint32)
        for condition in (history, betas, camera, states, templates):
            if condition is not None:
                condition.flags.writeable = False
        epoch = _Epoch(seq_id, segment_id, start_us, stamp, history, betas, camera,
                       native, {}, start_us, 0, np.ones(self.bins, dtype=bool),
                       states, carrier, templates, geometry_seconds, template_seconds)
        epoch.reset_seconds = time.perf_counter() - reset_start
        # A failed provider never destroys a previously valid state.
        self.states[seq_id] = epoch
        return self

    def release(self, seq_id):
        if seq_id not in self.states:
            raise KeyError(seq_id)
        del self.states[seq_id]

    def _get(self, seq_id):
        if seq_id not in self.states:
            raise KeyError(f"sequence {seq_id!r} has not been reset")
        return self.states[seq_id]

    def ingest(self, seq_id, events, source_ids, watermark_us,
               padding_mask=None, live=None):
        epoch = self._get(seq_id)
        local_start = time.perf_counter()
        watermark = _integer(watermark_us, "watermark_us")
        if not epoch.watermark_us <= watermark <= epoch.start_us + self.bins * 1000:
            raise ValueError("nonmonotone/outside-epoch watermark")
        a, ids = _array(events), _array(source_ids)
        if a.ndim != 2 or a.shape[1] != 4 or ids.shape != (len(a),):
            raise ValueError("events[N,4] absolute_us,x,y,stored_p and source_ids[N] required")
        if padding_mask is not None:
            mask = _array(padding_mask)
            if mask.shape != (len(a),) or mask.dtype != np.bool_:
                raise ValueError("padding_mask must be boolean [N], True means ignore")
            a, ids = a[~mask], ids[~mask]
        # Removing invalid padded rows can leave a float array even when all
        # retained fields are exact integers. Validate retained semantics before
        # conversion; no units/rounding inference or silent timestamp rounding.
        for values in (a, ids):
            if len(values) and not np.issubdtype(values.dtype, np.integer):
                if (values.dtype.kind not in 'fc' or np.iscomplexobj(values)
                        or not np.isfinite(values).all() or np.any(np.abs(values) > 2**53)
                        or np.any(values != np.floor(values))):
                    raise ValueError("integer event/source coordinates required")
        a, ids = a.astype(np.int64), ids.astype(np.int64)
        end = (watermark - epoch.start_us) // 1000
        available = epoch.live.copy() if live is None else _array(live)
        if available.shape != (self.bins,) or available.dtype != np.bool_:
            raise ValueError("live must be boolean [B] acquisition availability")
        if not np.array_equal(available[:epoch.end], epoch.live[:epoch.end]):
            raise ValueError("cannot rewrite consumed live/dropout state")
        if not available.all() and np.any(epoch.native_counts[~available] != 0):
            raise ValueError("dropout bin contains certified observations")
        incoming = {}
        for row, uid in zip(a, ids):
            uid = int(uid); stamp, x, y, polarity = map(int, row)
            if uid < 0 or not 0 <= x < 240 or not 0 <= y < 180 or polarity not in (0, 1):
                raise ValueError("source ID/pixel/stored polarity outside declared input contract")
            payload = (stamp, x, y, polarity)
            old = epoch.seen.get(uid, incoming.get(uid))
            if old is not None:
                if old != payload:
                    raise ValueError("source ID has conflicting payload")
                continue
            b = (stamp - epoch.start_us) // 1000
            if not epoch.end <= b < self.bins:
                raise ValueError("new observation is late or outside epoch")
            if not available[b]:
                raise ValueError("dropout bin also contains incoming observations")
            incoming[uid] = payload
        if len(epoch.seen) + len(incoming) > self.max_sources:
            raise OverflowError("source bound exceeded; no silent event dropping")
        if incoming:
            payloads = np.asarray(list(incoming.values()), dtype=np.int64)
            b = (payloads[:, 0] - epoch.start_us) // 1000
            x, y, polarity = payloads[:, 1], payloads[:, 2], payloads[:, 3]
            np.add.at(epoch.native_counts, (b, y * 240 + x, polarity), 1)
            if epoch.carrier is not None:
                low_x = (2 * x + 1) * LOW_SIZE[1] // (2 * 240)
                low_y = (2 * y + 1) * LOW_SIZE[0] // (2 * 180)
                low_index = (b, low_y * LOW_SIZE[1] + low_x, polarity)
                np.add.at(epoch.carrier.counts, low_index, 1)
                # Only touched low-resolution slots need saturation. Repeated
                # advanced indices all see the same post-add value, so this is
                # exactly the former full-buffer clip without global scanning.
                epoch.carrier.counts[low_index] = np.minimum(epoch.carrier.counts[low_index], SENSOR['cap'])
            epoch.seen.update(incoming)
        epoch.local_ingest_seconds += time.perf_counter() - local_start
        before = time.perf_counter()
        if epoch.carrier is not None:
            epoch.carrier.advance(epoch.start_us + end * 1000, watermark_us=watermark,
                                  live=np.broadcast_to(available[:, None], epoch.carrier.live.shape))
        epoch.phase_advance_seconds += time.perf_counter() - before
        epoch.live[:] = available
        epoch.watermark_us = watermark
        epoch.end = end
        return len(incoming)

    def readout(self, seq_id, cutoff_us):
        epoch = self._get(seq_id)
        cutoff = _integer(cutoff_us, "cutoff_us")
        if cutoff != epoch.start_us + epoch.end * 1000:
            raise ValueError("readout requires current sealed prefix; no future/history query")
        before = time.perf_counter()
        count = epoch.native_counts[:epoch.end]
        lnes = np.zeros((*NATIVE_SIZE, 2), dtype=np.float32)
        first = np.zeros_like(lnes)
        count_plane = np.zeros_like(lnes)
        if epoch.end:
            present = count > 0
            ticks = np.arange(epoch.end)[:, None, None]
            lnes[:] = (np.where(present, ticks, 0).max(0) / float(epoch.end)).reshape(*NATIVE_SIZE, 2)
            earliest = np.where(present, ticks, epoch.end).min(0)
            first[:] = np.where(earliest < epoch.end, earliest / float(epoch.end), 0).reshape(*NATIVE_SIZE, 2)
            # Match the existing EVENT_CHANNELS builder's float32 log-count
            # arithmetic, including its one-ulp rounding, for exact controls.
            totals = count.sum(0).astype(np.float32)
            count_plane[:] = (np.log1p(totals) / np.log1p(32.)).clip(max=1).reshape(*NATIVE_SIZE, 2)
        enhanced = np.concatenate([lnes, count_plane, first], axis=-1)
        R = np.zeros((3, 53), dtype=np.float32)
        templates = np.zeros((3, 2, *LOW_SIZE), dtype=np.float32)
        status = 'ok'
        if epoch.carrier is not None:
            coords = (epoch.states[:, epoch.end] - epoch.history) / COORDINATE_SCALE
            result = epoch.carrier.query(coords, cutoff_us=cutoff)
            R = result['R'].astype(np.float32)
            status = result['status']
            if epoch.end:
                templates[:] = (epoch.template_numerators[epoch.end] / epoch.end).reshape(3, 2, *LOW_SIZE)
        elapsed = time.perf_counter() - before
        return {'lnes': lnes, 'enhanced': enhanced, 'R': R, 'templates': templates,
                'has_events': bool(np.any(count)), 'history': epoch.history.copy(),
                'betas': epoch.betas.copy(), 'K_cam': epoch.K_cam.copy(), 'seq_id': seq_id,
                'meta': {'mode': self.mode, 'status': status, 'padding': False,
                         'epoch_start_us': epoch.start_us, 'cutoff_us': cutoff,
                         'target_time_us': cutoff - 1000 if epoch.end else cutoff,
                         'history_time_us': epoch.history_time_us, 'segment_id': epoch.segment_id,
                         'consumed_bins': epoch.end, 'watermark_us': epoch.watermark_us,
                         'native_source_count': sum(1 for stamp, *_ in epoch.seen.values() if stamp < cutoff),
                         'retained_source_count': len(epoch.seen), 'phase_boundary': 'uniform conditional restart',
                         'template_semantics': 'candidate predicted last-LNES, uniform initial phase and eta; no observed-last bypass',
                         'polarity': 'stored 0/1; assumed model mapping unverified',
                         'geometry_seconds': epoch.geometry_seconds, 'template_seconds': epoch.template_seconds,
                         'reset_seconds': epoch.reset_seconds,
                         'local_ingest_seconds': epoch.local_ingest_seconds,
                         'phase_advance_seconds': epoch.phase_advance_seconds,
                         'readout_seconds': elapsed}}

    def digest(self, seq_id=None):
        selected = self.states.items() if seq_id is None else [(seq_id, self._get(seq_id))]
        h = hashlib.sha256()
        for sid, state in sorted(selected, key=lambda row: repr(row[0])):
            h.update(repr((sid, state.segment_id, state.start_us, state.end, state.watermark_us,
                           state.history_time_us, sorted(state.seen.items()))).encode())
            for array in (state.history, state.betas, state.K_cam, state.native_counts, state.live):
                h.update(array.tobytes())
            if state.carrier is not None:
                h.update(state.states.tobytes())
                h.update(state.template_numerators.tobytes())
                h.update(state.carrier.digest().encode())
        return h.hexdigest()

    def memory_bytes(self, seq_id=None):
        selected = list(self.states.values()) if seq_id is None else [self._get(seq_id)]
        arrays = source = 0
        for state in selected:
            owned = [state.history, state.betas, state.K_cam, state.native_counts, state.live]
            if state.states is not None:
                owned += [state.states, state.template_numerators]
            arrays += sum(x.nbytes for x in owned)
            source += sys.getsizeof(state.seen) + sum(
                sys.getsizeof(uid) + sys.getsizeof(payload) + sum(sys.getsizeof(v) for v in payload)
                for uid, payload in state.seen.items())
            if state.carrier is not None:
                cost = state.carrier.memory_bytes()
                arrays += cost['array_bytes']; source += cost['source_bytes']
        return {'array_bytes': int(arrays), 'source_bytes': int(source),
                'total_accounted_bytes': int(arrays + source),
                'sequences': len(selected), 'max_sequences': self.max_sequences,
                'max_sources_per_epoch': self.max_sources, 'bins_per_epoch': self.bins,
                'source_count': sum(len(s.seen) for s in selected),
                'scope': 'owned arrays and source objects; not process RSS or temporary readout workspace'}


def _padding_readout(window, mode):
    return {'lnes': np.zeros((*NATIVE_SIZE, 2), np.float32),
            'enhanced': np.zeros((*NATIVE_SIZE, 6), np.float32),
            'R': np.zeros((3, 53), np.float32), 'templates': np.zeros((3, 2, *LOW_SIZE), np.float32),
            'has_events': False, 'history': np.zeros(51), 'betas': np.zeros(10),
            'K_cam': np.eye(3), 'seq_id': window.get('seq_id'),
            'meta': {'mode': mode, 'padding': True, 'status': 'padding', 'native_source_count': 0}}


def batch_from_windows(windows, *, mode="br", provider=None, bins=50,
                       max_sources=MAX_SOURCES_PER_EPOCH, image_size=LOW_SIZE):
    """Independent-window batch constructor using the exact streaming core.

    Windows contain seq_id/start_us/history/betas/K_cam/events/source_ids and
    optional cutoff_us, segment_id, event_padding_mask, live. Events are
    absolute unless the caller explicitly supplies time_is_relative=True.
    valid=False skips the row entirely, including invalid padded payloads.
    Repeated same sequence IDs are independent windows here (released after
    each row); a persistent prefix batch instead calls SequenceBank directly.
    """
    bank = SequenceBank(mode=mode, provider=provider, bins=bins,
                        max_sources=max_sources, max_sequences=1, image_size=image_size)
    output = []
    for index, window in enumerate(windows):
        if not bool(window.get('valid', True)):
            output.append(_padding_readout(window, mode)); continue
        sid = window.get('seq_id', index)
        start = _integer(window['start_us'], "start_us")
        history = window.get('history', window.get('prev_state'))
        camera = window.get('K_cam', window.get('camera_K'))
        bank.reset(sid, start, history, window['betas'], camera,
                   segment_id=window.get('segment_id'), history_time_us=window.get('history_time_us'))
        events = _array(window['events'])
        if bool(window.get('time_is_relative', False)):
            events[:, 0] += start
        cutoff = _integer(window.get('cutoff_us', start + bins * 1000), "cutoff_us")
        bank.ingest(sid, events, window['source_ids'],
                    watermark_us=window.get('watermark_us', cutoff),
                    padding_mask=window.get('event_padding_mask', window.get('padding_mask')),
                    live=window.get('live'))
        output.append(bank.readout(sid, cutoff))
        bank.release(sid)
    return output
