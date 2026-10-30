"""Causal F16 event fields, chronological lanes, and isolated XJ supervision.

The deployment input owner never opens pose metadata or retrieves ``betas``.
An input packet contains native ``[x, y, absolute_time_us, stored_polarity]``
events.  At every half-open 50 ms boundary, F16 consists of six latest-age
surfaces (tau 10/50/200 ms, each with two stored-polarity channels), followed
by five chronological 10 ms count bins, each with two polarity channels.
The count channels are raw counts unless ``log_counts=True`` is explicitly
selected and recorded.  The last-age history is clipped to 600 ms.

The original project split is only filtered by the requested training
subjects.  This module does not invent a validation/test partition.  It
deliberately refuses ``zgz`` in the training lane constructor.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import struct
from typing import Callable, Iterable, Sequence

import numpy as np


DATA_ROOT = Path('/data1/lyq/code/mesh/EventHands/data/hand_data51')
FIT7 = ('ch', 'lfz', 'lpc', 'lr', 'ly', 'lyh', 'lyq')
SELECT = ('ycy', 'ylf')
ALL_TRAIN_SUBJECTS = FIT7 + SELECT
HEIGHT, WIDTH = 180, 240
STEP_MS, HISTORY_MS = 50, 600
TAUS_MS = (10.0, 50.0, 200.0)
PUBLIC_INITIAL_STATE = np.asarray([0., 0., .6] + [0.] * 48, np.float32)
FIELD_CHANNELS = tuple(
    [f'latest_age_tau{int(tau)}_pol{p}' for tau in TAUS_MS for p in range(2)]
    + [f'count_bin{b}_pol{p}' for b in range(5) for p in range(2)]
)


def _readonly(array: np.ndarray) -> np.ndarray:
    array.flags.writeable = False
    return array


def _validate_native_events(events, *, height=HEIGHT, width=WIDTH):
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4 or events.dtype.kind not in 'iu':
        raise ValueError('native events must be integer [x,y,absolute_time_us,stored_polarity]')
    if len(events) and (
        np.any(events[:, 0] < 0) or np.any(events[:, 0] >= width)
        or np.any(events[:, 1] < 0) or np.any(events[:, 1] >= height)
        or np.any((events[:, 3] != 0) & (events[:, 3] != 1))
    ):
        raise ValueError('event coordinate/polarity outside declared sensor domain')
    return events


def advance_field16(latest_us: np.ndarray, events: np.ndarray, *,
                    start_ms: int, cutoff_ms: int, run_start_ms: int,
                    history_ms: int = HISTORY_MS, log_counts: bool = False):
    """One causal 50 ms update; return ``(F16, new_latest_us, metadata)``.

    ``latest_us`` is not modified.  Latest ages use the exact timestamp
    max, so tied timestamps and event-row permutations do not change F16.
    The caller must supply precisely the new ``[start_ms, cutoff_ms)``
    packet.  No state or target pose argument exists.
    """
    latest_us = np.asarray(latest_us)
    if latest_us.ndim != 3 or latest_us.shape[0] != 2 or latest_us.dtype.kind not in 'iu':
        raise ValueError('latest timestamps must be integer (2,H,W)')
    _, height, width = latest_us.shape
    start_ms, cutoff_ms, run_start_ms = map(int, (start_ms, cutoff_ms, run_start_ms))
    if cutoff_ms - start_ms != STEP_MS or start_ms < run_start_ms or history_ms < STEP_MS:
        raise ValueError('F16 updates require a causal 50 ms packet inside its valid run')
    events = _validate_native_events(events, height=height, width=width)
    lo_us, hi_us = start_ms * 1000, cutoff_ms * 1000
    if len(events) and (np.any(events[:, 2] < lo_us) or np.any(events[:, 2] >= hi_us)):
        raise ValueError('packet includes an event outside its half-open causal interval')
    if np.any(latest_us >= lo_us):
        raise ValueError('latest state already consumed this packet or future events')
    latest = latest_us.copy()
    counts = np.zeros((10, height, width), np.float32)
    if len(events):
        x = events[:, 0].astype(np.intp, copy=False)
        y = events[:, 1].astype(np.intp, copy=False)
        p = events[:, 3].astype(np.intp, copy=False)
        t = events[:, 2].astype(np.int64, copy=False)
        np.maximum.at(latest, (p, y, x), t)
        b = ((t - lo_us) // 10000).astype(np.intp)
        key = ((b * 2 + p) * height + y) * width + x
        counts = np.bincount(key, minlength=10 * height * width).reshape(10, height, width).astype(np.float32)
    age_ms = (hi_us - latest).astype(np.float64) / 1000.0
    observed = (latest >= run_start_ms * 1000) & (latest >= hi_us - int(history_ms) * 1000)
    age_faces = [np.where(observed, np.exp(-age_ms / tau), 0.).astype(np.float32)
                 for tau in TAUS_MS]
    if log_counts:
        np.log1p(counts, out=counts)
    field = np.concatenate([*age_faces, counts], axis=0)
    metadata = dict(
        start_ms=start_ms, cutoff_ms=cutoff_ms,
        history_exposure_ms=min(cutoff_ms - run_start_ms, int(history_ms)),
        current_exposure_ms=STEP_MS, event_count=len(events),
        current_window_observed=True, current_window_empty=len(events) == 0,
        memory_ms=int(history_ms), log_counts=bool(log_counts),
        timestamp_contract='absolute integer us; tied events are simultaneous',
    )
    return _readonly(field), _readonly(latest), metadata


class NativeInputReader:
    """Input-only mmap owner; only K and valid-run auxiliary members are read."""

    def __init__(self, recording: str, root=DATA_ROOT, *, legacy_dir=None):
        self.root, self.recording = Path(root), str(recording)
        manifest_path = self.root / 'splits_semkine.json'
        manifest = json.loads(manifest_path.read_text())
        training = manifest['train']
        if self.recording not in training['trials']:
            raise ValueError('input recording is outside the existing training split')
        actual_dir = training['legacy_dir'][self.recording]
        if legacy_dir is not None and str(legacy_dir) != actual_dir:
            raise ValueError('recording directory differs from the unique split manifest')
        self.base = self.root / actual_dir / self.recording
        self.events = np.load(str(self.base) + '_events.npy', mmap_mode='r', allow_pickle=False)
        self.offsets = np.load(str(self.base) + '_offsets.npy', mmap_mode='r', allow_pickle=False)
        self.tsub = np.load(str(self.base) + '_tsub.npy', mmap_mode='r', allow_pickle=False)
        with np.load(str(self.base) + '_aux.npz', allow_pickle=False) as aux:
            self.K_source = np.asarray(aux['camera_K'], np.float32).reshape(3, 3).copy()
            self.runs = np.asarray(aux['valid_runs_ms'], np.int64).reshape(-1, 2).copy()
        if self.events.ndim != 2 or self.events.shape[1] != 3 or self.events.dtype.kind not in 'iu':
            raise ValueError('native source must have integer [x,y,stored_polarity]')
        if self.offsets.ndim != 1 or self.offsets.dtype.kind not in 'iu' or int(self.offsets[0]) != 0:
            raise ValueError('native integer offsets beginning at zero required')
        if np.any(np.diff(self.offsets) < 0) or int(self.offsets[-1]) != len(self.events) or len(self.tsub) != len(self.events):
            raise ValueError('event/offset/sub-ms source identities differ')
        if not np.isfinite(self.K_source).all() or self.K_source[0, 0] <= 0 or self.K_source[1, 1] <= 0:
            raise ValueError('finite external camera intrinsics required')

    def packet(self, start_ms: int, cutoff_ms: int):
        start_ms, cutoff_ms = int(start_ms), int(cutoff_ms)
        if not 0 <= start_ms <= cutoff_ms < len(self.offsets):
            raise ValueError('packet outside source exposure')
        lo, hi = int(self.offsets[start_ms]), int(self.offsets[cutoff_ms])
        ev = np.asarray(self.events[lo:hi], np.int64)
        ms = np.repeat(np.arange(start_ms, cutoff_ms, dtype=np.int64),
                       np.diff(np.asarray(self.offsets[start_ms:cutoff_ms + 1], np.int64)))
        sub = np.asarray(self.tsub[lo:hi], np.int64)
        if len(sub) and (np.any(sub < 0) or np.any(sub >= 1000)):
            raise ValueError('sub-ms timestamp outside declared source bin')
        packet = np.empty((len(ev), 4), np.int64)
        packet[:, :2], packet[:, 2], packet[:, 3] = ev[:, :2], ms * 1000 + sub, ev[:, 2]
        _validate_native_events(packet)
        return dict(events=_readonly(packet), source_range=(lo, hi))

    def input_paths(self):
        return tuple(Path(str(self.base) + suffix) for suffix in
                     ('_events.npy', '_offsets.npy', '_tsub.npy', '_aux.npz'))


@dataclass(frozen=True)
class Run:
    recording: str
    run_id: int
    start_ms: int
    stop_ms: int


@dataclass(frozen=True)
class XJInputBatch:
    items: tuple
    token: str
    batch_id: int

    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __getitem__(self, index): return self.items[index]

    @property
    def supervised_mask(self):
        return np.asarray([item.get('supervised', False) for item in self.items], bool)


class XJRunLanes:
    """Transactional, chronological event-only lanes with two immutable F16 snapshots.

    The public initial pose is supplied on reset.  Own-state recurrence is
    entirely the trainer's responsibility; there is no GT/teacher-state input
    here.  ``peek_batch`` is repeatable, and only ``commit_batch`` advances
    the data cursor and the event field together.
    """

    def __init__(self, root=DATA_ROOT, *, subjects=FIT7, recordings=None,
                 batch_size=8, seed=3407, warmup_steps=2, finite=False,
                 log_counts=False, memory_ms=HISTORY_MS,
                 reader_factory: Callable = NativeInputReader):
        self.root = Path(root)
        self.subjects = tuple(subjects)
        self.batch_size, self.warmup_steps = int(batch_size), int(warmup_steps)
        self.finite, self.log_counts, self.memory_ms = bool(finite), bool(log_counts), int(memory_ms)
        if self.batch_size < 1 or self.warmup_steps < 1:
            raise ValueError('positive batch size and at least one label-free warmup step required')
        if not self.subjects or set(self.subjects) - set(ALL_TRAIN_SUBJECTS):
            raise ValueError('training lanes only use existing FIT7/SELECT subjects, never zgz')
        self.manifest_path = self.root / 'splits_semkine.json'
        manifest_bytes = self.manifest_path.read_bytes()
        self.manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest = json.loads(manifest_bytes)
        declared = manifest['train']['trials']
        names = sorted(recordings if recordings is not None else declared)
        if set(names) - set(declared):
            raise ValueError('recordings outside the unique training split manifest')
        self.readers = {}
        self.runs = []
        for name in names:
            if name.split('_')[0] not in self.subjects:
                continue
            reader = reader_factory(name, self.root)
            self.readers[name] = reader
            for rid, (a, b) in enumerate(reader.runs):
                a, b = int(a), int(b)
                if b - a >= STEP_MS:
                    self.runs.append(Run(name, rid, a, b))
        if not self.runs:
            raise ValueError('no 50 ms eligible runs in requested existing training recordings')
        rng = np.random.default_rng(int(seed))
        self._state = dict(rng=rng.bit_generator.state, order=np.empty(0, np.int64), position=0,
                           traversal=0, lanes=[None] * self.batch_size, batch_id=0,
                           outputs=0, supervised_outputs=0, events=0, resets=0,
                           tail_ms=0)
        self._pending = None

    def _new_run(self, state, lane):
        if state['position'] >= len(state['order']):
            if self.finite and state['traversal'] > 0:
                return None
            rng = np.random.default_rng()
            rng.bit_generator.state = state['rng']
            state['order'] = rng.permutation(len(self.runs))
            state['rng'] = rng.bit_generator.state
            state['position'] = 0
            state['traversal'] += 1
        run = self.runs[int(state['order'][state['position']])]
        state['position'] += 1
        state['resets'] += 1
        return dict(run=run, cursor_ms=run.start_ms, ordinal=0,
                    seq_id=f"xj:lane{lane}:pass{state['traversal']}:{state['position']}:{run.recording}:{run.run_id}",
                    latest_us=_readonly(np.full((2, HEIGHT, WIDTH), -1, np.int64)),
                    field=_readonly(np.zeros((16, HEIGHT, WIDTH), np.float32)),
                    exposure_ms=0)

    def _copy_state(self):
        state = {k: (copy.deepcopy(v) if k not in ('lanes',) else None) for k, v in self._state.items()}
        state['lanes'] = [None if lane is None else dict(lane) for lane in self._state['lanes']]
        return state

    def peek_batch(self):
        if self._pending is not None:
            return self._pending[0]
        state = self._copy_state()
        items = []
        h = hashlib.sha256(str(state['batch_id']).encode())
        for lane in range(self.batch_size):
            current = state['lanes'][lane]
            reset = current is None or current['cursor_ms'] + STEP_MS > current['run'].stop_ms
            if reset and current is not None:
                state['tail_ms'] += current['run'].stop_ms - current['cursor_ms']
            if reset:
                current = state['lanes'][lane] = self._new_run(state, lane)
            if current is None:
                items.append(dict(lane=lane, valid=False, padding=True, supervised=False))
                continue
            run, start = current['run'], current['cursor_ms']
            stop = start + STEP_MS
            reader = self.readers[run.recording]
            packet = reader.packet(start, stop)
            field1, latest, exposure = advance_field16(
                current['latest_us'], packet['events'], start_ms=start, cutoff_ms=stop,
                run_start_ms=run.start_ms, history_ms=self.memory_ms, log_counts=self.log_counts)
            supervised = current['ordinal'] >= self.warmup_steps
            item = dict(
                lane=lane, valid=True, padding=False, supervised=supervised, reset=reset,
                recording=run.recording, run_id=run.run_id, run_start_ms=run.start_ms,
                run_stop_ms=run.stop_ms, seq_id=current['seq_id'], packet_ordinal=current['ordinal'],
                start_ms=start, cutoff_ms=stop, reference_ms=start - 1, target_ms=stop - 1,
                field0=current['field'], field1=field1,
                field0_exposure_ms=current['exposure_ms'], field1_exposure_ms=exposure['history_exposure_ms'],
                both_snapshots_valid=current['exposure_ms'] >= STEP_MS,
                initial_state=_readonly(PUBLIC_INITIAL_STATE.copy()),
                beta_input=_readonly(np.zeros(10, np.float32)),
                K_source=_readonly(np.asarray(reader.K_source, np.float32).copy()),
                field_metadata=exposure, **packet,
            )
            if set(item) & {'target', 'targets', 'target0', 'target1', 'betas', 'beta_target', 'previous_GT'}:
                raise AssertionError('supervision entered an event input item')
            items.append(item)
            h.update(repr((item['seq_id'], start, stop, packet['source_range'], reset, supervised)).encode())
            h.update(packet['events'].tobytes())
            h.update(field1.tobytes())
            h.update(item['K_source'].tobytes())
            current.update(cursor_ms=stop, ordinal=current['ordinal'] + 1,
                           latest_us=latest, field=field1, exposure_ms=exposure['history_exposure_ms'])
            state['outputs'] += 1
            state['supervised_outputs'] += int(supervised)
            state['events'] += len(packet['events'])
        batch = XJInputBatch(tuple(items), h.hexdigest(), state['batch_id'])
        state['batch_id'] += 1
        self._pending = (batch, state)
        return batch

    next_batch = peek_batch

    def commit_batch(self, batch):
        if self._pending is None or batch is not self._pending[0]:
            raise ValueError('foreign or duplicate input cursor commit')
        self._state = self._pending[1]
        self._pending = None

    @property
    def exhausted(self):
        return (self.finite and self._state['traversal'] > 0
                and self._state['position'] >= len(self.runs)
                and all(lane is None for lane in self._state['lanes']))

    def identity(self):
        return dict(root=str(self.root), subjects=self.subjects, batch_size=self.batch_size,
                    finite=self.finite, warmup_steps=self.warmup_steps,
                    memory_ms=self.memory_ms, log_counts=self.log_counts,
                    manifest_sha256=self.manifest_sha256,
                    runs=[asdict(run) for run in self.runs])

    def state_dict(self):
        if self._pending is not None:
            raise RuntimeError('checkpoint only after a completed input/model commit')
        return copy.deepcopy(dict(schema='xvm_joint_lane_schedule_v1', identity=self.identity(), state=self._state))

    def load_state_dict(self, saved):
        if saved['schema'] != 'xvm_joint_lane_schedule_v1' or saved['identity'] != self.identity():
            raise ValueError('checkpoint schedule identity differs')
        self._state = copy.deepcopy(saved['state'])
        self._pending = None

    def manifest(self):
        return dict(self.identity(), window_ms=STEP_MS, hop_ms=STEP_MS,
                    field_shape=[16, HEIGHT, WIDTH], field_channels=list(FIELD_CHANNELS),
                    input_keys='events,F0,F1,K_source,beta0,public_initial,causal scheduling metadata',
                    target_pose_or_beta_in_input=False, input_state='trainer-owned self feedback only',
                    warmup='first warmup_steps outputs from public state; no targets read',
                    tail='short tail has no full50ms query, feedback or supervision',
                    time_contract='absolute native us, half-open packets; current target cutoff_ms-1',
                    reference_target_contract='start_ms-1, exactly50ms before target after warmup',
                    field_write='latest timestamp max + raw five-bin count, immutable query snapshots',
                    cursor='peek/commit; lane field and source cursor commit together')


class _SupervisionReader:
    """Score-only source; never constructed by the input lane owner."""

    def __init__(self, recording, root=DATA_ROOT):
        root = Path(root)
        manifest = json.loads((root / 'splits_semkine.json').read_text())
        if recording not in manifest['train']['trials']:
            raise ValueError('supervision recording outside existing training manifest')
        base = root / manifest['train']['legacy_dir'][recording] / recording
        meta = Path(str(base) + '.meta')
        with meta.open('rb') as stream:
            n, = struct.unpack('<i', stream.read(4))
        if n != 51:
            raise ValueError('expected51D supervision metadata')
        dtype = np.dtype([('data', '<f8', 51), ('m0', 'u1'), ('m1', 'u1')])
        if (meta.stat().st_size - 4) % dtype.itemsize:
            raise ValueError('truncated supervision metadata')
        self.labels = np.memmap(meta, dtype=dtype, mode='r', offset=4)
        with np.load(str(base) + '_aux.npz', allow_pickle=False) as aux:
            self.beta_target = np.asarray(aux['betas'], np.float32).reshape(10).copy()

    def target(self, index):
        row = self.labels[int(index)]
        if int(row['m0']) != 4 or int(row['m1']) != 13:
            raise ValueError('supervision metadata marker mismatch')
        return np.asarray(row['data'], np.float32).copy()


@dataclass(frozen=True)
class XJSupervisionBatch:
    target0: np.ndarray
    target1: np.ndarray
    beta_target: np.ndarray
    mask: np.ndarray
    input_token: str


class XJSupervisionReader:
    """Labels are returned only after the caller acknowledges completed inputs.

    ``read`` must be called after all current event fields, candidate geometry
    and forward inputs have been constructed.  Returned arrays are separate
    supervision slots; they must never be fed back as the deployment state.
    """

    def __init__(self, root=DATA_ROOT, *, reader_factory=_SupervisionReader):
        self.root, self._factory = Path(root), reader_factory
        self._readers = {}
        self.reads = 0

    def read(self, batch: XJInputBatch, *, inputs_complete: bool):
        if not inputs_complete or not isinstance(batch, XJInputBatch) or not batch.token:
            raise RuntimeError('construct all input fields and geometry before supervision access')
        mask = batch.supervised_mask
        target0 = np.zeros((len(batch), 51), np.float32)
        target1 = np.zeros_like(target0)
        beta = np.zeros((len(batch), 10), np.float32)
        for i, (item, selected) in enumerate(zip(batch, mask)):
            if not selected:
                continue
            name = item['recording']
            if name not in self._readers:
                self._readers[name] = self._factory(name, self.root)
            reader = self._readers[name]
            target0[i] = reader.target(item['reference_ms'])
            target1[i] = reader.target(item['target_ms'])
            beta[i] = reader.beta_target
            self.reads += 1
        return XJSupervisionBatch(_readonly(target0), _readonly(target1), _readonly(beta),
                                  _readonly(mask), batch.token)


def stack_inputs(batch: XJInputBatch):
    """Dense NumPy tensors for the model, deliberately without supervision."""
    valid = np.asarray([item.get('valid', False) for item in batch], bool)
    n = len(batch)
    result = dict(field0=np.zeros((n, 16, HEIGHT, WIDTH), np.float32),
                  field1=np.zeros((n, 16, HEIGHT, WIDTH), np.float32),
                  K_source=np.zeros((n, 3, 3), np.float32),
                  beta_input=np.zeros((n, 10), np.float32),
                  initial_state=np.repeat(PUBLIC_INITIAL_STATE[None], n, axis=0),
                  valid=valid, reset=np.zeros(n, bool), both_snapshots_valid=np.zeros(n, bool),
                  supervised_mask=batch.supervised_mask, input_token=batch.token)
    for i, item in enumerate(batch):
        if not valid[i]:
            continue
        for key in ('field0', 'field1', 'K_source', 'beta_input', 'initial_state', 'reset', 'both_snapshots_valid'):
            result[key][i] = item[key]
    return result
