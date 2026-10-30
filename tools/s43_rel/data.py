"""Chronological run lanes with continuous sensor history and separate targets.

Only valid-segment boundaries reset a memory.  Ordinary optimizer steps and
TBPTT detach points preserve its numeric A/Z state.  No per-window Z cache is
written to disk.  The source loader is read-only and uses native integer-ms
timestamps, ignoring the untrusted order within a quantized group.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
from pathlib import Path
import numpy as np

from tools.s43_br.data import DATA_ROOT, FIT7, SELECT, PUBLIC_INITIAL_STATE, SequenceReader, recording_dirs


@dataclass(frozen=True)
class Run:
    recording: str
    run_id: int
    start_ms: int
    stop_ms: int


class ContinuousRunLanes:
    """Deterministically shuffled runs, chronological W50 outputs inside each.

`next_batch` replaces finished lanes at a real run boundary.  A lane identity
includes its traversal number, so two lanes never share a mutable memory.
There is no GT warm start and no GT previous pose.  Targets are loaded by the
training caller only after inputs have been formed.
    """

    def __init__(self, root=DATA_ROOT, *, subjects=FIT7, batch_size=8,
                 hop_ms=50, seed=3407, allow_report_subject=False):
        self.root, self.subjects = Path(root), tuple(subjects)
        if "zgz" in subjects and not allow_report_subject:
            raise ValueError("zgz is report-only; explicit report scope required")
        if batch_size < 1 or hop_ms != 50:
            raise ValueError("training uses positive lane count and common W50/hop50")
        self.batch_size, self.hop_ms = int(batch_size), int(hop_ms)
        self.rng = np.random.default_rng(int(seed))
        self.readers = {}
        self.runs = []
        for name in sorted(recording_dirs(self.root)):
            if name.split("_")[0] not in self.subjects:
                continue
            reader = self.readers[name] = SequenceReader(name, self.root)
            for run_id, (a, b) in enumerate(reader.runs):
                if int(b) - int(a) >= self.hop_ms:
                    self.runs.append(Run(name, run_id, int(a), int(b)))
        if not self.runs:
            raise ValueError("no eligible training runs")
        self.order = np.empty(0, np.int64)
        self.position = 0
        self.traversal = 0
        self.lanes = [None] * self.batch_size
        self.outputs = self.events = self.resets = self.tail_events = self.tail_ms = 0

    def _new_run(self, lane):
        if self.position >= len(self.order):
            self.order = self.rng.permutation(len(self.runs))
            self.position = 0
            self.traversal += 1
        run = self.runs[int(self.order[self.position])]
        self.position += 1
        self.resets += 1
        return {"run": run, "cursor_ms": run.start_ms, "ordinal": 0,
                "seq_id": f"lane{lane}:pass{self.traversal}:{self.position}:{run.recording}:{run.run_id}"}

    def next_batch(self):
        batch = []
        for lane in range(self.batch_size):
            state = self.lanes[lane]
            reset = state is None or state["cursor_ms"] + self.hop_ms > state["run"].stop_ms
            released = None if state is None else state["seq_id"]
            tail = None
            if reset and state is not None and state["cursor_ms"] < state["run"].stop_ms:
                old_run = state["run"]
                old_reader = self.readers[old_run.recording]
                tail = self._packet(old_reader, state["cursor_ms"], old_run.stop_ms)
                tail.update(seq_id=released, packet_ordinal=state["ordinal"], cutoff_ms=old_run.stop_ms)
                self.tail_ms += old_run.stop_ms - state["cursor_ms"]
                self.tail_events += len(tail["events"])
                self.events += len(tail["events"])
            if reset:
                state = self.lanes[lane] = self._new_run(lane)
            run, start = state["run"], state["cursor_ms"]
            stop = start + self.hop_ms
            reader = self.readers[run.recording]
            raw = self._packet(reader, start, stop)
            lo, hi = raw["source_range"]
            events = raw["events"]
            item = {"seq_id": state["seq_id"], "released_seq_id": released if reset else None,
                    "reset": reset, "released_tail": tail, "recording": run.recording, "run_id": run.run_id,
                    "run_start_ms": run.start_ms, "run_stop_ms": run.stop_ms,
                    "start_ms": start, "cutoff_ms": stop, "target_ms": stop - 1,
                    "packet_ordinal": state["ordinal"], "source_range": (lo, hi), "events": events,
                    "initial_state": PUBLIC_INITIAL_STATE.copy(), "betas": reader.betas.copy(),
                    "K_cam": reader.K_cam.copy(), "lnes": last_lnes(events, start, stop)}
            if any(k in item for k in ("target", "targets", "previous_GT")):
                raise AssertionError("supervision entered the input record")
            batch.append(item)
            state["cursor_ms"], state["ordinal"] = stop, state["ordinal"] + 1
            self.outputs += 1
            self.events += len(events)
        return batch

    @staticmethod
    def _packet(reader, start, stop):
        lo, hi = int(reader.offsets[start]), int(reader.offsets[stop])
        native = np.asarray(reader.events[lo:hi], np.int64)
        timestamps = np.repeat(np.arange(start, stop, dtype=np.int64),
                               np.diff(np.asarray(reader.offsets[start:stop + 1], np.int64)))
        events = np.empty((len(native), 4), np.int64)
        events[:, :2], events[:, 2], events[:, 3] = native[:, :2], timestamps, native[:, 2]
        return {"events": events, "source_range": (lo, hi)}

    def targets(self, batch):
        """Explicitly separate supervision read, never used by next_batch."""
        return np.stack([self.readers[b["recording"]].target(b["target_ms"]) for b in batch])

    def state_dict(self):
        return copy.deepcopy({"rng": self.rng.bit_generator.state, "order": self.order, "position": self.position,
                "traversal": self.traversal, "lanes": self.lanes,
                "outputs": self.outputs, "events": self.events, "resets": self.resets,
                "tail_events": self.tail_events, "tail_ms": self.tail_ms,
                "subjects": self.subjects, "root": str(self.root), "hop_ms": self.hop_ms,
                "batch_size": self.batch_size, "runs": [asdict(r) for r in self.runs]})

    def load_state_dict(self, state):
        if (state["subjects"] != self.subjects or state["root"] != str(self.root)
                or state["batch_size"] != self.batch_size or state["hop_ms"] != self.hop_ms
                or state["runs"] != [asdict(r) for r in self.runs]):
            raise ValueError("sampling checkpoint differs from the fixed data schedule")
        self.rng.bit_generator.state = state["rng"]
        for key in ("order", "position", "traversal", "lanes", "outputs", "events", "resets", "tail_events", "tail_ms"):
            setattr(self, key, copy.deepcopy(state[key]))

    def manifest(self):
        return {"subjects": list(self.subjects), "recordings": sorted(self.readers),
                "runs": [asdict(r) for r in self.runs], "boundary": "valid_run only",
                "history": "full continuous exponential tail from segment boundary; no finite warmup truncation",
                "initial_state": PUBLIC_INITIAL_STATE.tolist(), "initialization_GT": False,
                "timestamp": "native integer-ms offset bins; no physical within-ms order",
                "output_lag_ms": 1, "hop_ms": self.hop_ms, "window_ms": 50,
                "short_final_tail": "consume all remaining valid-run events before release, no extra query/feedback/loss",
                "augmentation": "none; common raw sensor dictionary and chronological schedule"}


def last_lnes(events, start_ms, cutoff_ms, height=180, width=240):
    out = np.zeros((height, width, 2), np.float32)
    if len(events):
        ev = np.asarray(events)
        np.maximum.at(out, (ev[:, 1], ev[:, 0], ev[:, 3]),
                      ((ev[:, 2] - start_ms) / float(cutoff_ms - start_ms)).astype(np.float32))
    return out


def enhanced_channels(events, start_ms, cutoff_ms, A, cell_size=2, height=180, width=240):
    """20 channels: last/first/log-count, four log-count bins, six EMA.

Current-window statistics have exactly the same W50 prefix as LNES.  EMA
slots retain the entire continuous prefix with the relation arm's tau.
    """
    out = np.zeros((height, width, 20), np.float32)
    out[..., :2] = last_lnes(events, start_ms, cutoff_ms, height, width)
    if len(events):
        ev = np.asarray(events)
        y, x, p = ev[:, 1], ev[:, 0], ev[:, 3]
        relative = ev[:, 2] - start_ms
        first = np.full((height, width, 2), np.inf, np.float32)
        np.minimum.at(first, (y, x, p), relative.astype(np.float32) / float(cutoff_ms - start_ms))
        out[..., 2:4] = np.where(np.isfinite(first), first, 0.)
        count = np.zeros((height, width, 2), np.float32)
        np.add.at(count, (y, x, p), 1.)
        out[..., 4:6] = np.log1p(count)
        bins = np.minimum(3, (4 * relative // (cutoff_ms - start_ms)).astype(np.intp))
        coarse = np.zeros((height, width, 4, 2), np.float32)
        np.add.at(coarse, (y, x, bins, p), 1.)
        out[..., 6:14] = np.log1p(coarse).reshape(height, width, 8)
    out[..., 14:20] = sensor_A_grid(A, cell_size, height, width)
    return out


def sensor_A_grid(A, cell_size=2, height=180, width=240):
    values = np.asarray(A, np.float32)
    ch, cw = (height + cell_size - 1) // cell_size, (width + cell_size - 1) // cell_size
    if values.shape != (ch * cw, 2, 3):
        raise ValueError("the preregistered sensor adapter requires 2 polarities by 3 tau")
    grid = np.log1p(values).reshape(ch, cw, 6)
    return np.repeat(np.repeat(grid, cell_size, 0), cell_size, 1)[:height, :width]


def representation_from_snapshot(item, snapshot, arm, geometry=None, cell_size=2):
    """Convert only observed prefixes and predicted-condition geometry to inputs."""
    sensor = np.zeros((180, 240, 20), np.float32)
    if arm == "enhanced":
        sensor = enhanced_channels(item["events"], item["start_ms"], item["cutoff_ms"], snapshot.A, cell_size)
    elif arm in ("full", "no_order"):
        sensor[..., :6] = sensor_A_grid(snapshot.A, cell_size)
    result = {"lnes": item["lnes"], "sensor": sensor, "delta_commit_ms": 50.,
              "output_lag_ms": 1., "missing": False, "meta": {
                  "seq_id": item["seq_id"], "cutoff_ms": item["cutoff_ms"],
                  "source_range": list(item["source_range"]), "packet_ordinal": item["packet_ordinal"]}}
    if geometry is not None:
        result.update({key: np.asarray(geometry[key], np.float32) for key in ("f", "a", "lam")})
    return result


class EMATrace:
    """Strong control's exact first-order EMA without unused pair-Z work."""

    def __init__(self, boundary_ms, cell_size=2, tau_ms=(2., 8., 32.)):
        self.cell_size = int(cell_size)
        self.tau_ms = np.asarray(tau_ms, np.float64)
        self.cutoff_ms = int(boundary_ms)
        self.A = np.zeros(((180 // cell_size) * (240 // cell_size), 2, len(tau_ms)), np.float32)

    def ingest(self, events, cutoff_ms):
        if cutoff_ms < self.cutoff_ms:
            raise ValueError("EMA time cannot rewind")
        ev = np.asarray(events)
        if len(ev) and (np.any(ev[:, 2] < self.cutoff_ms) or np.any(ev[:, 2] >= cutoff_ms)):
            raise ValueError("EMA events outside the complete new prefix")
        self.A *= np.exp(-(cutoff_ms - self.cutoff_ms) / self.tau_ms).astype(np.float32)[None, None]
        if len(ev):
            cell = (ev[:, 1] // self.cell_size) * (240 // self.cell_size) + ev[:, 0] // self.cell_size
            weights = np.exp(-(cutoff_ms - ev[:, 2, None]) / self.tau_ms[None]).astype(np.float32)
            np.add.at(self.A, (cell, ev[:, 3]), weights)
        self.cutoff_ms = int(cutoff_ms)
        return self.A.copy()

    def state_dict(self):
        return {"A": self.A.copy(), "cutoff_ms": self.cutoff_ms, "cell_size": self.cell_size,
                "tau_ms": self.tau_ms.copy()}

    @classmethod
    def from_state(cls, state):
        obj = cls(state["cutoff_ms"], state["cell_size"], tuple(state["tau_ms"]))
        obj.A[:] = state["A"]
        return obj
