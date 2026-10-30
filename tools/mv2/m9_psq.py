#!/usr/bin/env python3
"""MV2 M9 / PSQ (P5-E9-R1): the prefix-stable query contract of the S39 stack.

Layer 1 (synthetic, no data): for every write step of the deployed S39 stack -- the W200 event image
(`semkine.eval_track.build_lnes` with the segment-start clip), the Count50 event count, the DT2 adaptive filter with the
ramp1000 evidence gate (`semkine.anchored.AdaptiveFilter`), and the readout (a random-initialised S39 `MNISTModel` in the
closed loop) -- check that changing the events AFTER the cut t_cut leaves every output bit for bit (float32, `torch.equal`)
unchanged. Positive controls (an in-window perturbation; a deliberately look-ahead writer) must be detected, otherwise
the harness is a "tool_defect". Also tested: the segment-start clip reads no event before the segment start, the ms
rebase is origin-invariant, a streaming ring-buffer writer (`RingLNES`, the minimal fix if a step had failed) equals the
batch event image bit for bit and answers past queries prefix-stably, and the S43 SAE-plane writer (read-only import of
`semkine.async_tracker.EventDrivenTracker`) is NOT stable for a past query once later events were ingested (a descriptive
fact about S43, not about S39).

Layer 2 (formal, SELECT only, frozen S39 x 3 seeds): the full closed-loop S39 stack (W200 clip, Count50, ramp1000 filter)
on the same event stream with the packet-boundary phase shifted by 10 / 25 ms (controls: 50 ms = the same grid, 1 ms =
a micro shift); |dRA| bands 0.03 / 0.15 mm per seed and domain (`decide`).

    python tools/mv2/m9_psq.py --smoke
    python tools/mv2/m9_psq.py --out outputs/mv2/m9_psq [--workers 24 --threads 2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mv2 import common as C  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine import events as EV  # noqa: E402
from semkine.anchored import AdaptiveFilter  # noqa: E402

RAMP1000 = REPO / "outputs" / "dt3" / "gate_specs" / "ramp1000.json"
RAMP1000_TAG = "af9f41a0"            # `filter_eval.spec_tag`: 'af' + md5(canonical spec)[:6], S39's evaluation files
H, W = C.H, C.W
STEP = C.STEP
WINDOW = C.WINDOW
SEED_RNG = 20261026
PHASES = (0, 10, 25, 50, 1)           # 0 = the evaluator's grid; 10 / 25 = the gated shifts; 50 = same grid (loop-history control)
GATED_PHASES = (10, 25)
BAND_INVARIANT = 0.03                 # mm, S44 P0b gate on h10 - h50 (#s44-p0b-results-20261020)
BAND_CADENCE = 0.15                   # mm, ~ the observed h10 - h50 (+0.165 / +0.194 / +0.188)
SELECT_SEQS = C.base_seqs(C.SELECT)


# ----------------------------------------------------------------------------------------------- S39 stack
def ramp1000_spec() -> dict:
    spec = AdaptiveFilter.canonical_spec(json.loads(RAMP1000.read_text()))
    tag = hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:6]
    if "af" + tag != RAMP1000_TAG[:8]:
        raise SystemExit(f"ramp1000 spec hash {tag} != the S39 evaluation spec {RAMP1000_TAG}")
    return spec


def build_stack(seed: int, device=torch.device("cpu")):
    """(bare S39 model, AdaptiveFilter(ramp1000) around it, cfg) -- `tools/dt/filter_eval.py` / `tools/trk6/bench_chain.py`."""
    model, cfg = C.load_s39(seed, device)
    net = AdaptiveFilter.from_spec(model, ramp1000_spec()).to(device).eval()
    return model, net, cfg


def count50(offsets, end: int) -> int:
    """The packet's raw event count handed to the gain schedule (`evalx.run_sequence` count_mode='last50')."""
    return int(offsets[end + 1] - offsets[end - STEP + 1])


def clip_window(end: int, floor: int, window: int = WINDOW) -> int:
    """The W200 evidence window clipped to the segment start (`evalx.window_of`, wmode 'fixed')."""
    return int(min(window, end + 1 - floor))


# ----------------------------------------------------------------------------------------------- closed loop
@torch.no_grad()
def closed_loop(net, cfg, seq: str, phase: int = 0, window: int = WINDOW, init_seed: int = 0,
                max_steps: Optional[int] = None, device=torch.device("cpu"), data=None, mano=None,
                event_window_fn=None) -> dict:
    """`evalx.run_sequence` (mode 'model', fixed window, clip_run_start, count_mode last50) with the packet grid shifted by
    `phase` ms: ends = a + 49 + phase + 50 k. phase 0 is the evaluator's loop bit for bit (checked by `parity_vs_evalx`).

    `event_window_fn(events, offsets, end, w, ev_ch) -> (x_image (H, W, 2) float32, n_count50 int)` lets a probe change the
    events of a window (default: the evaluator's `build_lnes` and the raw count). Returns the arrays of
    `evalx.run_sequence` plus the per-step RA (mm) when a MANO layer is given."""
    events, offsets, aux, pos51 = data if data is not None else C.load_seq(seq)
    betas = torch.tensor(aux["betas"], dtype=torch.float32, device=device).view(1, -1)
    K = torch.tensor(aux["camera_K"], dtype=torch.float32, device=device).view(1, 3, 3)
    if hasattr(net, "set_hand_context"):
        net.set_hand_context(betas, K)
    ev_ch = EV.event_channels(cfg)
    rng = np.random.default_rng(int(init_seed))
    preds, gts, ends_all, runs_all = [], [], [], []
    done = False
    for run_id, (a, b) in enumerate(np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)):
        ends = np.arange(a + STEP - 1 + phase, b, STEP, dtype=np.int64)
        if not len(ends):
            continue
        prev = pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)
        prev_t = torch.from_numpy(prev).view(1, -1).to(device)
        net.reset_state(prev_t)
        for end in ends:
            w = clip_window(int(end), int(a), window)
            if event_window_fn is None:
                x_np = ET.build_lnes(events, offsets, int(end), w, ev_ch)
                n_ev = count50(offsets, int(end))
            else:
                x_np, n_ev = event_window_fn(events, offsets, int(end), w, ev_ch)
            x = torch.from_numpy(x_np).unsqueeze(0).to(device)
            pred = net(x, prev_t, n_events=n_ev)
            prev_t = pred
            preds.append(pred.cpu().numpy()[0])
            gts.append(pos51[end])
            ends_all.append(int(end))
            runs_all.append(run_id)
            if max_steps is not None and len(preds) >= max_steps:
                done = True
                break
        if done:
            break
    out = {"pred": np.stack(preds).astype(np.float32), "gt": np.stack(gts).astype(np.float32),
           "end": np.asarray(ends_all), "run": np.asarray(runs_all), "betas": aux["betas"]}
    if mano is not None:
        from tracking import evalx as EX
        m = EX.per_step_metrics(mano, out, device)
        out["ra_mm"] = m["mpjpe_ra_mm"].astype(np.float64)
    return out


def get_mano(cfg, device=torch.device("cpu")):
    from mano_layer import ManoLayer
    return ManoLayer(str(REPO / "assets/mano_right.npz"), add_mean=False).to(device).eval()


def parity_vs_evalx(seed: int, seq: str, threads: int) -> dict:
    """phase 0 of `closed_loop` equals `tools/tracking/evalx.py::run_sequence` (S39 stack) bit for bit (same process,
    same thread count, CPU). Runs the evaluator on the whole sequence."""
    from tracking import evalx as EX
    torch.set_num_threads(threads)
    model, net, cfg = build_stack(seed)
    mine = closed_loop(net, cfg, seq, 0, init_seed=0)
    rng = np.random.default_rng(0)
    ref = EX.run_sequence(net, cfg, C.DATA_ROOT, C.seq_dir(seq), seq, torch.device("cpu"), rng, "model", "fixed", WINDOW,
                          0, 300, clip_run_start=True, count_mode="last50")
    same = {k: bool(np.array_equal(mine[k], ref[k] if k != "gt" else ref["gt"])) for k in ("pred", "gt", "end", "run")}
    return {"seed": seed, "seq": seq, "n_steps": int(len(ref["pred"])), "identical": all(same.values()), "per_key": same,
            "max_abs_pred_diff": float(np.abs(mine["pred"] - ref["pred"]).max()) if len(ref["pred"]) == len(mine["pred"]) else None}


# ----------------------------------------------------------------------------------------------- streaming writers
class RingLNES:
    """The minimal fix: a streaming, prefix-stable writer of the S39 event image (ring buffer + fixed-window rebase).

    `write_ms(ms, ev)` appends the events of millisecond `ms` (non-decreasing `ms`; `ev` (n, 3) uint8 x, y, p) to a ring of
    `capacity_ms` per-ms slots; nothing else is kept. `query(end, window)` reads ONLY the slots `[end - window + 1, end]`,
    rebases their times to the window start (`ms_rel = ms - start`, normalised by `window`) and splats them with the
    evaluator's own `semkine.events.splat_event_image`, so it equals `semkine.eval_track.build_lnes` bit for bit and does
    not depend on anything written after `end` -- also when the query is in the past of the last write (as long as the
    window is still inside the ring). `count(end, span)` is the Count50 of the same slots."""

    def __init__(self, capacity_ms: int = 512, channels=("last",)):
        self.cap = int(capacity_ms)
        self.channels = tuple(channels)
        self.slot_ms = np.full(self.cap, -1, np.int64)
        self.slots: List[Optional[np.ndarray]] = [None] * self.cap
        self.last_ms = -1

    def write_ms(self, ms: int, ev: np.ndarray):
        ms = int(ms)
        if ms < self.last_ms:
            raise ValueError(f"RingLNES.write_ms: ms {ms} before the last written {self.last_ms}")
        for m in range(max(self.last_ms + 1, ms - self.cap + 1), ms):            # skipped ms are empty slots
            self._put(m, np.zeros((0, 3), np.uint8))
        if ms == self.last_ms:                                                   # more events of the same ms
            i = ms % self.cap
            self.slots[i] = np.concatenate([self.slots[i], np.asarray(ev, np.uint8).reshape(-1, 3)])
        else:
            self._put(ms, np.asarray(ev, np.uint8).reshape(-1, 3))
        self.last_ms = ms

    def _put(self, ms: int, ev: np.ndarray):
        i = ms % self.cap
        self.slot_ms[i], self.slots[i] = ms, ev

    def _gather(self, lo: int, hi: int) -> List[np.ndarray]:
        if hi > self.last_ms:
            raise ValueError(f"query up to ms {hi} but only ms <= {self.last_ms} was written")
        if lo <= self.last_ms - self.cap:
            raise ValueError(f"query from ms {lo} is older than the ring ({self.cap} ms behind {self.last_ms})")
        out = []
        for m in range(lo, hi + 1):
            i = m % self.cap
            out.append(self.slots[i] if (m >= 0 and self.slot_ms[i] == m) else np.zeros((0, 3), np.uint8))
        return out

    def query(self, end: int, window: int) -> np.ndarray:
        start = end - window + 1
        evs = self._gather(start, end)
        counts = np.asarray([len(e) for e in evs], np.int64)
        if counts.sum() == 0:
            return np.zeros((H, W, 2 * len(self.channels)), np.float32)
        ev = np.concatenate(evs)
        ms_rel = np.repeat(np.arange(window, dtype=np.float32), counts)
        return EV.splat_event_image(ev[:, 0], ev[:, 1], np.clip(ev[:, 2], 0, 1), ms_rel, window, H, W, self.channels)

    def count(self, end: int, span: int = STEP) -> int:
        return int(sum(len(e) for e in self._gather(end - span + 1, end)))


class SAEPlane:
    """The S43 form of the writer: one 'latest ms' plane per (pixel, polarity). O(1) per event, but the plane keeps no
    history: once events of ms > q were written, a query at q is wrong (`semkine.async_tracker.EventDrivenTracker.lnes`)."""

    NEVER = -(2 ** 40)

    def __init__(self):
        self.plane = np.full((H, W, 2), self.NEVER, np.int64)

    def write_ms(self, ms: int, ev: np.ndarray):
        if len(ev):
            np.maximum.at(self.plane, (ev[:, 1].astype(np.intp), ev[:, 0].astype(np.intp),
                                       np.clip(ev[:, 2], 0, 1).astype(np.intp)), int(ms))

    def query(self, end: int, window: int) -> np.ndarray:
        start = end - window + 1
        val = (self.plane - start).astype(np.float32) / np.float32(window)
        return np.where(self.plane >= start, val, np.float32(0)).astype(np.float32)


# ----------------------------------------------------------------------------------------------- synthetic streams
def synth_stream(rng: np.random.Generator, n_ms: int = 700, mean_per_ms: float = 40.0):
    """A synthetic ms-binned event stream (uint8 x, y, p; int64 per-ms offsets, n_ms + 1) with a hot blob, so many events
    collide on a pixel and the 'latest event wins' rule is exercised."""
    counts = rng.poisson(mean_per_ms, n_ms)
    counts[rng.random(n_ms) < 0.1] = 0                                           # empty ms
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    n = int(offsets[-1])
    x = np.clip(rng.normal(120, 10, n), 0, W - 1).astype(np.uint8)
    y = np.clip(rng.normal(90, 8, n), 0, H - 1).astype(np.uint8)
    p = rng.integers(0, 2, n).astype(np.uint8)
    return np.stack([x, y, p], 1), offsets


def future_variant(events, offsets, t_cut: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    """The same stream up to and including ms `t_cut`; a different random stream (other counts and contents) after it."""
    n_ms = len(offsets) - 1
    keep = int(offsets[t_cut + 1])
    ev_b, off_b = synth_stream(rng, n_ms - (t_cut + 1), mean_per_ms=55.0)
    return (np.concatenate([events[:keep], ev_b]), np.concatenate([offsets[:t_cut + 2], off_b[1:] + keep]).astype(np.int64))


def past_variant(events, offsets, floor: int, rng) -> Tuple[np.ndarray, np.ndarray]:
    """The same stream from ms `floor` on; a different random stream before it (the segment-start clip must not read it)."""
    ev_a, off_a = synth_stream(rng, floor, mean_per_ms=60.0)
    keep0 = int(offsets[floor])
    return (np.concatenate([ev_a, events[keep0:]]),
            np.concatenate([off_a[:-1], offsets[floor:] - keep0 + off_a[-1]]).astype(np.int64))


def inwindow_variant(events, offsets, end: int, rng, n_edit: int = 8):
    """Positive control: move `n_edit` events of the last 50 ms by one pixel (changes the window and the packet content)."""
    ev = events.copy()
    a0, a1 = int(offsets[end - STEP + 1]), int(offsets[end + 1])
    if a1 - a0 < 2:
        return None
    idx = rng.choice(np.arange(a0, a1), size=min(n_edit, a1 - a0), replace=False)
    ev[idx, 0] = (ev[idx, 0].astype(np.int64) + 1) % W
    return ev


def build_lnes_lookahead(events, offsets, end: int, window: int, peek: int = 1, channels=("last",)):
    """Deliberately WRONG writer (negative control of the harness): it reads `peek` ms beyond the cut."""
    start = end - window + 1
    a0, a1 = int(offsets[start]), int(offsets[min(end + 1 + peek, len(offsets) - 1)])
    if a1 <= a0:
        return np.zeros((H, W, 2), np.float32)
    ev = events[a0:a1]
    counts = np.diff(offsets[start:min(end + 2 + peek, len(offsets))]).astype(np.int64)
    ms_rel = np.repeat(np.arange(len(counts), dtype=np.float32), counts)
    return EV.splat_event_image(ev[:, 0], ev[:, 1], np.clip(ev[:, 2], 0, 1), ms_rel, window, H, W, channels)


class _StubTracker(nn.Module):
    """A deterministic tracker whose output depends on the whole event image and on prev (stands in for the CNN when the
    filter alone is under test)."""

    def __init__(self, seed: int = 7):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.register_buffer("w", torch.randn(2, 51, generator=g) * 0.05)

    def forward(self, x, prevpos, betas=None, camera_K=None):
        return prevpos + x.mean(dim=(1, 2)) @ self.w


# ----------------------------------------------------------------------------------------------- layer 1
def _eq(a, b) -> bool:
    return bool(np.array_equal(a, b))


def layer1(n_trials: int = 24, seed: int = SEED_RNG, full_model: bool = True, verbose: bool = False) -> dict:
    """Prefix-stability unit tests of every write step (synthetic streams only). Returns {tests: {...}, steps: {...},
    controls: {...}, fix_needed: bool}. Each test: {passed, n, note}."""
    rng = np.random.default_rng(seed)
    tests: Dict[str, dict] = {}

    def record(name, passed, n, note="", **extra):
        tests[name] = dict(passed=bool(passed), n=int(n), note=note, **extra)
        if verbose:
            print(f"  [{'ok' if passed else 'FAIL'}] {name} (n={n}) {note}")

    ev_a, off_a = synth_stream(rng)
    n_ms = len(off_a) - 1
    ends = [int(e) for e in rng.integers(260, n_ms - 80, n_trials)]
    # --- T1 build_lnes is stable under changes after the cut (windows 50 / 100 / 200 / segment-start clipped)
    ok, n = True, 0
    for end in ends:
        ev_b, off_b = future_variant(ev_a, off_a, end, rng)
        for w in (50, 100, 200, clip_window(end, end - 63)):
            ok &= _eq(ET.build_lnes(ev_a, off_a, end, w), ET.build_lnes(ev_b, off_b, end, w))
            n += 1
    record("T1_build_lnes_future_stable", ok, n, "W in {50, 100, 200, clipped to a segment start 64 ms back}")
    # --- T2 Count50
    ok = all(count50(off_a, e) == count50(future_variant(ev_a, off_a, e, rng)[1], e) for e in ends)
    record("T2_count50_future_stable", ok, len(ends))
    # --- T3 the DT2 adaptive filter with the ramp1000 gate (stub tracker): the whole write+gain path
    spec = ramp1000_spec()
    filt = AdaptiveFilter.from_spec(_StubTracker(), spec).eval()
    ok, n = True, 0
    with torch.no_grad():
        for end in ends:
            ev_b, off_b = future_variant(ev_a, off_a, end, rng)
            prev = torch.from_numpy(rng.standard_normal((1, 51)).astype(np.float32) * 0.1)
            outs = []
            for evx, offx in ((ev_a, off_a), (ev_b, off_b)):
                x = torch.from_numpy(ET.build_lnes(evx, offx, end, clip_window(end, 0))).unsqueeze(0)
                filt.reset_state(prev)
                outs.append(filt(x, prev, n_events=count50(offx, end)))
            ok &= bool(torch.equal(outs[0], outs[1]))
            n += 1
    record("T3_adaptive_filter_ramp1000_future_stable", ok, n, f"spec {RAMP1000_TAG}")
    # --- T4 the segment-start clip reads nothing before the segment start
    ok, n = True, 0
    for end in ends:
        floor = end - int(rng.integers(40, 180))
        ev_b, off_b = past_variant(ev_a, off_a, floor, rng)
        w = clip_window(end, floor)
        ok &= _eq(ET.build_lnes(ev_a, off_a, end, w), ET.build_lnes(ev_b, off_b, end, w))
        n += 1
    record("T4_segment_start_clip_reads_no_pre_segment_event", ok, n, "events before the segment start replaced")
    # --- T5 ms rebase: the event image is invariant to the clock origin
    ok, n = True, 0
    for end in ends:
        shift = int(rng.integers(1, 5000))
        off_s = np.concatenate([np.zeros(shift, np.int64), off_a])
        for w in (50, 200):
            ok &= _eq(ET.build_lnes(ev_a, off_a, end, w), ET.build_lnes(ev_a, off_s, end + shift, w))
            n += 1
    record("T5_ms_rebase_origin_invariant", ok, n)
    # --- T6 ring buffer = batch writer, bit for bit, while streaming (the ring wraps) and for PAST queries
    ring = RingLNES(capacity_ms=256)
    ok_eq = ok_cnt = True
    n = 0
    for ms in range(n_ms):
        ring.write_ms(ms, ev_a[off_a[ms]:off_a[ms + 1]])
        if ms >= 120 and ms % 7 == 0:
            for end, lag_ok in ((ms, True), (ms - 30, ms - 30 >= 120)):
                if not lag_ok:
                    continue
                for w in (50, clip_window(end, 0), clip_window(end, end - 90)):
                    ok_eq &= _eq(ring.query(end, w), ET.build_lnes(ev_a, off_a, end, w))
                    n += 1
                ok_cnt &= ring.count(end) == count50(off_a, end)
    record("T6a_ring_streaming_equals_batch_build_lnes", ok_eq, n, "ring of 256 ms wraps; queries at the head and 30 ms back")
    record("T6b_ring_count50_equals_offsets", ok_cnt, n)
    ring_a, ring_b = RingLNES(capacity_ms=1024), RingLNES(capacity_ms=1024)
    cut = 420
    ev_b, off_b = future_variant(ev_a, off_a, cut, rng)
    for ms in range(n_ms):
        ring_a.write_ms(ms, ev_a[off_a[ms]:off_a[ms + 1]])
        ring_b.write_ms(ms, ev_b[off_b[ms]:off_b[ms + 1]])
    ok_pref, n = True, 0
    for end in range(cut - 150, cut + 1, 17):
        for w in (50, 200, clip_window(end, end - 90)):
            ref = ET.build_lnes(ev_a, off_a, end, w)
            ok_pref &= _eq(ring_a.query(end, w), ref) and _eq(ring_b.query(end, w), ref)   # B's later events must not matter
            n += 1
    record("T6c_ring_past_query_prefix_stable", ok_pref, n, "streams A / B differ after the cut and are fully written")
    # --- T7 S43 SAE-plane writer: equal at the cut, WRONG for a past query once later events were ingested (descriptive)
    sae_ok_at_cut = True
    for end in ends[:8]:
        sae = SAEPlane()
        for ms in range(end + 1):
            sae.write_ms(ms, ev_a[off_a[ms]:off_a[ms + 1]])
        sae_ok_at_cut &= _eq(sae.query(end, 200), ET.build_lnes(ev_a, off_a, end, 200))
    sae = SAEPlane()
    past_wrong = 0
    for ms in range(n_ms):
        sae.write_ms(ms, ev_a[off_a[ms]:off_a[ms + 1]])
    for end in ends[:8]:
        past_wrong += int(not _eq(sae.query(end, 200), ET.build_lnes(ev_a, off_a, end, 200)))
    record("T7a_sae_plane_equals_batch_at_the_cut", sae_ok_at_cut, 8)
    record("T7b_sae_plane_past_query_is_NOT_stable", past_wrong == 8, 8, "expected True: a plane keeps no history",
           descriptive=True, n_wrong=past_wrong)
    try:
        from semkine.async_tracker import EventDrivenTracker
        tr = EventDrivenTracker.__new__(EventDrivenTracker)
        tr.device, tr.h, tr.w, tr.window_ms, tr.floor = torch.device("cpu"), H, W, 200, 0
        tr.sae = torch.full((H * W * 2,), -(2 ** 40), dtype=torch.long)
        tr.ev_n, tr.timing = {}, {"ingest": 0.0, "query": 0.0, "n_query": 0, "n_events": 0}
        end = ends[0]
        for ms in range(end + 1):
            e = ev_a[off_a[ms]:off_a[ms + 1]]
            if len(e):
                tr.ingest(e[:, 0], e[:, 1], e[:, 2], ms)
        s43, _ = tr.lnes(end)
        record("T7c_s43_runtime_sae_equals_batch_at_the_cut", _eq(s43.numpy(), ET.build_lnes(ev_a, off_a, end, 200)), 1,
               "the actual EventDrivenTracker.ingest / lnes, read-only")
    except Exception as err:                                                     # pragma: no cover
        record("T7c_s43_runtime_sae_equals_batch_at_the_cut", False, 1, f"could not run: {type(err).__name__}: {err}")
    # --- T8 full model (random-initialised S39 architecture) closed loop: steps <= cut identical, steps after differ
    if full_model:
        from config import load_config
        from model import MNISTModel
        cfg = load_config(C.REPO / "outputs/semkine/dt_dz_l3_s3407/config_resolved.yaml")
        model = MNISTModel(cfg=cfg).eval()
        net = AdaptiveFilter.from_spec(model, spec).eval()
        net.set_hand_context(torch.zeros(1, 10), torch.tensor([[603.4507, 0, 325.09183], [0, 602.95654, 242.09796],
                                                              [0, 0, 1]]).view(1, 3, 3))
        ok_pre, diverged, n = True, True, 0
        for trial in range(3):
            n_ms2 = 520
            ev0, off0 = synth_stream(rng, n_ms2)
            cut_k = 3
            ends_k = [49 + 50 * k for k in range(9)]
            evb, offb = future_variant(ev0, off0, ends_k[cut_k], rng)
            prev0 = torch.from_numpy(rng.standard_normal((1, 51)).astype(np.float32) * 0.05)
            res = []
            for evx, offx in ((ev0, off0), (evb, offb)):
                p, o = prev0.clone(), []
                net.reset_state(p)
                with torch.no_grad():
                    for end in ends_k:
                        x = torch.from_numpy(ET.build_lnes(evx, offx, end, clip_window(end, 0))).unsqueeze(0)
                        p = net(x, p, n_events=count50(offx, end))
                        o.append(p.clone())
                res.append(o)
            ok_pre &= all(bool(torch.equal(res[0][k], res[1][k])) for k in range(cut_k + 1))
            diverged &= any(not torch.equal(res[0][k], res[1][k]) for k in range(cut_k + 1, len(ends_k)))
            n += cut_k + 1
        record("T8a_full_s39_stack_closed_loop_prefix_stable", ok_pre, n, "random-init S39 architecture + ramp1000 filter")
        record("T8b_closed_loop_does_diverge_after_the_cut", diverged, 3, "control: the harness can see a future change")
        # batch-composition determinism of the CPU forward (information for the batched probes, not a gate)
        x = torch.rand(4, H, W, 2)
        p = torch.zeros(4, 51)
        with torch.no_grad():
            ob = model(x, p)
            o1 = torch.cat([model(x[i:i + 1], p[i:i + 1]) for i in range(4)])
        tests["I1_cpu_forward_batch_composition_bitwise"] = dict(
            passed=bool(torch.equal(ob, o1)), n=4, descriptive=True,
            note="B=4 vs 4 x B=1; False means a probe must keep batch composition identical to compare bit for bit",
            max_abs_diff=float((ob - o1).abs().max()))
    # --- negative controls: the harness must flag a look-ahead writer and an in-window change
    flagged = 0
    for end in ends:
        ev_b, off_b = future_variant(ev_a, off_a, end, rng)
        flagged += int(not _eq(build_lnes_lookahead(ev_a, off_a, end, 200), build_lnes_lookahead(ev_b, off_b, end, 200)))
    record("C1_lookahead_writer_is_flagged", flagged == len(ends), len(ends), "control, must be True", control=True)
    flagged = n_try = 0
    for end in ends:
        evx = inwindow_variant(ev_a, off_a, end, rng)
        if evx is None:
            continue
        n_try += 1
        flagged += int(not _eq(ET.build_lnes(ev_a, off_a, end, 200), ET.build_lnes(evx, off_a, end, 200)))
    record("C2_in_window_change_is_flagged", n_try > 0 and flagged == n_try, n_try, "control, must be True", control=True)
    lk = 0
    for e in ends:
        ev_b, off_b = future_variant(ev_a, off_a, e, rng)
        lk += int(int(off_a[e + 11] - off_a[e - 39]) != int(off_b[e + 11] - off_b[e - 39]))
    record("C3_lookahead_count_is_flagged", lk == len(ends), len(ends),
           "control: a count over [end-39, end+10] is not prefix stable", control=True)
    steps = {
        "build_lnes(W200, clip, ms rebase)": all(tests[k]["passed"] for k in ("T1_build_lnes_future_stable",
                                                 "T4_segment_start_clip_reads_no_pre_segment_event",
                                                 "T5_ms_rebase_origin_invariant")),
        "count50": tests["T2_count50_future_stable"]["passed"],
        "adaptive_filter(ramp1000)": tests["T3_adaptive_filter_ramp1000_future_stable"]["passed"],
    }
    if full_model:
        steps["readout (random-init S39 closed loop)"] = tests["T8a_full_s39_stack_closed_loop_prefix_stable"]["passed"]
    control_names = [k for k, v in tests.items() if v.get("control")] + ["T8b_closed_loop_does_diverge_after_the_cut"] * full_model
    controls_ok = all(tests[k]["passed"] for k in control_names)
    return {"tests": tests, "steps": steps, "controls_ok": controls_ok, "fix_needed": not all(steps.values()),
            "ring_buffer_fix_equals_batch": all(tests[k]["passed"] for k in
                                                ("T6a_ring_streaming_equals_batch_build_lnes", "T6b_ring_count50_equals_offsets",
                                                 "T6c_ring_past_query_prefix_stable"))}


# ----------------------------------------------------------------------------------------------- layer 2
_WORKER: Dict[str, object] = {}


def _worker_init(threads: int):
    torch.set_num_threads(threads)
    _WORKER["stacks"] = {}


def _worker_task(args):
    seed, seq, phase, max_steps = args
    stacks = _WORKER["stacks"]
    if seed not in stacks:
        model, net, cfg = build_stack(seed)
        stacks[seed] = (net, cfg, get_mano(cfg))
    net, cfg, mano = stacks[seed]
    t0 = time.perf_counter()
    init_seed = 1000 + SELECT_SEQS.index(seq) if seq in SELECT_SEQS else 0
    r = closed_loop(net, cfg, seq, phase, init_seed=init_seed, max_steps=max_steps, mano=mano)
    return {"seed": seed, "seq": seq, "phase": phase, "n": int(len(r["ra_mm"])), "ra_mm": r["ra_mm"],
            "end": r["end"], "secs": time.perf_counter() - t0}


def run_layer2(seeds=C.SEEDS, seqs=SELECT_SEQS, phases=PHASES, workers: int = 1, threads: int = 2,
               max_steps: Optional[int] = None) -> dict:
    tasks = [(s, q, ph, max_steps) for s in seeds for q in seqs for ph in phases]
    t0 = time.time()
    if workers <= 1:
        _worker_init(threads)
        res = [_worker_task(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers, initializer=_worker_init, initargs=(threads,)) as pool:
            res = pool.map(_worker_task, tasks, chunksize=1)
    ra: Dict[str, Dict[str, Dict[str, dict]]] = {}
    for r in res:
        dom = "local" if r["seq"].endswith("_local") else "global"
        d = ra.setdefault(str(r["seed"]), {}).setdefault(dom, {}).setdefault(str(r["phase"]), {"sum": 0.0, "n": 0,
                                                                                                "per_seq": {}})
        d["sum"] += float(r["ra_mm"].sum())
        d["n"] += r["n"]
        d["per_seq"][r["seq"]] = {"mean_ra_mm": float(r["ra_mm"].mean()), "n": r["n"]}
    table = {s: {dom: {ph: {"ra_mm": v["sum"] / v["n"], "n": v["n"], "per_seq": v["per_seq"]} for ph, v in phs.items()}
                 for dom, phs in doms.items()} for s, doms in ra.items()}
    return {"table": table, "seeds": list(map(int, seeds)), "seqs": list(seqs), "phases": list(phases), "threads": threads,
            "secs": time.time() - t0, "max_steps": max_steps}


# ----------------------------------------------------------------------------------------------- decision
def _band(d: float) -> str:
    a = abs(d)
    return "invariant" if a <= BAND_INVARIANT else ("descriptive" if a <= BAND_CADENCE else "cadence")


def decide(result: dict) -> dict:
    """The pre-registered gate of PSQ (tools/mv2/prereg_m9.md section 1).

    tool validity (else `tool_defect`): layer-1 controls detected; layer-2 phase 0 equals `evalx.run_sequence` bit for bit.
    layer 1: any write step not future-stable => `batch_evaluation_identity` (fix: `RingLNES`, must equal the batch writer).
    layer 2 (per domain and gated phase, three seeds): all |dRA| <= 0.03 => invariant; all three > 0.15 with one sign =>
    cadence_dependent; else descriptive. Contract: layer 1 all pass and every (domain, phase) invariant => contract_holds;
    any cadence_dependent => `kill_frontier_contract` (report only a replay benchmark); else `descriptive`."""
    out: Dict[str, object] = {"verdict": None, "notes": []}
    l1 = result.get("layer1")
    if l1 is None:
        return {"verdict": "incomplete", "notes": ["layer1 missing"]}
    if not l1.get("controls_ok", False):
        return {"verdict": "tool_defect", "notes": ["layer-1 positive / negative controls not all detected"]}
    out["layer1"] = "pass" if not l1["fix_needed"] else "batch_evaluation_identity"
    if l1["fix_needed"]:
        out["notes"].append("a write step is not future-stable: S39 is registered as a batch evaluation; fix = RingLNES "
                            f"(equals batch writer: {l1.get('ring_buffer_fix_equals_batch')})")
    l2 = result.get("layer2")
    if l2 is None:
        out["verdict"] = "layer1_only"
        return out
    par = result.get("parity")
    if par is None or not par.get("identical", False):
        return {"verdict": "tool_defect", "notes": ["layer-2 phase 0 differs from evalx.run_sequence"], "layer1": out["layer1"]}
    table, cells, killed = l2["table"], {}, False
    seeds = sorted(table)
    for dom in ("local", "global"):
        for ph in GATED_PHASES + (50, 1):
            deltas = [table[s][dom][str(ph)]["ra_mm"] - table[s][dom]["0"]["ra_mm"] for s in seeds]
            bands = [_band(d) for d in deltas]
            same_sign = all(d > 0 for d in deltas) or all(d < 0 for d in deltas)
            if all(b == "invariant" for b in bands):
                cls = "invariant"
            elif all(b == "cadence" for b in bands) and same_sign:
                cls = "cadence_dependent"
            else:
                cls = "descriptive"
            cells[f"{dom}|{ph}"] = {"delta_mm": deltas, "bands": bands, "same_sign": same_sign, "class": cls}
            if ph in GATED_PHASES and cls == "cadence_dependent":
                killed = True
    out["cells"] = cells
    for dom in ("local", "global"):
        ctl = cells[f"{dom}|50"]
        if ctl["class"] == "cadence_dependent":
            out["notes"].append(f"{dom}: the same-grid control (50 ms) is itself > 0.15 mm with one sign: loop-history noise "
                                "floor not below the gate; the cadence reading is not separable")
    gated = [cells[f"{d}|{p}"]["class"] for d in ("local", "global") for p in GATED_PHASES]
    if killed:
        out["verdict"] = "kill_frontier_contract"
    elif out["layer1"] == "pass" and all(g == "invariant" for g in gated):
        out["verdict"] = "contract_holds"
    else:
        out["verdict"] = "descriptive"
    return out


# ----------------------------------------------------------------------------------------------- main
def smoke(args) -> int:
    torch.set_num_threads(max(1, args.threads))
    t0 = time.time()
    print("[m9_psq smoke] layer 1 (synthetic)")
    l1 = layer1(n_trials=args.trials, verbose=True)
    print(f"  layer1: steps={ {k: bool(v) for k, v in l1['steps'].items()} } controls_ok={l1['controls_ok']} "
          f"fix_needed={l1['fix_needed']} ring_fix_ok={l1['ring_buffer_fix_equals_batch']}  ({time.time() - t0:.1f} s)")
    ok = l1["controls_ok"] and l1["ring_buffer_fix_equals_batch"]
    # layer-2 code path on a tiny slice of a FIT7 sequence (never SELECT / zgz in smoke): shapes and timings only
    seq = "lyh_local"
    t1 = time.time()
    model, net, cfg = build_stack(3407)
    mano = get_mano(cfg)
    outs = {}
    for ph in (0, 10):
        outs[ph] = closed_loop(net, cfg, seq, ph, init_seed=0, max_steps=args.max_steps, mano=mano)
        print(f"  closed_loop phase {ph}: pred {outs[ph]['pred'].shape} ra_mm {outs[ph]['ra_mm'].shape} "
              f"finite={bool(np.isfinite(outs[ph]['ra_mm']).all())}")
        ok &= bool(np.isfinite(outs[ph]['ra_mm']).all())
    ends0, ends10 = outs[0]["end"], outs[10]["end"]
    grid_ok = bool(np.all((ends10 - ends0[:len(ends10)]) == 10))
    print(f"  phase-10 grid is the phase-0 grid + 10 ms: {grid_ok}  ({time.time() - t1:.1f} s)")
    ok &= grid_ok
    if not args.no_parity:
        t2 = time.time()
        par = parity_vs_evalx(3407, seq, args.threads)
        print(f"  parity phase 0 vs evalx.run_sequence on {seq}: identical={par['identical']} n_steps={par['n_steps']} "
              f"({time.time() - t2:.1f} s)")
        ok &= par["identical"]
    # decide() on a synthetic result (no effect reading): plumbing and the three classes
    fake = {"layer1": l1, "parity": {"identical": True}, "layer2": {"table": {
        str(s): {d: {str(p): {"ra_mm": 8.0 + (0.0 if p == 0 else 0.0) + 0.2 * (p in (10, 25))} for p in PHASES}
                 for d in ("local", "global")} for s in C.SEEDS}}}
    dec = decide(fake)
    print(f"  decide() plumbing on a synthetic table: verdict={dec['verdict']}")
    ok &= dec["verdict"] == "kill_frontier_contract"
    flat = {"layer1": l1, "parity": {"identical": True}, "layer2": {"table": {
        str(s): {d: {str(p): {"ra_mm": 8.0} for p in PHASES} for d in ("local", "global")} for s in C.SEEDS}}}
    ok &= decide(flat)["verdict"] == ("contract_holds" if not l1["fix_needed"] else "descriptive")
    out_dir = REPO / "outputs" / "mv2" / "smoke" / "m9_psq"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(json.dumps({"layer1_tests": {k: v["passed"] for k, v in l1["tests"].items()},
                                                    "ok": bool(ok)}, indent=1))
    print(f"[m9_psq smoke] {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--trials", type=int, default=24)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--no-parity", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        raise SystemExit(smoke(a))
    if not a.out:
        raise SystemExit("--out outputs/mv2/m9_psq (formal) or --smoke")
    C.verify_protocol()
    out_dir = Path(a.out)
    torch.set_num_threads(a.threads)
    result = {"layer1": layer1(n_trials=max(a.trials, 48))}
    result["parity"] = parity_vs_evalx(C.SEEDS[0], SELECT_SEQS[0], a.threads)
    result["layer2"] = run_layer2(workers=a.workers, threads=a.threads)
    result["decision"] = decide(result)
    C.save_json(out_dir / "result.json", result)
    C.receipt(out_dir, {"probe": "m9_psq", "verdict": result["decision"]["verdict"]},
              scripts=["tools/mv2/m9_psq.py", "tools/mv2/common.py"])
    print(json.dumps(result["decision"], indent=1, default=C._np))


if __name__ == "__main__":
    main()
