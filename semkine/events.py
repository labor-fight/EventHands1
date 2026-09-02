#!/usr/bin/env python3
"""S1 raw-event data contract: ragged packets, no `N_max` padding.

The legacy path hands the network a dense `(180, 240, 2)` surface, which throws away 70-88% of
the events (`ARCHITECTURE_AUDIT.md` §7) and all sub-millisecond ordering. This module defines
the alternative: a batch is a *concatenation* of variable-length event lists plus a `ptr` index,
the layout sparse point/graph backends use, so a batch costs `sum(N_b)` rather than
`B * max(N_b)`. With the observed per-window spread (1.5k to 29k events on the same camera),
`N_max` padding would waste roughly an order of magnitude of memory on the sparse windows.

Event columns, fixed for every downstream stage:

    0  batch_index  which packet the event belongs to (redundant with ptr, kept for scatter ops)
    1  x            event pixel column, already in the 240x180 event grid
    2  y            event pixel row
    3  t_rel        seconds since the packet's t_start, from the recovered microsecond stamps
    4  polarity     the *stored* polarity, 0/1, converted exactly once (see `POLARITY_NOTE`)

Timestamps come from `<seq>_tsub.npy` (`tools/extract_subms.py`), so `t_rel` has microsecond
resolution rather than the millisecond bins the LNES path is limited to. Events keep their
storage order, which is the AEDAT4 packet order; within one microsecond that order is the
tie-break and is stable across runs and workers because it is a property of the file.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

#: The stored polarity is used directly as the LNES channel index in `_build_lnes`, so "stored"
#: and "model" polarity are the same thing and no conversion happens on the raw path either.
#: The two training augmentations (global flip, per-pixel swap) are the only transformations,
#: and they are applied once, after packing. Nothing downstream may flip polarity again.
POLARITY_NOTE = "stored polarity == model polarity == LNES channel index; convert exactly once"

EV_BATCH, EV_X, EV_Y, EV_T, EV_P = range(5)
EV_COLS = 5

#: The event-image faces, each contributing one plane per polarity. `last` alone is LNES.
EVENT_CHANNELS = ("last", "count", "first")
#: Event multiplicity is heavy-tailed, so the count plane is log-compressed and saturates here.
#: At the 50 ms operating point the mean occupied slot holds ~10 events, which lands mid-range.
COUNT_REF = 32.0


def event_channels(cfg: dict) -> tuple:
    """`DATA.EVENT_CHANNELS`, validated. Unset means LNES, so every existing config is unchanged."""
    ch = tuple((cfg.get("DATA", {}) or {}).get("EVENT_CHANNELS", ("last",)))
    bad = [c for c in ch if c not in EVENT_CHANNELS]
    if bad or not ch or ch[0] != "last" or len(set(ch)) != len(ch):
        raise ValueError(
            f"DATA.EVENT_CHANNELS must be a duplicate-free subset of {list(EVENT_CHANNELS)} "
            f"beginning with 'last', got {list(ch)}")
    return ch


def splat_event_image(xs, ys, ps, ms_rel, window: int, height: int, width: int,
                      channels=("last",)) -> np.ndarray:
    """The dense event image, `(H, W, 2 * len(channels))`, planes ordered as `channels`.

    `last` is LNES exactly as it has always been: one normalised timestamp per (pixel, polarity),
    later events overwriting earlier ones. Its cost is measured, not assumed --
    `outputs/semkine/lnes_capacity.json` puts 23474 events into 2367 occupied slots at the 50 ms
    operating point, so **73.8% of the events are overwritten and discarded**, and the surface it
    produces is 97.3% empty. Every event that survives contributes exactly one number.

    The other two faces are the cheapest way to stop discarding, and they are chosen so that
    together the three recover what a single overwrite destroys:

    * `count`   how many events hit the slot, log-compressed. This is the multiplicity that
                overwriting deletes outright.
    * `first`   the *earliest* normalised timestamp in the slot. With `last`, this brackets the
                interval the slot was active in; the difference is the dwell time of a moving edge,
                which is a velocity cue that no single timestamp can carry.

    `first` requires the events to arrive in time order, which is the module's standing contract
    (storage order is AEDAT4 packet order, and the one place that perturbs it, hot-pixel injection,
    re-sorts). It is then a reversed splat: last-write-wins over a reversed stream leaves the
    earliest event standing. The reversal has to be materialised. A negative-stride *view* is
    iterated in memory order, so `plane[y[::-1], x[::-1], p[::-1]] = t[::-1]` silently returns
    `last` again -- it did, until `test_the_new_planes_carry_what_overwriting_deleted` caught it.
    `test_first_matches_the_unique_reference` pins the fast form against
    `np.unique(..., return_index=True)`, which is 32x slower but has a defined first-occurrence
    index; at 23k events per window the difference is 3.4 ms against 0.11 ms per sample, which is
    the whole data-loading budget at batch 1024.

    All three are functions of the events alone. That is the property that matters here: the
    repository's own architecture law (`docs/ASYNC_SPARSE_SOTA_MASTER_VERDICT_20260826.md` 3.5)
    is that evidence encoding must be state-independent, because `dPhi/dx != 0` is what made the
    pose-conditioned frontend's loop gain uncontrollable. Widening a state-independent encoder adds
    capacity in the one place that cannot feed the loop.
    """
    out = np.zeros((height, width, 2 * len(channels)), np.float32)
    if not len(xs):
        return out
    yi, xi, pi = ys.astype(np.intp), xs.astype(np.intp), ps.astype(np.intp)
    tval = ms_rel.astype(np.float32) / float(window)
    idx = None
    for k, name in enumerate(channels):
        face = np.zeros((height, width, 2), np.float32)
        if name == "last":
            face[yi, xi, pi] = tval
        else:
            if idx is None:
                idx = (yi * width + xi) * 2 + pi
            if name == "first":
                flat = face.reshape(-1)
                flat[np.ascontiguousarray(idx[::-1])] = np.ascontiguousarray(tval[::-1])
            elif name == "count":
                n = np.bincount(idx, minlength=height * width * 2).astype(np.float32)
                face = (np.log1p(n) / np.log1p(COUNT_REF)).clip(max=1.0
                                                                ).reshape(height, width, 2)
        out[:, :, 2 * k:2 * k + 2] = face
    return out


@dataclass
class EventPacket:
    """One sample's worth of raw events plus the state the tracker needs around it."""

    events: np.ndarray                 # (N, 4) float32: x, y, t_rel, polarity
    sequence_id: int
    t_start_us: int
    t_end_us: int
    is_sequence_start: bool
    is_sequence_end: bool
    target: np.ndarray                 # (51,) current state
    prev_state: np.ndarray             # (51,) previous state (possibly noised)
    betas: np.ndarray                  # (10,)
    camera_K: np.ndarray               # (3, 3)
    lnes: Optional[np.ndarray] = None  # (H, W, 2) when the legacy face is also requested
    meta: Dict = field(default_factory=dict)

    @property
    def n_events(self) -> int:
        return int(self.events.shape[0])

    @property
    def delta_t_s(self) -> float:
        return (self.t_end_us - self.t_start_us) * 1e-6


@dataclass
class EventPacketBatch:
    """Ragged batch. `events[:, 0]` is the packet index; `ptr[b]:ptr[b+1]` slices packet `b`."""

    events: torch.Tensor          # (N_total, 5) float32
    ptr: torch.Tensor             # (B+1,) int64
    sequence_id: torch.Tensor     # (B,) int64
    t_start_us: torch.Tensor      # (B,) int64
    t_end_us: torch.Tensor        # (B,) int64
    delta_t_s: torch.Tensor       # (B,) float32
    is_sequence_start: torch.Tensor   # (B,) bool
    is_sequence_end: torch.Tensor     # (B,) bool
    target: torch.Tensor          # (B, 51)
    prev_state: torch.Tensor      # (B, 51)
    betas: torch.Tensor           # (B, 10)
    camera_K: torch.Tensor        # (B, 3, 3)
    lnes: Optional[torch.Tensor] = None   # (B, H, W, 2)

    @property
    def batch_size(self) -> int:
        return int(self.ptr.numel() - 1)

    @property
    def counts(self) -> torch.Tensor:
        return self.ptr[1:] - self.ptr[:-1]

    def to(self, device) -> "EventPacketBatch":
        def mv(t):
            return t.to(device, non_blocking=True) if torch.is_tensor(t) else t
        return EventPacketBatch(**{k: mv(v) for k, v in self.__dict__.items()})

    def packet(self, b: int) -> torch.Tensor:
        """Events of packet `b`, columns 1..4 (batch index dropped)."""
        return self.events[self.ptr[b] : self.ptr[b + 1], 1:]

    def validate(self) -> None:
        """Contract checks that must hold for every batch, cheap enough to always run."""
        B = self.batch_size
        assert self.events.ndim == 2 and self.events.shape[1] == EV_COLS, self.events.shape
        assert int(self.ptr[0]) == 0 and int(self.ptr[-1]) == self.events.shape[0]
        assert bool((self.counts >= 0).all()), "negative packet length"
        for name in ("sequence_id", "t_start_us", "t_end_us", "delta_t_s",
                     "is_sequence_start", "is_sequence_end"):
            t = getattr(self, name)
            assert t.shape[0] == B, f"{name}: {t.shape} vs B={B}"
        assert self.target.shape[0] == B and self.prev_state.shape[0] == B
        if self.events.numel():
            bi = self.events[:, EV_BATCH]
            # batch_index must agree with ptr, else scatter-based pooling silently mixes samples
            expect = torch.repeat_interleave(
                torch.arange(B, device=bi.device, dtype=bi.dtype), self.counts
            )
            assert bool((bi == expect).all()), "events[:,0] disagrees with ptr"
            assert bool((self.events[:, EV_P] >= 0).all() and (self.events[:, EV_P] <= 1).all())


def _empty(n_cols: int = EV_COLS) -> np.ndarray:
    return np.zeros((0, n_cols), dtype=np.float32)


def collate_packets(items: Sequence[EventPacket]) -> EventPacketBatch:
    """Ragged collate. Never pads to `N_max`; never merges state across sequences.

    A zero-event packet is legal and keeps its slot: `ptr[b] == ptr[b+1]`, and `delta_t_s` is
    still the real interval, because "no events for 50 ms" is information the filter in S12
    consumes (covariance grows) rather than a sample to drop.
    """
    B = len(items)
    counts = np.array([it.n_events for it in items], dtype=np.int64)
    ptr = np.zeros(B + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    total = int(ptr[-1])

    ev = np.zeros((total, EV_COLS), dtype=np.float32)
    if total:
        idx = np.repeat(np.arange(B, dtype=np.float32), counts)
        ev[:, EV_BATCH] = idx
        pieces = [it.events for it in items if it.n_events]
        ev[:, 1:] = np.concatenate(pieces, axis=0)

    lnes = None
    if items[0].lnes is not None:
        lnes = torch.from_numpy(np.stack([it.lnes for it in items]))

    return EventPacketBatch(
        events=torch.from_numpy(ev),
        ptr=torch.from_numpy(ptr),
        sequence_id=torch.tensor([it.sequence_id for it in items], dtype=torch.int64),
        t_start_us=torch.tensor([it.t_start_us for it in items], dtype=torch.int64),
        t_end_us=torch.tensor([it.t_end_us for it in items], dtype=torch.int64),
        delta_t_s=torch.tensor([it.delta_t_s for it in items], dtype=torch.float32),
        is_sequence_start=torch.tensor([it.is_sequence_start for it in items], dtype=torch.bool),
        is_sequence_end=torch.tensor([it.is_sequence_end for it in items], dtype=torch.bool),
        target=torch.from_numpy(np.stack([it.target for it in items])),
        prev_state=torch.from_numpy(np.stack([it.prev_state for it in items])),
        betas=torch.from_numpy(np.stack([it.betas for it in items])),
        camera_K=torch.from_numpy(np.stack([it.camera_K for it in items])),
        lnes=lnes,
    )


def unpack(batch: EventPacketBatch, b: int) -> np.ndarray:
    """`(N_b, 4)` numpy view of packet `b` for the round-trip checks."""
    return batch.packet(b).detach().cpu().numpy()


def lnes_from_packet(events: np.ndarray, delta_t_s: float, window_ms: int,
                     height: int, width: int) -> np.ndarray:
    """Rebuild the legacy LNES from raw events, for the S1 parity check.

    The legacy surface quantises time to the millisecond bin index and writes
    `(ms_index - window_start) / window`. Recovering that from `t_rel` seconds means going back
    through the same floor, so the comparison tests the packing rather than the quantisation.
    """
    img = np.zeros((height, width, 2), np.float32)
    if len(events) == 0:
        return img
    ms = np.floor(events[:, 2] * 1000.0 + 1e-9).astype(np.int64)
    ms = np.clip(ms, 0, window_ms - 1)
    tval = (ms / float(window_ms)).astype(np.float32)
    xs = events[:, 0].astype(np.intp)
    ys = events[:, 1].astype(np.intp)
    ps = np.clip(events[:, 3].astype(np.intp), 0, 1)
    img[ys, xs, ps] = tval
    return img


def concat_check(batch: EventPacketBatch) -> Dict[str, object]:
    """Per-packet monotonicity and bounds, used by the S1 hard gates."""
    B = batch.batch_size
    nondec = 0
    oob = 0
    for b in range(B):
        e = batch.packet(b)
        if e.shape[0] > 1:
            nondec += int((e[1:, 2] - e[:-1, 2] < -1e-9).sum())
        if e.shape[0]:
            oob += int(((e[:, 0] < 0) | (e[:, 1] < 0)).sum())
    return {
        "n_packets": B,
        "n_events": int(batch.events.shape[0]),
        "n_nonmonotonic": nondec,
        "n_out_of_bounds": oob,
        "n_empty_packets": int((batch.counts == 0).sum()),
    }
