#!/usr/bin/env python3
r"""S15 information-driven adaptive packetization.

The rest of the pipeline inherits a 50 ms window from the LNES baseline. That window is a
property of the dense frontend, not of the measurement: a burst of contour events can already
constrain a finger in a millisecond, and a quiet interval can last much longer than 50 ms
without saying anything. This module emits a packet when a cheap information proxy crosses a
threshold, or when a maximum wait is reached, and refuses to emit a packet whose proxy is
still below a floor -- those events stay in the buffer.

The proxy is *not* the full S8 Fisher matrix. Building `Lambda` every event would make the
packetizer more expensive than the estimator it is feeding. Two proxies are offered, both
monotone in the evidence and both computable from the previous state's KSSF:

* `contour`  count of events whose pixel has a defined contour normal (they are the only
             events that enter the residual)
* `support`  number of distinct kinematic groups those events name, via the top LBS index

A packet that would have been empty under the 50 ms rule is still emitted at `max_dt` so the
filter can propagate, which is the principled form of `ZERO_EVENT_GATE`.

**Buffering is caller-owned and conservative.** The stream arrives in chunks the caller chooses,
and the chunking must not change the output -- otherwise the sparse arm scores on a different
number of events than the dense control it is compared against, and no equivalence claim survives.
Two rules make the split irrelevant:

* every input event leaves through exactly one emitted packet or stays in the buffer, once, in
  input order;
* a boundary at time `tau` is only finalised once an event with a stamp `> tau` has been seen, the
  watermark rule. Until then the last microsecond is still open and could receive more events, so
  closing it early is precisely what would make `push(A); push(B)` differ from
  `push(concat(A, B))`. `flush()` closes the stream at the end of a sequence.

Intervals are half-open, `[t_start_us, t_end_us)`, and contiguous: packet `k`'s `t_end_us` is
packet `k+1`'s `t_start_us`. An information-triggered packet ends at `t_trigger + 1`, so all events
sharing the trigger's microsecond travel together; a deadline packet ends exactly at
`t_start + max_dt_us` rather than at the first event past it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch

#: emitted event columns: x, y, polarity, absolute microsecond stamp
PK_X, PK_Y, PK_P, PK_T = range(4)


@dataclass
class Packet:
    events: np.ndarray                 # (N, 4) float32: x, y, p, t_us
    t_start_us: int
    t_end_us: int
    n_contour: int
    n_groups: int

    @property
    def dt_s(self) -> float:
        return (self.t_end_us - self.t_start_us) * 1e-6


@dataclass
class AdaptivePacketizer:
    min_contour: int = 80
    max_dt_us: int = 50_000
    min_dt_us: int = 1_000
    min_groups: int = 1
    proxy: str = "contour"
    _ev: Optional[np.ndarray] = field(default=None, repr=False)
    _t: Optional[np.ndarray] = field(default=None, repr=False)
    _mask: Optional[np.ndarray] = field(default=None, repr=False)
    _grp: Optional[np.ndarray] = field(default=None, repr=False)
    _t0: Optional[int] = None
    _last_t: Optional[int] = None
    n_emitted: int = 0
    #: events left in the buffer after the most recent push, i.e. current occupancy
    n_held: int = 0

    def reset(self) -> None:
        self._ev = self._t = self._mask = self._grp = None
        self._t0 = None
        self._last_t = None
        self.n_held = 0

    def buffered_events(self) -> np.ndarray:
        """The events the packetizer still holds, `(N, 4)` in input order."""
        if self._t is None or not len(self._t):
            return np.zeros((0, 4), np.float32)
        return np.concatenate([self._ev, self._t.astype(np.float32)[:, None]], 1)

    def should_emit(self, dt_us: int, n_contour: int, n_groups: int) -> bool:
        if dt_us < self.min_dt_us:
            return False
        if dt_us >= self.max_dt_us:
            return True
        if self.proxy == "support":
            return n_groups >= self.min_groups and n_contour >= self.min_contour
        return n_contour >= self.min_contour

    def _append(self, events_xypt, t_us, contour_mask, group_ids) -> None:
        t = np.asarray(t_us).astype(np.int64)
        ev = np.asarray(events_xypt, np.float32)[:, :3]
        if len(t) > 1 and bool((np.diff(t) < 0).any()):
            raise ValueError("packetizer input timestamps are not non-decreasing")
        if self._last_t is not None and int(t[0]) < self._last_t:
            raise ValueError(f"packetizer input goes back in time: {int(t[0])} < {self._last_t}")
        mask = (np.ones(len(t), bool) if contour_mask is None
                else np.asarray(contour_mask).astype(bool))
        grp = None if group_ids is None else np.asarray(group_ids).astype(np.int64)
        if self._t is None:
            self._ev, self._t, self._mask, self._grp = ev, t, mask, grp
        else:
            if (self._grp is None) != (grp is None):
                raise ValueError("group_ids must be supplied consistently across pushes")
            self._ev = np.concatenate([self._ev, ev])
            self._t = np.concatenate([self._t, t])
            self._mask = np.concatenate([self._mask, mask])
            self._grp = None if grp is None else np.concatenate([self._grp, grp])
        self._last_t = int(t[-1])

    def _cut(self, n: int, t_end: int) -> Packet:
        """Move the first `n` buffered events out as a packet covering `[_t0, t_end)`."""
        sel = slice(0, n)
        pk = Packet(
            events=np.concatenate(
                [self._ev[sel], self._t[sel].astype(np.float32)[:, None]], 1),
            t_start_us=int(self._t0), t_end_us=int(t_end),
            n_contour=int(self._mask[sel].sum()),
            n_groups=0 if self._grp is None else int(len(np.unique(self._grp[sel][self._mask[sel]]))),
        )
        self._ev, self._t, self._mask = self._ev[n:], self._t[n:], self._mask[n:]
        self._grp = None if self._grp is None else self._grp[n:]
        self._t0 = int(t_end)
        self.n_emitted += 1
        return pk

    def _trigger_index(self, safe: int) -> int:
        """First buffered event in `[0, safe)` that satisfies `should_emit`, or -1.

        Vectorised so a 29 k-event window costs one pass rather than the quadratic rescan a
        per-event `sum`/`unique` would need; the condition is the same one `should_emit` states.
        """
        if safe <= 0:
            return -1
        dt = self._t[:safe] - self._t0
        ok = dt >= self.min_dt_us
        fire = dt >= self.max_dt_us
        n_c = np.cumsum(self._mask[:safe])
        info = n_c >= self.min_contour
        if self.proxy == "support":
            # A group counts from the first contour event that names it, so the running distinct
            # count is a cumulative sum over first occurrences.
            first = np.zeros(safe, bool)
            idx = np.flatnonzero(self._mask[:safe])
            if len(idx) and self._grp is not None:
                _, pos = np.unique(self._grp[idx], return_index=True)
                first[idx[pos]] = True
            info &= np.cumsum(first) >= self.min_groups
        hit = np.flatnonzero(ok & (fire | info))
        return int(hit[0]) if len(hit) else -1

    def _drain(self, watermark: int) -> List[Packet]:
        """Emit every packet whose boundary the watermark has already decided.

        A boundary at `tau` is decided once an event with a stamp `>= tau` has been seen: only then
        can no future event still land before `tau`. `flush` passes `last_t + 1`, which decides
        every boundary.
        """
        out: List[Packet] = []
        while len(self._t):
            if self._t0 is None:
                self._t0 = int(self._t[0])
            deadline = self._t0 + self.max_dt_us
            if watermark >= deadline:
                n = int(np.searchsorted(self._t, deadline, side="left"))
                out.append(self._cut(n, deadline))
                continue
            safe = int(np.searchsorted(self._t, watermark, side="left"))
            hit = self._trigger_index(safe)
            if hit < 0:
                break
            t_end = int(self._t[hit]) + 1
            out.append(self._cut(int(np.searchsorted(self._t, t_end, side="left")), t_end))
        self.n_held = 0 if self._t is None else int(len(self._t))
        return out

    def push(self, events_xypt: np.ndarray, t_us: np.ndarray,
             contour_mask: Optional[np.ndarray] = None,
             group_ids: Optional[np.ndarray] = None) -> List[Packet]:
        """Ingest a stream slice. `events_xypt` is `(N, >=3)` [x, y, p, ...]; `t_us` is absolute."""
        if len(t_us) == 0:
            return []
        self._append(events_xypt, t_us, contour_mask, group_ids)
        return self._drain(self._last_t)

    def flush(self) -> List[Packet]:
        """End of stream: close every open boundary and empty the buffer."""
        if self._t is None or not len(self._t):
            self.n_held = 0
            return []
        out = self._drain(int(self._t[-1]) + 1)
        if len(self._t):
            out.append(self._cut(len(self._t), int(self._t[-1]) + 1))
        self.n_held = 0
        return out


def contour_mask_from_field(fields, xy: torch.Tensor, batch_index: torch.Tensor,
                            margin: float = 2.0) -> torch.Tensor:
    """Events that land in the contour band of a KSSF raster, i.e. the ones that constrain pose."""
    q = fields.query(xy, batch_index)
    return (q["visibility"] > 0.5) & (q["sdf"].abs() <= margin)
