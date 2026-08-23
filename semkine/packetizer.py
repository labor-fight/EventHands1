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
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch


@dataclass
class Packet:
    events: np.ndarray
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
    _buf: List[np.ndarray] = field(default_factory=list, repr=False)
    _t0: Optional[int] = None
    n_emitted: int = 0
    n_held: int = 0

    def reset(self) -> None:
        self._buf.clear()
        self._t0 = None

    def _stats(self, contour_mask: np.ndarray, group_ids: Optional[np.ndarray]) -> tuple:
        n_c = int(contour_mask.sum()) if contour_mask is not None else sum(len(e) for e in self._buf)
        n_g = 0
        if group_ids is not None and contour_mask is not None and len(group_ids):
            n_g = int(len(np.unique(group_ids[contour_mask])))
        return n_c, n_g

    def should_emit(self, dt_us: int, n_contour: int, n_groups: int) -> bool:
        if dt_us < self.min_dt_us:
            return False
        if dt_us >= self.max_dt_us:
            return True
        if self.proxy == "support":
            return n_groups >= self.min_groups and n_contour >= self.min_contour
        return n_contour >= self.min_contour

    def push(self, events_xypt: np.ndarray, t_us: np.ndarray,
             contour_mask: Optional[np.ndarray] = None,
             group_ids: Optional[np.ndarray] = None) -> List[Packet]:
        """Ingest a stream slice. `events_xypt` is `(N, 4)` [x,y,p,t_rel_unused]; `t_us` is absolute."""
        out: List[Packet] = []
        if len(t_us) == 0:
            return out
        if self._t0 is None:
            self._t0 = int(t_us[0])
        # Walk event by event only to decide cut points; the stored packet is a slice.
        start = 0
        if contour_mask is None:
            contour_mask = np.ones(len(t_us), dtype=bool)
        for i in range(len(t_us)):
            dt = int(t_us[i]) - int(self._t0)
            n_c, n_g = self._stats(contour_mask[start:i + 1],
                                   None if group_ids is None else group_ids[start:i + 1])
            if self.should_emit(dt, n_c, n_g):
                sl = slice(start, i + 1)
                ev = np.stack([events_xypt[sl, 0], events_xypt[sl, 1],
                               events_xypt[sl, 2] if events_xypt.shape[1] > 2
                               else np.zeros(i + 1 - start)], -1)
                out.append(Packet(events=ev, t_start_us=int(self._t0),
                                  t_end_us=int(t_us[i]), n_contour=n_c, n_groups=n_g))
                self.n_emitted += 1
                start = i + 1
                self._t0 = int(t_us[i]) if i + 1 < len(t_us) else None
            else:
                self.n_held += 1
        if start < len(t_us) and self._t0 is not None:
            # Remainder stays for the next call.
            self._buf = [events_xypt[start:], t_us[start:], contour_mask[start:],
                         None if group_ids is None else group_ids[start:]]
        else:
            self._buf = []
        return out


def contour_mask_from_field(fields, xy: torch.Tensor, batch_index: torch.Tensor,
                            margin: float = 2.0) -> torch.Tensor:
    """Events that land in the contour band of a KSSF raster, i.e. the ones that constrain pose."""
    q = fields.query(xy, batch_index)
    return (q["visibility"] > 0.5) & (q["sdf"].abs() <= margin)
