"""XKF evaluation: attach the causal long-window LNES to every evaluation packet without editing `semkine.eval_track`.

`LongWindowEval(runs)` is a context manager that rebinds `eval_track._window_events` (to learn the absolute endpoint
and the sequence's event arrays) and `eval_track.make_eval_packet` (to attach an incremental event timestamp surface
bitwise equal to `build_lnes(events, offsets, end, W)`, where `W = min(window_ms, end - run_start + 1)`).
The timestamp surface resets at run/sequence boundaries and on backward seeks.
The W50 events of the packet are unchanged, so the XYZ branch sees exactly what the XDL rows saw.
"""
from __future__ import annotations

import numpy as np
import torch

from semkine import eval_track as ET


class LongWindowEval:
    def __init__(self, runs, window_ms: int = 200, channels=("last",)):
        self.runs = np.asarray(runs, dtype=np.int64).reshape(-1, 2)
        self.window_ms = int(window_ms)
        if self.window_ms < 1:
            raise ValueError('XKF evaluation window must be positive')
        self.channels = tuple(channels)
        self._cur = None
        self._last = None
        self._surface_context = None

    def floor(self, end: int) -> int:
        hit = [int(a) for a, b in self.runs if a <= end < b]
        if not hit:
            raise RuntimeError(f'endpoint {end} outside every valid run')
        return hit[0]

    def lnes(self, events, offsets, end: int):
        end = int(end)
        run_a = self.floor(end)
        w = min(self.window_ms, end - run_a + 1)
        if self.channels != ('last',):
            return ET.build_lnes(events, offsets, end, w, self.channels), w
        start = end - w + 1
        previous = self._surface_context
        reset = (previous is None or previous[0] is not events or previous[1] is not offsets
                 or previous[2] != run_a or end < previous[3])
        if reset:
            self._last = np.full((ET.H, ET.W, 2), -1, np.int64)
            read_start = start
        else:
            read_start = max(start, previous[3] + 1)
        if read_start <= end:
            i, j = int(offsets[read_start]), int(offsets[end + 1])
            if i < j:
                ev = events[i:j]
                counts = np.diff(offsets[read_start:end + 2]).astype(np.int64)
                ms = np.repeat(np.arange(read_start, end + 1, dtype=np.int64), counts)
                self._last[ev[:, 1].astype(np.intp), ev[:, 0].astype(np.intp),
                           np.clip(ev[:, 2], 0, 1).astype(np.intp)] = ms
        self._surface_context = (events, offsets, run_a, end)
        img = (self._last - start).astype(np.float32) / float(w)
        img[self._last < start] = 0
        return img, w

    def __enter__(self):
        self._cur = None
        self._last = None
        self._surface_context = None
        self._orig = (ET._window_events, ET.make_eval_packet)
        orig_window, orig_make = self._orig

        def window_events(events, offsets, tsub, end, win):
            self._cur = (events, offsets, int(end))
            return orig_window(events, offsets, tsub, end, win)

        def make_eval_packet(ev5, prev, betas, camera_K, window_ms, device):
            pkt = orig_make(ev5, prev, betas, camera_K, window_ms, device)
            if self._cur is None:
                raise RuntimeError('make_eval_packet before _window_events: no endpoint known')
            events, offsets, end = self._cur
            img, w = self.lnes(events, offsets, end)
            pkt.lnes = torch.from_numpy(img).unsqueeze(0).to(device)
            pkt.lnes_window_ms = torch.tensor([w], dtype=torch.int64, device=device)
            self._cur = None
            return pkt

        ET._window_events, ET.make_eval_packet = window_events, make_eval_packet
        return self

    def __exit__(self, *exc):
        ET._window_events, ET.make_eval_packet = self._orig
        self._cur = self._last = self._surface_context = None
        return False


__all__ = ['LongWindowEval']
