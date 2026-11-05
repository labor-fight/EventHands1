r"""ACU causal history (docs/EXPERIMENT_RECORDS.md#acu-s1c-prereg-20261104): training packets that carry the H = 50 ms of
events before their window, so that causal per-event computations (semkine/acu_pr.py) see exactly what a streaming
system would have in memory when the window's first event arrives.

  window      [start, end] as SemKineDataset draws it (raw-packet training path)
  history     the events of [max(run_start, start - H), start - 1] (a stream starts at its valid run's first ms)
  realisation one domain-randomisation draw over history + window (geometry, keep mask, polarity), so both are one stream;
              this is a new sample stream (not B2's: the extended window changes the draws)
  counts      n_hist = history events after the keep mask (gives the window's causal stride); k_hist = the history's own
              stride from the count of the H before it (its raw count times the sample's keep fraction)
  labels      the XAG pseudo-sources of the window events (supervision only)
`HistBatch` = EventPacketBatch + hist_events (Nh, 5) [b, x, y, t_rel < 0, p], hist_ptr (B + 1), n_hist (B,), k_hist (B,).
Opt-in: `install_dataset()` replaces `SemKineDataset.__getitem__` for training datasets; `collate_hist` is the loader's
collate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from . import dataset as D
from . import domrand as DR
from .acu_pr import H_S, stride_k
from .events import EV_COLS, EventPacket, EventPacketBatch, collate_packets

H_MS = int(round(H_S * 1000))
_ORIGINAL_GETITEM = D.SemKineDataset.__getitem__
_STATE = {"on": False}


def _getitem(self, idx: int):
    if not self.train:
        return _ORIGINAL_GETITEM(self, idx)
    if (self.input_mode != "raw_packed" or self._fast_on or self.prev_lag_ms is not None or self.unroll_pair
            or self.clip_k or self.triplet or self.fixed_window is not None):
        raise RuntimeError("the causal history is defined on the raw-packet training path only")
    rng = np.random.default_rng((self.seed * 1_000_003 + idx) & 0x7FFFFFFF)
    si, end, window, run_a = self._window(idx, rng)
    h = self._handle(si)
    start = end - window + 1
    h0 = max(int(run_a), start - H_MS)                         # history start (the stream starts at the run start)
    ext = end - h0 + 1
    xs, ys, ps, us = self._raw_events(h, end, ext)              # us relative to h0
    source_rows = np.arange(int(h.offsets[h0]), int(h.offsets[end + 1]), dtype=np.int64) if self.xag_source_root else None
    target = self._to_target(h.pos51[end])
    prev = self._to_target(h.pos51[self._prev_index(start, end, run_a)])
    camera_K = h.camera_K.copy()
    n_slots = self.height * self.width * 2 * ext
    dr = DR.sample_params(self.domrand, rng, n_slots)
    if self.domrand.enabled:
        if source_rows is None:
            xs, ys, ps, us = self._domrand_events(xs, ys, ps, us, ext, dr, camera_K, rng)
        else:
            xs, ys, ps, us, source_rows = self._domrand_events(xs, ys, ps, us, ext, dr, camera_K, rng,
                                                               source_rows=source_rows)
        stacked, camera_K = DR.transform_labels(np.stack([prev, target]), dr, camera_K, h.j0, self.render_scale)
        prev, target = stacked[0], stacked[1]
    noise = self._sample_prev_noise(rng)
    if noise is not None:
        prev = prev + noise
    flip = self.polarity_flip and int(rng.integers(0, 2)) == 1
    swap_mask = None
    if self.pixel_polarity_swap and self.train:
        swap_mask = rng.integers(0, 2, size=(self.height, self.width)).astype(bool)
    p_eff = self._polarity(xs, ys, ps, flip, swap_mask)
    shift = (start - h0) * 1000
    win = us >= shift
    packet = self._pack(
        si, h, xs[win], ys[win], p_eff[win], us[win] - shift, window,
        t_start_us=int(start) * 1000, t_end_us=(int(end) + 1) * 1000,
        target=target, prev=prev, camera_K=camera_K,
        is_start=bool(start == int(self.index[idx][2])), is_end=bool(end == int(self.index[idx][3]) - 1),
        meta={"window_ms": window, "end_idx": end, "seq": h.seq, "category": h.category, "domrand": dr},
        source_rows=None if source_rows is None else source_rows[win],
    )
    hist = np.empty((int((~win).sum()), 4), np.float32)
    if len(hist):
        hist[:, 0], hist[:, 1] = xs[~win], ys[~win]
        hist[:, 2] = (us[~win].astype(np.float64) - shift) * 1e-6
        hist[:, 3] = p_eff[~win]
    pp0 = max(int(run_a), h0 - H_MS)
    n_pp = int(h.offsets[h0] - h.offsets[pp0]) * float(dr.keep)
    packet.meta.update(acu_hist=hist, acu_n_hist=int(len(hist)), acu_hist_len_s=(start - h0) * 1e-3,
                       acu_k_hist=int(stride_k(n_pp, (h0 - pp0) * 1e-3, (start - h0) * 1e-3)))
    return packet


@dataclass
class HistBatch(EventPacketBatch):
    hist_events: Optional[torch.Tensor] = None
    hist_ptr: Optional[torch.Tensor] = None
    n_hist: Optional[torch.Tensor] = None
    hist_len_s: Optional[torch.Tensor] = None
    k_hist: Optional[torch.Tensor] = None

    def k_win(self):
        """The window's causal stride from the history's rate."""
        return stride_k(self.n_hist.cpu(), self.hist_len_s.cpu(), self.delta_t_s.cpu()).to(self.ptr.device)

    def to(self, device) -> "HistBatch":
        def mv(t):
            return t.to(device, non_blocking=True) if torch.is_tensor(t) else t
        return HistBatch(**{k: mv(v) for k, v in self.__dict__.items()})


def collate_hist(items):
    base = collate_packets(items)
    hs = [it.meta["acu_hist"] for it in items]
    counts = np.array([len(x) for x in hs], np.int64)
    ptr = np.zeros(len(items) + 1, np.int64)
    np.cumsum(counts, out=ptr[1:])
    ev = np.zeros((int(ptr[-1]), EV_COLS), np.float32)
    if ptr[-1]:
        ev[:, 0] = np.repeat(np.arange(len(items), dtype=np.float32), counts)
        ev[:, 1:] = np.concatenate([x for x in hs if len(x)], 0)
    return HistBatch(**base.__dict__, hist_events=torch.from_numpy(ev), hist_ptr=torch.from_numpy(ptr),
                     n_hist=torch.tensor([it.meta["acu_n_hist"] for it in items], dtype=torch.int64),
                     hist_len_s=torch.tensor([it.meta["acu_hist_len_s"] for it in items], dtype=torch.float64),
                     k_hist=torch.tensor([it.meta["acu_k_hist"] for it in items], dtype=torch.int64))


def install_dataset():
    D.SemKineDataset.__getitem__ = _getitem
    _STATE["on"] = True


def uninstall():
    D.SemKineDataset.__getitem__ = _ORIGINAL_GETITEM
    _STATE["on"] = False


# --------------------------------------------------------------------------- evaluation streams
def eval_history(events, offsets, tsub, end, run_a, window=50):
    """For the evaluation window [end - window + 1, end] of a recording: (history events (Nh, 4) [x, y, t_rel < 0, p] of
    [max(run_a, start - H), start - 1], n_hist, hist_len_s, k_hist), as a stream that began at the run start holds them."""
    from .eval_track import _window_events
    start = int(end) - window + 1
    h0 = max(int(run_a), start - H_MS)
    if h0 < start:
        ev5 = _window_events(events, offsets, tsub, start - 1, start - h0)
        hist = ev5[:, 1:].copy()
        hist[:, 2] = hist[:, 2] - (start - h0) * 1e-3
    else:
        hist = np.zeros((0, 4), np.float32)
    pp0 = max(int(run_a), h0 - H_MS)
    n_pp = int(offsets[h0] - offsets[pp0])
    k_hist = int(stride_k(n_pp, (h0 - pp0) * 1e-3, (start - h0) * 1e-3))
    return hist.astype(np.float32), int(len(hist)), (start - h0) * 1e-3, k_hist


__all__ = ["H_MS", "HistBatch", "collate_hist", "eval_history", "install_dataset", "uninstall"]
