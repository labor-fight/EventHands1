r"""S45-TR arms N3 and N4: where the training prior comes from (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030).

S45 trains on `prior = GT(window start) + noise`, the noise drawn by `SemKineDataset._sample_prev_noise` from the
mixed contract: 50 % small (5 mm / 0.05 rad / 0.05 rad per DOF), 30 % large (50 mm / 0.3 / 0.3), 20 % directed.
Each arm here replaces the prior of a registered share of the training samples with an *alternative prior*,
and changes nothing else:

* **N3, time-offset prior.** Where S45 drew the large or the directed branch (its own first draw `u >= P_SMALL`),
  the prior is the ground truth of the same valid run at `start + s * D` instead, `D` uniform in [100, 1000] ms
  (integer ms), `s = +-1` with probability 1/2; if that time leaves the run the opposite sign is used, then it is
  clipped into the run. No noise is added to it. The small branch (`u < P_SMALL`) is S45's sample bit for bit.
* **N4, own closed-loop prior.** With probability 1/2 (an independent coin), the prior is the state the frozen S45
  held in its own W50 closed loop on this recording just before the window: the bank state at the latest grid
  endpoint `e < start` (`tools/s45tr/n4_bank.py`). No noise is added to it. Otherwise, or when the bank has no
  endpoint before `start`, the sample is S45's bit for bit.

Every draw the arms add comes from a separate per-sample stream `default_rng([seed, idx, TAG])`; S45's own stream
is peeked (state saved and restored) but never advanced differently, so the events, the window, the augmentation,
the polarity draws and the target of every sample are S45's. The alternative prior goes through the same
`domrand.transform_labels` realisation as the labels. Validation and evaluation datasets (`train=False`) are
untouched.

Opt-in only: `install_n3(...)` / `install_n4(...)` replace `SemKineDataset.__getitem__` for the process (DataLoader
workers fork after it). The copied training path is the raw-packet one S45 uses; any other dataset mode refuses.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from semkine import dataset as D
from semkine import domrand as DR

_ORIGINAL_GETITEM = D.SemKineDataset.__getitem__
TAG_N3 = 0x4E33
TAG_N4 = 0x4E34

#: the installed alternative-prior source, or None (S45 behaviour)
_SOURCE = None


def offset_ms(rng: np.random.Generator, start: int, run_a: int, run_b: int, lo: int, hi: int):
    """N3: `(prior_ms, sign, magnitude)` for a window starting at `start` inside the valid run `[run_a, run_b)`."""
    mag = int(rng.integers(int(lo), int(hi) + 1))
    sign = 1 if float(rng.random()) < 0.5 else -1
    cand = start + sign * mag
    if not (run_a <= cand <= run_b - 1):
        cand = start - sign * mag
    return int(min(max(cand, run_a), run_b - 1)), sign, mag


class OffsetSource:
    """N3: replaces the large and directed branches by the run's ground truth at a random time offset."""

    name = "n3"

    def __init__(self, lo_ms: int = 100, hi_ms: int = 1000):
        if not (0 < int(lo_ms) <= int(hi_ms)):
            raise ValueError("N3 offset range must satisfy 0 < lo <= hi")
        self.lo, self.hi = int(lo_ms), int(hi_ms)

    def draw(self, ds, idx, h, start, run_a, run_b):
        rng = np.random.default_rng([ds.seed, int(idx), TAG_N3])
        ms, _, _ = offset_ms(rng, start, run_a, run_b, self.lo, self.hi)
        return ds._to_target(h.pos51[ms])

    def use(self, ds, idx, u, alt):
        return u >= ds.prev_noise_mix[0]


class BankSource:
    """N4: half of the priors are S45's own closed-loop state just before the window."""

    name = "n4"

    def __init__(self, root, p: float = 0.5):
        self.root = Path(root)
        self.p = float(p)
        if not self.root.is_dir():
            raise FileNotFoundError(f"N4 state bank missing: {self.root}")
        self._cache = {}

    def bank(self, seq):
        b = self._cache.get(seq)
        if b is None:
            with np.load(self.root / f"{seq}.npz", allow_pickle=False) as z:
                b = (np.asarray(z["end_ms"], np.int64), np.asarray(z["state"], np.float32))
            if b[0].ndim != 1 or b[1].shape != (len(b[0]), 51) or not np.all(np.diff(b[0]) > 0) \
                    or not np.isfinite(b[1]).all():
                raise ValueError(f"malformed N4 bank for {seq}")
            self._cache[seq] = b
        return b

    def draw(self, ds, idx, h, start, run_a, run_b):
        rng = np.random.default_rng([ds.seed, int(idx), TAG_N4])
        if float(rng.random()) >= self.p:
            return None
        end_ms, state = self.bank(h.seq)
        k = int(np.searchsorted(end_ms, start, side="left")) - 1      # latest endpoint e < start
        if k < 0 or end_ms[k] < run_a:
            return None
        return state[k].copy()

    def use(self, ds, idx, u, alt):
        return alt is not None


def _require_supported(ds):
    if (ds.input_mode != "raw_packed" or getattr(ds, "_fast_on", False) or ds.unroll_pair or ds.clip_k
            or ds.triplet or ds.prev_lag_ms is not None or ds.prev_noise_mode != "mixed"
            or ds.fixed_window is not None):
        raise RuntimeError("S45-TR alternative priors are defined on the S45 raw-packet training path only")


def _getitem(self, idx: int):
    source = _SOURCE
    if source is None or not self.train:
        return _ORIGINAL_GETITEM(self, idx)
    _require_supported(self)
    # ---- SemKineDataset.__getitem__, raw-packet training path, verbatim except the marked S45-TR lines
    rng = np.random.default_rng((self.seed * 1_000_003 + idx) & 0x7FFFFFFF)
    si, end, window, run_a = self._window(idx, rng)
    h = self._handle(si)
    start = end - window + 1

    xs, ys, ps, us = self._raw_events(h, end, window)
    source_rows = np.arange(int(h.offsets[start]), int(h.offsets[end + 1]), dtype=np.int64) \
        if self.xag_source_root else None
    target = self._to_target(h.pos51[end])
    prev = self._to_target(h.pos51[self._prev_index(start, end, run_a)])
    camera_K = h.camera_K.copy()
    # S45-TR: the alternative prior (raw camera frame), from its own stream
    alt = source.draw(self, idx, h, int(start), int(run_a), int(self.index[idx][3]))
    target_raw, camera_K_raw = target.copy(), camera_K.copy()

    n_slots = self.height * self.width * 2 * window
    dr = DR.sample_params(self.domrand, rng, n_slots)
    if self.domrand.enabled:
        if source_rows is None:
            xs, ys, ps, us = self._domrand_events(xs, ys, ps, us, window, dr, camera_K, rng)
        else:
            xs, ys, ps, us, source_rows = self._domrand_events(
                xs, ys, ps, us, window, dr, camera_K, rng, source_rows=source_rows)
        stacked, camera_K = DR.transform_labels(
            np.stack([prev, target]), dr, camera_K, h.j0, self.render_scale
        )
        prev, target = stacked[0], stacked[1]
        # S45-TR: the same realisation mirrored onto the alternative prior
        if alt is not None:
            alt = DR.transform_labels(np.stack([alt, target_raw]), dr, camera_K_raw, h.j0, self.render_scale)[0][0]

    # -------- teacher forcing noise on the previous state
    # S45-TR: peek S45's branch draw, then draw exactly as S45 does
    state = rng.bit_generator.state
    u = float(rng.random())
    rng.bit_generator.state = state
    noise = self._sample_prev_noise(rng)
    if alt is not None and source.use(self, idx, u, alt):
        prev = np.asarray(alt, dtype=np.float32)
    elif noise is not None:
        prev = prev + noise

    flip = self.polarity_flip and int(rng.integers(0, 2)) == 1
    swap_mask = None
    if self.pixel_polarity_swap and self.train:
        swap_mask = rng.integers(0, 2, size=(self.height, self.width)).astype(bool)
    p_eff = self._polarity(xs, ys, ps, flip, swap_mask)

    return self._pack(
        si, h, xs, ys, p_eff, us, window,
        t_start_us=int(start) * 1000, t_end_us=(int(end) + 1) * 1000,
        target=target, prev=prev, camera_K=camera_K,
        is_start=bool(start == int(self.index[idx][2])),
        is_end=bool(end == int(self.index[idx][3]) - 1),
        meta={"window_ms": window, "end_idx": end, "seq": h.seq,
              "category": h.category, "domrand": dr},
        source_rows=source_rows,
    )


def _install(source) -> None:
    global _SOURCE
    _SOURCE = source
    D.SemKineDataset.__getitem__ = _getitem


def install_n3(lo_ms: int = 100, hi_ms: int = 1000) -> None:
    _install(OffsetSource(lo_ms, hi_ms))


def install_n4(bank_root, p: float = 0.5) -> None:
    _install(BankSource(bank_root, p))


def uninstall() -> None:
    global _SOURCE
    _SOURCE = None
    D.SemKineDataset.__getitem__ = _ORIGINAL_GETITEM


def installed():
    return None if _SOURCE is None else _SOURCE.name


__all__ = ["BankSource", "OffsetSource", "install_n3", "install_n4", "installed", "offset_ms", "uninstall"]
