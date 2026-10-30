"""XVM P4 samples (docs/EXPERIMENT_RECORDS.md#xvm-prereg-20261021 §3).

One sample = one 50 ms protocol endpoint `end` (not the first of its run): the anchor is the +-25 ms box-smoothed GT at
end - 50, the target the smoothed GT at end; the events are those of [max(run start, end - mem + 1), end], at most 8192
of the last 50 ms and 8192 older ones, both by uniform stride. Workers also build the per-(polarity, pixel) latest
event age (ms) for the patch-CNN reference.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from xvm import common as C
from xvm.common import EF

N_MEAS = 8192
N_OLD = 8192
H, W = EF.H, EF.W


def stride_take(n, m):
    if n <= m:
        return np.arange(n)
    return (np.arange(m) * (n / m)).astype(np.int64)


class P4Data(Dataset):
    def __init__(self, seqs, mem_ms: int, stride: int = 1, with_sae: bool = False, raw_window50: bool = False,
                 e3: bool = False):
        """`e3=True`: the XCL endpoint grid (every endpoint, the first of a run included), anchor = raw GT at the
        previous endpoint, target = raw GT at end (P5 / P6)."""
        self.mem = int(mem_ms)
        self.with_sae = with_sae
        self.raw50 = raw_window50
        self.e3 = e3
        self.seqs = []
        self.index = []
        for si, (s, d) in enumerate(seqs):
            sq = EF.Seq(s, d)
            self.seqs.append(sq)
            if e3:
                for rid, end, prev, first in C.endpoint_grid(sq.runs):
                    self.index.append((si, rid, end, prev))
                continue
            k = 0
            for rid, (a, b) in enumerate(sq.runs):
                for end in range(int(a) + 49, int(b), 50):
                    if end - 50 < a:
                        continue
                    if k % stride == 0:
                        self.index.append((si, rid, end, end - 50))
                    k += 1

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        si, rid, end, prev = self.index[i]
        sq = self.seqs[si]
        a, b = (int(v) for v in sq.runs[rid])
        lo = max(a, end - self.mem + 1)
        a0, a1 = int(sq.offsets[lo]), int(sq.offsets[end + 1])
        ev = np.asarray(sq.events[a0:a1])
        t_us = EF.event_times_us(sq, lo, end)
        age = (end + 1) * 1000.0 - t_us.astype(np.float64)                # us since the event, > 0
        age_ms = (age / 1000.0).astype(np.float32)
        meas = age_ms < 50.0
        im, io = np.nonzero(meas)[0], np.nonzero(~meas)[0]
        sel = np.concatenate([im[stride_take(len(im), N_MEAS)], io[stride_take(len(io), N_OLD)]])
        n = len(sel)
        xy = np.zeros((N_MEAS + N_OLD, 2), np.float32)
        ag = np.zeros(N_MEAS + N_OLD, np.float32)
        pol = np.zeros(N_MEAS + N_OLD, np.float32)
        msk = np.zeros(N_MEAS + N_OLD, bool)
        xy[:n] = ev[sel, :2]
        ag[:n] = age_ms[sel]
        pol[:n] = 2.0 * np.clip(ev[sel, 2], 0, 1) - 1.0
        msk[:n] = True
        if self.e3:
            x0, xt = np.asarray(sq.pos51[prev], np.float64), np.asarray(sq.pos51[end], np.float64)
        else:
            x0 = C.smoothed_gt(sq.pos51, prev, a, b - 1).astype(np.float64)
            xt = C.smoothed_gt(sq.pos51, end, a, b - 1).astype(np.float64)
        out = {"xy": xy, "age_ms": ag, "pol": pol, "ev_mask": msk, "x0": x0, "xt": xt, "prev": prev,
               "g0": np.asarray(sq.pos51[prev], np.float64), "gt": np.asarray(sq.pos51[end], np.float64),
               "betas": np.asarray(sq.betas, np.float64), "K": (sq.K * EF.RENDER_SCALE).astype(np.float64),
               "seq": si, "end": end, "n_events": int(a1 - a0)}
        if self.with_sae:
            sae = np.full((2, H * W), 1e9, np.float32)
            p = np.clip(ev[:, 2], 0, 1).astype(np.int64)
            key = p * (H * W) + ev[:, 1].astype(np.int64) * W + ev[:, 0].astype(np.int64)
            # events are time ordered: the last occurrence of a key is its latest event
            rk = key[::-1]
            _, first = np.unique(rk, return_index=True)
            last = len(key) - 1 - first
            sae.reshape(-1)[key[last]] = age_ms[last]
            out["sae_age"] = sae.reshape(2, H, W)
        if self.raw50:
            # C1 path: the events of [end-49, end] as the evaluator's ev5 rows [b, x, y, t_rel_s, p]
            lo5 = end - 49
            b0, b1 = int(sq.offsets[lo5]), int(sq.offsets[end + 1])
            e5 = np.asarray(sq.events[b0:b1])
            t5 = EF.event_times_us(sq, lo5, end) - lo5 * 1000
            sel5 = stride_take(len(e5), 2048)
            r = np.zeros((2048, 5), np.float32)
            m5 = np.zeros(2048, bool)
            k5 = len(sel5)
            r[:k5, 1:3] = e5[sel5, :2]
            r[:k5, 3] = t5[sel5] * 1e-6
            r[:k5, 4] = np.clip(e5[sel5, 2], 0, 1)
            m5[:k5] = True
            out["ev5"], out["ev5_mask"] = r, m5
        return out


def collate(items):
    out = {}
    for k in items[0]:
        v = [it[k] for it in items]
        out[k] = torch.as_tensor(np.stack(v)) if isinstance(v[0], np.ndarray) else torch.as_tensor(np.asarray(v))
    return out
