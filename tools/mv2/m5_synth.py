#!/usr/bin/env python3
"""M5 planted-signal generator (tool validity only; never an effect reading).

Draws Poisson events from the SBA mean field at a GT trajectory of a real recording (its own calibrated kappa, an
explicit background rate, no outlier component), then hands them to the normal `make_problem` pipeline on the
*injected* (biased, noisy) tracker trajectory.  A correct aligner must recover a planted bias on the axes the footprint
model is sensitive to; the formal run repeats this check as the 'tool_defect' condition.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from mv2 import m5_core as M
from semkine import sba as SB

BKG = 3e-4          # background events per source pixel per window


def pick_moving_anchor(sj: "M.Subject", w_ms: int, n_scan: int = 24) -> int:
    """Anchor (evaluator grid, >= 1 s of history) with the largest root-rotation + translation change over the window."""
    cand = sj.anchors(history_ms=max(w_ms, 1000), n=n_scan)
    best, bv = cand[0], -1.0
    for e in cand:
        a, b = sj.pos51[e - w_ms], sj.pos51[e]
        v = float(np.linalg.norm(b[3:6] - a[3:6])) + 10 * float(np.linalg.norm(b[:3] - a[:3])) \
            + 0.3 * float(np.linalg.norm(b[6:] - a[6:]))
        if v > bv:
            best, bv = e, v
    return best


@torch.no_grad()
def synth_events(g: "M.Geo", X_gt: torch.Tensor, betas, K, m_prev: int, m1: int, seed: int,
                 bkg: float = BKG) -> Tuple[np.ndarray, np.ndarray]:
    """(ev uint8 (n, 3), ms int64 (n,)) drawn from the full-hand swept-area mean field at the GT trajectory."""
    pb = M.make_problem(g, X_gt, betas, K, np.zeros((0, 3), np.uint8), np.zeros(0, np.int64), m_prev, m1, "full")
    win = pb.win
    plan_all, _, _ = M.edge_plans(pb, pb.V)
    param = M.ConstParam(pb.T)
    A = M.areas(pb, param, torch.zeros(1, 6), plan_all)                       # (1, 3, S, h, w)
    sl = (slice(win.crop.y0, win.crop.y0 + win.crop.h), slice(win.crop.x0, win.crop.x0 + win.crop.w))
    W = m1 - m_prev
    gain = g.geo.grid.gain[sl]
    win.bkg = (bkg * gain)[None] * (win.widths / W)[:, None, None]
    win.uni = None
    mod = SB.Model(kappa_sil=g.model.kappa_sil, kappa_int=g.model.kappa_int, signed=False, dispersion=1.0,
                   blur=g.model.blur, outlier=0.0)
    lam = SB.rates(A, win, mod)[0, 0]                                         # (S, h, w)
    rng = np.random.default_rng(seed)
    cnt = rng.poisson(lam.numpy())
    s_i, y_i, x_i = np.nonzero(cnt)
    c = cnt[s_i, y_i, x_i]
    s_r, y_r, x_r = np.repeat(s_i, c), np.repeat(y_i, c), np.repeat(x_i, c)
    n = len(s_r)
    ev = np.zeros((n, 3), np.uint8)
    ev[:, 0] = x_r + win.crop.x0
    ev[:, 1] = y_r + win.crop.y0
    ev[:, 2] = rng.integers(0, 2, n)
    bm = pb.bin_ms
    ms = m_prev + 1 + bm * s_r + rng.integers(0, bm, n)
    ms = np.minimum(ms, m1)
    return ev, ms.astype(np.int64)


def planted_case(g: "M.Geo", sj: "M.Subject", end: int, w_ms: int, axis: str, sign: int, deg: float, seed: int,
                 evidence: str = "palm", mode: str = "none"):
    """(Problem on the injected trajectory with synthetic events, true persistent offset (rad, 3))."""
    m_prev = end - w_ms
    edges = M.window_edges(m_prev, end)
    gt = sj.pos51[edges]
    X_gt = torch.tensor(gt, dtype=torch.float32)
    ev, ms = synth_events(g, X_gt, sj.betas, sj.K, m_prev, end, seed)
    rng = np.random.default_rng(seed + 1)
    nz = M.make_noise(mode, edges, end, m_prev, rng)
    b = M.axis_vec(axis, sign, deg)
    Xp = M.inject(gt, b, nz)
    pb = M.make_problem(g, Xp, sj.betas, sj.K, ev, ms, m_prev, end, evidence)
    return pb, M.true_offset(gt, Xp)
