#!/usr/bin/env python3
"""M5 (palm-rigid alignment + segment-level root bias) shared core: geometry, injection, SBA-footprint likelihood,
fixed-iteration Gauss-Newton on a 6-DoF palm SE(3) correction, and the joint (bias, trajectory) block used by the Schur
spectrum.  CPU only; nothing here imports a trained model.  Pre-registration draft: tools/mv2/prereg_m5.md.

Likelihood = the SBA swept-boundary Poisson footprint (`semkine/sba.py`, calibration `outputs/sba/s0/calibration.json`,
merged polarity, kappa profiled per window) restricted to the *palm* contour edges (both end vertices have wrist LBS
weight >= 0.9) with the cells swept by the non-palm (finger) contour masked out ("palm evidence"); `evidence='full'`
keeps every contour edge and no mask (the "full-hand solve" control).

Parametrisation (the only thing the aligner moves).  The tracker output trajectory x_pred(t) (51-D) is kept; a
correction acts on the whole hand about the wrist pivot (the LBS pivot j0 + t, so a root rotation never moves the wrist):

    v'(t) = w(t) + Exp(om(t)) (v(t) - w(t)) + tau(t)           w = FK wrist joint, v = FK vertex

i.e. R_root <- Exp(om) R_root (camera-frame left multiplication, the same convention as the injected bias) and the
wrist translation += tau.  `ConstParam`: om, tau constant over the window (6-D = the "palm rigid SE(3)").  `KnotParam`:
constant bias b (6-D) + knot trajectory s (6 per knot, linear hat interpolation) = the (bias, trajectory) pair of the
Schur identifiability spectrum; b + s(t) is the correction at time t, so b is exactly aliased with the constant mode of s
unless a level prior on s pins it (appendix L.B5).
"""
from __future__ import annotations

import json
import math
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mv2 import common as MC  # noqa: E402
from semkine import sba as SB  # noqa: E402
from semkine.lie import so3_exp, so3_log  # noqa: E402

SEED = 20261026
PALM_W = 0.9                                   # wrist LBS weight threshold of the palm vertex set
WINDOWS_MS = (50, 100, 200, 500, 1000)         # causal event windows of the injection recovery
TIERS_DEG = (2.0, 5.0, 10.0)                   # injected segment-level root rotation biases
AXES = ("x", "y", "z")                         # camera axes (balanced design; z = in-plane roll)
NOISE_SIG = {"T": 0.005, "R": 0.05, "POSE": 0.05}   # configs/dt/dt_dz_l3.yaml TRACK.PREV_NOISE_* (small class)
NOISE_KNOT_MS = 50                             # one noise innovation per tracker step
#: lag-1 (50 ms) autocorrelation of the injected tracker error: the measured closed-loop root-error autocorrelation of S39
#: is 0.95 (global) / 0.85 (local) (docs/EXPERIMENT_RECORDS.md#dt2-verdict 2.1); iid noise per step (rho = 0) is run as a
#: sensitivity arm only
NOISE_RHO = 0.9
CALIB = REPO / "outputs" / "sba" / "s0" / "calibration.json"
P0V = REPO / "outputs" / "sba" / "p0v" / "result.json"

# ---- fixed solver schedule (set on planted synthetic events only, never on real events; tools/mv2/prereg_m5.md)
K_ITERS = 8
POOLS = (2, 2, 1, 1, 1, 1, 1, 1)               # coarse-to-fine output-pixel pooling per iteration (SB.POOLS family)
BLURS = (8.0, 5.0, 3.0, 2.0, 1.5, 1.0, 1.0, 1.0)   # per-iteration blur (output px) of the mean field; the calibrated one is 1.0
MUS = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.05, 0.05)  # fixed Levenberg damping (relative to the Hessian diagonal)
#: Gaussian prior on the aligner's (tau [m], om [rad] per axis): the population scale of a segment bias, ~5 deg per
#: segment (docs/EXPERIMENT_RECORDS.md#dt2-verdict), and the PREV_NOISE_T scale (5 mm) of the translation nuisance
PRIOR_SIG_T, PRIOR_SIG_R = 0.005, math.radians(5.0)
SD_GATE_DEG = 2.5                              # runtime identifiability gate: worst-axis posterior SD of om <= this
PALM_MARGIN = 14                               # output px kept around the palm vertices' projection (likelihood cells)
BIN_SHORT, BIN_LONG, BIN_SWITCH = 5, 10, 200   # SBA bin width: 5 ms for windows <= 200 ms, 10 ms above (CPU budget)
FD_T, FD_R = SB.FD_T, SB.FD_R
CAP_T, CAP_R = SB.CAP_T, SB.CAP_R


def seed_of(*parts) -> int:
    return (zlib.crc32("|".join(str(p) for p in parts).encode()) ^ SEED) & 0x7FFFFFFF


# ------------------------------------------------------------------------------------------------ geometry
class Geo:
    """MANO layer + SBA geometry (topology, grid, part groups) on CPU, palm vertex set, calibrated likelihood model."""

    def __init__(self, device=torch.device("cpu")):
        from mano_layer import ManoLayer
        self.dev = device
        self.mano = ManoLayer(str(REPO / "assets" / "mano_right.npz"), add_mean=False).to(device).eval()
        self.geo = SB.Geometry(self.mano, device)
        w = self.mano.weights.detach().float()
        self.palm = (w[:, 0] >= PALM_W).to(device)                             # (778,) bool
        self.palm_idx = self.palm.nonzero(as_tuple=True)[0]
        e = self.geo.topo.edges
        self.palm_edge = self.palm[e[:, 0]] & self.palm[e[:, 1]]               # (E,) both end vertices in the set
        self.hands_mean = self.mano.hands_mean.detach().float().reshape(15, 3)
        c = json.loads(CALIB.read_text())
        self.model = SB.Model(kappa_sil=c["kappa_sil"], kappa_int=c["kappa_int"], signed=False,
                              dispersion=float(c["dispersion"]), blur=float(c["blur"]), outlier=float(c["outlier"]))

    @property
    def n_palm(self) -> int:
        return int(self.palm.sum())

    def fk(self, X: torch.Tensor, betas: torch.Tensor):
        return self.geo.fk(X, betas)


@dataclass
class Subject:
    """Recording-level data on CPU: events (mmap), offsets, GT per ms, runs, betas, source K."""
    seq: str
    events: np.ndarray
    offsets: np.ndarray
    pos51: np.ndarray
    runs: np.ndarray
    betas: torch.Tensor          # (1, 10)
    K: torch.Tensor              # (3, 3) 640 x 480 camera

    @staticmethod
    def load(seq: str) -> "Subject":
        ev, off, aux, pos = MC.load_seq(seq)
        return Subject(seq, ev, off, np.asarray(pos, np.float32), np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2),
                       torch.tensor(np.asarray(aux["betas"], np.float32).reshape(-1)[:10])[None],
                       torch.tensor(np.asarray(aux["camera_K"], np.float32).reshape(3, 3)))

    def window_events(self, m_prev: int, m1: int):
        """Events of (m_prev, m1] and their integer ms (the SBA `SeqData.events` convention)."""
        a0, a1 = int(self.offsets[m_prev + 1]), int(self.offsets[m1 + 1])
        ev = np.asarray(self.events[a0:a1])
        cnt = np.diff(self.offsets[m_prev + 1:m1 + 2]).astype(np.int64)
        return ev, np.repeat(np.arange(m_prev + 1, m1 + 1, dtype=np.int64), cnt)

    def anchors(self, history_ms: int = 1000, step: int = 50, n: int = 12) -> List[int]:
        """Evenly spaced window-end ms on the evaluator's 50 ms grid (a + 49 + 50 k) with >= `history_ms` of valid
        history inside the same run (so every causal window up to `history_ms` is unclipped)."""
        cand = []
        for a, b in self.runs:
            e = np.arange(a + step - 1, b, step, dtype=np.int64)
            cand += [int(x) for x in e[(e - history_ms) >= a]]
        if not cand:
            return []
        if len(cand) <= n:
            return cand
        return [cand[int(round(i))] for i in np.linspace(0, len(cand) - 1, n)]


def shuffle_times(ms: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Time-shuffled null: event positions / polarities fixed, their ms stamps permuted inside the window."""
    return ms[rng.permutation(len(ms))]


# ------------------------------------------------------------------------------------------------ injection
def sig51() -> np.ndarray:
    s = np.zeros(51, np.float64)
    s[0:3], s[3:6], s[6:] = NOISE_SIG["T"], NOISE_SIG["R"], NOISE_SIG["POSE"]
    return s


def noise_knots(end: int, t0: int, rng: np.random.Generator, scale: float = 1.0, rho: float = NOISE_RHO) -> np.ndarray:
    """(nk, 51) PREV_NOISE (small class) at the 50 ms tracker steps (index 0 = `end`, 1 = end - 50, ..., reaching back to
    `t0`): a stationary AR(1) process with marginal N(0, sig^2) and lag-1 autocorrelation `rho` (rho = 0: iid draws, the
    literal per-sample PREV_NOISE).  Deterministic given `rng`."""
    nk = int(math.ceil((end - t0) / NOISE_KNOT_MS)) + 1
    eps = rng.standard_normal((nk, 51))
    x = np.zeros((nk, 51))
    x[-1] = eps[-1]                                                           # earliest knot
    for k in range(nk - 2, -1, -1):
        x[k] = rho * x[k + 1] + math.sqrt(max(1.0 - rho * rho, 0.0)) * eps[k]
    return x * sig51()[None] * scale


def noise_at_edges(edges_ms: np.ndarray, end: int, draws: np.ndarray) -> np.ndarray:
    """(T, 51) noise at `edges_ms`: the knot draws, linear in between."""
    nk = draws.shape[0]
    pos = (end - np.asarray(edges_ms, np.float64)) / NOISE_KNOT_MS             # fractional knot index (0 at the end)
    i0 = np.clip(np.floor(pos).astype(int), 0, nk - 1)
    i1 = np.clip(i0 + 1, 0, nk - 1)
    f = (pos - i0)[:, None]
    return (1 - f) * draws[i0] + f * draws[i1]


NOISE_MODES = ("const", "none", "ar", "iid")


def make_noise(mode: str, edges_ms: np.ndarray, end: int, t0: int, rng: np.random.Generator) -> Optional[np.ndarray]:
    """(T, 51) injected tracker noise.  'const' (primary): ONE PREV_NOISE (small class) draw for the translation and the 45
    finger blocks, held over the window (the root part of PREV_NOISE is the injected bias itself); 'none': nothing;
    'ar': PREV_NOISE on every block as a stationary AR(1) process over the 50 ms tracker steps (lag-1 `NOISE_RHO`);
    'iid': the same with independent draws per step (literal per-sample PREV_NOISE)."""
    T = len(edges_ms)
    if mode == "none":
        return None
    if mode == "const":
        d = rng.standard_normal(51) * sig51()
        d[3:6] = 0.0
        return np.broadcast_to(d, (T, 51)).copy()
    if mode in ("ar", "iid"):
        return noise_at_edges(edges_ms, end, noise_knots(end, t0, rng, rho=NOISE_RHO if mode == "ar" else 0.0))
    raise ValueError(mode)


def true_offset(gt_edges: np.ndarray, X_pred: torch.Tensor) -> np.ndarray:
    """Window-mean root-rotation error vector (rad, camera frame, R_pred = Exp(e) R_gt): the persistent offset a bias
    estimator has to remove; equals the injected b exactly when there is no root noise."""
    Rp = so3_exp(X_pred[:, 3:6].double())
    Rg = so3_exp(torch.tensor(gt_edges[:, 3:6], dtype=torch.float64))
    return so3_log(Rp @ Rg.transpose(-1, -2)).mean(0).numpy()


def inject(gt_edges: np.ndarray, b_vec: np.ndarray, noise: Optional[np.ndarray]) -> torch.Tensor:
    """Tracker trajectory = GT + PREV_NOISE (additive axis-angle, as the evaluator's `prev = pos + noise`), then the
    root rotated by the segment-level bias b (rad, camera frame, R <- Exp(b) R)."""
    X = np.asarray(gt_edges, np.float64).copy()
    if noise is not None:
        X = X + noise
    Xt = torch.tensor(X, dtype=torch.float32)
    if float(np.linalg.norm(b_vec)) > 0:
        R = so3_exp(torch.tensor(b_vec, dtype=torch.float32))[None] @ so3_exp(Xt[:, 3:6])
        Xt[:, 3:6] = so3_log(R)
    return Xt


def axis_vec(axis: str, sign: int, deg: float) -> np.ndarray:
    v = np.zeros(3)
    v["xyz".index(axis)] = sign * math.radians(deg)
    return v


def geodesic_deg(w: torch.Tensor) -> float:
    return float(torch.rad2deg(w.norm()))


# ------------------------------------------------------------------------------------------------ rigid corrections
def apply_corr(V: torch.Tensor, Wr: torch.Tensor, tau: torch.Tensor, om: torch.Tensor) -> torch.Tensor:
    """V (T, N, 3), Wr (T, 3), tau / om (H, T, 3) -> (H, T, N, 3): rotate about the wrist by Exp(om), translate by tau."""
    R = so3_exp(om)                                                          # (H, T, 3, 3)
    d = V - Wr[:, None]                                                      # (T, N, 3)
    return torch.einsum("htij,tnj->htni", R, d) + (Wr[None] + tau)[:, :, None]


class ConstParam:
    """theta = [tau (3), om (3)] constant over the window (the palm rigid SE(3))."""

    def __init__(self, T: int):
        self.T, self.P = T, 6
        self.eps = torch.tensor([FD_T] * 3 + [FD_R] * 3, dtype=torch.float32)
        self.blocks = ((0, 3, "t"), (3, 6, "r"))

    def expand(self, theta: torch.Tensor):
        th = theta.view(-1, 6)
        return th[:, None, :3].expand(-1, self.T, -1), th[:, None, 3:].expand(-1, self.T, -1)


class KnotParam:
    """theta = [b (tau3, om3), s_0 (tau3, om3), ..., s_{nk-1}]; correction(t) = b + sum_k phi_k(t) s_k, hat functions on
    `nk` equally spaced knots over the window (nk >= 2)."""

    def __init__(self, T: int, nk: int):
        self.T, self.nk, self.P = T, nk, 6 + 6 * nk
        x = np.linspace(0.0, nk - 1.0, T)
        Phi = np.zeros((T, nk), np.float32)
        i0 = np.minimum(np.floor(x).astype(int), nk - 2)
        f = (x - i0).astype(np.float32)
        Phi[np.arange(T), i0] = 1 - f
        Phi[np.arange(T), i0 + 1] = f
        self.Phi = torch.tensor(Phi)
        e = [FD_T] * 3 + [FD_R] * 3
        self.eps = torch.tensor(e * (1 + nk), dtype=torch.float32)
        self.blocks = tuple((6 * k + a, 6 * k + a + 3, "t" if a == 0 else "r") for k in range(1 + nk) for a in (0, 3))

    def expand(self, theta: torch.Tensor):
        th = theta.view(-1, 1 + self.nk, 6)
        b, s = th[:, 0], th[:, 1:]                                           # (H, 6), (H, nk, 6)
        c = b[:, None] + torch.einsum("tk,hkc->htc", self.Phi, s)            # (H, T, 6)
        return c[..., :3], c[..., 3:]


# ------------------------------------------------------------------------------------------------ problem
import contextlib  # noqa: E402
from dataclasses import replace  # noqa: E402


def bin_for(w_ms: int) -> int:
    return BIN_SHORT if w_ms <= BIN_SWITCH else BIN_LONG


@contextlib.contextmanager
def with_bins(bin_ms: int):
    """`semkine.sba.BIN_MS` is a module constant read at call time by `bins_of` / `make_window`; set it for the duration
    of a window construction (process-local) without touching the frozen file."""
    old = SB.BIN_MS
    SB.BIN_MS = int(bin_ms)
    try:
        yield
    finally:
        SB.BIN_MS = old


def window_edges(m_prev: int, m1: int, bin_ms: Optional[int] = None) -> np.ndarray:
    with with_bins(bin_ms or bin_for(m1 - m_prev)):
        _, edges, _ = SB.bins_of(m_prev, m1)
    return np.asarray(edges, np.int64)


@dataclass
class Problem:
    g: Geo
    V: torch.Tensor              # (T, 778, 3) tracker trajectory vertices (before any correction)
    Wr: torch.Tensor             # (T, 3) wrist
    K: torch.Tensor
    win: SB.Window               # (palm evidence: cropped to the palm box)
    S: int
    evidence: str                # 'palm' | 'full'
    n_events: int
    bin_ms: int
    J_last: Optional[torch.Tensor] = None   # (21, 3) joints at the window end (capsule mask)
    uv0: Optional[torch.Tensor] = None      # (T, 778, 2) projected trajectory (crop)

    @property
    def T(self) -> int:
        return self.S + 1


def _subwindow(win: SB.Window, crop2: SB.Crop) -> SB.Window:
    c = win.crop
    ys, xs = slice(crop2.y0 - c.y0, crop2.y0 - c.y0 + crop2.h), slice(crop2.x0 - c.x0, crop2.x0 - c.x0 + crop2.w)
    return replace(win, N=win.N[:, :, ys, xs].contiguous(), bkg=win.bkg[:, ys, xs].contiguous(),
                   uni=None if win.uni is None else win.uni[:, ys, xs].contiguous(), crop=crop2)


def _build_window(g: Geo, V, J_last, uv, K, ev, ms, m_prev, m1, evidence, bm) -> SB.Window:
    with with_bins(bm), torch.no_grad():
        crop = SB.crop_from(uv)
        hand_box = SB.crop_from(uv[-1:], margin=0)
        cap = SB.capsule_mask(J_last, K, g.geo.grid)
        win = SB.make_window(ev, ms, m_prev, m1, crop, g.geo.grid, cap, hand_box, g.dev)
        if evidence == "palm":
            c2 = SB.crop_from(uv[:, g.palm_idx], margin=PALM_MARGIN)
            y0, x0 = max(c2.y0, crop.y0), max(c2.x0, crop.x0)
            y1, x1 = min(c2.y0 + c2.h, crop.y0 + crop.h), min(c2.x0 + c2.w, crop.x0 + crop.w)
            win = _subwindow(win, SB.Crop(y0, x0, max(1, y1 - y0), max(1, x1 - x0)))
    return win


def make_problem(g: Geo, X_edges: torch.Tensor, betas: torch.Tensor, K: torch.Tensor, ev: np.ndarray, ms: np.ndarray,
                 m_prev: int, m1: int, evidence: str = "palm", bin_ms: Optional[int] = None) -> Problem:
    """X_edges (S+1, 51): tracker states at the SBA bin edges (m_prev + bin s, ..., m1)."""
    bm = bin_ms or bin_for(m1 - m_prev)
    with with_bins(bm), torch.no_grad():
        S, edges, alphas = SB.bins_of(m_prev, m1)
        assert X_edges.shape[0] == S + 1, (X_edges.shape, S)
        V, J = g.fk(X_edges, betas)
        uv = SB.project_src(V, K)
    win = _build_window(g, V, J[-1], uv, K, ev, ms, m_prev, m1, evidence, bm)
    return Problem(g, V, J[:, 0].contiguous(), K, win, S, evidence, int(win.n_events), bm, J[-1], uv)


def rewindow(pb: Problem, ev: np.ndarray, ms: np.ndarray, m_prev: int, m1: int) -> Problem:
    """The same trajectory with other events (the time-shuffled null): geometry is reused, the window rebuilt."""
    win = _build_window(pb.g, pb.V, pb.J_last, pb.uv0, pb.K, ev, ms, m_prev, m1, pb.evidence, pb.bin_ms)
    return replace(pb, win=win, n_events=int(win.n_events))


def edge_plans(pb: Problem, V: torch.Tensor):
    """Contour plans at the (corrected) base trajectory V (T, 778, 3): (palm plan, finger plan or None)."""
    g, geo = pb.g, pb.g.geo
    uv = SB.project_src(V, pb.K)
    S = V.shape[0] - 1
    on, side, internal = SB.contour_entries(V[:S], uv[:S], geo.topo, geo.grid, geo.groups, geo.n_groups)
    plans = []
    keeps = (g.palm_edge, ~g.palm_edge) if pb.evidence == "palm" else (torch.ones_like(g.palm_edge), None)
    for keep in keeps:
        if keep is None:
            plans.append(None)
            continue
        plans.append(_plan(on & keep[None], side, internal, uv, geo.topo))
    return plans[0], plans[1], uv


def _plan(on, side, internal, uv, topo) -> SB.Plan:
    """`SB.build_plan` after the contour test, on a pre-masked edge set (same area elements)."""
    dev = uv.device
    s_idx, e_idx = on.nonzero(as_tuple=True)
    e0, e1 = topo.edges[e_idx, 0], topo.edges[e_idx, 1]
    sd = side[s_idx, e_idx]
    a, b = uv[s_idx, e0], uv[s_idx, e1]
    a1, b1 = uv[s_idx + 1, e0], uv[s_idx + 1, e1]
    ev = b - a
    L = ev.norm(dim=-1).clamp(min=1e-9)
    nrm = -sd[:, None] * torch.stack([-ev[:, 1], ev[:, 0]], -1) / L[:, None]
    da, db = (nrm * (a1 - a)).sum(-1), (nrm * (b1 - b)).sum(-1)
    nt = (L / SB.SPACING).ceil().clamp(1, SB.MAX_NT).long()
    nr = (torch.maximum(da.abs(), db.abs()) / SB.SPACING).ceil().clamp(1, SB.MAX_NR).long()
    n_per = nt * nr
    el = torch.repeat_interleave(torch.arange(len(nt), device=dev), n_per)
    start = torch.cumsum(n_per, 0) - n_per
    k = torch.arange(len(el), device=dev) - start[el]
    it, ir = k // nr[el], k % nr[el]
    t = (it.float() + 0.5) / nt[el].float()
    r = (ir.float() + 0.5) / nr[el].float()
    return SB.Plan(s_idx, e0, e1, sd, internal[s_idx, e_idx], nt, nr, el, t, r)


@torch.no_grad()
def areas(pb: Problem, param, thetas: torch.Tensor, plan: SB.Plan, chunk: int = 16) -> torch.Tensor:
    """(H, 3, S, h, w) swept area of `plan` under the corrections `thetas` (H, P)."""
    outs = []
    for i in range(0, thetas.shape[0], chunk):
        tau, om = param.expand(thetas[i:i + chunk])
        Vh = apply_corr(pb.V, pb.Wr, tau, om)
        uvh = SB.project_src(Vh, pb.K)
        outs.append(SB.sweep_area(uvh, plan, pb.g.geo.grid, pb.win.crop, pb.S))
    return torch.cat(outs)


@torch.no_grad()
def finger_keep(pb: Problem, param, theta0: torch.Tensor, plan_f: Optional[SB.Plan], thr: float = 0.02) -> Optional[torch.Tensor]:
    """(S, h, w) float mask: 0 where the non-palm (finger) contours sweep (blurred, 1-cell dilated), else 1."""
    if plan_f is None:
        return None
    A = areas(pb, param, theta0[None], plan_f)[0].sum(0)                     # (S, h, w)
    A = SB.blur_area(A, pb.g.model.blur)
    m = (A > thr).float()[None]
    m = torch.nn.functional.max_pool2d(m, 3, 1, 1)[0]
    return 1.0 - m


def _rates(pb: Problem, A: torch.Tensor, mod: SB.Model) -> torch.Tensor:
    """`SB.rates` (merged polarity) with the blur applied once to the combined channel (blur is linear): (H, S, h, w)."""
    comb = mod.kappa_sil * (A[:, SB.CH_SIL_POS] + A[:, SB.CH_SIL_NEG]) + mod.kappa_int * A[:, SB.CH_INT]
    comb = SB.blur_area(comb, mod.blur)
    win = pb.win
    bkg = win.bkg if (mod.outlier <= 0 or win.uni is None) else win.bkg + mod.outlier * win.uni
    return comb + bkg[None]


def _model_scaled(pb: Problem, A0: torch.Tensor, blur: Optional[float] = None) -> SB.Model:
    m = pb.g.model
    m = replace(m, blur=m.blur if blur is None else float(blur))
    k = float(SB.profile_kappa(A0, pb.win, m)[0])
    return SB.Model(kappa_sil=m.kappa_sil * k, kappa_int=m.kappa_int * k, signed=False, dispersion=m.dispersion,
                    blur=m.blur, outlier=m.outlier)


def _lam_flat(pb: Problem, A: torch.Tensor, mod: SB.Model, p: int, keep: Optional[torch.Tensor]):
    lam = _rates(pb, A, mod)                                                 # (H, S, h, w)
    N = pb.win.N.sum(0)                                                      # (S, h, w) merged polarity
    if keep is not None:
        lam = lam * keep
        N = N * keep
    return SB.pool(lam, p).flatten(1), SB.pool(N, p).flatten()


@dataclass
class Eval:
    ll: float
    g: torch.Tensor              # (P,) data gradient / dispersion
    F: torch.Tensor              # (P, P) Fisher / dispersion
    kappa_scale: float
    ctx: tuple = ()              # (plan, finger keep, scaled model, pooling) frozen at the base: for `trial_ll`


@torch.no_grad()
def evaluate(pb: Problem, param, theta: torch.Tensor, p: int, central: bool = True, chunk_params: int = 8,
             blur: Optional[float] = None) -> Eval:
    """Poisson log-likelihood, gradient and Fisher of the corrections at `theta` (finite differences of the mean field
    with contour set, visibility and finger mask frozen at `theta`, as `SB.solve`; central by default, forward for the
    cheaper Gauss-Newton iterations)."""
    tau, om = param.expand(theta[None])
    Vb = apply_corr(pb.V, pb.Wr, tau, om)[0]
    plan_p, plan_f, _ = edge_plans(pb, Vb)
    keep = finger_keep(pb, param, theta, plan_f) if pb.evidence == "palm" else None
    A0 = areas(pb, param, theta[None], plan_p)
    mod = _model_scaled(pb, A0, blur)
    lam0, N = _lam_flat(pb, A0, mod, p, keep)
    lam0, N = lam0[0].double(), N.double()
    P = param.P
    eps = param.eps
    rows = []
    per = 2 if central else 1
    for j0 in range(0, P, chunk_params):
        js = list(range(j0, min(P, j0 + chunk_params)))
        th = []
        for j in js:
            e = torch.zeros(P)
            e[j] = eps[j]
            th += [theta + e, theta - e] if central else [theta + e]
        A = areas(pb, param, torch.stack(th), plan_p)
        lam, _ = _lam_flat(pb, A, mod, p, keep)
        lam = lam.view(len(js), per, -1)
        if central:
            d = (lam[:, 0] - lam[:, 1]) / (2 * eps[js, None])
        else:
            d = (lam[:, 0] - lam0.float()[None]) / eps[js, None]
        rows.append(d.double())
    J = torch.cat(rows)                                                      # (P, M)
    lam0c = lam0.clamp(min=1e-12)
    disp = mod.dispersion
    g = (J @ (N / lam0c - 1.0)) / disp
    F = ((J / lam0c[None]) @ J.T) / disp
    ll = float((N * torch.log(lam0c) - lam0).sum() / disp)
    return Eval(ll, g, F, float(mod.kappa_sil / pb.g.model.kappa_sil), (plan_p, keep, mod, p))


@torch.no_grad()
def trial_ll(pb: Problem, param, thetas: torch.Tensor, ctx: tuple) -> torch.Tensor:
    """Log-likelihood / dispersion of candidate corrections: the contour set and visibility are re-planned at every
    candidate (the true objective, as `SB.solve`'s trial test), the finger mask and the scaled model stay those of the
    base so that all candidates are scored on the same cells."""
    _, keep, mod, p = ctx
    out = []
    for th in thetas:
        tau, om = param.expand(th[None])
        Vb = apply_corr(pb.V, pb.Wr, tau, om)[0]
        plan_p, _, _ = edge_plans(pb, Vb)
        A = areas(pb, param, th[None], plan_p)
        lam, N = _lam_flat(pb, A, mod, p, keep)
        lam, N = lam[0].double().clamp(min=1e-12), N.double()
        out.append((N * torch.log(lam) - lam).sum() / mod.dispersion)
    return torch.stack(out)


BACKTRACK = (1.0, 0.5, 0.25, 0.125)             # fixed candidate step scalings (always all evaluated; no early exit)


def safeguarded(pb: Problem, param, theta: torch.Tensor, step: torch.Tensor, ev: Eval, prior: torch.Tensor) -> torch.Tensor:
    """Best of {theta + c step : c in BACKTRACK} U {theta} by the penalised log-likelihood under the frozen base plan.
    The number of trial evaluations is fixed (5), so the compute pattern does not depend on the data."""
    cand = torch.stack([theta + c * step for c in BACKTRACK] + [theta])
    ll = trial_ll(pb, param, cand, ev.ctx)
    pen = 0.5 * ((cand.double() @ prior) * cand.double()).sum(1)
    return cand[int(torch.argmax(ll - pen))]


def _cap_step(step: torch.Tensor, param) -> torch.Tensor:
    s = step.clone()
    for a, b, kind in param.blocks:
        n = float(s[a:b].norm())
        cap = CAP_T if kind == "t" else CAP_R
        if n > cap:
            s[a:b] *= cap / n
    return s


def prior_const() -> torch.Tensor:
    return torch.diag(torch.tensor([1 / PRIOR_SIG_T ** 2] * 3 + [1 / PRIOR_SIG_R ** 2] * 3, dtype=torch.float64))


@dataclass
class Solve:
    theta: torch.Tensor
    F: torch.Tensor              # Fisher at the last iterate (p = 1)
    ll: float
    sd_deg: float                # worst-axis posterior SD of om (weak prior), degrees
    gate_open: bool
    secs: float


def rot_sd_deg(F6: torch.Tensor, prior: torch.Tensor) -> float:
    """Worst-axis posterior SD (deg) of the rotation block of a 6-D (tau, om) problem: invert (F + prior), take the om block."""
    C = torch.linalg.inv(F6 + prior)[3:6, 3:6]
    return float(torch.rad2deg(torch.linalg.eigvalsh(C).max().clamp(min=0).sqrt()))


def align6(pb: Problem, iters: int = K_ITERS, pools: Sequence[int] = POOLS, mus: Sequence[float] = MUS,
           theta0: Optional[torch.Tensor] = None, blurs: Sequence[Optional[float]] = BLURS) -> Solve:
    """Fixed-iteration Levenberg-Marquardt Fisher scoring of the 6-D palm correction (no early stop, no step
    acceptance test: K iterations, fixed damping schedule, trust caps)."""
    t0 = time.time()
    param = ConstParam(pb.T)
    prior = prior_const()
    theta = torch.zeros(6) if theta0 is None else theta0.clone().float()
    last = None
    for it in range(iters):
        ev = evaluate(pb, param, theta, pools[it], central=(it == iters - 1), blur=blurs[it])
        H = ev.F + prior
        rhs = ev.g - prior @ theta.double()
        step = torch.linalg.solve(H + mus[it] * torch.diag(H.diagonal().clamp(min=1e-9)), rhs).float()
        theta = safeguarded(pb, param, theta, _cap_step(step, param), ev, prior)
        last = ev
    # gate statistic from the last Fisher (evaluated at the second-to-last iterate; p = pools[-1] = 1)
    sd = rot_sd_deg(last.F, prior) if pools[iters - 1] == 1 else float("nan")
    return Solve(theta, last.F, last.ll, sd, bool(sd <= SD_GATE_DEG), time.time() - t0)


def recovery(theta: torch.Tensor, b_true: np.ndarray) -> Dict[str, float]:
    """Recovery of the persistent root offset b_true (rad) by the estimated rotation correction om = theta[3:6]
    (R <- Exp(om) R, perfect: Exp(om) Exp(b_true) = I):
        resid_deg = geodesic(Exp(om) Exp(b_true));  rho = 1 - resid / |b_true|  (fraction of the offset removed; negative
        = made worse, overshoot is not recovery);  slope = -<om, b_true> / |b_true|^2 (projection, can exceed 1)."""
    om = theta[3:6].double()
    b = torch.tensor(np.asarray(b_true, np.float64))
    nb = float(b.norm())
    resid = geodesic_deg(so3_log(so3_exp(om) @ so3_exp(b)))
    bdeg = math.degrees(nb)
    return {"rho": float(1.0 - resid / bdeg) if bdeg > 0 else float("nan"), "slope": float(-(om @ b) / nb ** 2) if nb > 0 else float("nan"),
            "resid_deg": resid, "corr_deg": geodesic_deg(om), "b_deg": bdeg}


# ------------------------------------------------------------------------------------------------ Schur / MAP (joint)
def label_stats(subjects: Sequence[Subject], delta_ms: int = 100) -> Dict[str, float]:
    """Robust label-derived temporal prior scales (calibrated on FIT7 GT only): SD (1.4826 MAD) of the per-`delta_ms`
    increment (velocity) and of the second difference over 2 delta (acceleration) of the root rotation (camera-frame
    left increment log(R(t+d) R(t)^T), rad) and of the wrist translation (m), per axis, pooled over axes and recordings."""
    v_r, a_r, v_t, a_t = [], [], [], []
    for sj in subjects:
        for a, b in sj.runs:
            if b - a < 2 * delta_ms + 1:
                continue
            t = np.arange(a, b - 2 * delta_ms, 25)
            R0, R1, R2 = (so3_exp(torch.tensor(sj.pos51[t + k * delta_ms, 3:6])) for k in (0, 1, 2))
            v_r.append(so3_log(R1 @ R0.transpose(-1, -2)).numpy().ravel())
            # second difference as the increment of increments: log(R2 R1^T) - log(R1 R0^T)
            a_r.append((so3_log(R2 @ R1.transpose(-1, -2)) - so3_log(R1 @ R0.transpose(-1, -2))).numpy().ravel())
            p = sj.pos51[:, :3]
            v_t.append((p[t + delta_ms] - p[t]).ravel())
            a_t.append((p[t + 2 * delta_ms] - 2 * p[t + delta_ms] + p[t]).ravel())

    def rsd(x):
        x = np.concatenate(x)
        return float(1.4826 * np.median(np.abs(x - np.median(x))))
    return {"vel_rot": rsd(v_r), "acc_rot": rsd(a_r), "vel_trans": rsd(v_t), "acc_trans": rsd(a_t), "delta_ms": delta_ms}


def knot_count(w_ms: int, delta_ms: int = 100) -> int:
    return int(math.ceil(w_ms / min(delta_ms, w_ms))) + 1


def prior_precision(nk: int, ls: Dict[str, float], regime: str = "full", sigma_b=(0.03, math.radians(10.0)),
                    level=(NOISE_SIG["T"], NOISE_SIG["R"]), level_mult: float = 1.0) -> torch.Tensor:
    """(6 + 6 nk) precision of [b, s_0..s_{nk-1}] (order per knot: tau3, om3):
       b ~ N(0, sigma_b^2) (regime 'noprior' drops it);  s per component: level (sd = level * level_mult; dropped in
       'temporal'), velocity (first differences between knots, sd = label vel_*), acceleration (second differences).
       regimes: full | temporal (no level on s) | noprior (nothing)."""
    P = 6 + 6 * nk
    out = torch.zeros(P, P, dtype=torch.float64)
    if regime != "noprior":
        for c in range(6):
            out[c, c] = 1.0 / (sigma_b[0] if c < 3 else sigma_b[1]) ** 2
    if regime == "noprior":
        return out
    for c in range(6):
        trans = c < 3
        s_lvl = level[0 if trans else 1] * level_mult
        s_v = ls["vel_trans" if trans else "vel_rot"]
        s_a = ls["acc_trans" if trans else "acc_rot"]
        L = torch.zeros(nk, nk, dtype=torch.float64)
        if regime == "full":
            L += torch.eye(nk, dtype=torch.float64) / s_lvl ** 2
        D1 = torch.zeros(nk - 1, nk, dtype=torch.float64)
        D1[torch.arange(nk - 1), torch.arange(nk - 1)] = -1
        D1[torch.arange(nk - 1), torch.arange(1, nk)] = 1
        L += D1.T @ D1 / s_v ** 2
        if nk >= 3:
            D2 = torch.zeros(nk - 2, nk, dtype=torch.float64)
            i = torch.arange(nk - 2)
            D2[i, i], D2[i, i + 1], D2[i, i + 2] = 1, -2, 1
            L += D2.T @ D2 / s_a ** 2
        idx = torch.tensor([6 * (1 + k) + c for k in range(nk)])
        out[idx[:, None], idx[None, :]] += L
    return out


def schur_b(F: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """S_b = (F_bb + P_bb) - (F_bs + P_bs) (F_ss + P_ss)^-1 (F_sb + P_sb) for b = first 6 coordinates (the prior is block
    diagonal between b and s)."""
    H = F + prior
    Hbb, Hbs, Hss = H[:6, :6], H[:6, 6:], H[6:, 6:]
    return Hbb - Hbs @ torch.linalg.solve(Hss + 1e-12 * torch.eye(Hss.shape[0], dtype=Hss.dtype), Hbs.T)


def posterior_sd_deg(Sb: torch.Tensor) -> float:
    """Worst-axis posterior SD (deg) of the bias rotation: rotation block of Sb^-1 (translation marginalised)."""
    C = torch.linalg.inv(Sb + 1e-12 * torch.eye(6, dtype=Sb.dtype))[3:6, 3:6]
    return float(torch.rad2deg(torch.linalg.eigvalsh(C).max().clamp(min=0).sqrt()))


def map_joint(pb: Problem, nk: int, prior: torch.Tensor, iters: int = 4, pools: Sequence[int] = (8, 4, 2, 1),
              mus: Sequence[float] = (1.0, 0.5, 0.3, 0.2)) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Fixed-iteration MAP of (b, s) under `prior` from theta = 0; returns (theta, Fisher at the last iterate, secs)."""
    t0 = time.time()
    param = KnotParam(pb.T, nk)
    theta = torch.zeros(param.P)
    last = None
    for it in range(iters):
        ev = evaluate(pb, param, theta, pools[it])
        H = ev.F + prior
        rhs = ev.g - prior @ theta.double()
        step = torch.linalg.solve(H + mus[it] * torch.diag(H.diagonal().clamp(min=1e-9)), rhs).float()
        theta = safeguarded(pb, param, theta, _cap_step(step, param), ev, prior)
        last = ev
    return theta, last.F, time.time() - t0


FISHER_EPS_SCALE = 0.1      # finite-difference steps of the Schur Fisher: 0.1 x (1 mm, 2e-3 rad); the exact aliasing b == sum_k s_k
                            # holds up to the FD error, which is ~5e-4 of |F_bb| at this scale (1e-3 at 0.25, 7e-2 at 1.0)


def fisher_joint(pb: Problem, nk: int, p: int = 1) -> torch.Tensor:
    """Fisher of the (b, s) parameters at the problem's own trajectory (theta = 0), the same likelihood."""
    param = KnotParam(pb.T, nk)
    param.eps = param.eps * FISHER_EPS_SCALE
    return evaluate(pb, param, torch.zeros(param.P), p).F


# ------------------------------------------------------------------------------------------------ sigma* covariate
def sigma_star(seq: str) -> Optional[float]:
    try:
        r = json.loads(P0V.read_text())["recordings"][seq]
        return float(r["sigma_star_q0_median"])
    except Exception:
        return None
