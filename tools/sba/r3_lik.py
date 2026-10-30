#!/usr/bin/env python3
"""SBA round 3 likelihood utilities (docs/EXPERIMENT_RECORDS.md#sba-r3-prereg-20261022).

The P4 emission model (outputs/sba/p4/fit_raw_b1.json, as tools/sba/q0_landscape.py loads it) evaluated per hypothesis
and scale gauge sigma (mesh about the wrist at fixed depth), deposited per part group:

    G_g = sum_{c in g} theta_c F_c            (6 groups, the expected hand events per group at unit multipliers)

Nuisances per hypothesis -- group multipliers m_g >= 0 and the outlier rate eta >= 0 -- are profiled by `newton`, a
batched copy of q0_landscape.profile (same projected Newton, float64, 40 iterations, same 1e-9 ridge). Variants on the
same features: 'mg' (6 multipliers + eta; Q0 V2), 'm1' (one multiplier + eta), 'geo' (material patch channels removed).
Chained windows concatenate their event cells and share the nuisances (`chain_problem`).
"""
from __future__ import annotations

import json
import sys
import zlib
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sba import common as CM  # noqa: E402
from semkine import sba_emission as EM  # noqa: E402

P4 = CM.OUT / "p4"
#: scale-gauge grid (Q0's 0.90 ... 1.04 is the sub-grid SIG[2:10]); SIG[7] == 1.00
SIG = tuple(round(0.86 + 0.02 * k, 2) for k in range(11))
Q0_GRID = tuple(range(2, 10))
GPUS = ("2", "3", "4", "5", "6", "7")


def seed_of(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


class Lik:
    """P4 group features of a trajectory on a window's crop."""

    def __init__(self, ctx):
        fit = json.loads((P4 / "fit_raw_b1.json").read_text())
        dev = ctx.dev
        self.ctx = ctx
        self.patch = torch.from_numpy(np.load(P4 / "patches.npy")).to(dev)
        self.n_p = int(fit["meta"]["n_patches"])
        self.blur = float(fit["meta"]["blur"])
        theta = torch.tensor(fit["selected"]["theta"], dtype=torch.float32, device=dev)
        # channel -> part group membership, exactly as tools/sba/q0_landscape.py::Q0.__init__
        grp_edge = ctx.geo.groups[ctx.geo.topo.edges[:, 0]]
        maj = torch.zeros(self.n_p, dtype=torch.long, device=dev)
        for i in range(self.n_p):
            g = grp_edge[self.patch == i]
            maj[i] = torch.bincount(g, minlength=6).argmax() if len(g) else 0
        n_ch = EM.n_channels(self.n_p)
        member = torch.zeros(6, n_ch, device=dev)
        for g in range(6):
            member[g, g] = 1.0
            member[g, 6 + g] = 1.0
            member[g, 12:][maj == g] = 1.0
        self.theta, self.member = theta, member
        self.W = member * theta[None]                                          # (6, n_ch)
        self.W_geo = self.W.clone()
        self.W_geo[:, 12:] = 0.0

    @torch.no_grad()
    def groups(self, sd, win, a: torch.Tensor, b: torch.Tensor, sigma: float):
        """(G (6, M), G_geo (6, M)) of the trajectory a -> b at scale sigma; M = S * h * w."""
        F = EM.features(self.ctx.geo, win, a, b, sd.betas, sd.K, self.patch, self.n_p, self.blur, sigma).flatten(1)
        return self.W @ F, self.W_geo @ F


# ------------------------------------------------------------------------------------------------ profiling
#: a profile whose last Newton step is larger than this (relative to max(1, |w|)) has not converged: its
#: log-likelihood is reported as NaN (a failure, scored as a tie and counted in the failure rate)
CONV_TOL = 1e-6


def newton(X: torch.Tensor, Xt: torch.Tensor, B: torch.Tensor, Bt, N: torch.Tensor, iters: int = 40,
           tol: float = 1e-12, nan_unconverged: bool = False):
    """Profile max_{w >= 0} sum N log(w.X + B) - Xt.w - Bt of P problems -> (ll (P,), w (P, k)), float64.
    X (P, k, n), Xt (P, k), B (n,), Bt float, N (n,).

    Same model, start (w = 0.5) and Hessian (with the 1e-9 ridge) as tools/sba/q0_landscape.py::profile, made
    convergent: weights at the bound with a negative gradient are held at zero (active set) and the projected Newton
    step is halved until the (concave) objective does not decrease. Where q0's plain projected Newton converges both
    reach the same optimum; where it oscillates (weights pinned at zero, e.g. wrong-content donor events) this one still
    converges. Stops once every problem's relative step is below `tol`; with `nan_unconverged` problems whose last
    relative step exceeds CONV_TOL get ll = NaN."""
    X, Xt, B, N = X.double(), Xt.double(), B.double(), N.double()
    P, k, _ = X.shape
    eye = torch.eye(k, dtype=torch.float64, device=X.device)
    w = torch.full((P, k), 0.5, dtype=torch.float64, device=X.device)

    def obj(w_):
        lam_ = (torch.einsum("pk,pkn->pn", w_, X) + B).clamp(min=1e-12)
        return (N * lam_.log()).sum(-1) - (Xt * w_).sum(-1), lam_

    f, lam = obj(w)
    rel = torch.zeros(P, dtype=torch.float64, device=X.device)
    for _ in range(iters):
        g = torch.einsum("pkn,pn->pk", X, N / lam) - Xt
        H = -torch.einsum("pkn,pjn->pkj", X * (N / lam ** 2)[:, None], X) - 1e-9 * eye
        free = ~((w <= 0.0) & (g < 0.0))
        ff = (free[:, :, None] & free[:, None, :]).to(H.dtype)
        Hm = H * ff - eye * (~free).to(H.dtype)[:, :, None]
        d = -torch.linalg.solve(Hm, (g * free)[..., None])[..., 0]             # ascent direction (H < 0)
        t = torch.ones(P, dtype=torch.float64, device=X.device)
        for _ in range(30):
            w_new = (w + t[:, None] * d).clamp(min=0.0)
            f_new, lam_new = obj(w_new)
            ok = f_new >= f - 1e-12 * f.abs().clamp(min=1.0)
            if bool(ok.all()):
                break
            t = torch.where(ok, t, 0.5 * t)
        keep = f_new >= f - 1e-12 * f.abs().clamp(min=1.0)
        w_new = torch.where(keep[:, None], w_new, w)
        f_new = torch.where(keep, f_new, f)
        lam = torch.where(keep[:, None], lam_new, lam)
        rel = (w_new - w).abs().amax(1) / w_new.abs().amax(1).clamp(min=1.0)
        w, f = w_new, f_new
        if float(rel.max()) <= tol:
            break
    ll = f - float(Bt)
    if nan_unconverged:
        ll = torch.where(rel > CONV_TOL, torch.full_like(ll, float("nan")), ll)
    return ll, w


def counts(win) -> torch.Tensor:
    return win.N.sum(0).flatten()


def problem(G: torch.Tensor, win, variant: str = "mg", N: torch.Tensor = None):
    """Newton inputs of hypotheses G (P, 6, M) on one window. variant 'mg' -> [G_1..G_6, U]; 'm1' -> [sum G, U].
    N: optional counts (M,) replacing the window's (Poisson twin)."""
    Nf = counts(win) if N is None else N
    cell = Nf > 0
    U = win.uni.flatten()
    Bf = win.bkg.flatten()
    Gx = G if variant != "m1" else G.sum(1, keepdim=True)
    P = Gx.shape[0]
    X = torch.cat([Gx[:, :, cell], U[cell].expand(P, 1, -1)], 1)
    Xt = torch.cat([Gx.sum(-1), U.sum().expand(P, 1)], 1)
    return X, Xt, Bf[cell], float(Bf.sum()), Nf[cell]


def chain_problem(Gs: Sequence[torch.Tensor], wins: Sequence, variant: str = "mg"):
    """Several windows sharing one set of nuisances: event cells concatenated, totals summed."""
    parts = [problem(G, w, variant) for G, w in zip(Gs, wins)]
    X = torch.cat([p[0] for p in parts], -1)
    Xt = sum(p[1] for p in parts)
    B = torch.cat([p[2] for p in parts])
    Bt = sum(p[3] for p in parts)
    N = torch.cat([p[4] for p in parts])
    return X, Xt, B, Bt, N


def dense_rate(G: torch.Tensor, win, w: torch.Tensor) -> torch.Tensor:
    """lambda (P, M) of hypotheses G (P, 6, M) at nuisances w (7,) = [m_1..m_6, eta]."""
    w = w.to(G.dtype)
    return torch.einsum("g,pgm->pm", w[:6], G) + w[6] * win.uni.flatten() + win.bkg.flatten()


def kl_poisson(lam_ref: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """sum over cells of KL(Poisson(lam_ref) || Poisson(lam)); lam_ref (M,), lam (P, M)."""
    r = lam_ref.double().clamp(min=1e-12)
    q = lam.double().clamp(min=1e-12)
    return (r * (r.log() - q.log()) - r + q).sum(-1)


# ------------------------------------------------------------------------------------------------ events
def thin(ev_ms, p: float, seed: int):
    """Uniform Bernoulli thinning of (ev, ms) with keep probability p (deterministic seed)."""
    ev, ms = ev_ms
    keep = np.random.default_rng(seed).random(len(ms)) < p
    return ev[keep], ms[keep]


def shuffle_times(ev_ms, seed: int):
    """Permute the timestamps among the events (positions and polarities kept; per-ms counts kept)."""
    ev, ms = ev_ms
    return ev, ms[np.random.default_rng(seed).permutation(len(ms))]


def take(ev_ms, m_prev: int, m1: int):
    """Events with ms in (m_prev, m1]."""
    ev, ms = ev_ms
    k = (ms > m_prev) & (ms <= m1)
    return ev[k], ms[k]


# ------------------------------------------------------------------------------------------------ sigma arms
def refine_max(y: np.ndarray, grid: Sequence[float] = SIG) -> Tuple[float, float, bool]:
    """(max, argmax sigma, edge) of y over the grid with a 3-point parabolic refinement at the discrete argmax."""
    y = np.asarray(y, np.float64)
    if not np.all(np.isfinite(y)):
        return float("nan"), float("nan"), False
    i = int(np.argmax(y))
    h = grid[1] - grid[0]
    if i == 0 or i == len(y) - 1:
        return float(y[i]), float(grid[i]), True
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    if den >= 0:
        return float(b), float(grid[i]), False
    d = 0.5 * (a - c) / den
    return float(b - 0.25 * (a - c) * d), float(grid[i] + d * h), False


def interp(y: np.ndarray, s: float, grid: Sequence[float] = SIG) -> float:
    """Quadratic (3-point Lagrange) interpolation of y on the grid at s (clamped to the grid)."""
    y = np.asarray(y, np.float64)
    g = np.asarray(grid)
    s = float(np.clip(s, g[0], g[-1]))
    i = int(np.clip(np.argmin(np.abs(g - s)), 1, len(g) - 2))
    x = g[i - 1:i + 2]
    v = y[i - 1:i + 2]
    out = 0.0
    for j in range(3):
        L = 1.0
        for m in range(3):
            if m != j:
                L *= (s - x[m]) / (x[j] - x[m])
        out += v[j] * L
    return float(out)
