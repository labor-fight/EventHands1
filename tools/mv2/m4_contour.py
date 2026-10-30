#!/usr/bin/env python3
"""MV2 / M4 -- contour-normal 1-D measurement (CNM), R3 probe set (misspecification-aware real-event measurement).

Draft pre-registration: tools/mv2/prereg_m4.md.  CPU only, zero training, GT previous state.  Reads FIT7 + SELECT base
recordings only (never zgz).  Three R3 probes share one extraction of real event / FK-contour pairs:

  E5  (P3-E5-R2)  per-class pseudo-true offset b_c, delta_sigma = b / R, sandwich A^-1 B A^-1 and VIF
  E4  (P2-E4-R2)  contour-normal hypothesis volume: mode-peak bias, within-packet jitter, 3-family nuisance
                  decomposition (translation, scale about the palm centre, latency x velocity), peakiness vs nulls
  E1  (P3-E1-R1)  class-biased 1-D PDA gate: b_c stability FIT7 -> SELECT, de-biasing, NIS, position recovery after a
                  known image-plane shift (PDA vs nearest-contour-point baseline), oracle-visibility control
  E8  (P4-E8-R2)  descriptive: scale-about-wrist coefficient of the same decomposition (sigma* 0.93-1.01 range, CV)

Geometry.  Occluding edges, their visibility and the material class (visible non-contour edges) are SBA's
(`semkine.sba.contour_entries`, `semkine.sba_emission.material_entries`); the silhouette / internal decision is taken on the
exact triangle raster of `semkine.kssf` (pixel 1.5 px outward of the edge midpoint: background -> silhouette, a hand surface
>= 10 mm farther -> internal, anything else dropped).  Event cells are placed at their continuous render-plane centre through
the 3/3/2 bin LUT (`semkine.sba.make_grid`, mean phase 0.4375 px).  Offsets d are signed distances along the outward contour
normal in render pixels (240 x 180).  Every contour point is one mesh edge (midpoint); its search strip is +-1 px along the
contour and [-16.5, 16.5) px along the normal.

    python tools/mv2/m4_contour.py --smoke
    python tools/mv2/m4_contour.py --out outputs/mv2/m4_contour [--stage extract|analyze|all] [--workers N]
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import pickle
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mv2 import common as C  # noqa: E402

# ------------------------------------------------------------------------------------------------ constants
SEED = 20261026
SCALE = 240.0 / 640.0                  # source px -> render px
CLS = ("sil", "int", "mat")
SIL, INT, MAT = 0, 1, 2
VIEWS = ("V200", "V50", "Vc50")
NULLS = ("real", "tshuf_layer", "tshuf_event", "pol_flip", "pol_perm", "cls_shuf", "wrongwin")
NSLAB, SLAB_MS, TAU_LO = 8, 25.0, -150.0      # tau = event time - (end - 49) ms; W200: [-150, 50)
VIEW_SLABS = {"V200": tuple(range(8)), "V50": (6, 7), "Vc50": (5, 6)}
BIN, RMAX, NB, D0 = 0.25, 16.5, 132, -16.5    # fine offset bins; 1 px bins = groups of 4 (centres -16 .. 16)
TW = 1.0                                       # strip half-width along the contour (render px)
CELL, NABIN = 16.0, 8                          # segment = (class, 16 px cell, 45 deg normal bin)
EVERY = 4                                      # endpoint stride (x 50 ms = 200 ms: windows do not overlap)
SHIFT_MM = (5, 10, 14, 19, 25)
SHIFT_PRIMARY = 14
SHIFT_FIT7_EVERY = 4                           # FIT7 samples that also get the shift conditions
NULL_SLOPE_MAX = 0.25                          # scattered-event control: slope CI must reach into +-this
DISP_PX = 24.0                                 # scattered-event null: each event moved this far in a random direction
N_SEG_MIN = 50                                 # pairs in a segment for it to count
N_SLAB_MIN = 25
N_UNIT_MIN = 30                                # packets with pairs for a class to be analysed in a recording
N_FIT_MIN = 5000                               # pairs to fit a class mixture
N_JIT_PKT_MIN = 30                             # packets with >= 3 qualified segments for a jitter number
BLOCK = 10                                     # bootstrap block (consecutive samples of a recording)
CLUTTER_FLOOR = 1e-4
THETA = np.arange(-12.0, 12.0 + 1e-9, 0.05)    # location grid of the pseudo-true offset (px)
DELTA = np.arange(-16.0, 16.0 + 1e-9, 0.25)    # shift grid of the PDA posterior (px)
CENTRES = D0 + BIN * (np.arange(NB) + 0.5)
PALM_JOINTS = (0, 5, 9, 13, 17)
T_REF_OFFSET = 49                              # tau = ms + tsub/1000 - (end - 49)
GEOM_PROBE_PX = 1.5
INWARD_PX = 1.0
GEOM_MIN_FRAC = 0.85                           # tool-validity: contour class agrees with the exact raster
Z_INT_MARGIN = 0.010

# gate constants (all from the plan / R3 entries; see prereg_m4.md)
GATE = dict(b_enter=2.0, dsig_enter=0.03, vif_enter=3.0, b_kill=4.0, dsig_kill=0.06,
            jitter=2.0, bias_r2=4.0, r2=0.5, peak_ratio=3.0,
            drift=1.5, drift_kill=3.0, debias_red=0.5, floor_kill=0.8)
RECORDING_RULE = "all"


# ------------------------------------------------------------------------------------------------ small utilities
def fine_index(d: np.ndarray) -> np.ndarray:
    return np.clip(np.floor((np.asarray(d, np.float64) - D0) / BIN).astype(np.int64), 0, NB - 1)


def h1px(h: np.ndarray) -> np.ndarray:
    """(..., 132) fine histogram -> (..., 33) 1-px histogram (bin j centred on j - 16)."""
    return np.asarray(h).reshape(h.shape[:-1] + (NB // 4, 4)).sum(-1)


def smooth3(x: np.ndarray) -> np.ndarray:
    xp = np.concatenate([x[..., :1], x, x[..., -1:]], -1).astype(np.float64)
    return 0.25 * xp[..., :-2] + 0.5 * xp[..., 1:-1] + 0.25 * xp[..., 2:]


def peak_of(h1: np.ndarray) -> Tuple[float, float]:
    """Mode (px, parabolic sub-bin) and peakiness (max / median of the [1 2 1]-smoothed 33-bin profile)."""
    s = smooth3(np.asarray(h1, np.float64))
    i = int(np.argmax(s))
    off = 0.0
    if 0 < i < len(s) - 1:
        den = s[i - 1] - 2.0 * s[i] + s[i + 1]
        if den < -1e-12:
            off = float(np.clip(0.5 * (s[i - 1] - s[i + 1]) / den, -0.5, 0.5))
    return float(i - 16) + off, float(s.max() / max(float(np.median(s)), 1.0))


def median_from_hist(h: np.ndarray) -> float:
    h = np.asarray(h, np.float64)
    n = h.sum()
    if n <= 0:
        return float("nan")
    cs = np.cumsum(h)
    k = int(np.searchsorted(cs, 0.5 * n))
    prev = cs[k - 1] if k > 0 else 0.0
    frac = (0.5 * n - prev) / max(h[k], 1e-12)
    return float(D0 + BIN * (k + frac))


def rng_for(*key) -> np.random.Generator:
    return np.random.default_rng([SEED] + [int(zlib.crc32(str(k).encode())) if isinstance(k, str) else int(k) for k in key])


# ------------------------------------------------------------------------------------------------ mixture / PDA
def logp_theta(theta: np.ndarray, s: float, beta: float, q: np.ndarray) -> np.ndarray:
    """log P_k(theta): bin probabilities of (1 - beta) truncated-Gaussian(theta, s) + beta q over the 132 fine bins."""
    g = np.exp(-0.5 * ((CENTRES[None, :] - np.asarray(theta, np.float64)[:, None]) / s) ** 2)
    g /= np.maximum(g.sum(1, keepdims=True), 1e-300)
    return np.log((1.0 - beta) * g + beta * q[None, :])


def clutter_table(h_wrong: np.ndarray) -> np.ndarray:
    """Empirical non-contour offset distribution (wrong-window events), smoothed over +-1.5 px, floored, normalised."""
    k = np.array([1, 2, 3, 4, 3, 2, 1], np.float64)
    x = np.convolve(np.asarray(h_wrong, np.float64), k / k.sum(), mode="same")
    x = x / max(x.sum(), 1e-12) + CLUTTER_FLOOR / NB
    return x / x.sum()


def fit_mixture(h: np.ndarray, q: np.ndarray) -> Optional[dict]:
    """Binned maximum likelihood of (theta, s, beta) with the clutter table q fixed (multi-start L-BFGS-B)."""
    from scipy.optimize import minimize
    h = np.asarray(h, np.float64)
    n = h.sum()
    if n < N_FIT_MIN:
        return None

    def nll(x):
        return -float(logp_theta(np.array([x[0]]), x[1], x[2], q)[0] @ h) / n

    best = None
    for t0 in (-4.0, 0.0, 4.0):
        r = minimize(nll, np.array([t0, 2.0, 0.7]), method="L-BFGS-B",
                     bounds=[(-12.0, 12.0), (0.5, 8.0), (0.001, 0.999)])
        if best is None or r.fun < best.fun:
            best = r
    th, sg, be = float(best.x[0]), float(best.x[1]), float(best.x[2])
    return dict(theta=th, s=sg, beta=be, n=float(n), nll=float(best.fun),
                at_bound=bool(abs(th) > 11.95 or sg < 0.51 or sg > 7.95 or be < 0.0015 or be > 0.9985))


def mle_location(Hu: np.ndarray, s: float, beta: float, q: np.ndarray) -> dict:
    """Pseudo-true offset of the frozen mixture: argmax of the pooled log-likelihood over THETA (+ parabola), the
    Hessian A and clustered score variance B over the units (rows of Hu), sandwich variance and VIF = B / A."""
    Hu = np.asarray(Hu, np.float64)
    U = Hu.shape[0]
    LP = logp_theta(THETA, s, beta, q)
    ll = LP @ Hu.sum(0)
    t = int(np.argmax(ll))
    th = float(THETA[t])
    if 0 < t < len(THETA) - 1:
        den = ll[t - 1] - 2 * ll[t] + ll[t + 1]
        if den < -1e-12:
            th += float(np.clip(0.5 * (ll[t - 1] - ll[t + 1]) / den, -0.5, 0.5)) * (THETA[1] - THETA[0])
    eps, hh = 0.02, 0.1
    lp = logp_theta(np.array([th - eps, th + eps, th - hh, th, th + hh]), s, beta, q)
    psi = (lp[1] - lp[0]) / (2 * eps)
    Psi = Hu @ psi
    tot = Hu.sum(0)
    ll3 = lp[[2, 3, 4]] @ tot
    A = float(-(ll3[0] - 2 * ll3[1] + ll3[2]) / hh ** 2 / U)
    Bv = float(np.mean(Psi ** 2))
    return dict(theta=th, A=A, B=Bv, vif=Bv / A if A > 1e-12 else float("inf"),
                se=math.sqrt(Bv / (A * A * U)) if A > 1e-12 else float("inf"),
                at_bound=bool(t in (0, len(THETA) - 1)), n_units=int(U))


def pda_posterior(Hseg: np.ndarray, s: float, beta: float, q: np.ndarray, b: float) -> Tuple[np.ndarray, np.ndarray]:
    """1-D PDA posterior of the contour shift delta (flat prior on DELTA) for segment histograms Hseg (S, 132): events
    appear at d = delta + b + noise (Gaussian s) or as clutter (table q, weight beta).  Returns mean, variance (S,)."""
    LP = logp_theta(DELTA + b, s, beta, q)                       # (129, 132)
    ll = np.asarray(Hseg, np.float64) @ LP.T                     # (S, 129)
    ll -= ll.max(1, keepdims=True)
    p = np.exp(ll)
    p /= p.sum(1, keepdims=True)
    mu = p @ DELTA
    var = np.maximum(p @ DELTA ** 2 - mu ** 2, 1e-9)
    return mu, var


def nis_band(n: int, level: float = 0.95) -> Tuple[float, float]:
    """Central `level` band of the median of n iid chi2(1) draws (order-statistic law of the sample median)."""
    from scipy import stats
    n = max(int(n), 1)
    k = n // 2
    a = k + 1
    b = n - k if n % 2 == 1 else k + 1
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return (float(stats.chi2.ppf(stats.beta.ppf(lo, a, b), 1)), float(stats.chi2.ppf(stats.beta.ppf(hi, a, b), 1)))


def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def r2(y: np.ndarray, yhat: np.ndarray) -> Tuple[float, float]:
    """(uncentred R2 = explained fraction of the bias energy, centred R2)."""
    ss = float(np.sum((y - yhat) ** 2))
    u = 1.0 - ss / max(float(np.sum(y ** 2)), 1e-12)
    c = 1.0 - ss / max(float(np.sum((y - y.mean()) ** 2)), 1e-12)
    return u, c


def slope_from_sums(n, sx, sy, sxx, sxy) -> float:
    den = n * sxx - sx * sx
    return float((n * sxy - sx * sy) / den) if den > 1e-12 else float("nan")


# ------------------------------------------------------------------------------------------------ geometry
class Geo:
    """FK (the frozen S39 model's MANO layer), SBA contour classes, exact raster, bin-phase LUT. CPU, one per process."""

    def __init__(self):
        from semkine import sba as SB
        from semkine import sba_emission as SE
        from semkine.kssf import KSSF
        torch.set_num_threads(1)
        self.dev = torch.device("cpu")
        self.model, _ = C.load_s39(3407, self.dev)
        self.SB, self.SE = SB, SE
        self.geo = SB.Geometry(self.model.mano, self.dev)
        self.kssf = KSSF(self.model.mano)
        self.topo = self.geo.topo
        self.edges = self.topo.edges.numpy().astype(np.int64)
        self.groups = self.geo.groups.numpy().astype(np.int64)
        self.lut_x = (self.geo.grid.cx_src.numpy().astype(np.float64) * SCALE).astype(np.float32)   # (240,) render x
        self.lut_y = (self.geo.grid.cy_src.numpy().astype(np.float64) * SCALE).astype(np.float32)   # (180,)

    # -- one sample's geometry: prev (GT end-50) and end (GT end)
    @torch.no_grad()
    def scene(self, prev51: np.ndarray, gt51: np.ndarray, betas: np.ndarray, K: np.ndarray) -> dict:
        P = torch.from_numpy(np.stack([prev51, gt51]).astype(np.float32))
        Kt = torch.from_numpy(np.asarray(K, np.float32))
        fk = C.fk_surface(self.model, P, torch.from_numpy(np.asarray(betas, np.float32)), Kt)
        verts, joints = fk["verts"], fk["joints"]
        uv_src = self.SB.project_src(verts, Kt)
        j = joints[0, list(PALM_JOINTS)].mean(0, keepdim=True)
        c_palm = self.SB.project_src(j, Kt)[0].numpy().astype(np.float64) * SCALE
        c_wrist = self.SB.project_src(joints[0, :1], Kt)[0].numpy().astype(np.float64) * SCALE
        return dict(verts=verts, K=Kt, uv=uv_src.numpy().astype(np.float64), c_palm=c_palm, c_wrist=c_wrist,
                    z=verts[..., 2].numpy().astype(np.float64))

    @torch.no_grad()
    def edge_sets(self, verts_b: torch.Tensor, uv_src_b: np.ndarray, K: torch.Tensor):
        """Classes of the mesh edges of one geometry: -1 none, 0 silhouette, 1 internal, 2 material; plus the outward
        side of every edge and a dict with the exact raster (mask, depth) and SBA's own class for the agreement count.
        Occluding edges and their visibility are SBA's (`contour_entries`, part-group vertex z-buffer); the class is
        decided on the exact triangle raster: the pixel `GEOM_PROBE_PX` outward of the edge midpoint is background
        (silhouette) or a hand surface at least `Z_INT_MARGIN` farther than the edge (internal); anything else
        (a hand surface at about the same depth, or nearer) is ambiguous and dropped."""
        uvt = torch.from_numpy(uv_src_b.astype(np.float32))[None]
        on, side, internal = self.SB.contour_entries(verts_b[None], uvt, self.topo, self.geo.grid, self.geo.groups,
                                                     self.geo.n_groups)
        mat = self.SE.material_entries(verts_b[None], uvt, self.topo, self.geo.groups, self.geo.n_groups)
        on, side, internal, mat = on[0].numpy(), side[0].numpy().astype(np.float64), internal[0].numpy(), mat[0].numpy()
        mask, depth = self.raster(verts_b, K)
        e0, e1 = self.edges[:, 0], self.edges[:, 1]
        a, b = uv_src_b[e0], uv_src_b[e1]
        ev = b - a
        L = np.maximum(np.linalg.norm(ev, axis=1), 1e-9)
        nrm = -side[:, None] * np.stack([-ev[:, 1], ev[:, 0]], 1) / L[:, None]
        q = 0.5 * (a + b) * SCALE + GEOM_PROBE_PX * nrm
        qx, qy = np.round(q[:, 0]).astype(np.int64), np.round(q[:, 1]).astype(np.int64)
        inb = (qx >= 0) & (qx < 240) & (qy >= 0) & (qy < 180)
        qx, qy = np.clip(qx, 0, 239), np.clip(qy, 0, 179)
        inh, zq = mask[qy, qx], depth[qy, qx]
        ze = 0.5 * (verts_b[e0, 2].numpy() + verts_b[e1, 2].numpy()).astype(np.float64)
        cls = np.full(len(on), -1, np.int64)
        cls[mat & ~on] = MAT
        cls[on & inb & ~inh] = SIL
        cls[on & inb & inh & (zq > ze + Z_INT_MARGIN)] = INT
        sba = np.full(len(on), -1, np.int64)
        sba[on & ~internal] = SIL
        sba[on & internal] = INT
        return cls, side, dict(mask=mask, depth=depth, sba=sba)

    def points(self, uv_s, uv_ref, uv_end, cls_e, side_e, c_palm, c_wrist, zv, include_mat=True) -> dict:
        """Contour points (one per selected mesh edge, at its midpoint) with outward image normals.
        uv_s: positions used for the contour (source px); uv_ref: GT previous positions; uv_end: GT end positions."""
        keep = (cls_e >= 0) & (include_mat | (cls_e != MAT))
        eid = np.nonzero(keep)[0]
        e0, e1 = self.edges[eid, 0], self.edges[eid, 1]
        a, b = uv_s[e0], uv_s[e1]
        ev = b - a
        L = np.maximum(np.linalg.norm(ev, axis=1), 1e-9)
        rot = np.stack([-ev[:, 1], ev[:, 0]], 1) / L[:, None]
        nrm = -side_e[eid][:, None] * rot
        ecls = cls_e[eid]
        mid = 0.5 * (a + b) * SCALE
        mm = ecls == MAT
        if mm.any():                                           # material: orientation ambiguous -> radially outward
            flip = ((mid[mm] - c_palm) * rot[mm]).sum(1) < 0
            nrm[mm] = np.where(flip[:, None], -rot[mm], rot[mm])
        for c in (SIL, INT):                                   # vertex-smoothed normals within a contour class
            m = ecls == c
            if not m.any():
                continue
            acc = np.zeros((uv_s.shape[0], 2))
            np.add.at(acc, e0[m], nrm[m] * L[m, None])
            np.add.at(acc, e1[m], nrm[m] * L[m, None])
            sm = acc[e0[m]] + acc[e1[m]]
            nn = np.linalg.norm(sm, axis=1)
            ok = nn > 1e-9
            tmp = nrm[m].copy()
            tmp[ok] = sm[ok] / nn[ok, None]
            nrm[m] = tmp
        dend = uv_end - uv_ref
        dsh = uv_s - uv_ref
        vn = ((0.5 * (dend[e0] + dend[e1])) * SCALE * nrm).sum(1) / 50.0
        sn = ((0.5 * (dsh[e0] + dsh[e1])) * SCALE * nrm).sum(1)
        return dict(eid=eid, pos=mid, nrm=nrm, cls=ecls.astype(np.int64), grp=self.groups[e0], vn=vn, sn=sn,
                    rad=((mid - c_palm) * nrm).sum(1), radw=((mid - c_wrist) * nrm).sum(1),
                    z=0.5 * (zv[e0] + zv[e1]))

    def raster(self, verts_b: torch.Tensor, K: torch.Tensor):
        with torch.no_grad():
            f = self.kssf.rasterize(verts_b[None], K[None])
        return f.visibility[0].numpy() > 0, f.depth[0].numpy().astype(np.float64)


def segment_ids(pos: np.ndarray, nrm: np.ndarray, cls: np.ndarray) -> Tuple[np.ndarray, int]:
    """Segment of every point: (class, 16 px cell, 45 deg bin of the normal)."""
    cx = np.floor(pos[:, 0] / CELL).astype(np.int64) + 8
    cy = np.floor(pos[:, 1] / CELL).astype(np.int64) + 8
    ab = np.floor((np.arctan2(nrm[:, 1], nrm[:, 0]) + math.pi) / (2 * math.pi / NABIN)).astype(np.int64) % NABIN
    key = ((cls * 128 + cx) * 128 + cy) * NABIN + ab
    _, inv = np.unique(key, return_inverse=True)
    return inv.astype(np.int64), int(inv.max()) + 1 if len(inv) else 0


def segment_meta(pts: dict, seg: np.ndarray, S: int) -> np.ndarray:
    """(S, 10): class, group, mean nx, ny, rad_palm, rad_wrist, vn, sn, n points, mean pos x."""
    cnt = np.bincount(seg, minlength=S).astype(np.float64)

    def mean(x):
        return np.bincount(seg, weights=x, minlength=S) / np.maximum(cnt, 1)

    nx, ny = mean(pts["nrm"][:, 0]), mean(pts["nrm"][:, 1])
    nn = np.maximum(np.hypot(nx, ny), 1e-9)
    _, first = np.unique(seg, return_index=True)
    return np.stack([pts["cls"][first], pts["grp"][first], nx / nn, ny / nn, mean(pts["rad"]), mean(pts["radw"]),
                     mean(pts["vn"]), mean(pts["sn"]), cnt, mean(pts["pos"][:, 0])], 1).astype(np.float32)


# ------------------------------------------------------------------------------------------------ events
class SeqData:
    """One recording's W200 packets at a fixed endpoint stride (events from `common.load_seq`, sub-ms times)."""

    def __init__(self, seq: str, G: Geo, every: int = EVERY, max_n: Optional[int] = None):
        assert seq not in C.ZGZ and C.subject_of(seq) in C.FIT7 + C.SELECT and seq.split("_")[-1] in ("local", "global"), seq
        self.seq, self.G = seq, G
        self.ss = C.SeqSamples(seq, every=every, max_n=max_n)
        self.tsub = np.load(C.DATA_ROOT / C.seq_dir(seq) / f"{seq}_tsub.npy", mmap_mode="r")
        self.domain = seq.split("_")[-1]
        self.subject = C.subject_of(seq)

    def __len__(self):
        return len(self.ss)

    def events(self, i: int) -> dict:
        ss = self.ss
        end, w = int(ss.ends[i]), int(ss.windows[i])
        start = end - w + 1
        a0, a1 = int(ss._offsets[start]), int(ss._offsets[end + 1])
        ev = np.asarray(ss._events[a0:a1])
        counts = np.diff(ss._offsets[start:end + 2]).astype(np.int64)
        ms = np.repeat(np.arange(start, end + 1, dtype=np.int64), counts)
        ts = np.asarray(self.tsub[a0:a1]).astype(np.int64)
        tus = ms * 1000 + ts
        tau = (tus - (end - T_REF_OFFSET) * 1000) / 1000.0
        _, layer = np.unique(tus, return_inverse=True)
        return dict(X=self.G.lut_x[ev[:, 0].astype(np.int64)], Y=self.G.lut_y[ev[:, 1].astype(np.int64)],
                    tau=tau.astype(np.float32), pol=np.clip(ev[:, 2], 0, 1).astype(np.uint8),
                    layer=layer.astype(np.int64), end=end, window=w)


def slab_of(tau: np.ndarray) -> np.ndarray:
    return np.clip(np.floor((np.asarray(tau, np.float64) - TAU_LO) / SLAB_MS), 0, NSLAB - 1).astype(np.int64)


def shuffle_tau(ev: dict, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Time-shuffle null inside the window.  `layer`: permute the time stamps among the 0.93 ms layers (events of one
    layer stay together); `event`: permute the stamps among events.  Positions and polarities are untouched."""
    tau = ev["tau"]
    if mode == "event":
        return tau[rng.permutation(len(tau))]
    L = int(ev["layer"].max()) + 1 if len(tau) else 0
    lt = np.zeros(L, np.float32)
    lt[ev["layer"]] = tau
    return lt[rng.permutation(L)][ev["layer"]]


# ------------------------------------------------------------------------------------------------ pairs
def find_pairs(pos: np.ndarray, nrm: np.ndarray, EX: np.ndarray, EY: np.ndarray, tw: float = TW,
               rmax: float = RMAX, chunk: int = 16):
    """(contour point, event, signed normal offset) triples: events within +-tw of a point along the contour and in
    [-rmax, rmax) along its outward normal."""
    m = len(pos)
    if m == 0 or len(EX) == 0:
        return np.zeros(0, np.int32), np.zeros(0, np.int32), np.zeros(0, np.float32)
    lo, hi = pos.min(0) - rmax - 1, pos.max(0) + rmax + 1
    idx = np.nonzero((EX >= lo[0]) & (EX <= hi[0]) & (EY >= lo[1]) & (EY <= hi[1]))[0]
    ex, ey = EX[idx].astype(np.float32), EY[idx].astype(np.float32)
    pf, nf = pos.astype(np.float32), nrm.astype(np.float32)
    pi_l, pj_l, d_l = [], [], []
    for c0 in range(0, m, chunk):
        sl = slice(c0, min(c0 + chunk, m))
        dx = ex[None, :] - pf[sl, 0, None]
        dy = ey[None, :] - pf[sl, 1, None]
        nx, ny = nf[sl, 0, None], nf[sl, 1, None]
        d = dx * nx + dy * ny
        t = dy * nx - dx * ny
        a, b = np.nonzero((np.abs(t) <= tw) & (d >= -rmax) & (d < rmax))
        pi_l.append((a + c0).astype(np.int32))
        pj_l.append(idx[b].astype(np.int32))
        d_l.append(d[a, b])
    return np.concatenate(pi_l), np.concatenate(pj_l), np.concatenate(d_l).astype(np.float32)


def nearest_contour(pos: np.ndarray, nrm: np.ndarray, EX: np.ndarray, EY: np.ndarray, rmax: float = RMAX,
                    chunk: int = 20000):
    """Per event (inside the contour bounding box + rmax): nearest contour point, its Euclidean distance and the
    signed normal offset of the event at that point.  Returns (event idx, point idx, dist, dn)."""
    m = len(pos)
    if m == 0 or len(EX) == 0:
        z = np.zeros(0)
        return z.astype(np.int64), z.astype(np.int64), z.astype(np.float32), z.astype(np.float32)
    lo, hi = pos.min(0) - rmax - 1, pos.max(0) + rmax + 1
    idx = np.nonzero((EX >= lo[0]) & (EX <= hi[0]) & (EY >= lo[1]) & (EY <= hi[1]))[0]
    pf, nf = pos.astype(np.float32), nrm.astype(np.float32)
    P_i, D_i, DN = [], [], []
    for c0 in range(0, len(idx), chunk):
        ii = idx[c0:c0 + chunk]
        dx = EX[ii][:, None].astype(np.float32) - pf[None, :, 0]
        dy = EY[ii][:, None].astype(np.float32) - pf[None, :, 1]
        d2 = dx * dx + dy * dy
        k = np.argmin(d2, 1)
        r = np.arange(len(ii))
        P_i.append(k)
        D_i.append(np.sqrt(d2[r, k]))
        DN.append(dx[r, k] * nf[k, 0] + dy[r, k] * nf[k, 1])
    return idx, np.concatenate(P_i), np.concatenate(D_i), np.concatenate(DN)


def pair_weights(cls_pt: np.ndarray, pi: np.ndarray, pj: np.ndarray) -> np.ndarray:
    """Soft de-duplication: an event inside the strips of several points of the same class carries total weight 1 in that
    class (1 / number of its pairs in the class).  Without it neighbouring strips count one event up to ~8 times and the
    clustered score variance (VIF) measures this multiplicity instead of the misspecification."""
    if len(pi) == 0:
        return np.zeros(0, np.float32)
    key = pj.astype(np.int64) * 3 + cls_pt[pi]
    return (1.0 / np.bincount(key)[key]).astype(np.float32)


def hist_unit(cls_pt: np.ndarray, pi, pj, d, slab_ev, pol_ev, w: Optional[np.ndarray] = None) -> np.ndarray:
    """(view, class, polarity, 132) weighted counts of the pairs (float64)."""
    out = np.zeros((len(VIEWS), 3, 2, NB), np.float64)
    if len(pi) == 0:
        return out
    w = np.ones(len(pi)) if w is None else w
    cl, b, sl, pl = cls_pt[pi], fine_index(d), slab_ev[pj], pol_ev[pj].astype(np.int64)
    for vi, v in enumerate(VIEWS):
        m = np.isin(sl, VIEW_SLABS[v])
        out[vi] = np.bincount((cl[m] * 2 + pl[m]) * NB + b[m], weights=w[m], minlength=3 * 2 * NB).reshape(3, 2, NB)
    return out


def hist_seg(seg_pt: np.ndarray, S: int, pi, pj, d, slab_ev, w: Optional[np.ndarray] = None) -> np.ndarray:
    """(segment, slab, 132) weighted counts of the pairs (float32)."""
    if len(pi) == 0 or S == 0:
        return np.zeros((S, NSLAB, NB), np.float32)
    w = np.ones(len(pi)) if w is None else w
    idx = (seg_pt[pi] * NSLAB + slab_ev[pj]) * NB + fine_index(d)
    return np.bincount(idx, weights=w, minlength=S * NSLAB * NB).reshape(S, NSLAB, NB).astype(np.float32)


# ------------------------------------------------------------------------------------------------ extraction
def _shift_condition(G: Geo, sc: dict, ev: dict, slab: np.ndarray, mag_mm: float, phi: float, oracle: bool,
                     cls_end: np.ndarray, side_end: np.ndarray) -> dict:
    """Contour of the previous state translated by `mag_mm` in the image plane (direction phi), segment histograms of the
    real events (V200) with PDA inputs and the nearest-contour-point baseline sums.  Labels (visibility, classes, side)
    come from the shifted geometry (deployable) or from the GT end geometry (oracle)."""
    verts = sc["verts"][0].clone()
    verts[:, 0] += mag_mm / 1000.0 * math.cos(phi)
    verts[:, 1] += mag_mm / 1000.0 * math.sin(phi)
    uv_s = G.SB.project_src(verts, sc["K"]).numpy().astype(np.float64)
    if oracle:
        cls_e, side_e = cls_end.copy(), side_end
        cls_e[cls_e == MAT] = -1
    else:
        cls_e, side_e, _ = G.edge_sets(verts, uv_s, sc["K"])
    pts = G.points(uv_s, sc["uv"][0], sc["uv"][1], cls_e, side_e, sc["c_palm"], sc["c_wrist"], sc["z"][0],
                   include_mat=False)
    if len(pts["pos"]) == 0:
        return dict(meta=np.zeros((0, 10), np.float32), hist=np.zeros((0, NB), np.float32), ncp=np.zeros((0, 2), np.float32))
    seg, S = segment_ids(pts["pos"], pts["nrm"], pts["cls"])
    pi, pj, d = find_pairs(pts["pos"], pts["nrm"], ev["X"], ev["Y"])
    h = hist_seg(seg, S, pi, pj, d, slab, pair_weights(pts["cls"], pi, pj)).sum(1)
    eidx, pidx, dist, dn = nearest_contour(pts["pos"], pts["nrm"], ev["X"], ev["Y"])
    ok = dist <= RMAX
    sgi = seg[pidx[ok]]
    ncp = np.stack([np.bincount(sgi, weights=dn[ok].astype(np.float64), minlength=S),
                    np.bincount(sgi, minlength=S).astype(np.float64)], 1)
    return dict(meta=segment_meta(pts, seg, S), hist=h.astype(np.float32), ncp=ncp.astype(np.float32))


def extract_sample(G: Geo, sd: SeqData, i: int, partner: int, with_shift: bool) -> Optional[dict]:
    """Everything the analysis needs from one packet; None when the hand has no usable silhouette."""
    ss = sd.ss
    sc = G.scene(ss.prev[i], ss.gt[i], ss.betas, ss.K)
    cls_e, side_e, info = G.edge_sets(sc["verts"][0], sc["uv"][0], sc["K"])
    pts = G.points(sc["uv"][0], sc["uv"][0], sc["uv"][1], cls_e, side_e, sc["c_palm"], sc["c_wrist"], sc["z"][0])
    n_cls = np.bincount(pts["cls"], minlength=3)
    ev, wev = sd.events(i), sd.events(partner)
    if n_cls[SIL] < 20 or len(ev["tau"]) == 0:
        return None
    slab, wslab = slab_of(ev["tau"]), slab_of(wev["tau"])
    pi, pj, d = find_pairs(pts["pos"], pts["nrm"], ev["X"], ev["Y"])
    wpi, wpj, wd = find_pairs(pts["pos"], pts["nrm"], wev["X"], wev["Y"])
    w, ww = pair_weights(pts["cls"], pi, pj), pair_weights(pts["cls"], wpi, wpj)
    seg, S = segment_ids(pts["pos"], pts["nrm"], pts["cls"])
    end = int(ss.ends[i])
    # -- unit (packet x class) histograms under the real data and every null
    unit = {"real": hist_unit(pts["cls"], pi, pj, d, slab, ev["pol"], w)}
    tau_l = shuffle_tau(ev, "layer", rng_for(sd.seq, end, 1))
    slab_l = slab_of(tau_l)
    unit["tshuf_layer"] = hist_unit(pts["cls"], pi, pj, d, slab_l, ev["pol"], w)
    unit["tshuf_event"] = hist_unit(pts["cls"], pi, pj, d, slab_of(shuffle_tau(ev, "event", rng_for(sd.seq, end, 2))),
                                    ev["pol"], w)
    unit["pol_flip"] = hist_unit(pts["cls"], pi, pj, d, slab, (1 - ev["pol"]).astype(np.uint8), w)
    unit["pol_perm"] = hist_unit(pts["cls"], pi, pj, d, slab, ev["pol"][rng_for(sd.seq, end, 3).permutation(len(ev["pol"]))], w)
    unit["cls_shuf"] = hist_unit(pts["cls"][rng_for(sd.seq, end, 4).permutation(len(pts["cls"]))], pi, pj, d, slab, ev["pol"], w)
    unit["wrongwin"] = hist_unit(pts["cls"], wpi, wpj, wd, wslab, wev["pol"], ww)
    # -- segment histograms: real (8 slabs), layer-shuffled (8 slabs), wrong window (V200)
    meta = segment_meta(pts, seg, S)
    segd = dict(meta=meta, hist=hist_seg(seg, S, pi, pj, d, slab, w), hist_tsh=hist_seg(seg, S, pi, pj, d, slab_l, w),
                hist_wrong=hist_seg(seg, S, wpi, wpj, wd, wslab, ww).sum(1))
    # -- exact raster: class sanity (inward probe of silhouette points lies in the hand) and SBA-vs-raster agreement
    mask = info["mask"]
    cm = pts["cls"] < 2
    qi = pts["pos"] - INWARD_PX * pts["nrm"]
    qx, qy = np.round(qi[:, 0]).astype(np.int64), np.round(qi[:, 1]).astype(np.int64)
    inb = (qx >= 0) & (qx < 240) & (qy >= 0) & (qy < 180)
    inner = inb & (pts["cls"] == SIL) & mask[np.clip(qy, 0, 179), np.clip(qx, 0, 239)]
    sba = info["sba"]
    geom = np.array([int(((pts["cls"] == SIL) & inb).sum()), int(inner.sum()),
                     int((sba == SIL).sum()), int(((sba == SIL) & (cls_e == SIL)).sum()),
                     int((sba == INT).sum()), int(((sba == INT) & (cls_e == INT)).sum())], np.int64)
    eidx, pidx, dist, _ = nearest_contour(pts["pos"][cm], pts["nrm"][cm], ev["X"], ev["Y"])
    ex = np.clip(np.round(ev["X"][eidx]).astype(np.int64), 0, 239)
    ey = np.clip(np.round(ev["Y"][eidx]).astype(np.int64), 0, 179)
    inside = mask[ey, ex]
    comp = np.array([(dist <= 2.0).sum(), ((dist > 2.0) & inside).sum(), ((dist > 2.0) & ~inside & (dist <= RMAX)).sum(),
                     ((dist > RMAX) & ~inside).sum() + (len(ev["tau"]) - len(eidx))], np.int64)
    out = dict(seq=sd.seq, i=int(i), end=end, subject=sd.subject, domain=sd.domain, n_events=int(len(ev["tau"])),
               window=int(ev["window"]), n_pts=n_cls.astype(np.int64), n_pairs=float(w.sum()), n_seg=int(S),
               R=float(np.median(np.linalg.norm(pts["pos"][pts["cls"] == SIL] - sc["c_palm"], axis=1))),
               geom=geom, comp=comp, unit=unit, seg=segd, shift={})
    if with_shift:
        cls_end, side_end, _ = G.edge_sets(sc["verts"][1], sc["uv"][1], sc["K"])
        for mi, mag in enumerate(SHIFT_MM):
            phi = float(rng_for(sd.seq, end, 100 + mi).uniform(0.0, 2.0 * math.pi))
            out["shift"][mag] = dict(
                phi=phi,
                dep=_shift_condition(G, sc, ev, slab, mag, phi, False, cls_end, side_end),
                orc=_shift_condition(G, sc, ev, slab, mag, phi, True, cls_end, side_end))
            if mag == SHIFT_PRIMARY:           # null: every event scattered by DISP_PX in its own random direction
                psi = rng_for(sd.seq, end, 200).uniform(0.0, 2.0 * math.pi, len(ev["tau"]))
                evd = dict(ev, X=(ev["X"] + DISP_PX * np.cos(psi)).astype(np.float32),
                           Y=(ev["Y"] + DISP_PX * np.sin(psi)).astype(np.float32))
                out["shift"][mag]["dep_disp"] = _shift_condition(G, sc, evd, slab, mag, phi, False, cls_end, side_end)
    return out


_G: Optional[Geo] = None


def _worker_init():
    global _G
    _G = Geo()


def run_chunk(task: dict) -> dict:
    """Extract samples [i0, i1) of one recording and write them (gzip pickle) under task['dir']."""
    global _G
    if _G is None:
        _G = Geo()
    t0 = time.time()
    sd = SeqData(task["seq"], _G, EVERY)
    N = len(sd)
    recs = []
    for i in range(task["i0"], min(task["i1"], N)):
        shift = task["shift_all"] or (i % SHIFT_FIT7_EVERY == 0)
        r = extract_sample(_G, sd, i, (i + N // 2) % N, shift)
        if r is not None:
            recs.append(r)
    path = Path(task["dir"]) / f"{task['seq']}__{task['i0']:05d}_{task['i1']:05d}.pkl.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=3) as f:
        pickle.dump(dict(task=task, n_total=N, recs=recs), f, protocol=4)
    return dict(path=str(path), n=len(recs), seconds=time.time() - t0)


def seq_list() -> Dict[str, str]:
    return {s: ("SELECT" if C.subject_of(s) in C.SELECT else "FIT7") for s in C.base_seqs(C.FIT7 + C.SELECT)}


def make_tasks(out_dir: Path, chunk: int = 40) -> List[dict]:
    tasks = []
    for seq, role in seq_list().items():
        _, _, aux, _ = C.load_seq(seq)
        n = len(C.endpoints(aux, EVERY))
        for i0 in range(0, n, chunk):
            tasks.append(dict(seq=seq, i0=i0, i1=i0 + chunk, shift_all=(role == "SELECT"), dir=str(out_dir / "cache")))
    return tasks


def extract_all(out_dir: Path, workers: int) -> List[dict]:
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp
    tasks = make_tasks(out_dir)
    done = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"), initializer=_worker_init) as ex:
        for r in ex.map(run_chunk, tasks):
            done.append(r)
    return done


def load_cache(cache_dir: Path) -> Dict[str, List[dict]]:
    """All extracted samples by recording, ordered by packet index."""
    by: Dict[str, List[dict]] = {}
    for p in sorted(Path(cache_dir).glob("*.pkl.gz")):
        with gzip.open(p, "rb") as f:
            blob = pickle.load(f)
        by.setdefault(blob["task"]["seq"], []).extend(blob["recs"])
    for s in by:
        by[s].sort(key=lambda r: r["i"])
    return by


# ------------------------------------------------------------------------------------------------ analysis
N_NCP_MIN = 20
N_BOOT = 20                                    # multinomial resamples per segment for the mode's sampling variance
TAU_SLAB = TAU_LO + SLAB_MS * (np.arange(NSLAB) + 0.5)      # slab centres (ms)
VIEW_TAU = {"V200": -50.0, "V50": 25.0}                       # nominal mean event lag of the view (ms), reporting only


def unit_stack(recs: List[dict], null: str, view: str, c: int, pol: Optional[int] = None) -> np.ndarray:
    """(U, 132) per-packet histograms of class c under `null` (polarities summed unless `pol`)."""
    vi = VIEWS.index(view)
    if not recs:
        return np.zeros((0, NB))
    a = np.stack([r["unit"][null][vi, c] for r in recs]).astype(np.float64)        # (U, 2, NB)
    return a.sum(1) if pol is None else a[:, pol]


def mode_boot_var(h: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> float:
    """Sampling variance of the 1-px mode of a (weighted) segment histogram: multinomial resampling of its effective events."""
    n = int(round(float(h.sum())))
    if n < 2:
        return float("nan")
    p = h / h.sum()
    modes = [peak_of(h1px(x.astype(np.float64)))[0] for x in rng.multinomial(n, p, size=n_boot)]
    return float(np.var(modes, ddof=1))


def seg_rows(recs: List[dict], c: int, n_min: Optional[int] = None, key: str = "hist", boot: bool = False,
             seed_key: str = "") -> dict:
    """Qualified segments of class c (>= n_min pairs in V200): meta, 1-px mode of V200 / V50, V200 histograms; with `boot`
    the bootstrap variance of each V200 mode (noise floor of the within-packet jitter)."""
    n_min = N_SEG_MIN if n_min is None else n_min
    rng = rng_for("boot", seed_key, c)
    vb: List[float] = []
    ks, meta, h200, p200, p50, n50 = [], [], [], [], [], []
    for k, r in enumerate(recs):
        m, H = r["seg"]["meta"], r["seg"][key]
        sel = np.nonzero(m[:, 0] == c)[0]
        if not len(sel):
            continue
        h = H[sel].sum(1).astype(np.float64)
        ok = h.sum(1) >= n_min
        for j in np.nonzero(ok)[0]:
            ks.append(k)
            meta.append(m[sel[j]])
            h200.append(h[j])
            p200.append(peak_of(h1px(h[j]))[0])
            if boot:
                vb.append(mode_boot_var(h[j], rng))
            h5 = H[sel[j]][list(VIEW_SLABS["V50"])].sum(0).astype(np.float64)
            n50.append(h5.sum())
            p50.append(peak_of(h1px(h5))[0] if h5.sum() >= n_min else float("nan"))
    return dict(k=np.asarray(ks, np.int64), meta=np.asarray(meta, np.float64).reshape(-1, 10),
                h200=np.asarray(h200, np.float64).reshape(-1, NB), p200=np.asarray(p200, np.float64),
                p50=np.asarray(p50, np.float64), n50=np.asarray(n50, np.float64), vb=np.asarray(vb, np.float64))


def wrong_hists(recs: List[dict], c: int, n_min: Optional[int] = None) -> np.ndarray:
    """V200 histograms of the qualified class-c segments built from wrong-window events (null for the PDA / NIS)."""
    n_min = N_SEG_MIN if n_min is None else n_min
    out = []
    for r in recs:
        m, H = r["seg"]["meta"], r["seg"]["hist_wrong"].astype(np.float64)
        sel = (m[:, 0] == c) & (H.sum(1) >= n_min)
        out.append(H[sel])
    return np.concatenate(out) if out else np.zeros((0, NB))


def design(meta: np.ndarray, kind: str = "geom") -> np.ndarray:
    """Nuisance design of the segment mode-peak bias (px): translation (nx, ny), scale about the palm centre (radial
    arm of the normal), latency x velocity (normal speed, px/ms).  `literal` = plan wording (constant, distance, speed);
    `wrist` = geom with the scale arm about the wrist (E8)."""
    nx, ny, rad, radw, vn = meta[:, 2], meta[:, 3], meta[:, 4], meta[:, 5], meta[:, 6]
    if kind == "geom":
        return np.stack([nx, ny, rad, vn], 1)
    if kind == "wrist":
        return np.stack([nx, ny, radw, vn], 1)
    return np.stack([np.ones_like(rad), rad, vn], 1)


def jitter_sd(rows: dict) -> Tuple[float, float, int]:
    """Within-packet spread of the segment mode peaks over packets with >= 3 qualified segments.  Returns (raw SD, noise-
    corrected SD, n packets): raw = sqrt(mean packet variance); corrected subtracts each packet's mean bootstrap variance
    of its segment modes (floored at 0), i.e. the systematic wandering beyond sampling noise."""
    if not len(rows["k"]):
        return float("nan"), float("nan"), 0
    raw, cor = [], []
    for k in np.unique(rows["k"]):
        m = rows["k"] == k
        if m.sum() >= 3:
            v = float(np.var(rows["p200"][m], ddof=1))
            raw.append(v)
            vb = rows["vb"][m] if len(rows["vb"]) else np.zeros(m.sum())
            cor.append(max(v - float(np.nanmean(vb)), 0.0))
    if not raw:
        return float("nan"), float("nan"), 0
    return math.sqrt(float(np.mean(raw))), math.sqrt(float(np.mean(cor))), len(raw)


def kappa(recs: List[dict], c: int, key: str = "hist") -> Tuple[float, int]:
    """Motion-following coefficient: per segment the slope of its slab-wise mode peak against slab time (px/ms),
    regressed through the origin on the segment's GT normal speed.  1 = events follow the moving contour."""
    num = den = 0.0
    n = 0
    for r in recs:
        m, H = r["seg"]["meta"], r["seg"][key]
        for j in np.nonzero(m[:, 0] == c)[0]:
            hs = H[j].astype(np.float64)
            ok = hs.sum(1) >= N_SLAB_MIN
            if ok.sum() < 4:
                continue
            pk = np.array([peak_of(h1px(hs[l]))[0] for l in np.nonzero(ok)[0]])
            t = TAU_SLAB[ok]
            sl = float(np.polyfit(t, pk, 1)[0])
            v = float(m[j, 6])
            num += sl * v
            den += v * v
            n += 1
    return (num / den if den > 1e-12 else float("nan")), n


def fit_constants(by_seq: Dict[str, List[dict]]) -> dict:
    """FIT7-only constants per (domain, class, view): clutter table, mixture (theta, s, beta); nuisance coefficients."""
    out: Dict[str, dict] = {}
    for dom in ("local", "global"):
        seqs = [s for s in by_seq if s.endswith("_" + dom) and C.subject_of(s) in C.FIT7]
        recs = [r for s in seqs for r in by_seq[s]]
        d = {}
        for c, cn in enumerate(CLS):
            d[cn] = {}
            for v in ("V200", "V50"):
                hr = unit_stack(recs, "real", v, c).sum(0) if recs else np.zeros(NB)
                hw = unit_stack(recs, "wrongwin", v, c).sum(0) if recs else np.zeros(NB)
                q = clutter_table(hw)
                f = fit_mixture(hr, q) if hr.sum() > 0 else None
                d[cn][v] = dict(fit=f, q=q.tolist(), n_pairs=float(hr.sum()))
            if c < 2:
                rows = seg_rows(recs, c)
                co = {}
                if len(rows["p200"]) >= 20:
                    for kind in ("geom", "literal", "wrist"):
                        co[kind] = ols(design(rows["meta"], kind), rows["p200"]).tolist()
                d[cn]["coef"] = co
                d[cn]["n_seg"] = int(len(rows["p200"]))
        out[dom] = d
    return out


def _fit_of(fit: dict, dom: str, cn: str, v: str = "V200"):
    e = fit[dom][cn][v]
    return e["fit"], np.asarray(e["q"], np.float64)


def eval_recording(seq: str, recs: List[dict], fit: dict) -> dict:
    """Per-class gate inputs of one recording (E5, E4, E1 (i)(ii), E8) plus null contrasts."""
    dom = seq.split("_")[-1]
    out: Dict[str, dict] = {}
    R = float(np.median([r["R"] for r in recs])) if recs else float("nan")
    for c, cn in enumerate(CLS):
        e: dict = {"insufficient": False}
        U200 = unit_stack(recs, "real", "V200", c)
        has = U200.sum(1) > 0
        e["n_units"], e["n_pairs"] = int(has.sum()), float(U200.sum())
        f, q = _fit_of(fit, dom, cn, "V200")
        if e["n_units"] < N_UNIT_MIN or f is None or f.get("at_bound", False):
            e["insufficient"] = True
            e["reason"] = "units" if e["n_units"] < N_UNIT_MIN else ("no_fit" if f is None else "fit_at_bound")
            out[cn] = e
            continue
        # E5: pseudo-true offset, sandwich, VIF; median companion; scale equivalent
        m = mle_location(U200[has], f["s"], f["beta"], q)
        e.update(b_m=m["theta"], A=m["A"], B=m["B"], vif=m["vif"], se=m["se"], at_bound=m["at_bound"],
                 b_med=median_from_hist(U200.sum(0)), R=R, dsig=m["theta"] / R)
        f50, q50 = _fit_of(fit, dom, cn, "V50")
        U50 = unit_stack(recs, "real", "V50", c)
        h50 = U50.sum(1) > 0
        if f50 is not None and h50.sum() >= N_UNIT_MIN:
            m50 = mle_location(U50[h50], f50["s"], f50["beta"], q50)
            e.update(b_m50=m50["theta"], vif50=m50["vif"])
        # E4: recording-level volume peak, peakiness vs nulls, jitter
        pk, pkn = peak_of(h1px(U200.sum(0)))
        e["peak"], e["pk_real"] = pk, pkn
        e["pk_wrong"] = peak_of(h1px(unit_stack(recs, "wrongwin", "V200", c).sum(0)))[1]
        e["pk_tshuf"] = peak_of(h1px(unit_stack(recs, "tshuf_layer", "V200", c).sum(0)))[1]
        e["r_pk"] = e["pk_real"] / max(e["pk_wrong"], 1e-9)
        for v in ("V50", "Vc50"):
            e[f"pk_real_{v}"] = peak_of(h1px(unit_stack(recs, "real", v, c).sum(0)))[1]
            e[f"pk_wrong_{v}"] = peak_of(h1px(unit_stack(recs, "wrongwin", v, c).sum(0)))[1]
            e[f"pk_tshufl_{v}"] = peak_of(h1px(unit_stack(recs, "tshuf_layer", v, c).sum(0)))[1]
            e[f"pk_tshufe_{v}"] = peak_of(h1px(unit_stack(recs, "tshuf_event", v, c).sum(0)))[1]
        # polarity-resolved peak separation (descriptive) and its nulls
        for nm in ("real", "pol_perm", "pol_flip"):
            a = unit_stack(recs, nm, "V200", c, 1).sum(0)
            b = unit_stack(recs, nm, "V200", c, 0).sum(0)
            e[f"dpol_{nm}"] = (peak_of(h1px(a))[0] - peak_of(h1px(b))[0]) if a.sum() > 50 and b.sum() > 50 else float("nan")
        if c < 2:
            rows = seg_rows(recs, c, boot=True, seed_key=seq)
            e["n_seg"] = int(len(rows["p200"]))
            e["jitter_raw"], e["jitter"], e["n_jit_pkts"] = jitter_sd(rows)
            co = fit[dom][cn].get("coef", {})
            if len(rows["p200"]) >= 10 and "geom" in co:
                for kind in ("geom", "literal"):
                    yh = design(rows["meta"], kind) @ np.asarray(co[kind])
                    e[f"r2u_{kind}"], e[f"r2c_{kind}"] = r2(rows["p200"], yh)
                for kind in ("geom", "literal", "wrist"):
                    w = ols(design(rows["meta"], kind), rows["p200"])
                    e[f"r2u_within_{kind}"] = r2(rows["p200"], design(rows["meta"], kind) @ w)[0]
                    e[f"coef_within_{kind}"] = w.tolist()
                e["sigma_hat_wrist"] = 1.0 + e["coef_within_wrist"][2]
                e["sigma_hat_palm"] = 1.0 + e["coef_within_geom"][2]
                # lag coefficient on V200 vs V50 (events follow motion: the difference ~ +75 ms)
                ok = np.isfinite(rows["p50"]) & (rows["n50"] >= N_SEG_MIN)
                if ok.sum() >= 10:
                    w50 = ols(design(rows["meta"][ok], "geom"), rows["p50"][ok])
                    e["lag_v200"] = e["coef_within_geom"][3]
                    e["lag_v50"] = float(w50[3])
            e["kappa_real"], e["n_kappa"] = kappa(recs, c)
            e["kappa_tshuf"] = kappa(recs, c, "hist_tsh")[0]
            # E1 (i): drift of b_c, de-biasing of the segment mode peaks; (ii) NIS of the 1-D PDA posterior
            e["drift"] = abs(m["theta"] - f["theta"])
            if len(rows["p200"]) >= 10:
                raw = np.median(np.abs(rows["p200"]))
                db = np.median(np.abs(rows["p200"] - f["theta"]))
                e["debias_red"] = 1.0 - float(db / max(raw, 1e-9))
                mu, var = pda_posterior(rows["h200"], f["s"], f["beta"], q, f["theta"])
                nis = mu ** 2 / var
                ns = len(np.unique(rows["k"]))
                lo, hi = nis_band(ns)
                e.update(nis_med=float(np.median(nis)), nis_lo=lo, nis_hi=hi, n_nis_pkts=int(ns))
                hw = wrong_hists(recs, c)
                if len(hw) >= 10:
                    muw, varw = pda_posterior(hw, f["s"], f["beta"], q, f["theta"])
                    e["nis_med_wrong"] = float(np.median(muw ** 2 / varw))
        out[cn] = e
    # class specificity of b_m (descriptive) real vs class-label-shuffled
    for nm in ("real", "cls_shuf"):
        bs = []
        for c, cn in enumerate(CLS):
            f, q = _fit_of(fit, dom, cn, "V200")
            U = unit_stack(recs, nm, "V200", c)
            h = U.sum(1) > 0
            if f is not None and h.sum() >= N_UNIT_MIN:
                bs.append(mle_location(U[h], f["s"], f["beta"], q)["theta"])
        out.setdefault("_spread", {})[nm] = (max(bs) - min(bs)) if len(bs) >= 2 else float("nan")
    comp = np.sum([r["comp"] for r in recs], 0).astype(np.float64) if recs else np.zeros(4)
    out["_outside"] = (comp / max(comp.sum(), 1.0)).tolist()
    out["_n"] = dict(samples=len(recs), events_med=float(np.median([r["n_events"] for r in recs])) if recs else 0.0, R=R)
    return out


def _role(seq: str) -> str:
    return "SELECT" if C.subject_of(seq) in C.SELECT else "FIT7"


def position_recovery(by_seq: Dict[str, List[dict]], fit: dict, group: str) -> dict:
    """Recovery of a known image-plane shift of the previous-state contour (E1 iii).  Per qualified segment (>= 50 pairs
    and >= 20 nearest-point events, contour classes) the shift estimate is -(PDA posterior mean) and -(mean normal
    offset of the events nearest to the segment's points); both are regressed on the true normal shift of the segment.
    Pooled OLS slopes; the PDA - baseline slope difference carries a block bootstrap (blocks = 10 consecutive packets of
    a recording) and per-subject differences.  `dep`: labels from the shifted geometry; `orc`: labels from GT end."""
    out: Dict[str, dict] = {}
    seqs = [s for s in by_seq if _role(s) == group]
    for mode in ("dep", "orc", "dep_disp"):
        for mag in SHIFT_MM:
            if mode == "dep_disp" and mag != SHIFT_PRIMARY:
                continue
            blocks: Dict[Tuple[str, int], np.ndarray] = {}
            for s in seqs:
                dom = s.split("_")[-1]
                for r in by_seq[s]:
                    sh = r["shift"].get(mag)
                    if sh is None or mode not in sh:
                        continue
                    d = sh[mode]
                    meta, H, ncp = d["meta"], d["hist"].astype(np.float64), d["ncp"].astype(np.float64)
                    ok = (H.sum(1) >= N_SEG_MIN) & (ncp[:, 1] >= N_NCP_MIN) & (meta[:, 0] < 2)
                    if not ok.any():
                        continue
                    r_pda = np.full(len(meta), np.nan)
                    for c in (SIL, INT):
                        f, q = _fit_of(fit, dom, CLS[c], "V200")
                        m = ok & (meta[:, 0] == c)
                        if f is None or not m.any():
                            continue
                        mu, _ = pda_posterior(H[m], f["s"], f["beta"], q, f["theta"])
                        r_pda[m] = -mu
                    use = ok & np.isfinite(r_pda)
                    if not use.any():
                        continue
                    r_ncp = -ncp[use, 0] / ncp[use, 1]
                    st = meta[use, 7].astype(np.float64)
                    cl = meta[use, 0].astype(np.int64)
                    blk = blocks.setdefault((s, r["i"] // BLOCK), np.zeros((3, 2, 5)))
                    for gi, mk in enumerate((np.ones_like(cl, bool), cl == SIL, cl == INT)):
                        for ei, rr in enumerate((r_pda[use], r_ncp)):
                            x, y = st[mk], rr[mk]
                            blk[gi, ei] += [len(x), x.sum(), y.sum(), (x * x).sum(), (x * y).sum()]
            key = f"{mode}/{mag}"
            if not blocks:
                out[key] = dict(n_seg=0)
                continue
            keys = list(blocks)
            A = np.stack([blocks[k] for k in keys])                                   # (B, 3, 2, 5)
            tot = A.sum(0)
            sl = lambda T, gi, ei: slope_from_sums(*T[gi, ei])                         # noqa: E731
            rng = np.random.default_rng(SEED + mag)
            bs, bp, bn = [], [], []
            for _ in range(2000):
                T = A[rng.integers(0, len(A), len(A))].sum(0)
                bp.append(sl(T, 0, 0))
                bn.append(sl(T, 0, 1))
                bs.append(bp[-1] - bn[-1])
            bs, bp, bn = np.asarray(bs), np.asarray(bp), np.asarray(bn)
            subj = {}
            for sj in sorted({C.subject_of(k[0]) for k in keys}):
                T = np.stack([blocks[k] for k in keys if C.subject_of(k[0]) == sj]).sum(0)
                subj[sj] = dict(delta=sl(T, 0, 0) - sl(T, 0, 1), n=float(T[0, 0, 0]))
            percls = {cn: dict(pda=sl(tot, gi + 1, 0), ncp=sl(tot, gi + 1, 1), n=float(tot[gi + 1, 0, 0]))
                      for gi, cn in enumerate(CLS[:2])}
            out[key] = dict(n_seg=float(tot[0, 0, 0]), n_blocks=len(keys), slope_pda=sl(tot, 0, 0),
                            slope_ncp=sl(tot, 0, 1), delta=sl(tot, 0, 0) - sl(tot, 0, 1),
                            ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))],
                            slope_pda_ci=[float(np.nanpercentile(bp, 2.5)), float(np.nanpercentile(bp, 97.5))],
                            slope_ncp_ci=[float(np.nanpercentile(bn, 2.5)), float(np.nanpercentile(bn, 97.5))],
                            per_subject=subj, per_class=percls)
    return out


def validity_checks(by_seq: Dict[str, List[dict]]) -> dict:
    """Tool-validity conditions that do not depend on the effect (any False -> `tool_defect`)."""
    v = dict(tshuf_pooled_invariant=True, pol_pooled_invariant=True, pol_flip_swaps=True, cls_total_invariant=True,
             seg_tshuf_invariant=True, no_zgz=True, finite_counts=True)
    g = np.zeros(6)
    for s, recs in by_seq.items():
        v["no_zgz"] &= s not in C.ZGZ
        for r in recs:
            u = r["unit"]
            re = u["real"]
            eq = lambda a, b: bool(np.allclose(a, b, rtol=1e-6, atol=1e-4))            # noqa: E731  (float sums)
            for nm in ("tshuf_layer", "tshuf_event"):
                v["tshuf_pooled_invariant"] &= eq(re[0], u[nm][0])                       # V200 does not see time
            for nm in ("pol_flip", "pol_perm"):
                v["pol_pooled_invariant"] &= eq(re.sum(2), u[nm].sum(2))
            v["pol_flip_swaps"] &= eq(re[:, :, 0], u["pol_flip"][:, :, 1])
            v["cls_total_invariant"] &= eq(re.sum(1), u["cls_shuf"].sum(1))
            v["seg_tshuf_invariant"] &= eq(r["seg"]["hist"].sum(1), r["seg"]["hist_tsh"].sum(1))
            v["finite_counts"] &= bool(np.isfinite(r["R"]) and r["n_events"] > 0)
            g += r["geom"]
    v["inward_probe_in_hand"] = bool(g[0] > 0 and g[1] / g[0] >= GEOM_MIN_FRAC)
    v["geom_counts"] = [float(x) for x in g]       # [n sil, inward in hand, SBA sil, agree, SBA int, agree]: counts, not effects
    return v


def protocol_dict() -> dict:
    return dict(seed=SEED, every=EVERY, step_ms=C.STEP, window=C.WINDOW, scale=SCALE, bin=BIN, rmax=RMAX, tw=TW,
                cell=CELL, nabin=NABIN, n_seg_min=N_SEG_MIN, n_slab_min=N_SLAB_MIN, n_unit_min=N_UNIT_MIN,
                n_fit_min=N_FIT_MIN, n_jit_pkt_min=N_JIT_PKT_MIN, block=BLOCK, shift_mm=list(SHIFT_MM),
                shift_primary=SHIFT_PRIMARY, shift_fit7_every=SHIFT_FIT7_EVERY, probe_px=GEOM_PROBE_PX,
                inward_px=INWARD_PX, z_int_margin=Z_INT_MARGIN, geom_min_frac=GEOM_MIN_FRAC, gate=GATE,
                disp_px=DISP_PX, null_slope_max=NULL_SLOPE_MAX, n_ncp_min=N_NCP_MIN,
                views={k: list(v) for k, v in VIEW_SLABS.items()}, classes=list(CLS), nulls=list(NULLS),
                recording_rule=RECORDING_RULE, fit7=list(C.FIT7), select=list(C.SELECT))


def analyze(by_seq: Dict[str, List[dict]], unit_tests_ok: bool = True) -> dict:
    fit = fit_constants(by_seq)
    by_rec = {s: eval_recording(s, recs, fit) for s, recs in sorted(by_seq.items())}
    val = validity_checks(by_seq)
    val["unit_tests"] = bool(unit_tests_ok)
    for dom in ("local", "global"):
        val[f"fit_sil_{dom}"] = fit[dom]["sil"]["V200"]["fit"] is not None
    pos = {g: position_recovery(by_seq, fit, g) for g in ("SELECT", "FIT7")}
    for g, P in pos.items():                                   # the displaced-event control must not recover the shift
        d = P.get(f"dep_disp/{SHIFT_PRIMARY}", {})
        enough = d.get("n_seg", 0) >= 30
        flat = lambda ci: ci[0] <= NULL_SLOPE_MAX and ci[1] >= -NULL_SLOPE_MAX            # noqa: E731
        val[f"disp_null_flat_{g}"] = bool((not enough) or (flat(d["slope_pda_ci"]) and flat(d["slope_ncp_ci"])))
        val[f"disp_null_n_seg_{g}"] = float(d.get("n_seg", 0))
    return dict(protocol=protocol_dict(), fit=fit, by_rec=by_rec, position=pos, validity=val,
                select_recs=[s for s in sorted(by_seq) if _role(s) == "SELECT"],
                fit7_recs=[s for s in sorted(by_seq) if _role(s) == "FIT7"],
                n_samples={s: len(r) for s, r in by_seq.items()})


# ------------------------------------------------------------------------------------------------ decision
def _none_to_nan(o):
    if isinstance(o, dict):
        return {k: _none_to_nan(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_none_to_nan(v) for v in o]
    return float("nan") if o is None else o


def _med(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    return float(np.median(xs)) if xs else float("nan")


def decide(result: dict) -> dict:
    """Apply the pre-registered gates (tools/mv2/prereg_m4.md) to `analyze()`'s result.  Verdicts: pass / kill /
    descriptive; a violated tool-validity condition gives `tool_defect` for everything (never a pass)."""
    G = GATE
    result = dict(result, by_rec=_none_to_nan(result["by_rec"]))
    bad = [k for k, x in result.get("validity", {}).items() if isinstance(x, bool) and not x]
    if bad:
        return dict(overall="tool_defect", e5="tool_defect", e4="tool_defect", e1="tool_defect", reasons=["validity:" + k for k in bad])
    sel, by = result["select_recs"], result["by_rec"]

    def recs_of(cn):
        return [by[s].get(cn, {"insufficient": True}) for s in sel]

    def usable(cn):
        return all(not e.get("insufficient", True) for e in recs_of(cn))

    def allrec(cn, pred):
        return usable(cn) and all(pred(e) for e in recs_of(cn))

    def medrec(cn, key):
        return _med([e.get(key) for e in recs_of(cn)]) if usable(cn) else float("nan")

    det = [cn for cn in CLS if usable(cn) and medrec(cn, "r_pk") >= G["peak_ratio"]]
    contour_det = [cn for cn in det if cn in CLS[:2]]
    # ---- E5
    admit = {cn: allrec(cn, lambda e: abs(e["b_m"]) <= G["b_enter"] and abs(e["b_med"]) <= G["b_enter"]
                        and abs(e["dsig"]) <= G["dsig_enter"] and e["vif"] <= G["vif_enter"] and not e["at_bound"])
             for cn in CLS}
    k5_bias = bool(det) and all(abs(medrec(cn, "b_m")) > G["b_kill"] and abs(medrec(cn, "b_med")) > G["b_kill"] for cn in det)
    k5_scale = ("sil" in det) and abs(medrec("sil", "dsig")) >= G["dsig_kill"]
    r5 = dict(admit=admit, kill_bias=k5_bias, kill_scale=k5_scale,
              verdict="kill" if (k5_bias or k5_scale) else ("pass" if any(admit.values()) else "descriptive"))
    # ---- E4
    jit_ok = {cn: allrec(cn, lambda e: e.get("n_jit_pkts", 0) >= N_JIT_PKT_MIN and e["jitter"] <= G["jitter"]) for cn in CLS[:2]}
    r2_ok = {cn: allrec(cn, lambda e: abs(e["peak"]) < G["bias_r2"] or e.get("r2u_geom", -1.0) >= G["r2"]) for cn in CLS[:2]}
    pk_ok = {cn: allrec(cn, lambda e: e["r_pk"] >= G["peak_ratio"]) for cn in CLS[:2]}
    e4_pass = {cn: jit_ok[cn] and r2_ok[cn] and pk_ok[cn] for cn in CLS[:2]}

    def violates(cn):
        j = medrec(cn, "jitter")
        return (np.isfinite(j) and j > G["jitter"]) or (abs(medrec(cn, "peak")) >= G["bias_r2"] and medrec(cn, "r2u_geom") < G["r2"])

    k4_nosignal = len(contour_det) == 0 and any(usable(cn) for cn in CLS[:2])     # insufficient data is not "no signal"
    k4_bias = bool(contour_det) and all(bool(violates(cn)) for cn in contour_det)
    r4 = dict(jitter_ok=jit_ok, r2_ok=r2_ok, peak_ok=pk_ok, pass_by_class=e4_pass, kill_nosignal=k4_nosignal, kill_biased=k4_bias,
              verdict="kill" if (k4_nosignal or k4_bias) else ("pass" if any(e4_pass.values()) else "descriptive"))
    # ---- E1
    i_ok = {cn: allrec(cn, lambda e: e["drift"] <= G["drift"] and e.get("debias_red", -1.0) >= G["debias_red"]) for cn in CLS[:2]}
    ii_ok = {cn: allrec(cn, lambda e: "nis_med" in e and e["nis_lo"] <= e["nis_med"] <= e["nis_hi"]) for cn in CLS[:2]}
    P = result["position"]["SELECT"]
    pd, po = P.get(f"dep/{SHIFT_PRIMARY}", {}), P.get(f"orc/{SHIFT_PRIMARY}", {})
    subj_ok = bool(pd.get("per_subject")) and all(x["delta"] > 0 for x in pd["per_subject"].values())
    iii_ok = bool(pd.get("ci")) and pd["ci"][0] > 0 and subj_ok
    ci_has0 = bool(pd.get("ci")) and pd["ci"][0] <= 0 <= pd["ci"][1]
    k1_drift = bool(contour_det) and all(medrec(cn, "drift") > G["drift_kill"] for cn in contour_det)
    k1_floor = ci_has0 and pd.get("slope_ncp", 0.0) >= G["floor_kill"]
    k1_oracle = bool(pd.get("ci")) and bool(po.get("ci")) and pd["ci"][0] <= 0 < po["ci"][0]
    r1 = dict(stable_debias=i_ok, nis_ok=ii_ok, position_ok=iii_ok, position_delta=pd.get("delta"), position_ci=pd.get("ci"),
              baseline_slope=pd.get("slope_ncp"), kill_drift=k1_drift, kill_floor=k1_floor, kill_oracle_only=k1_oracle,
              verdict="kill" if (k1_drift or k1_floor or k1_oracle) else
              ("pass" if iii_ok and any(i_ok[cn] and ii_ok[cn] for cn in CLS[:2]) else "descriptive"))
    # ---- combination (draft rule: position gain + calibration + stable or small bias + 1-D peak gates)
    entered = [cn for cn in CLS[:2] if e4_pass[cn] and ii_ok[cn] and iii_ok and (i_ok[cn] or admit[cn])]
    reasons = []
    for nm, r in (("e5", r5), ("e4", r4), ("e1", r1)):
        reasons += [f"{nm}:{k}" for k, x in r.items() if k.startswith("kill") and x is True]
    overall = "kill" if "kill" in (r5["verdict"], r4["verdict"], r1["verdict"]) else ("pass" if entered else "descriptive")
    sig = [by[s][cn].get("sigma_hat_wrist") for s in result["select_recs"] + result.get("fit7_recs", []) for cn in ("sil",)
           if cn in by[s]]
    sig = [x for x in sig if x is not None and np.isfinite(x)]
    r8 = dict(sigma_hat_wrist=sig, in_range_frac=float(np.mean([0.93 <= x <= 1.01 for x in sig])) if sig else None,
              cv_sigma=float(np.std(sig) / abs(np.mean(sig))) if len(sig) > 1 else None,
              cv_rho=float(np.std(np.asarray(sig) - 1.0) / abs(np.mean(np.asarray(sig) - 1.0)))
              if len(sig) > 1 and abs(np.mean(np.asarray(sig) - 1.0)) > 1e-9 else None, gate="descriptive only")
    return dict(overall=overall, e5=r5["verdict"], e4=r4["verdict"], e1=r1["verdict"], entered_classes=entered, detectable=det,
                reasons=reasons, detail=dict(e5=r5, e4=r4, e1=r1, e8=r8))


# ------------------------------------------------------------------------------------------------ unit tests (synthetic only)
def _draw_hist(rng, theta, s, beta, q, n):
    p = np.exp(logp_theta(np.array([theta]), s, beta, q)[0])
    return rng.multinomial(int(n), p / p.sum()).astype(np.float64)


def _smooth_q():
    x = np.exp(-0.5 * ((CENTRES + 3.0) / 9.0) ** 2) + 0.2           # asymmetric empirical-looking clutter
    return x / x.sum()


def _ut_lut(G: Optional[Geo]) -> bool:
    from semkine import sba as SB
    g = SB.make_grid(torch.device("cpu"))
    cx = g.cx_src.numpy().astype(np.float64)
    src = np.arange(640)
    cell = (src * 240) // 640
    mean_idx = np.array([src[cell == j].mean() for j in range(240)])
    phase = SCALE * cx - np.arange(240)
    return bool(np.allclose(cx, mean_idx) and abs(phase.mean() - 0.4375) < 0.01 and phase.min() > 0.37 and phase.max() < 0.51)


def _ut_pairs() -> bool:
    x = np.arange(20.0, 60.0, 1.0)
    pos = np.stack([x, np.full_like(x, 50.0)], 1)
    nrm = np.tile(np.array([[0.0, -1.0]]), (len(x), 1))                 # outward = up (-y)
    delta = np.array([-3.0, 0.0, 2.5, 7.25, -15.0])
    ex = np.array([30.0, 31.0, 32.0, 33.0, 34.0, 30.0, 30.0, 100.0], np.float32)
    ey = np.array([50.0 - delta[0], 50.0 - delta[1], 50.0 - delta[2], 50.0 - delta[3], 50.0 - delta[4], 50 - 3.0, 50.0 - 17.0, 50.0])
    ey = ey.astype(np.float32)
    ex[5], ex[7] = 31.6, 70.0                                           # 70: 11 px beyond the contour's end
    pi, pj, d = find_pairs(pos, nrm, ex, ey.astype(np.float32))
    got = {int(j): [] for j in range(len(ex))}
    for a, b, c in zip(pi, pj, d):
        got[int(b)].append(float(c))
    ok = all(len(got[k]) >= 1 and abs(got[k][0] - delta[k]) < 1e-4 for k in range(5))
    ok &= len(got[6]) == 0 and len(got[7]) == 0                         # beyond rmax, off the contour
    ok &= len(got[5]) >= 1 and all(abs(v - 3.0) < 1e-4 for v in got[5])  # between two points: offset +3 (outward) for both
    return bool(ok)


def _ut_mixture() -> bool:
    rng = np.random.default_rng(1)
    q = _smooth_q()
    h = _draw_hist(rng, 2.0, 2.0, 0.6, q, 300000)
    f = fit_mixture(h, q)
    return bool(f and abs(f["theta"] - 2.0) < 0.1 and abs(f["s"] - 2.0) < 0.15 and abs(f["beta"] - 0.6) < 0.05)


def _ut_sandwich() -> bool:
    rng = np.random.default_rng(2)
    q = _smooth_q()
    U, n = 400, 120
    well = np.stack([_draw_hist(rng, 1.5, 2.0, 0.5, q, n) for _ in range(U)])
    m = mle_location(well, 2.0, 0.5, q)
    ok = abs(m["theta"] - 1.5) < 4 * m["se"] + 0.05 and 0.7 < m["vif"] < 1.4
    over = np.stack([_draw_hist(rng, 1.5 + rng.normal(0, 1.5), 2.0, 0.5, q, n) for _ in range(U)])
    mo = mle_location(over, 2.0, 0.5, q)
    ok &= mo["vif"] > 2.0 and mo["se"] > 1.5 * m["se"]                  # overdispersion shows in the sandwich, not in A
    return bool(ok)


def _ut_pda_nis() -> bool:
    rng = np.random.default_rng(3)
    q = _smooth_q()
    b, s, beta = 2.0, 2.0, 0.5
    truth = rng.uniform(-8, 8, 600)
    H = np.stack([_draw_hist(rng, t + b, s, beta, q, 150) for t in truth])
    mu, var = pda_posterior(H, s, beta, q, b)
    ok = float(np.mean(np.abs(mu - truth))) < 0.6
    nis = (mu - truth) ** 2 / var
    lo, hi = nis_band(len(nis))
    ok &= lo <= float(np.median(nis)) <= hi
    ok &= abs(float(np.polyfit(truth, mu, 1)[0]) - 1.0) < 0.1
    # a profile with no events at the right place (clutter only) must be wrong with high NIS
    Hc = np.stack([_draw_hist(rng, 0.0, s, 0.999, q, 150) for _ in range(300)])
    muc, varc = pda_posterior(Hc, s, beta, q, b)
    return bool(ok and abs(float(np.polyfit(truth[:300], muc, 1)[0])) < 0.2)


def _ut_decomposition() -> bool:
    rng = np.random.default_rng(4)
    S = 300
    ang = rng.uniform(0, 2 * np.pi, S)
    meta = np.zeros((S, 10))
    meta[:, 2], meta[:, 3] = np.cos(ang), np.sin(ang)
    meta[:, 4] = rng.uniform(-30, 30, S)
    meta[:, 5] = rng.uniform(-60, 60, S)
    meta[:, 6] = rng.normal(0, 0.1, S)
    w = np.array([1.5, -0.8, -0.05, -40.0])
    y = design(meta, "geom") @ w
    c = ols(design(meta, "geom"), y)
    ok = np.allclose(c, w, atol=1e-8) and abs(r2(y, design(meta, "geom") @ c)[0] - 1.0) < 1e-9
    yn = y + rng.normal(0, 0.3, S)
    ok &= r2(yn, design(meta, "geom") @ ols(design(meta, "geom"), yn))[0] > 0.9
    ok &= r2(rng.normal(0, 3, S), design(meta, "geom") @ c)[0] < 0.5       # an unrelated bias is not "explained"
    return bool(ok)


def _ut_peak() -> bool:
    x = np.arange(-16, 17, dtype=np.float64)
    h = 200 * np.exp(-0.5 * ((x - 2.3) / 1.6) ** 2) + 5
    pk, ratio = peak_of(h)
    flat = peak_of(np.full(33, 30.0))[1]
    return bool(abs(pk - 2.3) < 0.15 and ratio > 5 and flat < 1.01)


def _ut_shuffles() -> bool:
    rng = np.random.default_rng(5)
    n = 4000
    layer = np.sort(rng.integers(0, 150, n))
    lt = np.sort(rng.uniform(-150, 50, 150)).astype(np.float32)
    ev = dict(tau=lt[layer], layer=layer, pol=rng.integers(0, 2, n).astype(np.uint8))
    t1 = shuffle_tau(ev, "layer", rng_for("ut", 1))
    ok = np.array_equal(np.sort(np.unique(t1)), np.sort(np.unique(ev["tau"])))
    ok &= all(len(np.unique(t1[layer == l])) == 1 for l in np.unique(layer))      # events of a layer stay together
    ok &= not np.array_equal(t1, ev["tau"])
    t2 = shuffle_tau(ev, "event", rng_for("ut", 2))
    ok &= np.array_equal(np.sort(t2), np.sort(ev["tau"])) and not np.array_equal(t2, ev["tau"])
    ok &= np.array_equal(shuffle_tau(ev, "layer", rng_for("ut", 1)), t1)          # deterministic
    # the V200 pooled histogram does not see the time stamps; V50 does
    pos = np.stack([np.linspace(10, 60, 60), np.full(60, 40.0)], 1)
    nrm = np.tile(np.array([[0.0, -1.0]]), (60, 1))
    EX, EY = rng.uniform(10, 60, n).astype(np.float32), rng.uniform(30, 50, n).astype(np.float32)
    pi, pj, d = find_pairs(pos, nrm, EX, EY)
    cl = np.zeros(60, np.int64)
    a = hist_unit(cl, pi, pj, d, slab_of(ev["tau"]), ev["pol"])
    b = hist_unit(cl, pi, pj, d, slab_of(t1), ev["pol"])
    ok &= np.array_equal(a[0], b[0]) and not np.array_equal(a[1], b[1])
    # polarity nulls
    pf = hist_unit(cl, pi, pj, d, slab_of(ev["tau"]), (1 - ev["pol"]).astype(np.uint8))
    ok &= np.array_equal(a.sum(2), pf.sum(2)) and np.array_equal(a[:, :, 0], pf[:, :, 1])
    return bool(ok)


def _ut_segments() -> bool:
    rng = np.random.default_rng(6)
    pos = rng.uniform(0, 200, (400, 2))
    ang = rng.uniform(-np.pi, np.pi, 400)
    nrm = np.stack([np.cos(ang), np.sin(ang)], 1)
    cls = rng.integers(0, 3, 400)
    seg, S = segment_ids(pos, nrm, cls)
    seg2, S2 = segment_ids(pos, nrm, cls)
    perm = rng.permutation(400)
    return bool(S == S2 and np.array_equal(seg, seg2) and S > 20 and np.bincount(cls[perm], minlength=3).tolist()
                == np.bincount(cls, minlength=3).tolist())


def _synthetic_position_recs(rng, slope_pda: float, noise_pda: float, slope_ncp: float, fit: dict, n_samples: int = 160) -> Dict[str, List[dict]]:
    """Fake shifted-contour data: PDA histograms generated from the mixture at the true shift (slope_pda scales it);
    NCP baseline = slope_ncp * truth + noise.  Unit-test input only."""
    q = np.asarray(fit["local"]["sil"]["V200"]["q"])
    f = fit["local"]["sil"]["V200"]["fit"]
    recs = []
    for i in range(n_samples):
        S = 12
        st = rng.uniform(-7, 7, S)
        H = np.stack([_draw_hist(rng, -slope_pda * t + f["theta"] + rng.normal(0, noise_pda), f["s"], f["beta"], q, 200) for t in st])
        meta = np.zeros((S, 10), np.float32)
        meta[:, 0], meta[:, 7] = 0, st
        ncp = np.stack([-(slope_ncp * st + rng.normal(0, 0.5, S)) * 60, np.full(S, 60.0)], 1)
        sh = {mg: dict(dep=dict(meta=meta, hist=H.astype(np.float32), ncp=ncp), orc=dict(meta=meta, hist=H.astype(np.float32), ncp=ncp))
              for mg in SHIFT_MM}
        recs.append(dict(i=i, shift=sh))
    return {"ycy_local": recs[: n_samples // 2], "ylf_local": recs[n_samples // 2:]}


def _ut_position() -> bool:
    rng = np.random.default_rng(7)
    q = _smooth_q()
    fit = {"local": {"sil": {"V200": {"fit": dict(theta=1.0, s=2.0, beta=0.5), "q": q.tolist()}},
                     "int": {"V200": {"fit": None, "q": q.tolist()}}}}
    by = _synthetic_position_recs(rng, 0.95, 0.5, 0.45, fit)
    P = position_recovery(by, fit, "SELECT")[f"dep/{SHIFT_PRIMARY}"]
    ok = P["ci"][0] > 0 and abs(P["slope_pda"] - 0.95) < 0.1 and abs(P["slope_ncp"] - 0.45) < 0.1
    by0 = _synthetic_position_recs(rng, 0.45, 0.5, 0.45, fit)
    P0 = position_recovery(by0, fit, "SELECT")[f"dep/{SHIFT_PRIMARY}"]
    ok &= P0["ci"][0] <= 0 <= P0["ci"][1]
    byz = _synthetic_position_recs(rng, 0.0, 0.5, 0.45, fit)
    Pz = position_recovery(byz, fit, "SELECT")[f"dep/{SHIFT_PRIMARY}"]
    ok &= Pz["ci"][1] < 0 and abs(Pz["slope_pda"]) < 0.1                       # no information: PDA is worse than the baseline
    return bool(ok)


def _mk_class(b=0.5, b_med=0.5, dsig=0.01, vif=1.5, rpk=5.0, jit=1.0, njit=100, peak=0.5, r2u=0.8, drift=0.5, red=0.7,
              nis=0.5, lo=0.3, hi=0.7):
    return dict(insufficient=False, b_m=b, b_med=b_med, dsig=dsig, vif=vif, at_bound=False, r_pk=rpk, jitter=jit, n_jit_pkts=njit,
                peak=peak, r2u_geom=r2u, drift=drift, debias_red=red, nis_med=nis, nis_lo=lo, nis_hi=hi, sigma_hat_wrist=0.97)


def _mk_result(sil: dict, int_: Optional[dict] = None, pos=None, validity=None) -> dict:
    ins = {"insufficient": True}
    sel = ["ycy_local", "ycy_global", "ylf_local", "ylf_global"]
    by = {s: {"sil": dict(sil), "int": dict(int_) if int_ else dict(ins), "mat": dict(ins)} for s in sel}
    pos = pos or dict(delta=0.3, ci=[0.1, 0.5], slope_ncp=0.3, per_subject={"ycy": dict(delta=0.3), "ylf": dict(delta=0.3)})
    P = {f"dep/{SHIFT_PRIMARY}": pos, f"orc/{SHIFT_PRIMARY}": dict(pos)}
    return dict(validity=validity if validity is not None else dict(unit_tests=True), select_recs=sel, fit7_recs=[],
                by_rec=by, position=dict(SELECT=P))


def _ut_decide() -> bool:
    ok = True
    # a clean case passes
    ok &= decide(_mk_result(_mk_class()))["overall"] == "pass"
    # tool defect never passes
    ok &= decide(_mk_result(_mk_class(), validity=dict(unit_tests=False)))["overall"] == "tool_defect"
    # a large stable bias, calibrated and recovered: passes through E1 (bias stable and removable) but E5 would kill
    big = _mk_class(b=6.0, b_med=6.0, dsig=0.2, peak=6.0, r2u=0.7, drift=0.4)
    d = decide(_mk_result(big))
    ok &= d["e5"] == "kill" and d["overall"] == "kill"
    # peak not above the wrong-window null: no detectable class -> E4 kill
    ok &= decide(_mk_result(_mk_class(rpk=1.5)))["e4"] == "kill"
    # jitter too big for the only detectable class -> kill
    ok &= decide(_mk_result(_mk_class(jit=3.5)))["e4"] == "kill"
    # bias >= 4 px not explained by the nuisances -> kill
    ok &= decide(_mk_result(_mk_class(peak=5.0, r2u=0.2)))["e4"] == "kill"
    # drift of b_c above 3 px -> kill
    ok &= decide(_mk_result(_mk_class(drift=3.5)))["e1"] == "kill"
    # position gain CI includes 0 with a high baseline -> kill; with a low baseline -> descriptive
    p0 = dict(delta=0.01, ci=[-0.1, 0.1], slope_ncp=0.9, per_subject={"ycy": dict(delta=0.0), "ylf": dict(delta=0.0)})
    ok &= decide(_mk_result(_mk_class(), pos=p0))["e1"] == "kill"
    p1 = dict(delta=0.01, ci=[-0.1, 0.1], slope_ncp=0.3, per_subject={"ycy": dict(delta=0.0), "ylf": dict(delta=0.0)})
    d1 = decide(_mk_result(_mk_class(), pos=p1))
    ok &= d1["e1"] == "descriptive" and d1["overall"] == "descriptive"
    # oracle-only gain -> kill
    r = _mk_result(_mk_class(), pos=p1)
    r["position"]["SELECT"][f"orc/{SHIFT_PRIMARY}"] = dict(delta=0.3, ci=[0.1, 0.5], slope_ncp=0.3, per_subject={})
    ok &= decide(r)["e1"] == "kill"
    # NIS outside its band: not calibrated -> descriptive, never pass
    ok &= decide(_mk_result(_mk_class(nis=3.0)))["overall"] == "descriptive"
    # insufficient data (no usable class) is neither a pass nor a kill
    ins = _mk_result(_mk_class())
    for sq in ins["by_rec"]:
        ins["by_rec"][sq]["sil"] = {"insufficient": True}
    di = decide(ins)
    ok &= di["overall"] == "descriptive" and di["e4"] == "descriptive"
    # one SELECT recording violating a condition blocks the class (all-recordings rule)
    r = _mk_result(_mk_class())
    r["by_rec"]["ylf_global"]["sil"] = _mk_class(vif=9.0, drift=2.0, red=0.2)
    ok &= decide(r)["overall"] == "descriptive"
    # unbiased + well calibrated but the position gain is absent: not entered
    ok &= decide(_mk_result(_mk_class(), pos=dict(delta=0.2, ci=[-0.05, 0.4], slope_ncp=0.3,
                                                 per_subject={"ycy": dict(delta=0.2), "ylf": dict(delta=0.2)})))["overall"] == "descriptive"
    return bool(ok)


def _ut_injection(G: Geo) -> bool:
    """Synthetic events planted at a known outward offset from a real FIT7 pose's contour: the class estimators must
    recover the offset with the right sign (an orientation / phase / class-wiring test; no data statistics)."""
    sd = SeqData("lyq_local", G, EVERY, max_n=6)
    sc = G.scene(sd.ss.prev[2], sd.ss.gt[2], sd.ss.betas, sd.ss.K)
    cls_e, side_e, _ = G.edge_sets(sc["verts"][0], sc["uv"][0], sc["K"])
    pts = G.points(sc["uv"][0], sc["uv"][0], sc["uv"][1], cls_e, side_e, sc["c_palm"], sc["c_wrist"], sc["z"][0])
    rng = np.random.default_rng(8)
    sel = pts["cls"] == SIL
    P, N = pts["pos"][sel], pts["nrm"][sel]
    k = rng.integers(0, len(P), 30000)
    off = 3.0 + rng.normal(0, 1.2, len(k))
    EX = (P[k, 0] + off * N[k, 0]).astype(np.float32)
    EY = (P[k, 1] + off * N[k, 1]).astype(np.float32)
    pi, pj, d = find_pairs(pts["pos"], pts["nrm"], EX, EY)
    cl = pts["cls"][pi]
    h = np.bincount(fine_index(d[cl == SIL]), minlength=NB).astype(np.float64)
    q = np.full(NB, 1.0 / NB)
    f = fit_mixture(h, q)
    return bool(f and abs(f["theta"] - 3.0) < 0.5 and abs(f["s"] - 1.2) < 0.5)


def _ut_shift_sign(G: Geo) -> bool:
    """Events planted on the UNSHIFTED contour, model contour shifted by 14 mm: the recovered shift (PDA and nearest-point
    baseline) must regress on the segment's true normal shift with an O(1) positive slope (sign and unit wiring of E1 iii; the
    value depends on the assumed mixture, which is deliberately mis-specified here)."""
    sd = SeqData("lyq_local", G, EVERY, max_n=6)
    sc = G.scene(sd.ss.prev[2], sd.ss.gt[2], sd.ss.betas, sd.ss.K)
    cls_e, side_e, _ = G.edge_sets(sc["verts"][0], sc["uv"][0], sc["K"])
    pts = G.points(sc["uv"][0], sc["uv"][0], sc["uv"][1], cls_e, side_e, sc["c_palm"], sc["c_wrist"], sc["z"][0])
    rng = np.random.default_rng(9)
    sel = pts["cls"] == SIL
    P, N = pts["pos"][sel], pts["nrm"][sel]
    k = rng.integers(0, len(P), 60000)
    off = rng.normal(0, 0.8, len(k))
    n = len(k)
    ev = dict(X=(P[k, 0] + off * N[k, 0]).astype(np.float32), Y=(P[k, 1] + off * N[k, 1]).astype(np.float32),
              tau=np.zeros(n, np.float32), pol=np.zeros(n, np.uint8), layer=np.zeros(n, np.int64))
    cls_end, side_end, _ = G.edge_sets(sc["verts"][1], sc["uv"][1], sc["K"])
    dd = _shift_condition(G, sc, ev, slab_of(ev["tau"]), 14.0, 0.7, False, cls_end, side_end)
    meta, H, ncp = dd["meta"], dd["hist"].astype(np.float64), dd["ncp"].astype(np.float64)
    ok = (H.sum(1) >= 30) & (ncp[:, 1] >= 10) & (meta[:, 0] == SIL)
    st = meta[ok, 7].astype(np.float64)
    q = np.full(NB, 1.0 / NB)
    mu, _ = pda_posterior(H[ok], 1.0, 0.3, q, 0.0)
    sp = float(np.polyfit(st, -mu, 1)[0])
    sn = float(np.polyfit(st, -ncp[ok, 0] / ncp[ok, 1], 1)[0])
    return bool(ok.sum() >= 15 and st.min() < -2.0 and st.max() > 2.0 and 0.5 < sp < 1.5 and 0.4 < sn < 1.3)   # sign + scale wiring


def _ut_vn_units(G: Geo) -> bool:
    """An event planted where the GT end pose puts a contour point lies at the offset v_n * 50 ms (outward positive) of the
    previous-state contour: units, sign and offset convention of the velocity regressor."""
    sd = SeqData("lyq_global", G, EVERY, max_n=8)
    sc = G.scene(sd.ss.prev[3], sd.ss.gt[3], sd.ss.betas, sd.ss.K)
    cls_e, side_e, _ = G.edge_sets(sc["verts"][0], sc["uv"][0], sc["K"])
    pts = G.points(sc["uv"][0], sc["uv"][0], sc["uv"][1], cls_e, side_e, sc["c_palm"], sc["c_wrist"], sc["z"][0])
    sel = np.nonzero(pts["cls"] == SIL)[0]
    P, N, vn = pts["pos"][sel], pts["nrm"][sel], pts["vn"][sel]
    ex = (P + (vn * 50.0)[:, None] * N).astype(np.float32)
    pi, pj, d = find_pairs(pts["pos"], pts["nrm"], ex[:, 0], ex[:, 1])
    own = pi == sel[pj]
    return bool(own.sum() >= 0.9 * len(sel) and np.abs(d[own] - (vn * 50.0)[pj[own]]).max() < 1e-3 and np.abs(vn).max() > 0.01)


def _ut_decomp_real_geometry(G: Geo) -> bool:
    """Known nuisance offsets (translation, scale about the palm centre, latency x velocity) injected as synthetic events on
    real FIT7 contours (poses only, the recording's own events are not read) with ideal association; the production segment
    pipeline (segment histograms -> mode peaks -> design -> OLS) must recover them and explain the bias energy."""
    sd = SeqData("lyq_global", G, EVERY, max_n=70)
    w = np.array([1.2, -0.7, -0.04, -30.0])
    rng = np.random.default_rng(10)
    recs = []
    for i in range(1, 60):
        sc = G.scene(sd.ss.prev[i], sd.ss.gt[i], sd.ss.betas, sd.ss.K)
        cls_e, side_e, _ = G.edge_sets(sc["verts"][0], sc["uv"][0], sc["K"])
        pts = G.points(sc["uv"][0], sc["uv"][0], sc["uv"][1], cls_e, side_e, sc["c_palm"], sc["c_wrist"], sc["z"][0])
        sel = np.nonzero(pts["cls"] == SIL)[0]
        if len(sel) < 20:
            continue
        k = np.repeat(sel, 40)
        X = np.stack([pts["nrm"][k, 0], pts["nrm"][k, 1], pts["rad"][k], pts["vn"][k]], 1)
        off = X @ w + rng.normal(0, 0.8, len(k))
        EX = (pts["pos"][k, 0] + off * pts["nrm"][k, 0]).astype(np.float32)
        EY = (pts["pos"][k, 1] + off * pts["nrm"][k, 1]).astype(np.float32)
        pi, pj, d = find_pairs(pts["pos"], pts["nrm"], EX, EY)
        own = pi == k[pj]                                  # ideal association: the strip pairs of the planting point only
        pi, pj, d = pi[own], pj[own], d[own]
        seg, S = segment_ids(pts["pos"], pts["nrm"], pts["cls"])
        H = hist_seg(seg, S, pi, pj, d, np.full(len(EX), 6, np.int64))
        recs.append(dict(seg=dict(meta=segment_meta(pts, seg, S), hist=H)))
    rows = seg_rows(recs, SIL, n_min=40)
    if len(rows["p200"]) < 100:
        return False
    Xd = design(rows["meta"], "geom")
    co = ols(Xd, rows["p200"])
    r2u = r2(rows["p200"], Xd @ co)[0]
    return bool(r2u > 0.8 and abs(co[2] - w[2]) < 0.02 and abs(co[3] - w[3]) < 12.0)


G_VIF_SYNTH = 3.0


class _SynthSeq(SeqData):
    """Real FIT7 poses (labels), SYNTHETIC events: even packets plant `SIL` events at a known outward offset of the previous-state
    silhouette (Gaussian, random times / polarities / 0.93 ms layers) plus uniform clutter; odd packets are clutter only.  Used only
    by the plumbing unit test of extract -> fit -> evaluate."""
    OFFSET, WIDTH = 3.0, 1.2

    def events(self, i: int) -> dict:
        ss = self.ss
        rng = np.random.default_rng(1000 + int(i))
        sc = self.G.scene(ss.prev[i], ss.gt[i], ss.betas, ss.K)
        cls_e, side_e, _ = self.G.edge_sets(sc["verts"][0], sc["uv"][0], sc["K"])
        pts = self.G.points(sc["uv"][0], sc["uv"][0], sc["uv"][1], cls_e, side_e, sc["c_palm"], sc["c_wrist"], sc["z"][0])
        lo, hi = pts["pos"].min(0) - 20, pts["pos"].max(0) + 20
        n_cl = 6000
        X = [rng.uniform(lo[0], hi[0], n_cl)]
        Y = [rng.uniform(lo[1], hi[1], n_cl)]
        if i % 2 == 0:
            k = np.repeat(np.nonzero(pts["cls"] == SIL)[0], 25)
            off = self.OFFSET + rng.normal(0, self.WIDTH, len(k))
            tan = rng.uniform(-0.5, 0.5, len(k))
            X.append(pts["pos"][k, 0] + off * pts["nrm"][k, 0] - tan * pts["nrm"][k, 1])
            Y.append(pts["pos"][k, 1] + off * pts["nrm"][k, 1] + tan * pts["nrm"][k, 0])
        X, Y = np.concatenate(X).astype(np.float32), np.concatenate(Y).astype(np.float32)
        n = len(X)
        tau = np.sort(rng.uniform(-150.0, 50.0, n)).astype(np.float32)
        layer = np.floor((tau + 150.0) / 0.93).astype(np.int64)
        tau = ((layer + 0.5) * 0.93 - 150.0).astype(np.float32)             # events of a layer share one time stamp
        perm = rng.permutation(n)                                            # time independent of position
        return dict(X=X, Y=Y, tau=tau[perm], pol=rng.integers(0, 2, n).astype(np.uint8), layer=layer[perm],
                    end=int(ss.ends[i]), window=200)


def _ut_pipeline_synthetic(G: Geo) -> bool:
    """extract_sample -> fit_constants -> eval_recording on real FIT7 poses with planted synthetic events: the class estimators,
    the null contrasts and the invariance checks must behave as built (offset recovered, peak above the wrong-window null,
    validity flags True)."""
    sd = _SynthSeq("lyq_local", G, EVERY, max_n=92)
    recs = []
    for i in range(2, 90, 2):
        r = extract_sample(G, sd, i, i + 1, False)
        if r is not None:
            recs.append(r)
    if len(recs) < 35:
        return False
    by = {"lyq_local": recs}
    v = validity_checks(by)
    if not all(x for x in v.values() if isinstance(x, bool)):
        return False
    fit = fit_constants(by)
    e = eval_recording("lyq_local", recs, fit)["sil"]
    if e.get("insufficient"):
        return False
    ok = abs(e["b_m"] - _SynthSeq.OFFSET) < 0.4 and abs(e["peak"] - _SynthSeq.OFFSET) < 1.0 and e["r_pk"] > 2.0
    ok &= abs(e["b_med"] - _SynthSeq.OFFSET) < 1.2 and e["drift"] < 0.5 and 0.0 < e["dsig"] < 0.3
    ok &= e["pk_real"] > 2.0 * e["pk_wrong"] and abs(e["pk_real"] - e["pk_tshuf"]) < 1e-9        # V200 is time-blind
    ok &= abs(e["kappa_real"]) < 0.5 or not np.isfinite(e["kappa_real"])                         # events do not follow motion here
    # the gate quantities are achievable when the events are independent and well specified (no gate is replaced by this)
    ok &= e["vif"] < G_VIF_SYNTH and e["jitter"] < GATE["jitter"] and e["nis_lo"] <= e["nis_med"] <= e["nis_hi"]
    ok &= e["nis_med_wrong"] > 5.0                                                               # wrong-window events are not calibrated
    return bool(ok)


def run_unit_tests(G: Optional[Geo] = None, verbose: bool = False) -> bool:
    tests = {"lut_phase": lambda: _ut_lut(G), "pairs_geometry": _ut_pairs, "mixture_roundtrip": _ut_mixture,
             "sandwich_vif": _ut_sandwich, "pda_nis": _ut_pda_nis, "decomposition": _ut_decomposition, "peak": _ut_peak,
             "shuffles_nulls": _ut_shuffles, "segments": _ut_segments, "position_slope": _ut_position, "decide": _ut_decide}
    if G is not None:
        tests["injection_real_contour"] = lambda: _ut_injection(G)
        tests["shift_sign_real_contour"] = lambda: _ut_shift_sign(G)
        tests["vn_units_real_contour"] = lambda: _ut_vn_units(G)
        tests["decomp_real_contour"] = lambda: _ut_decomp_real_geometry(G)
        tests["pipeline_synthetic_events"] = lambda: _ut_pipeline_synthetic(G)
    res = {}
    for k, fn in tests.items():
        try:
            res[k] = bool(fn())
        except Exception as ex:                      # a crashing test is a failing test
            res[k] = False
            if verbose:
                print(f"  {k}: exception {type(ex).__name__}: {ex}")
        if verbose:
            print(f"  unit {k:<24s} {'PASS' if res[k] else 'FAIL'}")
    return all(res.values())


# ------------------------------------------------------------------------------------------------ entry points
def to_py(o):
    """JSON-safe copy (numpy scalars / arrays -> python)."""
    if isinstance(o, dict):
        return {str(k): to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_py(v) for v in o]
    if isinstance(o, np.ndarray):
        return to_py(o.tolist())
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


class _relaxed:
    """Smoke only: shrink the minimum-support constants so the tiny slice exercises the full analysis code path."""
    KEYS = dict(N_SEG_MIN=8, N_SLAB_MIN=3, N_UNIT_MIN=3, N_FIT_MIN=50, N_JIT_PKT_MIN=2, N_NCP_MIN=3)

    def __enter__(self):
        g = globals()
        self.old = {k: g[k] for k in self.KEYS}
        g.update(self.KEYS)

    def __exit__(self, *a):
        globals().update(self.old)


def smoke() -> int:
    t_all = time.time()
    out_dir = REPO / "outputs" / "mv2" / "smoke" / "m4_contour"
    flags: Dict[str, object] = {}
    G = Geo()
    print("[m4 smoke] unit tests (synthetic data only)")
    flags["unit_tests"] = run_unit_tests(G, verbose=True)
    # --- tiny real-data slice: FIT7 lyq_local, base recording, never SELECT / zgz
    sd = SeqData("lyq_local", G, EVERY, max_n=14)
    t0 = time.time()
    recs = []
    for i in range(1, 13):
        r = extract_sample(G, sd, i, (i + 7) % len(sd), with_shift=(i % 6 == 0))
        if r is not None:
            recs.append(r)
    dt = time.time() - t0
    r0 = recs[0]
    sh = [r for r in recs if r["shift"]][0]["shift"]
    print(f"[m4 smoke] slice lyq_local: {len(sd)} packets at stride {EVERY}, extracted {len(recs)}, "
          f"{dt / max(len(recs), 1):.2f} s / packet (with shifts on {sum(1 for r in recs if r['shift'])})")
    print(f"[m4 smoke] shapes: unit {{null: {r0['unit']['real'].shape}}} x {len(r0['unit'])} nulls, "
          f"seg hist {r0['seg']['hist'].shape} meta {r0['seg']['meta'].shape} wrong {r0['seg']['hist_wrong'].shape}, "
          f"shift mags {sorted(sh)} dep seg {sh[SHIFT_MM[0]]['dep']['hist'].shape}")
    print(f"[m4 smoke] counts: events/packet {int(np.mean([r['n_events'] for r in recs]))}, points/class "
          f"{np.mean([r['n_pts'] for r in recs], 0).round(0).astype(int).tolist()}, pairs/packet "
          f"{int(np.mean([r['n_pairs'] for r in recs]))}, segments/packet {int(np.mean([r['n_seg'] for r in recs]))}")
    by = {"lyq_local": recs}
    v = validity_checks(by)
    for k, x in v.items():
        if isinstance(x, bool):
            print(f"  validity {k:<28s} {'PASS' if x else 'FAIL'}")
    flags["validity"] = {k: x for k, x in v.items() if isinstance(x, bool)}
    # --- analysis code path with relaxed minima (nothing from it is printed beyond keys / finiteness)
    with _relaxed():
        fit = fit_constants(by)
        rec_out = eval_recording("lyq_local", recs, fit)
        pos = position_recovery(by, fit, "FIT7")
        result = dict(protocol=protocol_dict(), fit=fit, by_rec={"lyq_local": rec_out}, position={"SELECT": pos, "FIT7": pos},
                      validity=dict(v, unit_tests=flags["unit_tests"]), select_recs=["lyq_local"], fit7_recs=[],
                      n_samples={"lyq_local": len(recs)})
        dec = decide(result)
    res_json = json.dumps(to_py(result))
    print(f"[m4 smoke] analysis path ran: by_rec keys {sorted(k for k in rec_out if not k.startswith('_'))}, position keys "
          f"{len(pos)}, result json {len(res_json) / 1e3:.0f} kB, decide keys {sorted(dec.keys())}")
    flags["analysis_path"] = True
    flags["json_ok"] = True
    flags["seconds"] = round(time.time() - t_all, 1)
    flags["counts"] = dict(packets=len(recs), events=int(sum(r["n_events"] for r in recs)), pairs=int(sum(r["n_pairs"] for r in recs)))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke_receipt.json").write_text(json.dumps(to_py(dict(flags=flags, sha=C.sha(__file__))), indent=1))
    ok = bool(flags["unit_tests"]) and all(flags["validity"].values())
    print(f"[m4 smoke] {'PASS' if ok else 'FAIL'} in {flags['seconds']} s (no effect statistic is printed or stored)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=str(REPO / "outputs" / "mv2" / "m4_contour"))
    ap.add_argument("--stage", choices=("extract", "analyze", "all"), default="all")
    ap.add_argument("--workers", type=int, default=min(96, max(1, (os.cpu_count() or 2) // 2)))
    args = ap.parse_args()
    if args.smoke:
        return smoke()
    C.verify_protocol()                                  # exits until outputs/mv2/protocol.json (the MV2 freeze) exists
    out = Path(args.out)
    cache = out / "cache"
    if args.stage in ("extract", "all"):
        if list(cache.glob("*.pkl.gz")):
            raise SystemExit(f"{cache} already holds extracted packets; preserve it (use --stage analyze)")
        t0 = time.time()
        done = extract_all(out, args.workers)
        C.save_json(out / "extract_log.json", dict(chunks=done, seconds=time.time() - t0))
    if args.stage in ("analyze", "all"):
        ut = run_unit_tests(Geo())
        by = load_cache(cache)
        result = analyze(by, ut)
        result["decision"] = decide(result)
        C.save_json(out / "result.json", to_py(result))
        C.receipt(out, dict(stage=args.stage, unit_tests=ut, decision=result["decision"]["overall"]),
                  scripts=["tools/mv2/m4_contour.py", "semkine/sba.py", "semkine/sba_emission.py", "semkine/kssf.py"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
