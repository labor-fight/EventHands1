#!/usr/bin/env python3
"""MV2 / M8 (mesh physically-consistent self-supervision, MCS) zero-training CPU probes.

    SRCA  source-rate ceiling audit (R3 source P1-E4-R1, prerequisite of M8): (i) a synthetic frequency-response closed loop -- a
          rendered MANO hand with known > 30 Hz motion components, an ESIM-like (1 kHz, per-pixel crossing times) and a v2e-like
          (60 Hz frames, motion-aware interpolation, common time layers) event generator, and the IWE contrast response to the
          sub-frame residual against the source-interval shuffle null; (ii) the sub-frame residual RMS (mm) of the FIT7 1 ms labels
    TBC   event-label time-base closure (P1-E10-R1): T0 metadata e(n) = stamp_n - t0 - n / avg_fps times the FK image-plane speed;
          T1 event-rate x FK-predicted-rate cross-correlation, run only if T0 says nuisance
    PF    IWE loss landscape (M8 PF, R3 revised): multi-reference contrast versus +-5 / 10 mm time-varying anatomical perturbations
          at 100 / 300 / 600 ms, correct-gradient-direction rate, cluster-permutation null, equal-event / static / noise / flash
          controls, collapse check with the Shiba-style area normalisation

    python tools/mv2/m8_probes.py --probe srca --smoke        (also tbc, pf)
    python tools/mv2/m8_probes.py --probe srca --out outputs/mv2/srca          (formal: needs the freeze)

`decide(result)` applies exactly the gates of tools/mv2/prereg_m8.md. Smoke mode prints shapes / counts / timings and pass / fail of
internal unit checks only. CPU only, base recordings of FIT7 / SELECT only, never zgz.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mv2 import common as MC  # noqa: E402
from mv2 import iwe as I  # noqa: E402
from mv2 import m3_probes as M3  # noqa: E402
from sba import r3_probe as PR  # noqa: E402

SEED = 20261026
FIT7, SELECT = MC.FIT7, MC.SELECT
RAW = Path("/data1/lyq/data/hand_data/hand_data")
BOOT = M3.GroupBoot
FINGERS5, LONG4 = M3.FINGERS5, M3.LONG4
ci = M3.ci
domain_of = M3.domain_of
MDE_LOCAL, MDE_GLOBAL = 1.149, 0.793                     # KS0 planning MDEs (mm), ledger #kstar-results-20261025 (not a gate)

# ================================================================================================ SRCA constants
SRCA_LEVELS_MM = (0.4, 0.8, 1.149, 2.3, 4.6)                # requested residual RMS (mm) of the synthetic trajectories
SRCA_N_TRAJ = 40                                         # paired base trajectories (each evaluated at every level)
SRCA_DURATION_MS = 400
SRCA_FPS = 60.0                                          # source frame rate of the synthetic video
SRCA_INTERP = 9                                          # sub-frame steps per source interval (v2e: ceil(16.7 ms / 2 ms))
SRCA_THETA = 0.15                                        # contrast threshold (log intensity), the data's v2e value
SRCA_EPS_LOG = 0.05                                      # log(eps + I)
SRCA_HF_BAND = (32.0, 58.0)                              # Hz, above the 60 Hz Nyquist of 30 Hz
SRCA_N_CAP = 20000
SRCA_NULL_DRAWS = 2
SRCA_MIN_MOTION_PX = 6.0                                 # reject base trajectories whose hand moves less than this (px)
LABEL_RMS_PASS_MM, LABEL_RMS_BLIND_MM = 1.149, 0.793     # plan thresholds (KS0 MDE local / global), see prereg
LABEL_MIN_RUN = 48                                       # frames of a continuous valid and non-interpolated run (Stage 0 A2)
LABEL_STEP_MS = 2


# ================================================================================================ renderer
class Renderer:
    """Z-buffered Gouraud rasteriser of the MANO mesh on the 240 x 180 render grid (integer pixel coordinates = pixel centres):
    Lambertian shading of a fixed directional light times a smooth 3-D albedo texture attached to the template surface (so the
    texture moves with the material points)."""

    S = 12                                                # stencil: faces whose bounding box exceeds S pixels are skipped

    def __init__(self, hand: "I.Hand", seed: int = SEED):
        g = I.geo()
        self.hand = hand
        self.faces = g["faces"]
        v0 = g["ctx"].mano.v_template.float()
        lo, hi = v0.min(0).values, v0.max(0).values
        r = (v0 - lo) / (hi - lo)
        rng = np.random.default_rng(seed)
        ph = torch.as_tensor(rng.uniform(0, 2 * math.pi, 6), dtype=torch.float32)
        self.albedo = 0.55 + 0.20 * torch.sin(2 * math.pi * 4 * r[:, 0] + ph[0]) * torch.sin(2 * math.pi * 4 * r[:, 1] + ph[1]) \
            * torch.sin(2 * math.pi * 3 * r[:, 2] + ph[2]) + 0.08 * torch.sin(2 * math.pi * 9 * r[:, 1] + ph[3])
        self.light = torch.as_tensor([0.3, -0.4, -1.0])
        self.light = self.light / self.light.norm()
        o = torch.arange(self.S, dtype=torch.float32)
        self.ox, self.oy = torch.meshgrid(o, o, indexing="xy")
        self.bg = 0.12

    @torch.no_grad()
    def render(self, verts: torch.Tensor) -> torch.Tensor:
        """verts (778, 3) camera frame -> intensity image (H, W) in [0, 1]."""
        from semkine.xyz_mesh import vertex_normals
        faces = self.faces
        n = vertex_normals(verts[None], faces)[0]
        shade = 0.35 + 0.65 * (n * self.light).sum(-1).abs()
        col = (self.albedo * shade).clamp(0, 1)
        uv = self.hand.uv(verts[None])[0]
        z = verts[:, 2]
        tri = uv[faces]                                    # (F, 3, 2)
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        lo = torch.floor(tri.min(1).values)
        hi = torch.ceil(tri.max(1).values)
        ok = ((hi - lo) < self.S).all(-1)
        px = lo[:, 0, None, None] + self.ox[None]
        py = lo[:, 1, None, None] + self.oy[None]
        p = torch.stack([px, py], -1)                      # (F, S, S, 2)
        v0, v1 = (b - a)[:, None, None], (c - a)[:, None, None]
        v2 = p - a[:, None, None]
        d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
        d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
        den = d00 * d11 - d01 * d01
        good = den.abs() > 1e-9
        den = torch.where(good, den, torch.ones_like(den))
        bv = (d11 * d20 - d01 * d21) / den
        bw = (d00 * d21 - d01 * d20) / den
        bu = 1 - bv - bw
        area = (b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0]
        inside = (bu >= -1e-4) & (bv >= -1e-4) & (bw >= -1e-4) & good & ok[:, None, None] & (area > 0)[:, None, None]
        pxi, pyi = px.round().long(), py.round().long()
        inb = (pxi >= 0) & (pxi < I.W) & (pyi >= 0) & (pyi < I.H)
        m = inside & inb
        f_idx = faces[:, None, None, :].expand(-1, self.S, self.S, -1)
        zz = (bu[..., None] * 0 + 0)                       # placeholder to keep shapes explicit
        zc = bu * z[f_idx[..., 0]] + bv * z[f_idx[..., 1]] + bw * z[f_idx[..., 2]]
        cc = bu * col[f_idx[..., 0]] + bv * col[f_idx[..., 1]] + bw * col[f_idx[..., 2]]
        pix = (pyi * I.W + pxi)[m]
        zc, cc = zc[m], cc[m]
        zbuf = torch.full((I.H * I.W,), float("inf"))
        zbuf.scatter_reduce_(0, pix, zc, reduce="amin", include_self=True)
        win = zc <= zbuf[pix] + 1e-7
        img = torch.full((I.H * I.W,), self.bg)
        img[pix[win]] = cc[win]
        return img.view(I.H, I.W)


# ================================================================================================ event simulation
def simulate_events(logI: torch.Tensor, t_ms: np.ndarray, theta: float, layers: bool):
    """Threshold-crossing events of a log-intensity sequence logI (K, H, W) sampled at t_ms (K,).
    layers=True (v2e): per consecutive pair the maximum crossing count m over the image defines m common time layers
    t_a + i (t_b - t_a) / m, every pixel with n >= i emits in layer i; layers=False (ESIM-like): per-pixel crossing times by linear
    interpolation. Returns (x, y, p, t_us) int arrays sorted by time."""
    ref = logI[0].clone()
    xs, ys, ps, ts = [], [], [], []
    for k in range(1, logI.shape[0]):
        dL = logI[k] - ref
        n = torch.floor(dL.abs() / theta).long()
        if int(n.max()) == 0:
            continue
        sg = torch.sign(dL).long()
        ta, tb = float(t_ms[k - 1]), float(t_ms[k])
        m = int(n.max())
        if layers:
            for i in range(1, m + 1):
                yy, xx = torch.nonzero(n >= i, as_tuple=True)
                t = ta + i * (tb - ta) / m
                xs.append(xx.numpy())
                ys.append(yy.numpy())
                ps.append((sg[yy, xx] > 0).numpy().astype(np.uint8))
                ts.append(np.full(len(xx), t * 1000.0))
        else:
            yy, xx = torch.nonzero(n >= 1, as_tuple=True)
            nn = n[yy, xx]
            rep = torch.repeat_interleave(torch.arange(len(nn)), nn)
            first = torch.cumsum(nn, 0) - nn
            idx = torch.arange(len(rep)) - first[rep] + 1                  # crossing index 1..n of its pixel
            frac = idx.float() * theta / dL.abs()[yy, xx][rep]
            xs.append(xx[rep].numpy())
            ys.append(yy[rep].numpy())
            ps.append((sg[yy, xx][rep] > 0).numpy().astype(np.uint8))
            ts.append((ta + frac.numpy() * (tb - ta)) * 1000.0)
        ref = ref + sg * n * theta
    if not xs:
        z = np.zeros(0, np.int64)
        return z, z, np.zeros(0, np.uint8), z
    x, y, p, t = (np.concatenate(a) for a in (xs, ys, ps, ts))
    o = np.argsort(t, kind="stable")
    return x[o].astype(np.int64), y[o].astype(np.int64), p[o], np.rint(t[o]).astype(np.int64)


# ================================================================================================ synthetic trajectories
class SynthTraj:
    """x(t) = x_base + low-frequency components (1-6 Hz) + s * HF components (32-58 Hz) in 51-d parameter space, with the
    piecewise-linear (PL) interpolation of its 60 Hz samples, the stand-in for what a motion-aware frame interpolator can
    reconstruct from the source video."""

    def __init__(self, x_base: np.ndarray, hand: "I.Hand", seed: int, duration_ms: int = SRCA_DURATION_MS):
        rng = np.random.default_rng(seed)
        self.hand, self.duration = hand, duration_ms
        self.x_base = np.asarray(x_base, np.float64)
        low = np.zeros(51)
        ch = {"t": [0, 1], "r": [3, 4, 5], "f": list(rng.choice(np.arange(6, 51), 8, replace=False))}
        amp = np.zeros(51)
        amp[ch["t"]] = 0.02
        amp[ch["r"]] = 0.2
        amp[ch["f"]] = 0.45
        self.lf = [(c, amp[c] * rng.uniform(0.5, 1.0), rng.uniform(1.0, 6.0), rng.uniform(0, 2 * math.pi)) for c in
                   ch["t"] + ch["r"] + ch["f"]]
        hf_amp = {"t": 0.0015, "r": 0.012, "f": 0.05}
        hf_ch = [0, 1, 3, 4] + list(rng.choice(ch["f"], 3, replace=False))
        self.hf = []
        for c in hf_ch:
            kind = "t" if c < 3 else ("r" if c < 6 else "f")
            for _ in range(2):
                self.hf.append((c, hf_amp[kind] * rng.uniform(0.6, 1.0), rng.uniform(*SRCA_HF_BAND), rng.uniform(0, 2 * math.pi)))
        self.scale = 0.0

    def x_at(self, t_ms: np.ndarray, hf_scale: Optional[float] = None) -> np.ndarray:
        """(n, 51) true poses at times t_ms (ms)."""
        t = np.asarray(t_ms, np.float64)[:, None] / 1000.0
        X = np.repeat(self.x_base[None], len(t), 0)
        for c, a, f, ph in self.lf:
            X[:, c] += a * np.sin(2 * math.pi * f * t[:, 0] + ph)
        s = self.scale if hf_scale is None else hf_scale
        for c, a, f, ph in self.hf:
            X[:, c] += s * a * np.sin(2 * math.pi * f * t[:, 0] + ph)
        return X

    def x_pl(self, t_ms: np.ndarray) -> np.ndarray:
        """Piecewise-linear interpolation of the 60 Hz samples of x(t) (parameter space)."""
        ts = np.arange(0.0, self.duration + 1000.0 / SRCA_FPS, 1000.0 / SRCA_FPS)
        Xs = self.x_at(ts)
        return np.stack([np.interp(np.asarray(t_ms, np.float64), ts, Xs[:, c]) for c in range(51)], 1)

    def residual_rms_mm(self, step_ms: float = 2.0) -> float:
        """RMS over times and the 778 vertices (mm) of FK(x(t)) - FK(PL(t))."""
        t = np.arange(0.0, self.duration + 1e-9, step_ms)
        va, _ = self.hand.fk(torch.as_tensor(self.x_at(t), dtype=torch.float32))
        vb, _ = self.hand.fk(torch.as_tensor(self.x_pl(t), dtype=torch.float32))
        return float(((va - vb).norm(dim=-1) ** 2).mean().sqrt() * 1000.0)

    def calibrate(self, target_mm: float) -> float:
        """HF scale s such that the residual RMS equals target_mm (rms^2 = c^2 + s^2 h^2: curvature floor c, HF part h)."""
        self.scale = 0.0
        c = self.residual_rms_mm()
        self.scale = 1.0
        r1 = self.residual_rms_mm()
        h2 = max(r1 ** 2 - c ** 2, 1e-12)
        s = math.sqrt(max(target_mm ** 2 - c ** 2, 0.0) / h2)
        self.scale = s
        return s

    def motion_px(self) -> float:
        """Peak mean-vertex image displacement (px) from t = 0 of the low-frequency trajectory (HF scale 0): the acceptance statistic of a
        base trajectory, identical for every level of the pair."""
        ts = np.arange(0.0, self.duration + 1e-9, 4.0)
        uv = self.hand.uv(self.hand.fk(torch.as_tensor(self.x_at(ts, hf_scale=0.0), dtype=torch.float32))[0])
        return float((uv - uv[0:1]).norm(dim=-1).mean(-1).max())

    def hf_energy_above_30hz(self, fs: float = 1000.0) -> float:
        """Share of the HF component's variance above 30 Hz (a unit-test quantity: the added components are > 30 Hz by design)."""
        t = np.arange(0, 1.0, 1.0 / fs)
        h = np.zeros(len(t))
        for c, a, f, ph in self.hf:
            h += a * np.sin(2 * math.pi * f * t + ph)
        F = np.abs(np.fft.rfft(h - h.mean())) ** 2
        fr = np.fft.rfftfreq(len(t), 1.0 / fs)
        return float(F[fr > 30.0].sum() / max(F.sum(), 1e-30))


def render_sequence(rend: "Renderer", X: np.ndarray) -> torch.Tensor:
    """Log intensities (K, H, W) of poses X (K, 51)."""
    verts, _ = rend.hand.fk(torch.as_tensor(X, dtype=torch.float32))
    imgs = torch.stack([rend.render(v) for v in verts])
    return torch.log(SRCA_EPS_LOG + imgs)


def srca_streams(rend: "Renderer", st: SynthTraj):
    """The two event streams of one synthetic trajectory: ESIM-like (1 kHz true trajectory, per-pixel times) and v2e-like (60 Hz
    samples of the true trajectory, pose-linear interpolation to SRCA_INTERP sub-frames, common time layers)."""
    dur = st.duration
    t_e = np.arange(0.0, dur + 1e-9, 1.0)
    esim = simulate_events(render_sequence(rend, st.x_at(t_e)), t_e, SRCA_THETA, layers=False)
    t_src = np.arange(0.0, dur + 1000.0 / SRCA_FPS, 1000.0 / SRCA_FPS)
    t_sub = np.arange(0.0, t_src[-1] + 1e-9, 1000.0 / SRCA_FPS / SRCA_INTERP)
    t_sub = t_sub[t_sub <= dur + 1000.0 / SRCA_FPS]
    v2e = simulate_events(render_sequence(rend, st.x_pl(t_sub)), t_sub, SRCA_THETA, layers=True)
    return {"esim": esim, "v2e": v2e}


def iwe_contrast_synth(hand: "I.Hand", st: SynthTraj, ev_arrays, hyp: str, t_us_override: Optional[np.ndarray] = None) -> float:
    """Multi-reference whole-hand IWE variance (start / mid / end references) of the events warped by the hypothesised trajectory
    `hyp` in {'true', 'pl'}; events thinned to SRCA_N_CAP with a fixed seed (the same events for every hypothesis)."""
    x, y, p, t_us = ev_arrays
    n = len(x)
    keep = I.thin_index(n, SRCA_N_CAP, np.random.default_rng(SEED))
    xy = np.stack([x[keep], y[keep]], 1).astype(np.float32)
    t_use = (t_us if t_us_override is None else t_us_override)[keep] / 1000.0
    ts = np.arange(0.0, st.duration + 1e-9, I.FK_STEP)
    X = st.x_at(ts) if hyp == "true" else st.x_pl(ts)
    traj = I.build_traj(hand, X.astype(np.float32), 0.0)
    asg = I.assign(traj, xy, t_use)
    box = I.box_from_points(np.concatenate([traj.uv[traj.vis].numpy(), xy]), 10.0)
    vals = []
    for tr in (0.0, st.duration / 2.0, float(st.duration)):
        d = I.displacement(traj, asg, t_use, tr)
        vals.append(I.contrast_stats((xy + d).astype(np.float32), box)["var"])
    return float(np.mean(vals))


def srca_one(job: dict) -> dict:
    """One (base trajectory, level): residual RMS, and for both pipelines the contrast responses (true-warp versus PL-warp, relative)
    for real event times and for the source-interval shuffle null."""
    torch.set_num_threads(1)
    rec = M3.RecCtx.get(job["seq"]).rec
    hand = rec.hand
    t0 = int(job["t_base"])
    x_base = np.asarray(rec.pos51[t0], np.float64)
    st = SynthTraj(x_base, hand, seed=job["seed"])
    s = st.calibrate(job["level_mm"])
    rms = st.residual_rms_mm()
    motion = st.motion_px()                                              # low-frequency base only: level independent
    out = {"seq": job["seq"], "traj": job["traj"], "level_mm": job["level_mm"], "hf_scale": s, "rms_mm": rms, "motion_px": motion}
    if motion < SRCA_MIN_MOTION_PX:
        out["rejected"] = True
        return out
    rend = Renderer(hand, seed=job["seed"])
    streams = srca_streams(rend, st)
    interval_us = 1e6 / SRCA_FPS
    for name, ev in streams.items():
        out[f"n_{name}"] = int(len(ev[0]))
        if len(ev[0]) < 500:
            out[f"resp_{name}"] = float("nan")
            continue
        c_t, c_p = (iwe_contrast_synth(hand, st, ev, h) for h in ("true", "pl"))
        out[f"resp_{name}"] = (c_t - c_p) / max(c_p, 1e-12)
        nulls = []
        for dr in range(SRCA_NULL_DRAWS):
            tsh = I.shuffle_times(ev[3], "interval", np.random.default_rng(I.seed_of(job["seq"], job["traj"], name, dr)), interval_us)
            ct, cp = (iwe_contrast_synth(hand, st, ev, h, tsh) for h in ("true", "pl"))
            nulls.append((ct - cp) / max(cp, 1e-12))
        out[f"null_{name}"] = float(np.mean(nulls))
    return out


def srca_jobs(n_traj: int = SRCA_N_TRAJ, levels=SRCA_LEVELS_MM) -> List[dict]:
    """Base trajectories: poses of FIT7 base recordings (one subject per trajectory, cycling), fixed seeds; each at every level."""
    jobs = []
    subs = list(FIT7)
    for i in range(n_traj):
        sub = subs[i % len(subs)]
        seq = f"{sub}_{'local' if (i // len(subs)) % 2 == 0 else 'global'}"
        rng = np.random.default_rng(I.seed_of("srca", i))
        rec_runs = M3.RecCtx.get(seq).rec.runs
        run = rec_runs[np.argmax(rec_runs[:, 1] - rec_runs[:, 0])]
        t_base = int(rng.integers(run[0] + 100, run[1] - 100))
        for lv in levels:
            jobs.append({"seq": seq, "traj": i, "t_base": t_base, "seed": int(I.seed_of("srca", i, "s")), "level_mm": float(lv)})
    return jobs


def srca_aggregate(rows: List[dict]) -> dict:
    """Response slope (relative contrast response per mm of residual RMS) of each pipeline, against the interval-shuffle null,
    OLS with intercept over (trajectory, level) pairs, group bootstrap over trajectories."""
    ok = [r for r in rows if not r.get("rejected") and "error" not in r]
    out = {"n_rows": len(rows), "n_used": len(ok), "pipelines": {}}
    gb_keys = np.asarray([f"{r['seq']}|{r['traj']}" for r in ok])
    for name in ("esim", "v2e"):
        x = np.asarray([r["rms_mm"] for r in ok])
        y = np.asarray([r.get(f"resp_{name}", np.nan) for r in ok])
        yn = np.asarray([r.get(f"null_{name}", np.nan) for r in ok])
        fin = np.isfinite(y) & np.isfinite(yn)
        if fin.sum() < 10:
            out["pipelines"][name] = {"error": "too few finite rows"}
            continue
        gb = BOOT(gb_keys[fin])
        res = {}
        for tag, v in (("real", y[fin]), ("null", yn[fin]), ("real_minus_null", y[fin] - yn[fin])):
            S = M3._slope_moments(x[fin], v)
            s_, _ = gb.gsum(S)
            pt = M3._slope_of(s_.sum(0))
            bt = np.asarray([M3._slope_of(w @ s_) for w in gb.W])
            res[f"slope_{tag}"] = {"value": pt, "ci": ci(bt)}
        # level-wise mean responses (descriptive)
        lv = {}
        for L_ in sorted({r["level_mm"] for r in ok}):
            m = np.asarray([r["level_mm"] == L_ for r in ok])[fin]
            lv[str(L_)] = {"rms_mm": float(np.mean(x[fin][m])), "resp": float(np.mean(y[fin][m])), "null": float(np.mean(yn[fin][m])), "n": int(m.sum())}
        res["by_level"] = lv
        res["n_events_median"] = float(np.median([r.get(f"n_{name}", 0) for r in ok]))
        out["pipelines"][name] = res
    return out


# ------------------------------------------------------------------------------------------------ SRCA (ii): label residual
def _import_ph():
    import importlib.util
    spec = importlib.util.spec_from_file_location("prepare_hand_data", str(REPO / "tools" / "prepare_hand_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def label_residual_core(hand: "I.Hand", pos_at, t_ms: np.ndarray, tr: np.ndarray, go: np.ndarray, hp: np.ndarray,
                        runs, min_run: int = LABEL_MIN_RUN, step_ms: int = LABEL_STEP_MS) -> dict:
    """Core of the label sub-frame residual: `pos_at(ms) -> (51,)` the 1 ms label, source samples (t_ms, translation tr (N, 3), root
    rotation vector go (N, 3), residual joint rotation vectors hp (N, 45)) and the continuous runs (frame index pairs). On every run
    of >= min_run source frames: FK(label_1ms(t)) - FK(PL(t)), PL = translation linear and every rotation geodesic (slerp) between
    consecutive source samples. Returns sums of squares for pooling and the RMS over vertices / joints (mm)."""
    from scipy.spatial.transform import Rotation as R, Slerp
    sq_v, n_v, sq_j, n_j, n_runs = 0.0, 0, 0.0, 0, 0
    for a, b in runs:
        if b - a < min_run:
            continue
        i0 = int(math.ceil(t_ms[a] - 1e-9))
        i1 = int(math.floor(t_ms[b - 1] + 1e-9))
        grid = np.arange(i0, i1 + 1, step_ms)
        if len(grid) < 4:
            continue
        lab = np.stack([np.asarray(pos_at(int(g)), np.float64) for g in grid])
        pl = np.zeros_like(lab)
        for c in range(3):
            pl[:, c] = np.interp(grid, t_ms[a:b], tr[a:b, c])
        rots = [go[a:b]] + [hp[a:b, 3 * k:3 * k + 3] for k in range(15)]
        for k, rv in enumerate(rots):
            sl = Slerp(t_ms[a:b], R.from_rotvec(rv))
            col = slice(3, 6) if k == 0 else slice(6 + 3 * (k - 1), 9 + 3 * (k - 1))
            pl[:, col] = sl(np.clip(grid, t_ms[a], t_ms[b - 1])).as_rotvec()
        va, ja = hand.fk(torch.as_tensor(lab, dtype=torch.float32))
        vb, jb = hand.fk(torch.as_tensor(pl, dtype=torch.float32))
        dv = (va - vb).norm(dim=-1).numpy() * 1000.0
        dj = (ja - jb).norm(dim=-1).numpy() * 1000.0
        sq_v += float((dv ** 2).sum())
        n_v += dv.size
        sq_j += float((dj ** 2).sum())
        n_j += dj.size
        n_runs += 1
    return {"n_runs": n_runs, "n_samples": n_v // 778 if n_v else 0,
            "rms_vertex_mm": math.sqrt(sq_v / n_v) if n_v else float("nan"),
            "rms_joint_mm": math.sqrt(sq_j / n_j) if n_j else float("nan"), "sum_sq_vertex": sq_v, "n_vertex": n_v}


def label_subframe_rms(seq: str) -> dict:
    """Sub-frame residual of the 1 ms labels of one BASE recording (`label_residual_core` on mesh.npz source samples; valid and
    non-interpolated frames, `prepare_hand_data.continuous_valid_runs`)."""
    PH = _import_ph()
    rc = M3.RecCtx.get(seq)
    rec = rc.rec
    z = np.load(RAW / MC.seq_dir(seq) / seq / "mesh" / f"{seq}_mesh.npz", allow_pickle=False)
    ts, valid, interp = z["timestamp_ns"], z["valid"], z["interpolated"]
    runs = PH.continuous_valid_runs(valid & ~interp)
    out = label_residual_core(rec.hand, lambda k: rec.pos51[k], (ts - ts[0]) / 1e6,
                              z["camera_transl_right_canonical_m"].astype(np.float64),
                              z["global_orient_rotvec"].astype(np.float64), z["hand_pose_residual_aa"].astype(np.float64), runs)
    out["seq"] = seq
    return out


def label_job(seq: str) -> dict:
    torch.set_num_threads(1)
    try:
        return label_subframe_rms(seq)
    except Exception as ex:
        return {"seq": seq, "error": f"{type(ex).__name__}: {ex}"}


def label_aggregate(rows: List[dict]) -> dict:
    ok = [r for r in rows if "error" not in r and r["n_vertex"] > 0]
    out = {"per_recording": {r["seq"]: {"rms_vertex_mm": r["rms_vertex_mm"], "rms_joint_mm": r["rms_joint_mm"], "n_runs": r["n_runs"]} for r in ok}}
    for name, sel in (("all", ok), ("local", [r for r in ok if domain_of(r["seq"]) == "local"]),
                      ("global", [r for r in ok if domain_of(r["seq"]) == "global"])):
        if sel:
            n = sum(r["n_vertex"] for r in sel)
            out[name] = {"rms_vertex_mm": math.sqrt(sum(r["sum_sq_vertex"] for r in sel) / n), "n_recordings": len(sel),
                         "median_recording_rms_mm": float(np.median([r["rms_vertex_mm"] for r in sel]))}
    return out


# ================================================================================================ decide (SRCA)
def decide_srca(result: dict) -> dict:
    """Gate of prereg_m8.md section 1. Tool validity: the ESIM-like positive control must respond (slope of the real-minus-null response
    on the residual RMS has CI lower bound > 0) and the label residual must be available. v2e-like response: significant iff the CI
    lower bound of its real-minus-null slope is > 0.
      kill        v2e response not significant (CI contains 0 or lies below 0): training-time self-supervision is dead; the extra tag
                  'blind_confirmed' if the pooled label sub-frame RMS < 0.793 mm
      pass        v2e response significant and label sub-frame RMS >= 1.149 mm: release in-training self-supervision to PF
      descriptive v2e response significant but label RMS < 1.149 mm"""
    agg = result["aggregate"]
    lab = result["label_residual"]["all"]["rms_vertex_mm"] if "all" in result["label_residual"] else float("nan")
    if agg.get("n_rows") and (agg["n_rows"] - agg["n_used"]) / agg["n_rows"] > 0.5:
        return {"probe": "SRCA", "verdict": "tool_defect", "detail": {"note": "more than half of the synthetic rows were rejected or failed"}}
    pe, pv = agg["pipelines"].get("esim", {}), agg["pipelines"].get("v2e", {})
    r = {"label_rms_mm": lab}
    if "slope_real_minus_null" not in pe or "slope_real_minus_null" not in pv or not np.isfinite(lab):
        return {"probe": "SRCA", "verdict": "tool_defect", "detail": dict(r, note="missing pipeline or label residual")}
    r["esim_slope"] = pe["slope_real_minus_null"]
    r["v2e_slope"] = pv["slope_real_minus_null"]
    if not (pe["slope_real_minus_null"]["ci"][0] > 0):
        return {"probe": "SRCA", "verdict": "tool_defect", "detail": dict(r, note="ESIM-like positive control does not respond")}
    sig = pv["slope_real_minus_null"]["ci"][0] > 0
    if not sig:
        r["blind_confirmed"] = bool(lab < LABEL_RMS_BLIND_MM)
        return {"probe": "SRCA", "verdict": "kill", "detail": r}
    if lab >= LABEL_RMS_PASS_MM:
        return {"probe": "SRCA", "verdict": "pass", "detail": r}
    return {"probe": "SRCA", "verdict": "descriptive", "detail": r}


# ================================================================================================ TBC
TBC_T0_NUISANCE_MM, TBC_T0_CLOSE_MM = 0.40, 0.08         # 0.5 x and 0.1 x MDE_global (0.793 mm), plan section 3 M8 (2)
TBC_POINTS = (0, 4, 8, 12, 16, 20)                       # OpenPose joints: wrist + 5 fingertips
TBC_DIFF_MS = 4                                          # central difference half-width for the FK speed
TBC_BLOCK_MS = 2000
TBC_MAX_LAG = 40
TBC_SMOOTH_MS = 9
TBC_RHO_MIN = 0.5                                        # corr(delta_hat, e) gate
TBC_N_PERM = 2000
TBC_STATIC_Q = 25.0                                      # blocks below this percentile of the recording's predicted rate are dropped
TBC_MIN_BLOCKS = 30


def _csv_stamps(seq: str):
    path = RAW / MC.seq_dir(seq) / seq / "event" / f"{seq}_source_timestamps.csv"
    stamps = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1,), dtype=np.int64)
    rel = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(2,), dtype=np.float64)
    return stamps, rel


def tbc_e_of(stamps_ns: np.ndarray):
    """(t_label_ms, e_ms, avg_fps): label time of frame n relative to the first frame, e(n) = label time - n / avg_fps, and the average
    frame rate (N - 1) / duration that v2e received as --input_frame_rate."""
    t0 = stamps_ns[0]
    t_l = (stamps_ns - t0) / 1e6
    n = len(stamps_ns)
    fps = (n - 1) / ((stamps_ns[-1] - t0) / 1e9)
    return t_l, t_l - np.arange(n) * 1000.0 / fps, float(fps)


def fk_speeds(rec: "I.Rec", t_ms: np.ndarray, half: int = TBC_DIFF_MS):
    """Image-plane speeds of the wrist + fingertips at label times t_ms (integer ms indices): max over the points and the mean, in
    mm / ms (camera-plane) and px / ms (render grid): central difference of the GT FK over +-half ms."""
    idx = np.rint(t_ms).astype(np.int64)
    a = np.clip(idx - half, 0, rec.n_ms - 1)
    b = np.clip(idx + half, 0, rec.n_ms - 1)
    dt = np.maximum(b - a, 1).astype(np.float64)
    Xa = torch.as_tensor(np.stack([np.asarray(rec.pos51[int(i)], np.float32) for i in a]))
    Xb = torch.as_tensor(np.stack([np.asarray(rec.pos51[int(i)], np.float32) for i in b]))
    _, ja = rec.hand.fk(Xa)
    _, jb = rec.hand.fk(Xb)
    ja, jb = ja[:, list(TBC_POINTS)], jb[:, list(TBC_POINTS)]
    mm = ((jb[..., :2] - ja[..., :2]).norm(dim=-1) * 1000.0).numpy() / dt[:, None]

    def proj(j):
        z = j[..., 2].clamp(min=1e-6)
        return torch.stack([rec.hand.fx * j[..., 0] / z + rec.hand.cx, rec.hand.fy * j[..., 1] / z + rec.hand.cy], -1)
    px = ((proj(jb) - proj(ja)).norm(dim=-1)).numpy() / dt[:, None]
    return {"mm_max": mm.max(1), "mm_mean": mm.mean(1), "px_max": px.max(1), "px_mean": px.mean(1)}


def tbc_t0(seq: str) -> dict:
    """T0 of one BASE recording: metadata only (CSV header stamps, aux t0) plus the GT FK speed; no events, no model."""
    rc = M3.RecCtx.get(seq)
    rec = rc.rec
    stamps, rel = _csv_stamps(seq)
    t_l, e, fps = tbc_e_of(stamps)
    ok_valid = np.zeros(len(t_l), bool)
    for ra, rb in rec.runs:
        ok_valid |= (t_l >= ra + 6) & (t_l <= rb - 7)
    sp = fk_speeds(rec, t_l[ok_valid])
    ae = np.abs(e[ok_valid])
    t_e = np.arange(len(t_l)) * 1000.0 / fps
    slope = float(np.polyfit(t_e / 1000.0, e, 1)[0])                    # ms of drift per second
    checks = {"n_frames_equals_csv_rows": bool(len(stamps) == len(rel)),
              "csv_relative_time_matches": bool(np.abs(rel - (stamps - stamps[0]) / 1e9).max() < 2e-6),
              "t0_equals_aux": bool(int(rec.aux["t0_ns"]) == int(stamps[0])),
              "e_zero_at_ends": bool(abs(e[0]) < 1e-9 and abs(e[-1]) < 1e-6)}
    return {"seq": seq, "n_frames": int(len(stamps)), "n_valid_frames": int(ok_valid.sum()), "avg_fps": fps,
            "e_abs_ms": {"max": float(np.abs(e).max()), "p95": float(np.percentile(np.abs(e), 95)), "median": float(np.median(np.abs(e)))},
            "e_drift_ms_per_s": slope, "checks": checks,
            "err_mm_max_speed": (ae * sp["mm_max"]).tolist(), "err_px_max_speed": (ae * sp["px_max"]).tolist(),
            "err_mm_mean_speed": (ae * sp["mm_mean"]).tolist(), "err_px_mean_speed": (ae * sp["px_mean"]).tolist()}


def tbc_t0_job(seq: str) -> dict:
    torch.set_num_threads(1)
    try:
        return tbc_t0(seq)
    except Exception as ex:
        return {"seq": seq, "error": f"{type(ex).__name__}: {ex}"}


def tbc_t0_aggregate(rows: List[dict]) -> dict:
    ok = [r for r in rows if "error" not in r]
    out = {"n_recordings": len(ok), "n_failed": len(rows) - len(ok), "per_recording": {}, "pooled": {}}
    for r in ok:
        em = np.asarray(r["err_mm_max_speed"])
        out["per_recording"][r["seq"]] = {"avg_fps": r["avg_fps"], "e_abs_ms": r["e_abs_ms"], "e_drift_ms_per_s": r["e_drift_ms_per_s"],
                                          "err_mm_p95": float(np.percentile(em, 95)) if len(em) else float("nan"),
                                          "n_valid_frames": r["n_valid_frames"], "checks": r["checks"]}
    for name, sel in (("all", ok), ("local", [r for r in ok if domain_of(r["seq"]) == "local"]),
                      ("global", [r for r in ok if domain_of(r["seq"]) == "global"])):
        if not sel:
            continue
        d = {}
        for key in ("err_mm_max_speed", "err_px_max_speed", "err_mm_mean_speed", "err_px_mean_speed"):
            v = np.concatenate([np.asarray(r[key]) for r in sel])
            d[key] = {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "p99": float(np.percentile(v, 99)),
                      "max": float(v.max()), "n": int(len(v))}
        out["pooled"][name] = d
    out["checks_all_pass"] = bool(all(all(r["checks"].values()) for r in ok)) if ok else False
    return out


# ------------------------------------------------------------------------------------------------ TBC T1
def xcorr_lag(z_ev: np.ndarray, z_pred: np.ndarray, max_lag: int = TBC_MAX_LAG):
    """Lag delta (ms, sub-sample by a parabola) maximising sum_t z_ev(t) z_pred(t + delta) over |delta| <= max_lag and the peak
    normalised correlation. z_pred must extend max_lag samples beyond z_ev at both ends: len(z_pred) = len(z_ev) + 2 max_lag."""
    n = len(z_ev)
    zs = (z_ev - z_ev.mean()) / (z_ev.std() + 1e-12)
    c = np.empty(2 * max_lag + 1)
    for i, d in enumerate(range(-max_lag, max_lag + 1)):
        seg = z_pred[max_lag + d: max_lag + d + n]
        c[i] = float(np.dot(zs, (seg - seg.mean()) / (seg.std() + 1e-12)) / n)
    k = int(np.argmax(c))
    d = float(k - max_lag)
    if 0 < k < len(c) - 1:
        a, b, g = c[k - 1], c[k], c[k + 1]
        den = a - 2 * b + g
        if den < 0:
            d += 0.5 * (a - g) / den
    return d, float(c[k])


def smooth_box(x: np.ndarray, w: int = TBC_SMOOTH_MS) -> np.ndarray:
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


@torch.no_grad()
def predicted_rate(hand: "I.Hand", X: np.ndarray, t_first: int, n_ms: int) -> np.ndarray:
    """Predicted event-rate proxy (px^2 / ms swept by the visible occluding contour) of the GT trajectory at 1 ms for
    ms [t_first, t_first + n_ms): contour every 4 ms from semkine.sba.contour_entries, summed edge length x |normal velocity of the edge
    midpoint| (central difference over +-2 ms from the 2 ms FK samples), linearly interpolated to 1 ms. X (n, 51) at 2 ms from t_first - 2."""
    from semkine import sba as SB
    g = I.geo()
    traj = I.build_traj(hand, X, float(t_first - 2), dt=2.0)
    ks = np.arange(1, traj.n - 1, 2)
    on, _, _ = SB.contour_entries(traj.verts[ks], traj.uv[ks] / I.RS, g["topo"], g["grid"], g["groups"], 6)
    e0, e1 = g["topo"].edges[:, 0], g["topo"].edges[:, 1]
    r4 = np.zeros(len(ks))
    for j, k in enumerate(ks):
        m = on[j]
        if not bool(m.any()):
            continue
        a, b = traj.uv[k][e0[m]], traj.uv[k][e1[m]]
        mid_p = 0.5 * (traj.uv[k + 1][e0[m]] + traj.uv[k + 1][e1[m]])
        mid_m = 0.5 * (traj.uv[k - 1][e0[m]] + traj.uv[k - 1][e1[m]])
        v = (mid_p - mid_m) / 4.0
        ed = b - a
        L_ = ed.norm(dim=-1)
        nrm = torch.stack([-ed[:, 1], ed[:, 0]], -1) / L_.clamp(min=1e-9)[:, None]
        r4[j] = float((L_ * (nrm * v).sum(-1).abs()).sum())
    t4 = traj.t0 + ks * 2.0
    return np.interp(np.arange(t_first, t_first + n_ms, dtype=np.float64), t4, r4)


def tbc_blocks(rec: "I.Rec", block: int = TBC_BLOCK_MS) -> List[int]:
    out = []
    for ra, rb in rec.runs:
        b0 = int(ra) + TBC_MAX_LAG + 4
        while b0 + block + TBC_MAX_LAG + 4 <= int(rb):
            out.append(b0)
            b0 += block
    return out


def tbc_t1_recording(seq: str, stride: int = 1) -> dict:
    """T1 of one recording: per 2 s block, the lag delta_hat between the event rate inside the GT hand crop and the FK-predicted contour
    sweep rate, together with the mean e(n) of the block, and the lag under the source-interval-shuffled events."""
    rc = M3.RecCtx.get(seq)
    rec = rc.rec
    stamps, _ = _csv_stamps(seq)
    t_l, e, fps = tbc_e_of(stamps)
    t_e = np.arange(len(e)) * 1000.0 / fps
    interval_us = 1e6 / fps
    blocks = tbc_blocks(rec)[::stride]
    rows = []
    for b0 in blocks:
        n_ms = TBC_BLOCK_MS
        pad = TBC_MAX_LAG
        X = rec.pose(np.arange(b0 - pad - 2, b0 + n_ms + pad + 2, 2))
        r_pred = predicted_rate(rec.hand, X.astype(np.float32), b0 - pad, n_ms + 2 * pad)
        ev = rec.events_in(b0, b0 + n_ms - 1)
        uv = rec.hand.uv(rec.hand.fk(torch.as_tensor(X[::25].astype(np.float32)))[0])
        vis = rec.hand.visible(rec.hand.fk(torch.as_tensor(X[::25].astype(np.float32)))[0])
        pts = uv[vis].numpy()
        if len(pts) == 0 or len(ev) == 0:
            continue
        lo, hi = pts.min(0) - 10, pts.max(0) + 10
        xy = ev.xy()
        m = (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])
        t_ms = (ev.t_us[m] // 1000 - b0).astype(np.int64)
        rate = np.bincount(np.clip(t_ms, 0, n_ms - 1), minlength=n_ms).astype(np.float64)
        sh = I.shuffle_times(ev.t_us[m], "interval", np.random.default_rng(I.seed_of(seq, b0, "tbc")), interval_us)
        rate_n = np.bincount(np.clip((sh // 1000 - b0).astype(np.int64), 0, n_ms - 1), minlength=n_ms).astype(np.float64)
        zp = smooth_box(r_pred)
        d, rho = xcorr_lag(smooth_box(rate), zp)
        dn, rhon = xcorr_lag(smooth_box(rate_n), zp)
        sel = (t_e >= b0) & (t_e < b0 + n_ms)
        if sel.sum() < 10:
            continue
        rows.append({"b0": b0, "delta": d, "rho": rho, "delta_null": dn, "rho_null": rhon, "e_mean": float(e[sel].mean()),
                     "pred_mean": float(r_pred.mean()), "n_events_crop": int(m.sum())})
    return {"seq": seq, "blocks": rows}


def tbc_t1_job(seq: str) -> dict:
    torch.set_num_threads(1)
    try:
        return tbc_t1_recording(seq)
    except Exception as ex:
        return {"seq": seq, "error": f"{type(ex).__name__}: {ex}"}


def tbc_t1_aggregate(rows: List[dict]) -> dict:
    """Pearson correlation of delta_hat with the block e over all moving blocks, against (i) the permutation of e among the blocks of the
    same recording (95th percentile of the null correlation) and (ii) the correlation of the interval-shuffled-event delta_hat."""
    ok = [r for r in rows if "error" not in r and r["blocks"]]
    d, dn, e, rec_id = [], [], [], []
    for i, r in enumerate(ok):
        b = r["blocks"]
        thr = np.percentile([x["pred_mean"] for x in b], TBC_STATIC_Q)
        for x in b:
            if x["pred_mean"] >= thr:
                d.append(x["delta"])
                dn.append(x["delta_null"])
                e.append(x["e_mean"])
                rec_id.append(i)
    d, dn, e, rec_id = map(np.asarray, (d, dn, e, rec_id))
    out = {"n_blocks": int(len(d)), "n_recordings": len(ok)}
    if len(d) < TBC_MIN_BLOCKS or np.std(e) < 1e-9:
        out["underpowered"] = True
        return out
    corr = lambda a, b: float(np.corrcoef(a, b)[0, 1])                  # noqa: E731
    rng = np.random.default_rng(SEED)
    perm = []
    for _ in range(TBC_N_PERM):
        ep = e.copy()
        for i in np.unique(rec_id):
            m = rec_id == i
            ep[m] = rng.permutation(e[m])
        perm.append(corr(d, ep))
    gb = BOOT(rec_id)
    # bootstrap of the correlation over recordings
    def corr_from_moments(s):
        n, sx, sy, sxx, syy, sxy = s
        den = math.sqrt(max(sxx - sx * sx / n, 1e-18) * max(syy - sy * sy / n, 1e-18))
        return (sxy - sx * sy / n) / den
    S = np.stack([np.ones_like(d), d, e, d * d, e * e, d * e], 1)
    s_, _ = gb.gsum(S)
    bt = np.asarray([corr_from_moments(w @ s_) for w in gb.W])
    out.update({"corr_delta_e": corr(d, e), "corr_ci": ci(bt), "corr_null_interval_shuffle": corr(dn, e),
                "perm_null_p95": float(np.percentile(perm, 95)), "perm_null_mean": float(np.mean(perm)),
                "delta_std_ms": float(d.std()), "e_std_ms": float(e.std()), "underpowered": False})
    return out


def xcorr_selfcheck() -> bool:
    """Synthetic: a smooth random rate series, a copy delayed by a known fractional lag with Poisson noise; the lag is recovered within
    1 ms and a pure-noise series is not recovered systematically."""
    rng = np.random.default_rng(SEED)
    ok = []
    for lag in (-7.3, 0.0, 5.6, 22.4):
        n = 2000
        base = np.convolve(rng.normal(0, 1, n + 2 * TBC_MAX_LAG + 200), np.ones(15) / 15, mode="same")
        base = np.exp(0.8 * base / base.std())
        t = np.arange(len(base))
        shifted = np.interp(t + lag, t, base)                       # events see the content `lag` ms later: delta = +lag
        seg = shifted[TBC_MAX_LAG + 100: TBC_MAX_LAG + 100 + n]
        ev = rng.poisson(40.0 * seg / seg.mean()).astype(np.float64)
        d, rho = xcorr_lag(smooth_box(ev), smooth_box(base[100: 100 + n + 2 * TBC_MAX_LAG]))
        ok.append(abs(d - lag) < 1.0 and rho > 0.5)
    return bool(all(ok))


def decide_tbc(result: dict) -> dict:
    """Gate of prereg_m8.md section 2. Tool validity: metadata checks pass for every recording (CSV rows, relative_time column, t0 equals aux,
    e(0) = e(N-1) = 0) and the synthetic cross-correlation self-check recovers the injected lag.
      T0 pooled p95 registration error (max FK image-plane speed of wrist + fingertips x |e(n)|, mm)
        < 0.08 mm  -> kill ('close'): the time base is not a nuisance, write into the M7 exclusion table
        >= 0.40 mm -> nuisance: T1 decides  (pass: corr(delta_hat, e) >= 0.5 and above the permutation p95 and the
                      interval-shuffle correlation; kill: not above both nulls; else descriptive)
        else       -> descriptive"""
    t0 = result["t0"]
    if not t0.get("checks_all_pass") or not result.get("xcorr_selfcheck", False) or t0["n_recordings"] < 14:
        return {"probe": "TBC", "verdict": "tool_defect", "detail": {"checks_all_pass": t0.get("checks_all_pass"),
                                                                      "xcorr_selfcheck": result.get("xcorr_selfcheck"),
                                                                      "n_recordings": t0.get("n_recordings")}}
    p95 = t0["pooled"]["all"]["err_mm_max_speed"]["p95"]
    det = {"p95_reg_err_mm": p95, "p95_reg_err_px": t0["pooled"]["all"]["err_px_max_speed"]["p95"]}
    if p95 < TBC_T0_CLOSE_MM:
        return {"probe": "TBC", "verdict": "kill", "detail": dict(det, note="closed: not a nuisance; M7 exclusion table")}
    if p95 < TBC_T0_NUISANCE_MM:
        return {"probe": "TBC", "verdict": "descriptive", "detail": dict(det, note="between the close and nuisance lines; T1 not run")}
    t1 = result.get("t1")
    if not t1 or t1.get("underpowered", True):
        return {"probe": "TBC", "verdict": "descriptive", "detail": dict(det, note="nuisance by T0; T1 missing or underpowered")}
    det.update({"corr_delta_e": t1["corr_delta_e"], "corr_ci": t1["corr_ci"], "perm_p95": t1["perm_null_p95"],
                "corr_interval_null": t1["corr_null_interval_shuffle"]})
    above = t1["corr_delta_e"] > max(t1["perm_null_p95"], t1["corr_null_interval_shuffle"])
    if t1["corr_delta_e"] >= TBC_RHO_MIN and above:
        return {"probe": "TBC", "verdict": "pass", "detail": det}
    if not above:
        return {"probe": "TBC", "verdict": "kill", "detail": dict(det, note="delta_hat indistinguishable from the nulls")}
    return {"probe": "TBC", "verdict": "descriptive", "detail": det}


# ================================================================================================ PF
PF_T = (100, 300, 600)
PF_DELTAS = (5, 10)
PF_EVERY = 8
PF_MIN_EVENTS = 20
PF_SCALES = (0.5, 1.0, 1.5, 2.0, 3.0)                   # warp-magnitude scales of the collapse check
PF_COLLAPSE_ARGMAX = 2.0                                 # argmax scale at or above this counts as collapse
PF_STATIC_PX = 1.0                                       # static control: mean GT displacement of the finger's events over 300 ms below this (px)
PF_FLASH_FRAC = 0.2
PF_CTRL_TOL = 0.05                                       # |control rate - 0.5| tolerance on top of the 1.96 SE
PF_JK_MULT = 2.0                                         # margin = 2 x jackknife SE over the FIT7 subjects (leave-one-subject-out)
PF_PRIMARY_T = 300
PF_BOX_MARGIN = 12.0


def pf_ramp_traj(sm: "M3.FbwSample", spec: "PR.Spec", T: int) -> "I.Traj":
    """Time-varying anatomical error: the offset ramps linearly from 0 at the window end to its full size at the window start
    (T ms earlier) and stays at full size before; the E0 spec amplitude (true FK joint error at the end knot) is the full size."""
    rc = sm.rc
    t = sm.ts.astype(np.float64)
    a = np.clip((sm.end - t) / float(T - 1), 0.0, 1.0)
    B = len(t)
    deg = torch.as_tensor(a * spec.sign * spec.deg, dtype=torch.float32)
    om = None
    if spec.fam == "synrc":
        om = torch.as_tensor(a[:, None] * np.asarray(spec.omega)[None], dtype=torch.float32)
    Xp = PR.apply(rc.rec.hand.ctx, rc.fr, spec, torch.as_tensor(sm.X_gt), rc.rec.hand.betas, deg=deg, omega=om).numpy()
    return I.build_traj(rc.rec.hand, Xp, float(sm.t_lo))


def _multiref_var(xy: np.ndarray, d_by_ref: List[np.ndarray], box: "I.Box", sel: np.ndarray) -> float:
    vals = [I.contrast_stats((xy[sel] + d[sel]).astype(np.float32), box)["var"] for d in d_by_ref]
    return float(np.mean(vals))


class PfEval:
    """Evaluates the multi-reference contrast of one finger subtree (or the long-finger union) at one window under a hypothesised
    trajectory, for the real events and the control variants; the boxes cover the warps to all three references."""

    def __init__(self, sm: "M3.FbwSample"):
        self.sm = sm
        self.box = {}
        self.refs = {}
        for T in PF_T:
            self.refs[T] = (sm.end + 1.0 - T, sm.end + 1.0 - T / 2.0, sm.end + 1.0)
        for f in list(FINGERS5) + ["long4"]:
            sel = sm.member[f]
            if sel.sum() < 3:
                continue
            pts = [sm.xy[sel]]
            for T in PF_T:
                for tr in self.refs[T]:
                    pts.append(sm.xy[sel] + I.displacement(sm.traj, sm.asg, sm.t_ms, tr)[sel])
            self.box[f] = I.box_from_points(np.concatenate(pts), PF_BOX_MARGIN)

    def contrast(self, traj_h, f: str, T: int, variants: Sequence[str], ev_seed: int, thin_to: Optional[int] = None) -> Dict[str, float]:
        """`ev_seed` seeds the control event variants (null permutation, thinning, noise, flash); it must be the same for the GT and for
        every hypothesis of a (sample, T) so that all hypotheses are scored on the identical event variant."""
        sm = self.sm
        subs = LONG4 if f == "long4" else (f,)
        sel0 = np.any([sm.member[s] for s in subs], axis=0) & sm.tmask[T]
        idx0 = np.nonzero(sel0)[0]
        out = {}
        if len(idx0) < PF_MIN_EVENTS or f not in self.box:
            return {v: float("nan") for v in variants}
        for v in variants:
            xy, t_use, idx = sm.xy, sm.t_ms, idx0
            if v == "null":
                t_use = I.shuffle_times(sm.ev.t_us, "cluster", np.random.default_rng(ev_seed)) / 1000.0
            elif v == "thin":
                n_ref = int((np.any([sm.member[s] for s in subs], axis=0) & sm.tmask[PF_T[0]]).sum())
                if thin_to is None or len(idx0) <= n_ref or n_ref < PF_MIN_EVENTS:
                    out[v] = float("nan")
                    continue
                idx = idx0[I.thin_index(len(idx0), n_ref, np.random.default_rng(ev_seed + 7))]
            elif v in ("noise", "flash"):
                rng = np.random.default_rng(ev_seed + (11 if v == "noise" else 13))
                b = self.box[f]
                xy = sm.xy.copy()
                if v == "noise":
                    xy[idx0] = np.stack([rng.uniform(b.x0, b.x0 + b.w, len(idx0)), rng.uniform(b.y0, b.y0 + b.h, len(idx0))], 1).astype(np.float32)
                else:
                    m = int(round(PF_FLASH_FRAC * len(idx0)))
                    t_fl = float(sm.t_ms[idx0[rng.integers(len(idx0))]])
                    fx = np.stack([rng.uniform(b.x0, b.x0 + b.w, m), rng.uniform(b.y0, b.y0 + b.h, m)], 1).astype(np.float32)
                    xy = np.concatenate([sm.xy, fx])
                    t_use = np.concatenate([sm.t_ms, np.full(m, t_fl)])
                    idx = np.concatenate([idx0, np.arange(len(sm.xy), len(sm.xy) + m)])
            asg = I.assign(traj_h, xy[idx], t_use[idx])
            dd = [I.displacement(traj_h, asg, t_use[idx], tr) for tr in self.refs[T]]
            vals = [I.contrast_stats((xy[idx] + d).astype(np.float32), self.box[f])["var"] for d in dd]
            out[v] = float(np.mean(vals))
        return out

    def scale_curve(self, f: str, T: int, scales: Sequence[float], null: bool, seed: int) -> Dict[str, List[float]]:
        """Contrast of the GT warp with the displacement scaled by s (collapse check): raw variance and the area-normalised variance."""
        sm = self.sm
        sel = sm.member[f] & sm.tmask[T]
        idx = np.nonzero(sel)[0]
        if len(idx) < PF_MIN_EVENTS or f not in self.box:
            return {"raw": [float("nan")] * len(scales), "norm": [float("nan")] * len(scales)}
        t_use = sm.t_ms if not null else I.shuffle_times(sm.ev.t_us, "cluster", np.random.default_rng(seed)) / 1000.0
        asg = I.assign(sm.traj, sm.xy[idx], t_use[idx])
        ds = [I.displacement(sm.traj, asg, t_use[idx], tr) for tr in self.refs[T]]
        raw, nrm = [], []
        for s in scales:
            vr, vn = [], []
            for d in ds:
                xw = (sm.xy[idx] + s * d).astype(np.float32)
                v = I.contrast_stats(xw, self.box[f])["var"]
                vr.append(v)
                vn.append(v * I.cloud_area_ratio(sm.xy[idx], xw) ** 2)
            raw.append(float(np.mean(vr)))
            nrm.append(float(np.mean(vn)))
        return {"raw": raw, "norm": nrm}


def pf_one(seq: str, end: int, deltas=PF_DELTAS, quick: bool = False) -> dict:
    rc = M3.RecCtx.get(seq)
    seed = I.seed_of(seq, end, "pf")
    sm = M3.FbwSample(rc, end, seed, Ts=PF_T)
    ev = PfEval(sm)
    x_end = torch.as_tensor(sm.X_gt[-1])
    specs = M3.make_specs(rc, x_end, deltas=deltas)
    out = {"seq": seq, "end": int(end), "n_cap": len(sm.ev)}
    # GT contrasts (multi-reference, real / null / thin)
    gt = {}
    for f in FINGERS5 + ("long4",):
        for T in PF_T:
            gt[(f, T)] = ev.contrast(sm.traj, f, T, ("real", "null", "thin", "noise", "flash"), seed + T, thin_to=PF_T[0])   # ev_seed = seed + T
    out["gt"] = gt
    # static flag per finger: mean GT displacement of its members over 300 ms to the end reference
    disp = np.linalg.norm(I.displacement(sm.traj, sm.asg, sm.t_ms, sm.end + 1.0), axis=1)
    out["motion_px"] = {f: float(disp[sm.member[f] & sm.tmask[300]].mean()) if (sm.member[f] & sm.tmask[300]).sum() > 0 else float("nan")
                        for f in FINGERS5}
    hyp = {}
    for key, sp in specs.items():
        fam, f, sign, d = key
        for T in PF_T:
            tr = pf_ramp_traj(sm, sp, T)
            vars_ = ("real", "null", "thin") + (("noise", "flash") if (T == PF_PRIMARY_T and d == 10) else ())
            hyp[(key, T)] = ev.contrast(tr, f, T, vars_, seed + T, thin_to=PF_T[0])
    out["hyp"] = hyp
    # collapse check on the GT trajectory at the primary window
    out["collapse"] = {f: {"real": ev.scale_curve(f, PF_PRIMARY_T, PF_SCALES, False, seed), "null": ev.scale_curve(f, PF_PRIMARY_T, PF_SCALES, True, seed + 5)}
                       for f in FINGERS5}
    return out


def pf_job(job):
    seq, ends, kw = job
    torch.set_num_threads(1)
    res = []
    for e in ends:
        try:
            res.append(pf_one(seq, int(e), **kw))
        except Exception as ex:
            res.append({"seq": seq, "end": int(e), "error": f"{type(ex).__name__}: {ex}"})
    return res


def _pf_pref(rec: dict, T: int, delta: int, variant: str, fingers=FINGERS5, static_only: Optional[bool] = None):
    """Per-sample preference of the GT contrast over the +-delta hypotheses of the given fingers (ties 0.5) at window T for an event
    variant; nan if no hypothesis is defined. `static_only` True keeps only static fingers, False only moving ones."""
    vals = []
    for (key, Tk), c in rec["hyp"].items():
        fam, f, sign, d = key
        if Tk != T or d != delta or f not in fingers:
            continue
        if static_only is not None:
            st = bool(rec["motion_px"].get(f, np.nan) < PF_STATIC_PX)
            if st != static_only:
                continue
        a = rec["gt"].get((f, T), {}).get(variant, np.nan)
        b = c.get(variant, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            vals.append(1.0 if a > b else (0.5 if a == b else 0.0))
    return float(np.mean(vals)) if vals else float("nan")


def _pf_grad(rec: dict, T: int, variant: str):
    """Fraction of (finger, family, sign) triples whose contrast falls from delta 5 to delta 10 (correct local gradient direction)."""
    vals = []
    h = rec["hyp"]
    for (key, Tk), c10 in h.items():
        fam, f, sign, d = key
        if Tk != T or d != 10:
            continue
        c5 = h.get(((fam, f, sign, 5), T))
        if c5 is None:
            continue
        a, b = c5.get(variant, np.nan), c10.get(variant, np.nan)
        if np.isfinite(a) and np.isfinite(b):
            vals.append(1.0 if b < a else (0.5 if b == a else 0.0))
    return float(np.mean(vals)) if vals else float("nan")


def _collapse(rec: dict, world: str, kind: str):
    """Fraction of fingers whose scale curve peaks at s >= PF_COLLAPSE_ARGMAX."""
    v = []
    for f in FINGERS5:
        c = np.asarray(rec["collapse"][f][world][kind], np.float64)
        if np.all(np.isfinite(c)):
            v.append(float(PF_SCALES[int(np.argmax(c))] >= PF_COLLAPSE_ARGMAX))
    return float(np.mean(v)) if v else float("nan")


def _nm(v) -> float:
    v = np.asarray(v, np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else float("nan")


def pf_aggregate(recs: List[dict]) -> dict:
    ok = [r for r in recs if "error" not in r]
    out = {"n_ok": len(ok), "n_failed": len(recs) - len(ok), "domains": {}}
    for dom in ("local", "global"):
        d = {}
        for role, subj in (("fit", FIT7), ("sel", SELECT)):
            rs = [r for r in ok if domain_of(r["seq"]) == dom and r["seq"].split("_")[0] in subj]
            if not rs:
                continue
            gb = BOOT(M3.group_keys([r["seq"] for r in rs], [r["end"] for r in rs]))
            o = {"n": len(rs), "cells": {}}
            for T in PF_T:
                for delta in PF_DELTAS:
                    cell = {}
                    for var in ("real", "null", "thin", "noise", "flash"):
                        if var in ("noise", "flash") and (T != PF_PRIMARY_T or delta != 10):
                            continue
                        v = np.asarray([_pf_pref(r, T, delta, var) for r in rs])
                        pt, bt = gb.wmean(v)
                        cell[var] = {"pref": float(pt[0]), "ci": ci(bt[:, 0])}
                    pr, bt_r = gb.wmean(np.asarray([_pf_pref(r, T, delta, "real") for r in rs]))
                    pn, bt_n = gb.wmean(np.asarray([_pf_pref(r, T, delta, "null") for r in rs]))
                    diff = np.asarray([_pf_pref(r, T, delta, "real") - _pf_pref(r, T, delta, "null") for r in rs])
                    pd, bd = gb.wmean(diff)
                    cell["real_minus_null"] = {"mean": float(pd[0]), "ci": ci(bd[:, 0])}
                    if T != PF_T[0]:
                        dt = np.asarray([_pf_pref(r, T, delta, "real") - _pf_pref(r, T, delta, "thin") for r in rs])
                        ptn, btn = gb.wmean(dt)
                        cell["real_minus_equal_event"] = {"mean": float(ptn[0]), "ci": ci(btn[:, 0])}
                    o["cells"][f"T{T}|d{delta}"] = cell
                gv = np.asarray([_pf_grad(r, T, "real") for r in rs])
                gn = np.asarray([_pf_grad(r, T, "null") for r in rs])
                pt, bt = gb.wmean(gv)
                pn_, bn_ = gb.wmean(gn)
                pdg, bdg = gb.wmean(gv - gn)
                o["cells"][f"T{T}|grad_5to10"] = {"real": float(pt[0]), "real_ci": ci(bt[:, 0]), "null": float(pn_[0]),
                                                  "real_minus_null": {"mean": float(pdg[0]), "ci": ci(bdg[:, 0])}}
            # static control (T = primary, delta 10): fingers with GT motion below PF_STATIC_PX
            sv = np.asarray([_pf_pref(r, PF_PRIMARY_T, 10, "real", static_only=True) for r in rs])
            mv = np.asarray([_pf_pref(r, PF_PRIMARY_T, 10, "real", static_only=False) for r in rs])
            ps, bs = gb.wmean(sv)
            pm, bm = gb.wmean(mv)
            o["static_control"] = {"pref_static": float(ps[0]), "ci": ci(bs[:, 0]), "n_static_samples": int(np.isfinite(sv).sum()),
                                   "pref_moving": float(pm[0])}
            # collapse (T = primary): raw and area-normalised argmax >= PF_COLLAPSE_ARGMAX, real versus null
            for kind in ("raw", "norm"):
                cr = np.asarray([_collapse(r, "real", kind) for r in rs])
                cn = np.asarray([_collapse(r, "null", kind) for r in rs])
                pr_, br_ = gb.wmean(cr)
                pn_, bn_ = gb.wmean(cn)
                pd_, bd_ = gb.wmean(cr - cn)
                o.setdefault("collapse", {})[kind] = {"real": float(pr_[0]), "real_ci": ci(br_[:, 0]), "null": float(pn_[0]),
                                                      "real_minus_null": {"mean": float(pd_[0]), "ci": ci(bd_[:, 0])}}
            d[role] = o
        # jackknife SE over FIT7 subjects of the primary real-minus-null difference (the pre-declared margin)
        fit = [r for r in ok if domain_of(r["seq"]) == dom and r["seq"].split("_")[0] in FIT7]
        if fit:
            subs = sorted({r["seq"].split("_")[0] for r in fit})
            dif = lambda r: _pf_pref(r, PF_PRIMARY_T, 10, "real") - _pf_pref(r, PF_PRIMARY_T, 10, "null")         # noqa: E731
            per = {s: _nm([dif(r) for r in fit if r["seq"].split("_")[0] == s]) for s in subs}
            loso = np.asarray([_nm([dif(r) for r in fit if r["seq"].split("_")[0] != s]) for s in subs])
            n = len(subs)
            se = float(math.sqrt((n - 1) / n * np.sum((loso - loso.mean()) ** 2))) if n > 1 else float("nan")
            d["margin"] = {"fit_diff_mean": _nm(list(per.values())), "loso_values": loso.tolist(),
                           "jackknife_se": se, "margin": PF_JK_MULT * se, "n_subjects": n}
        out["domains"][dom] = d
    return out


def decide_pf(result: dict) -> dict:
    """Gate of prereg_m8.md section 3, per domain on SELECT (the FIT7 jackknife margin is a noise yardstick, not a fit):
    tool validity (all must hold): the static and noise controls have preference within 0.05 + 1.96 SE of 0.5 and the flash control does
    not exceed the real preference by more than 0.05; the GT warp is not worse than the null at T = 100 (real-minus-null difference at
    delta 10 has CI lower bound > -0.05).
      pass     primary cell (T = 300, delta = 10 mm): real minus cluster-null preference >= margin = 2 x jackknife SE of the FIT7
               leave-one-subject-out differences AND its CI lower bound > 0 AND the same CI lower bound > 0 at T = 600 AND no collapse
               (area-normalised collapse rate real minus null has CI lower bound <= 0)
      kill     difference <= margin or CI lower bound <= 0 at the primary cell, or collapse flagged
      else     descriptive
    The equal-event-count attribution (real minus thinned) is reported with a flag `event_count_only` when its CI lower bound is <= 0 at T = 300.
    Overall: both domains pass -> pass; both kill -> kill; tool defect anywhere -> tool_defect; else descriptive."""
    agg = result["aggregate"]
    per = {}
    fail_frac = agg.get("n_failed", 0) / max(1, agg.get("n_ok", 1) + agg.get("n_failed", 0))
    for dom, d in agg["domains"].items():
        s = d.get("sel")
        mg = d.get("margin", {}).get("margin", float("nan"))
        if not s or not np.isfinite(mg):
            per[dom] = {"verdict": "descriptive", "note": "no SELECT samples or margin"}
            continue
        c300, c600, c100 = s["cells"][f"T{PF_PRIMARY_T}|d10"], s["cells"][f"T600|d10"], s["cells"][f"T{PF_T[0]}|d10"]
        n_static = s["static_control"]["n_static_samples"]
        noise = c300.get("noise", {"pref": float("nan"), "ci": [np.nan, np.nan]})
        flash = c300.get("flash", {"pref": float("nan")})
        se_n = (noise["ci"][1] - noise["ci"][0]) / (2 * 1.96) if np.isfinite(noise["pref"]) else float("nan")
        v = {"noise_ok": bool(np.isfinite(noise["pref"]) and abs(noise["pref"] - 0.5) <= PF_CTRL_TOL + 1.96 * se_n),
             "flash_ok": bool(np.isfinite(flash["pref"]) and flash["pref"] <= c300["real"]["pref"] + PF_CTRL_TOL),
             "static_ok": bool(n_static < 30 or abs(s["static_control"]["pref_static"] - 0.5) <= PF_CTRL_TOL +
                               (s["static_control"]["ci"][1] - s["static_control"]["ci"][0]) / 2.0),
             "not_worse_than_null_T100": bool(c100["real_minus_null"]["ci"][0] > -0.05)}
        r = {"validity": v, "margin": mg, "diff_T300": c300["real_minus_null"], "diff_T600": c600["real_minus_null"],
             "collapse_norm": s["collapse"]["norm"]["real_minus_null"], "collapse_raw": s["collapse"]["raw"]["real_minus_null"],
             "event_count_only": bool(c300["real_minus_equal_event"]["ci"][0] <= 0)}
        collapsed = s["collapse"]["norm"]["real_minus_null"]["ci"][0] > 0
        r["failed_fraction"] = fail_frac
        if not all(v.values()) or fail_frac > M3.MAX_FAIL_FRAC:
            r["verdict"] = "tool_defect"
        elif collapsed:
            r["verdict"] = "kill"
            r["note"] = "collapse flagged"
        elif c300["real_minus_null"]["mean"] >= mg and c300["real_minus_null"]["ci"][0] > 0 and c600["real_minus_null"]["ci"][0] > 0:
            r["verdict"] = "pass"
        elif c300["real_minus_null"]["mean"] <= mg or c300["real_minus_null"]["ci"][0] <= 0:
            r["verdict"] = "kill"
        else:
            r["verdict"] = "descriptive"
        per[dom] = r
    vv = [x["verdict"] for x in per.values()]
    overall = ("tool_defect" if "tool_defect" in vv else "pass" if vv and all(x == "pass" for x in vv) else
               "kill" if vv and all(x == "kill" for x in vv) else "descriptive")
    return {"probe": "PF", "verdict": overall, "per_domain": per}


def decide(result: dict) -> dict:
    """Dispatch on result['probe'] in {'SRCA', 'TBC', 'PF'}; a dict {'SRCA': ..., 'TBC': ..., 'PF': ...} decides each."""
    if "probe" in result:
        return {"SRCA": decide_srca, "TBC": decide_tbc, "PF": decide_pf}[result["probe"]](result)
    return {k: decide(v) for k, v in result.items() if k in ("SRCA", "TBC", "PF")}


# ================================================================================================ unit checks / smoke
class _StubRec:
    """Minimal recording stand-in for the speed / residual unit tests: a pose function of ms, hand, n_ms."""

    def __init__(self, hand, fn, n_ms):
        self.hand, self.n_ms = hand, n_ms
        self.pos51 = _Lazy(fn)


class _Lazy:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, k):
        return self.fn(int(k))


def _chk_srca(hand: "I.Hand") -> Dict[str, bool]:
    res = {}
    rng = np.random.default_rng(SEED)
    rend = Renderer(hand)
    base = np.zeros(51)
    base[2] = 0.6
    base[6:] = 0.0
    # --- renderer: translation moves the image by f dx / z; foreground exists; intensity range
    v0, _ = hand.fk(torch.as_tensor(base[None], dtype=torch.float32))
    img0 = rend.render(v0[0])
    b2 = base.copy()
    b2[0] += 0.008
    v1, _ = hand.fk(torch.as_tensor(b2[None], dtype=torch.float32))
    img1 = rend.render(v1[0])

    def centroid(img):
        w = (img - rend.bg).clamp(min=0)
        yy, xx = torch.meshgrid(torch.arange(I.H, dtype=torch.float32), torch.arange(I.W, dtype=torch.float32), indexing="ij")
        return float((w * xx).sum() / w.sum()), float((w * yy).sum() / w.sum())
    c0, c1 = centroid(img0), centroid(img1)
    expect = hand.fx * 0.008 / float(v0[0, :, 2].mean())
    res["render_foreground"] = bool((img0 > rend.bg + 1e-3).float().mean() > 0.01 and float(img0.max()) <= 1.0)
    res["render_translation_shift"] = bool(abs((c1[0] - c0[0]) - expect) < 0.35 and abs(c1[1] - c0[1]) < 0.35)
    # --- event simulator: static, step, polarity, layer and per-pixel timestamps
    static = torch.zeros(3, 6, 8)
    x, y, p, t = simulate_events(static, np.array([0.0, 1.0, 2.0]), SRCA_THETA, True)
    res["sim_static_no_events"] = bool(len(x) == 0)
    step = torch.zeros(2, 6, 8)
    step[1] += 3.07 * SRCA_THETA
    for layers in (True, False):
        x, y, p, t = simulate_events(step, np.array([0.0, 10.0]), SRCA_THETA, layers)
        cnt = np.bincount(y * 8 + x, minlength=48)
        ok = bool(np.all(cnt == 3) and p.all() and np.all((t >= 0) & (t <= 10000)))
        if layers:
            ok = ok and bool(np.array_equal(np.unique(t), np.rint(np.array([1, 2, 3]) * 10000.0 / 3).astype(np.int64)))
        else:
            ok = ok and bool(np.array_equal(np.unique(t), np.rint(10000.0 * np.array([1, 2, 3]) * SRCA_THETA / (3.07 * SRCA_THETA)).astype(np.int64)))
        res[f"sim_step_{'layers' if layers else 'per_pixel'}"] = ok
    neg = -step
    x, y, p, t = simulate_events(neg, np.array([0.0, 10.0]), SRCA_THETA, True)
    res["sim_negative_polarity"] = bool((p == 0).all() and len(x) == 3 * 48)
    # --- synthetic trajectory: PL passes through its samples, HF component is above 30 Hz, calibration hits the target
    st = SynthTraj(base, hand, seed=5)
    ts = np.arange(0.0, st.duration + 1.0, 1000.0 / SRCA_FPS)
    st.scale = 0.7
    res["pl_exact_at_samples"] = bool(np.abs(st.x_pl(ts) - st.x_at(ts)).max() < 1e-9)
    res["hf_above_30hz"] = bool(st.hf_energy_above_30hz() > 0.95)
    st.calibrate(1.149)
    res["calibration_hits_target"] = bool(abs(st.residual_rms_mm() - 1.149) / 1.149 < 0.05)
    st2 = SynthTraj(base, hand, seed=5)
    st2.scale = 0.0
    r_low, st2.scale = st2.residual_rms_mm(), 1.0
    res["hf_raises_residual"] = bool(st2.residual_rms_mm() > 3 * max(r_low, 1e-3))
    # --- label residual core: constant velocity -> 0, parabola -> analytic RMS
    h_ms = 1000.0 / SRCA_FPS
    n_fr = 60
    t_src = np.arange(n_fr) * h_ms
    a_par = 0.02 / (h_ms / 1000.0) ** 2 * 0.2
    lab_lin = lambda k: np.r_[0.05 * k / 1000.0, 0, 0.6, np.zeros(48)]                       # noqa: E731
    lab_par = lambda k: np.r_[a_par * (k / 1000.0) ** 2, 0, 0.6, np.zeros(48)]               # noqa: E731
    for name, fn, expected in (("lin", lab_lin, 0.0), ("par", lab_par, 0.1826 * a_par * (h_ms / 1000.0) ** 2 * 1000.0)):
        tr = np.stack([fn(t)[:3] for t in t_src])
        out = label_residual_core(hand, fn, t_src, tr, np.zeros((n_fr, 3)), np.zeros((n_fr, 45)), [(0, n_fr)], min_run=10, step_ms=1)
        res[f"label_residual_{name}"] = bool(abs(out["rms_vertex_mm"] - expected) < (1e-3 if expected == 0 else 0.05 * expected))
    # --- TBC metadata arithmetic and speeds
    stamps = (1_700_000_000_000_000_000 + np.arange(101) * 16_700_000 + (3_000_000 * np.sin(np.arange(101) / 5.0))).astype(np.int64)
    t_l, e, fps = tbc_e_of(stamps)
    res["tbc_e_zero_at_ends"] = bool(abs(e[0]) < 1e-9 and abs(e[-1]) < 1e-6 and abs(fps * (stamps[-1] - stamps[0]) / 1e9 - 100) < 1e-9)
    stub = _StubRec(hand, lambda k: np.r_[0.0005 * k, 0.0, 0.6, np.zeros(48)], 4000)           # 0.5 mm per ms along x
    sp = fk_speeds(stub, np.array([100.0, 500.0, 1000.0]))
    res["tbc_fk_speed"] = bool(np.allclose(sp["mm_max"], 0.5, atol=0.01))
    res["tbc_xcorr_selfcheck"] = bool(xcorr_selfcheck())
    # --- predicted rate: static trajectory ~ 0, moving trajectory > 0
    Xs = np.repeat(base[None].astype(np.float32), 400, 0)
    Xm = Xs.copy()
    Xm[:, 0] += 0.0008 * np.arange(400)
    r_static = predicted_rate(hand, Xs, 10, 600)
    r_move = predicted_rate(hand, Xm, 10, 600)
    res["predicted_rate_static_zero"] = bool(np.abs(r_static).max() < 1e-6 and r_move.mean() > 1e-3)
    return res


def _chk_decide_m8() -> Dict[str, bool]:
    res = {}

    def srca_res(esim_lo, v2e_lo, lab):
        return {"aggregate": {"pipelines": {"esim": {"slope_real_minus_null": {"value": 1.0, "ci": [esim_lo, 2.0]}},
                                            "v2e": {"slope_real_minus_null": {"value": 0.0, "ci": [v2e_lo, v2e_lo + 0.5]}}}},
                "label_residual": {"all": {"rms_vertex_mm": lab}}}
    res["srca_kill_blind"] = decide_srca(srca_res(0.5, -0.4, 0.3))["verdict"] == "kill"
    res["srca_blind_confirmed_flag"] = bool(decide_srca(srca_res(0.5, -0.4, 0.3))["detail"]["blind_confirmed"])
    res["srca_pass"] = decide_srca(srca_res(0.5, 0.1, 1.5))["verdict"] == "pass"
    res["srca_descriptive"] = decide_srca(srca_res(0.5, 0.1, 0.9))["verdict"] == "descriptive"
    res["srca_tool_defect_control"] = decide_srca(srca_res(-0.1, 0.1, 1.5))["verdict"] == "tool_defect"

    def tbc_res(p95, corr=0.7, perm=0.2, nul=0.1, under=False, checks=True):
        t0 = {"checks_all_pass": checks, "n_recordings": 18, "pooled": {"all": {"err_mm_max_speed": {"p95": p95}, "err_px_max_speed": {"p95": 1.0}}}}
        t1 = {"underpowered": under, "corr_delta_e": corr, "corr_ci": [0.5, 0.8], "perm_null_p95": perm, "corr_null_interval_shuffle": nul}
        return {"t0": t0, "t1": t1, "xcorr_selfcheck": True}
    res["tbc_close"] = decide_tbc(tbc_res(0.05))["verdict"] == "kill"
    res["tbc_between"] = decide_tbc(tbc_res(0.2))["verdict"] == "descriptive"
    res["tbc_pass"] = decide_tbc(tbc_res(0.6))["verdict"] == "pass"
    res["tbc_kill_t1"] = decide_tbc(tbc_res(0.6, corr=0.1))["verdict"] == "kill"
    res["tbc_nuisance_below_rho"] = decide_tbc(tbc_res(0.6, corr=0.4, perm=0.1, nul=0.0))["verdict"] == "descriptive"
    res["tbc_tool_defect"] = decide_tbc(tbc_res(0.6, checks=False))["verdict"] == "tool_defect"

    def pf_res(diff300=0.1, lo300=0.05, lo600=0.03, margin=0.04, collapse_lo=-0.1, noise=0.5, flash=0.6, real=0.7):
        cell = lambda diff, lo: {"real": {"pref": real, "ci": [real - 0.03, real + 0.03]}, "null": {"pref": 0.5},       # noqa: E731
                                 "noise": {"pref": noise, "ci": [noise - 0.02, noise + 0.02]}, "flash": {"pref": flash},
                                 "real_minus_null": {"mean": diff, "ci": [lo, diff + 0.05]},
                                 "real_minus_equal_event": {"mean": 0.05, "ci": [0.01, 0.09]}}
        s = {"cells": {"T300|d10": cell(diff300, lo300), "T600|d10": cell(0.1, lo600), "T100|d10": cell(0.1, 0.05)},
             "static_control": {"pref_static": 0.5, "ci": [0.45, 0.55], "n_static_samples": 50},
             "collapse": {"norm": {"real_minus_null": {"mean": 0.0, "ci": [collapse_lo, 0.1]}}, "raw": {"real_minus_null": {"mean": 0.0, "ci": [-0.1, 0.1]}}}}
        d = {"sel": s, "margin": {"margin": margin}}
        return {"aggregate": {"domains": {"local": d, "global": d}}}
    res["pf_pass"] = decide_pf(pf_res())["verdict"] == "pass"
    res["pf_kill_no_gain"] = decide_pf(pf_res(diff300=0.0, lo300=-0.05))["verdict"] == "kill"
    res["pf_kill_below_margin"] = decide_pf(pf_res(diff300=0.03, lo300=0.01, margin=0.05))["verdict"] == "kill"
    res["pf_kill_collapse"] = decide_pf(pf_res(collapse_lo=0.05))["verdict"] == "kill"
    res["pf_tool_defect_noise"] = decide_pf(pf_res(noise=0.8))["verdict"] == "tool_defect"
    res["pf_tool_defect_flash"] = decide_pf(pf_res(flash=0.95))["verdict"] == "tool_defect"
    res["pf_descriptive_T600"] = decide_pf(pf_res(lo600=-0.02))["verdict"] == "descriptive"
    return res


_SMOKE_LOG: List[str] = []


def say(msg: str):
    print(msg, flush=True)
    _SMOKE_LOG.append(msg)


def _write_smoke_log(name: str, ok: bool):
    d = REPO / "outputs" / "mv2" / "smoke" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "smoke.json").write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "ok": bool(ok),
                                              "note": "shapes / counts / timings / pass-fail of unit checks only; no effect statistic",
                                              "lines": _SMOKE_LOG}, indent=1, ensure_ascii=False))


def smoke(real: bool = True) -> bool:
    t0 = time.time()
    ok_all = True
    say("[smoke] iwe toolkit selftest")
    ok_all &= I.selftest(verbose=False)
    rc = M3.RecCtx.get("ch_local")
    for name, fn in (("SRCA / TBC / label-residual machinery (synthetic)", lambda: _chk_srca(rc.rec.hand)),
                     ("decide gates (logic)", _chk_decide_m8)):
        r = fn()
        bad = [k for k, v in r.items() if not v]
        say(f"[smoke] {name}: {len(r)} checks, {'all pass' if not bad else 'FAIL ' + str(bad)}")
        ok_all &= not bad
    if not real:
        return ok_all
    # tiny real slices, shapes / counts only
    t = time.time()
    job = srca_jobs(n_traj=1, levels=(1.149,))[0]
    o = srca_one(job)
    say(f"[smoke] SRCA one (trajectory, level): keys {sorted(k for k in o if k.startswith(('resp', 'null', 'n_')))} "
          f"events esim {o.get('n_esim')} v2e {o.get('n_v2e')} {time.time() - t:.1f} s")
    ok_all &= bool("resp_esim" in o and "null_v2e" in o)
    agg = srca_aggregate([dict(o, traj=i, rms_mm=o["rms_mm"] * (1 + 0.2 * (i % 5)), resp_esim=0.1 * i, resp_v2e=-0.05 * i, null_esim=0.0, null_v2e=0.0)
                          for i in range(30)])
    say(f"[smoke] SRCA aggregate path runs: pipelines {sorted(agg['pipelines'])}")
    t = time.time()
    lab = label_subframe_rms("ycy_local")
    say(f"[smoke] label residual of one recording: runs {lab['n_runs']} samples {lab['n_samples']} finite {bool(np.isfinite(lab['rms_vertex_mm']))} {time.time() - t:.1f} s")
    ok_all &= bool(lab["n_runs"] > 0 and np.isfinite(lab["rms_vertex_mm"]))
    t = time.time()
    t0r = tbc_t0("ycy_local")
    say(f"[smoke] TBC T0 of one recording: frames {t0r['n_frames']} valid {t0r['n_valid_frames']} metadata checks "
          f"{'pass' if all(t0r['checks'].values()) else 'FAIL ' + str(t0r['checks'])} {time.time() - t:.1f} s")
    ok_all &= bool(all(t0r["checks"].values()))
    agg0 = tbc_t0_aggregate([t0r, dict(t0r, seq="ycy_global")])
    say(f"[smoke] TBC T0 aggregate path runs: pooled keys {sorted(agg0['pooled'])}")
    rec = rc.rec
    t = time.time()
    t1 = tbc_t1_recording("ycy_local", stride=40)
    say(f"[smoke] TBC T1 on {len(t1['blocks'])} blocks of one recording {time.time() - t:.1f} s")
    agg1 = tbc_t1_aggregate([t1] * 3)
    say(f"[smoke] TBC T1 aggregate path runs: keys {sorted(agg1)}")
    ends = M3.list_endpoints(rc.rec, PF_EVERY)
    t = time.time()
    o = pf_one("ch_local", int(ends[4]), deltas=(5, 10))
    say(f"[smoke] PF one endpoint: hyp cells {len(o['hyp'])} gt cells {len(o['gt'])} {time.time() - t:.1f} s")
    recs = []
    for seq in ("ch_local", "ycy_local", "ch_global", "ycy_global"):
        recs += [dict(o, seq=seq, end=o["end"] + 1000 * i) for i in range(4)]
    agg = pf_aggregate(recs)
    say(f"[smoke] PF aggregate + decide path runs: domains {sorted(agg['domains'])} roles {sorted(agg['domains']['local'])}")
    ok_all &= bool("sel" in agg["domains"]["local"] and "margin" in agg["domains"]["local"])
    dec = decide_pf({"aggregate": agg})
    ok_all &= bool("verdict" in dec)
    sm = M3.FbwSample(M3.RecCtx.get("ch_local"), int(ends[4]), 1, Ts=PF_T)
    specs = M3.make_specs(M3.RecCtx.get("ch_local"), torch.as_tensor(sm.X_gt[-1]), fingers=("index",), deltas=(10,), with_syn=False)
    sp = next(iter(specs.values()))
    tr_ramp = pf_ramp_traj(sm, sp, 300)
    tr_end = M3.pert_traj(sm, sp)
    d_end = float((tr_ramp.verts[-1] - sm.traj.verts[-1]).norm(dim=-1).max()) * 1000
    d_start = float((tr_ramp.verts[0] - tr_end.verts[0]).norm(dim=-1).max()) * 1000
    ok_ramp = bool(d_end < 1e-3 and d_start < 1e-3)
    say(f"[smoke] PF ramp trajectory: zero offset at the window end and the full persistent offset before the window: {'pass' if ok_ramp else 'FAIL'}")
    ok_all &= ok_ramp
    try:
        MC.verify_protocol()
        say("[smoke] formal-mode gate: verify_protocol accepts (the freeze is present and valid)")
    except SystemExit:
        say("[smoke] formal-mode gate: verify_protocol refuses (no valid freeze yet), as required")
    say(f"[smoke] m8_probes: {'ALL PASS' if ok_all else 'FAIL'} ({time.time() - t0:.0f} s)")
    _write_smoke_log("m8", ok_all)
    return ok_all


# ================================================================================================ formal run
def _run_pool(fn, jobs: list, procs: int, tag: str):
    import random
    from multiprocessing import get_context
    random.Random(SEED).shuffle(jobs)
    out = []
    t0 = time.time()
    with get_context("fork").Pool(procs) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
            out += res if isinstance(res, list) else [res]
            if (i + 1) % max(1, len(jobs) // 10) == 0:
                print(f"[{tag}] {i + 1}/{len(jobs)} jobs {time.time() - t0:.0f} s", flush=True)
    out.sort(key=lambda r: (str(r.get("seq")), int(r.get("end", 0)), int(r.get("traj", 0)), float(r.get("level_mm", 0.0))))    # completion order must not matter
    return out


def _jdefault(o):
    if isinstance(o, np.bool_):
        return bool(o)
    return MC._np(o)


def run_formal(probe: str, out_dir: Path, procs: int, every: Optional[int] = None):
    import pickle
    MC.verify_protocol()
    out_dir = Path(out_dir)
    if (out_dir / "result.json").exists():
        raise SystemExit(f"{out_dir}/result.json exists; preserve it")
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    seqs_all = MC.base_seqs(FIT7 + SELECT)
    meta = {"probe": probe.upper(), "procs": procs, "protocol_sha": MC.sha(MC.PROTOCOL), "stride": None}
    scripts = [Path(__file__), REPO / "tools/mv2/iwe.py", REPO / "tools/mv2/m3_probes.py", REPO / "tools/mv2/common.py",
               REPO / "tools/sba/r3_probe.py"]
    raw = None
    if probe == "srca":
        jobs = srca_jobs()
        rows = _run_pool(srca_one, jobs, procs, "srca")
        lab = _run_pool(label_job, list(MC.base_seqs(FIT7)), min(procs, 14), "label")
        res = {"probe": "SRCA", "aggregate": srca_aggregate(rows), "label_residual": label_aggregate(lab), "meta": meta,
               "constants": {"levels_mm": SRCA_LEVELS_MM, "n_traj": SRCA_N_TRAJ, "hf_band_hz": SRCA_HF_BAND, "interp": SRCA_INTERP,
                             "theta": SRCA_THETA, "fps": SRCA_FPS}}
        raw = {"rows": rows, "label": lab}
    elif probe == "tbc":
        rows = _run_pool(tbc_t0_job, list(seqs_all), min(procs, len(seqs_all)), "tbc-t0")
        t0agg = tbc_t0_aggregate(rows)
        res = {"probe": "TBC", "t0": t0agg, "xcorr_selfcheck": xcorr_selfcheck(), "meta": meta}
        raw = {"t0": rows}
        if t0agg["pooled"]["all"]["err_mm_max_speed"]["p95"] >= TBC_T0_NUISANCE_MM:
            t1rows = _run_pool(tbc_t1_job, list(seqs_all), min(procs, len(seqs_all)), "tbc-t1")
            res["t1"] = tbc_t1_aggregate(t1rows)
            raw["t1"] = t1rows
        else:
            res["t1"] = None
            res["t1_note"] = "T1 not run: T0 below the nuisance line"
    elif probe == "pf":
        ev = every or PF_EVERY
        meta["stride"] = ev
        jobs = []
        for s in seqs_all:
            ends = M3.list_endpoints(I.Rec(s), ev)
            jobs += [(s, ends[i:i + 2].tolist(), {}) for i in range(0, len(ends), 2)]
        recs = _run_pool(pf_job, jobs, procs, "pf")
        res = {"probe": "PF", "aggregate": pf_aggregate(recs), "meta": meta,
               "constants": {"T": PF_T, "deltas": PF_DELTAS, "scales": PF_SCALES, "static_px": PF_STATIC_PX}}
        raw = recs
    else:
        raise SystemExit(f"unknown probe {probe}")
    res["decision"] = decide(res)
    res["meta"]["wall_s"] = time.time() - t0
    pickle.dump(raw, open(out_dir / "raw.pkl", "wb"))
    res = json.loads(json.dumps(res, default=_jdefault))          # numpy scalars (incl. bool) -> plain JSON values
    MC.save_json(out_dir / "result.json", res)
    MC.receipt(out_dir, {"probe": probe, "wall_s": time.time() - t0, "verdict": res["decision"]["verdict"]}, scripts=scripts)
    print(f"[{probe}] done in {time.time() - t0:.0f} s -> {out_dir}/result.json (verdict {res['decision']['verdict']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=("srca", "tbc", "pf"), default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--procs", type=int, default=140)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    if not a.probe or not a.out:
        raise SystemExit("formal mode needs --probe and --out (and the MV2 freeze)")
    run_formal(a.probe, Path(a.out), a.procs, a.stride)


if __name__ == "__main__":
    main()
