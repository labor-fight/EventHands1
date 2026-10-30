#!/usr/bin/env python3
"""MV2 M10 / SAD-Ladder + STCA (P1-E4-R2, P1-E5-R1): how much the frozen S39 depends on simulator artefacts and on the sensor
contract. FIT7 + SELECT (the 9 training subjects' base recordings), frozen S39 x 3 seeds, CPU, no training.

Single-factor perturbations of the raw event stream inside each sample's W200 window, applied before the evaluator's own splat
(`semkine.events.splat_event_image`); every random draw is a counter-based hash of the event's global index (or of the absolute
millisecond), so an event is perturbed identically in every window that contains it. Doses are pre-registered priors, not fits.

  SAD ladder (additions / re-timing; compared with the unperturbed run)
    time_layer   retime_cluster_J{1,2,4}  whole 1-ms clusters shifted by a common random +-J ms     (keeps the cluster)
                 retime_indep_J{1,2,4}    every event shifted independently by +-J ms                (breaks the cluster)
    bin_phase    binphase_x_q{0.375,0.75} / binphase_y_q{...}  a fraction q of events moved one output pixel (an upper bound: the
                 source coordinates are gone; q = 0.375 is one source pixel at 3/8 binning)
    mask_border  border_noise_b{0.1,0.3}  b x (window events) uniform noise events in a +-3 px band around the FK silhouette of GT(end)
    background   background_r{0.05,0.2,1.0}  uniform Poisson noise, Hz per output pixel (DAVIS346 background activity ~ 0.1-1 Hz)
  STCA (deletions; compared with an equal-expected-count random thinning)
    polarity     on_keep{0.8,0.6} / off_keep{0.8,0.6}   events of one polarity kept with probability rho
    refractory   refractory_tau{0.5,1,2,5}   per merged output pixel (any polarity) an event within tau ms of the last KEPT event is
                 dropped (greedy, microsecond times from <seq>_tsub.npy)
  controls     thin_equal(<deletion>)   each deletion's equal-count random thinning; ctl_thin_keep_0.1 (positive control: must hurt)

Observables per sample (E3): the bare network on the perturbed W200 image with prev = GT(end - 50): root-aligned MPJPE (mm), and the
carry of a previous-state perturbation (root 8 deg, fingers sigma 0.10, translation 15 mm; `tools/trk6/carry_probe.py`'s draws)
through that network. Optional E1 (SELECT only, closed loop of the full S39 stack with Count50 recomputed on the perturbed events).

    python tools/mv2/m10_sad.py --smoke
    python tools/mv2/m10_sad.py --out outputs/mv2/m10_sad [--workers 24 --threads 4] [--e1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
for _p in (str(REPO), str(REPO / "model"), str(REPO / "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mv2 import common as C  # noqa: E402
from mv2.m9_age import forward_e3, ra_err  # noqa: E402
from semkine import eval_track as ET  # noqa: E402
from semkine import events as EV  # noqa: E402

H, W = C.H, C.W
BASE_KEY = 20261026
TRAIN_SUBJECTS = C.FIT7 + C.SELECT
DEP_MM = 0.38                         # SAD "dependent": dRA >= 0.38 mm (upper edge of the seed SD), 3 seeds same sign
INS_MM = 0.27                         # SAD "insensitive": all |dRA| < 0.27 mm (five-seed RA SD) ...
INS_CARRY = 0.05                      # ... and |dcarry| < 0.05 (DT2 G6 amp / carry scale)
MDE = {"local": 1.149, "global": 0.793}           # planning approximation, KS0 (#kstar-results-20261025)
MDE15 = {d: 1.5 * v for d, v in MDE.items()}      # 1.72 / 1.19 mm
ZERO_BAND = {"local": 0.38, "global": 0.26}       # STCA 'insensitive': CI contains 0 and |delta| <= seed SD
CARRY_KINDS = ("root", "fingers", "transl")
BORDER_BAND_PX = 3
CTL_POS = "ctl_thin_keep_0.1"


# ----------------------------------------------------------------------------------------------- specs
def build_specs() -> Dict[str, dict]:
    S: Dict[str, dict] = {}

    def add(name, family, kind, stca=False, **params):
        S[name] = {"name": name, "family": family, "kind": kind, "stca": stca, "params": params,
                   "key": int(hashlib.md5(name.encode()).hexdigest()[:8], 16) ^ BASE_KEY, "control_of": None, "positive_control": False}
    for J in (1, 2, 4):
        add(f"retime_cluster_J{J}", "time_layer", "retime_cluster", J=J)
        add(f"retime_indep_J{J}", "time_layer", "retime_indep", J=J)
    for q in (0.375, 0.75):
        add(f"binphase_x_q{q}", "bin_phase", "binphase", axis="x", q=q)
        add(f"binphase_y_q{q}", "bin_phase", "binphase", axis="y", q=q)
    for b in (0.1, 0.3):
        add(f"border_noise_b{b}", "mask_border", "border_noise", beta=b)
    for r in (0.05, 0.2, 1.0):
        add(f"background_r{r}", "background", "background", rate_hz=r)
    for rho in (0.8, 0.6):
        add(f"on_keep{rho}", "polarity", "polarity_thin", stca=True, pol=1, rho=rho)
        add(f"off_keep{rho}", "polarity", "polarity_thin", stca=True, pol=0, rho=rho)
    for tau in (0.5, 1.0, 2.0, 5.0):
        add(f"refractory_tau{tau}", "refractory", "refractory", stca=True, tau_ms=tau)
    for name in [n for n, s in list(S.items()) if s["stca"]]:
        cn = f"thin_equal({name})"
        S[cn] = {"name": cn, "family": "control", "kind": "thin_equal", "stca": False, "params": {}, "control_of": name,
                 "key": int(hashlib.md5(cn.encode()).hexdigest()[:8], 16) ^ BASE_KEY, "positive_control": False}
    S[CTL_POS] = {"name": CTL_POS, "family": "control", "kind": "thin_fixed", "stca": False, "params": {"keep": 0.1},
                  "control_of": None, "key": int(hashlib.md5(CTL_POS.encode()).hexdigest()[:8], 16) ^ BASE_KEY,
                  "positive_control": True}
    return S


SPECS = build_specs()
E1_SPECS = ("on_keep0.6", "off_keep0.6", "refractory_tau5.0")           # strongest deletions of each STCA family
E1_CONDS = E1_SPECS + tuple(f"thin_equal({n})" for n in E1_SPECS)


# ----------------------------------------------------------------------------------------------- randomness
_G = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def hash_uniform(idx: np.ndarray, key: int) -> np.ndarray:
    """Counter-based uniform [0, 1) (splitmix64 of idx and key): the same (idx, key) gives the same value in every window."""
    with np.errstate(over="ignore"):
        z = np.asarray(idx).astype(np.uint64) * _G + np.uint64(int(key) & 0xFFFFFFFFFFFFFFFF)
        z = (z ^ (z >> np.uint64(30))) * _M1
        z = (z ^ (z >> np.uint64(27))) * _M2
        z = z ^ (z >> np.uint64(31))
    return (z >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))


# ----------------------------------------------------------------------------------------------- windows
class Win:
    __slots__ = ("x", "y", "p", "ms", "tsub", "gidx", "w", "end", "start")

    def __init__(self, x, y, p, ms, tsub, gidx, w, end):
        self.x, self.y, self.p, self.ms, self.tsub, self.gidx = x, y, p, ms, tsub, gidx
        self.w, self.end, self.start = int(w), int(end), int(end) - int(w) + 1

    def take(self, sel) -> "Win":
        return Win(self.x[sel], self.y[sel], self.p[sel], self.ms[sel], self.tsub[sel], self.gidx[sel], self.w, self.end)

    @property
    def n(self) -> int:
        return len(self.x)


def get_window(events, offsets, tsub, end: int, w: int) -> Win:
    start = end - w + 1
    a0, a1 = int(offsets[start]), int(offsets[end + 1])
    ev = np.asarray(events[a0:a1])
    counts = np.diff(offsets[start:end + 2]).astype(np.int64)
    ms = np.repeat(np.arange(w, dtype=np.int64), counts)
    ts = np.asarray(tsub[a0:a1]).astype(np.int64) if tsub is not None else np.zeros(a1 - a0, np.int64)
    return Win(ev[:, 0].astype(np.int64), ev[:, 1].astype(np.int64), np.clip(ev[:, 2], 0, 1).astype(np.int64), ms, ts,
               np.arange(a0, a1, dtype=np.int64), w, end)


def image_of(win: Win, channels=("last",)) -> np.ndarray:
    if win.n == 0:
        return np.zeros((H, W, 2 * len(channels)), np.float32)
    return EV.splat_event_image(win.x, win.y, win.p, win.ms.astype(np.float32), win.w, H, W, channels)


def load_tsub(seq: str):
    f = C.DATA_ROOT / C.seq_dir(seq) / f"{seq}_tsub.npy"
    return np.load(f, mmap_mode="r") if f.exists() else None


# ----------------------------------------------------------------------------------------------- perturbations
def refractory_keep(pix: np.ndarray, t_us: np.ndarray, tau_us: int) -> np.ndarray:
    """Exact greedy refractory filter per pixel: an event is dropped iff a KEPT event of the same pixel lies within tau_us before
    it (ties in time and pixel are ordered by the input order). Vectorised by rounds over the blocker ranges."""
    n = len(pix)
    if n == 0:
        return np.zeros(0, bool)
    order = np.lexsort((np.arange(n), t_us, pix))
    p_s, t_s = pix[order], t_us[order]
    ck = p_s.astype(np.int64) * (1 << 40) + t_s
    lo = np.searchsorted(ck, ck - tau_us, side="right")
    idx = np.arange(n)
    nblock = idx - lo
    kept = np.zeros(n, bool)
    dropped = np.zeros(n, bool)
    undecided = np.ones(n, bool)
    for _ in range(100000):
        ck_cum = np.concatenate([[0], np.cumsum(kept)])
        dr_cum = np.concatenate([[0], np.cumsum(dropped)])
        nk = ck_cum[idx] - ck_cum[lo]
        nd = dr_cum[idx] - dr_cum[lo]
        new_drop = undecided & (nk > 0)
        new_keep = undecided & ~new_drop & (nd == nblock)
        if not (new_drop.any() or new_keep.any()):
            break
        dropped |= new_drop
        kept |= new_keep
        undecided &= ~(new_drop | new_keep)
        if not undecided.any():
            break
    out = np.empty(n, bool)
    out[order] = kept
    return out


def refractory_keep_reference(pix, t_us, tau_us):
    """Plain-Python greedy reference (for tests)."""
    n = len(pix)
    keep = np.zeros(n, bool)
    order = np.lexsort((np.arange(n), t_us, pix))
    last = {}
    for i in order:
        pxl, t = int(pix[i]), int(t_us[i])
        if pxl in last and t - last[pxl] < tau_us:
            continue
        keep[i] = True
        last[pxl] = t
    return keep


def _sorted_by_ms(win: Win, ms_new: np.ndarray) -> Win:
    order = np.argsort(ms_new, kind="stable")
    out = win.take(order)
    out.ms = ms_new[order]
    return out


def apply_spec(spec: dict, win: Win, ctx: Optional[dict] = None) -> Tuple[Win, dict]:
    """Perturb the events of one window. Returns the new window and dose counters."""
    ctx = ctx or {}
    k, pr, key = spec["kind"], spec["params"], spec["key"]
    info = {"n_in": win.n, "n_out": win.n, "n_added": 0, "n_deleted_target": 0, "n_target": 0, "n_moved": 0, "max_shift": 0,
            "refractory_violations": 0}
    if k == "identity":
        return win, info
    if k in ("retime_cluster", "retime_indep"):
        J = int(pr["J"])
        src = (win.start + win.ms) if k == "retime_cluster" else win.gidx
        sh = np.floor(hash_uniform(src, key) * (2 * J + 1)).astype(np.int64) - J
        ms_new = np.clip(win.ms + sh, 0, win.w - 1)
        info.update(n_moved=int((ms_new != win.ms).sum()), max_shift=int(np.abs(ms_new - win.ms).max()) if win.n else 0)
        return _sorted_by_ms(win, ms_new), info
    if k == "binphase":
        mv = hash_uniform(win.gidx, key) < float(pr["q"])
        out = win.take(np.ones(win.n, bool))
        if pr["axis"] == "x":
            out.x = win.x + mv
            keep = out.x < W
        else:
            out.y = win.y + mv
            keep = out.y < H
        info["n_moved"] = int(mv.sum())
        out = out.take(keep)
        info["n_out"] = out.n
        return out, info
    if k in ("border_noise", "background"):
        if k == "border_noise":
            band = ctx["band"]
            pix = np.flatnonzero(band.ravel())
            n_add = int(round(float(pr["beta"]) * win.n)) if len(pix) else 0
            rng = np.random.default_rng([key, win.end, 1])
            sel = pix[rng.integers(0, len(pix), n_add)] if n_add else np.zeros(0, np.int64)
            xa, ya = (sel % W).astype(np.int64), (sel // W).astype(np.int64)
            pa = rng.integers(0, 2, n_add).astype(np.int64)
            ma = rng.integers(0, win.w, n_add).astype(np.int64)
        else:
            lam = float(pr["rate_hz"]) * H * W / 1000.0
            xs_, ys_, ps_, ms_ = [], [], [], []
            for m in range(win.w):
                r = np.random.default_rng([key, win.start + m])
                c = int(r.poisson(lam))
                if c:
                    xs_.append(r.integers(0, W, c)); ys_.append(r.integers(0, H, c)); ps_.append(r.integers(0, 2, c))
                    ms_.append(np.full(c, m))
            cat = lambda L: np.concatenate(L).astype(np.int64) if L else np.zeros(0, np.int64)  # noqa: E731
            xa, ya, pa, ma = cat(xs_), cat(ys_), cat(ps_), cat(ms_)
            n_add = len(xa)
        base = len(win.gidx)
        merged = Win(np.concatenate([win.x, xa]), np.concatenate([win.y, ya]), np.concatenate([win.p, pa]),
                     np.concatenate([win.ms, ma]), np.concatenate([win.tsub, np.zeros(n_add, np.int64)]),
                     np.concatenate([win.gidx, np.full(n_add, -1, np.int64)]), win.w, win.end)
        out = _sorted_by_ms(merged, merged.ms)
        info.update(n_added=int(n_add), n_out=out.n)
        return out, info
    if k == "polarity_thin":
        tgt = win.p == int(pr["pol"])
        drop = tgt & (hash_uniform(win.gidx, key) >= float(pr["rho"]))
        out = win.take(~drop)
        info.update(n_out=out.n, n_target=int(tgt.sum()), n_deleted_target=int(drop.sum()))
        return out, info
    if k == "refractory":
        t_us = (win.start + win.ms) * 1000 + win.tsub
        keep = refractory_keep(win.y * W + win.x, t_us, int(round(float(pr["tau_ms"]) * 1000)))
        out = win.take(keep)
        info["n_out"] = out.n
        return out, info
    if k == "thin_equal":
        p_w = float(ctx["keep_prob"])
        keep = hash_uniform(win.gidx, key) < p_w
        out = win.take(keep)
        info.update(n_out=out.n, keep_prob=p_w)
        return out, info
    if k == "thin_fixed":
        keep = hash_uniform(win.gidx, key) < float(pr["keep"])
        out = win.take(keep)
        info["n_out"] = out.n
        return out, info
    raise ValueError(f"unknown perturbation kind {k!r}")


def silhouette_band(model, P_gt: np.ndarray, betas: np.ndarray, K: np.ndarray, chunk: int = 64) -> np.ndarray:
    """(N, H, W) bool: a +-BORDER_BAND_PX band around the FK silhouette of GT states (the model's own vertex-splat render,
    dilated to a filled region)."""
    from scipy.ndimage import binary_dilation, binary_erosion
    out = []
    with torch.no_grad():
        for i in range(0, len(P_gt), chunk):
            p = torch.from_numpy(np.asarray(P_gt[i:i + chunk], np.float32))
            b = torch.from_numpy(betas.astype(np.float32)).view(1, -1).expand(len(p), -1)
            k = torch.from_numpy(K.astype(np.float32)).view(1, 3, 3).expand(len(p), -1, -1)
            sil = model._render_prev(p, b, k)[..., 0].numpy() > 0
            for s in sil:
                filled = binary_dilation(s, iterations=3)
                band = binary_dilation(filled, iterations=BORDER_BAND_PX) & ~binary_erosion(filled, iterations=BORDER_BAND_PX)
                out.append(band)
    return np.stack(out)


# ----------------------------------------------------------------------------------------------- E3 measurement
def _carry_perturbations(P0: np.ndarray, root_deg=8.0, finger_sigma=0.10, t_mm=15.0, seed=1234) -> Dict[str, np.ndarray]:
    """`carry_probe.py`'s draws (same order) of the three previous-state perturbations."""
    from scipy.spatial.transform import Rotation as Rr
    rng = np.random.default_rng(seed)
    n = len(P0)
    ax = rng.standard_normal((n, 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    pr = P0.copy()
    for i in range(n):
        pr[i, 3:6] = (Rr.from_rotvec(ax[i] * np.deg2rad(root_deg)) * Rr.from_rotvec(P0[i, 3:6])).as_rotvec().astype(np.float32)
    pf = P0.copy()
    pf[:, 6:] += rng.standard_normal((n, 45)).astype(np.float32) * finger_sigma
    dt = rng.standard_normal((n, 3)); dt /= np.linalg.norm(dt, axis=1, keepdims=True)
    pt = P0.copy()
    pt[:, :3] += (dt * t_mm / 1000).astype(np.float32)
    return {"root": pr, "fingers": pf, "transl": pt}


def _carry(model, O0, Op, P0, Pp, kind: str, betas) -> float:
    if kind == "transl":
        din, dout = (Pp - P0)[:, :3], (Op - O0)[:, :3]
    else:
        dev = torch.device("cpu")
        raj = lambda p: C.root_aligned_joints(model, p.astype(np.float32), betas, dev)  # noqa: E731
        J0 = raj(O0)
        din = (raj(Pp) - raj(P0)).reshape(len(P0), -1)
        dout = (raj(Op) - J0).reshape(len(P0), -1)
    return float((din * dout).sum() / (din * din).sum())


def measure_seq(seed: int, seq: str, spec_names: Sequence[str], every: int = 8, max_n: Optional[int] = None, threads: int = 4,
                carry: bool = True, chunk: int = 64, model_cache: Optional[dict] = None) -> dict:
    torch.set_num_threads(threads)
    if model_cache is not None and seed in model_cache:
        model = model_cache[seed]
    else:
        model, _ = C.load_s39(seed, torch.device("cpu"))
        if model_cache is not None:
            model_cache[seed] = model
    ss = C.SeqSamples(seq, every=every, max_n=max_n)
    N = len(ss)
    model.set_hand_context(torch.from_numpy(ss.betas).view(1, -1), torch.from_numpy(ss.K).view(1, 3, 3))
    tsub = load_tsub(seq)
    wins = [get_window(ss._events, ss._offsets, tsub, int(ss.ends[i]), int(ss.windows[i])) for i in range(N)]
    P0, G, betas = ss.prev, ss.gt, ss.betas
    pert = _carry_perturbations(P0) if carry else {}
    band = silhouette_band(model, G, betas, ss.K) if any(SPECS[n]["kind"] == "border_noise" for n in spec_names) else None
    checks = {"identity_bit_exact": True, "deterministic": True}
    keepfrac: Dict[str, np.ndarray] = {}
    dose: Dict[str, dict] = {}
    table: Dict[str, dict] = {}

    def run_cond(name: str, spec: dict, with_carry: bool):
        ctx_list = []
        imgs, infos, kf = [], [], np.zeros(N)
        for i, win in enumerate(wins):
            ctx = {}
            if spec["kind"] == "border_noise":
                ctx["band"] = band[i]
            if spec["kind"] == "thin_equal":
                ctx["keep_prob"] = keepfrac[spec["control_of"]][i]
            out, info = apply_spec(spec, win, ctx)
            imgs.append(image_of(out))
            infos.append(info)
            kf[i] = info["n_out"] / max(info["n_in"], 1)
            if i < 3 and spec["kind"] != "identity":
                out2, _ = apply_spec(spec, win, ctx)
                checks["deterministic"] &= bool(np.array_equal(out.gidx, out2.gidx) and np.array_equal(out.ms, out2.ms)
                                                and np.array_equal(out.x, out2.x))
        keepfrac[name] = kf
        X = torch.from_numpy(np.stack(imgs))
        O0 = forward_e3(model, X, P0, chunk)
        err = ra_err(model, O0, G, betas)
        row = {"ra_mm": float(err.mean()), "n": int(N)}
        if with_carry:
            row["carry"] = {kd: _carry(model, O0, forward_e3(model, X, pert[kd], chunk), P0, pert[kd], kd, betas) for kd in CARRY_KINDS}
        table[name] = row
        keys = ("n_in", "n_out", "n_added", "n_deleted_target", "n_target", "n_moved", "refractory_violations")
        dose[name] = {k: int(sum(i[k] for i in infos)) for k in keys}
        dose[name]["max_shift"] = int(max(i["max_shift"] for i in infos))
        dose[name]["expected_out"] = float(sum(i["n_in"] for i in infos))
        if spec["kind"] == "thin_equal":
            dose[name]["expected_out"] = float(sum(i["n_in"] * keepfrac[spec["control_of"]][j] for j, i in enumerate(infos)))
        return X, O0

    ident = {"name": "plain", "kind": "identity", "params": {}, "key": 0}
    X_plain, O_plain = run_cond("plain", ident, carry)
    # identity path == the evaluator's `build_lnes`, bit for bit
    X_ref = torch.from_numpy(np.stack([ET.build_lnes(ss._events, ss._offsets, int(ss.ends[i]), int(ss.windows[i]), ss.channels)
                                       for i in range(N)]))
    checks["identity_bit_exact"] = bool(torch.equal(X_plain, X_ref))
    order = [n for n in spec_names if SPECS[n]["kind"] != "thin_equal"] + [n for n in spec_names if SPECS[n]["kind"] == "thin_equal"]
    for n in order:
        spec = SPECS[n]
        if spec["kind"] == "thin_equal" and spec["control_of"] not in keepfrac:
            continue
        run_cond(n, spec, carry and spec["kind"] not in ("thin_equal", "thin_fixed"))
    return {"seed": seed, "seq": seq, "table": table, "dose": dose, "checks": checks, "n": N}


# ----------------------------------------------------------------------------------------------- E1 (closed loop, SELECT)
def make_window_fn(spec: Optional[dict], tsub):
    if spec is None:
        return None

    def fn(events, offsets, end, w, ev_ch):
        win = get_window(events, offsets, tsub, end, w)
        ctx = {}
        if spec["kind"] == "thin_equal":
            _, ref = apply_spec(SPECS[spec["control_of"]], win)
            ctx["keep_prob"] = ref["n_out"] / max(ref["n_in"], 1)
        out, _ = apply_spec(spec, win, ctx)
        return image_of(out, ev_ch), int((out.ms >= w - 50).sum())
    return fn


_E1: Dict[str, object] = {}


def _e1_init(threads: int):
    torch.set_num_threads(threads)
    _E1["stacks"] = {}


def _e1_task(args):
    from mv2 import m9_psq as PSQ
    seed, seq, cond = args
    if seed not in _E1["stacks"]:
        model, net, cfg = PSQ.build_stack(seed)
        _E1["stacks"][seed] = (net, cfg, PSQ.get_mano(cfg))
    net, cfg, mano = _E1["stacks"][seed]
    tsub = load_tsub(seq)
    spec = None if cond == "plain" else SPECS[cond]
    r = PSQ.closed_loop(net, cfg, seq, 0, init_seed=2000 + (C.SELECT.index(C.subject_of(seq)) if C.subject_of(seq) in C.SELECT else 0), mano=mano,
                        event_window_fn=make_window_fn(spec, tsub))
    return {"seed": seed, "seq": seq, "cond": cond, "ra_mm": float(r["ra_mm"].mean()), "n": int(len(r["ra_mm"]))}


# ----------------------------------------------------------------------------------------------- aggregation / decision
def _domain(seq: str) -> str:
    return "local" if seq.endswith("_local") else "global"


def derive(result: dict) -> dict:
    """Per seed and domain: dRA (mean over the 9 subjects of ra_cond - ra_plain), dcarry, and for each STCA deletion the
    differential against its equal-count thinning with a subject-block bootstrap CI."""
    out: Dict[str, dict] = {}
    for s, r in result["seeds"].items():
        tab = r["table"]
        o: Dict[str, dict] = {}
        for dom in ("local", "global"):
            seqs = [q for q in tab if _domain(q) == dom]
            subj = [C.subject_of(q) for q in seqs]
            for cond in SPECS:
                if not all(cond in tab[q] for q in seqs):
                    continue
                d = np.array([tab[q][cond]["ra_mm"] - tab[q]["plain"]["ra_mm"] for q in seqs])
                rec = {"dRA": float(d.mean()), "per_subject": {u: float(v) for u, v in zip(subj, d)}}
                if all("carry" in tab[q][cond] for q in seqs):
                    rec["dcarry"] = {kd: float(np.mean([tab[q][cond]["carry"][kd] - tab[q]["plain"]["carry"][kd] for q in seqs]))
                                     for kd in CARRY_KINDS}
                if SPECS[cond]["stca"]:
                    cn = f"thin_equal({cond})"
                    if all(cn in tab[q] for q in seqs):
                        diff = np.array([tab[q][cond]["ra_mm"] - tab[q][cn]["ra_mm"] for q in seqs])
                        lo, hi = C.block_bootstrap_ci(diff, np.array(subj), n_boot=2000, seed=BASE_KEY)
                        rec["diff_vs_thinning"] = {"mean": float(diff.mean()), "ci": [lo, hi]}
                o.setdefault(dom, {})[cond] = rec
        out[s] = o
    e1 = result.get("e1")
    if e1:
        for s, per in e1.items():
            for dom in ("local", "global"):
                seqs = [q for q in per if _domain(q) == dom]
                if not seqs or "plain" not in per[seqs[0]]:
                    continue
                for cond in E1_CONDS:
                    if all(cond in per[q] for q in seqs):
                        rec = out[s][dom].setdefault(cond, {})
                        rec["dRA_e1"] = float(np.mean([per[q][cond] - per[q]["plain"] for q in seqs]))
                for n in E1_SPECS:
                    cn = f"thin_equal({n})"
                    if n in out[s][dom] and all(n in per[q] and cn in per[q] for q in seqs):
                        out[s][dom][n]["diff_e1_vs_thinning"] = float(np.mean([per[q][n] - per[q][cn] for q in seqs]))
    return out


def _dose_ok(name: str, d: dict) -> Tuple[bool, str]:
    spec = SPECS[name]
    k, pr = spec["kind"], spec["params"]
    if k == "polarity_thin":
        frac = d["n_deleted_target"] / max(d["n_target"], 1)
        return abs(frac - (1 - pr["rho"])) <= 0.01, f"deleted target fraction {frac:.4f} vs {1 - pr['rho']:.4f}"
    if k == "thin_equal":
        return abs(d["n_out"] - d["expected_out"]) <= 0.01 * max(d["n_in"], 1), "equal-count thinning"
    if k == "thin_fixed":
        return abs(d["n_out"] / max(d["n_in"], 1) - pr["keep"]) <= 0.01, "fixed thinning"
    if k == "background":
        return True, "n/a"
    if k in ("retime_cluster", "retime_indep"):
        return d["max_shift"] <= pr["J"], f"max shift {d['max_shift']}"
    if k == "binphase":
        frac = d["n_moved"] / max(d["n_in"], 1)
        return abs(frac - pr["q"]) <= 0.01, f"moved fraction {frac:.4f} vs {pr['q']}"
    if k == "refractory":
        return d["refractory_violations"] == 0 and d["n_out"] <= d["n_in"], "greedy filter"
    return True, "n/a"


def decide(result: dict) -> dict:
    """Pre-registered gates of the SAD-Ladder and STCA (tools/mv2/prereg_m10.md section 3).

    tool validity (else `tool_defect`): the unperturbed path equals `build_lnes` bit for bit and perturbations are deterministic
    (every seed / sequence); the delivered doses match the specification (polarity deletion fraction +-0.01, binphase moved fraction
    +-0.01, equal-count thinning +-1 %, retime shift <= J, refractory output <= input); the positive control (keep 10 % of the events)
    raises RA by >= 0.38 mm in both domains in every seed.
    SAD classes per perturbation and domain (3 seeds): `dependent` = mean dRA >= 0.38 mm and all seeds > 0; `insensitive` = every
    |dRA| < 0.27 mm and every |dcarry| < 0.05; else `intermediate`.
    STCA (deletions): differential = RA(T) - RA(equal-count thinning), per-seed subject-block bootstrap CI. `insensitive` = every
    seed's CI contains 0 and |dRA| <= 0.38 (local) / 0.26 (global); `sensitive` = differential > 1.5 x MDE (1.72 / 1.19 mm) in every
    seed; else `intermediate`. A sensitive STCA item => `robust_training_needed` (before any real-sensor claim; M3 / M4 / M8 signed /
    count channels closed). E1 (SELECT) is reported, not gated.
    verdict: robust_training_needed > artifact_dependent > insensitive_all > intermediate."""
    seeds = result["seeds"]
    for s, r in seeds.items():
        for q, rec in r["per_seq"].items():
            if not rec["checks"]["identity_bit_exact"] or not rec["checks"]["deterministic"]:
                return {"verdict": "tool_defect", "notes": [f"seed {s} {q}: identity path / determinism check failed"]}
    pooled: Dict[str, dict] = {}
    for s, r in seeds.items():
        for q, rec in r["per_seq"].items():
            for n, d in rec["dose"].items():
                if n == "plain":
                    continue
                acc = pooled.setdefault(n, {})
                for k, v in d.items():
                    acc[k] = max(acc.get(k, 0), v) if k == "max_shift" else acc.get(k, 0) + v
    bad = [f"{n}: {msg}" for n in pooled for ok, msg in [_dose_ok(n, pooled[n])] if not ok]
    if bad:
        return {"verdict": "tool_defect", "notes": ["dose not delivered as specified"] + bad}
    der = derive(result)
    sd = sorted(der)
    for dom in ("local", "global"):
        if not all(CTL_POS in der[s][dom] and der[s][dom][CTL_POS]["dRA"] >= DEP_MM for s in sd):
            return {"verdict": "tool_defect", "notes": [f"positive control (keep 10 % of events) did not raise RA by >= {DEP_MM} mm in {dom}"]}
    sad, stca, rows = {}, {}, {}
    for name, spec in SPECS.items():
        if spec["control_of"] or spec["positive_control"]:
            continue
        for dom in ("local", "global"):
            d = [der[s][dom][name] for s in sd]
            dra = [x["dRA"] for x in d]
            carry = [abs(v) for x in d for v in x.get("dcarry", {}).values()]
            cls = ("dependent" if (np.mean(dra) >= DEP_MM and all(v > 0 for v in dra)) else
                   "insensitive" if (max(abs(v) for v in dra) < INS_MM and (max(carry) if carry else 0.0) < INS_CARRY) else "intermediate")
            rows[f"{name}|{dom}"] = {"dRA": dra, "max_abs_dcarry": max(carry) if carry else None, "sad_class": cls}
            if spec["stca"]:
                df = [x["diff_vs_thinning"] for x in d]
                zero_ok = all(x["ci"][0] <= 0 <= x["ci"][1] for x in df) and all(abs(v) <= ZERO_BAND[dom] for v in dra)
                sens = all(x["mean"] > MDE15[dom] for x in df)
                rows[f"{name}|{dom}"]["stca"] = {"diff": [x["mean"] for x in df], "ci": [x["ci"] for x in df],
                                                  "class": "sensitive" if sens else "insensitive" if zero_ok else "intermediate"}
                stca[f"{name}|{dom}"] = rows[f"{name}|{dom}"]["stca"]["class"]
            else:
                sad[f"{name}|{dom}"] = cls
    robust = sorted(k for k, v in stca.items() if v == "sensitive")
    dependent = sorted(k for k, v in sad.items() if v == "dependent")
    all_ins = all(v == "insensitive" for v in sad.values()) and all(v == "insensitive" for v in stca.values())
    verdict = ("robust_training_needed" if robust else "artifact_dependent" if dependent else
               "insensitive_all" if all_ins else "intermediate")
    return {"verdict": verdict, "robust_training_needed": robust, "dependent": dependent, "rows": rows,
            "families": sorted({SPECS[k.split("|")[0]]["family"] for k in dependent + robust})}


# ----------------------------------------------------------------------------------------------- driver
def _seq_task(args):
    seed, seq, names, every, threads, carry = args
    cache = _E1.setdefault("model_cache", {})
    return measure_seq(seed, seq, names, every=every, threads=threads, carry=carry, model_cache=cache)


def run(workers: int, every: int, threads: int, e1: bool, seqs=None, names=None, carry: bool = True) -> dict:
    seqs = list(seqs) if seqs is not None else C.base_seqs(TRAIN_SUBJECTS)
    names = list(names) if names is not None else list(SPECS)
    t0 = time.time()
    tasks = [(s, q, names, every, threads, carry) for s in C.SEEDS for q in seqs]
    if workers <= 1:
        res = [_seq_task(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_seq_task, tasks, chunksize=1)
    out = {"seeds": {}, "specs": {n: {k: v for k, v in SPECS[n].items()} for n in names}, "every": every}
    for r in res:
        out["seeds"].setdefault(str(r["seed"]), {"table": {}, "per_seq": {}})
        out["seeds"][str(r["seed"])]["table"][r["seq"]] = r["table"]
        out["seeds"][str(r["seed"])]["per_seq"][r["seq"]] = {"dose": r["dose"], "checks": r["checks"], "n": r["n"]}
    if e1:
        sel = C.base_seqs(C.SELECT)
        tasks = [(s, q, c) for s in C.SEEDS for q in sel for c in ("plain",) + E1_CONDS]
        with mp.get_context("spawn").Pool(workers, initializer=_e1_init, initargs=(threads,)) as pool:
            r1 = pool.map(_e1_task, tasks, chunksize=1)
        e1o: Dict[str, dict] = {}
        for r in r1:
            e1o.setdefault(str(r["seed"]), {}).setdefault(r["seq"], {})[r["cond"]] = r["ra_mm"]
        out["e1"] = e1o
    out["secs"] = time.time() - t0
    return out


# ----------------------------------------------------------------------------------------------- smoke
def _synth_win(rng, w=200, n_per_ms=60, end=1000) -> Win:
    counts = rng.poisson(n_per_ms, w)
    ms = np.repeat(np.arange(w, dtype=np.int64), counts)
    n = len(ms)
    x = np.clip(rng.normal(120, 12, n), 0, W - 1).astype(np.int64)
    y = np.clip(rng.normal(90, 10, n), 0, H - 1).astype(np.int64)
    return Win(x, y, rng.integers(0, 2, n), ms, rng.integers(0, 1000, n), np.arange(5000, 5000 + n), w, end)


def smoke(args) -> int:
    t0 = time.time()
    torch.set_num_threads(max(1, args.threads))
    ok = True
    rng = np.random.default_rng(3)
    win = _synth_win(rng)
    # hash: uniform and counter-based
    u = hash_uniform(np.arange(200000), 99)
    hash_ok = abs(u.mean() - 0.5) < 0.005 and abs(u.var() - 1 / 12) < 0.002 and np.array_equal(u[100:110], hash_uniform(np.arange(100, 110), 99))
    print(f"[m10_sad smoke] counter-based hash uniform and window-independent: {bool(hash_ok)}")
    ok &= bool(hash_ok)
    # refractory: exact against the Python reference on random events
    r_ok = True
    for tau in (500, 1000, 2000, 5000):
        pix = rng.integers(0, 40, 3000)
        t = np.sort(rng.integers(0, 400000, 3000))
        r_ok &= bool(np.array_equal(refractory_keep(pix, t, tau), refractory_keep_reference(pix, t, tau)))
    print(f"  refractory filter equals the greedy Python reference (4 tau, 3000 events each): {r_ok}")
    ok &= r_ok
    # every perturbation kind on a synthetic window: delivered dose / shapes
    band = np.zeros((H, W), bool)
    band[60:120, 90:150] = True
    ctx = {"band": band, "keep_prob": 0.7}
    checks = {}
    for name, spec in SPECS.items():
        c = dict(ctx)
        if spec["kind"] == "thin_equal":
            c["keep_prob"] = 0.7
        out, info = apply_spec(spec, win, c)
        img = image_of(out)
        fine = img.shape == (H, W, 2) and np.isfinite(img).all() and (np.diff(out.ms) >= 0).all()
        if spec["kind"] == "refractory":
            t = (out.start + out.ms) * 1000 + out.tsub
            pix = out.y * W + out.x
            o = np.lexsort((t, pix))
            same = pix[o][1:] == pix[o][:-1]
            fine &= bool((np.diff(t[o])[same] >= round(spec["params"]["tau_ms"] * 1000)).all())
        checks[name] = bool(fine)
    print(f"  all {len(SPECS)} conditions run on a synthetic window with finite images, time-ordered output (refractory gap respected): "
          f"{all(checks.values())}")
    ok &= all(checks.values())
    # dose delivery on the synthetic window
    d = {n: apply_spec(SPECS[n], win, ctx)[1] for n in ("on_keep0.6", "binphase_x_q0.375", "retime_indep_J2", CTL_POS)}
    dose_ok = all(_dose_ok(n, d[n])[0] for n in d)
    print(f"  delivered doses within tolerance on the synthetic window (polarity, bin-phase, retime, fixed thinning): {dose_ok}")
    ok &= dose_ok
    # the real path on a tiny FIT7 slice: identity == build_lnes bit for bit, shapes only
    names = ["retime_indep_J2", "binphase_x_q0.375", "border_noise_b0.1", "background_r0.2", "on_keep0.6", "refractory_tau2.0",
             "thin_equal(on_keep0.6)", "thin_equal(refractory_tau2.0)", CTL_POS]
    r = measure_seq(3407, "ch_local", names, every=8, max_n=args.max_n, threads=args.threads, carry=True)
    print(f"  tool path on ch_local ({r['n']} samples, {len(r['table']) - 1} conditions + plain, E3 + 3 carries): identity path == "
          f"build_lnes bit for bit: {r['checks']['identity_bit_exact']}, deterministic: {r['checks']['deterministic']}")
    ok &= r["checks"]["identity_bit_exact"] and r["checks"]["deterministic"] and len(r["table"]) == len(names) + 1
    # window-independence: an event is perturbed identically in two overlapping windows (hash keyed by global index / absolute ms)
    from mv2 import m9_psq as PSQ
    ev_s, off_s = PSQ.synth_stream(np.random.default_rng(5), 600, 40.0)
    w1, w2 = get_window(ev_s, off_s, None, 400, 200), get_window(ev_s, off_s, None, 450, 200)
    cons = True
    for nm in ("retime_indep_J2", "retime_cluster_J2", "on_keep0.6", "binphase_x_q0.375", "background_r0.2", "thin_equal(on_keep0.6)"):
        sp = SPECS[nm]
        c1 = {"keep_prob": 0.7, "band": band}
        o1, _ = apply_spec(sp, w1, c1)
        o2, _ = apply_spec(sp, w2, {"keep_prob": 0.7, "band": band})
        t1 = o1.start + o1.ms                     # absolute ms of the perturbed events
        t2 = o2.start + o2.ms
        lo, hi = max(w1.start, w2.start) + 8, w1.end - 8             # overlap, away from the clip edges of the re-timing
        a1 = sorted(zip(t1[(t1 >= lo) & (t1 <= hi)].tolist(), o1.gidx[(t1 >= lo) & (t1 <= hi)].tolist()))
        a2 = sorted(zip(t2[(t2 >= lo) & (t2 <= hi)].tolist(), o2.gidx[(t2 >= lo) & (t2 <= hi)].tolist()))
        cons &= a1 == a2
    print(f"  perturbed events identical in overlapping windows (retime x2, polarity, bin-phase, background, thinning): {cons}")
    ok &= cons
    # the carry estimator recovers a known gain on the translation block
    P0r = r_prev = np.zeros((40, 51), np.float32)
    Pp = P0r.copy(); Pp[:, :3] = np.random.default_rng(2).standard_normal((40, 3)).astype(np.float32) * 0.01
    Op = P0r + 0.37 * (Pp - P0r)
    cr = _carry(None, P0r, Op, P0r, Pp, "transl", None)
    print(f"  carry estimator recovers a known translation gain: {abs(cr - 0.37) < 1e-6}")
    ok &= abs(cr - 0.37) < 1e-6
    # E1 plumbing: an identity event-window function reproduces the evaluator's loop bit for bit
    model1, net1, cfg1 = PSQ.build_stack(3407)
    ident_spec = {"kind": "identity", "params": {}, "key": 0, "control_of": None}
    a_ = PSQ.closed_loop(net1, cfg1, "ch_local", 0, max_steps=6)
    b_ = PSQ.closed_loop(net1, cfg1, "ch_local", 0, max_steps=6, event_window_fn=make_window_fn(ident_spec, load_tsub("ch_local")))
    e1_ok = bool(np.array_equal(a_["pred"], b_["pred"]))
    print(f"  E1 plumbing: identity event-window function == plain closed loop over 6 steps: {e1_ok}")
    ok &= e1_ok
    # decide() plumbing on a synthetic result (no effects): two subjects per domain, consistent dose records
    def fake_dose(sp):
        d = {"n_in": 1000, "n_out": 1000, "n_added": 0, "n_deleted_target": 0, "n_target": 1000, "n_moved": 0, "max_shift": 0,
             "refractory_violations": 0, "expected_out": 1000.0}
        if sp["kind"] == "polarity_thin":
            d["n_deleted_target"] = int(round(1000 * (1 - sp["params"]["rho"])))
            d["n_out"] = 1000 - d["n_deleted_target"]
        if sp["kind"] == "binphase":
            d["n_moved"] = int(round(1000 * sp["params"]["q"]))
        if sp["kind"] == "thin_fixed":
            d["n_out"] = int(round(1000 * sp["params"]["keep"]))
        return d

    def fake(dra_pos=2.0):
        seeds = {}
        for sd_ in C.SEEDS:
            tab, per = {}, {}
            for q in ("a_local", "b_local", "a_global", "b_global"):
                t = {"plain": {"ra_mm": 4.0, "n": 10, "carry": {k: 0.5 for k in CARRY_KINDS}}}
                for n, spec in SPECS.items():
                    row = {"ra_mm": 4.0 + (dra_pos if n == CTL_POS else 0.0), "n": 10}
                    if not (spec["control_of"] or spec["positive_control"]):
                        row["carry"] = {k: 0.5 for k in CARRY_KINDS}
                    t[n] = row
                tab[q] = t
                per[q] = {"dose": {n: fake_dose(sp) for n, sp in SPECS.items()},
                          "checks": {"identity_bit_exact": True, "deterministic": True}, "n": 10}
            seeds[str(sd_)] = {"table": tab, "per_seq": per}
        return {"seeds": seeds}
    d1, d2 = decide(fake(2.0)), decide(fake(0.0))
    print(f"  decide() plumbing on synthetic tables: null effect -> {d1['verdict']}, dead positive control -> {d2['verdict']}")
    ok &= d1["verdict"] == "insensitive_all" and d2["verdict"] == "tool_defect"
    out_dir = REPO / "outputs" / "mv2" / "smoke" / "m10_sad"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "smoke.json").write_text(json.dumps({"checks": r["checks"], "ok": bool(ok)}, indent=1))
    print(f"[m10_sad smoke] {'PASS' if ok else 'FAIL'} ({time.time() - t0:.1f} s)")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--max-n", type=int, default=6)
    ap.add_argument("--e1", action="store_true", help="also the closed-loop E1 on SELECT for the strongest deletions")
    a = ap.parse_args()
    if a.smoke:
        raise SystemExit(smoke(a))
    if not a.out:
        raise SystemExit("--out outputs/mv2/m10_sad (formal) or --smoke")
    C.verify_protocol()
    out_dir = Path(a.out)
    res = run(a.workers, a.every, a.threads, a.e1)
    res["derived"] = derive(res)
    res["decision"] = decide(res)
    C.save_json(out_dir / "result.json", res)
    C.receipt(out_dir, {"probe": "m10_sad", "verdict": res["decision"]["verdict"]},
              scripts=["tools/mv2/m10_sad.py", "tools/mv2/m9_age.py", "tools/mv2/m9_psq.py", "tools/mv2/common.py"])
    print(json.dumps({k: res["decision"][k] for k in res["decision"] if k != "rows"}, indent=1, default=C._np))


if __name__ == "__main__":
    main()
