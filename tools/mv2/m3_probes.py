#!/usr/bin/env python3
"""MV2 / M3 (material-point event accumulation, MPES) zero-training CPU probes.

    FBW   focus-basin width of the subtree IWE under anatomical FK-warp errors (R3 source P2-E2-R1)
    SMG   surface-sharpness-gradient position innovation, FIT7-fitted ridge readout with augmentation (P3-E10-R1)
    EXPO  censoring-corrected exposure accumulation: single / class-calibrated / class + censoring + ageing emission,
          priority rate at 50 / 100 / 200 ms (P1-E2-R2; a CPU port of tools/sba/r3_e8.py)

    python tools/mv2/m3_probes.py --probe fbw  --smoke
    python tools/mv2/m3_probes.py --probe fbw  --out outputs/mv2/fbw           # formal (needs the freeze)
    python tools/mv2/m3_probes.py --probe smg  --out outputs/mv2/smg
    python tools/mv2/m3_probes.py --probe expo --out outputs/mv2/expo

`decide(result)` applies exactly the gates of tools/mv2/prereg_m3.md. Smoke mode prints shapes / counts / timings and pass / fail of
internal unit checks only (never a statistic that could be an effect) and is the only mode that runs without
`outputs/mv2/protocol.json`. Formal mode first calls `common.verify_protocol()`. Everything runs on the CPU
(`CUDA_VISIBLE_DEVICES=""`), reads only BASE recordings of FIT7 / SELECT, never zgz.
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
from sba import r3_probe as PR  # noqa: E402  (E0 anatomical perturbation machinery, read-only)

SEED = 20261026
FIT7, SELECT = MC.FIT7, MC.SELECT
FINGERS5 = I.FINGERS5                                   # thumb, index, middle, ring, pinky
GID = {f: I.GROUP_OF[f] for f in FINGERS5}
LONG4 = ("index", "middle", "ring", "pinky")
S39_CONFIG = REPO / "outputs" / "semkine" / "dt_dz_l3_s3407" / "config_resolved.yaml"

# ------------------------------------------------------------------------------------------------ FBW constants
DELTAS = (2, 5, 10, 14, 19)                              # mm of true FK joint error (E0 definition, wrist-aligned)
T_GRID = (50, 100, 200, 400, 600)                        # ms
T_MAX = max(T_GRID)
FAM_LONG = ("pip", "mcpf")                               # single-finger flexion families of E0 (PIP / MCP)
CELL_F5 = (5,)
CELL_F10 = (10,)
FBW_MIN_EVENTS = 20                                      # events a (subtree, T, channel) cell needs for a contrast
N_CAP = 30000                                            # event cap per 600 ms window (uniform, seeded), same for every T
CROP_MARGIN = 14.0                                       # px around the GT hand extents
PN_CLASSES = ("small", "large", "directed")              # TRACK.PREV_NOISE_MODE = mixed, the three classes
PN_SEEDS = 4
NULL_DRAWS = 2
FBW_EVERY = 8                                            # endpoint stride (x 50 ms)
HALF_GAIN = 0.5                                          # basin edge: focus gain falls below this fraction of the GT gain
T_STAR_PASS, T_STAR_KILL = 200, 100
N_SUB_PASS = 3
PREF_FRAC_PASS, PREF_FRAC_KILL = 0.8, 0.5
CH = ("all", "texture", "contour")
N_BOOT = 2000
BOOT_SEED = 20261026
MAX_FAIL_FRAC = 0.02                                     # more failed endpoints than this is a tool defect
BLOCK_MS = 5000                                          # bootstrap block: 5 s of endpoints of one recording
E1_CHECK = REPO / "outputs" / "sba" / "r3" / "e1" / "result_check.json"     # ledger anchor: SBA preference rates


def domain_of(seq: str) -> str:
    return "local" if seq.endswith("_local") else "global"


def list_endpoints(rec: "I.Rec", every: int, phase: int = 0, min_hist: int = T_MAX + 1) -> np.ndarray:
    """Endpoint ms of the evaluator's 50 ms grid with >= min_hist ms of history inside the run (first step of a run
    skipped as in `common.endpoints`), every `every`-th."""
    out = []
    for ra, rb in rec.runs:
        ends = np.arange(ra + MC.STEP - 1, rb, MC.STEP, dtype=np.int64)
        for k, end in enumerate(ends):
            if k == 0 or (k - phase) % every:
                continue
            if end + 1 - ra >= min_hist:
                out.append(int(end))
    return np.asarray(out, np.int64)


# ------------------------------------------------------------------------------------------------ statistics
class GroupBoot:
    """Group bootstrap: resamples whole groups; every statistic is a function of weighted means of per-sample arrays."""

    def __init__(self, groups: Sequence, n_boot: int = N_BOOT, seed: int = BOOT_SEED):
        ug, self.inv = np.unique(np.asarray(groups).astype(str), return_inverse=True)
        self.G = len(ug)
        rng = np.random.default_rng(seed)
        self.W = rng.multinomial(self.G, np.full(self.G, 1.0 / self.G), size=n_boot).astype(np.float64)   # (B, G)

    def gsum(self, x: np.ndarray):
        """Per-group sums and counts of finite entries of x (n,) or (n, k)."""
        x = np.asarray(x, np.float64)
        if x.ndim == 1:
            x = x[:, None]
        fin = np.isfinite(x)
        s = np.zeros((self.G, x.shape[1]))
        c = np.zeros((self.G, x.shape[1]))
        for j in range(x.shape[1]):
            s[:, j] = np.bincount(self.inv, weights=np.where(fin[:, j], x[:, j], 0.0), minlength=self.G)
            c[:, j] = np.bincount(self.inv, weights=fin[:, j].astype(float), minlength=self.G)
        return s, c

    def wmean(self, x: np.ndarray):
        """(point (k,), boot (B, k)) of the mean over finite entries."""
        s, c = self.gsum(x)
        pt = s.sum(0) / np.maximum(c.sum(0), 1e-12)
        bt = (self.W @ s) / np.maximum(self.W @ c, 1e-12)
        pt = np.where(c.sum(0) > 0, pt, np.nan)
        return pt, bt


def ci(bt: np.ndarray, q=(2.5, 97.5)):
    return [float(np.nanpercentile(bt, q[0])), float(np.nanpercentile(bt, q[1]))]


def group_keys(seqs, ends) -> np.ndarray:
    return np.asarray([f"{s}|{int(e) // BLOCK_MS}" for s, e in zip(seqs, ends)])


# ------------------------------------------------------------------------------------------------ noise / perturbations
def s39_noise_constants() -> dict:
    """TRACK.PREV_NOISE_* of the frozen S39 configuration."""
    cfg = {}
    for line in S39_CONFIG.read_text().splitlines():
        line = line.strip()
        if line.startswith("PREV_NOISE"):
            k, v = line.split(":")
            cfg[k.strip()] = v.strip()
    return {"small": (float(cfg["PREV_NOISE_T"]), float(cfg["PREV_NOISE_R"]), float(cfg["PREV_NOISE_POSE"])),
            "large": (float(cfg["PREV_NOISE_LARGE_T"]), float(cfg["PREV_NOISE_LARGE_R"]),
                      float(cfg["PREV_NOISE_LARGE_POSE"])),
            "mix": (float(cfg["PREV_NOISE_P_SMALL"]), float(cfg["PREV_NOISE_P_LARGE"]), float(cfg["PREV_NOISE_P_CORR"]))}


def prev_noise(rng: np.random.Generator, cls: str, const: Optional[dict] = None) -> np.ndarray:
    """One draw of the 51-d additive `PREV_NOISE` of class small / large / directed, as `SemKineDataset._sample_prev_noise`
    (semkine/dataset.py:367-397) forced into the class."""
    const = const or s39_noise_constants()

    def fill(a, b, c):
        n = np.empty(51, np.float32)
        n[0:3] = rng.standard_normal(3) * a
        n[3:6] = rng.standard_normal(3) * b
        n[6:51] = rng.standard_normal(45) * c
        return n

    if cls == "small":
        return fill(*const["small"])
    if cls == "large":
        return fill(*const["large"])
    lt, lr, lp = const["large"]

    def directed(n, mag):
        v = rng.standard_normal(n).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-8) * np.float32(rng.uniform(0.0, mag))

    n = np.empty(51, np.float32)
    n[0:3], n[3:6], n[6:51] = directed(3, lt), directed(3, lr), directed(45, lp)
    return n


class RecCtx:
    """Per-process cache of one BASE recording with its E0 axis frame."""
    _cache: Dict[str, "RecCtx"] = {}

    def __init__(self, seq: str):
        self.seq = seq
        self.rec = I.Rec(seq)
        self.fr = PR.make_frame(self.rec.hand.ctx, self.rec.hand.betas)

    @classmethod
    def get(cls, seq: str) -> "RecCtx":
        if seq not in cls._cache:
            cls._cache.clear()
            cls._cache[seq] = RecCtx(seq)
        return cls._cache[seq]


def make_specs(rc: RecCtx, x_end: torch.Tensor, fingers=FINGERS5, deltas=DELTAS, with_syn=True):
    """E0 hypotheses scaled by the true FK joint error at the end knot: {(kind, finger|None, sign, delta): Spec}.
    single-finger flexion (pip / mcpf) for the long fingers, thumb opposition for the thumb, de-rooted synergy."""
    ctx, betas, fr = rc.rec.hand.ctx, rc.rec.hand.betas, rc.fr
    j_clean = PR.joints(ctx, x_end[None], betas)
    specs: Dict[tuple, "PR.Spec"] = {}
    for f in fingers:
        fams = ("thm",) if f == "thumb" else FAM_LONG
        ax = PR.thumb_axis(ctx, fr, x_end, betas).tolist() if f == "thumb" else None
        for fam in fams:
            for sign in (1, -1):
                for d in deltas:
                    deg, ok = PR.bisect(ctx, fr, fam, sign, float(d), x_end, betas, f if f != "thumb" else "index", ax,
                                        None, j_clean)
                    if ok:
                        specs[(fam, f, sign, d)] = PR.Spec(fam, sign, deg, f if f != "thumb" else "index", ax)
    if with_syn:
        Bj = PR.root_jacobian(ctx, x_end, betas)
        for sign in (1, -1):
            for d in deltas:
                deg, ok = PR.bisect(ctx, fr, "synrc", sign, float(d), x_end, betas, "index", None, Bj, j_clean)
                if ok:
                    sp = PR.Spec("synrc", sign, deg, "index", None)
                    _, om = PR._syn_rc(ctx, fr, x_end[None], torch.tensor([sign * deg]), Bj, j_clean, betas)
                    sp.omega = om[0].tolist()
                    specs[("synrc", "long4", sign, d)] = sp
    return specs


def finger_node_error(rc: RecCtx, x_hat: torch.Tensor, x_gt: torch.Tensor) -> Dict[str, float]:
    """Wrist-aligned mean FK node error (mm) of the four OpenPose nodes of every finger, hat vs GT (the E0 metric)."""
    ctx, betas = rc.rec.hand.ctx, rc.rec.hand.betas
    jh = PR.joints(ctx, x_hat.view(1, 51), betas)
    jc = PR.joints(ctx, x_gt.view(1, 51), betas)
    e21, e_abs = PR.errors(jh, jc)
    out = {f: float(e21[0, list(PR.OP[f])].mean()) for f in FINGERS5}
    out["j21"] = float(e21[0].mean())
    out["abs21"] = float(e_abs[0])
    return out


# ================================================================================================ FBW
def _crop_events(traj: "I.Traj", ev: "I.Events", margin: float = CROP_MARGIN):
    """Events inside the GT hand's extents (visible vertices over the window) + margin."""
    uv = traj.uv[traj.vis]
    lo, hi = uv.min(0).values.numpy() - margin, uv.max(0).values.numpy() + margin
    xy = ev.xy()
    return (xy[:, 0] >= lo[0]) & (xy[:, 0] <= hi[0]) & (xy[:, 1] >= lo[1]) & (xy[:, 1] <= hi[1])


class FbwSample:
    """Everything one endpoint needs: GT trajectory, capped events, GT subtree / channel membership, boxes."""

    def __init__(self, rc: RecCtx, end: int, seed: int, Ts: Sequence[int] = T_GRID):
        rec = rc.rec
        self.rc, self.end = rc, end
        self.t_lo = end - T_MAX
        self.t_ref = end + 1.0
        self.ts = np.arange(self.t_lo, end + 1, I.FK_STEP)
        self.X_gt = rec.pose(self.ts.astype(np.int64))
        self.traj = I.build_traj(rec.hand, self.X_gt, float(self.t_lo))
        ev = rec.events_in(end - T_MAX + 1, end)
        self.n_raw = len(ev)
        keep = np.nonzero(_crop_events(self.traj, ev))[0]
        self.n_crop = len(keep)
        keep = keep[I.thin_index(len(keep), N_CAP, np.random.default_rng(seed))]
        self.ev = ev.take(keep)
        self.xy = self.ev.xy()
        self.t_ms = self.ev.t_ms
        self.asg = I.assign(self.traj, self.xy, self.t_ms)
        isc, nrm, _ = I.contour_band(self.traj, self.xy, self.t_ms)
        self.isc, self.nrm = isc & self.asg.ok, nrm
        self.tmask = {T: self.t_ms >= (end + 1 - T) for T in sorted(set(Ts) | set(T_GRID))}
        self.member = {f: self.asg.ok & (self.asg.group == GID[f]) for f in FINGERS5}
        self.member["long4"] = np.any([self.member[f] for f in LONG4], axis=0)
        self.member["all5"] = np.any([self.member[f] for f in FINGERS5], axis=0)
        d_gt = I.displacement(self.traj, self.asg, self.t_ms, self.t_ref)
        self.d_gt = d_gt
        self.box = {}
        for f in list(FINGERS5) + ["long4"]:
            sel = self.member[f]
            if sel.sum() >= 3:
                self.box[f] = I.box_from_points(np.concatenate([self.xy[sel], self.xy[sel] + d_gt[sel]]), 10.0)
        self.subs = list(FINGERS5) + ["long4"]

    def chan_mask(self, f: str, ch: int, T: int) -> np.ndarray:
        m = self.member[f] & self.tmask[T]
        if ch == 1:
            return m & ~self.isc
        if ch == 2:
            return m & self.isc
        return m

    def contrasts(self, xw: np.ndarray, xw_n: np.ndarray, only: Optional[Sequence[str]] = None) -> np.ndarray:
        """(n_sub, nT, 3) variance contrast of every subtree cell for warped positions xw (full) / xw_n (contour channel:
        normal component only)."""
        out = np.full((len(self.subs), len(T_GRID), 3), np.nan, np.float32)
        for i, f in enumerate(self.subs):
            if f not in self.box or (only is not None and f not in only):
                continue
            for j, T in enumerate(T_GRID):
                for ch in range(3):
                    m = self.chan_mask(f, ch, T)
                    if m.sum() >= FBW_MIN_EVENTS:
                        out[i, j, ch] = I.contrast_stats((xw_n if ch == 2 else xw)[m], self.box[f])["var"]
        return out

    def counts(self) -> np.ndarray:
        out = np.zeros((len(self.subs), len(T_GRID), 3), np.int32)
        for i, f in enumerate(self.subs):
            for j, T in enumerate(T_GRID):
                for ch in range(3):
                    out[i, j, ch] = int(self.chan_mask(f, ch, T).sum())
        return out

    def eval_traj(self, traj_h: "I.Traj", subtrees: Sequence[str], t_override: Optional[np.ndarray] = None):
        """Warp every event of the union of `subtrees` with the hypothesised trajectory: assignment on the hypothesis at
        the events' own (or overridden) times, displacement to the reference, contrasts of those subtrees."""
        sel = np.any([self.member[f] for f in subtrees], axis=0)
        idx = np.nonzero(sel)[0]
        t_use = self.t_ms if t_override is None else t_override
        asg = I.assign(traj_h, self.xy[idx], t_use[idx])
        d = np.zeros_like(self.xy)
        d[idx] = I.displacement(traj_h, asg, t_use[idx], self.t_ref)
        xw = (self.xy + d).astype(np.float32)
        d_n = d.copy()
        c = self.isc
        d_n[c] = self.nrm[c] * (self.nrm[c] * d[c]).sum(-1, keepdims=True)
        xw_n = (self.xy + d_n).astype(np.float32)
        return self.contrasts(xw, xw_n, only=list(subtrees))


def pert_traj(sm: FbwSample, spec: Optional["PR.Spec"] = None, noise: Optional[np.ndarray] = None) -> "I.Traj":
    """Persistent-offset trajectory: the same anatomical spec (or additive 51-d noise) at every sample of the window."""
    rc = sm.rc
    if spec is not None:
        Xp = PR.apply(rc.rec.hand.ctx, rc.fr, spec, torch.as_tensor(sm.X_gt), rc.rec.hand.betas).numpy()
    else:
        Xp = sm.X_gt + noise[None]
    return I.build_traj(rc.rec.hand, Xp, float(sm.t_lo))


def fbw_one(seq: str, end: int, deltas=DELTAS, n_pn: int = PN_SEEDS, full: bool = True) -> dict:
    """All FBW readings of one endpoint (raw arrays; aggregation in `fbw_aggregate`)."""
    rc = RecCtx.get(seq)
    seed = I.seed_of(seq, end, "fbw")
    sm = FbwSample(rc, end, seed)
    out = {"seq": seq, "end": int(end), "n_raw": sm.n_raw, "n_crop": sm.n_crop, "n_cap": len(sm.ev),
           "subs": sm.subs, "counts": sm.counts()}
    xy_id = sm.xy
    out["C0"] = sm.contrasts(xy_id, xy_id)
    out["Cgt"] = sm.eval_traj(sm.traj, sm.subs)
    x_end = torch.as_tensor(sm.X_gt[-1])
    # --- anatomical hypotheses (persistent offsets)
    specs = make_specs(rc, x_end, deltas=deltas)
    hyp, C = [], []
    thin_sel = {}
    rng = np.random.default_rng(seed + 1)
    for f in FINGERS5:                                       # equal-event control sets: T = 400 thinned to the T = 100 count
        n100 = int(sm.chan_mask(f, 0, 100).sum())
        pool = np.nonzero(sm.chan_mask(f, 0, 400))[0]
        thin_sel[f] = pool[I.thin_index(len(pool), n100, rng)] if n100 >= FBW_MIN_EVENTS and len(pool) > n100 else None
    Cthin = []
    for key, sp in specs.items():
        fam, f, sign, d = key
        subs = LONG4 if f == "long4" else (f,)
        tr = pert_traj(sm, sp)
        sel = np.any([sm.member[s] for s in subs], axis=0)
        idx = np.nonzero(sel)[0]
        asg = I.assign(tr, sm.xy[idx], sm.t_ms[idx])
        dd = np.zeros_like(sm.xy)
        dd[idx] = I.displacement(tr, asg, sm.t_ms[idx], sm.t_ref)
        xw = (sm.xy + dd).astype(np.float32)
        dn = dd.copy()
        c = sm.isc
        dn[c] = sm.nrm[c] * (sm.nrm[c] * dd[c]).sum(-1, keepdims=True)
        xw_n = (sm.xy + dn).astype(np.float32)
        hyp.append(key)
        C.append(sm.contrasts(xw, xw_n, only=list(subs) + (["long4"] if f == "long4" else [])))
        if f in thin_sel and thin_sel[f] is not None and d in (5, 10):
            Cthin.append((key, float(I.contrast_stats(xw[thin_sel[f]], sm.box[f])["var"])))
    out["hyp"] = hyp
    out["C"] = np.stack(C) if C else np.zeros((0,) + out["C0"].shape, np.float32)
    # equal-event control: GT and no-warp contrasts on the same thinned sets
    out["thin"] = {}
    for f in FINGERS5:
        if thin_sel.get(f) is not None:
            out["thin"][f] = {"gt": float(I.contrast_stats((sm.xy + sm.d_gt)[thin_sel[f]], sm.box[f])["var"]),
                              "id": float(I.contrast_stats(sm.xy[thin_sel[f]], sm.box[f])["var"]),
                              "hyp": [(k, v) for (k, v) in Cthin if k[1] == f]}
    # --- PREV_NOISE classes (persistent), all five finger subtrees; node errors for the closed-loop-like error scale
    pn_meta, Cpn, pn_err = [], [], []
    const = s39_noise_constants()
    for cls in PN_CLASSES:
        for r in range(n_pn):
            nz = prev_noise(np.random.default_rng(I.seed_of(seq, end, "pn", cls, r)), cls, const)
            tr = pert_traj(sm, noise=nz)
            Cpn.append(sm.eval_traj(tr, FINGERS5 + ("long4",)))
            pn_meta.append((cls, r))
            pn_err.append(finger_node_error(rc, torch.as_tensor(sm.X_gt[-1] + nz), x_end))
    out["pn_meta"], out["Cpn"], out["pn_err"] = pn_meta, np.stack(Cpn), pn_err
    # --- time null (cluster-preserving permutation of the time labels used by the warp), GT trajectory
    if full:
        Cn = []
        for dr in range(NULL_DRAWS):
            tsh = I.shuffle_times(sm.ev.t_us, "cluster", np.random.default_rng(I.seed_of(seq, end, "null", dr))) / 1000.0
            Cn.append(sm.eval_traj(sm.traj, sm.subs, t_override=tsh))
        out["Cgt_null"] = np.mean(np.stack(Cn), axis=0)
    return out


def fbw_job(job):
    seq, ends, kw = job
    torch.set_num_threads(1)
    res = []
    for e in ends:
        try:
            res.append(fbw_one(seq, int(e), **kw))
        except Exception as ex:                                  # a failed endpoint is recorded, never silently dropped
            res.append({"seq": seq, "end": int(e), "error": f"{type(ex).__name__}: {ex}"})
    return res


# ------------------------------------------------------------------------------------------------ FBW aggregation
def sba_pref_reference() -> dict:
    """SBA E1 preference of the 5 mm single-finger-flexion cell (F5) on the training-subject check packets, per domain
    (n-informative-weighted fit + select); ledger anchor `#sba-r3-results-20261022` (outputs/sba/r3/e1/result_check.json)."""
    r = json.loads(E1_CHECK.read_text())["domains"]
    out = {}
    for dom in ("local", "global"):
        num = den = 0.0
        for who in ("fit", "sel"):
            d = r[f"check_{who}_{dom}"]
            n = d["n_informative"]
            num += n * d["P"]["F5"]["pref"]
            den += n
        out[dom] = num / den
    return out


def _pref(cg: np.ndarray, ch: np.ndarray) -> np.ndarray:
    """Preference of the GT contrast over hypothesis contrasts, ties 0.5; nan where undefined. cg (n,), ch (n, h)."""
    cg = cg[:, None]
    v = np.where(cg > ch, 1.0, np.where(cg == ch, 0.5, 0.0))
    return np.where(np.isfinite(cg) & np.isfinite(ch), v, np.nan)


def interp_basin(gain: np.ndarray, deltas=DELTAS, edge=HALF_GAIN) -> np.ndarray:
    """First crossing of `edge` of the focus-gain curves gain (..., len(deltas)) along deltas (gain(0) = 1), linear
    interpolation; right-censored at the largest delta; 0 if the first delta is already below the edge; nan if undefined."""
    g = np.concatenate([np.ones(gain.shape[:-1] + (1,)), gain], -1)
    x = np.concatenate([[0.0], np.asarray(deltas, np.float64)])
    out = np.full(gain.shape[:-1], np.nan)
    flat = g.reshape(-1, g.shape[-1])
    res = np.full(len(flat), np.nan)
    for i, row in enumerate(flat):
        if not np.all(np.isfinite(row)):
            continue
        below = np.nonzero(row < edge)[0]
        if len(below) == 0:
            res[i] = x[-1]
        else:
            k = below[0]
            res[i] = x[k - 1] + (row[k - 1] - edge) / max(row[k - 1] - row[k], 1e-12) * (x[k] - x[k - 1])
    return res.reshape(gain.shape[:-1])


def fbw_aggregate(recs: List[dict], ch: int = 1) -> dict:
    """Per domain x subtree x window: preference of the GT contrast over the F5 / F10 hypotheses, focus-gain curve and basin
    half-width (bootstrap CI), closed-loop-like error scale, T*_s; channel `ch` (default texture)."""
    ok = [r for r in recs if "error" not in r]
    for r in ok:                                              # hypothesis columns by (subtree, delta)
        cols = defaultdict(list)
        for q, h in enumerate(r["hyp"]):
            cols[(h[1], h[3])].append(q)
        r["_cols"] = cols
    res = {"n_ok": len(ok), "n_failed": len(recs) - len(ok), "channel": CH[ch], "domains": {}}
    sba = sba_pref_reference()
    res["sba_pref_F5"] = sba
    for dom in ("local", "global"):
        rs = [r for r in ok if domain_of(r["seq"]) == dom]
        if not rs:
            continue
        gb = GroupBoot(group_keys([r["seq"] for r in rs], [r["end"] for r in rs]))
        subs = rs[0]["subs"]
        dres = {"n_samples": len(rs), "subtrees": {}, "n_events_median": {}}
        Cgt = np.stack([r["Cgt"] for r in rs])                     # (n, S, T, 3)
        C0 = np.stack([r["C0"] for r in rs])
        cnt = np.stack([r["counts"] for r in rs])
        for i, f in enumerate(subs):
            fd = {}
            for j, T in enumerate(T_GRID):
                cg, c0 = Cgt[:, i, j, ch], C0[:, i, j, ch]
                cell = {"n_events_median": float(np.median(cnt[:, i, j, ch])),
                        "n_valid": int(np.isfinite(cg).sum())}
                # preference + normalised contrast ratios per (family, delta); hypotheses pooled over family and sign
                for dname, dl in (("F5", CELL_F5), ("F10", CELL_F10)):
                    pv = np.full((len(rs),), np.nan)
                    for n_, r in enumerate(rs):
                        cols = [q for dd in dl for q in r["_cols"].get((f, dd), [])]
                        if cols:
                            ph = _pref(np.array([cg[n_]]), r["C"][cols][:, i, j, ch][None])[0]
                            pv[n_] = np.nanmean(ph) if np.isfinite(ph).any() else np.nan
                    pt, bt = gb.wmean(pv)
                    cell[f"pref_{dname}"] = float(pt[0])
                    cell[f"pref_{dname}_ci"] = ci(bt[:, 0])
                # focus-gain curve over deltas: per-sample ratios to C_gt, pooled over family / sign
                R = np.full((len(rs), len(DELTAS)), np.nan)
                for n_, r in enumerate(rs):
                    for k, dl in enumerate(DELTAS):
                        cols = r["_cols"].get((f, dl), [])
                        if cols and np.isfinite(cg[n_]) and cg[n_] > 0:
                            R[n_, k] = np.nanmean(r["C"][cols][:, i, j, ch]) / cg[n_]
                r0 = np.where(np.isfinite(cg) & (cg > 0), c0 / cg, np.nan)
                pt_R, bt_R = gb.wmean(R)
                pt_0, bt_0 = gb.wmean(r0)
                gain_pt = (pt_R - pt_0[0]) / max(1.0 - pt_0[0], 1e-9)
                gain_bt = (bt_R - bt_0) / np.maximum(1.0 - bt_0, 1e-9)
                w_pt = float(interp_basin(gain_pt[None])[0]) if pt_0[0] < 1.0 else 0.0
                w_bt = interp_basin(gain_bt)
                w_bt = np.where(bt_0[:, 0] < 1.0, w_bt, 0.0)
                cell.update({"r0_mean": float(pt_0[0]), "gain_curve": gain_pt.tolist(), "w_mm": w_pt,
                             "w_ci": ci(w_bt), "w_censored_at_max": bool(np.isfinite(w_pt) and w_pt >= DELTAS[-1] - 1e-9)})
                # no-warp vs GT preference (positive control of the focus statistic)
                pg = np.where(np.isfinite(cg) & np.isfinite(c0), np.where(cg > c0, 1.0, np.where(cg == c0, 0.5, 0.0)), np.nan)
                p_pt, p_bt = gb.wmean(pg)
                cell["pref_gt_over_nowarp"] = float(p_pt[0])
                cell["pref_gt_over_nowarp_ci"] = ci(p_bt[:, 0])
                # time-null advantage: mean log(C_gt / C_0), real versus cluster-permuted warp times
                if "Cgt_null" in rs[0]:
                    cn = np.stack([r["Cgt_null"] for r in rs])[:, i, j, ch]
                    adv_real = np.where(np.isfinite(cg) & np.isfinite(c0) & (cg > 0) & (c0 > 0), np.log(cg / c0), np.nan)
                    adv_null = np.where(np.isfinite(cn) & np.isfinite(c0) & (cn > 0) & (c0 > 0), np.log(cn / c0), np.nan)
                    ok2 = np.isfinite(adv_real) & np.isfinite(adv_null)
                    diff = np.where(ok2, adv_real - adv_null, np.nan)
                    d_pt, d_bt = gb.wmean(diff)
                    cell["adv_real_minus_null"] = float(d_pt[0])
                    cell["adv_real_minus_null_ci"] = ci(d_bt[:, 0])
                fd[str(T)] = cell
            dres["subtrees"][f] = fd
        # closed-loop-like error scale per finger: p95 of the wrist-aligned finger node error under PREV_NOISE classes
        err = {}
        for cls in PN_CLASSES:
            for f in FINGERS5:
                v = [e[f] for r in rs for (m, e) in zip(r["pn_meta"], r["pn_err"]) if m[0] == cls]
                err[f"{cls}|{f}"] = {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                                     "n": len(v)}
        dres["pn_error_mm"] = err
        # focus survival under PREV_NOISE: fraction of draws whose contrast stays above half the GT gain
        pn_surv = {}
        Cpn = [r["Cpn"] for r in rs]                              # list of (n_pn, S, T, 3)
        for cls in PN_CLASSES:
            for i, f in enumerate(FINGERS5):
                for j, T in enumerate(T_GRID):
                    vals = []
                    for r, cp in zip(rs, Cpn):
                        sel = [q for q, m in enumerate(r["pn_meta"]) if m[0] == cls]
                        cg, c0 = r["Cgt"][i, j, ch], r["C0"][i, j, ch]
                        if np.isfinite(cg) and np.isfinite(c0) and cg > c0:
                            g = (cp[sel][:, i, j, ch] - c0) / (cg - c0)
                            vals += list(g[np.isfinite(g)] >= HALF_GAIN)
                    pn_surv[f"{cls}|{f}|{T}"] = float(np.mean(vals)) if vals else float("nan")
        dres["pn_focus_survival"] = pn_surv
        # equal-event control: T=400 events thinned to the T=100 count (F5 / F10 preference, all channels pooled)
        th = {}
        for f in FINGERS5:
            for dname, dl in (("F5", 5), ("F10", 10)):
                pv = []
                for r in rs:
                    t = r["thin"].get(f)
                    if not t:
                        continue
                    hv = [v for (k, v) in t["hyp"] if k[3] == dl]
                    if hv:
                        pv.append(float(np.mean([1.0 if t["gt"] > v else (0.5 if t["gt"] == v else 0.0) for v in hv])))
                th[f"{f}|{dname}"] = {"pref_thin400to100": float(np.mean(pv)) if pv else float("nan"), "n": len(pv)}
        dres["equal_event"] = th
        res["domains"][dom] = dres
    return res


def fbw_tstar(res: dict, edge_source: str = "small", scale: float = 1.0) -> dict:
    """T*_s = max{T : basin half-width w_s(T) >= p95 closed-loop-like warp error of subtree s}; `scale` multiplies the error
    (descriptive 'zgz-like' scale 12.53 / 4.48)."""
    out = {}
    for dom, d in res["domains"].items():
        o = {}
        for f in FINGERS5:
            e95 = d["pn_error_mm"][f"{edge_source}|{f}"]["p95"] * scale
            ok = [T for T in T_GRID if np.isfinite(d["subtrees"][f][str(T)]["w_mm"]) and d["subtrees"][f][str(T)]["w_mm"] >= e95]
            o[f] = {"e95_mm": e95, "T_star": max(ok) if ok else 0}
        out[dom] = o
    return out


# ================================================================================================ decide (FBW)
def decide_fbw(result: dict) -> dict:
    """Gate of prereg_m3.md section 1, per domain, on the texture channel. A finger subtree is *powered* if its T = 200 cell has
    >= FBW_MIN_EVENTS events (median) and >= 20 valid samples; fewer than 3 powered subtrees -> descriptive (underpowered).
    pass iff T*_s >= 200 ms for >= 3 powered subtrees and the contrast preference (F5, T = 200) pooled over those subtrees is
    >= 0.8 x the SBA F5 preference of the domain; kill iff >= 3 powered subtrees have T*_s < 100 ms or the pooled preference is
    < 0.5 x SBA. Tool validity (all-channel aggregate, T = 200, pooled over the fingers): the GT warp beats no warp (median CI lower
    bound of the preference > 0.5) and beats the cluster time-null (median CI lower bound of the paired log-advantage > 0).
    Overall: both domains pass -> pass; both kill -> kill; a tool defect in any domain -> tool_defect; else descriptive."""
    agg = result["aggregate"]
    val = result.get("aggregate_all", agg)
    ts = result["t_star"]
    sba = agg["sba_pref_F5"]
    per = {}
    fail_frac = agg.get("n_failed", 0) / max(1, agg.get("n_ok", 1) + agg.get("n_failed", 0))
    for dom, d in agg["domains"].items():
        r = {"failed_fraction": fail_frac}
        subs = d["subtrees"]
        vsubs = val["domains"][dom]["subtrees"]
        vc = [vsubs[f]["200"] for f in FINGERS5]
        lo_gain = [c["pref_gt_over_nowarp_ci"][0] for c in vc if np.isfinite(c["pref_gt_over_nowarp"])]
        adv = [c.get("adv_real_minus_null_ci", [np.nan, np.nan])[0] for c in vc]
        r["valid_gt_beats_nowarp"] = bool(lo_gain and np.nanmedian(lo_gain) > 0.5)
        r["valid_beats_time_null"] = bool(np.isfinite(adv).any() and np.nanmedian(adv) > 0.0)
        powered = [f for f in FINGERS5 if subs[f]["200"]["n_events_median"] >= FBW_MIN_EVENTS and subs[f]["200"]["n_valid"] >= 20]
        r["powered_subtrees"] = powered
        n_pass = sum(1 for f in powered if ts[dom][f]["T_star"] >= T_STAR_PASS)
        n_kill = sum(1 for f in powered if ts[dom][f]["T_star"] < T_STAR_KILL)
        good = [f for f in powered if ts[dom][f]["T_star"] >= T_STAR_PASS]
        pf = [subs[f]["200"]["pref_F5"] for f in (good if good else powered)]
        pref = float(np.nanmean(pf)) if pf else float("nan")
        r.update({"n_subtrees_Tstar_ge_200": n_pass, "n_subtrees_Tstar_lt_100": n_kill, "pref_F5_T200": pref,
                  "pref_ratio_to_SBA": pref / sba[dom] if np.isfinite(pref) else float("nan"), "sba_pref_F5": sba[dom]})
        if not (r["valid_gt_beats_nowarp"] and r["valid_beats_time_null"]) or fail_frac > MAX_FAIL_FRAC:
            r["verdict"] = "tool_defect"
        elif len(powered) < N_SUB_PASS:
            r["verdict"] = "descriptive"
            r["note"] = "underpowered: fewer than 3 finger subtrees with enough texture events"
        elif n_pass >= N_SUB_PASS and r["pref_ratio_to_SBA"] >= PREF_FRAC_PASS:
            r["verdict"] = "pass"
        elif n_kill >= N_SUB_PASS or r["pref_ratio_to_SBA"] < PREF_FRAC_KILL:
            r["verdict"] = "kill"
        else:
            r["verdict"] = "descriptive"
        per[dom] = r
    v = {d: per[d]["verdict"] for d in per}
    if any(x == "tool_defect" for x in v.values()):
        overall = "tool_defect"
    elif v and all(x == "pass" for x in v.values()):
        overall = "pass"
    elif v and all(x == "kill" for x in v.values()):
        overall = "kill"
    else:
        overall = "descriptive"
    return {"probe": "FBW", "verdict": overall, "per_domain": per}


# ================================================================================================ SMG
SMG_WINDOW = 200                                         # ms of events of the surface map
SMG_N_CAP = 20000
SMG_EVERY_FIT = 8
SMG_EVERY_SEL = 8
SMG_T_MM = (5, 10, 14, 19)                               # image-plane shifts of the SELECT readout (mm)
SMG_R_DEG = (5, 11)                                      # root rotations (deg)
SMG_N_DIR = 2                                            # directions per size
SMG_FD = (("tx", 3e-3), ("tx", 8e-3), ("ty", 3e-3), ("ty", 8e-3), ("tz", 8e-3),
          ("rx", math.radians(2.5)), ("ry", math.radians(2.5)), ("rz", math.radians(2.5)))     # (DoF, step) of the contrast probe
DOF = ("tx", "ty", "tz", "rx", "ry", "rz")
SMG_SETS = ("tex_hyp", "con_hyp", "all_hyp", "tex_orc", "tex_nowarp", "tex_signed", "tex_signed_pol")
SMG_B_PASS, SMG_B_KILL = 0.30, 0.10
SMG_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
SMG_AUG = (0.2, 0.4, 0.4)                                # clean / KS4 random_spec / PREV_NOISE mixture


def _so3(w: np.ndarray):
    from semkine.lie import so3_exp, so3_log
    return so3_exp, so3_log


def apply_root_offset(X: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Anchor displacement of poses X (n, 51): off = (tx, ty, tz in m, rotation vector in rad right-multiplied on the root
    axis-angle, as tools/kstar/head_probe.py::perturb)."""
    so3_exp, so3_log = _so3(None)
    Xp = X.copy()
    Xp[:, 0:3] += off[None, 0:3].astype(np.float32)
    if np.linalg.norm(off[3:6]) > 0:
        R = so3_exp(torch.as_tensor(X[:, 3:6])) @ so3_exp(torch.as_tensor(off[3:6].astype(np.float32))[None].expand(len(X), 3))
        Xp[:, 3:6] = so3_log(R).numpy()
    return Xp


def offset_target(off_applied: np.ndarray) -> np.ndarray:
    """Correction that returns the anchor to the GT: minus the applied translation, minus the applied rotation vector."""
    return -off_applied


def fd_offsets() -> List[np.ndarray]:
    """Finite-difference displacement vectors (6,) of SMG_FD, + and - (in the order [+, -] per entry)."""
    out = []
    for name, step in SMG_FD:
        v = np.zeros(6, np.float64)
        v[DOF.index(name)] = step
        out += [v, -v]
    return out


class SmgSample:
    """One endpoint of the SMG probe: W200 events (capped), GT trajectory, GT-flag sets for the oracle control."""

    def __init__(self, rc: RecCtx, end: int, seed: int, ev_override: Optional["I.Events"] = None, n_cap: int = SMG_N_CAP,
                 equal_event: bool = False):
        rec = rc.rec
        self.rc, self.end = rc, end
        self.t_lo = end - SMG_WINDOW
        self.t_ref = end + 1.0
        self.ts = np.arange(self.t_lo, end + 1, I.FK_STEP)
        self.X_gt = rec.pose(self.ts.astype(np.int64))
        self.traj = I.build_traj(rec.hand, self.X_gt, float(self.t_lo))
        ev = rec.events_in(end - SMG_WINDOW + 1, end) if ev_override is None else ev_override
        keep = np.nonzero(_crop_events(self.traj, ev, margin=26.0))[0]
        self.n_crop = len(keep)
        keep = keep[I.thin_index(len(keep), n_cap, np.random.default_rng(seed))]
        if equal_event:                                   # equal-event control: the 200 ms events thinned to the count of the last 50 ms
            tt = ev.t_ms[keep]
            n50 = int((tt >= end + 1 - 50).sum())
            keep = keep[I.thin_index(len(keep), max(n50, 1), np.random.default_rng(seed + 3))]
        self.ev = ev.take(keep)
        self.xy, self.t_ms = self.ev.xy(), self.ev.t_ms
        self.w_pol = np.where(self.ev.p == 1, 1.0, -1.0)
        self.w_pol_swap = np.where(I.swap_polarity(self.ev, "pixel_half", np.random.default_rng(seed + 9)).p == 1, 1.0, -1.0)
        uv = self.traj.uv[self.traj.vis]
        self.box = I.box_from_points(np.concatenate([uv.numpy(), self.xy]), 12.0) if len(self.xy) else None
        self.asg_gt = I.assign(self.traj, self.xy, self.t_ms)
        isc, _, _ = I.contour_band(self.traj, self.xy, self.t_ms)
        self.tex_orc = self.asg_gt.ok & ~isc

    def stats_for(self, X: np.ndarray, t_override: Optional[np.ndarray] = None) -> Dict[str, float]:
        """log pair-concentration of the four event sets under the (hypothesised) pose sequence X."""
        rc = self.rc
        tr = I.build_traj(rc.rec.hand, X, float(self.t_lo))
        tms = self.t_ms if t_override is None else t_override
        asg = I.assign(tr, self.xy, tms)
        isc, nrm, _ = I.contour_band(tr, self.xy, tms)
        d = np.zeros_like(self.xy)
        idx = np.nonzero(asg.ok)[0]
        d[idx] = I.displacement(tr, I.Assign(asg.face[idx], asg.bary[idx], asg.resid[idx], asg.vert[idx], asg.group[idx],
                                             asg.ok[idx], asg.front[idx], asg.dist[idx]), tms[idx], self.t_ref)
        xw = (self.xy + d).astype(np.float32)
        d_n = d.copy()
        c = isc & asg.ok
        d_n[c] = nrm[c] * (nrm[c] * d[c]).sum(-1, keepdims=True)
        xw_n = (self.xy + d_n).astype(np.float32)
        m_tex = asg.ok & ~isc
        sets = {"tex_hyp": m_tex, "con_hyp": asg.ok & isc, "all_hyp": asg.ok, "tex_orc": self.tex_orc, "tex_nowarp": m_tex,
                "tex_signed": m_tex, "tex_signed_pol": m_tex}
        out = {}
        for name, m in sets.items():
            if m.sum() >= 30:
                pos = self.xy if name == "tex_nowarp" else (xw_n if name == "con_hyp" else xw)
                wts = self.w_pol[m] if name == "tex_signed" else (self.w_pol_swap[m] if name == "tex_signed_pol" else None)
                st = I.contrast_stats(pos[m], self.box, weights=wts)
                out[name] = (math.log(max(st["conc"], 1e-9)) if np.isfinite(st["conc"]) else float("nan"), float(st["n_in"]))
            else:
                out[name] = (float("nan"), float(m.sum()))
        return out


def smg_features(sm: SmgSample, off: np.ndarray, null_draw: Optional[int] = None) -> Dict[str, np.ndarray]:
    """Feature vectors (per event set) of the anchor displaced by `off` (6,): base log-concentration, log event count, then the
    antisymmetric (gradient) and symmetric (curvature) parts of the finite differences of the log-concentration along SMG_FD."""
    t_override = None
    if null_draw is not None:
        t_override = I.shuffle_times(sm.ev.t_us, "cluster", np.random.default_rng(I.seed_of(sm.rc.seq, sm.end, "smgnull", null_draw))) / 1000.0
    Xa = apply_root_offset(sm.X_gt, off)
    base = sm.stats_for(Xa, t_override)
    fds = fd_offsets()
    vals = [sm.stats_for(apply_root_offset(Xa, v), t_override) for v in fds]
    feats = {}
    for s in SMG_SETS:
        b, n = base[s]
        f = [b, math.log(max(n, 1.0))]
        for k in range(len(SMG_FD)):
            lp, lm = vals[2 * k][s][0], vals[2 * k + 1][s][0]
            f += [lp - lm, lp + lm - 2 * b]
        feats[s] = np.asarray(f, np.float64)
    return feats


def smg_specs_fit(rng: np.random.Generator, const: dict) -> np.ndarray:
    """Anchor offset (6,) of one FIT augmentation draw: clean / KS4 random_spec (translation <= 25 mm, depth +-45 mm, root
    rotation <= 11 deg) / PREV_NOISE mixture (root block only)."""
    u = rng.random()
    off = np.zeros(6)
    if u < SMG_AUG[0]:
        return off
    if u < SMG_AUG[0] + SMG_AUG[1]:
        k = rng.integers(3)
        if k == 0:
            d, th = rng.uniform(0, 25e-3), rng.uniform(0, 2 * math.pi)
            off[0], off[1] = d * math.cos(th), d * math.sin(th)
        elif k == 1:
            off[2] = rng.uniform(-45e-3, 45e-3)
        else:
            ax = rng.standard_normal(3)
            off[3:6] = ax / np.linalg.norm(ax) * math.radians(rng.uniform(0, 11))
        return off
    cls = rng.choice(PN_CLASSES, p=np.asarray(const["mix"]) / sum(const["mix"]))     # S39 PREV_NOISE_P_SMALL / LARGE / CORR
    nz = prev_noise(rng, str(cls), const)
    off[0:3], off[3:6] = nz[0:3], nz[3:6]
    return off


def smg_specs_select(rng: np.random.Generator) -> List[Tuple[str, np.ndarray]]:
    """SELECT readout anchors: clean, image-plane shifts of SMG_T_MM in SMG_N_DIR directions, root rotations of SMG_R_DEG."""
    out = [("clean", np.zeros(6))]
    for d in SMG_T_MM:
        for k in range(SMG_N_DIR):
            th = rng.uniform(0, 2 * math.pi)
            off = np.zeros(6)
            off[0], off[1] = d * 1e-3 * math.cos(th), d * 1e-3 * math.sin(th)
            out.append((f"t{d}", off))
    for deg in SMG_R_DEG:
        for k in range(SMG_N_DIR):
            ax = rng.standard_normal(3)
            off = np.zeros(6)
            off[3:6] = ax / np.linalg.norm(ax) * math.radians(deg)
            out.append((f"r{deg}", off))
    return out


def smg_one(seq: str, end: int, role: str, synthetic: bool = False) -> dict:
    """role 'fit': a clean anchor and three augmentation draws; role 'select': the readout anchors. Features are computed for the real
    events (and, unless synthetic, for the cluster-permuted null). `synthetic`: events replaced by ideal on-surface events of the GT
    trajectory (positive control of the whole pipeline)."""
    rc = RecCtx.get(seq)
    seed = I.seed_of(seq, end, "smg")
    rng = np.random.default_rng(seed)
    const = s39_noise_constants()
    sm = SmgSample(rc, end, seed)
    if synthetic:
        n = max(200, min(len(sm.ev), SMG_N_CAP))
        xy, t_ev, _, _ = I._synthetic_events(sm.traj, n, np.random.default_rng(seed + 5))
        sm = SmgSample(rc, end, seed, ev_override=_ContinuousEvents(xy, (t_ev * 1000).astype(np.int64)))
    out = {"seq": seq, "end": int(end), "role": role, "synthetic": synthetic, "n_ev": len(sm.ev), "anchors": []}
    sm_eq = None
    if role == "fit":
        specs = [("clean", np.zeros(6))] + [("aug", smg_specs_fit(rng, const)) for _ in range(3)]
    else:
        specs = smg_specs_select(rng)
    for name, off in specs:
        rec_ = {"name": name, "off": off.tolist(), "target": offset_target(off).tolist()}
        f = smg_features(sm, off)
        rec_["feat"] = {s: f[s].tolist() for s in SMG_SETS}
        if not synthetic:
            fn = smg_features(sm, off, null_draw=0)
            rec_["feat_null"] = {s: fn[s].tolist() for s in SMG_SETS}
        if role == "select" and not synthetic and name in ("clean", "t14"):
            if sm_eq is None:
                sm_eq = SmgSample(rc, end, seed, equal_event=True)
            fe = smg_features(sm_eq, off)
            rec_["feat_eq"] = {s: fe[s].tolist() for s in ("tex_hyp",)}
        out["anchors"].append(rec_)
    return out


class _ContinuousEvents(I.Events):
    """Synthetic events with continuous positions (positive control only): `xy()` returns the stored positions."""

    def __init__(self, xy: np.ndarray, t_us: np.ndarray):
        n = len(xy)
        super().__init__(np.zeros(n, np.int16), np.zeros(n, np.int16), np.zeros(n, np.uint8), np.asarray(t_us, np.int64))
        self._xy = np.asarray(xy, np.float32)

    def xy(self) -> np.ndarray:
        return self._xy

    def take(self, idx) -> "_ContinuousEvents":
        return _ContinuousEvents(self._xy[idx], self.t_us[idx])

    def with_times(self, t_us) -> "_ContinuousEvents":
        return _ContinuousEvents(self._xy, t_us)


def smg_job(job):
    seq, ends, role, kw = job
    torch.set_num_threads(1)
    out = []
    for e in ends:
        try:
            out.append(smg_one(seq, int(e), role, **kw))
        except Exception as ex:
            out.append({"seq": seq, "end": int(e), "role": role, "error": f"{type(ex).__name__}: {ex}"})
    return out


# ------------------------------------------------------------------------------------------------ SMG readout / statistics
SCALE6 = np.array([1e3, 1e3, 1e3, 1.0, 1.0, 1.0])        # mm / rad: comparable target scales for the ridge


def ridge_primal(X: np.ndarray, Y: np.ndarray, lam: float):
    """Standardised-feature ridge with an unpenalised intercept: returns (mu, sd, W (d+1, k))."""
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1)
    A = Z.T @ Z + lam * len(X) * np.diag(np.r_[np.ones(X.shape[1]), 1e-8])
    return mu, sd, np.linalg.solve(A, Z.T @ Y)


def ridge_apply(model, X: np.ndarray) -> np.ndarray:
    mu, sd, W = model
    return np.concatenate([(X - mu) / sd, np.ones((len(X), 1))], 1) @ W


def _rows(recs: List[dict], s: str, with_null: bool = True, null_from: Optional[str] = None, key: str = "feat"):
    """(X (n, d), X_null or None, Y (n, 6), cells (n,), subjects (n,), blocks (n,)) of every anchor, rows finite in both worlds.
    `null_from`: take the null world from another feature set of the same real run (the polarity-swap world of the signed IWE);
    `key='feat_eq'` reads the equal-event control features (only the anchors that carry them)."""
    X, Xn, Y, cells, subj, blk = [], [], [], [], [], []
    for r in recs:
        for a in r["anchors"]:
            if key not in a:
                continue
            f = np.asarray(a[key][s], np.float64)
            if null_from is not None:
                fn = np.asarray(a["feat"][null_from], np.float64)
            else:
                fn = np.asarray(a["feat_null"][s], np.float64) if (with_null and "feat_null" in a) else None
            if not np.all(np.isfinite(f)) or (fn is not None and not np.all(np.isfinite(fn))):
                continue
            X.append(f)
            if fn is not None:
                Xn.append(fn)
            Y.append(np.asarray(a["target"], np.float64))
            cells.append(a["name"])
            subj.append(r["seq"].split("_")[0])
            blk.append(f"{r['seq']}|{int(r['end']) // BLOCK_MS}")
    return (np.stack(X) if X else np.zeros((0, 18)), np.stack(Xn) if Xn else None, np.stack(Y) if Y else np.zeros((0, 6)),
            np.asarray(cells), np.asarray(subj), np.asarray(blk))


def fit_readout(X: np.ndarray, Y: np.ndarray, subj: np.ndarray) -> dict:
    """Ridge readout of the six root-offset corrections; lambda by leave-subjects-out folds (four) inside the fit set, then refit."""
    Ys = Y * SCALE6
    us = sorted(set(subj))
    fold = np.asarray([us.index(s) % 4 for s in subj])
    best = None
    for lam in SMG_LAMBDAS:
        err, used = 0.0, 0
        for f in range(4):
            tr, te = fold != f, fold == f
            if tr.sum() < 10 or te.sum() < 1:
                continue
            err += float(((ridge_apply(ridge_primal(X[tr], Ys[tr], lam), X[te]) - Ys[te]) ** 2).sum())
            used += 1
        if used and (best is None or err < best[1]):
            best = (lam, err)
    lam = best[0] if best else SMG_LAMBDAS[2]
    return {"model": ridge_primal(X, Ys, lam), "lambda": lam, "n_fit": int(len(X))}


def _hit(tg: np.ndarray, pr: np.ndarray) -> np.ndarray:
    dot = (tg * pr).sum(1)
    return np.where(dot > 0, 1.0, np.where(dot == 0, 0.5, 0.0))


def _slope_moments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.stack([np.ones_like(x), x, y, x * x, x * y], 1)


def _slope_of(m: np.ndarray) -> float:
    n, sx, sy, sxx, sxy = m
    den = sxx - sx * sx / max(n, 1e-12)
    return float((sxy - sx * sy / n) / den) if den > 0 else float("nan")


def read_cells(X: np.ndarray, Y: np.ndarray, cells: np.ndarray, blk: np.ndarray, pred_real: np.ndarray,
               pred_null: Optional[np.ndarray]) -> dict:
    """Per readout cell (t5 / t10 / t14 / t19 / r5 / r11): OLS slope b_off of the prediction on the true correction (translation: the two
    in-plane components pooled, rotation: the three components pooled), group-bootstrap CI, the paired difference to the null
    readout, the direction hit rate (CI), and the mean predicted magnitude of the clean anchors."""
    out = {}
    for cell in ("t5", "t10", "t14", "t19", "r5", "r11"):
        m = cells == cell
        if m.sum() < 5:
            continue
        comp = (0, 1) if cell.startswith("t") else (3, 4, 5)
        gb = GroupBoot(blk[m])
        per = lambda P: np.concatenate([_slope_moments(Y[m][:, c], P[m][:, c])[:, None, :] for c in comp], 1).sum(1)    # noqa: E731
        res = {"n": int(m.sum())}
        sums = {}
        for tag, P in (("real", pred_real), ("null", pred_null)):
            if P is None:
                continue
            mom = per(P)                                                            # (n_cell, 5)
            s_, _ = gb.gsum(mom)
            pt = _slope_of(s_.sum(0))
            bt = np.asarray([_slope_of(w @ s_) for w in gb.W])
            sums[tag] = (pt, bt)
            res["b_off" if tag == "real" else "b_off_null"] = pt
            res["b_off_ci" if tag == "real" else "b_off_null_ci"] = ci(bt)
        if "null" in sums:
            res["b_off_minus_null"] = sums["real"][0] - sums["null"][0]
            res["b_off_minus_null_ci"] = ci(sums["real"][1] - sums["null"][1])
            res["b_off_minus_null_ci_holm"] = ci(sums["real"][1] - sums["null"][1], (1.25, 98.75))
        h = _hit(Y[m][:, comp], pred_real[m][:, comp])
        pt, bt = gb.wmean(h)
        res["hit"], res["hit_ci"] = float(pt[0]), ci(bt[:, 0])
        if pred_null is not None:
            res["hit_null"] = float(np.mean(_hit(Y[m][:, comp], pred_null[m][:, comp])))
        out[cell] = res
    cl = cells == "clean"
    if cl.any():
        out["clean_pred_rms_mm"] = float(np.sqrt(np.mean(pred_real[cl][:, :2] ** 2)) * 1e3)
    return out


def smg_aggregate(fit_recs: List[dict], sel_recs: List[dict], fit_syn: Optional[List[dict]] = None,
                  sel_syn: Optional[List[dict]] = None) -> dict:
    """Per domain: FIT-fitted ridge readouts of the real and the cluster-null worlds read on the SELECT anchors (all four event
    sets), plus the ideal-event positive control (fit and read on synthetic on-surface events)."""
    out = {"domains": {}, "sets": list(SMG_SETS), "synthetic_control": {}}
    ok = lambda rs, dom: [r for r in rs if "error" not in r and domain_of(r["seq"]) == dom]     # noqa: E731
    for dom in ("local", "global"):
        fit, sel = ok(fit_recs, dom), ok(sel_recs, dom)
        if not fit or not sel:
            continue
        d = {"n_fit_samples": len(fit), "n_sel_samples": len(sel), "sets": {}}
        for s in SMG_SETS:
            if s == "tex_signed_pol":
                continue                                                # it is the null world of tex_signed
            nf = "tex_signed_pol" if s == "tex_signed" else None
            Xf, Xnf, Yf, _, sf, _ = _rows(fit, s, null_from=nf)
            Xs, Xns, Ys, cs, _, bs = _rows(sel, s, null_from=nf)
            m_real, m_null = fit_readout(Xf, Yf, sf), fit_readout(Xnf, Yf, sf)
            p_real = ridge_apply(m_real["model"], Xs) / SCALE6
            p_null = ridge_apply(m_null["model"], Xns) / SCALE6
            cells = read_cells(Xs, Ys, cs, bs, p_real, p_null)
            cells.update({"lambda_real": m_real["lambda"], "lambda_null": m_null["lambda"], "n_fit_rows": m_real["n_fit"]})
            d["sets"][s] = {"cells": {k: v for k, v in cells.items() if k.startswith(("t", "r"))},
                            **{k: v for k, v in cells.items() if not k.startswith(("t", "r"))}}
            if s == "tex_hyp":                                          # equal-event control, read by the real tex_hyp readout
                Xe, _, Ye, ce, _, be = _rows(sel, s, with_null=False, key="feat_eq")
                if len(Xe):
                    pe = ridge_apply(m_real["model"], Xe) / SCALE6
                    d["sets"][s]["equal_event"] = {k: v for k, v in read_cells(Xe, Ye, ce, be, pe, None).items() if k in ("t14",)}
        out["domains"][dom] = d
        if fit_syn and sel_syn:
            fs, ss = ok(fit_syn, dom), ok(sel_syn, dom)
            if fs and ss:
                Xf, _, Yf, _, sf, _ = _rows(fs, "tex_hyp", False)
                Xs, _, Ys, cs, _, bs = _rows(ss, "tex_hyp", False)
                m = fit_readout(Xf, Yf, sf)
                cells = read_cells(Xs, Ys, cs, bs, ridge_apply(m["model"], Xs) / SCALE6, None)
                out["synthetic_control"][dom] = {"b_off_t14": cells.get("t14", {}).get("b_off"), "n": len(ss)}
    return out


# ================================================================================================ decide (SMG)
def decide_smg(result: dict) -> dict:
    """Gate of prereg_m3.md section 2, per domain on SELECT, texture channel read at the perturbed anchor (`tex_hyp`):
    pass iff b_off(14 mm) >= 0.30 and its paired difference to the cluster-null readout has CI lower bound > 0 and the direction hit
    rate has CI lower bound > 0.5; kill iff b_off(14 mm) <= 0.10, or the paired difference to the null has CI lower bound <= 0, or the
    slope exists only with oracle flags (`tex_orc` b_off(14) >= 0.30 while `tex_hyp` < 0.30); tool defect iff the ideal-event positive
    control (synthetic on-surface events, same pipeline) has b_off(14 mm) < 0.30. Overall: both domains pass -> pass; both kill ->
    kill; tool defect anywhere -> tool_defect; else descriptive."""
    agg = result["aggregate"]
    per = {}
    n_total = max(1, result.get("n_total", 1))
    fail_frac = result.get("n_failed", 0) / n_total
    for dom, d in agg["domains"].items():
        c = d["sets"]["tex_hyp"].get("cells", {}).get("t14")
        o = d["sets"]["tex_orc"].get("cells", {}).get("t14")
        if c is None:
            per[dom] = {"verdict": "descriptive", "note": "no t14 cell"}
            continue
        r = {"b_off_t14": c["b_off"], "b_off_t14_ci": c["b_off_ci"], "b_off_minus_null_ci": c["b_off_minus_null_ci"],
             "hit": c["hit"], "hit_ci": c["hit_ci"], "b_off_oracle_flags": None if o is None else o["b_off"]}
        syn = agg.get("synthetic_control", {}).get(dom, {}).get("b_off_t14")
        r["failed_fraction"] = fail_frac
        if fail_frac > MAX_FAIL_FRAC:
            r["verdict"] = "tool_defect"
            r["note"] = "too many failed endpoints"
        elif syn is None or not np.isfinite(syn) or syn < SMG_B_PASS:
            r["verdict"] = "tool_defect"
            r["note"] = "ideal-event positive control missing or below the gate level"
            r["synthetic_b_off_t14"] = syn
        elif c["b_off"] >= SMG_B_PASS and c["b_off_minus_null_ci"][0] > 0 and c["hit_ci"][0] > 0.5:
            r["verdict"] = "pass"
        elif (c["b_off"] <= SMG_B_KILL or c["b_off_minus_null_ci"][0] <= 0 or
              (o is not None and o["b_off"] >= SMG_B_PASS and c["b_off"] < SMG_B_PASS)):
            r["verdict"] = "kill"
        else:
            r["verdict"] = "descriptive"
        per[dom] = r
    v = [x["verdict"] for x in per.values()]
    overall = ("tool_defect" if "tool_defect" in v else "pass" if v and all(x == "pass" for x in v) else
               "kill" if v and all(x == "kill" for x in v) else "descriptive")
    return {"probe": "SMG", "verdict": overall, "per_domain": per}


def decide(result: dict) -> dict:
    """Dispatch on `result["probe"]` ('FBW' / 'SMG' / 'EXPO'); a dict {'FBW': ..., 'SMG': ..., 'EXPO': ...} decides each."""
    if "probe" in result:
        return {"FBW": decide_fbw, "SMG": decide_smg, "EXPO": lambda r: decide_expo(r)}[result["probe"]](result)
    return {k: decide(v) for k, v in result.items() if k in ("FBW", "SMG", "EXPO")}




# ================================================================================================ EXPO
from sba import common as SCM  # noqa: E402
from sba import r3_lik as L  # noqa: E402
from sba import r3_e8 as E8  # noqa: E402
from semkine import sba_emission as EM  # noqa: E402

EXPO_EVERY = 6
EXPO_HYPS = ["clean", "pip+5", "pip-5", "mcpf+5", "mcpf-5", "synrc+5", "synrc-5"]
EXPO_F5 = ["pip+5", "pip-5", "mcpf+5", "mcpf-5"]
EXPO_S5 = ["synrc+5", "synrc-5"]
EXPO_RHO = (1.0, 0.7, 0.5)                 # ageing: window weight rho ** (windows between it and the terminal window)
EXPO_KC = (0.0, 0.01, 0.05)                # censoring floor, fraction of the window's peak union exposure
EXPO_N_THIN = 3
EXPO_MIN_EVENTS = 100                      # terminal window events (E8 MIN_EVENTS)
EXPO_DELTA_PASS, EXPO_DELTA_KILL = 0.02, 0.01
EXPO_REG_TOL = 0.10                        # engineering regression tolerance versus the E8 K = 1 preference
LONG_ORDER = ("index", "middle", "ring", "pinky")


class SbaCache:
    _c: Dict[str, dict] = {}

    @classmethod
    def get(cls, seq: str) -> dict:
        if seq not in cls._c:
            cls._c.clear()
            g = I.geo()
            dev = torch.device("cpu")
            d = MC.seq_dir(seq)
            sd = SCM.SeqData(seq, d, dev)
            lik = L.Lik(g["ctx"])
            fr = PR.make_frame(g["ctx"], sd.betas)
            cls._c[seq] = {"sd": sd, "lik": lik, "fr": fr, "ctx": g["ctx"]}
        return cls._c[seq]


def cluster_perm_ms(ev_ms, rng: np.random.Generator):
    """Cluster-preserving permutation at 1 ms resolution (the SBA window grid): events of one ms bin keep their bin-mates and the
    occupied bins exchange their time labels."""
    ev, ms = ev_ms
    if len(ms) == 0:
        return ev, ms
    uq, inv = np.unique(ms, return_inverse=True)
    return ev, uq[rng.permutation(len(uq))][inv]


def expo_specs(rc_: dict, x1: torch.Tensor, k: int) -> Dict[str, "PR.Spec"]:
    """E0 specs of the chain's terminal knot: the finger of the F5 cell rotates through the four long fingers by chain index."""
    ctx, betas, fr = rc_["ctx"], rc_["sd"].betas, rc_["fr"]
    finger = LONG_ORDER[k % 4]
    j_clean = PR.joints(ctx, x1[None], betas)
    Bj = PR.root_jacobian(ctx, x1, betas)
    specs = {}
    for fam in ("pip", "mcpf"):
        for sign in (1, -1):
            deg, ok = PR.bisect(ctx, fr, fam, sign, 5.0, x1, betas, finger, None, None, j_clean)
            if ok:
                specs[f"{fam}{sign * 5:+d}"] = PR.Spec(fam, sign, deg, finger, None)
    for sign in (1, -1):
        deg, ok = PR.bisect(ctx, fr, "synrc", sign, 5.0, x1, betas, finger, None, Bj, j_clean)
        if ok:
            sp = PR.Spec("synrc", sign, deg, finger, None)
            _, om = PR._syn_rc(ctx, fr, x1[None], torch.tensor([sign * deg]), Bj, j_clean, betas)
            sp.omega = om[0].tolist()
            specs[f"synrc{sign * 5:+d}"] = sp
    return specs


def _expo_problem(G: torch.Tensor, win, variant: str, keep: Optional[torch.Tensor], a: float):
    """`r3_lik.problem` with a censoring mask over the window's cells and a window weight (power likelihood)."""
    N = win.N.sum(0).flatten()
    U = win.uni.flatten()
    Bk = win.bkg.flatten()
    if keep is None:
        keep = torch.ones_like(N, dtype=torch.bool)
    cell = (N > 0) & keep
    Gx = G if variant == "mg" else G.sum(1, keepdim=True)
    P = Gx.shape[0]
    X = torch.cat([Gx[:, :, cell], U[cell].expand(P, 1, -1)], 1)
    Xt = torch.cat([Gx[:, :, keep].sum(-1), U[keep].sum().expand(P, 1)], 1)
    return X, a * Xt, Bk[cell], a * float(Bk[keep].sum()), a * N[cell]


def _expo_ll(Gs: Sequence[torch.Tensor], wins: Sequence, variant: str, keeps=None, weights=None) -> np.ndarray:
    """Chained profile log-likelihood (P,) of hypotheses with nuisances shared over the windows."""
    parts = [_expo_problem(G, w, variant, None if keeps is None else keeps[i], 1.0 if weights is None else weights[i])
             for i, (G, w) in enumerate(zip(Gs, wins))]
    X = torch.cat([p[0] for p in parts], -1)
    Xt = sum(p[1] for p in parts)
    B = torch.cat([p[2] for p in parts])
    Bt = sum(p[3] for p in parts)
    N = torch.cat([p[4] for p in parts])
    return L.newton(X, Xt, B, Bt, N, nan_unconverged=True)[0].cpu().numpy()


def expo_one(seq: str, end: int, k: int, light: bool = False) -> dict:
    """All EXPO log-likelihoods of one terminal endpoint: conditions x K x emission arms x hypotheses (EXPO_HYPS)."""
    rc_ = SbaCache.get(seq)
    sd, lik, ctx = rc_["sd"], rc_["lik"], rc_["ctx"]
    geo = ctx.geo
    kms = [end - 200 + 50 * i for i in range(5)]
    Xg = torch.stack([sd.gt(m) for m in kms])
    ev_all = sd.events(kms[0], kms[-1])
    nat = E8.windows_from(ctx, sd, kms, Xg, ev_all)
    rec = {"seq": seq, "end": int(end), "k": k, "n_events": [w.n_events for w in nat], "n_crop": [w.n_crop for w in nat]}
    if nat[-1].n_events < EXPO_MIN_EVENTS:
        return rec
    specs = expo_specs(rc_, Xg[-1], k)
    hyps = [h for h in EXPO_HYPS if h == "clean" or h in specs]
    rec["hyps"] = hyps
    XS = [Xg.clone() if h == "clean" else PR.apply(ctx, rc_["fr"], specs[h], Xg, sd.betas) for h in hyps]
    # features: (windows) x (hyps) -> G (6, M); computed once, every condition re-uses them (they do not depend on events)
    Gw = []
    for j in range(4):
        Gj = []
        for hi in range(len(hyps)):
            F = EM.features(geo, nat[j], XS[hi][j], XS[hi][j + 1], sd.betas, sd.K, lik.patch, lik.n_p, lik.blur, 1.0).flatten(1)
            Gj.append(lik.W @ F)
        Gw.append(torch.stack(Gj))                                          # (P, 6, M)
    rec["G_shape"] = list(Gw[0].shape)
    # union exposure per window (hypothesis-independent censoring mask)
    Ebar = [G.sum(1).amax(0) for G in Gw]
    keeps = {kc: [(E >= kc * float(E.max())) for E in Ebar] for kc in EXPO_KC}
    T = nat[-1].n_crop
    conds = {"nat": nat}
    for K in (2, 4):
        sub = L.take(ev_all, kms[4 - K], kms[-1])
        n_k = sum(w.n_crop for w in nat[4 - K:])
        for dr in range(EXPO_N_THIN):
            conds[f"thin{K}_{dr}"] = E8.windows_from(ctx, sd, kms, Xg, L.thin(sub, T / max(n_k, 1),
                                                                             L.seed_of(sd.name, end, K, dr)))
    conds["shufc4"] = E8.windows_from(ctx, sd, kms, Xg, cluster_perm_ms(ev_all, np.random.default_rng(L.seed_of(sd.name, end, "shufc"))))
    conds["shufw4"] = E8.windows_from(ctx, sd, kms, Xg, L.shuffle_times(ev_all, L.seed_of(sd.name, end, "shufw")))
    arms = {"a": ("m1", None, None), "b": ("mg", None, None)}
    for rho in EXPO_RHO:
        for kc in EXPO_KC:
            arms[f"c_{rho}_{kc}"] = ("mg", kc, rho)
    ll = {}
    for cname, wins in conds.items():
        Ks = (1, 2, 4) if cname == "nat" else ((int(cname[4]),) if cname.startswith("thin") else (4,))
        for K in Ks:
            idx = list(range(4 - K, 4))
            cell = {}
            for an, (var, kc, rho) in arms.items():
                keeps_ = None if kc is None else [keeps[kc][j] for j in idx]
                w_ = None if rho is None else [rho ** (3 - j) for j in idx]
                cell[an] = _expo_ll([Gw[j] for j in idx], [wins[j] for j in idx], var, keeps_, w_).tolist()
            ll[f"{cname}|{K}"] = cell
    rec["ll"] = ll
    return rec


def expo_job(job):
    seq, items = job
    torch.set_num_threads(1)
    out = []
    for e, k in items:
        try:
            out.append(expo_one(seq, int(e), int(k)))
        except Exception as ex:
            out.append({"seq": seq, "end": int(e), "k": k, "error": f"{type(ex).__name__}: {ex}"})
    return out


def expo_endpoints(rec_runs, every: int, phase: int = 0) -> np.ndarray:
    """Terminal endpoints with three predecessors inside the run (k >= 4 on the 50 ms grid), every `every`-th."""
    out = []
    for ra, rb in rec_runs:
        ends = np.arange(ra + MC.STEP - 1, rb, MC.STEP, dtype=np.int64)
        for kk, end in enumerate(ends):
            if kk >= 4 and (kk - 4 - phase) % every == 0:
                out.append(int(end))
    return np.asarray(out, np.int64)


# ------------------------------------------------------------------------------------------------ EXPO aggregation
def _pref_vec(ll: Sequence[float], hyps: Sequence[str], cell: Sequence[str]) -> float:
    ll = np.asarray(ll, np.float64)
    a = ll[hyps.index("clean")]
    v = []
    for h in cell:
        if h not in hyps:
            continue
        b = ll[hyps.index(h)]
        v.append(0.5 if not (np.isfinite(a) and np.isfinite(b)) else (1.0 if a > b else (0.5 if a == b else 0.0)))
    return float(np.mean(v)) if v else float("nan")


def expo_pref_table(recs: List[dict], cell: Sequence[str] = EXPO_F5) -> Dict[str, np.ndarray]:
    """{ 'cond|K|arm': (n,) preference per informative chain }."""
    inf = [r for r in recs if "ll" in r]
    keys = list(inf[0]["ll"].keys()) if inf else []
    out = {}
    for ck in keys:
        for an in inf[0]["ll"][ck]:
            out[f"{ck}|{an}"] = np.array([_pref_vec(r["ll"][ck][an], r["hyps"], cell) for r in inf])
    return out


def _thin_mean(tab: Dict[str, np.ndarray], K: int, arm: str) -> np.ndarray:
    return np.mean([tab[f"thin{K}_{d}|{K}|{arm}"] for d in range(EXPO_N_THIN)], axis=0)


def e8_anchor_k1() -> dict:
    """E8 K = 1 F5 preference (P arm) on the training-subject check chains, n-informative-weighted fit + select, per domain."""
    r = json.loads((REPO / "outputs" / "sba" / "r3" / "e8" / "result.json").read_text())["domains"]
    out = {}
    for dom in ("local", "global"):
        num = den = 0.0
        for who in ("fit", "sel"):
            d = r[f"check_{who}_{dom}"]
            num += d["n_informative"] * d["P"]["F5"]["nat1"]["pref"]
            den += d["n_informative"]
        out[dom] = num / den
    return out


def expo_aggregate(recs: List[dict]) -> dict:
    """Per domain: FIT7 selection of the (rho, kc) censoring / ageing constants, SELECT preferences of the arms, increments over the
    single-rate arm with group-bootstrap CIs, the same increments on the cluster-permuted null, and the validity checks."""
    ok = [r for r in recs if "error" not in r]
    out = {"n_chains": len(ok), "n_failed": len(recs) - len(ok), "domains": {}, "e8_anchor_K1": e8_anchor_k1()}
    for dom in ("local", "global"):
        fit = [r for r in ok if domain_of(r["seq"]) == dom and r["seq"].split("_")[0] in FIT7 and "ll" in r]
        sel = [r for r in ok if domain_of(r["seq"]) == dom and r["seq"].split("_")[0] in SELECT and "ll" in r]
        if not fit or not sel:
            continue
        tf, ts = expo_pref_table(fit), expo_pref_table(sel)
        c_arms = [f"c_{rho}_{kc}" for rho in EXPO_RHO for kc in EXPO_KC]
        sel_arm = max(c_arms, key=lambda a: np.nanmean(tf[f"nat|4|{a}"]))
        d = {"n_fit": len(fit), "n_sel": len(sel), "selected_c": sel_arm,
             "fit_c_grid": {a: float(np.nanmean(tf[f"nat|4|{a}"])) for a in c_arms}}
        gb = GroupBoot(group_keys([r["seq"] for r in sel], [r["end"] for r in sel]))
        pool = {"fit+sel": tf}
        # SELECT preferences
        arms = ["a", "b", sel_arm]
        pr = {}
        for an in arms:
            for K in (1, 2, 4):
                pt, bt = gb.wmean(ts[f"nat|{K}|{an}"])
                pr[f"nat|{K}|{an}"] = {"pref": float(pt[0]), "ci": ci(bt[:, 0])}
            for K in (2, 4):
                pt, bt = gb.wmean(_thin_mean(ts, K, an))
                pr[f"thin|{K}|{an}"] = {"pref": float(pt[0]), "ci": ci(bt[:, 0])}
            pt, bt = gb.wmean(ts[f"shufc4|4|{an}"])
            pr[f"shufc|4|{an}"] = {"pref": float(pt[0]), "ci": ci(bt[:, 0])}
            pt, bt = gb.wmean(ts[f"shufw4|4|{an}"])
            pr[f"shufw|4|{an}"] = {"pref": float(pt[0]), "ci": ci(bt[:, 0])}
        d["pref"] = pr
        # increments over the single-rate arm at 200 ms, paired
        inc = {}
        for an in ("b", sel_arm):
            real = ts[f"nat|4|{an}"] - ts["nat|4|a"]
            null = ts[f"shufc4|4|{an}"] - ts["shufc4|4|a"]
            for tag, v in (("real", real), ("null", null), ("real_minus_null", real - null)):
                pt, bt = gb.wmean(v)
                inc[f"{an}|{tag}"] = {"mean": float(pt[0]), "ci95": ci(bt[:, 0]), "ci_holm2": ci(bt[:, 0], (1.25, 98.75))}
            # accumulation gain K=4 minus K=1 of each arm (descriptive, E8's g200)
            g = ts[f"nat|4|{an}"] - ts[f"nat|1|{an}"]
            pt, bt = gb.wmean(g)
            inc[f"{an}|g200"] = {"mean": float(pt[0]), "ci95": ci(bt[:, 0])}
        g = ts["nat|4|a"] - ts["nat|1|a"]
        pt, bt = gb.wmean(g)
        inc["a|g200"] = {"mean": float(pt[0]), "ci95": ci(bt[:, 0])}
        d["increment_over_a"] = inc
        # S5 cell (descriptive)
        t5 = expo_pref_table(sel, EXPO_S5)
        d["S5_nat4"] = {an: float(np.nanmean(t5[f"nat|4|{an}"])) for an in arms}
        # validity: c(rho=1, kc=0) == b exactly; clean ll finite; E8 regression; real beats the cluster null
        eq = [np.array_equal(np.asarray(r["ll"]["nat|4"]["c_1.0_0.0"]), np.asarray(r["ll"]["nat|4"]["b"]), equal_nan=True)
              for r in fit + sel]
        fin = [bool(np.isfinite(r["ll"]["nat|4"]["b"][r["hyps"].index("clean")])) for r in fit + sel]
        allp = expo_pref_table(fit + sel)
        d["validity"] = {"c_equals_b_when_off": bool(np.all(eq)), "clean_finite_frac": float(np.mean(fin)),
                         "pref_b_K1_fit_sel": float(np.nanmean(allp["nat|1|b"])),
                         "e8_anchor_K1": out["e8_anchor_K1"][dom],
                         "real_beats_cluster_null_b": bool(pr["nat|4|b"]["pref"] >= pr["shufc|4|b"]["pref"])}
        out["domains"][dom] = d
    return out


def decide_expo(result: dict) -> dict:
    """Gate of prereg_m3.md section 3, per domain on SELECT at 200 ms (K = 4), F5 cell: with `a` = single multiplier, `b` = class
    multipliers (E8), `c` = b + censoring / ageing (constants chosen on FIT7): pass iff max(b - a, c - a) >= 0.02 with the
    Holm-2 (98.75 %) CI above 0 and its excess over the same increment on the cluster-permuted null has CI lower bound > 0;
    kill iff max increment <= 0.01 or its CI lower bound <= 0; tool defect iff c(off) != b, a clean likelihood is not finite in
    >= 99 % chains, the K = 1 `b` preference is farther than 0.10 from the E8 anchor, or `b` does not beat its cluster null.
    Overall: both domains pass -> pass, both kill -> kill, tool defect anywhere -> tool_defect, else descriptive."""
    agg = result["aggregate"]
    per = {}
    for dom, d in agg["domains"].items():
        v = d["validity"]
        r = {"selected_c": d["selected_c"], "validity": v}
        inc = d["increment_over_a"]
        best = max(("b", d["selected_c"]), key=lambda an: inc[f"{an}|real"]["mean"])
        real, diff = inc[f"{best}|real"], inc[f"{best}|real_minus_null"]
        r.update({"best_arm": best, "increment": real["mean"], "increment_ci_holm2": real["ci_holm2"],
                  "excess_over_null_ci": diff["ci95"]})
        fail_frac = agg.get("n_failed", 0) / max(1, agg.get("n_chains", 1) + agg.get("n_failed", 0))
        r["failed_fraction"] = fail_frac
        defect = (fail_frac > MAX_FAIL_FRAC or not v["c_equals_b_when_off"] or v["clean_finite_frac"] < 0.99 or
                  abs(v["pref_b_K1_fit_sel"] - v["e8_anchor_K1"]) > EXPO_REG_TOL or not v["real_beats_cluster_null_b"])
        if defect:
            r["verdict"] = "tool_defect"
        elif real["mean"] >= EXPO_DELTA_PASS and real["ci_holm2"][0] > 0 and diff["ci95"][0] > 0:
            r["verdict"] = "pass"
        elif real["mean"] <= EXPO_DELTA_KILL or real["ci_holm2"][0] <= 0:
            r["verdict"] = "kill"
        else:
            r["verdict"] = "descriptive"
        per[dom] = r
    vv = [x["verdict"] for x in per.values()]
    overall = ("tool_defect" if "tool_defect" in vv else "pass" if vv and all(x == "pass" for x in vv) else
               "kill" if vv and all(x == "kill" for x in vv) else "descriptive")
    return {"probe": "EXPO", "verdict": overall, "per_domain": per}


# ================================================================================================ unit checks / smoke
def _chk_decide_logic() -> Dict[str, bool]:
    """Pure-logic checks of the three gates on hand-made result dicts (no data): every branch must be reachable."""
    res = {}
    mk_cell = lambda w, pref, adv=0.2, gtn=0.9, n=50: {"w_mm": w, "pref_F5": pref, "pref_gt_over_nowarp": gtn,   # noqa: E731
                                                       "pref_gt_over_nowarp_ci": [gtn - 0.05, gtn + 0.05],
                                                       "adv_real_minus_null_ci": [adv - 0.1, adv + 0.1],
                                                       "n_events_median": n, "n_valid": 50}

    def fbw_result(w, pref, tstar, sba=0.7, adv=0.2, gtn=0.9, n=50):
        subs = {f: {str(T): mk_cell(w, pref, adv, gtn, n) for T in T_GRID} for f in FINGERS5}
        d = {"subtrees": subs}
        return {"aggregate": {"sba_pref_F5": {"local": sba, "global": sba}, "domains": {"local": d, "global": d}},
                "t_star": {dm: {f: {"T_star": tstar, "e95_mm": 5.0} for f in FINGERS5} for dm in ("local", "global")}}
    res["fbw_pass"] = decide_fbw(fbw_result(12, 0.7, 400))["verdict"] == "pass"
    res["fbw_kill_short_basin"] = decide_fbw(fbw_result(3, 0.7, 50))["verdict"] == "kill"
    res["fbw_kill_low_pref"] = decide_fbw(fbw_result(12, 0.3, 400))["verdict"] == "kill"
    res["fbw_tool_defect_no_gain"] = decide_fbw(fbw_result(12, 0.7, 400, gtn=0.4))["verdict"] == "tool_defect"
    res["fbw_tool_defect_null"] = decide_fbw(fbw_result(12, 0.7, 400, adv=-0.3))["verdict"] == "tool_defect"
    res["fbw_descriptive_between"] = decide_fbw(fbw_result(12, 0.6, 100))["verdict"] == "descriptive"
    res["fbw_underpowered_descriptive"] = decide_fbw(fbw_result(12, 0.7, 400, n=5))["verdict"] == "descriptive"

    def smg_result(b, lo_null, hit_lo, orc, syn):
        cell = {"b_off": b, "b_off_ci": [b - 0.05, b + 0.05], "b_off_minus_null_ci": [lo_null, lo_null + 0.2], "hit": 0.7, "hit_ci": [hit_lo, 0.8]}
        d = {"sets": {"tex_hyp": {"cells": {"t14": cell}}, "tex_orc": {"cells": {"t14": {"b_off": orc}}}}}
        return {"aggregate": {"domains": {"local": d, "global": d}, "synthetic_control": {dm: {"b_off_t14": syn} for dm in ("local", "global")}}}
    res["smg_pass"] = decide_smg(smg_result(0.4, 0.1, 0.55, 0.4, 0.6))["verdict"] == "pass"
    res["smg_kill_flat"] = decide_smg(smg_result(0.05, 0.1, 0.55, 0.05, 0.6))["verdict"] == "kill"
    res["smg_kill_not_above_null"] = decide_smg(smg_result(0.4, -0.05, 0.55, 0.4, 0.6))["verdict"] == "kill"
    res["smg_kill_oracle_only"] = decide_smg(smg_result(0.2, 0.1, 0.55, 0.5, 0.6))["verdict"] == "kill"
    res["smg_tool_defect_synthetic"] = decide_smg(smg_result(0.4, 0.1, 0.55, 0.4, 0.1))["verdict"] == "tool_defect"
    res["smg_descriptive"] = decide_smg(smg_result(0.2, 0.1, 0.55, 0.2, 0.6))["verdict"] == "descriptive"

    def expo_result(inc, lo, null_lo, valid=True):
        v = {"c_equals_b_when_off": valid, "clean_finite_frac": 1.0, "pref_b_K1_fit_sel": 0.7, "e8_anchor_K1": 0.7,
             "real_beats_cluster_null_b": True}
        incd = {"b|real": {"mean": inc, "ci95": [lo, inc + 0.05], "ci_holm2": [lo, inc + 0.06]},
                "b|real_minus_null": {"mean": inc, "ci95": [null_lo, inc + 0.05], "ci_holm2": [null_lo, inc + 0.06]},
                "c_0.7_0.01|real": {"mean": inc - 0.005, "ci95": [lo, inc], "ci_holm2": [lo, inc]},
                "c_0.7_0.01|real_minus_null": {"mean": inc, "ci95": [null_lo, inc], "ci_holm2": [null_lo, inc]}}
        d = {"selected_c": "c_0.7_0.01", "validity": v, "increment_over_a": incd}
        return {"aggregate": {"domains": {"local": d, "global": d}}}
    res["expo_pass"] = decide_expo(expo_result(0.05, 0.01, 0.01))["verdict"] == "pass"
    res["expo_kill_small"] = decide_expo(expo_result(0.005, -0.01, 0.01))["verdict"] == "kill"
    res["expo_tool_defect"] = decide_expo(expo_result(0.05, 0.01, 0.01, valid=False))["verdict"] == "tool_defect"
    res["expo_descriptive_not_above_null"] = decide_expo(expo_result(0.05, 0.01, -0.02))["verdict"] == "descriptive"
    return res


def _chk_stats() -> Dict[str, bool]:
    rng = np.random.default_rng(SEED)
    res = {}
    # basin interpolation: linear gain 1 -> 0.2 over the delta grid crosses 0.5 where expected; censored; immediate
    g = np.array([[0.9, 0.8, 0.6, 0.45, 0.3]])
    w = float(interp_basin(g)[0])
    res["basin_interp"] = bool(10 < w < 14 and abs(w - (10 + (0.6 - 0.5) / (0.6 - 0.45) * 4)) < 1e-9)
    res["basin_censored"] = bool(interp_basin(np.array([[0.9, 0.9, 0.8, 0.7, 0.6]]))[0] == DELTAS[-1])
    res["basin_immediate"] = bool(0 <= float(interp_basin(np.array([[0.3, 0.2, 0.1, 0.0, 0.0]]))[0]) < DELTAS[0])
    # group bootstrap: the CI of a known mean brackets it; resampling keeps group structure
    grp = np.repeat(np.arange(30), 10)
    x = rng.normal(2.0, 1.0, 300)
    pt, bt = GroupBoot(grp).wmean(x)
    lo, hi = ci(bt[:, 0])
    res["groupboot_ci"] = bool(lo < pt[0] < hi and hi - lo < 1.0)
    # slope from moments equals polyfit
    xx = rng.normal(0, 1, 500)
    yy = 0.37 * xx + rng.normal(0, 0.1, 500)
    s_ = _slope_moments(xx, yy).sum(0)
    res["slope_moments"] = bool(abs(_slope_of(s_) - np.polyfit(xx, yy, 1)[0]) < 1e-9)
    # ridge recovers a linear map
    A = rng.normal(0, 1, (400, 5))
    Wt = rng.normal(0, 1, (5, 2))
    m = ridge_primal(A, A @ Wt + 0.3, 1e-6)
    res["ridge_recovers"] = bool(np.abs(ridge_apply(m, A) - (A @ Wt + 0.3)).max() < 1e-3)
    # preference: ties and nan
    pv = _pref(np.array([2.0, 1.0, np.nan]), np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 1.0]]))
    res["pref_ties_nan"] = bool(np.allclose(pv[0], [1, 1]) and np.allclose(pv[1], [0.5, 0.0]) and np.isnan(pv[2]).all())
    # PREV_NOISE classes: directed norm bounded, small has the S39 scales
    const = s39_noise_constants()
    sm = np.stack([prev_noise(np.random.default_rng(i), "small", const) for i in range(4000)])
    dr = np.stack([prev_noise(np.random.default_rng(i), "directed", const) for i in range(500)])
    res["noise_small_scales"] = bool(abs(sm[:, 0:3].std() - const["small"][0]) < 3e-4 and abs(sm[:, 6:].std() - const["small"][2]) < 2e-3)
    res["noise_directed_bounded"] = bool(np.linalg.norm(dr[:, 0:3], axis=1).max() <= const["large"][0] + 1e-6 and
                                         np.linalg.norm(dr[:, 6:], axis=1).max() <= const["large"][2] + 1e-6)
    # anchor offset round trip: applying the offset and then the correction restores the pose (translation exactly, rotation by the
    # right-multiplied inverse)
    X = rng.normal(0, 0.1, (5, 51)).astype(np.float32)
    off = np.array([0.01, -0.02, 0.03, 0.05, -0.04, 0.02])
    Xp = apply_root_offset(X, off)
    Xb = apply_root_offset(Xp, offset_target(off))
    res["offset_round_trip"] = bool(np.abs(Xb[:, :3] - X[:, :3]).max() < 1e-6 and np.abs(Xb[:, 3:6] - X[:, 3:6]).max() < 2e-4)
    res["offset_changes_pose"] = bool(np.abs(Xp - X).max() > 1e-3)
    return res


def _chk_smg_aggregate() -> Dict[str, bool]:
    """End-to-end check of the SMG readout / bootstrap / decision path on fabricated anchor records whose real-world features carry the
    offset linearly and whose null-world features are noise: the readout must recover a slope well above the gate and a null near zero."""
    rng = np.random.default_rng(SEED)
    d = 2 + 2 * len(SMG_FD)

    def anchor(name, off, signal):
        tg = -off
        f = rng.normal(0, 1, d)
        fn = rng.normal(0, 1, d)
        if signal:
            f[2] = 80.0 * off[0] * 1e3 / 10 + rng.normal(0, 0.2)
            f[3] = 80.0 * off[1] * 1e3 / 10 + rng.normal(0, 0.2)
            f[4] = 50.0 * off[0] * 1e3 / 10 + rng.normal(0, 0.2)
        return {"name": name, "off": off.tolist(), "target": tg.tolist(),
                "feat": {s_: f.tolist() for s_ in SMG_SETS}, "feat_null": {s_: fn.tolist() for s_ in SMG_SETS}}

    def make(seqs, role, n_end):
        recs = []
        for sq in seqs:
            for e in range(n_end):
                if role == "fit":
                    offs = [("clean", np.zeros(6))] + [("aug", np.r_[rng.normal(0, 0.012, 2), 0, rng.normal(0, 0.05, 3)]) for _ in range(3)]
                else:
                    offs = [("clean", np.zeros(6))]
                    for dd in SMG_T_MM:
                        for _ in range(2):
                            th = rng.uniform(0, 2 * math.pi)
                            offs.append((f"t{dd}", np.r_[dd * 1e-3 * math.cos(th), dd * 1e-3 * math.sin(th), 0, 0, 0, 0]))
                    for dg in SMG_R_DEG:
                        for _ in range(2):
                            offs.append((f"r{dg}", np.r_[0, 0, 0, rng.normal(0, math.radians(dg), 3)]))
                recs.append({"seq": sq, "end": 1000 + 700 * e, "role": role, "anchors": [anchor(n, o, True) for n, o in offs]})
        return recs
    fit = make([f"{s_}_{d_}" for s_ in FIT7 for d_ in ("local", "global")], "fit", 12)
    sel = make([f"{s_}_{d_}" for s_ in SELECT for d_ in ("local", "global")], "select", 12)
    syn_fit, syn_sel = [dict(r, synthetic=True) for r in fit], [dict(r, synthetic=True) for r in sel]
    agg = smg_aggregate(fit, sel, syn_fit, syn_sel)
    c = agg["domains"]["local"]["sets"]["tex_hyp"]["cells"]["t14"]
    dec = decide_smg({"aggregate": agg})
    return {"smg_agg_signal_recovered": bool(c["b_off"] > 0.5), "smg_agg_null_near_zero": bool(abs(c["b_off_null"]) < 0.2),
            "smg_agg_null_ci_covers_zero": bool(c["b_off_null_ci"][0] < 0 < c["b_off_null_ci"][1]),
            "smg_agg_hit_high": bool(c["hit_ci"][0] > 0.5), "smg_agg_decision_pass": bool(dec["verdict"] == "pass")}


def _chk_expo_problem(sm_seq: str = "ycy_local") -> Dict[str, bool]:
    """On one real window: `_expo_ll` without mask / weight equals `r3_lik.problem` + Newton; a unit weight equals no weight; the
    cluster permutation keeps the size multiset of the ms clusters."""
    rc_ = SbaCache.get(sm_seq)
    sd, lik, ctx = rc_["sd"], rc_["lik"], rc_["ctx"]
    ra, rb = (int(v) for v in sd.runs[int(np.argmax(sd.runs[:, 1] - sd.runs[:, 0]))])
    end = ra + 49 + 50 * 6
    kms = [end - 50, end]
    x0, x1 = sd.gt(kms[0]), sd.gt(kms[1])
    win = SCM.window(ctx, sd, kms[0], kms[1], x0, x1)
    F = EM.features(ctx.geo, win, x0, x1, sd.betas, sd.K, lik.patch, lik.n_p, lik.blur, 1.0).flatten(1)
    G = (lik.W @ F)[None]
    ref = L.newton(*L.problem(G, win, "mg"), nan_unconverged=True)[0].cpu().numpy()
    mine = _expo_ll([G], [win], "mg")
    one = _expo_ll([G], [win], "mg", None, [1.0])
    ref1 = L.newton(*L.problem(G, win, "m1"), nan_unconverged=True)[0].cpu().numpy()
    mine1 = _expo_ll([G], [win], "m1")
    keep = (G.sum(1)[0] >= 0.01 * float(G.sum(1).max()))
    cen = _expo_ll([G], [win], "mg", [keep], None)
    ev = sd.events(kms[0], kms[1])
    _, ms2 = cluster_perm_ms(ev, np.random.default_rng(0))
    cs = lambda m: np.sort(np.unique(m, return_counts=True)[1])          # noqa: E731
    return {"expo_problem_equals_r3_lik": bool(np.allclose(ref, mine, rtol=1e-9, atol=1e-9)),
            "expo_unit_weight_equals": bool(np.allclose(one, mine)),
            "expo_m1_equals_r3_lik": bool(np.allclose(ref1, mine1, rtol=1e-9, atol=1e-9)),
            "expo_censor_runs_finite": bool(np.isfinite(cen).all()),
            "expo_cluster_perm_sizes": bool(np.array_equal(cs(ev[1]), cs(ms2)))}


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
    """Engineering smoke: shapes / counts / timings and pass-fail of internal unit checks only; no statistic that could be an effect."""
    t0 = time.time()
    ok_all = True
    say("[smoke] iwe toolkit selftest")
    ok_all &= I.selftest(verbose=False)
    for name, fn in (("decide gates (logic)", _chk_decide_logic), ("statistics helpers", _chk_stats)):
        r = fn()
        bad = [k for k, v in r.items() if not v]
        say(f"[smoke] {name}: {len(r)} checks, {'all pass' if not bad else 'FAIL ' + str(bad)}")
        ok_all &= not bad
    if not real:
        return ok_all
    r = _chk_smg_aggregate()
    bad = [k for k, v in r.items() if not v]
    say(f"[smoke] SMG readout / bootstrap / decision path on fabricated records: {len(r)} checks, {'all pass' if not bad else 'FAIL ' + str(bad)}")
    ok_all &= not bad
    r = _chk_expo_problem()
    bad = [k for k, v in r.items() if not v]
    say(f"[smoke] expo likelihood plumbing: {len(r)} checks, {'all pass' if not bad else 'FAIL ' + str(bad)}")
    ok_all &= not bad
    # tiny real slice: one endpoint per path, shapes only
    seq = "ycy_local"
    rc = RecCtx.get(seq)
    ends = list_endpoints(rc.rec, FBW_EVERY)
    t = time.time()
    o = fbw_one(seq, int(ends[min(3, len(ends) - 1)]), deltas=(5,), n_pn=1)
    say(f"[smoke] FBW one endpoint: hyp {len(o['hyp'])} C {tuple(o['C'].shape)} Cpn {tuple(o['Cpn'].shape)} "
          f"cap {o['n_cap']} finite C0 cells {int(np.isfinite(o['C0']).sum())}/{o['C0'].size} {time.time() - t:.1f} s")
    ok_all &= bool(len(o["hyp"]) > 0 and o["C"].shape[1:] == o["C0"].shape and np.isfinite(o["Cgt"]).any())
    agg = fbw_aggregate([o, {**o, "seq": "ycy_global"}] * 3)
    ts = fbw_tstar(agg)
    dec = decide_fbw({"aggregate": agg, "t_star": ts})
    say(f"[smoke] FBW aggregate+decide path runs: domains {sorted(agg['domains'])} verdict-key {'verdict' in dec}")
    sm = SmgSample(rc, int(list_endpoints(rc.rec, FBW_EVERY, min_hist=SMG_WINDOW + 1)[2]), 1)
    t = time.time()
    f = smg_features(sm, np.array([0.01, 0.0, 0.0, 0, 0, 0]))
    say(f"[smoke] SMG one anchor: sets {sorted(f)} feature dim {len(f['tex_hyp'])} events {len(sm.ev)} {time.time() - t:.1f} s")
    ok_all &= bool(len(f["tex_hyp"]) == 2 + 2 * len(SMG_FD))
    syn = smg_one(seq, int(list_endpoints(rc.rec, FBW_EVERY, min_hist=SMG_WINDOW + 1)[2]), "fit", synthetic=True)
    say(f"[smoke] SMG synthetic-control record: anchors {len(syn['anchors'])} events {syn['n_ev']}")
    chains = expo_endpoints(rc.rec.runs, EXPO_EVERY)
    t = time.time()
    e = expo_one(seq, int(chains[10]), 0)
    say(f"[smoke] EXPO one chain: conditions {len(e.get('ll', {}))} arms {len(next(iter(e['ll'].values())))} hyps {len(e['hyps'])} {time.time() - t:.1f} s")
    ok_all &= bool("ll" in e)
    same = np.array_equal(np.asarray(e["ll"]["nat|4"]["c_1.0_0.0"]), np.asarray(e["ll"]["nat|4"]["b"]), equal_nan=True)
    say(f"[smoke] EXPO arm c with censoring and ageing off equals b: {'pass' if same else 'FAIL'}")
    ok_all &= bool(same)
    try:
        MC.verify_protocol()
        say("[smoke] formal-mode gate: verify_protocol accepts (the freeze is present and valid)")
    except SystemExit:
        say("[smoke] formal-mode gate: verify_protocol refuses (no valid freeze yet), as required")
    say(f"[smoke] m3_probes: {'ALL PASS' if ok_all else 'FAIL'} ({time.time() - t0:.0f} s)")
    _write_smoke_log("m3", ok_all)
    return ok_all


# ================================================================================================ formal run
def _seqs(subjects) -> List[str]:
    return MC.base_seqs(subjects)


def _run_pool(fn, jobs: list, procs: int, tag: str):
    import random
    from multiprocessing import get_context
    random.Random(SEED).shuffle(jobs)
    out = []
    t0 = time.time()
    with get_context("fork").Pool(procs) as pool:
        for i, res in enumerate(pool.imap_unordered(fn, jobs, chunksize=1)):
            out += res
            if (i + 1) % max(1, len(jobs) // 10) == 0:
                print(f"[{tag}] {i + 1}/{len(jobs)} jobs {time.time() - t0:.0f} s", flush=True)
    out.sort(key=lambda r: (str(r.get("seq")), int(r.get("end", 0)), str(r.get("role", "")), bool(r.get("synthetic", False))))   # completion order must not matter
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
    seqs = _seqs(FIT7 + SELECT)
    t0 = time.time()
    meta = {"probe": probe.upper(), "stride": None, "seqs": seqs, "procs": procs, "protocol_sha": MC.sha(MC.PROTOCOL)}
    scripts = [Path(__file__), REPO / "tools/mv2/iwe.py", REPO / "tools/mv2/common.py", REPO / "tools/sba/r3_probe.py"]
    if probe == "fbw":
        ev = every or FBW_EVERY
        meta["stride"] = ev
        jobs = []
        for s in seqs:
            ends = list_endpoints(I.Rec(s), ev)
            jobs += [(s, ends[i:i + 3].tolist(), {}) for i in range(0, len(ends), 3)]
        recs = _run_pool(fbw_job, jobs, procs, "fbw")
        agg = {CH[c]: fbw_aggregate(recs, c) for c in range(3)}
        res = {"probe": "FBW", "aggregate": agg["texture"], "aggregate_all": agg["all"], "aggregate_contour": agg["contour"],
               "t_star": fbw_tstar(agg["texture"]), "t_star_all": fbw_tstar(agg["all"]),
               "t_star_zgz_like_scale": fbw_tstar(agg["texture"], scale=12.53 / 4.48), "meta": meta}
        res["decision"] = decide(res)
        pickle.dump(recs, open(out_dir / "raw.pkl", "wb"))
    elif probe == "smg":
        ev_fit, ev_sel = every or SMG_EVERY_FIT, every or SMG_EVERY_SEL // 2
        meta["stride"] = {"fit": ev_fit, "select": ev_sel, "synthetic_fit": 4 * ev_fit, "synthetic_select": 2 * ev_sel}
        jobs = []
        for s in _seqs(FIT7):
            ends = list_endpoints(I.Rec(s), ev_fit, min_hist=SMG_WINDOW + 1)
            jobs += [(s, [int(e)], "fit", {}) for e in ends]
            ends = list_endpoints(I.Rec(s), 4 * ev_fit, min_hist=SMG_WINDOW + 1)
            jobs += [(s, [int(e)], "fit", {"synthetic": True}) for e in ends]
        for s in _seqs(SELECT):
            ends = list_endpoints(I.Rec(s), ev_sel, min_hist=SMG_WINDOW + 1)
            jobs += [(s, [int(e)], "select", {}) for e in ends]
            ends = list_endpoints(I.Rec(s), 2 * ev_sel, min_hist=SMG_WINDOW + 1)
            jobs += [(s, [int(e)], "select", {"synthetic": True}) for e in ends]
        recs = _run_pool(smg_job, jobs, procs, "smg")
        fit = [r for r in recs if r.get("role") == "fit" and not r.get("synthetic")]
        sel = [r for r in recs if r.get("role") == "select" and not r.get("synthetic")]
        fs = [r for r in recs if r.get("role") == "fit" and r.get("synthetic")]
        ss = [r for r in recs if r.get("role") == "select" and r.get("synthetic")]
        res = {"probe": "SMG", "aggregate": smg_aggregate(fit, sel, fs, ss), "meta": meta,
               "n_failed": sum(1 for r in recs if "error" in r), "n_total": len(recs)}
        res["decision"] = decide(res)
        pickle.dump(recs, open(out_dir / "raw.pkl", "wb"))
    elif probe == "expo":
        ev = every or EXPO_EVERY
        meta["stride"] = ev
        jobs = []
        for s in seqs:
            rec = I.Rec(s)
            ends = expo_endpoints(rec.runs, ev)
            items = [(int(e), i) for i, e in enumerate(ends)]
            jobs += [(s, items[i:i + 4]) for i in range(0, len(items), 4)]
        recs = _run_pool(expo_job, jobs, procs, "expo")
        res = {"probe": "EXPO", "aggregate": expo_aggregate(recs), "meta": meta}
        res["decision"] = decide(res)
        pickle.dump(recs, open(out_dir / "raw.pkl", "wb"))
    else:
        raise SystemExit(f"unknown probe {probe}")
    res["meta"]["wall_s"] = time.time() - t0
    res = json.loads(json.dumps(res, default=_jdefault))          # numpy scalars (incl. bool) -> plain JSON values
    MC.save_json(out_dir / "result.json", res)
    MC.receipt(out_dir, {"probe": probe, "wall_s": time.time() - t0, "verdict": res["decision"]["verdict"]}, scripts=scripts)
    print(f"[{probe}] done in {time.time() - t0:.0f} s -> {out_dir}/result.json (verdict {res['decision']['verdict']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", choices=("fbw", "smg", "expo"), default=None)
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
