#!/usr/bin/env python3
"""SBA round 3, E0 probe contract (docs/EXPERIMENT_RECORDS.md#sba-r3-prereg-20261022); CPU, GT geometry only.

Anatomical finger perturbations scaled per packet by the actual FK joint error. They replace P0 / Q0's 'finger'
probe (local x right-multiplied: a twist about the bone axis, about 0.5 mm of joint error, #sba-why-20261022):

    pip / mcpf / mcpa   one long finger (packet index % 4): PIP flexion, MCP flexion, MCP abduction; target = the
                        affected finger's mean joint error over its 4 OpenPose nodes (wrist-aligned), 5 / 10 mm
    thm                 thumb CMC opposition; thumb 4 nodes, 5 / 10 mm
    syn                 four long fingers, MCP : PIP : DIP = 1 : 1 : 0.67; 21-joint wrist-aligned mean, 3 / 5 / 10 mm
    synrc               syn with its best-fit 3-DoF root rotation removed in the same joint metric, re-scaled to 5 / 10 mm
    root / trans        rigid references: root about camera axis index % 3 (wrist fixed), 21-joint 5 / 10 mm; lateral
                        translation along x / y (index % 2), exactly 5 / 10 mm
    tw                  the old probe (local x, +-20 deg, joint index % 15) -- regression and near-zero reference
    fsw                 the finger block of another packet of the same recording (natural direction), root / t kept
    gauge0.97 / 1.03    mesh scale about the wrist at fixed depth (state = GT; E1 scores it at sigma * s)
    synM                syn at the end knot only (increment error)
    sm25                both knots replaced by the +-25 ms smoothed GT (label-jitter sentinel)

Frame from the recording's shaped rest geometry: palm normal n = (J_indexMCP - J_wrist) x (J_pinkyMCP - J_wrist), oriented
so the mean pose curls the long-finger tips towards +n; flexion axis b x n with b = MCP -> tip projected on the palm plane
(one hinge direction for MCP / PIP / DIP: + curls towards the palm); abduction axis +-n (+ away from the middle finger;
middle: towards the index); thumb opposition axis = shortest-arc axis turning the CMC -> tip direction (palm frame, at
the packet's end knot) towards the palm centre + 30 mm n. A joint is rotated in its parent's frame -- the hinge is fixed
in the proximal segment: full = residual + hands_mean, R <- exp(theta a) R, residual = log R - hands_mean.

Amplitudes are found by bisection on the end knot (18 steps on [0, 80] deg, <= 2 %); the same angle (and axis, root
compensation) is applied at the start knot (condition A: both knots displaced).

    python tools/sba/r3_probe.py run
    python tools/sba/r3_probe.py summarize
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sba import common as CM  # noqa: E402
from sba.common import SB, C  # noqa: E402
from sba.p0_landscape import packets  # noqa: E402
from sba.p0_visibility import render, BIN2  # noqa: E402

OUT = CM.OUT / "r3" / "e0"
DOMAINS = ("zgz_local", "zgz_global", "check")
FINGERS = ("index", "middle", "ring", "pinky")
#: MANO joints (1..15; hands_mean row = joint - 1) and OpenPose nodes of every finger
MANO_J = {"index": (1, 2, 3), "middle": (4, 5, 6), "pinky": (7, 8, 9), "ring": (10, 11, 12), "thumb": (13, 14, 15)}
OP = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
      "pinky": (17, 18, 19, 20)}
TIP = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}            # rows of fingertip_vertex_ids
SYN_RATIO = (1.0, 1.0, 0.67)
THUMB_LIFT = 0.030
TIERS_F, TIERS_W, TIERS_RC = (5, 10), (3, 5, 10), (5, 10)
HI_DEG, ITERS, TOL = 80.0, 18, 0.02
GAUGE = {"gauge0.97": 0.97, "gauge1.03": 1.03}
FSW_MIN_MM = 5.0
ROOT_FD = 1e-3                                                                 # rad, root-rotation Jacobian step
#: frozen E0 validity rules (written into outputs/sba/r3/protocol.json)
VALID = {"bisect_ok_frac": 0.999, "linear_band": (1.8, 2.1), "linear_frac": 0.95, "flex_tip_cos": 0.98,
         "twist_template_mm": 1.0}


def hyp_names() -> List[str]:
    h = ["clean", "sm25"]
    for fam in ("pip", "mcpf", "mcpa", "thm"):
        h += [f"{fam}{s * t:+d}" for t in TIERS_F for s in (1, -1)]
    h += [f"syn{s * t:+d}" for t in TIERS_W for s in (1, -1)]
    h += [f"synrc{s * t:+d}" for t in TIERS_RC for s in (1, -1)]
    h += [f"root{s * t:+d}" for t in TIERS_F for s in (1, -1)]
    h += [f"trans{s * t:+d}" for t in TIERS_F for s in (1, -1)]
    h += ["tw+20", "tw-20"]
    h += [f"synM{s * t:+d}" for t in TIERS_F for s in (1, -1)]
    return h + ["fsw"] + list(GAUGE)


HYP = hyp_names()
#: families whose amplitude is bisected, with their target metric
BISECT = {"pip": "finger", "mcpf": "finger", "mcpa": "finger", "thm": "thumb", "syn": "j21", "synrc": "j21",
          "root": "j21"}


def split(h: str):
    """'pip+5' -> ('pip', +1, 5); 'tw-20' -> ('tw', -1, 20); others -> (h, 0, 0)."""
    for k in ("synrc", "synM", "syn", "pip", "mcpf", "mcpa", "thm", "root", "trans", "tw"):
        if h.startswith(k) and h[len(k):len(k) + 1] in "+-" and len(h) > len(k):
            v = int(h[len(k):])
            return k, (1 if v > 0 else -1), abs(v)
    return h, 0, 0


# ------------------------------------------------------------------------------------------------ geometry
def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)


@dataclass
class Frame:
    n: torch.Tensor                    # (3,) palm normal, palmar
    flex: Dict[str, torch.Tensor]      # long finger -> (3,) flexion axis (parent frame)
    abd: Dict[str, torch.Tensor]       # long finger -> (3,) abduction axis (+ away from the middle finger)
    J: torch.Tensor                    # (16, 3) shaped rest joints (MANO order)
    target: torch.Tensor               # (3,) thumb opposition target, palm frame


@torch.no_grad()
def make_frame(ctx, betas) -> Frame:
    m = ctx.mano
    b = betas.view(1, -1).float()
    vs = m.v_template + torch.einsum("bl,mkl->bmk", b, m.shapedirs)[0]
    J = m.J_regressor @ vs
    tips = vs[m.fingertip_vertex_ids]
    n = unit(torch.cross(J[1] - J[0], J[7] - J[0], dim=0))
    z3, z45 = torch.zeros(1, 3), torch.zeros(1, 45)
    _, j_rest = m(b, z3, z45, z3)
    _, j_mean = m(b, z3, m.hands_mean.view(1, 45), z3)
    tip_op = [OP[f][3] for f in FINGERS]
    if float(((j_mean[0, tip_op] - j_rest[0, tip_op]) @ n).mean()) < 0:
        n = -n
    flex, abd = {}, {}
    for f in FINGERS:
        mcp = MANO_J[f][0]
        r = tips[TIP[f]] - J[mcp]
        bdir = unit(r - (r @ n) * n)
        flex[f] = unit(torch.cross(bdir, n, dim=0))
        lat = torch.cross(n, bdir, dim=0)
        ref = J[1] if f == "middle" else J[mcp]
        s = 1.0 if float((ref - J[4]) @ lat) >= 0 else -1.0
        abd[f] = s * n
    return Frame(n, flex, abd, J, 0.5 * (J[1] + J[7]) + THUMB_LIFT * n)


def rot_parent(ctx, X: torch.Tensor, jidx: int, axis: torch.Tensor, rad: torch.Tensor) -> torch.Tensor:
    """X (B, 51): MANO joint `jidx` (1..15) rotated by `rad` (B,) about `axis` ((3,) or (B, 3)) fixed in the parent's
    frame (left-multiplied on the full local rotation, hands_mean restored first)."""
    k = jidx - 1
    X = X.clone()
    full = X[:, 6 + 3 * k:9 + 3 * k] + ctx.mean[k]
    w = rad[:, None] * axis.expand(len(X), 3)
    X[:, 6 + 3 * k:9 + 3 * k] = CM.so3_log(CM.so3_exp(w) @ CM.so3_exp(full)) - ctx.mean[k]
    return X


def root_rotvec(ctx, X: torch.Tensor, w: torch.Tensor, betas) -> torch.Tensor:
    """Root left-multiplied in the camera frame by exp(w) (B, 3) rad, translation compensated so that the FK wrist
    stays (as tools/sba/common.py::perturb)."""
    X = X.clone()
    w0 = ctx.wrist(X, betas)
    X[:, 3:6] = CM.so3_log(CM.so3_exp(w) @ CM.so3_exp(X[:, 3:6]))
    X[:, :3] += w0 - ctx.wrist(X, betas)
    return X


@torch.no_grad()
def joints(ctx, X: torch.Tensor, betas) -> torch.Tensor:
    return ctx.geo.fk(X, betas)[1]


def errors(jh: torch.Tensor, jc: torch.Tensor):
    """(B, 21) per-node wrist-aligned displacement (mm) and (B,) 21-joint absolute error (mm)."""
    d = (jh - jh[:, :1]) - (jc - jc[:, :1])
    return d.norm(dim=-1) * 1000.0, (jh - jc).norm(dim=-1).mean(1) * 1000.0


def metric(e21: torch.Tensor, kind: str, finger: str) -> torch.Tensor:
    if kind == "j21":
        return e21.mean(1)
    nodes = list(OP["thumb" if kind == "thumb" else finger])
    return e21[:, nodes].mean(1)


def thumb_axis(ctx, fr: Frame, x: torch.Tensor, betas) -> torch.Tensor:
    """Opposition axis of one pose: CMC -> thumb tip (palm frame) turned towards the opposition target."""
    xp = x.clone().view(1, 51)
    xp[0, :6] = 0.0                                                            # palm frame: no root, no translation
    j = joints(ctx, xp, betas)[0]
    t = unit(j[OP["thumb"][3]] - fr.J[13])
    p = unit(fr.target - fr.J[13])
    return unit(torch.cross(t, p, dim=0))


def root_jacobian(ctx, x: torch.Tensor, betas) -> torch.Tensor:
    """(63, 3) d(wrist-aligned joints, m) / d(root rotation vector in the camera frame, rad) by central differences."""
    E = torch.eye(3) * ROOT_FD
    X = x.view(1, 51).expand(6, 51)
    Xs = root_rotvec(ctx, X, torch.cat([E, -E]), betas)
    j = joints(ctx, Xs, betas)
    j = j - j[:, :1]
    return ((j[:3] - j[3:]).reshape(3, 63) / (2 * ROOT_FD)).T


@dataclass
class Spec:
    """One hypothesis, appliable to any knot."""
    fam: str
    sign: int = 0
    deg: float = 0.0
    finger: Optional[str] = None
    axis: Optional[List[float]] = None         # thumb opposition / root rotation axis
    omega: Optional[List[float]] = None        # synrc: root rotation vector (rad) that was removed
    delta: Optional[List[float]] = None        # fsw: additive finger-block offset
    mm: float = 0.0                            # trans
    t_axis: int = 0
    joint: int = 0                             # tw: joint index (0..14)
    scale: float = 1.0                         # gauge


def apply(ctx, fr: Frame, sp: Spec, X: torch.Tensor, betas, deg: Optional[torch.Tensor] = None,
          omega: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply a hypothesis to knots X (B, 51). `deg` (B,) / `omega` (B, 3) override the spec (bisection)."""
    B = X.shape[0]
    d = torch.full((B,), sp.sign * sp.deg) if deg is None else deg
    rad = torch.deg2rad(d)
    f = sp.fam
    if f == "pip":
        return rot_parent(ctx, X, MANO_J[sp.finger][1], fr.flex[sp.finger], rad)
    if f == "mcpf":
        return rot_parent(ctx, X, MANO_J[sp.finger][0], fr.flex[sp.finger], rad)
    if f == "mcpa":
        return rot_parent(ctx, X, MANO_J[sp.finger][0], fr.abd[sp.finger], rad)
    if f == "thm":
        return rot_parent(ctx, X, 13, torch.tensor(sp.axis, dtype=X.dtype), rad)
    if f in ("syn", "synrc", "synM"):
        for fg in FINGERS:
            for j, r in zip(MANO_J[fg], SYN_RATIO):
                X = rot_parent(ctx, X, j, fr.flex[fg], rad * r)
        if f == "synrc":
            om = torch.tensor(sp.omega, dtype=X.dtype).expand(B, 3) if omega is None else omega
            X = root_rotvec(ctx, X, -om, betas)
        return X
    if f == "root":
        return root_rotvec(ctx, X, rad[:, None] * torch.tensor(sp.axis, dtype=X.dtype), betas)
    if f == "trans":
        X = X.clone()
        X[:, sp.t_axis] += sp.sign * sp.mm / 1000.0
        return X
    if f == "tw":
        return torch.stack([CM.perturb(ctx, x, "finger", sp.sign * sp.deg, sp.joint, betas) for x in X])
    if f == "fsw":
        X = X.clone()
        X[:, 6:] += torch.tensor(sp.delta, dtype=X.dtype)
        return X
    return X.clone()                                                           # clean, gauge


def knots(ctx, fr: Frame, sp: Spec, x0: torch.Tensor, x1: torch.Tensor, betas):
    """(x0', x1') of a hypothesis under condition A (both knots); synM displaces the end knot only."""
    if sp.fam == "synM":
        return x0.clone(), apply(ctx, fr, sp, x1[None], betas)[0]
    X = apply(ctx, fr, sp, torch.stack([x0, x1]), betas)
    return X[0], X[1]


# ------------------------------------------------------------------------------------------------ bisection
def _syn_rc(ctx, fr, X, deg, Bj, j_clean, betas):
    """Root-compensated synergy at angles deg (B,): remove the least-squares root rotation twice (one refinement)."""
    sp = Spec("syn")
    Xs = apply(ctx, fr, sp, X, betas, deg=deg)
    om = torch.zeros(len(X), 3)
    pinv = torch.linalg.pinv(Bj)                                               # (3, 63)
    jc = (j_clean - j_clean[:, :1]).reshape(1, 63)
    for _ in range(2):
        Xc = root_rotvec(ctx, Xs, -om, betas)
        jh = joints(ctx, Xc, betas)
        dlt = (jh - jh[:, :1]).reshape(len(X), 63) - jc
        om = om + dlt @ pinv.T
    return root_rotvec(ctx, Xs, -om, betas), om


def bisect(ctx, fr, fam: str, sign: int, target: float, x: torch.Tensor, betas, finger, axis, Bj, j_clean):
    """Angle (deg >= 0) of `fam` with sign `sign` whose error metric at the knot x reaches `target` mm."""
    kind = BISECT[fam]
    lo, hi = 0.0, HI_DEG
    sp = Spec(fam, sign, 0.0, finger, axis)

    def err(a):
        d = torch.tensor([sign * a])
        if fam == "synrc":
            Xh, om = _syn_rc(ctx, fr, x[None], d, Bj, j_clean, betas)
        else:
            Xh, om = apply(ctx, fr, sp, x[None], betas, deg=d), None
        e21, _ = errors(joints(ctx, Xh, betas), j_clean)
        return float(metric(e21, kind, finger)[0]), om

    e_hi, _ = err(hi)
    if e_hi < target:
        return hi, False
    for _ in range(ITERS):
        mid = 0.5 * (lo + hi)
        e, _ = err(mid)
        lo, hi = (mid, hi) if e < target else (lo, mid)
    return 0.5 * (lo + hi), True


# ------------------------------------------------------------------------------------------------ run
def fsw_spec(ctx, sd, rows, index: int, x1: torch.Tensor, j_clean, betas):
    """Finger block of the first row from index + n/2 on (cyclic) whose finger change at the end knot is >= 5 mm."""
    n = len(rows)
    for k in range(n):
        r = rows[(index + n // 2 + k) % n]
        g = sd.gt(int(r["end_ms"]))
        delta = g[6:] - x1[6:]
        xh = x1.clone()
        xh[6:] += delta
        e21, _ = errors(joints(ctx, xh[None], betas), j_clean)
        D = float(e21.mean())
        if D >= FSW_MIN_MM:
            return Spec("fsw", delta=delta.tolist()), {"donor_index": int(r["index"]), "D_mm": D}
    return Spec("fsw", delta=[0.0] * 45), {"donor_index": None, "D_mm": 0.0}


def work(job):
    torch.set_num_threads(1)
    seq, d, items = job
    ctx = CM.Ctx(torch.device("cpu"))
    sd = CM.SeqData(seq, d, torch.device("cpu"))
    betas = sd.betas
    fr = make_frame(ctx, betas)
    faces = ctx.mano.f.long().numpy()
    vg = ctx.geo.groups.numpy()
    fgroup = np.array([np.bincount(vg[f], minlength=6).argmax() for f in faces])
    recs, states = [], []
    for index, row, rows in items:
        prev, end = int(row["previous_end_ms"]), int(row["end_ms"])
        x0, x1 = sd.gt(prev), sd.gt(end)
        finger = FINGERS[index % 4]
        j_clean = joints(ctx, x1[None], betas)
        j_clean0 = joints(ctx, x0[None], betas)
        Bj = root_jacobian(ctx, x1, betas)
        pinvB = torch.linalg.pinv(Bj)
        th_axis = thumb_axis(ctx, fr, x1, betas).tolist()
        specs: Dict[str, Spec] = {}
        meta: Dict[str, dict] = {}
        for h in HYP:
            fam, sign, tier = split(h)
            if fam in BISECT:
                ax = th_axis if fam == "thm" else ([1.0 if i == index % 3 else 0.0 for i in range(3)]
                                                  if fam == "root" else None)
                deg, ok = bisect(ctx, fr, fam, sign, float(tier), x1, betas, finger, ax, Bj, j_clean)
                sp = Spec(fam, sign, deg, finger, ax)
                if fam == "synrc":
                    _, om = _syn_rc(ctx, fr, x1[None], torch.tensor([sign * deg]), Bj, j_clean, betas)
                    sp.omega = om[0].tolist()
                specs[h], meta[h] = sp, {"deg": deg, "ok": ok, "target": float(tier)}
            elif fam == "synM":
                base = specs[f"syn{sign * tier:+d}"]
                specs[h], meta[h] = Spec("synM", sign, base.deg, finger), {"deg": base.deg, "target": float(tier)}
            elif fam == "trans":
                specs[h], meta[h] = Spec("trans", sign, mm=float(tier), t_axis=index % 2), {"target": float(tier)}
            elif fam == "tw":
                specs[h], meta[h] = Spec("tw", sign, float(tier), joint=index % 15), {"deg": float(tier)}
            elif h == "fsw":
                sp, info = fsw_spec(ctx, sd, rows, index, x1, j_clean, betas)
                specs[h], meta[h] = sp, info
            elif h in GAUGE:
                specs[h], meta[h] = Spec("gauge", scale=GAUGE[h]), {"scale": GAUGE[h]}
            elif h == "sm25":
                lo0, hi0 = sd.run_bounds(prev)
                lo1, hi1 = sd.run_bounds(end)
                s0 = torch.tensor(C.smoothed_gt(sd.pos51, prev, lo0, hi0), dtype=torch.float32)
                s1 = torch.tensor(C.smoothed_gt(sd.pos51, end, lo1, hi1), dtype=torch.float32)
                specs[h], meta[h] = Spec("sm25"), {}
                meta[h]["_knots"] = (s0, s1)
            else:
                specs[h], meta[h] = Spec("clean"), {}
        X = torch.zeros(len(HYP), 2, 51)
        for i, h in enumerate(HYP):
            if h == "sm25":
                X[i, 0], X[i, 1] = meta[h].pop("_knots")
            else:
                X[i, 0], X[i, 1] = knots(ctx, fr, specs[h], x0, x1, betas)
        # errors at both knots, root-explainable fraction, linearity (2x angle), visible change at the end knot
        v1, j1 = ctx.geo.fk(X[:, 1], betas)
        _, j0 = ctx.geo.fk(X[:, 0], betas)
        e21_1, eabs_1 = errors(j1, j_clean.expand_as(j1))
        e21_0, _ = errors(j0, j_clean0.expand_as(j0))
        jr = (j1 - j1[:, :1]).reshape(len(HYP), 63) - (j_clean - j_clean[:, :1]).reshape(1, 63)
        om_fit = jr @ pinvB.T
        res = jr - om_fit @ Bj.T
        rexp = 1.0 - (res ** 2).sum(1) / (jr ** 2).sum(1).clamp(min=1e-18)
        rexp[e21_1.mean(1) < 0.05] = float("nan")                              # no wrist-aligned displacement
        uv = SB.project_src(v1, sd.K).numpy()
        z = v1[..., 2].numpy()
        for g, s in GAUGE.items():
            i = HYP.index(g)
            w = j1[i:i + 1, :1]
            vg_ = w + s * (v1[i:i + 1] - w)
            uv[i] = SB.project_src(vg_, sd.K)[0].numpy()
        lo_ = np.floor(uv.reshape(-1, 2).min(0)).astype(int) - 2
        hi_ = np.ceil(uv.reshape(-1, 2).max(0)).astype(int) + 2
        box = (int(lo_[0]), int(lo_[1]), int(hi_[0]), int(hi_[1]))
        maps = [render(uv[i], z[i], faces, fgroup, box) for i in range(len(HYP))]
        hyps = {}
        for i, h in enumerate(HYP):
            fam = split(h)[0]
            kind = BISECT.get(fam if fam != "synM" else "syn", "j21")
            m = meta[h]
            m.update({"e_finger": float(metric(e21_1[i:i + 1], "finger", finger)[0]),
                      "e_thumb": float(metric(e21_1[i:i + 1], "thumb", finger)[0]),
                      "e_j21": float(e21_1[i].mean()), "e_abs": float(eabs_1[i]),
                      "e_metric_x0": float(metric(e21_0[i:i + 1], kind, finger)[0]),
                      "e_metric_x1": float(metric(e21_1[i:i + 1], kind, finger)[0]),
                      "root_explained": float(rexp[i]),
                      "area": float((maps[i] != maps[0]).sum() * BIN2),
                      "dmax": float(np.linalg.norm(uv[i] - uv[0], axis=1).max() * 3 / 8)})
            if fam in BISECT and fam != "synrc":
                X2 = apply(ctx, fr, specs[h], x1[None], betas, deg=torch.tensor([2 * specs[h].sign * specs[h].deg]))
                e2, _ = errors(joints(ctx, X2, betas), j_clean)
                m["linear_ratio"] = float(metric(e2, kind, finger)[0]) / max(m["e_metric_x1"], 1e-9)
            elif fam == "synrc":
                X2, _ = _syn_rc(ctx, fr, x1[None], torch.tensor([2 * specs[h].sign * specs[h].deg]), Bj, j_clean,
                                betas)
                e2, _ = errors(joints(ctx, X2, betas), j_clean)
                m["linear_ratio"] = float(e2.mean()) / max(m["e_metric_x1"], 1e-9)
            m["spec"] = {k: v for k, v in specs[h].__dict__.items() if v not in (None, 0, 0.0) or k == "fam"}
            hyps[h] = m
        recs.append({"seq": seq, "dir": d, "index": index, "end": end, "prev": prev, "run_id": int(row["run_id"]),
                     "finger": finger, "joint15": index % 15, "depth": float(x1[2]),
                     "hand_px": float(np.sqrt((maps[0] > 0).sum() * BIN2)), "hyps": hyps})
        states.append(X.numpy())
    return recs, states


@torch.no_grad()
def template_checks(ctx, betas) -> dict:
    """Axis sanity on the template pose of one hand shape (frozen E0 validity rules)."""
    fr = make_frame(ctx, betas)
    x = torch.zeros(1, 51)
    x[0, 6:] = -ctx.mean.reshape(-1)                                           # full local rotation = 0 (template)
    x[0, 2] = 0.5
    j0 = joints(ctx, x, betas)
    out = {"flex_tip_cos": {}, "abd_lateral_cos": {}, "twist_mm": {}}
    for f in FINGERS:
        for name, jj, ax in (("mcp", MANO_J[f][0], fr.flex[f]), ("pip", MANO_J[f][1], fr.flex[f])):
            jh = joints(ctx, rot_parent(ctx, x, jj, ax, torch.tensor([math.radians(5.0)])), betas)
            dt = jh[0, OP[f][3]] - j0[0, OP[f][3]]
            out["flex_tip_cos"][f"{f}_{name}"] = float(unit(dt) @ fr.n)
        jh = joints(ctx, rot_parent(ctx, x, MANO_J[f][0], fr.abd[f], torch.tensor([math.radians(5.0)])), betas)
        dt = unit(jh[0, OP[f][3]] - j0[0, OP[f][3]])
        lat = unit(torch.cross(fr.n, unit(j0[0, OP[f][3]] - j0[0, OP[f][0]]), dim=0))
        out["abd_lateral_cos"][f] = float(abs(dt @ lat))
    xt = x.clone()
    ax = thumb_axis(ctx, fr, xt[0], betas)
    jh = joints(ctx, rot_parent(ctx, xt, 13, ax, torch.tensor([math.radians(5.0)])), betas)
    # target in palm frame == camera frame here (zero root, translation only along z)
    tgt = fr.target + x[0, :3]
    out["thumb_target_dist_mm"] = [float((j0[0, 4] - tgt).norm() * 1000), float((jh[0, 4] - tgt).norm() * 1000)]
    for k in range(15):
        jh = joints(ctx, CM.perturb(ctx, x[0], "finger", 20, k, betas)[None], betas)
        e21, _ = errors(jh, j0)
        out["twist_mm"][str(k)] = float(e21.mean())
    return out


#: twist rows held to the near-zero rule: index / middle / ring, whose rest bones lie within 4-18 deg of local x
#: (pinky 33-36 deg and thumb 35-50 deg off, so local x is not a twist there; reported, not held)
TWIST_HELD = tuple(str(k) for k in (0, 1, 2, 3, 4, 5, 9, 10, 11))


def run(procs: int):
    jobs = []
    for domain in DOMAINS:
        by = {}
        for seq, d, index, row, rows in packets(domain):
            by.setdefault((seq, d), []).append((index, row, rows))
        for (seq, d), items in by.items():
            jobs += [(seq, d, items[k:k + 20]) for k in range(0, len(items), 20)]
    OUT.mkdir(parents=True, exist_ok=True)
    with Pool(procs) as pool:
        res = pool.map(work, jobs)
    dom_of = lambda s: s if s in CM.ZGZ else "check"                           # noqa: E731
    for domain in DOMAINS:
        recs = [r for rr, _ in res for r in rr if dom_of(r["seq"]) == domain]
        st = [s for rr, ss in res for r, s in zip(rr, ss) if dom_of(r["seq"]) == domain]
        (OUT / f"probes_{domain}.json").write_text(json.dumps({"hyp": HYP, "records": recs}))
        np.savez_compressed(OUT / f"states_{domain}.npz", keys=np.array([f"{r['seq']}|{r['index']}" for r in recs]),
                            X=np.stack(st).astype(np.float32))
    ctx = CM.Ctx(torch.device("cpu"))
    seqs = sorted({r["seq"].replace("_v2", "").replace("_v3", "").replace("_v4", "") for rr, _ in res for r in rr})
    tc = {}
    for s in seqs:
        d = "val" if s in CM.ZGZ else "train"
        tc[s] = template_checks(ctx, CM.SeqData(s, d, torch.device("cpu")).betas)
    (OUT / "template_checks.json").write_text(json.dumps(tc, indent=1))
    CM.receipt(OUT / "run", {"probe": "SBA r3 E0", "packets": sum(len(rr) for rr, _ in res)}, scripts=(__file__,))
    print(sum(len(rr) for rr, _ in res), "packets")


def summarize():
    res = {"domains": {}, "validity": {}}
    tc = json.loads((OUT / "template_checks.json").read_text())
    oks, lin = [], []
    for domain in DOMAINS:
        recs = json.loads((OUT / f"probes_{domain}.json").read_text())["records"]
        for sub in (("local", "global") if domain == "check" else ("all",)):
            rs = [r for r in recs if sub == "all" or f"_{sub}" in r["seq"]]
            name = domain if sub == "all" else f"{domain}_{sub}"
            tab = {}
            for h in HYP:
                ms = [r["hyps"][h] for r in rs]
                tab[h] = {k: float(np.median([m[k] for m in ms if k in m])) for k in
                          ("deg", "e_finger", "e_thumb", "e_j21", "e_abs", "e_metric_x0", "e_metric_x1",
                           "root_explained", "area", "dmax", "D_mm") if any(k in m for m in ms)}
                tab[h]["area_p10"] = float(np.percentile([m["area"] for m in ms], 10))
                tab[h]["frac_area_ge40"] = float(np.mean([m["area"] >= 40 for m in ms]))
                if split(h)[0] in BISECT:
                    tab[h]["ok_frac"] = float(np.mean([m["ok"] for m in ms]))
            res["domains"][name] = {"n": len(rs), "hyps": tab,
                                    "hand_px_median": float(np.median([r["hand_px"] for r in rs])),
                                    "depth_median": float(np.median([r["depth"] for r in rs]))}
        for r in recs:
            for h in HYP:
                m = r["hyps"][h]
                if split(h)[0] in BISECT:
                    oks.append(abs(m["e_metric_x1"] / m["target"] - 1.0) <= TOL)
                    lo, hi = VALID["linear_band"]
                    lin.append(lo <= m["linear_ratio"] <= hi)
    flex_cos = min(v for t in tc.values() for v in t["flex_tip_cos"].values())
    abd_cos = min(v for t in tc.values() for v in t["abd_lateral_cos"].values())
    twist = max(v for t in tc.values() for k, v in t["twist_mm"].items() if k in TWIST_HELD)
    thumb = all(t["thumb_target_dist_mm"][1] < t["thumb_target_dist_mm"][0] for t in tc.values())
    v = {"bisect_ok_frac": float(np.mean(oks)), "linear_frac": float(np.mean(lin)), "flex_tip_cos_min": flex_cos,
         "abd_lateral_cos_min": abd_cos, "twist_template_mm_max": twist, "thumb_opposition_shortens": thumb}
    v["pass"] = bool(v["bisect_ok_frac"] >= VALID["bisect_ok_frac"] and v["linear_frac"] >= VALID["linear_frac"]
                     and flex_cos >= VALID["flex_tip_cos"] and twist < VALID["twist_template_mm"] and thumb)
    res["validity"] = v
    (OUT / "result.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(v, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("run", "summarize"))
    ap.add_argument("--procs", type=int, default=48)
    a = ap.parse_args()
    run(a.procs) if a.cmd == "run" else summarize()


if __name__ == "__main__":
    main()
