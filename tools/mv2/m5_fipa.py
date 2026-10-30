#!/usr/bin/env python3
"""M5 probe 3: fixed-iteration static-graph palm alignment FIPA (P5-E2-R1).

A branch-free, static-shape Gauss-Newton on a 6-DoF palm correction (R, tau) about the wrist:

  * input  : the palm vertices (wrist LBS weight >= 0.9, 132 of 778) of the initial pose with their vertex normals and the
             wrist point; a FIXED-CAPACITY event buffer (C in {256, 512, 1024} source-pixel positions + validity flags,
             built on the host: the in-ROI events of the 50 ms window, time-stratified stride, zero padded);
  * model  : an event-to-contour distance field (not the SBA footprint, whose plan has data-dependent shapes): the palm
             vertices with a soft occluding-contour weight w_i = exp(-(n_i . ray_i)^2 / 2 s^2) are the contour samples;
             each event takes its nearest weighted sample j* (cost D^2 / 2 sigma_a^2 - log w), the residual is the signed
             distance along the sample's outward image normal, Cauchy weighted and distance gated;
  * solver : K unrolled iterations of  H = A^T Q A + P0 + mu_abs I  (6 x 6, Cholesky), step = -H^-1 A^T Q r, caps by clamp,
             R <- Exp(delta) R, tau += delta tau.  The identifiability 'mask' is the absolute Levenberg damping mu_abs in
             unit-scaled (mm, deg) coordinates: eigen-directions with data information << mu_abs are not moved, no branch;
  * checks : `count_dynamic_shape_ops` replays the forward under a TorchDispatchMode on several inputs and counts aten
             calls whose output shape varies with the data or that are data-dependent-shape / host-sync ops (nonzero,
             masked_select, unique, item, ...).  The target of the plan is 0.
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mv2 import m5_core as M
from mv2 import m5_ib as IB
from semkine import sba as SB
from semkine.lie import so3_exp, so3_log

CAPACITIES = (256, 512, 1024)
KS = (1, 2, 4, 8)
ROT_TIERS_DEG = (2.0, 5.0, 10.0)
TRANS_TIERS_MM = (5.0, 10.0)
WIN_MS = 50                                      # the S39 step window
ROI_MARGIN = 24.0                                # source px around the palm projection
TAU_S, SIG_A, C_ROB, DMAX = 0.15, 4.0, 3.0, 14.0
MU_ABS = 2.0                                     # absolute LM damping in (mm, deg) units
UNITS = torch.tensor([1e-3] * 3 + [math.pi / 180.0] * 3)
CAPS_STEP = torch.tensor([20.0] * 3 + [6.0] * 3)  # per-iteration step caps (mm, deg)
P0_DIAG = torch.tensor([1 / 20.0 ** 2] * 3 + [1 / 8.0 ** 2] * 3)
GRID = None


def grid():
    global GRID
    if GRID is None:
        GRID = SB.make_grid(torch.device("cpu"))
    return GRID


def skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[..., 0])
    return torch.stack([torch.stack([z, -v[..., 2], v[..., 1]], -1), torch.stack([v[..., 2], z, -v[..., 0]], -1),
                        torch.stack([-v[..., 1], v[..., 0], z], -1)], -2)


# ----------------------------------------------------------------------------------------------- static graph
class Tabs:
    """Static index tables of the palm contour model (built once from the topology; the graph never recomputes them)."""

    def __init__(self, g: M.Geo):
        topo = g.geo.topo
        idx = g.palm_edge.nonzero(as_tuple=True)[0]
        self.e0, self.e1 = topo.edges[idx, 0].clone(), topo.edges[idx, 1].clone()          # (E,) vertex ids of the palm edges
        f = topo.edge_f[idx]
        self.has2 = (f[:, 1] >= 0).float()                                                  # boundary (wrist cut) edges: 0
        self.f0, self.f1 = f[:, 0].clone(), torch.where(f[:, 1] >= 0, f[:, 1], f[:, 0])
        self.faces = g.mano.f.detach().cpu().long()
        self.n_edges = int(len(idx))


def edge_state(Vh: torch.Tensor, tabs: Tabs, fx, fy, cx, cy):
    """Projected vertices (778, 2), soft silhouette weight (E,) and projected end points (E, 2) x 2 of the palm edges."""
    z = Vh[:, 2].clamp(min=1e-3)
    iz = 1.0 / z
    P2 = torch.stack([fx * Vh[:, 0] * iz + cx, fy * Vh[:, 1] * iz + cy], -1)
    fa = tabs.faces

    def facing(fi):
        tri = Vh[fa[fi]]                                                                      # (E, 3, 3)
        nrm = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
        nrm = nrm / (nrm.norm(dim=-1, keepdim=True) + 1e-9)
        cen = tri.mean(1)
        ray = cen / cen.norm(dim=-1, keepdim=True)
        return torch.sigmoid(-(nrm * ray).sum(-1) / TAU_S)
    ph0, ph1 = facing(tabs.f0), facing(tabs.f1)
    w_e = (ph0 * (1 - ph1) + ph1 * (1 - ph0)) * tabs.has2
    return P2, w_e, P2[tabs.e0], P2[tabs.e1]


def linearize(Vall, wr, tabs, ev_xy, ev_w, intr, Rm, tau):
    """One association + linearisation at the state (Rm, tau): residual r (C,), weights q (C,), the analytic residual Jacobian
    a (C, 6) per (mm, deg) units, and the association (segment j, foot parameter t_j, unit normal nj)."""
    fx, fy, cx, cy = intr[0], intr[1], intr[2], intr[3]
    Rd = (Vall - wr) @ Rm.T
    Vh = wr + Rd + tau                                                                        # (778, 3)
    z = Vh[:, 2].clamp(min=1e-3)
    iz = 1.0 / z
    P2, w_e, a2, b2 = edge_state(Vh, tabs, fx, fy, cx, cy)
    ab = b2 - a2
    L2 = (ab * ab).sum(-1) + 1e-6
    xe = ev_xy[:, None, :] - a2[None]                                                         # (C, E, 2)
    tpar = ((xe * ab[None]).sum(-1) / L2[None]).clamp(0.0, 1.0)                               # (C, E)
    foot = a2[None] + tpar[..., None] * ab[None]
    dist2 = ((ev_xy[:, None, :] - foot) ** 2).sum(-1)
    cost = 0.5 * dist2 / SIG_A ** 2 - torch.log(w_e + 1e-3)[None]
    j = cost.argmin(1)                                                                        # (C,)
    t_j = tpar.gather(1, j[:, None])[:, 0]
    abj = ab[j]
    nj = torch.stack([-abj[:, 1], abj[:, 0]], -1)
    nj = nj / (nj.norm(dim=-1, keepdim=True) + 1e-6)
    footj = a2[j] + t_j[:, None] * abj
    r = ((ev_xy - footj) * nj).sum(-1)
    dj = dist2.gather(1, j[:, None])[:, 0]
    q = ev_w * w_e[j] / (1.0 + (r / C_ROB) ** 2) * torch.exp(-0.5 * dj / DMAX ** 2)
    zero = torch.zeros_like(iz)
    Jpi = torch.stack([torch.stack([fx * iz, zero, -fx * Vh[:, 0] * iz * iz], -1),
                       torch.stack([zero, fy * iz, -fy * Vh[:, 1] * iz * iz], -1)], -2)      # (778, 2, 3)
    Jv = torch.cat([Jpi, Jpi @ (-skew(Rd))], -1)                                             # (778, 2, 6): [d tau, d omega (left)]
    Jfoot = (1 - t_j)[:, None, None] * Jv[tabs.e0[j]] + t_j[:, None, None] * Jv[tabs.e1[j]]  # (C, 2, 6)
    a = -(nj[:, :, None] * Jfoot).sum(1) * UNITS                                             # (C, 6) per (mm, deg)
    return r, q, a, j, t_j, nj


def fipa_forward(Vall: torch.Tensor, wr: torch.Tensor, tabs: Tabs, ev_xy: torch.Tensor, ev_w: torch.Tensor,
                 intr: torch.Tensor, iters: int = 8) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Vall (778, 3) vertices of the initial pose (camera frame), wr (3,) wrist, tabs the static palm-edge tables, ev_xy (C, 2)
    source-pixel event positions, ev_w (C,) validity in {0, 1}, intr (4,) = (fx, fy, cx, cy) of the 640 x 480 camera.
    Returns the (R (3, 3), tau (3,)) after every iteration.  Only static-shape tensor ops (no nonzero / masked index / item).

    Contour model: the 337 palm mesh edges (both end vertices in the wrist-weight >= 0.9 set) as image segments, each with a
    soft silhouette weight  w_e = phi_0 (1 - phi_1) + phi_1 (1 - phi_0),  phi_f = sigmoid(-(n_f . ray_f) / tau_s)  (the two faces
    of the edge face opposite ways); an event takes the segment minimising  d^2 / 2 sigma_a^2 - log(w_e + eps)  (d = distance
    to the segment), the residual is the signed point-to-line distance along the segment normal."""
    units, caps, p0 = UNITS, CAPS_STEP, P0_DIAG
    Rm = torch.eye(3, dtype=Vall.dtype)
    tau = torch.zeros(3, dtype=Vall.dtype)
    eye6 = torch.eye(6, dtype=Vall.dtype)
    outs = []
    for _ in range(iters):
        r, q, a, _, _, _ = linearize(Vall, wr, tabs, ev_xy, ev_w, intr, Rm, tau)
        Aq = a.T * q[None]
        H = Aq @ a + torch.diag(p0) + MU_ABS * eye6
        g_ = Aq @ r
        L, _ = torch.linalg.cholesky_ex(H)
        step = -torch.cholesky_solve(g_[:, None], L)[:, 0]
        step = torch.minimum(torch.maximum(step, -caps), caps)
        dth = step * units
        Rm = so3_exp(dth[3:]) @ Rm
        tau = tau + dth[:3]
        outs.append((Rm, tau))
    return outs


# ----------------------------------------------------------------------------------------------- host-side preparation
class SampleGeom:
    """FK of the reference pose of one sample (GT at the window middle): all vertices and the wrist."""

    def __init__(self, g: M.Geo, sj: M.Subject, end: int):
        self.end = end
        x = torch.tensor(sj.pos51[end - WIN_MS // 2], dtype=torch.float32)[None]
        with torch.no_grad():
            V, J = g.fk(x, sj.betas)
        self.V = V[0].clone()
        self.wr = J[0, 0].clone()
        self.palm_idx = g.palm_idx


def perturb(V, wr, om0: torch.Tensor, tau0: torch.Tensor):
    """Rigid perturbation about the wrist (rotation Exp(om0)) plus translation tau0: returns (V', wr', R0)."""
    R0 = so3_exp(om0)
    return wr + (V - wr) @ R0.T + tau0, wr + tau0, R0


def event_bucket(ev_src: np.ndarray, uv_roi: np.ndarray, capacity: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """ROI-select the in-box events (ev_src (n, 2) source px, time ordered), take `capacity` of them by a deterministic
    time-stratified stride, zero pad; returns (xy (C, 2) float32, valid (C,) float32, n_in_roi)."""
    lo, hi = uv_roi.min(0) - ROI_MARGIN, uv_roi.max(0) + ROI_MARGIN
    m = (ev_src[:, 0] >= lo[0]) & (ev_src[:, 0] <= hi[0]) & (ev_src[:, 1] >= lo[1]) & (ev_src[:, 1] <= hi[1])
    sel = ev_src[m]
    n_in = len(sel)
    xy = np.zeros((capacity, 2), np.float32)
    valid = np.zeros(capacity, np.float32)
    if n_in:
        idx = np.arange(n_in) if n_in <= capacity else np.linspace(0, n_in - 1, capacity).astype(np.int64)
        xy[:len(idx)] = sel[idx]
        valid[:len(idx)] = 1.0
    return xy, valid, n_in


def window_events_src(sj: M.Subject, end: int) -> np.ndarray:
    ev, _ = sj.window_events(end - WIN_MS, end)
    gr = grid()
    return np.stack([gr.cx_src.numpy()[ev[:, 0].astype(np.int64)], gr.cy_src.numpy()[ev[:, 1].astype(np.int64)]], -1).astype(np.float32)


def cases_of(seq: str, end: int) -> List[dict]:
    out = []
    for t in ROT_TIERS_DEG:
        for ax in M.AXES:
            sg = 1 if (M.seed_of("fsign", seq, end, t, ax) & 1) else -1
            v = np.zeros(3)
            v["xyz".index(ax)] = sg * math.radians(t)
            out.append({"kind": "rot", "tier": t, "axis": ax, "om0": v, "tau0": np.zeros(3)})
    for t in TRANS_TIERS_MM:
        for ax in M.AXES:
            sg = 1 if (M.seed_of("fsign", seq, end, t, ax, "t") & 1) else -1
            v = np.zeros(3)
            v["xyz".index(ax)] = sg * t / 1000.0
            out.append({"kind": "trans", "tier": t, "axis": ax, "om0": np.zeros(3), "tau0": v})
    return out


_TABS = {}


def tabs_of(g: M.Geo) -> Tabs:
    if "t" not in _TABS:
        _TABS["t"] = Tabs(g)
    return _TABS["t"]


def run_samples(task: dict) -> List[dict]:
    """Worker: FIPA over the listed ends of one recording, every case x capacity; K in KS read from one 8-iteration run."""
    seq = task["seq"]
    g, sj = IB._geo(), IB._subj(seq)
    K = sj.K
    tabs = tabs_of(g)
    intr = torch.tensor([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=torch.float32)
    out = []
    for end in task["ends"]:
        sg = SampleGeom(g, sj, int(end))
        evs = window_events_src(sj, int(end))
        for case in cases_of(seq, int(end)):
            om0, tau0 = torch.tensor(case["om0"], dtype=torch.float32), torch.tensor(case["tau0"], dtype=torch.float32)
            V0, wr0, R0 = perturb(sg.V, sg.wr, om0, tau0)
            uv = SB.project_src(V0[g.palm_idx], K).numpy()
            for C in task.get("capacities", CAPACITIES):
                xy, valid, n_in = event_bucket(evs, uv, C)
                it = fipa_forward(V0, wr0, tabs, torch.from_numpy(xy), torch.from_numpy(valid), intr, iters=max(KS))
                rec = {"seq": seq, "end": int(end), "kind": case["kind"], "tier": case["tier"], "axis": case["axis"], "C": C,
                       "n_in_roi": n_in, "n_valid": int(valid.sum())}
                init_rot = float(torch.rad2deg(om0.norm()))
                init_tr = float(tau0.norm() * 1000)
                rec["init_rot_deg"], rec["init_trans_mm"] = init_rot, init_tr
                for k in KS:
                    Rm, tau = it[k - 1]
                    rec[f"rot_{k}"] = float(torch.rad2deg(so3_log(Rm @ R0).norm()))
                    rec[f"trans_{k}"] = float((tau0 + tau).norm() * 1000)
                out.append(rec)
    return out


# ----------------------------------------------------------------------------------------------- S39 TF root error
def s39_tf_root_errors(seq: str, every: int, seeds: Sequence[int]) -> Dict[int, Dict[str, np.ndarray]]:
    """Teacher-forced S39 root rotation / translation error on the W200 single-step samples of `seq` (prev = GT(end - 50)),
    per seed: geodesic angle (deg) of the output root against GT(end) and wrist-translation error (mm)."""
    from mv2 import common as MC
    ss = MC.SeqSamples(seq, every=every, window=MC.WINDOW)
    out = {}
    for seed in seeds:
        model, _ = MC.load_s39(seed, torch.device("cpu"))
        model.set_hand_context(torch.tensor(ss.betas).view(1, -1), torch.tensor(ss.K).view(1, 3, 3))
        rot, tr = [], []
        with torch.no_grad():
            for i0 in range(0, len(ss), 64):
                idx = np.arange(i0, min(len(ss), i0 + 64))
                X = torch.from_numpy(ss.lnes(idx))
                P = torch.from_numpy(ss.prev[idx])
                o, _, _ = MC.s39_forward_taps(model, X, P, taps=("layer3",), check=(i0 == 0))   # == model(X, P) bit for bit
                o = o.float()
                G = torch.from_numpy(ss.gt[idx])
                R = so3_exp(o[:, 3:6]) @ so3_exp(G[:, 3:6]).transpose(-1, -2)
                rot.append(torch.rad2deg(so3_log(R).norm(dim=-1)).numpy())
                tr.append(((o[:, :3] - G[:, :3]).norm(dim=-1) * 1000).numpy())
        out[seed] = {"rot_deg": np.concatenate(rot), "trans_mm": np.concatenate(tr), "ends": ss.ends.copy()}
    return out


# ----------------------------------------------------------------------------------------------- op counter
_BLACKLIST = ("nonzero", "_unique", "unique", "masked_select", "_local_scalar_dense", "argwhere", "repeat_interleave",
              "bincount", "item", "where.default", "index_put", "masked_fill_")


class _Tracer(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.log: List[Tuple[str, tuple]] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        flat = torch.utils._pytree.tree_flatten(out)[0]
        self.log.append((str(func), tuple(tuple(o.shape) for o in flat if isinstance(o, torch.Tensor))))
        return out


def trace_ops(fn, *args) -> List[Tuple[str, tuple]]:
    with _Tracer() as t:
        fn(*args)
    return t.log


def count_dynamic_shape_ops(traces: Sequence[List[Tuple[str, tuple]]]) -> dict:
    """Aten calls whose output shapes differ between traces (or whose traces differ in length / op sequence) plus calls to
    blacklisted data-dependent-shape / host-sync ops.  Returns the counts and the offending op names."""
    ref = traces[0]
    shape_var, same_seq = set(), all(len(t) == len(ref) for t in traces) and all([n for n, _ in t] == [n for n, _ in ref] for t in traces)
    if same_seq:
        for i in range(len(ref)):
            if any(t[i][1] != ref[i][1] for t in traces):
                shape_var.add((i, ref[i][0]))
    bl = [n for t in traces for n, _ in t if any(b in n for b in _BLACKLIST)]
    import collections
    hist = dict(collections.Counter(n for n, _ in ref).most_common())
    return {"op_histogram": hist, "n_ops_per_forward": len(ref), "same_op_sequence": bool(same_seq), "n_shape_varying_ops": len(shape_var),
            "n_blacklisted_calls": len(bl), "blacklisted": sorted(set(bl)), "varying": sorted({n for _, n in shape_var}),
            "count": len(shape_var) + len(bl) + (0 if same_seq else 1)}


# ----------------------------------------------------------------------------------------------- summaries
def summarize(records: List[dict], s39: Dict[int, Dict[str, np.ndarray]], ops: dict, synth: dict) -> dict:
    """Residual medians per (K, C) and tier family against the S39 teacher-forced root error and the no-op baseline."""
    import collections
    out = {"n_records": len(records), "cells": {}, "no_op": {}, "ops": ops, "synthetic": synth, "s39_tf": {}}
    med_s39 = {str(sd): {"rot_deg_median": float(np.median(v["rot_deg"])), "trans_mm_median": float(np.median(v["trans_mm"])), "n": int(len(v["rot_deg"]))}
               for sd, v in s39.items()}
    out["s39_tf"] = dict(med_s39)
    out["s39_tf"]["mean_of_seed_medians_rot_deg"] = float(np.mean([m["rot_deg_median"] for m in med_s39.values()]))
    out["s39_tf"]["mean_of_seed_medians_trans_mm"] = float(np.mean([m["trans_mm_median"] for m in med_s39.values()]))
    rot = [r for r in records if r["kind"] == "rot"]
    tr = [r for r in records if r["kind"] == "trans"]
    out["no_op"] = {"rot_deg_median": float(np.median([r["init_rot_deg"] for r in rot])) if rot else float("nan"),
                    "trans_mm_median": float(np.median([r["init_trans_mm"] for r in tr])) if tr else float("nan")}
    for C in CAPACITIES:
        for k in KS:
            rr = [r for r in rot if r["C"] == C]
            tt = [r for r in tr if r["C"] == C]
            cell = {"rot_deg_median": float(np.median([r[f"rot_{k}"] for r in rr])) if rr else float("nan"),
                    "trans_mm_median_rot_cases": float(np.median([r[f"trans_{k}"] for r in rr])) if rr else float("nan"),
                    "trans_mm_median": float(np.median([r[f"trans_{k}"] for r in tt])) if tt else float("nan"),
                    "starved_frac": float(np.mean([r["n_valid"] < 64 for r in rr])) if rr else float("nan"),
                    "full_bucket_frac": float(np.mean([r["n_valid"] == C for r in rr])) if rr else float("nan"),
                    "per_tier_rot": {str(t): float(np.median([r[f"rot_{k}"] for r in rr if r["tier"] == t])) for t in ROT_TIERS_DEG if any(r["tier"] == t for r in rr)},
                    "per_axis_rot": {ax: float(np.median([r[f"rot_{k}"] for r in rr if r["axis"] == ax])) for ax in M.AXES if any(r["axis"] == ax for r in rr)}}
            out["cells"][f"K{k}_C{C}"] = cell
    return out


def decide(summary: dict) -> dict:
    """Gate (SELECT): some configuration inside the budget K <= 4, capacity <= 512 has a median root-rotation residual
    (tier-pooled 2 / 5 / 10 deg injections) <= 0.8 x the S39 teacher-forced root error (mean over the three seeds of the
    per-seed median, same samples) AND <= 0.8 x the no-op residual (the aligner must beat doing nothing, otherwise the first
    criterion could pass vacuously), and the number of data-dependent-shape ops is 0.
    Kill: K = 8 still above the S39 teacher-forced root error at every capacity, or the fixed-K form diverges (K = 8 residual
    > 1.25 x the K = 4 residual at capacity 512), or the op count > 0.  Planted-signal failure = tool_defect."""
    ops = summary["ops"]["count"]
    syn = summary["synthetic"]
    if not syn.get("ok", False) or not summary["ops"].get("counter_self_test", True):
        return {"verdict": "tool_defect", "synthetic": syn, "counter_self_test": summary["ops"].get("counter_self_test")}
    ref = summary["s39_tf"]["mean_of_seed_medians_rot_deg"]
    noop = summary["no_op"]["rot_deg_median"]
    cells = summary["cells"]
    ok_cells = {f"K{k}_C{C}": (cells[f"K{k}_C{C}"]["rot_deg_median"] <= 0.8 * ref and cells[f"K{k}_C{C}"]["rot_deg_median"] <= 0.8 * noop)
                for k in KS for C in CAPACITIES}
    gate_cells = [f"K{k}_C{C}" for k in (1, 2, 4) for C in (256, 512)]
    passed = any(ok_cells[c] for c in gate_cells) and ops == 0      # some configuration inside the budget K <= 4, C <= 512 passes
    k8_above = all(cells[f"K8_C{C}"]["rot_deg_median"] > ref for C in CAPACITIES)
    diverge = cells["K8_C512"]["rot_deg_median"] > 1.25 * cells["K4_C512"]["rot_deg_median"]
    kill_reasons = []
    if k8_above:
        kill_reasons.append("K=8 still above the S39 TF root error at every capacity")
    if diverge:
        kill_reasons.append("fixed-K form diverges (K=8 > 1.25 x K=4 at C=512)")
    if ops > 0:
        kill_reasons.append(f"{ops} data-dependent-shape ops")
    verdict = "pass" if passed else ("kill" if kill_reasons else "descriptive")
    return {"verdict": verdict, "s39_tf_root_deg": ref, "no_op_deg": noop, "gate_cells_ok": {c: ok_cells[c] for c in gate_cells},
            "k8_ok_any_capacity": {f"C{C}": ok_cells[f"K8_C{C}"] for C in CAPACITIES}, "n_dynamic_shape_ops": ops,
            "kill_reasons": kill_reasons,
            "disposition": {"pass": "FIPA root row may be paralleled to S39's root row (single-variable M1 + root pathway arm)",
                            "kill": "palm pre-alignment has no deployable fixed-budget form",
                            "descriptive": "needs K > 4 or capacity > 512: outside the latency budget, no claim"}[verdict]}
