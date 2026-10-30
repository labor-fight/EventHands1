#!/usr/bin/env python3
"""M5 probe 4: root-block closed-loop contraction audit RCA (P3-E9-R1).

CPU closed-loop self-rollout of the frozen S39 (seeds 3407 / 3408 / 3409) on the FIT7 / SELECT base recordings with the
evaluator's protocol (W200 evidence window, 50 ms step, valid runs, first window [a, a + 199], initial state
GT(a) + `sample_init_noise`, `semkine.eval_track.track_sequence` loop), WITHOUT the DT2 adaptive filter / evidence gate
(the filter acts on the output stream, not in the loop's network call; documented deviation: absolute numbers are those of the
bare tracker `dt_dz_l3`, not of S39's filtered row).  At every step the same network is also run
  TF : prev = GT(end - 50)                                  (the E3 / teacher-forced single step)
  SH : prev = GT of a random other step of the same run     (time-shuffled previous state, control)
so that, per block b in {root, transl, fingers}:
  e_t   = closed-loop error of the output against GT(end_t)            (root: left error log(R_out R_gt^T), rad)
  tau_t = TF error                                                      (what the step does with a correct prev)
  d_t   = CL output - TF output = J_t e_{t-1}                           (the part due to the carried previous-state error)
exposure gap  : Shapley split over the three blocks of (CL - TF) mean RA-MPJPE (and abs-MPJPE), by hybrid states that take
                each block from CL or TF;
contraction   : lambda_b = sum <d_t, e_{t-1}> / sum |e_{t-1}|^2   (regressing out tau_t by construction: no AR(1) confound with
                a persistent bias), bootstrap over (recording, 20-step block) groups;
bias share    : fraction of the root block's second moment sum |e_t|^2 carried by the means over non-overlapping L-step blocks
                (L = 20 = 1 s; 5 and 60 descriptive) against the same statistic on the error series permuted over all steps of the
                recording (white-noise null with the same marginal).
"""
from __future__ import annotations

import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from mv2 import m5_core as M
from mv2 import m5_ib as IB
from semkine.lie import so3_exp, so3_log

STEP, WIN = 50, 200
BLOCK_STEPS = 20                                # 1 s of 50 ms steps: bootstrap group and the primary bias-share block
SHARE_BLOCKS = (5, 20, 60)
SEEDS = (3407, 3408, 3409)
N_BOOT = 2000
HYB = list(range(8))                            # bit 0: root from CL, bit 1: transl from CL, bit 2: fingers from CL
_MODELS: Dict[int, tuple] = {}


def _model(seed: int):
    if seed not in _MODELS:
        from mv2 import common as MC
        _MODELS[seed] = MC.load_s39(seed, torch.device("cpu"))
    return _MODELS[seed]


def _split(P: torch.Tensor, blk: int) -> torch.Tensor:
    return P[..., 0:3] if blk == 1 else (P[..., 3:6] if blk == 0 else P[..., 6:51])


def hybrid_states(cl: np.ndarray, tf: np.ndarray, mask: int) -> np.ndarray:
    """(n, 51): root (3:6) from CL if bit 0, transl (0:3) from CL if bit 1, fingers (6:51) from CL if bit 2, else TF."""
    h = tf.copy()
    if mask & 1:
        h[:, 3:6] = cl[:, 3:6]
    if mask & 2:
        h[:, 0:3] = cl[:, 0:3]
    if mask & 4:
        h[:, 6:] = cl[:, 6:]
    return h


def root_err(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n, 3) left error log(R_a R_b^T) between the root axis-angles of two states (rad)."""
    Ra, Rb = so3_exp(torch.tensor(a[:, 3:6], dtype=torch.float64)), so3_exp(torch.tensor(b[:, 3:6], dtype=torch.float64))
    return so3_log(Ra @ Rb.transpose(-1, -2)).numpy()


def joints_of(g: M.Geo, P: np.ndarray, betas: torch.Tensor, chunk: int = 2048) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(P), chunk):
            _, J = g.fk(torch.tensor(P[i:i + chunk], dtype=torch.float32), betas)
            out.append(J.numpy())
    return np.concatenate(out)


def run_sequence(task: dict) -> dict:
    """Worker: closed loop + TF + shuffled-prev of one (recording, seed); returns per-step arrays and hybrid MPJPEs."""
    from semkine import eval_track as ET
    from semkine import events as EV
    from mv2 import common as MC
    seq, seed = task["seq"], int(task["seed"])
    max_steps = task.get("max_steps")
    g = IB._geo()
    model, cfg = _model(seed)
    events, offsets, aux, pos51 = MC.load_seq(seq)
    betas = torch.tensor(np.asarray(aux["betas"], np.float32).reshape(-1)[:10])[None]
    K = torch.tensor(np.asarray(aux["camera_K"], np.float32).reshape(3, 3))
    model.set_hand_context(betas.view(1, -1), K.view(1, 3, 3))
    ev_ch = EV.event_channels(cfg)
    rng = np.random.default_rng(M.seed_of("rca", seq, seed))
    shuf_rng = np.random.default_rng(M.seed_of("rca-shuf", seq, seed))
    runs = np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)
    cols = {k: [] for k in ("end", "run", "first", "gt", "cl", "tf", "sh")}
    t0 = time.time()
    with torch.no_grad():
        for rid, (a, b) in enumerate(runs):
            ends = np.arange(a + max(STEP, WIN) - 1, b, STEP, dtype=np.int64)
            if max_steps:
                ends = ends[:max_steps]
            if not len(ends):
                continue
            prev = torch.from_numpy((pos51[a].copy() + ET.sample_init_noise(cfg, rng, 1.0)).astype(np.float32)).view(1, -1)
            perm = shuf_rng.permutation(len(ends))
            for k, end in enumerate(ends):
                x = torch.from_numpy(ET.build_lnes(events, offsets, int(end), WIN, ev_ch)).unsqueeze(0)
                gt = np.asarray(pos51[end], np.float32)
                tf_in = torch.from_numpy(np.asarray(pos51[end - STEP], np.float32)).view(1, -1)
                sh_in = torch.from_numpy(np.asarray(pos51[int(ends[perm[k]]) - STEP], np.float32)).view(1, -1)
                out = model(torch.cat([x, x, x]), torch.cat([prev, tf_in, sh_in])).float()
                cl, tf, sh = out[0:1], out[1:2], out[2:3]
                for key, v in (("end", end), ("run", rid), ("first", int(k == 0)), ("gt", gt), ("cl", cl[0].numpy()),
                               ("tf", tf[0].numpy()), ("sh", sh[0].numpy())):
                    cols[key].append(v)
                prev = cl
    if not cols["end"]:
        return {"seq": seq, "seed": seed, "n": 0}
    A = {k: np.asarray(v) for k, v in cols.items()}
    n = len(A["end"])
    # MPJPE of the eight hybrids against GT (abs and root-aligned, mm) and of the shuffled-prev control
    Jg = joints_of(g, A["gt"], betas)
    abs_h, ra_h = np.zeros((n, 8), np.float32), np.zeros((n, 8), np.float32)
    for m in HYB:
        Jh = joints_of(g, hybrid_states(A["cl"], A["tf"], m), betas)
        abs_h[:, m] = np.linalg.norm(Jh - Jg, axis=-1).mean(-1) * 1000
        ra_h[:, m] = np.linalg.norm((Jh - Jh[:, :1]) - (Jg - Jg[:, :1]), axis=-1).mean(-1) * 1000
    Js = joints_of(g, A["sh"], betas)
    res = {"seq": seq, "seed": seed, "n": n, "secs": time.time() - t0, "end": A["end"], "run": A["run"], "first": A["first"],
           "abs_h": abs_h, "ra_h": ra_h,
           "sh_abs": np.linalg.norm(Js - Jg, axis=-1).mean(-1) * 1000,
           "sh_ra": np.linalg.norm((Js - Js[:, :1]) - (Jg - Jg[:, :1]), axis=-1).mean(-1) * 1000,
           # block error vectors
           "e_root": root_err(A["cl"], A["gt"]), "tau_root": root_err(A["tf"], A["gt"]), "d_root": root_err(A["cl"], A["tf"]),
           "e_tr": A["cl"][:, :3] - A["gt"][:, :3], "tau_tr": A["tf"][:, :3] - A["gt"][:, :3], "d_tr": A["cl"][:, :3] - A["tf"][:, :3],
           "e_f": A["cl"][:, 6:] - A["gt"][:, 6:], "tau_f": A["tf"][:, 6:] - A["gt"][:, 6:], "d_f": A["cl"][:, 6:] - A["tf"][:, 6:],
           "e_root_sh": root_err(A["sh"], A["gt"]), "finite": bool(np.isfinite(A["cl"]).all() and np.isfinite(A["tf"]).all())}
    # the TF step must equal the single-step E3 forward bit for bit
    return res


# ----------------------------------------------------------------------------------------------- statistics
def shapley_root(v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-step Shapley value of the root block and the total gap from hybrid metrics v (n, 8) (bit 0 root, 1 transl, 2 fingers):
    phi_r = 1/3 [v(r)-v(0)] + 1/6 [v(rt)-v(t)] + 1/6 [v(rf)-v(f)] + 1/3 [v(rtf)-v(tf)];  gap = v(rtf) - v(0)."""
    r, t, f = 1, 2, 4
    phi = (v[:, r] - v[:, 0]) / 3 + (v[:, r | t] - v[:, t]) / 6 + (v[:, r | f] - v[:, f]) / 6 + (v[:, r | t | f] - v[:, t | f]) / 3
    return phi, v[:, 7] - v[:, 0]


def shapley_all(v: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-step Shapley values of the three blocks (they sum to the gap exactly)."""
    out = {}
    for name, bit in (("root", 1), ("transl", 2), ("fingers", 4)):
        others = [b for b in (1, 2, 4) if b != bit]
        o1, o2 = others
        out[name] = ((v[:, bit] - v[:, 0]) / 3 + (v[:, bit | o1] - v[:, o1]) / 6 + (v[:, bit | o2] - v[:, o2]) / 6
                     + (v[:, 7] - v[:, o1 | o2]) / 3)
    return out


def valid_steps(first: np.ndarray, run: np.ndarray) -> np.ndarray:
    """Steps whose previous step is in the same run (the first step of a run uses the initial state, not a loop state)."""
    return first == 0


def group_ids(res: dict) -> np.ndarray:
    """Bootstrap group of each step: (recording, run, 20-step block) as a string id."""
    run, first = res["run"], res["first"]
    out = []
    k = {}
    for i in range(len(run)):
        r = int(run[i])
        k[r] = 0 if first[i] else k[r] + 1
        out.append(f"{res['seq']}|{r}|{k[r] // BLOCK_STEPS}")
    return np.asarray(out)


def collect(results: List[dict], select: Sequence[str]) -> Dict[str, dict]:
    """Per-step stacked arrays per set ('SELECT', 'FIT7', 'ALL') over all seeds."""
    sets = {"SELECT": [], "FIT7": []}
    for r in results:
        if r.get("n", 0) == 0:
            continue
        sets[IB.subject_set(r["seq"], select)].append(r)
    sets["ALL"] = sets["SELECT"] + sets["FIT7"]
    return sets


def per_group_stats(rs: List[dict]) -> dict:
    """Group-additive sufficient statistics (one row per (recording, run, block), pooled over the seeds)."""
    rows = {}

    def add(gid, key, val):
        rows.setdefault(gid, {}).setdefault(key, 0.0)
        rows[gid][key] += val

    rng = np.random.default_rng(M.SEED)
    for r in rs:
        v = valid_steps(r["first"], r["run"])
        gid = group_ids(r)
        sh = shapley_all(r["ra_h"]), shapley_all(r["abs_h"])
        gap_ra, gap_abs = r["ra_h"][:, 7] - r["ra_h"][:, 0], r["abs_h"][:, 7] - r["abs_h"][:, 0]
        # lambda: previous-step CL error within the same run
        e_prev = np.zeros_like(r["e_root"])
        e_prev[1:] = r["e_root"][:-1]
        e_prev_tr = np.zeros_like(r["e_tr"])
        e_prev_tr[1:] = r["e_tr"][:-1]
        e_prev_f = np.zeros_like(r["e_f"])
        e_prev_f[1:] = r["e_f"][:-1]
        for i in np.flatnonzero(v):
            g_ = gid[i]
            add(g_, "n", 1.0)
            add(g_, "gap_ra", float(gap_ra[i]))
            add(g_, "gap_abs", float(gap_abs[i]))
            for nm in ("root", "transl", "fingers"):
                add(g_, f"phi_ra_{nm}", float(sh[0][nm][i]))
                add(g_, f"phi_abs_{nm}", float(sh[1][nm][i]))
            add(g_, "lam_root_num", float(r["d_root"][i] @ e_prev[i]))
            add(g_, "lam_root_den", float(e_prev[i] @ e_prev[i]))
            add(g_, "lam_tr_num", float(r["d_tr"][i] @ e_prev_tr[i]))
            add(g_, "lam_tr_den", float(e_prev_tr[i] @ e_prev_tr[i]))
            add(g_, "lam_f_num", float(r["d_f"][i] @ e_prev_f[i]))
            add(g_, "lam_f_den", float(e_prev_f[i] @ e_prev_f[i]))
            add(g_, "e2_root", float(r["e_root"][i] @ r["e_root"][i]))
            add(g_, "tau2_root", float(r["tau_root"][i] @ r["tau_root"][i]))
            add(g_, "ra_cl", float(r["ra_h"][i, 7]))
            add(g_, "ra_tf", float(r["ra_h"][i, 0]))
            add(g_, "abs_cl", float(r["abs_h"][i, 7]))
            add(g_, "abs_tf", float(r["abs_h"][i, 0]))
            add(g_, "ra_sh", float(r["sh_ra"][i]))
            add(g_, "abs_sh", float(r["sh_abs"][i]))
    return rows


def bias_share(rs: List[dict], L: int, null: bool = False, seed: int = M.SEED) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Share of the root-block second moment carried by the means over non-overlapping L-step blocks (inside runs, valid steps
    only), per (recording, run, 20-step group) for the bootstrap.  The null permutes the root errors over ALL valid steps of the
    recording (not inside a run: a within-run shuffle keeps the run mean and so keeps a run-constant bias): white noise with
    the same marginal distribution.  Returns (share, group ids, numerators, denominators)."""
    rng = np.random.default_rng(seed)
    gids, num, den = [], [], []
    for r in rs:
        v = valid_steps(r["first"], r["run"])
        gid = group_ids(r)
        e_all = r["e_root"].copy()
        if null:
            iv = np.flatnonzero(v)
            e_all[iv] = e_all[iv[rng.permutation(len(iv))]]
        for rid in np.unique(r["run"]):
            idx = np.flatnonzero((r["run"] == rid) & v)
            if not len(idx):
                continue
            e = e_all[idx]
            for b0 in range(0, len(idx), L):
                blk = e[b0:b0 + L]
                n = len(blk)
                gids.append(gid[idx[b0]])
                num.append(n * float(blk.mean(0) @ blk.mean(0)))
                den.append(float((blk * blk).sum()))
    num, den = np.asarray(num), np.asarray(den)
    return float(num.sum() / max(den.sum(), 1e-30)), np.asarray(gids), num, den


def _ratio_boot(gids: np.ndarray, num: np.ndarray, den: np.ndarray, n_boot: int = N_BOOT, seed: int = M.SEED):
    ug, inv = np.unique(gids, return_inverse=True)
    gn = np.bincount(inv, weights=num, minlength=len(ug))
    gd = np.bincount(inv, weights=den, minlength=len(ug))
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(ug), len(ug))
        bs[i] = gn[pick].sum() / max(gd[pick].sum(), 1e-30)
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def summarize_set(rs: List[dict], n_boot: int = N_BOOT) -> dict:
    """Everything of one recording set (pooled over seeds)."""
    if not rs:
        return {}
    G = per_group_stats(rs)
    gids = np.array(list(G.keys()))

    def col(k):
        return np.array([G[g].get(k, 0.0) for g in gids])

    out = {"n_steps": int(col("n").sum()), "n_groups": int(len(gids)), "seeds": sorted({int(r["seed"]) for r in rs})}
    out["ra_mm"] = {"closed_loop": float(col("ra_cl").sum() / col("n").sum()), "teacher_forced": float(col("ra_tf").sum() / col("n").sum()),
                    "shuffled_prev": float(col("ra_sh").sum() / col("n").sum())}
    out["abs_mm"] = {"closed_loop": float(col("abs_cl").sum() / col("n").sum()), "teacher_forced": float(col("abs_tf").sum() / col("n").sum()),
                     "shuffled_prev": float(col("abs_sh").sum() / col("n").sum())}
    for metric in ("ra", "abs"):
        gap = col(f"gap_{metric}")
        tot = float(gap.sum())
        d = {"gap_mm": tot / col("n").sum()}
        for nm in ("root", "transl", "fingers"):
            ph = col(f"phi_{metric}_{nm}")
            share = float(ph.sum() / tot) if abs(tot) > 1e-9 else float("nan")
            lo, hi = _ratio_boot(gids, ph, gap, n_boot) if abs(tot) > 1e-9 else (float("nan"), float("nan"))
            d[nm] = {"share": share, "ci95": [lo, hi], "gap_mm": float(ph.sum() / col("n").sum())}
        out[f"exposure_gap_{metric}"] = d
    for nm, key in (("root", "root"), ("transl", "tr"), ("fingers", "f")):
        num, den = col(f"lam_{key}_num"), col(f"lam_{key}_den")
        lam = float(num.sum() / max(den.sum(), 1e-30))
        lo, hi = _ratio_boot(gids, num, den, n_boot)
        out.setdefault("lambda", {})[nm] = {"value": lam, "ci95": [lo, hi]}
    out["root_error_rms_rad"] = {"closed_loop": math.sqrt(float(col("e2_root").sum() / col("n").sum() / 3)),
                                 "teacher_forced": math.sqrt(float(col("tau2_root").sum() / col("n").sum() / 3))}
    out["bias_share"] = {}
    for L in SHARE_BLOCKS:
        s, gi, nu, de = bias_share(rs, L)
        sn, gin, nun, den_ = bias_share(rs, L, null=True)
        lo, hi = _ratio_boot(gi, nu, de, n_boot)
        out["bias_share"][str(L)] = {"share": s, "ci95": [lo, hi], "null_share": sn, "excess_over_null": s - sn}
    return out


def decide(summary: dict) -> dict:
    """Gate (SELECT, 3 seeds pooled): pass  = root exposure-gap share (Shapley, RA-MPJPE) >= 0.5  and  lambda_root bootstrap
    95 % upper bound < 1  and  bias share of the root-block second moment (1 s blocks) >= 0.5  -> the root-block-restricted
    self-previous-state training arm is worth a GPU debug run;  kill = gap share < 0.5 (exposure not in the root) or
    lambda_root upper bound >= 1 (the root block does not contract);  else descriptive (-> M7 provenance check).
    A degenerate gap (|CL - TF| < 0.05 mm) or non-finite outputs are not a pass."""
    s = summary.get("SELECT", {})
    if not s:
        return {"verdict": "tool_defect", "reason": "no SELECT rollout"}
    if not summary.get("validity", {}).get("ok", True):
        return {"verdict": "tool_defect", "validity": summary.get("validity")}
    gap = s["exposure_gap_ra"]["gap_mm"]
    if not (abs(gap) >= 0.05):
        return {"verdict": "descriptive", "reason": f"degenerate exposure gap {gap:.4f} mm", "gap_mm": gap}
    share = s["exposure_gap_ra"]["root"]["share"]
    lam_ub = s["lambda"]["root"]["ci95"][1]
    bias = s["bias_share"][str(BLOCK_STEPS)]["share"]
    kill_reasons = []
    if share < 0.5:
        kill_reasons.append("root exposure-gap share < 0.5")
    if lam_ub >= 1.0:
        kill_reasons.append("lambda_root 95% upper bound >= 1")
    passed = share >= 0.5 and lam_ub < 1.0 and bias >= 0.5
    verdict = "pass" if passed else ("kill" if kill_reasons else "descriptive")
    return {"verdict": verdict, "root_gap_share_ra": share, "root_gap_share_ra_ci95": s["exposure_gap_ra"]["root"]["ci95"],
            "lambda_root": s["lambda"]["root"]["value"], "lambda_root_ub95": lam_ub, "bias_share_1s": bias,
            "bias_share_1s_null": s["bias_share"][str(BLOCK_STEPS)]["null_share"], "kill_reasons": kill_reasons,
            "disposition": {"pass": "root-block-restricted self-prev-state GPU arm (S39 recipe + TRACK.SELF_P on the root block) may be debugged (single seed 3407, stop at 2000)",
                            "kill": "exposure is not in a contracting-resistant root block: no block-restricted self-rollout arm; go to M7 provenance",
                            "descriptive": "bias share < 0.5: persistence is not bias-dominated; go to M7 provenance"}[verdict]}
