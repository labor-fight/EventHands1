#!/usr/bin/env python3
"""K-star KS1 (PE'): fusion ceiling of an increment measurement on top of S39 (docs/EXPERIMENT_RECORDS.md#kstar-prereg-20261025).

Question: if a fast path measured the 50 ms state increment perfectly (or with normal correlation r), how much could a
grouped constant-gain fusion with the S39 closed-loop states lower RA-MPJPE on zgz? Inputs are only the stored S39
three-seed archives (`outputs/s44/p10/s39_s{seed}_control.npz`) and the GT. Within a run:

    x_0 = a_0;   x_k = a_k + K_g * (x_{k-1} + d_k - a_k)   per DOF group g in {transl, root, fingers}

with a_k the S39 state and d_k = (gt_k - gt_{k-1}) + noise, noise_dof ~ N(0, (kappa * rms_dof)^2): kappa is the relative
error of the measured increment (an R_art-equivalent level: 0 oracle; 0.80 ~ XVM P3n c = 0.9; 0.87 ~ c = 0.8, the r* line;
0.94 = the frozen patch head's measured E3 local R_art). Mapping vertex-normal correlation to DOF noise through
sqrt(1/r^2 - 1) would be optimistic (K-Battle R2 P6), so the levels are set in R_art units.
The gains are chosen on the scored data itself (root / fingers on RA-MPJPE, transl on wrist error), so the result is an
optimistic upper bound and is used one way only: a bound below the kill line ends the accuracy claim.

Controls: V0 the evaluator's RA of the archive equals the stored RA; V1 K = 0 reproduces S39; negative = the same search
with the increments permuted in time (gain-tuning optimism); positive = an anchor that is GT + white noise of S39's
per-DOF error RMS (the bound must be large).

    python tools/kstar/fusion_ceiling.py --out outputs/kstar/s1_fusion_ceiling
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kstar import common as KC  # noqa: E402
from xvm import common as XC  # noqa: E402

GROUPS = {"transl": slice(0, 3), "root": slice(3, 6), "fingers": slice(6, 51)}
GRID = np.round(np.arange(0.0, 0.951, 0.05), 2)
KAPPAS = (0.0, 0.5, 0.80, 0.87, 0.94)
CV_KAPPAS = (0.0, 0.87)
N_FOLDS = 5
NOISE_SEED = 777


def fuse(a, d, first, K):
    """a (n,51) S39 states, d (n,51) increments, first (n,) bool run starts, K (51,) per-DOF gain."""
    x = np.empty_like(a)
    for k in range(len(a)):
        if first[k]:
            x[k] = a[k]
        else:
            x[k] = a[k] + K * (x[k - 1] + d[k] - a[k])
    return x


class Scorer:
    def __init__(self, betas):
        self.dev = torch.device("cpu")
        self.mano = XC.mano32(self.dev)
        self.betas = betas

    def __call__(self, seq, pred, gt):
        return XC.score_ra(self.mano, pred, gt, self.betas[seq], self.dev)


def gains_vec(kt, kr, kf):
    K = np.zeros(51)
    K[GROUPS["transl"]], K[GROUPS["root"]], K[GROUPS["fingers"]] = kt, kr, kf
    return K


def search(score, seq, a, d, gt, first, mask=None):
    """Coordinate search over the grid: (root, fingers) on mean RA-MPJPE, transl on mean wrist error; scored on the
    endpoints in `mask` (all by default). Runs are independent (the fusion restarts at every run start)."""
    mask = np.ones(len(a), bool) if mask is None else mask
    best = {"kr": 0.0, "kf": 0.0, "kt": 0.0}
    for _ in range(2):
        for key in ("kf", "kr"):
            vals = []
            for g in GRID:
                trial = dict(best, **{key: g})
                x = fuse(a, d, first, gains_vec(0.0, trial["kr"], trial["kf"]))
                vals.append(score(seq, x, gt)["mpjpe_ra_mm"][mask].mean())
            best[key] = float(GRID[int(np.argmin(vals))])
    vals = []
    for g in GRID:
        x = fuse(a, d, first, gains_vec(g, best["kr"], best["kf"]))
        vals.append(score(seq, x, gt)["wrist_mm"][mask].mean())
    best["kt"] = float(GRID[int(np.argmin(vals))])
    x = fuse(a, d, first, gains_vec(best["kt"], best["kr"], best["kf"]))
    return best, x


def error_spectrum(e, first):
    """Per-group lag-1 autocorrelation of the S39 error within runs and the slow / fast energy split
    (slow = per-run mean + the 5-endpoint moving average about it; fast = the remainder)."""
    out = {}
    for gname, sl in GROUPS.items():
        eg = e[:, sl]
        num = den = 0.0
        slow = fast = 0.0
        run_id = np.cumsum(first)
        for r in np.unique(run_id):
            m = run_id == r
            x = eg[m] - eg[m].mean(0)
            if len(x) >= 2:
                num += float((x[1:] * x[:-1]).sum())
                den += float((x * x).sum())
            k = 5
            if len(x) >= k:
                ker = np.ones(k) / k
                sm = np.stack([np.convolve(x[:, j], ker, mode="same") for j in range(x.shape[1])], 1)
            else:
                sm = np.repeat(x.mean(0, keepdims=True), len(x), 0)
            slow += float((eg[m].mean(0) ** 2).sum() * m.sum() + (sm ** 2).sum())
            fast += float(((x - sm) ** 2).sum())
        out[gname] = {"lag1": num / den if den > 0 else None, "slow_energy": slow, "fast_energy": fast,
                      "slow_fast_rms_ratio": float(np.sqrt(slow / fast)) if fast > 0 else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true", help="engineering smoke: 1 seed, 60 steps, no result file")
    a = ap.parse_args()
    torch.set_num_threads(8)
    out = Path(a.out)
    if not a.smoke:
        KC.verify_protocol()
        if (out / "result.json").exists():
            raise SystemExit(f"{out}/result.json exists; preserve it")
    betas = KC.zgz_betas()
    score = Scorer(betas)
    seeds = KC.SEEDS[:1] if a.smoke else KC.SEEDS
    res = {"grid": GRID.tolist(), "kappas": list(KAPPAS), "cv_kappas": list(CV_KAPPAS), "noise_seed": NOISE_SEED,
           "per": {}, "validity": {}}
    pooled = {}
    t0 = time.time()
    for seed in seeds:
        arc = KC.archive(seed)
        for seq in KC.ZGZ:
            A = arc[seq]
            n = 60 if a.smoke else len(A["pred"])
            pred = A["pred"][:n].astype(np.float64)
            gt = A["gt"][:n].astype(np.float64)
            run = A["run"][:n]
            first = np.r_[True, run[1:] != run[:-1]]
            base = score(seq, pred, gt)
            v0 = float(np.abs(base["mpjpe_ra_mm"] - A["mpjpe_ra_mm"][:n]).max())
            ra0 = float(base["mpjpe_ra_mm"].mean())
            inc = np.zeros_like(gt)
            inc[~first] = (gt[1:] - gt[:-1])[~first[1:]]
            rms = np.sqrt((inc[~first] ** 2).mean(0))
            x_k0 = fuse(pred, inc, first, np.zeros(51))
            v1 = float(np.abs(x_k0 - pred).max())
            key = f"s{seed}|{seq}"
            ent = {"n": int(n), "ra_s39": ra0, "V0_max_abs_mm": v0, "V1_max_abs": v1,
                   "error_spectrum": error_spectrum(pred - gt, first), "levels": {}}
            rng = np.random.default_rng(NOISE_SEED + seed)
            fold_of = {r: i % N_FOLDS for i, r in enumerate(np.unique(run))}
            fstep = np.array([fold_of[r] for r in run])
            for kap in KAPPAS:
                d = inc + rng.standard_normal(inc.shape) * kap * rms
                d[first] = 0.0
                K, x = search(score, seq, pred, d, gt, first)
                sc = score(seq, x, gt)
                delta = sc["mpjpe_ra_mm"] - base["mpjpe_ra_mm"]
                lv = {"gains": K, "ra": float(sc["mpjpe_ra_mm"].mean()), "delta_ra": float(delta.mean()),
                      "delta_wrist": float((sc["wrist_mm"] - base["wrist_mm"]).mean())}
                if kap in CV_KAPPAS:                                  # run-held-out gains (5 folds of runs)
                    dcv = np.zeros(len(pred))
                    cv_ok = True
                    for f in range(N_FOLDS):
                        test = fstep == f
                        if not test.any():
                            continue
                        if test.all():
                            cv_ok = False
                            break
                        Kf, _ = search(score, seq, pred, d, gt, first, mask=~test)
                        xf = fuse(pred, d, first, gains_vec(Kf["kt"], Kf["kr"], Kf["kf"]))
                        dcv[test] = (score(seq, xf, gt)["mpjpe_ra_mm"] - base["mpjpe_ra_mm"])[test]
                    lv["delta_ra_cv"] = float(dcv.mean()) if cv_ok else None
                ent["levels"][str(kap)] = lv
                pooled.setdefault((seq, kap), []).append((delta, run, seed))
            # negative control: increments permuted in time (same marginal), gains re-searched
            perm = np.random.default_rng(NOISE_SEED).permutation(np.nonzero(~first)[0])
            dneg = np.zeros_like(inc)
            dneg[np.nonzero(~first)[0]] = inc[perm]
            Kn, xn = search(score, seq, pred, dneg, gt, first)
            ent["negative_shuffled"] = {"gains": Kn,
                                        "delta_ra": float((score(seq, xn, gt)["mpjpe_ra_mm"] - base["mpjpe_ra_mm"]).mean())}
            # positive control: white-noise anchor with S39's per-DOF error RMS
            erms = np.sqrt(((pred - gt) ** 2).mean(0))
            apos = gt + np.random.default_rng(NOISE_SEED + 1).standard_normal(gt.shape) * erms
            bpos = score(seq, apos, gt)["mpjpe_ra_mm"]
            Kp, xp = search(score, seq, apos, inc, gt, first)
            ent["positive_white_anchor"] = {"gains": Kp, "ra_anchor": float(bpos.mean()),
                                            "delta_ra": float((score(seq, xp, gt)["mpjpe_ra_mm"] - bpos).mean())}
            res["per"][key] = ent
            if a.smoke:                                   # engineering smoke prints no effect numbers
                continue
            print(key, f"V0 {v0:.2e} V1 {v1:.1e} RA {ra0:.3f}",
                  {k: round(ent["levels"][str(k)]["delta_ra"], 3) for k in KAPPAS},
                  f"neg {ent['negative_shuffled']['delta_ra']:.3f} pos {ent['positive_white_anchor']['delta_ra']:.3f}",
                  f"{time.time() - t0:.0f} s", flush=True)
    summary = {}
    for (seq, kap), rows in pooled.items():
        vals = np.concatenate([d for d, _, _ in rows])
        bids = np.concatenate([KC.blocks_of(run) + 100000 * i for i, (_, run, _) in enumerate(rows)])
        ent = KC.block_ci(vals, bids)
        ent["per_seed_mean"] = [float(d.mean()) for d, _, _ in rows]
        cv = [res["per"][f"s{sd}|{seq}"]["levels"][str(kap)].get("delta_ra_cv") for _, _, sd in rows]
        ent["per_seed_cv_mean"] = cv if all(c is not None for c in cv) else None
        summary[f"{seq}|{kap}"] = ent
    res["summary_delta_ra"] = summary
    res["validity"] = {"V0_ok": all(v["V0_max_abs_mm"] < 1e-3 for v in res["per"].values()),
                       "V1_ok": all(v["V1_max_abs"] == 0.0 for v in res["per"].values())}
    res["wall_s"] = time.time() - t0
    if a.smoke:
        print("SMOKE", res["validity"], f"{res['wall_s']:.0f} s")
        return
    KC.save_json(out / "result.json", res)
    KC.receipt(out, {"probe": "S1 fusion ceiling"}, scripts=(__file__,), inputs=KC.archive_files())


if __name__ == "__main__":
    main()
