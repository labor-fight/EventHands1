#!/usr/bin/env python3
"""S44 P0.8 (under-response counterfactual) and P0.9 (closed-loop state error vs the training noise classes),
docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019 section 2. GT and the P0.10 control trajectories only; no model.

P0.8: per segment x_t = x_0 + k (GT_t - GT_0), x_0 the protocol's noisy initial state (the control's recorded state of
the segment's first step), GT_0 = GT at the segment start; RA of the protocol steps, k in {0.65, 0.71, 1.0}.
P0.9: the state each non-first step was conditioned on (= the previous output) against GT(end - 50): per finger-joint
axis-angle error magnitude, translation / root errors, lag-1 autocorrelation within segments, mean inter-joint
correlation of the finger error components; three S39 seeds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "dt"))
from s44 import evflow as EF  # noqa: E402
import filter_eval as FE  # noqa: E402
from mano_layer import ManoLayer  # noqa: E402

EX = FE.EX
KS = (0.65, 0.71, 1.0)
S39_LOCAL = 10.27


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    mano = ManoLayer(str(EF.MANO_NPZ), add_mean=False).to(dev).eval()
    out = {"P08": {}, "P09": {}}
    ctl = {sd: np.load(REPO / f"outputs/s44/p10/s39_s{sd}_control.npz") for sd in (3407, 3408, 3409)}
    for s, d in EF.split_seqs("val_core"):
        sq = EF.Seq(s, d)
        c = ctl[3407]
        ends, runs, prev0 = c[f"{s}|end"], c[f"{s}|run"], c[f"{s}|prev"]
        res = {}
        for k in KS:
            pred = np.zeros((len(ends), 51), np.float32)
            for i, (e, rid) in enumerate(zip(ends, runs)):
                a = int(sq.runs[rid][0])
                first = int(np.flatnonzero(runs == rid)[0])
                pred[i] = prev0[first] + k * (sq.pos51[int(e)] - sq.pos51[a])
            r = {"pred": pred, "gt": sq.pos51[ends].astype(np.float32), "betas": sq.betas.astype(np.float32)}
            m = EX.per_step_metrics(mano, r, dev)
            res[str(k)] = float(m["mpjpe_ra_mm"].mean())
        out["P08"][s] = res
        per_seed = {}
        for sd, cc in ctl.items():
            ends, runs, prev = cc[f"{s}|end"], cc[f"{s}|run"], cc[f"{s}|prev"].astype(np.float64)
            keep = np.array([e - EF.STEP >= sq.runs[rr][0] for e, rr in zip(ends, runs)])
            gt = sq.pos51[ends[keep] - EF.STEP].astype(np.float64)
            err = prev[keep] - gt
            fe = err[:, EF.FINGER].reshape(-1, 15, 3)
            mag = np.linalg.norm(fe, axis=-1)                                     # (n, 15) rad
            rk = runs[keep]
            same = rk[1:] == rk[:-1]
            def lag1(x):
                x0, x1 = x[:-1][same], x[1:][same]
                x0, x1 = x0 - x0.mean(0), x1 - x1.mean(0)
                return float((x0 * x1).sum() / np.sqrt((x0 ** 2).sum() * (x1 ** 2).sum()))
            C = np.corrcoef(err[:, EF.FINGER].T)
            off = C[~np.eye(45, dtype=bool)]
            per_seed[sd] = {"finger_err_rad_median": float(np.median(mag)), "finger_err_rad_p90": float(np.percentile(mag, 90)),
                            "finger_err_per_joint_median_rad": np.median(mag, 0).tolist(),
                            "transl_err_mm_median": float(np.median(np.linalg.norm(err[:, :3], axis=-1)) * 1000),
                            "root_err_rad_median": float(np.median(np.linalg.norm(err[:, 3:6], axis=-1))),
                            "lag1_autocorr_finger": lag1(err[:, EF.FINGER]), "lag1_autocorr_root": lag1(err[:, 3:6]),
                            "mean_abs_interjoint_corr": float(np.abs(off).mean())}
        out["P09"][s] = per_seed
    # training noise classes, per finger joint |N(0, sigma^2 I_3)|: median sigma * 1.538
    out["P09"]["train_noise_joint_median_rad"] = {"small(0.05)": 0.05 * 1.5382, "large(0.3)": 0.3 * 1.5382}
    loc = out["P08"]["zgz_local"]["0.71"]
    out["P08"]["gate"] = {"k071_local_mm": loc, "pass_ge_0.7xS39": loc >= 0.7 * S39_LOCAL,
                          "bias_lt_0.4xS39": loc < 0.4 * S39_LOCAL}
    p = REPO / "outputs/s44/p08_p09.json"
    p.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["P08"], indent=1))
    print(json.dumps({s: {sd: {k: v for k, v in e.items() if not isinstance(v, list)} for sd, e in out["P09"][s].items()}
                      for s in ("zgz_global", "zgz_local")}, indent=1))


if __name__ == "__main__":
    main()
