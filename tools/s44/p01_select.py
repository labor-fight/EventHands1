#!/usr/bin/env python3
"""S44 P0.1 hyperparameter selection on the TRAINING subjects only (docs/EXPERIMENT_RECORDS.md#s44-prereg-20261019
section 1 + deviation 1). zgz is never read here.

One (r, dfit) per invocation (the measurements depend only on these); for every step all (eps cap, kappa, lam_p)
solves are evaluated. Selection set: the 18 base sequences {subj}_{local,global}, every 4th protocol step (the first
step of a segment has no state 50 ms earlier inside the segment and is skipped), prev = GT(end - 50) + the training
"small" noise class (5 mm, 0.05 rad, 0.05 rad; one rng seeded 0 over the sequences in order). Deployable owners and
the Jacobian are computed at that noisy prev. Output: sums of e_art (finger-articulation increment error, mm) and of
e_zero per category and config.

    python tools/s44/p01_select.py --r 2 --dfit 30 --out outputs/s44/p01_select/r2_d30.json
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from s44 import evflow as EF  # noqa: E402

KAPPAS = (0.3, 1.0)
LAMS = (0.01, 0.1, 1.0, 10.0)
NOISE_T, NOISE_R, NOISE_P = 0.005, 0.05, 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--dfit", type=float, required=True, help="ms")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: steps per sequence")
    ap.add_argument("--mem", type=int, default=EF.W_MEM, help="SAE memory ms (P0b round)")
    ap.add_argument("--eps-grid", default="", help="comma list overriding EF.EPS_PX_GRID (P0b round)")
    ap.add_argument("--lam-grid", default="", help="comma list overriding LAMS (P0b round)")
    ap.add_argument("--layer", choices=("deploy", "oracle"), default="deploy",
                    help="oracle: post-hoc diagnostic, prev = GT(end - 50) exactly and XAG GT-source owners")
    ap.add_argument("--fix-root-gt", action="store_true",
                    help="diagnostic (oracle layer): translation / root rates given from GT, only the 45 finger rates solved")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    mano = EF.Mano64(dev)
    prec, _ = EF.rate_prior(dev)
    eps_grid = tuple(float(v) for v in a.eps_grid.split(",")) if a.eps_grid else EF.EPS_PX_GRID
    lams = tuple(float(v) for v in a.lam_grid.split(",")) if a.lam_grid else LAMS
    combos = list(itertools.product(eps_grid, KAPPAS, lams))
    rng = np.random.default_rng(0)
    acc = {}
    t0 = time.time()
    n_steps = 0
    for name, d in EF.training_base_seqs():
        cat = "local" if "_local" in name else "global"
        sq = EF.Seq(name, d)
        intr = sq.intrinsics()
        betas = torch.tensor(sq.betas, dtype=torch.float64, device=dev)[None]
        steps = [s for s in sq.steps() if s[2] - EF.STEP >= s[1]][::a.stride]
        if a.max_steps:
            steps = steps[:a.max_steps]
        A_sum = acc.setdefault(cat, {"e_art": np.zeros(len(combos)), "e_zero": 0.0, "n": 0,
                                     "n_valid_meas": np.zeros(len(eps_grid)), "n_meas": 0})
        for rid, run_a, end in steps:
            gp = sq.pos51[end - EF.STEP].astype(np.float64)
            ge = sq.pos51[end].astype(np.float64)
            noise = np.zeros(51)
            noise[0:3] = rng.standard_normal(3) * NOISE_T
            noise[3:6] = rng.standard_normal(3) * NOISE_R
            noise[6:] = rng.standard_normal(45) * NOISE_P
            prev = torch.tensor(gp + (0.0 if a.layer == "oracle" else noise), dtype=torch.float64, device=dev)[None]
            meas = EF.step_measurements(sq, run_a, end, a.r, a.dfit, dev, mem_ms=a.mem)
            incs = torch.zeros(len(combos), 51, dtype=torch.float64, device=dev)
            A_sum["n_meas"] += meas["n_meas"]
            if meas["n_valid"] > 0:
                _, uv, z, J = mano.jacobian(prev, betas, intr)
                if a.layer == "oracle":
                    ids, w, _ = EF.owners_oracle(sq, meas["rows"].cpu().numpy(), mano.faces)
                else:
                    vis = EF.visible(uv[0], z[0])
                    ids, w = EF.owners_deploy(meas["px"], uv[0], vis)
                A, mass = EF.rows_A(J[0], meas["n"], ids, w)
                for ci, cap in enumerate(eps_grid):
                    sel = (meas["eps_px"] <= cap) & (mass > 1e-6)
                    A_sum["n_valid_meas"][ci] += int(sel.sum())
                for k, (cap, kap, lam) in enumerate(combos):
                    sel = (meas["eps_px"] <= cap) & (mass > 1e-6)
                    if int(sel.sum()) == 0:
                        continue
                    if a.fix_root_gt:
                        rate_gt = torch.tensor((ge - gp) / (EF.STEP * 1e-3), dtype=torch.float64, device=dev)
                        s_res = meas["s"][sel] - A[sel][:, :6] @ rate_gt[:6]
                        thf, _ = EF.solve_rate(A[sel][:, 6:], s_res, mass[sel], kap, lam, prec[6:, 6:])
                        th = torch.cat([rate_gt[:6], thf])
                    else:
                        th, _ = EF.solve_rate(A[sel], meas["s"][sel], mass[sel], kap, lam, prec)
                    incs[k] = th * (EF.STEP * 1e-3)
            P = torch.cat([prev, prev.expand(len(combos), -1).clone()], 0)
            P[1:, EF.FINGER] += incs[:, EF.FINGER]
            gpe = torch.tensor(np.stack([gp, ge]), dtype=torch.float64, device=dev)
            jh = mano.fk_hand(torch.cat([P, gpe]), betas)                          # (2 + C + 1? , 21, 3)
            j_prev, j_est, j_gp, j_ge = jh[0], jh[1:1 + len(combos)], jh[-2], jh[-1]
            true_d = j_ge - j_gp
            e = ((j_est - j_prev[None]) - true_d[None]).norm(dim=-1).mean(-1) * 1000.0
            A_sum["e_art"] += e.cpu().numpy()
            A_sum["e_zero"] += float(true_d.norm(dim=-1).mean() * 1000.0)
            A_sum["n"] += 1
            n_steps += 1
        print(f"{name}: {len(steps)} steps, {time.time() - t0:.0f} s", flush=True)
    out = {"r": a.r, "dfit_ms": a.dfit, "mem_ms": a.mem, "stride": a.stride, "layer": a.layer,
           "combos": [{"eps_px": c[0], "kappa": c[1], "lam_p": c[2]} for c in combos], "categories": {}}
    for cat, v in acc.items():
        out["categories"][cat] = {"n_steps": v["n"], "mean_e_art_mm": (v["e_art"] / max(v["n"], 1)).tolist(),
                                  "mean_e_zero_mm": v["e_zero"] / max(v["n"], 1),
                                  "valid_frac_by_eps": (v["n_valid_meas"] / max(v["n_meas"], 1)).tolist()}
    loc, glo = out["categories"]["local"], out["categories"]["global"]
    J = (np.array(loc["mean_e_art_mm"]) + np.array(glo["mean_e_art_mm"])) / 2
    out["objective_mm"] = J.tolist()
    k = int(np.argmin(J))
    out["best"] = {"index": k, **out["combos"][k], "objective_mm": float(J[k]),
                   "zero_objective_mm": (loc["mean_e_zero_mm"] + glo["mean_e_zero_mm"]) / 2}
    out["wall_s"] = time.time() - t0
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out["best"]), flush=True)


if __name__ == "__main__":
    main()
