#!/usr/bin/env python3
"""K-star KS0: A/A audit of the statistical unit (docs/EXPERIMENT_RECORDS.md#kstar-prereg-20261025).

The three S39 seeds are the same recipe; their per-endpoint RA-MPJPE differences on zgz are an A/A comparison whose true
mean effect is a seed effect, not a method effect. For each domain and seed pair: the 20-endpoint block bootstrap CI,
the 200-endpoint block CI, the run-cluster CI, the integrated autocorrelation time of the difference within runs and
the resulting effective n (all descriptive), and the seed-level SD and minimum detectable effect that registered
comparisons use. Reads only the stored S39 archives (`outputs/s44/p10`).

    python tools/kstar/aa_audit.py --out outputs/kstar/s0_aa
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kstar import common as KC  # noqa: E402


def tau_int(d, run, max_lag=50):
    """Integrated autocorrelation time of d within runs (initial positive sequence)."""
    x = d - d.mean()
    var = (x * x).mean()
    tau = 1.0
    for lag in range(1, max_lag + 1):
        num = cnt = 0.0
        for r in np.unique(run):
            xr = x[run == r]
            if len(xr) > lag:
                num += float((xr[lag:] * xr[:-lag]).sum())
                cnt += len(xr) - lag
        if cnt == 0:
            break
        rho = num / cnt / var
        if rho <= 0:
            break
        tau += 2 * rho
    return tau


def run_ci(d, run, n_boot=KC.N_BOOT, seed=KC.BOOT_SEED):
    return KC.block_ci(d, run, n_boot=n_boot, seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    KC.verify_protocol()
    arcs = {s: KC.archive(s) for s in KC.SEEDS}
    res = {"pairs": {}}
    excl20 = 0
    for seq in KC.ZGZ:
        for s1, s2 in itertools.combinations(KC.SEEDS, 2):
            A, B = arcs[s1][seq], arcs[s2][seq]
            assert np.array_equal(A["end"], B["end"]) and np.array_equal(A["run"], B["run"])
            d = A["mpjpe_ra_mm"].astype(np.float64) - B["mpjpe_ra_mm"].astype(np.float64)
            run = A["run"]
            b20 = KC.block_ci(d, KC.blocks_of(run, 20))
            b200 = KC.block_ci(d, KC.blocks_of(run, 200))
            rc = run_ci(d, run)
            tau = tau_int(d, run)
            n_eff = len(d) / tau
            se_eff = d.std(ddof=1) / np.sqrt(n_eff)
            ent = {"mean_diff": float(d.mean()), "sd": float(d.std(ddof=1)), "n": int(len(d)),
                   "ci_block20": b20, "ci_block200": b200, "ci_run": rc, "tau_int": tau, "n_eff": n_eff,
                   "mde_neff_mm": 2.8 * se_eff,
                   "block20_excludes_0": bool(b20["lo"] > 0 or b20["hi"] < 0)}
            excl20 += ent["block20_excludes_0"]
            res["pairs"][f"{seq}|{s1}-{s2}"] = ent
            print(seq, s1, s2, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in ent.items()
                                if k in ("mean_diff", "tau_int", "n_eff", "mde_neff_mm", "block20_excludes_0")}, flush=True)
    taus = [v["tau_int"] for v in res["pairs"].values()]
    seed_sd = {}
    for seq in KC.ZGZ:
        means = np.array([arcs[s][seq]["mpjpe_ra_mm"].astype(np.float64).mean() for s in KC.SEEDS])
        sd = float(means.std(ddof=1))
        seed_sd[seq] = {"per_seed_mean": means.tolist(), "sd_seed": sd,
                        "mde_3v3_mm": float((2.776 + 0.941) * np.sqrt(2.0 / 3.0) * sd)}
    res["decision"] = {
        "block20_excludes_0_count": int(excl20),
        "block_len": int(max(20, 2 * np.ceil(max(taus)))),
        "seed_level": seed_sd,
        "rule": "registered (K-Battle R2 P6): block bootstrap CIs only describe endpoint sampling; model comparisons on "
                "zgz use seed-level units (3 vs 3 two-sample t, df 4, 80 % power: MDE = 3.04 * sqrt(2/3) * SD_seed); "
                "descriptive block length = max(20, 2 * max tau_int)"}
    KC.save_json(out / "result.json", res)
    KC.receipt(out, {"probe": "S0b A/A"}, scripts=(__file__,), inputs=KC.archive_files())
    print("DECISION", res["decision"])


if __name__ == "__main__":
    main()
