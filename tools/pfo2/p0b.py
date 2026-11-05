#!/usr/bin/env python3
"""PFO2 P0-B, routing-precision dose-response (docs/EXPERIMENT_RECORDS.md#pfo2-p0-prereg-20261103): F's trunks (PFO
arm f1, seeds 3407-3409, step 6000) read with the label routes plus location noise sigma (mm, calibrated median
displacement) and part misassignment `mis` (semkine/pfo2_routes.py), zero training, diagnostic only.

    python -m tools.pfo2.p0b run --seed S --sigma X --mis M --task e1|pert_rf
        one cell; tag pfo_f1_s<seed>__g<sigma>m<mis*100>__label (sigma 0 and mis 0 = the plain label routes, no noise)
    python -m tools.pfo2.p0b calib
        the (scale, keep) calibration of every grid sigma on the rest template (no data), decisions/p0b_calib.json
    python -m tools.pfo2.p0b decide
        sigma* and the written gate, decisions/p0b.json (+ the descriptive grid table analysis/p0b.md)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo2.common import OUT, PFO_OUT, SEEDS, ZGZ, now, read, require, save  # noqa: E402

SIGMAS = (2, 4, 6, 8, 12)
MIS = (0.0, 0.1, 0.2, 0.3)
NPZ = ROOT / "assets/mano_right.npz"
OPEN_AT, CLOSE_AT = 8.0, 5.0


def tag(seed, sigma, mis):
    return f"pfo_f1_s{seed}__g{sigma:g}m{int(round(mis * 100))}"


def cells():
    out = [(0, 0.0)]
    for s in SIGMAS:
        for m in MIS:
            out.append((s, m))
    return out


def run_cell(seed, sigma, mis, task):
    from tools.pfo2 import run as R
    from tools.pfo import nets as N
    from semkine import pfo2_routes as PR2
    _, sp = R.pfo_spec("f1", seed, 6000)
    R.prepare(sp)
    noisy = sigma > 0 or mis > 0
    native_load = N.load

    def load(*args, **kwargs):
        net = native_load(*args, **kwargs)
        if noisy:
            from semkine import xag as XAG
            XAG.sample_sources = PR2.install(XAG.sample_sources, NPZ, float(sigma), float(mis), seed=0)
        return net
    N.load = load
    name = tag(seed, sigma, mis)
    side = (lambda: PR2.summary()) if noisy else None
    if task == "e1":
        out = R.engine_run(name, "e1", sp, route="label")
        if side:
            save(out / "noise.json", side(), exclusive=True)
    else:
        from tools.pfo2 import engine as E2
        E2.run(name, "pert_rf", "label", str(sp), sidecar=side)


def calib():
    from semkine import pfo2_routes as PR2
    geo = PR2.build_geo(NPZ)
    rows = {}
    for s in SIGMAS:
        scale, keep, floor = PR2.calibrate(geo["v"], geo["f"], geo["D"], geo["area"], float(s))
        check = PR2.displacement_median(geo["v"], geo["f"], geo["D"], geo["area"], scale, keep, seed=1)
        rows[str(s)] = {"scale_m": scale, "keep": keep, "floor_mm": floor, "median_mm_heldout_draw": check}
    save(OUT / "decisions" / "p0b_calib.json", {"rows": rows, "written_at_utc": now()}, exclusive=True)
    print(rows)


def ra(t, task="e1", root=None):
    p = (root or OUT) / "readings" / t / task / "report.json"
    return read(p) if p.exists() else None


def sigma_star(curve, half):
    """First crossing of gain(sigma) <= half along the grid (sigma 0 first), linear interpolation; None if never."""
    xs = sorted(curve)
    for a, b in zip(xs[:-1], xs[1:]):
        if curve[a] > half >= curve[b]:
            return a + (b - a) * (curve[a] - half) / (curve[a] - curve[b])
        if curve[a] <= half:
            return a
    return None if curve[xs[-1]] > half else xs[-1]


def decide():
    dest = OUT / "decisions" / "p0b.json"
    require(not dest.exists(), "decided once")
    dep = {s: ra(f"pfo_f1_s{s}", "e1", PFO_OUT)["overall"]["mpjpe_ra_mm"] for s in SEEDS}
    grid, retention = {}, {}
    for sg, m in cells():
        vals, r15, f15 = [], [], []
        for s in SEEDS:
            e = ra(f"{tag(s, sg, m)}__label")
            require(e is not None, f"missing E1 cell {sg} {m} seed {s}")
            vals.append(e["overall"]["mpjpe_ra_mm"])
            p = ra(f"{tag(s, sg, m)}__label", "pert_rf")
            if p is not None:
                r15.append(p["per_sequence"]["zgz_local"]["R15"]["rho_median"])
                f15.append(p["per_sequence"]["zgz_local"]["F15"]["rho_median"])
        grid[f"{sg}/{m}"] = {"ra_by_seed": vals, "ra_mean": float(np.mean(vals))}
        if len(r15) == len(SEEDS):
            retention[f"{sg}/{m}"] = {"R15_local": float(np.mean(r15)), "F15_local": float(np.mean(f15))}
    dep_mean = float(np.mean(list(dep.values())))
    gain0 = dep_mean - grid["0/0.0"]["ra_mean"]
    curve = {0: gain0, **{s: dep_mean - grid[f"{s}/0.0"]["ra_mean"] for s in SIGMAS}}
    half = gain0 / 2.0
    star = sigma_star(curve, half) if gain0 > 0 else None
    if star is None:
        verdict = "OPEN" if gain0 > 0 else "CLOSE_PERMANENT"
    elif star >= OPEN_AT:
        verdict = "OPEN"
    elif star <= CLOSE_AT:
        verdict = "CLOSE_PERMANENT"
    else:
        verdict = "NOT_OPEN"
    out = {"deployed_ra_mean": dep_mean, "deployed_ra_by_seed": dep, "label_gain": gain0, "half_gain": half,
           "gain_curve_mis0": {str(k): v for k, v in curve.items()}, "sigma_star_mm": star,
           "sigma_star_note": "None = the half-gain level is never reached on the grid (sigma* > 12 mm)" if star is None else "",
           "verdict": verdict, "rule": "sigma* >= 8 -> OPEN (A3); sigma* <= 5 -> CLOSE_PERMANENT; else NOT_OPEN",
           "grid": grid, "retention_local": retention, "decided_at_utc": now()}
    save(dest, out, exclusive=True)
    lines = ["## P0-B 路由精度剂量—反应（F 三种子，E1 RA 均值，mm；行 = 噪声 σ，列 = 错派率）", "",
             f"部署路由（r_s2d）{dep_mean:.2f}；无噪声标签 {grid['0/0.0']['ra_mean']:.2f}；标签收益 {gain0:.2f}，"
             f"一半 {half:.2f}；σ* = {star if star is None else round(star, 2)} mm → {verdict}", "",
             "| σ \\ 错派 | 0 % | 10 % | 20 % | 30 % |", "|---|---|---|---|---|"]
    for s in SIGMAS:
        lines.append(f"| {s} | " + " | ".join(f"{grid[f'{s}/{m}']['ra_mean']:.2f}" for m in MIS) + " |")
    lines += ["", "local 保留率 R15 / F15（三种子均值）：", "", "| 单元 | R15 | F15 |", "|---|---|---|"]
    for k, v in retention.items():
        lines.append(f"| {k} | {v['R15_local']:.2f} | {v['F15_local']:.2f} |")
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "p0b.md").write_text("\n".join(lines) + "\n")
    print(verdict, star, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    r = sub.add_parser("run")
    r.add_argument("--seed", required=True, type=int)
    r.add_argument("--sigma", required=True, type=float)
    r.add_argument("--mis", required=True, type=float)
    r.add_argument("--task", required=True, choices=("e1", "pert_rf"))
    sub.add_parser("calib")
    sub.add_parser("decide")
    a = p.parse_args(argv)
    if a.verb == "run":
        require((a.sigma, a.mis) in [(float(s), float(m)) for s, m in cells()], "not a grid cell")
        run_cell(a.seed, a.sigma, a.mis, a.task)
    elif a.verb == "calib":
        calib()
    else:
        decide()


if __name__ == "__main__":
    main()
