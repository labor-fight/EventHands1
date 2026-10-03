#!/usr/bin/env python3
"""DT round: raw tracker versus the preregistered constant-gain filter (0.5, 1.0, 0.5), two-seed means, for trained arms.

    python tools/dt/filter_summary.py dt_base dt_dz dt_trdz

Raw = outputs/semkine/<arm>_s<seed>/evalx_val_core_last_tf_pert.json; filtered = outputs/dt/filter/<arm>_s<seed>/r0.5_f1.0_t0.5.json
(written by tools/dt/filter_eval.py, same loop, same init, same protocol).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
TAG = "r0.5_f1.0_t0.5"


def row(js):
    m = js["model"]
    g, l = m["zgz_global"], m["zgz_local"]
    j = m.get("jitter", {})
    ret = js.get("perturb", {}).get("10", {}).get("retention_k1_k2_k5_k10_k20", [None] * 5)
    return {"RA": m["overall"]["mpjpe_ra_mm"], "RAl": l["mpjpe_ra_mm"][0], "ABS": m["overall"]["mpjpe_abs_mm"],
            "rot": m["overall"]["root_rot_deg"], "jit": j.get("jit_pred_mm"), "acc": j.get("acc_err_mm"),
            "accr": j.get("acc_ratio"), "rsp": g["motion"]["root_speed_ratio"],
            "fsp": (g["motion"]["finger_speed_ratio"] + l["motion"]["finger_speed_ratio"]) / 2,
            "rotacc": j.get("rot_acc_pred_deg"), "k1": ret[0], "k5": ret[2], "fail": g["failure"]["episodes"] + l["failure"]["episodes"]}


def main():
    arms = sys.argv[1:]
    print("| 臂 | 版本 | RA | 绝对 MPJPE | 根旋转 ° | jit_pred | **acc_err** | acc_ratio | 根角加速度 ° | 根速度比 global | 手指速度比 | 10° 扰动保留 k1 / k5 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        pair = {}
        have = [sd for sd in ("3407", "3408")
                if (REPO / f"outputs/dt/filter/{arm}_s{sd}/{TAG}_pert.json").exists() or (REPO / f"outputs/dt/filter/{arm}_s{sd}/{TAG}.json").exists()]
        for kind in ("raw", "filtered"):
            rows = []
            for seed in have:                                   # raw and filtered rows always use the same seeds
                f = (REPO / f"outputs/semkine/{arm}_s{seed}/evalx_val_core_last_tf_pert.json") if kind == "raw" else \
                    (REPO / f"outputs/dt/filter/{arm}_s{seed}/{TAG}_pert.json")
                if not f.exists() and kind == "filtered":
                    f = REPO / f"outputs/dt/filter/{arm}_s{seed}/{TAG}.json"
                if f.exists():
                    rows.append(row(json.loads(f.read_text())))
            if rows:
                pair[kind] = {k: (None if rows[0][k] is None else float(np.mean([r[k] for r in rows]))) for k in rows[0]} | {"n": len(rows)}
        for kind, r in pair.items():
            fm = lambda k, n=2: "-" if r[k] is None else f"{r[k]:.{n}f}"            # noqa: E731
            dacc = ""
            if kind == "filtered" and "raw" in pair and r["acc"] is not None:
                dacc = f"（{(r['acc'] / pair['raw']['acc'] - 1) * 100:+.0f}%）"
            dra = f"（{r['RA'] - pair['raw']['RA']:+.2f}）" if kind == "filtered" and "raw" in pair else ""
            dabs = f"（{r['ABS'] - pair['raw']['ABS']:+.1f}）" if kind == "filtered" and "raw" in pair else ""
            print(f"| `{arm}` | {'滤波 (0.5, 1, 0.5)' if kind == 'filtered' else '裸跟踪器'}{' (种子 ' + ','.join(have) + ')' if len(have) < 2 else ''} | {fm('RA')}{dra} | {fm('ABS', 1)}{dabs} | {fm('rot')} | "
                  f"{fm('jit')} | **{fm('acc')}**{dacc} | {fm('accr')} | {fm('rotacc')} | {fm('rsp')} | {fm('fsp')} | {fm('k1')} / {fm('k5')} |")


if __name__ == "__main__":
    main()
