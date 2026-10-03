#!/usr/bin/env python3
"""DT round: jitter tables in BOTH conventions, generated through evalx.jitter_decomp (never typed by hand).

    python tools/dt/jitter_tables.py rt_cnntrack dt_base dt_dz ...      # markdown on stdout

Absolute joints (translation included) and root-aligned joints (joint 0 subtracted, the main table's convention; translation
drops out). Per arm and seed the recorded closed-loop arrays (`evalx_val_core_last_tf_pert.npz`, or the filter evaluation's
npz for the filtered rows) go through `evalx.jitter_decomp`, sequences are combined frame-weighted as `evalx.evaluate` does
(`tools/dt/jitter_offline.py` machinery), then averaged over seeds 3407 and 3408. Runs without arrays are skipped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools"), str(REPO / "tools" / "dt"), str(REPO / "tools" / "tracking")]
import jitter_offline as JO                                           # noqa: E402
from config import load_config                                        # noqa: E402

SEEDS = ("3407", "3408")
TAG = "r0.5_f1.0_t0.5"
KEYS = ("acc_err_mm", "acc_err_ra_mm", "acc_ratio_ra", "jit_pred_mm", "jit_ra_pred_mm", "jit_ra_gt_mm", "acc_ra_gt_mm",
        "acc_ra_pred_mm", "acc_err_ra_only_root_mm", "acc_err_ra_only_fingers_mm", "acc_err_only_transl_mm")


def overall(npz: Path, st: dict) -> dict:
    return JO.combine(JO.analyse_npz(npz, st))


def mean_over_seeds(paths, st):
    rows = [overall(p, st) for p in paths]
    return {k: float(np.mean([r[k] for r in rows])) for k in KEYS if all(k in r for r in rows)}, len(rows)


def main() -> None:
    arms = sys.argv[1:]
    cfg = load_config(REPO / "configs/dt/dt_base.yaml")
    st = JO._state(cfg["MANO"]["NPZ"], cfg["DATA"]["ROOT"], cfg["DATA"]["SPLITS_MANIFEST"], 2, 0.5)
    print("| 臂 | 版本 | acc_err 绝对 mm/步² | **acc_err 根对齐** | 根对齐 GT 自己的加速度 | 根对齐 预测/GT 加速度比 | acc_err 根对齐：只保留 根 / 手指 | jit_pred 绝对 | jit_pred 根对齐（GT） |")
    print("|---|---|---|---|---|---|---|---|---|")
    for arm in arms:
        raw = [REPO / f"outputs/semkine/{arm}_s{s}/evalx_val_core_last_tf_pert.npz" for s in SEEDS]
        filt = [REPO / f"outputs/dt/filter/{arm}_s{s}/{TAG}.npz" for s in SEEDS]
        base = None
        for kind, paths in (("裸跟踪器", raw), ("滤波 (0.5, 1, 0.5)", filt)):
            if not all(p.exists() for p in paths):
                continue
            m, n = mean_over_seeds(paths, st)
            if base is None:
                base = m
            dra = f"（{(m['acc_err_ra_mm'] / base['acc_err_ra_mm'] - 1) * 100:+.0f}%）" if kind != "裸跟踪器" else ""
            dab = f"（{(m['acc_err_mm'] / base['acc_err_mm'] - 1) * 100:+.0f}%）" if kind != "裸跟踪器" else ""
            print(f"| `{arm}` | {kind} | {m['acc_err_mm']:.2f}{dab} | **{m['acc_err_ra_mm']:.2f}**{dra} | {m['acc_ra_gt_mm']:.2f} | "
                  f"{m['acc_ratio_ra']:.2f} | {m['acc_err_ra_only_root_mm']:.2f} / {m['acc_err_ra_only_fingers_mm']:.2f} | "
                  f"{m['jit_pred_mm']:.2f} | {m['jit_ra_pred_mm']:.2f}（{m['jit_ra_gt_mm']:.2f}） |")


if __name__ == "__main__":
    main()
