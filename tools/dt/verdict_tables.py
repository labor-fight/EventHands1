#!/usr/bin/env python3
"""DT round: the result tables of docs/DT_RENDER_TRACK_VERDICT.md, generated from the evaluation jsons (nothing typed by hand).

    python tools/dt/verdict_tables.py > /tmp/tables.md        # all arms that have an evaluation
    python tools/dt/verdict_tables.py --arms dt_base dt_dz     # a subset

Primary statistics use seeds 3407 and 3408 (the preregistered pair); seed 3409 is listed separately.
Source: outputs/semkine/<arm>_s<seed>/evalx_val_core_last_tf_pert.json (last checkpoint, no selection).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RUNS = REPO / "outputs" / "semkine"
LABEL = [
    ("rt_cnntrack", "同网络，MSE 目标（参照）"),
    ("dt_base", "目标架构：so3_trans_fk，统一配方（基线）"),
    ("dt_tr", "C1 平移监督（TRANS_BETA 0.01，ABS_FK 1）"),
    ("dt_nos", "C2a 去掉尺度增强"),
    ("dt_dz", "C2c 深度一致的尺度增强"),
    ("dt_trnos", "C1 + C2a"),
    ("dt_trdz", "C1 + C2c"),
    ("dt_nopm", "C9a 去掉 prev_mlp"),
    ("dt_pmt", "C9b 屏蔽 prev_mlp 的平移输出"),
    ("dt_cam", "C2b 射线平面输入"),
    ("dt_so3c", "C3 根旋转在 SO(3) 上合成"),
    ("dt_w05", "C5 半宽 ResNet18"),
    ("dt_l3", "C6 去掉 layer4"),
    ("dt_dz_w05", "C2c + C5"),
    ("dt_dz_l3", "C2c + C6"),
]
SEEDS = ("3407", "3408", "3409")


def load(arm: str, seed: str):
    f = RUNS / f"{arm}_s{seed}" / "evalx_val_core_last_tf_pert.json"
    if not f.exists():
        return None
    j = json.loads(f.read_text())
    m = j["model"]
    g, l = m["zgz_global"], m["zgz_local"]
    jit = m.get("jitter", {})
    fing = (g["motion"]["finger_speed_ratio"] + l["motion"]["finger_speed_ratio"]) / 2
    return {"RA": m["overall"]["mpjpe_ra_mm"], "RAg": g["mpjpe_ra_mm"][0], "RAl": l["mpjpe_ra_mm"][0],
            "ABS": m["overall"]["mpjpe_abs_mm"], "ABSg": g["mpjpe_abs_mm"][0], "ABSl": l["mpjpe_abs_mm"][0],
            "rot": m["overall"]["root_rot_deg"], "trg": g["transl_mm"][0], "trl": l["transl_mm"][0],
            "TF": j["tf"]["overall"]["mpjpe_ra_mm"], "amp": j["amplification"]["mpjpe_ra_mm"],
            "fail": g["failure"]["episodes"] + l["failure"]["episodes"],
            "rsp": g["motion"]["root_speed_ratio"], "fsp": fing,
            "jit": jit.get("jit_pred_mm"), "acc": jit.get("acc_err_mm"), "accr": jit.get("acc_ratio"),
            "a_t": jit.get("acc_err_only_transl_mm"), "a_r": jit.get("acc_err_only_root_mm"),
            "a_f": jit.get("acc_err_only_fingers_mm"), "jitgt": jit.get("jit_gt_mm")}


def params_of(arm: str):
    for seed in SEEDS:
        f = RUNS / f"{arm}_s{seed}" / "training_metadata.json"
        if f.exists():
            return json.loads(f.read_text()).get("params_total") or json.loads(f.read_text()).get("total_params")
    return None


def fmt(v, n=2):
    return "-" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{n}f}"


def mean(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None)
    a = ap.parse_args()
    arms = [(k, d) for k, d in LABEL if a.arms is None or k in a.arms]
    data = {k: {s: load(k, s) for s in SEEDS} for k, _ in arms}
    arms = [(k, d) for k, d in arms if any(data[k].values())]
    out = ["### 表 A 精度（最后一步，zgz，两个种子 3407 / 3408 的均值；第三个种子单列）", "",
           "| 臂 | 说明 | 种子 | RA（逐种子） | **RA 均值** | RA global / local | **绝对 MPJPE 均值** | 绝对 global / local | zgz_local 平移误差 | 根旋转 ° | 教师强制 RA | 放大 | 失败段 |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, d in arms:
        r = [data[k][s] for s in SEEDS[:2] if data[k][s]]
        seeds = ",".join(s for s in SEEDS[:2] if data[k][s])
        per = " / ".join(fmt(data[k][s]["RA"]) for s in SEEDS[:2] if data[k][s])
        extra = f"（3409：{fmt(data[k]['3409']['RA'])}）" if data[k]["3409"] else ""
        if not r:
            continue
        mk = lambda key: mean([x[key] for x in r])                                  # noqa: E731
        out.append(f"| `{k}` | {d} | {seeds}{'+3409' if data[k]['3409'] else ''} | {per}{extra} | **{fmt(mk('RA'))}** | "
                   f"{fmt(mk('RAg'))} / {fmt(mk('RAl'))} | **{fmt(mk('ABS'), 1)}** | {fmt(mk('ABSg'), 1)} / {fmt(mk('ABSl'), 1)} | "
                   f"{fmt(mk('trl'), 1)} | {fmt(mk('rot'))} | {fmt(mk('TF'))} | {fmt(mk('amp'), 3)} | {fmt(mk('fail'), 0)} |")
    out += ["", "### 表 B 抖动与运动（两个种子均值；`rt_cnntrack` 的抖动块来自离线归因文件，不在此表）", "",
            "| 臂 | jit_pred mm/步 | jit_gt | **acc_err mm/步²** | acc_ratio | acc_err 只保留 平移 / 根 / 手指 | 根速度比 global | 手指速度比 |",
            "|---|---|---|---|---|---|---|---|"]
    for k, d in arms:
        r = [data[k][s] for s in SEEDS[:2] if data[k][s]]
        if not r or r[0]["jit"] is None:
            continue
        mk = lambda key: mean([x[key] for x in r])                                  # noqa: E731
        out.append(f"| `{k}` | {fmt(mk('jit'))} | {fmt(mk('jitgt'))} | **{fmt(mk('acc'))}** | {fmt(mk('accr'))} | "
                   f"{fmt(mk('a_t'))} / {fmt(mk('a_r'))} / {fmt(mk('a_f'))} | {fmt(mk('rsp'))} | {fmt(mk('fsp'))} |")
    print("\n".join(out))


if __name__ == "__main__":
    main()
