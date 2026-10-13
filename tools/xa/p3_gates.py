#!/usr/bin/env python3
"""[XYZ-ASSOC] P3 gate table from the last-step main rows. Prints a markdown block and writes
outputs/xa/step5/p3_gates.md + p3_gates.json. Thresholds are the pre-registered ones."""
import json
from pathlib import Path

REPO = Path("/data1/lyq/code/mesh/EventHands1")
rows = {r: json.load(open(REPO / f"outputs/semkine/{r}_main_row_last.json"))
        for r in ["xa_c1", "xa_c1r", "xa_c0", "xa_orc", "xa_ctr"]
        if (REPO / f"outputs/semkine/{r}_main_row_last.json").exists()}


def ra(run, seed, split):
    return rows[run]["per_seed"][str(seed)][split]["mpjpe_ra_mm"]


SPLITS = ["overall", "local", "global"]
delta0 = {s: abs(ra("xa_c1", 3407, s) - ra("xa_c1r", 3407, s)) for s in SPLITS}
tau = {s: max(1.1, 2 * delta0[s]) for s in SPLITS}
out = {"delta0": delta0, "tau": tau, "gates": {}}
lines = []
lines.append("| 门 | 分列 | 配对差（3407 / 3408） | 两种子均值 | τ | 判定 |")
lines.append("|---|---|---|---|---|---|")


def pair(a, b, s):
    d = [ra(a, sd, s) - ra(b, sd, s) for sd in (3407, 3408)]
    return d, sum(d) / 2


for s in SPLITS:
    # G1: d_orc = RA(c1) - RA(orc)
    d, m = pair("xa_c1", "xa_orc", s)
    same = (d[0] > 0) == (d[1] > 0)
    if abs(m) <= tau[s]:
        v = "≤ τ：分配正确性对本架构无增量"
    elif m >= tau[s] and same:
        v = "≥ τ 且同号：分配正确性有价值（含真值几何泄漏，上界）→ 进入 G2"
    else:
        v = "种子分裂或方向为负：不成立"
    out["gates"][f"G1_{s}"] = {"d": d, "mean": m, "verdict": v}
    lines.append(f"| G1 上界 d_orc = RA(c1) − RA(orc) | {s} | {d[0]:+.2f} / {d[1]:+.2f} | {m:+.2f} | {tau[s]:.2f} | {v} |")
for s in SPLITS:
    if "xa_ctr" not in rows:
        lines.append(f"| G2 φ = (RA(c1) − RA(ctr)) / d_orc | {s} | — | — | — | 未测（ctr 臂未完成） |")
        continue
    d, m = pair("xa_c1", "xa_ctr", s)
    dorc = out["gates"][f"G1_{s}"]["mean"]
    phi = m / dorc if dorc else float("nan")
    same = (d[0] > 0) == (d[1] > 0)
    if abs(m) <= tau[s]:
        v = f"|Δ| ≤ τ：规则级修正无可测收益（φ = {phi:.2f}）"
    elif phi >= 0.5 and same:
        v = f"φ = {phi:.2f} ≥ 0.5 且同号：规则级修正收回主要差距"
    elif phi <= 0.25:
        v = f"φ = {phi:.2f} ≤ 0.25：规则级不够"
    else:
        v = f"φ = {phi:.2f} 介于其间：补 3409 种子"
    out["gates"][f"G2_{s}"] = {"d": d, "mean": m, "phi": phi, "verdict": v}
    lines.append(f"| G2 规则修正 RA(c1) − RA(ctr) | {s} | {d[0]:+.2f} / {d[1]:+.2f} | {m:+.2f} | {tau[s]:.2f} | {v} |")
for s in SPLITS:
    d, m = pair("xa_c1", "xa_c0", s)
    same = (d[0] > 0) == (d[1] > 0)
    if abs(m) <= tau[s]:
        v = "|Δ| ≤ τ：显式 3D 通道无因果贡献"
    else:
        v = ("≥ τ 且同号：" if same else "≥ τ 但种子分裂：") + ("C0 优于 C1（显式 3D 通道有害）" if m > 0 else "C1 优于 C0（显式 3D 通道有益）")
    out["gates"][f"G3_{s}"] = {"d": d, "mean": m, "verdict": v}
    lines.append(f"| G3 通道 RA(c1) − RA(c0) | {s} | {d[0]:+.2f} / {d[1]:+.2f} | {m:+.2f} | {tau[s]:.2f} | {v} |")
m_c1 = (ra("xa_c1", 3407, "overall") + ra("xa_c1", 3408, "overall")) / 2
g4 = abs(m_c1 - 21.923)
out["gates"]["G4"] = {"mean_c1_last": m_c1, "hist_last_mean": 21.923, "diff": g4,
                      "verdict": "复现" if g4 <= 2 else "配方漂移"}
lines.append(f"| G4 复现（末步两种子均值 vs 历史末步均值 21.92） | overall | {m_c1:.2f} vs 21.92 | {g4:+.2f} | 2.00 | {out['gates']['G4']['verdict']} |")
hdr = (f"噪声底 δ0（xa_c1_s3407 − xa_c1r_s3407 末步）：overall {delta0['overall']:.3f}，local {delta0['local']:.3f}，"
       f"global {delta0['global']:.3f} mm；τ = max(1.1, 2·δ0)：overall {tau['overall']:.2f}，local {tau['local']:.2f}，global {tau['global']:.2f} mm。\n\n")
md = hdr + "\n".join(lines) + "\n"
(REPO / "outputs/xa/step5/p3_gates.md").write_text(md)
(REPO / "outputs/xa/step5/p3_gates.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
print(md)
