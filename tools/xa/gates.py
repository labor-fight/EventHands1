#!/usr/bin/env python3
"""[XYZ-ASSOC] P3 gates G1-G4 from the last-step main rows (outputs/semkine/<run>_main_row_last.json).
Pre-registered: tau = max(1.1, 2*delta0), delta0 = |RA(xa_c1_s3407) - RA(xa_c1r_s3407)| per scope;
paired same-seed differences, two-seed mean, both seeds must share the sign to count as 'established'.
Prints markdown; `--json out.json` also dumps the numbers."""
import argparse
import json
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--json", default=None)
a = ap.parse_args()

ROWS = {}
for run in ("xa_c1", "xa_c1r", "xa_c0", "xa_orc", "xa_ctr"):
    p = Path(f"outputs/semkine/{run}_main_row_last.json")
    ROWS[run] = json.loads(p.read_text()) if p.exists() else None

def ra(run, seed, scope):
    r = ROWS[run]
    if r is None:
        return None
    return float(r["per_seed"][str(seed)][scope]["mpjpe_ra_mm"])

SCOPES = ("overall", "global", "local")
HIST_LAST_MEAN = 21.923
out = {"tau": {}, "delta0": {}, "scopes": {}}
lines = []
for scope in SCOPES:
    c1 = {s: ra("xa_c1", s, scope) for s in (3407, 3408)}
    c1r = ra("xa_c1r", 3407, scope)
    d0 = abs(c1[3407] - c1r)
    tau = max(1.1, 2 * d0)
    out["delta0"][scope] = d0
    out["tau"][scope] = tau
    sc = {"c1": c1, "c1r": c1r, "arms": {}}
    lines.append(f"\n**{scope}**：δ0 = {d0:.3f} mm，τ = {tau:.3f} mm；xa_c1 末步 {c1[3407]:.3f} / {c1[3408]:.3f}（均值 {(c1[3407]+c1[3408])/2:.3f}），xa_c1r {c1r:.3f}\n")
    lines.append("| 臂 | s3407 | s3408 | 配对差 c1−臂（s3407 / s3408） | 均值 | 两种子同号 | 判定 |")
    lines.append("|---|---|---|---|---|---|---|")
    d_orc_mean = None
    for arm, gate in (("xa_orc", "G1"), ("xa_ctr", "G2"), ("xa_c0", "G3")):
        v = {s: ra(arm, s, scope) for s in (3407, 3408)}
        if v[3407] is None or v[3408] is None:
            lines.append(f"| {arm} | — | — | — | — | — | {gate} 未测（无主行） |")
            sc["arms"][arm] = None
            continue
        d = {s: c1[s] - v[s] for s in (3407, 3408)}
        dm = (d[3407] + d[3408]) / 2
        same = (d[3407] > 0) == (d[3408] > 0)
        if gate == "G1":
            d_orc_mean = dm
            if abs(dm) <= tau:
                verdict = "G1：|d_orc| ≤ τ → 分配正确性对本架构无增量"
            elif dm >= tau and same:
                verdict = "G1：d_orc ≥ τ 且同号 → 分配正确性有价值（上界含泄漏）"
            elif dm <= -tau and same:
                verdict = "G1：oracle 反而更差 ≥ τ（同号）"
            else:
                verdict = "G1：超过 τ 但两种子不同号 → 不计成立"
        elif gate == "G2":
            if d_orc_mean is None or abs(d_orc_mean) <= tau:
                verdict = "G2：G1 未判有价值 → 不适用"
            else:
                phi = dm / d_orc_mean
                phis = {s: d[s] / (c1[s] - ra("xa_orc", s, scope)) for s in (3407, 3408)}
                sc["phi"] = {"mean": phi, "per_seed": phis}
                if phi >= 0.5:
                    verdict = f"G2：φ = {phi:.2f}（{phis[3407]:.2f} / {phis[3408]:.2f}）≥ 0.5 → 规则级修正收回主要差距"
                elif phi <= 0.25:
                    verdict = f"G2：φ = {phi:.2f}（{phis[3407]:.2f} / {phis[3408]:.2f}）≤ 0.25 → 规则级不够"
                else:
                    verdict = f"G2：φ = {phi:.2f}（{phis[3407]:.2f} / {phis[3408]:.2f}）介于 0.25–0.5 → 补 3409"
                if abs(dm) <= tau:
                    verdict += f"；注意 |d_ctr| = {abs(dm):.2f} ≤ τ，ctr 对 c1 本身未超噪声底"
        else:
            if abs(dm) <= tau:
                verdict = "G3：|d| ≤ τ → 显式 3D 通道无因果贡献"
            else:
                verdict = f"G3：{'C0 更好' if dm > 0 else 'C1 更好'} ≥ τ{'（同号）' if same else '（两种子不同号，不计成立）'}"
        sc["arms"][arm] = {"ra": v, "d": d, "d_mean": dm, "same_sign": same, "verdict": verdict}
        lines.append(f"| {arm} | {v[3407]:.3f} | {v[3408]:.3f} | {d[3407]:+.3f} / {d[3408]:+.3f} | {dm:+.3f} | {'是' if same else '否'} | {verdict} |")
    out["scopes"][scope] = sc

c1m = (ra("xa_c1", 3407, "overall") + ra("xa_c1", 3408, "overall")) / 2
g4 = abs(c1m - HIST_LAST_MEAN)
out["G4"] = {"c1_last_mean": c1m, "hist_last_mean": HIST_LAST_MEAN, "diff": g4, "pass": g4 <= 2.0}
lines.append(f"\n**G4 复现**：xa_c1 末步两种子均值 {c1m:.3f} 对历史末步均值 {HIST_LAST_MEAN}，差 {g4:.3f} mm → {'复现成立' if g4 <= 2 else '配方漂移'}。")
print("\n".join(lines))
if a.json:
    Path(a.json).write_text(json.dumps(out, indent=2, ensure_ascii=False))
