#!/usr/bin/env python3
"""S45-TR report (docs/EXPERIMENT_RECORDS.md#s45tr-prereg-20261030): single seed 3407, descriptive. CPU only.

Writes outputs/s45tr/20261030/report.md and summary.json from the finished artifacts only:
  * the nine-column main table (tools/report_table.py rows: baseline, S39, S45, every arm with a main row);
  * zgz E1 / E2 / E3 RA (local / global) of every arm next to S45's C1SF readings, as arm - S45;
  * the four mechanism readouts of every network with a readout summary (S45 reproduced first);
  * the T0 gate.
Arms without a main row are listed with their state, never as a result.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45tr.common import FORMAL_ARMS, OUT, T0_DIR, now, read, run_name, save  # noqa: E402

S45_EVAL = ROOT / "outputs/c1sf/20261029/evaluations/so3"
S45_ROW = ROOT / "outputs/semkine/s45_main_row.json"
LABELS = {
    "n1": "S45-TR N1：读出不见前态姿态（prev48 置零）",
    "n2": "S45-TR N2：同包不动点一致性（λ=1）",
    "n3": "S45-TR N3：时间偏移前态（±100–1000 ms 的 GT）",
    "n4": "S45-TR N4：S45 自身闭环回放前态（DAgger 一轮）",
    "t1a": "S45-TR T1a：事件图节点特征替换 token7",
    "t1b": "S45-TR T1b：冻结 T0 对应路由（τ=0.5）†",
    "t1d": "S45-TR T1d：XAG GT 面路由（oracle，不可部署）",
}
ORDER = ("n1", "n2", "n3", "n4", "t1a", "t1b", "t1d")


def main_table():
    from tools import report_table as original
    paths = {"dt3_dz_l3_w200cfg": ROOT / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json", "s45": S45_ROW}
    labels = {"s45": "S45：XYZ-C1 + SO(3)+平移+FK 损失（seed3407）"}
    for arm in ORDER:
        prefix = run_name(arm).removesuffix("_s3407")
        f = OUT / "evaluations" / arm / "row_step6000" / f"{prefix}_main_row_step6000.json"
        if f.is_file():
            paths[prefix] = f
            labels[prefix] = LABELS[arm]

    class Rows:
        def __truediv__(self, filename):
            return paths[str(filename).removesuffix("_main_row.json")]
    old = original.OUT
    original.OUT = Rows()
    try:
        lines = [original.HEADER, original.RULE, original.BASELINE,
                 original.row("dt3_dz_l3_w200cfg", "S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *")]
        lines += [original.row(p, labels[p]) for p in paths if p != "dt3_dz_l3_w200cfg"]
    finally:
        original.OUT = old
    return "\n".join(lines), [a for a in ORDER if run_name(a).removesuffix("_s3407") in paths]


def trace(directory):
    f = directory / "report.json"
    if not f.is_file():
        return None
    m = read(f)["metrics"]
    return {d: m[d]["mpjpe_ra_mm"] for d in ("local", "global")}


def three():
    # E2 here = fed S45's own E1 loop states (tools/s45tr/evaluate.py), so S45's own E2 is its E1 by definition
    # (C1SF's S45 E2 was fed C1's loop states and is not this quantity)
    ref = {m: trace(S45_EVAL / f"{m}_step6000_repeat1_none") for m in ("e1", "e3")}
    ref["e2"] = ref["e1"]
    rows = {"s45": ref}
    for arm in ORDER:
        rows[arm] = {m: trace(OUT / "evaluations" / arm / f"{m}_step6000") for m in ("e1", "e2", "e3")}
        rows[arm]["e1_late"] = {s: trace(OUT / "evaluations" / arm / f"e1_step{s}") for s in (5000, 5500)}
    return rows


def readouts():
    out = {}
    for net in ("s45",) + ORDER:
        f = OUT / "readouts" / net / "summary.json"
        if f.is_file():
            out[net] = read(f)
    return out


def fmt(v, d=2):
    return "—" if v is None else f"{v:.{d}f}"


def main():
    table, with_rows = main_table()
    e = three()
    r = readouts()
    gate = read(T0_DIR / "gate.json") if (T0_DIR / "gate.json").is_file() else None
    lines = ["# S45-TR 结果（自动生成；单 seed 3407，描述性）", "", table, "",
             "† T1b 的 Params / FLOPs 不含冻结 T0（另计）；时延含 T0 前向。", ""]
    missing = [a for a in ORDER if a not in with_rows]
    if missing:
        status = read(OUT / "status.json")["jobs"]
        lines += ["无主行的臂（不作结果）：" + "；".join(
            f"{a}（" + ", ".join(f"{k}={v['status']}" for k, v in status.items()
                                 if k.split(":")[0] in ("formal", "preflight") and k.split(":")[1].startswith(a)) + "）"
            for a in missing), ""]
    lines += ["## zgz E1 / E2 / E3（RA，mm；括号内为 臂 − S45；E2 = 喂 S45 自己的闭环状态，S45 的 E2 即其 E1）", "",
              "| 网络 | E1 local | E1 global | E2 local | E2 global | E3 local | E3 global |", "|---|---|---|---|---|---|---|"]
    ref = e["s45"]
    for net in ("s45",) + ORDER:
        row = e[net]
        if all(row[m] is None for m in ("e1", "e2", "e3")):
            continue
        cells = []
        for m in ("e1", "e2", "e3"):
            for d in ("local", "global"):
                v = None if row[m] is None else row[m][d]
                dv = None if (v is None or ref[m] is None or net == "s45") else v - ref[m][d]
                cells.append(fmt(v) + ("" if dv is None else f"（{dv:+.2f}）"))
        lines.append(f"| {net} | " + " | ".join(cells) + " |")
    lines += ["", "## 机制读数（zgz_local / 训练 local 中位）", "",
              "| 网络 | 保留率 T10 / R5 / Fs | 流形内 / 外 | 收缩 k（全部 / 低运动） | 反复送入 GT 起点 RA / k @20 | 均值起点 k @20 | 闭环起点 RA / k @20 | 钳位份额 姿态 / 平移 |",
              "|---|---|---|---|---|---|---|---|"]
    for net, s in r.items():
        for dom in ("zgz_local", "train_local"):
            r1, r2, r3, r4 = s["r1_retention"][dom], s["r2_shrinkage"][dom], s["r3_fixed_point"][dom], s["r4_clamp"][dom]
            lines.append(
                f"| {net} {dom} | {fmt(r1['T10'])} / {fmt(r1['R5'])} / {fmt(r1['Fs'])} | {fmt(r1['top'])} / {fmt(r1['bottom'])} | "
                f"{fmt(r2['k_all'])} / {fmt(r2['k_low_motion'])} | {fmt(r3['gt']['20']['ra_mm_median'], 1)} / {fmt(r3['gt']['20']['shrink_k_median'])} | "
                f"{fmt(r3['mean']['20']['shrink_k_median'])} | {fmt(r3['loop']['20']['ra_mm_median'], 1)} / {fmt(r3['loop']['20']['shrink_k_median'])} | "
                f"{fmt(r4['pose_clamp_share'])} / {fmt(r4['transl_clamp_share'])} |")
    if "s45" in r:
        lines += ["", "S45 读数工具逐位复现 S45-LOC / S45-DEEP：" + ("是" if all(r["s45"].get("s45_bitwise_reproduction", {}).values()) else "否")]
    lines += ["", "## T0 门（留出受试者 ylf）", ""]
    if gate is None:
        lines.append("未读")
    else:
        g, d = gate["gate"], gate["descriptive"]
        lines.append(f"- {gate['status']}：门集合模板误差中位 T0 {g['t0_median_mm']:.2f} mm 对 S45 射线关联 {g['assoc_median_mm']:.2f} mm（门槛 {g['threshold_mm']} mm），"
                     f"{gate['gate_events']} 个事件")
        lines.append(f"- 描述：手上事件中位 {d['t0_onhand_median_mm']:.2f} mm；面准确率 {d['face_accuracy_onhand']:.3f}；部位准确率 {d['part_accuracy_onhand']:.3f}；"
                     f"在手上 AUC {d['onhand_auc']:.3f}")
    text = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(text)
    save(OUT / "summary.json", {"computed_at_utc": now(), "arms_with_main_row": with_rows, "zgz_three": e,
                                "readouts": {k: {kk: v[kk] for kk in ("r1_retention", "r2_shrinkage", "r3_fixed_point", "r4_clamp")}
                                             for k, v in r.items()},
                                "t0_gate": gate, "single_seed_descriptive_only": True, "s39_s43_c1_s45_unchanged": True})
    print(text)


if __name__ == "__main__":
    main()
