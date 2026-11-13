#!/usr/bin/env python3
"""Real-study tables with EvHandTrack and its paper ablations (docs/EXPERIMENT_RECORDS.md#real-amend5-20261111).

1. The comparison table (tools/real/table.py's rows, cells and cost rule unchanged) plus the EvHandTrack row: the Full
   model d1r_rol5 read in the streaming form (tools/real/evht.py), three-seed means, the RA cell with the seeds; its cost
   cells from outputs/real/20261110/costs/<tag>/report.json once measured, else "-".
2. The paper's ablation table (user's figure: Local motion MPJPE / MPVPE, Global motion MPJPE / MPVPE; rows (a) w/o
   surface association, Global MLP update, State-independent gating, (b) Single-step only, Full model), every row its
   trained arm read in the streaming form, three-seed means with the seeds in parentheses: per light (normal, dark) and
   both lights pooled (local = the four local sequences of kqj / lqh, global = the four global ones).
The dev subject (wh) gets the same tables, descriptive. Writes table2.md and rows2.json next to the readings.

Added 2026-11-12 before the first run (docs/EXPERIMENT_RECORDS.md#real-amend5-run-20261111; user: (a) "按照之前的仿真数据的
处理方法", i.e. tools/pmr/abltable3.py): a seed whose reading failed (the engine keeps a non-finite state as a failure)
is left out of the mean and named under the table; step 1's before-fine-tune comparison (the two from-scratch bases d1r
and abgeo, 6000 steps, no rollout, read in the same streaming form, descriptive); and the association during the rollout
fine-tune (training log: `train_xyz_valid` and `train_joint_error_mm_k0` by step range, per seed).

    python tools/real/table2.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.real import evht as X  # noqa: E402
from tools.real import ns  # noqa: E402
from tools.real import table as TB  # noqa: E402

EVHT = "d1r_rol5"
EVHT_LABEL = "EvHandTrack（运动学树 D1 + rol5，整幅 346×260，流式形式；路由器在真实训练集上训练）"
EVHT_COST = "pmr_d1r_rol5_s3407__str"
LIGHTS = (("normal", "正常光"), ("dark", "暗光"), ("both", "正常光 + 暗光"))
BASES = (("EvHandTrack 的从头基座（D1 R + ESMNet 路由器）", "d1r"), ("(a) 的从头基座（几何邻近关联）", "abgeo"))
ROLL = (("完整 EvHandTrack（rol5）", "d1r_rol5"), ("(a) w/o surface association（rol5）", "abgeo_rol5"))
RANGES = ((0, 100), (100, 300), (300, 600), (600, 1500))
ABL_HEAD = ["| Configuration | Local MPJPE↓ | Local MPVPE↓ | Global MPJPE↓ | Global MPVPE↓ | 种子 |",
            "|---|---|---|---|---|---|"]


def failed(arm):
    """Seeds whose reading ran and ended without a COMPLETE report, with the engine's last error line."""
    out = []
    for s in ns.SEEDS:
        p = X.report(arm, s)
        if (p.parent / "launch.json").exists() and not (p.exists() and json.loads(p.read_text()).get("status") == "COMPLETE"):
            log = ns.OUT / "logs" / f"ev_e1_{arm}_{s}.log"
            err = [x.strip() for x in log.read_text(errors="replace").splitlines() if "Error" in x] if log.exists() else []
            out.append((s, err[-1] if err else "no report"))
    return out


def abl_row(label, arm, subjects, light):
    per = per_seed([(s, X.report(arm, s)) for s in ns.SEEDS], subjects, light)
    if not per:
        return f"| {label} | - | - | - | - | - |", None, per
    m = mean(per)
    cells = [f"{m[k][f]:.2f}{seeds_txt(per, k, f)}" for k in ("local", "global") for f in ("mpjpe_ra_mm", "mpvpe_ra_mm")]
    return f"| {label} | " + " | ".join(cells) + f" | {' / '.join(str(s) for s, _ in per)} |", m, per


def scalars(arm, seed, tag):
    import glob
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    d = sorted(glob.glob(str(X.run_dir(arm, seed) / "logs" / "*")))[-1]
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def rollout_log():
    head = " | ".join(f"第 {a}–{b} 步" for a, b in RANGES)
    L = ["| 训练（K = 5 闭环展开微调） | 种子 | 关联有效比例：" + head + " | 首包关节误差 mm：" + head + " |",
         "|---|---|" + "---|" * (2 * len(RANGES))]
    for label, arm in ROLL:
        for s in ns.SEEDS:
            try:
                v, e = scalars(arm, s, "train_xyz_valid"), scalars(arm, s, "train_joint_error_mm_k0")
            except (IndexError, KeyError):
                continue

            def m(x, a, b):
                y = [val for st, val in x if a <= st < b]
                return sum(y) / len(y) if y else float("nan")
            L.append(f"| {label} | {s} | " + " | ".join(f"{m(v, a, b):.2f}" for a, b in RANGES) + " | "
                     + " | ".join(f"{m(e, a, b):.1f}" for a, b in RANGES) + " |")
    return L


def cell(blocks, subjects, light):
    if light != "both":
        return TB.cell(blocks, subjects, light)
    loc = [f"{s}_local{l}" for s in subjects for l in ("", "_dark")]
    glo = [f"{s}_global{l}" for s in subjects for l in ("", "_dark")]
    return {"local": TB.pooled(blocks, loc), "global": TB.pooled(blocks, glo), "overall": TB.pooled(blocks, loc + glo)}


def per_seed(paths, subjects, light):
    out = []
    for seed, p in paths:
        if p.exists():
            r = json.loads(p.read_text())
            if r.get("status") == "COMPLETE":
                out.append((seed, cell(r["per_sequence"], subjects, light)))
    return out


def mean(per):
    return {k: {f: sum(c[k][f] for _, c in per) / len(per) for f in TB.FIELDS} for k in ("local", "global", "overall")}


def seeds_txt(per, key, field):
    return "（" + " / ".join(f"{c[key][field]:.2f}" for _, c in per) + "）" if len(per) > 1 else ""


def evht_cost():
    p = ns.OUT / "costs" / EVHT_COST / "report.json"
    if not p.exists():
        return "- | - | -"
    c = json.loads(p.read_text())
    if c.get("status") != "COMPLETE":
        return "- | - | -"
    return (f"{c['latency_row']['scaled_ms']:.2f} ms | {c['macs_layers']['median'] / 1e9:.3f} G "
            f"| {c['params_total'] / 1e6:.2f} M")


def main():
    rows, lines = {}, []
    for group, subjects in (("test", ("kqj", "lqh")), ("dev", ("wh",))):
        title = "测试 kqj、lqh" if group == "test" else "调参 wh（描述）"
        for light, lname in LIGHTS[:2]:
            lines += ["", f"**{title} · {lname}**", "", TB.HEADER]
            for arm in TB.ARMS:
                per = per_seed([(s, ns.OUT / "readings" / f"cmp_{arm}_s{s}" / "e1" / "report.json") for s in ns.SEEDS],
                               subjects, light)
                if per:
                    lines.append(row(TB.LABELS[arm], per, TB.cost(arm)))
                    rows[f"{group}/{light}/{arm}"] = {"seeds": [s for s, _ in per], "mean": mean(per)}
            per = per_seed([(s, X.report(EVHT, s)) for s in ns.SEEDS], subjects, light)
            if per:
                lines.append(row(EVHT_LABEL, per, evht_cost()))
                rows[f"{group}/{light}/evhandtrack"] = {"seeds": [s for s, _ in per], "mean": mean(per)}
        for light, lname in LIGHTS:
            lines += ["", f"**消融（{title} · {lname}；训练结果，流式 E1，去根对齐，mm）**", "",
                      "| Configuration | Local MPJPE↓ | Local MPVPE↓ | Global MPJPE↓ | Global MPVPE↓ | 种子 |",
                      "|---|---|---|---|---|---|"]
            for i, (label, arm) in enumerate(X.PAPER):
                if i == 0:
                    lines.append("| **(a) Model components** | | | | | |")
                if i == 3:
                    lines.append("| **(b) Training strategy** | | | | | |")
                per = per_seed([(s, X.report(arm, s)) for s in ns.SEEDS], subjects, light)
                if not per:
                    lines.append(f"| {label} | - | - | - | - | - |")
                    continue
                m = mean(per)
                rows[f"{group}/{light}/abl/{arm}"] = {"seeds": [s for s, _ in per], "mean": m,
                                                     "per_seed": {str(s): c for s, c in per}}
                cells = [f"{m[k][f]:.2f}{seeds_txt(per, k, f)}" for k in ("local", "global")
                         for f in ("mpjpe_ra_mm", "mpvpe_ra_mm")]
                lines.append(f"| {'**Full model**' if arm == EVHT else label} | " + " | ".join(cells)
                             + f" | {' / '.join(str(s) for s, _ in per)} |")
            notes = [f"- {'Full model' if arm == EVHT else label}，种子 {s}：读数失败（{err}），不计入均值。"
                     for label, arm in X.PAPER for s, err in failed(arm)]
            if notes:
                lines += [""] + notes
        lines += ["", f"**第 1 步 · 微调前对照（{title}；从头 6000 步、未做闭环微调，同配方同数据流；流式 E1，去根对齐，mm；"
                      "描述）**"]
        for light, lname in LIGHTS:
            lines += ["", f"*{lname}*", ""] + ABL_HEAD
            for label, arm in BASES:
                text, m, per = abl_row(label, arm, subjects, light)
                lines.append(text)
                if per:
                    rows[f"{group}/{light}/base/{arm}"] = {"seeds": [s for s, _ in per], "mean": m,
                                                          "per_seed": {str(s): c for s, c in per}}
    lines += ["", "**第 1 步 · 闭环展开微调中的关联（训练日志；描述）**", ""] + rollout_log()
    (ns.OUT / "rows2.json").write_text(json.dumps(rows, indent=1) + "\n")
    text = "\n".join(lines).strip() + "\n"
    (ns.OUT / "table2.md").write_text(text)
    print(text)


def row(label, per, cost):
    m = mean(per)
    ra = f"{m['overall']['mpjpe_ra_mm']:.2f}" + seeds_txt(per, "overall", "mpjpe_ra_mm")
    tag = "" if len(per) == 3 else f"（{len(per)} 种子）"
    return (f"| {label}{tag} | {m['local']['mpjpe_ra_mm']:.2f} | {m['global']['mpjpe_ra_mm']:.2f} "
            f"| {m['local']['mpvpe_ra_mm']:.2f} | {m['global']['mpvpe_ra_mm']:.2f} | {ra} | {cost} |")


if __name__ == "__main__":
    main()
