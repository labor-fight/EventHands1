#!/usr/bin/env python3
"""Paper tables with route-first EvHandTrack everywhere (docs/EXPERIMENT_RECORDS.md#rfsu-20261113). CPU only; every
verb reads finished readings and leaves the published tables of the other studies untouched.

  sim   the simulated comparison table: CMP's rows (eho, epx, ehpe, fme, emrol5, eg; their main-row files as published)
        with EvHandTrack = route-first rfsr_rol5 (router r_s2dx2; seeds 3407 and 3408 -- user 2026-11-13 "种子2个就行" --,
        streaming E1 on zgz; cost from
        outputs/rfsu/20261113/sim/costs/pmr_rfsr_rol5_s3407__str once measured, else "-"), written as
        outputs/semkine/cmp_evhandtrack_rfsr_main_row.json (tools/report_table.py's schema, when the cost exists); and
        the figure's ablation table (Local / Global MPJPE / MPVPE, root-aligned; (a) w/o surface association, Global
        MLP update, State-independent gating, (b) Single-step only, Full model; the seeds each variant has, per seed in
        parentheses) plus the paired differences to the full model.
  real  tools/real/table4.py's two tables (kqj, lqh, wh pooled by endpoints, normal light) with EvHandTrack = rfsr_rol5
        and the ablation rows = the route-first r_s2dx2 arms.
  erh   tools/erh/table.py's tables (test per scene and all scenes, dev) with EvHandTrack = rfsr_rol5 (EvRealHands' own
        r_s2dx2); cost cells = the real study's rfsr_rol5 reading (tools/erh/table.py takes the real study's costs: same
        networks, same tensors).
The router is r_s2dx2 on every dataset (user 2026-11-13: "论文用 r_s2dx2，真实数据，仿真数据，和 EvRealHands ... 都换成
r_s2dx2"); the route-first rows with r_s2d (rfs_rol5) are listed under the tables where they exist, for the record.
  all   the three, each in its own process (the real and EvRealHands namespaces do not mix).
Each table also lists the original (sample-first) EvHandTrack in one line under it, for the record.
Output: outputs/rfsu/20261113/tables/<verb>.md (+ <verb>.json).

    python tools/rfsu/tables.py all
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

OUT = ROOT / "outputs/rfsu/20261113/tables"
SIM = ROOT / "outputs/rfsu/20261113/sim"
ROWS = ROOT / "outputs/semkine"
EVHT_LABEL = "EvHandTrack（先路由采样，路由器 r_s2dx2，运动学树 D1 + rol5，流式形式）"
FULL = "rfsr_rol5"
ABL = (("w/o surface association", "rfsrgeo_rol5"), ("Global MLP update", "rfsrm_rol5"),
       ("State-independent gating", "rfsrg0_rol5"), ("Single-step only", "rfsr_rol1"), ("Full model", FULL))
ABL_HEAD = ["| Configuration | Local MPJPE↓ | Local MPVPE↓ | Global MPJPE↓ | Global MPVPE↓ | 种子 |",
            "|---|---|---|---|---|---|"]
FIELDS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")


def write(name, lines, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).strip() + "\n"
    (OUT / f"{name}.md").write_text(text)
    (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(text)


def cost_cells(path):
    p = Path(path) / "report.json"
    if not p.exists():
        return "- | - | -", None
    c = json.loads(p.read_text())
    if c.get("status") != "COMPLETE":
        return "- | - | -", None
    return (f"{c['latency_row']['scaled_ms']:.2f} ms | {c['macs_layers']['median'] / 1e9:.3f} G "
            f"| {c['params_total'] / 1e6:.2f} M"), c


def abl_block(cells_of, key_local="local", key_global="global"):
    """The figure's ablation rows from `cells_of(arm) -> {seed: {local / global: {field: v}}}`."""
    lines = list(ABL_HEAD)
    full = cells_of(FULL)
    diffs = []
    for i, (label, arm) in enumerate(ABL):
        if i == 0:
            lines.append("| **(a) Model components** | | | | | |")
        if i == 3:
            lines.append("| **(b) Training strategy** | | | | | |")
        per = cells_of(arm)
        if not per:
            lines.append(f"| {label} | - | - | - | - | - |")
            continue
        seeds = sorted(per)
        cells = []
        for k in (key_local, key_global):
            for f in ("mpjpe_ra_mm", "mpvpe_ra_mm"):
                vals = [per[s][k][f] for s in seeds]
                cells.append(f"{sum(vals) / len(vals):.2f}" + ("（" + " / ".join(f"{v:.2f}" for v in vals) + "）"
                                                              if len(vals) > 1 else ""))
        lines.append(f"| {'**Full model**' if arm == FULL else label} | " + " | ".join(cells)
                     + f" | {' / '.join(str(s) for s in seeds)} |")
        if arm != FULL:
            both = [s for s in seeds if s in full]
            if both:
                d = []
                for k in (key_local, key_global):
                    for f in ("mpjpe_ra_mm", "mpvpe_ra_mm"):
                        v = [per[s][k][f] - full[s][k][f] for s in both]
                        d.append(f"{sum(v) / len(v):+.2f}（" + " / ".join(f"{x:+.2f}" for x in v) + "）")
                diffs.append(f"| {label} | " + " | ".join(d) + f" | {' / '.join(str(s) for s in both)} |")
    if diffs:
        lines += ["", "同种子配对差（变体 − 完整模型；均值，括号里逐种子）", ""] + ABL_HEAD + diffs
    return lines


# --------------------------------------------------------------------------------------------- simulated
def sim():
    from tools.cmp import rows as CR
    from tools.rfsu import sim as S

    def per_seed(arm):
        out = {}
        for s in S.SEEDS:
            p = S.report(arm, s)
            if p.exists():
                r = json.loads(p.read_text())
                if r.get("status") == "COMPLETE":
                    out[s] = {k: {f: r["per_sequence"][q][f] for f in FIELDS} for k, q in CR.SEQ.items()}
                    out[s]["overall"] = {f: r["overall"][f] for f in FIELDS}
                    out[s]["reading"] = str(p.parent.relative_to(ROOT))
        return out
    rows = {}
    lines = ["", "**仿真 zgz（流式 E1；EvHandTrack 为先路由，种子 3407 / 3408，其余方法三种子）**", "",
             "| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name in CR.MAIN[:-1]:
        p = ROWS / f"cmp_{name}_main_row.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        m = r["two_seed_mean"]
        ra = f"{m['overall']['mpjpe_ra_mm']:.2f}（" + " / ".join(
            f"{r['per_seed'][k]['overall']['mpjpe_ra_mm']:.2f}" for k in sorted(r["per_seed"])) + "）"
        lines.append(f"| {CR.LABELS[name]} | {m['local']['mpjpe_ra_mm']:.2f} | {m['global']['mpjpe_ra_mm']:.2f} "
                     f"| {m['local']['mpvpe_ra_mm']:.2f} | {m['global']['mpvpe_ra_mm']:.2f} | {ra} "
                     f"| {r['latency_ms_scaled_full1p75']:.2f} ms | {r['macs_forward_packet'] / 1e9:.3f} G "
                     f"| {r['params_total'] / 1e6:.2f} M |")
    per = per_seed(FULL)
    cost_txt, c = cost_cells(SIM / "costs" / f"pmr_{FULL}_s3407__str")
    if per:
        seeds = sorted(per)
        mean = {k: {f: sum(per[s][k][f] for s in seeds) / len(seeds) for f in FIELDS} for k in ("local", "global", "overall")}
        ra = f"{mean['overall']['mpjpe_ra_mm']:.2f}（" + " / ".join(f"{per[s]['overall']['mpjpe_ra_mm']:.2f}" for s in seeds) + "）"
        lines.append(f"| {EVHT_LABEL} | {mean['local']['mpjpe_ra_mm']:.2f} | {mean['global']['mpjpe_ra_mm']:.2f} "
                     f"| {mean['local']['mpvpe_ra_mm']:.2f} | {mean['global']['mpvpe_ra_mm']:.2f} | {ra} | {cost_txt} |")
        rows["evhandtrack_rfsr"] = {"seeds": seeds, "mean": mean, "per_seed": {str(s): per[s] for s in seeds}}
        if c is not None and seeds == [3407, 3408]:
            from tools.cmp.common import now
            row = {"arm": "cmp_evhandtrack_rfsr", "label": EVHT_LABEL, "split": "val_core", "n_seeds": 2, "form": "main",
                   "two_seed_mean": mean, "per_seed": {str(i): per[s] for i, s in enumerate(seeds)},
                   "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
                   "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"],
                   "macs_forward_packet": c["macs_layers"]["median"], "params_total": c["params_total"],
                   "latency_e2e_ms": c["latency_e2e_ms"],
                   "cost_reading": str((SIM / "costs" / f"pmr_{FULL}_s3407__str").relative_to(ROOT)),
                   "written_utc": now(),
                   "note": "RFSU (#rfsu-20261113): route-first EvHandTrack rfsr_rol5 (simulated r_s2dx2), PFO engine E1 on zgz in the streaming "
                           "form, seeds 3407 and 3408 (user 2026-11-13: two seeds are enough; 'two_seed_mean' is their "
                           "mean); cost = seed 3407, "
                           "deployable route-first step (tools/rfsu/cost.py)"}
            (ROWS / "cmp_evhandtrack_rfsr_main_row.json").write_text(json.dumps(row, indent=1) + "\n")
    orig = ROWS / "cmp_evhandtrack_main_row.json"
    if orig.exists():
        r = json.loads(orig.read_text())
        m = r["two_seed_mean"]
        lines += ["", f"记录：原版 EvHandTrack（先采样再路由）{m['local']['mpjpe_ra_mm']:.2f} / {m['global']['mpjpe_ra_mm']:.2f} / "
                      f"{m['local']['mpvpe_ra_mm']:.2f} / {m['global']['mpvpe_ra_mm']:.2f} / RA {m['overall']['mpjpe_ra_mm']:.2f}，"
                      f"{r['latency_ms_scaled_full1p75']:.2f} ms / {r['macs_forward_packet'] / 1e9:.3f} G / "
                      f"{r['params_total'] / 1e6:.2f} M。"]
    lines += ["", "**仿真消融（zgz；训练结果，流式 E1，去根对齐，mm）**", ""]
    cache = {}

    def cells_of(arm):
        if arm not in cache:
            cache[arm] = per_seed(arm)
        return cache[arm]
    lines += abl_block(cells_of)
    rows["abl"] = {arm: {str(s): v for s, v in cells_of(arm).items()} for _, arm in ABL}
    write("sim", lines, rows)


# --------------------------------------------------------------------------------------------- real
def real():
    from tools.real import rfs as RFS
    X = RFS.extend()
    from tools.real import ns
    from tools.real import table as TB
    from tools.real import table2 as T2
    from tools.real import table4 as T4
    rows, lines = {}, ["", f"**{T4.TITLE}（EvHandTrack 为先路由）**", "", TB.HEADER]
    for arm in TB.ARMS:
        per = T2.per_seed([(s, ns.OUT / "readings" / f"cmp_{arm}_s{s}" / "e1" / "report.json") for s in ns.SEEDS],
                          T4.SUBJECTS, T4.LIGHT)
        if per:
            lines.append(T2.row(TB.LABELS[arm], per, TB.cost(arm)))
            rows[arm] = {"seeds": [s for s, _ in per], "mean": T2.mean(per)}
    per = T2.per_seed([(s, X.report(FULL, s)) for s in ns.SEEDS], T4.SUBJECTS, T4.LIGHT)
    cost_txt, _ = cost_cells(ns.OUT / "costs" / f"pmr_{FULL}_s3407__str")
    if per:
        lines.append(T2.row(EVHT_LABEL.replace("流式形式", "整幅 346×260，流式形式"), per, cost_txt))
        rows["evhandtrack_rfsr"] = {"seeds": [s for s, _ in per], "mean": T2.mean(per)}
    per = T2.per_seed([(s, X.report("rfs_rol5", s)) for s in ns.SEEDS], T4.SUBJECTS, T4.LIGHT)
    if per:
        m = T2.mean(per)
        c_txt, _ = cost_cells(ns.OUT / "costs" / "pmr_rfs_rol5_s3407__str")
        lines += ["", f"记录：先路由 + 路由器 r_s2d（rfs_rol5）{m['local']['mpjpe_ra_mm']:.2f} / {m['global']['mpjpe_ra_mm']:.2f} / "
                      f"{m['local']['mpvpe_ra_mm']:.2f} / {m['global']['mpvpe_ra_mm']:.2f} / RA {m['overall']['mpjpe_ra_mm']:.2f}；"
                      f"成本 {c_txt}。"]
    per = T2.per_seed([(s, X.report("d1r_rol5", s)) for s in ns.SEEDS], T4.SUBJECTS, T4.LIGHT)
    if per:
        m = T2.mean(per)
        lines += ["", f"记录：原版 EvHandTrack（先采样再路由）{m['local']['mpjpe_ra_mm']:.2f} / {m['global']['mpjpe_ra_mm']:.2f} / "
                      f"{m['local']['mpvpe_ra_mm']:.2f} / {m['global']['mpvpe_ra_mm']:.2f} / RA {m['overall']['mpjpe_ra_mm']:.2f}；"
                      f"成本 {T2.evht_cost()}。"]
    lines += ["", f"**消融（{T4.TITLE}；训练结果，流式 E1，去根对齐，mm）**", ""]

    def cells_of(arm):
        return {s: c for s, c in T2.per_seed([(s, X.report(arm, s)) for s in ns.SEEDS], T4.SUBJECTS, T4.LIGHT)}
    lines += abl_block(cells_of)
    notes = [f"- {label}，种子 {s}：读数失败（{err}），不计入均值。" for label, arm in ABL for s, err in T2.failed(arm)]
    if notes:
        lines += [""] + notes
    rows["abl"] = {arm: {str(s): v for s, v in cells_of(arm).items()} for _, arm in ABL}
    write("real", lines, rows)


# --------------------------------------------------------------------------------------------- EvRealHands
def erh():
    from tools.erh import table as ET            # installs the EvRealHands namespace first thing
    import numpy as np
    from tools.real import table as TB
    R = ET.R
    native_reading = ET.reading
    ET.reading = lambda arm, seed: (ET.X.report(FULL, seed).parent if arm == "evht_rfs" else
                                    ET.X.report("rfs_rol5", seed).parent if arm == "evht_rfs_r1" else native_reading(arm, seed))
    real_cost, _ = cost_cells(ET.REAL_OUT / "costs" / f"pmr_{FULL}_s3407__str")
    arms = ("eho", "epx", "ehpe", "fme", "evht_rfs")
    labels = {**ET.LABELS, "evht_rfs": EVHT_LABEL.replace("流式形式", "整幅 346×260，流式形式；路由器在 EvRealHands 训练集上训练")}
    rows, lines = {}, []
    groups = [(f"测试（受试者 1、3）· {s}", [q for q in R.TEST if ET.info(q)["scene"] == s]) for s in ET.SCENES]
    groups += [("测试（受试者 1、3）· 全部场景", list(R.TEST)), ("调参（受试者 7，均为 normal；描述）", list(R.DEV))]
    for title, seqs in groups:
        lines += ["", f"**{title}**（{len(seqs)} 条序列；只在标注时刻计分；EvHandTrack 为先路由）", "", TB.HEADER]
        for arm in arms:
            per = ET.per_seed(arm, seqs)
            if not per:
                continue
            m = {k: {f: float(np.mean([c[k][f] for _, c in per])) for f in ET.FIELDS} for k in ("local", "global", "overall")}
            rows[f"{title}/{arm}"] = {"seeds": [s for s, _ in per], "mean": m}
            ra = f"{m['overall']['mpjpe_ra_mm']:.2f}" + ("（" + " / ".join(
                f"{c['overall']['mpjpe_ra_mm']:.2f}" for _, c in per) + "）" if len(per) > 1 else "")
            tag = "" if len(per) == 3 else f"（{len(per)} 种子）"
            cost = real_cost if arm == "evht_rfs" else ET.cost(arm)
            lines.append(f"| {labels[arm]}{tag} | {m['local']['mpjpe_ra_mm']:.2f} | {m['global']['mpjpe_ra_mm']:.2f} "
                         f"| {m['local']['mpvpe_ra_mm']:.2f} | {m['global']['mpvpe_ra_mm']:.2f} | {ra} | {cost} |")
        per = ET.per_seed("evht_rfs_r1", seqs)
        if per:
            m = {k: {f: float(np.mean([c[k][f] for _, c in per])) for f in ET.FIELDS} for k in ("local", "global", "overall")}
            lines += ["", f"记录：先路由 + 路由器 r_s2d（rfs_rol5，{len(per)} 种子）{m['local']['mpjpe_ra_mm']:.2f} / "
                          f"{m['global']['mpjpe_ra_mm']:.2f} / {m['local']['mpvpe_ra_mm']:.2f} / {m['global']['mpvpe_ra_mm']:.2f} / "
                          f"RA {m['overall']['mpjpe_ra_mm']:.2f}。"]
        per = ET.per_seed("evht", seqs)
        if per:
            m = {k: {f: float(np.mean([c[k][f] for _, c in per])) for f in ET.FIELDS} for k in ("local", "global", "overall")}
            lines += ["", f"记录：原版 EvHandTrack（先采样再路由）{m['local']['mpjpe_ra_mm']:.2f} / {m['global']['mpjpe_ra_mm']:.2f} / "
                          f"{m['local']['mpvpe_ra_mm']:.2f} / {m['global']['mpvpe_ra_mm']:.2f} / RA {m['overall']['mpjpe_ra_mm']:.2f}。"]
    write("erh", lines, rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("verb", choices=("sim", "real", "erh", "all"))
    a = ap.parse_args()
    if a.verb == "all":
        for v in ("sim", "real", "erh"):
            subprocess.run([sys.executable, __file__, v], check=False)
    else:
        {"sim": sim, "real": real, "erh": erh}[a.verb]()


if __name__ == "__main__":
    main()
