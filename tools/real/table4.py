#!/usr/bin/env python3
"""Real-study paper tables over the three held-out subjects (docs/EXPERIMENT_RECORDS.md#real-amend5-run-20261111).

User 2026-11-12: the paper uses normal light only ("放在论文里的话，就用正常光照就行了") and the three held-out subjects
together ("你三位受试者算平均就行"): kqj, lqh (test) and wh (dev; no method was tuned on it, every recipe ran as
registered). Both tables of tools/real/table2.py -- the comparison table (same rows, labels, cost cells; EvHandTrack =
the original d1r_rol5 streaming reading) and the paper's ablation table (figure format) -- with every column pooled
over the normal-light local / global sequences of kqj, lqh and wh, weighted by endpoints (tools/real/table.py's rule
within a group), three-seed means with the seeds in parentheses. Writes table4.md and rows4.json; table.py / table2.py
are imported unchanged.

    python tools/real/table4.py
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
from tools.real import table2 as T2  # noqa: E402

SUBJECTS = ("kqj", "lqh", "wh")
LIGHT = "normal"
TITLE = "留出受试者 kqj、lqh、wh · 正常光"


def main():
    rows, lines = {}, ["", f"**{TITLE}**", "", TB.HEADER]
    for arm in TB.ARMS:
        per = T2.per_seed([(s, ns.OUT / "readings" / f"cmp_{arm}_s{s}" / "e1" / "report.json") for s in ns.SEEDS],
                          SUBJECTS, LIGHT)
        if per:
            lines.append(T2.row(TB.LABELS[arm], per, TB.cost(arm)))
            rows[arm] = {"seeds": [s for s, _ in per], "mean": T2.mean(per), "per_seed": {str(s): c for s, c in per}}
    per = T2.per_seed([(s, X.report(T2.EVHT, s)) for s in ns.SEEDS], SUBJECTS, LIGHT)
    if per:
        lines.append(T2.row(T2.EVHT_LABEL, per, T2.evht_cost()))
        rows["evhandtrack"] = {"seeds": [s for s, _ in per], "mean": T2.mean(per), "per_seed": {str(s): c for s, c in per}}
    lines += ["", f"**消融（{TITLE}；训练结果，流式 E1，去根对齐，mm）**", ""] + T2.ABL_HEAD
    for i, (label, arm) in enumerate(X.PAPER):
        if i == 0:
            lines.append("| **(a) Model components** | | | | | |")
        if i == 3:
            lines.append("| **(b) Training strategy** | | | | | |")
        per = T2.per_seed([(s, X.report(arm, s)) for s in ns.SEEDS], SUBJECTS, LIGHT)
        if not per:
            lines.append(f"| {label} | - | - | - | - | - |")
            continue
        m = T2.mean(per)
        rows[f"abl/{arm}"] = {"seeds": [s for s, _ in per], "mean": m, "per_seed": {str(s): c for s, c in per}}
        cells = [f"{m[k][f]:.2f}{T2.seeds_txt(per, k, f)}" for k in ("local", "global") for f in ("mpjpe_ra_mm", "mpvpe_ra_mm")]
        lines.append(f"| {'**Full model**' if arm == T2.EVHT else label} | " + " | ".join(cells)
                     + f" | {' / '.join(str(s) for s, _ in per)} |")
    notes = [f"- {'Full model' if arm == T2.EVHT else label}，种子 {s}：读数失败（{err}），不计入均值。"
             for label, arm in X.PAPER for s, err in T2.failed(arm)]
    if notes:
        lines += [""] + notes
    (ns.OUT / "rows4.json").write_text(json.dumps(rows, indent=1) + "\n")
    text = "\n".join(lines).strip() + "\n"
    (ns.OUT / "table4.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
