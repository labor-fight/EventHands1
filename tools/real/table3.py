#!/usr/bin/env python3
"""Real-study comparison table with both lights pooled (docs/EXPERIMENT_RECORDS.md#real-amend5-run-20261111).

User 2026-11-12: the real-data metrics of EventHands, EvHandPose, EventHPE, FastMETRO-Event and EvHandTrack, "正常光照+黑暗光照
求均值就行". tools/real/table2.py's comparison table (same rows, labels, cost cells, three-seed means, RA cell with the
seeds) with light = "both": each column pooled over the subjects' sequences of both lights, weighted by endpoints, as in
table2.py's "正常光 + 暗光" ablation table, so the EvHandTrack row equals that table's Full model row. A note gives the
endpoint counts per light and the largest difference to the equal-weight mean of the normal and dark rows.
Writes table3.md and rows3.json next to the readings; table.py / table2.py are imported unchanged.

    python tools/real/table3.py
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

COLS = (("local", "mpjpe_ra_mm", "MPJPE-local"), ("global", "mpjpe_ra_mm", "MPJPE-global"),
        ("local", "mpvpe_ra_mm", "MPVPE-local"), ("global", "mpvpe_ra_mm", "MPVPE-global"),
        ("overall", "mpjpe_ra_mm", "RA"))


def reports(arm):
    if arm == "evhandtrack":
        return [(s, X.report(T2.EVHT, s)) for s in ns.SEEDS]
    return [(s, ns.OUT / "readings" / f"cmp_{arm}_s{s}" / "e1" / "report.json") for s in ns.SEEDS]


def counts(subjects):
    """Endpoints per light and motion class, from the first complete EvHandTrack reading (equal for every arm)."""
    for _, p in reports("evhandtrack"):
        if p.exists() and json.loads(p.read_text()).get("status") == "COMPLETE":
            b = json.loads(p.read_text())["per_sequence"]
            return {(m, l): sum(b[f"{s}_{m}{'_dark' if l == 'dark' else ''}"]["n"] for s in subjects)
                    for m in ("local", "global") for l in ("normal", "dark")}
    return {}


def main():
    rows, lines = {}, []
    for group, subjects in (("test", ("kqj", "lqh")), ("dev", ("wh",))):
        title = "测试 kqj、lqh" if group == "test" else "调参 wh（描述）"
        lines += ["", f"**{title} · 正常光 + 暗光**", "", TB.HEADER]
        worst = (0.0, "", "")
        for arm in TB.ARMS + ("evhandtrack",):
            per = T2.per_seed(reports(arm), subjects, "both")
            if not per:
                continue
            label, cost = (T2.EVHT_LABEL, T2.evht_cost()) if arm == "evhandtrack" else (TB.LABELS[arm], TB.cost(arm))
            lines.append(T2.row(label, per, cost))
            m = T2.mean(per)
            rows[f"{group}/both/{arm}"] = {"seeds": [s for s, _ in per], "mean": m,
                                          "per_seed": {str(s): c for s, c in per}}
            light = {l: T2.mean(T2.per_seed(reports(arm), subjects, l)) for l in ("normal", "dark")}
            for k, f, name in COLS:
                d = abs(m[k][f] - (light["normal"][k][f] + light["dark"][k][f]) / 2)
                if d > worst[0]:
                    worst = (d, label.split("（")[0], name)
        n = counts(subjects)
        if n:
            lines += ["", f"- 两种光照按标注帧数合并：local 正常光 {n['local', 'normal']} / 暗光 {n['local', 'dark']} 帧，"
                          f"global 正常光 {n['global', 'normal']} / 暗光 {n['global', 'dark']} 帧；与两行等权平均最多相差 "
                          f"{worst[0]:.2f} mm（{worst[1]}，{worst[2]}）。"]
    (ns.OUT / "rows3.json").write_text(json.dumps(rows, indent=1) + "\n")
    text = "\n".join(lines).strip() + "\n"
    (ns.OUT / "table3.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
