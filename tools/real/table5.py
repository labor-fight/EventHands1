#!/usr/bin/env python3
"""Route-first EvHandTrack in the paper's real-data table (docs/EXPERIMENT_RECORDS.md#real-rfs-20261112).

tools/real/table4.py's comparison table (three held-out subjects kqj, lqh, wh; normal light; pooled by endpoints;
three-seed means, the RA cell with the seeds) with the route-first EvHandTrack rows added under the original one:
  rfs_rol5    EvHandTrack + route-first sampling (residual slot: the hand's vicinity), the study's router r_s2d
  rfsr_rol5   the same with the router chosen by the pre-registered rule (routers/chosen.json), when read
Cost cells: the original EvHandTrack's from the study's cost block; the route-first rows "-" until their own cost
reading exists (costs/<run>__str/report.json). Writes table5.md.

    python tools/real/table5.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.real import rfs as RFS  # noqa: E402
from tools.real import ns  # noqa: E402
from tools.real import table as TB  # noqa: E402
from tools.real import table2 as T2  # noqa: E402
from tools.real import table4 as T4  # noqa: E402

LABELS = {"rfs_rol5": "EvHandTrack + 先路由采样（残差槽只收手边事件，路由器 r_s2d）",
          "rfsr_rol5": "EvHandTrack + 先路由采样 + 新路由器（{variant}）"}


def cost(arm):
    p = ns.OUT / "costs" / f"pmr_{arm}_s3407__str" / "report.json"
    if not p.exists():
        return "- | - | -"
    c = json.loads(p.read_text())
    return (f"{c['latency_row']['scaled_ms']:.2f} ms | {c['macs_layers']['median'] / 1e9:.3f} G "
            f"| {c['params_total'] / 1e6:.2f} M") if c.get("status") == "COMPLETE" else "- | - | -"


def cost_note():
    """The route-first rows' latency puts the node selection inside the step (tools/real/rfs_cost.py); the original
    EvHandTrack's two definitions re-measured in the same block, when that block has run."""
    def ms(tag):
        p = ns.OUT / "costs" / tag / "report.json"
        if not p.exists():
            return None
        c = json.loads(p.read_text())
        return c["latency_row"]["scaled_ms"] if c.get("status") == "COMPLETE" else None
    step, staging = ms("pmr_d1r_rol5_s3407__str__step_rfsblk"), ms("pmr_d1r_rol5_s3407__str__staging_rfsblk")
    if step is None or staging is None or cost("rfs_rol5").startswith("-"):
        return None
    return (f"注：先路由各行的时延把节点选择算在每步之内（选择要用路由器对包内全部事件的判断，不能放进预处理）；"
            f"路由器每包只算一次，节点的路由直接取用。同一成本块里，原版 EvHandTrack 同口径为 {step:.2f} ms，"
            f"按表中原版的口径（节点选择算在预处理里）为 {staging:.2f} ms。")


def main():
    X = RFS.extend()
    lines = ["", f"**{T4.TITLE}**", "", TB.HEADER]
    for arm in TB.ARMS:
        per = T2.per_seed([(s, ns.OUT / "readings" / f"cmp_{arm}_s{s}" / "e1" / "report.json") for s in ns.SEEDS],
                          T4.SUBJECTS, T4.LIGHT)
        if per:
            lines.append(T2.row(TB.LABELS[arm], per, TB.cost(arm)))
    per = T2.per_seed([(s, X.report(T2.EVHT, s)) for s in ns.SEEDS], T4.SUBJECTS, T4.LIGHT)
    lines.append(T2.row(T2.EVHT_LABEL + "（原版）", per, T2.evht_cost()))
    for arm in ("rfs_rol5", "rfsr_rol5"):
        per = T2.per_seed([(s, X.report(arm, s)) for s in ns.SEEDS], T4.SUBJECTS, T4.LIGHT)
        if per:
            label = LABELS[arm]
            if arm == "rfsr_rol5":
                label = label.format(variant=json.loads(RFS.CHOSEN.read_text())["variant"])
            lines.append(T2.row(label, per, cost(arm)))
    note = cost_note()
    if note:
        lines += ["", note]
    text = "\n".join(lines).strip() + "\n"
    (ns.OUT / "table5.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
