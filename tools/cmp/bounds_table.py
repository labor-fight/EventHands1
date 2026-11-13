#!/usr/bin/env python3
"""Per-seed bounds table of the CMP methods (user 2026-11-10: baselines at their worst seed, EvHandTrack at its best
seed, unmeasured cells empty), in the nine columns of tools/report_table.py.

    python -m tools.cmp.bounds_table

A row is ONE seed of the method, picked by the overall RA-MPJPE of the main reading (worst = highest, best = lowest);
its accuracy cells are that seed's, the cost cells the method's cost reading (seed 3407's network; the same for every
seed). Accuracy comes from the CMP main row when it exists (outputs/semkine/cmp_<name>_main_row.json), else from the
seed readings (readings/cmp_<name>_s<seed>/e1/report.json, cost cells empty), else the row is empty.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.cmp.common import OUT, SEEDS, read  # noqa: E402

ROWS = ROOT / "outputs/semkine"
HEADER = ("| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |\n"
          "|---|---|---|---|---|---|---|---|---|")
BASELINE = "| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |"
#: (row name, display name, pick)
TABLE = (("eho", "EventHands（ICCV 2021 原版配方）", "worst"), ("epx", "EvHandPose（修复版，在线裁剪）", "worst"),
         ("ehpe", "EventHPE（在线跟踪）", "worst"), ("fme", "FastMETRO-Event", "worst"),
         ("evhandtrack", "EvHandTrack（运动学树 D1 + rol5，流式形式）", "best"))


def seeds_of(name):
    """[(seed label, {local, global, overall})] of the main reading, and the cost fields or None."""
    p = ROWS / f"cmp_{name}_main_row.json"
    if p.exists():
        r = read(p)
        per = [(k, v) for k, v in sorted(r["per_seed"].items())]
        labels = [str(s) for s in SEEDS] if len(per) == len(SEEDS) else [k for k, _ in per]
        return [(lab, v) for lab, (_, v) in zip(labels, per)], r
    out = []
    for s in SEEDS:
        q = OUT / "readings" / f"cmp_{name}_s{s}" / "e1" / "report.json"
        if not q.exists():
            return [], None
        r = read(q)
        if r["status"] != "COMPLETE":
            return [], None
        ps = r["per_sequence"]
        out.append((str(s), {"local": ps["zgz_local"], "global": ps["zgz_global"], "overall": r["overall"]}))
    return out, None


def line(name, label, pick):
    per, row = seeds_of(name)
    if not per:
        return f"| {label}（{'最差' if pick == 'worst' else '最好'}种子） |  |  |  |  |  |  |  |  |"
    key = lambda x: x[1]["overall"]["mpjpe_ra_mm"]  # noqa: E731
    seed, v = (max if pick == "worst" else min)(per, key=key)
    cost = ("", "", "")
    if row is not None:
        cost = (f"{row['latency_ms_scaled_full1p75']:.2f} ms", f"{row['macs_forward_packet'] / 1e9:.3f} G",
                f"{row['params_total'] / 1e6:.2f} M")
    tag = "最差" if pick == "worst" else "最好"
    return (f"| {label}（{tag}种子 {seed}） | {v['local']['mpjpe_ra_mm']:.2f} | {v['global']['mpjpe_ra_mm']:.2f} "
            f"| {v['local']['mpvpe_ra_mm']:.2f} | {v['global']['mpvpe_ra_mm']:.2f} | {v['overall']['mpjpe_ra_mm']:.2f} "
            f"| {cost[0]} | {cost[1]} | {cost[2]} |")


def main():
    print(HEADER)
    print(BASELINE)
    for name, label, pick in TABLE:
        print(line(name, label, pick))


if __name__ == "__main__":
    main()
