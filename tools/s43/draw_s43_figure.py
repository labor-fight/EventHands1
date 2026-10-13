#!/usr/bin/env python3
"""S43 in the S36/S37 flow-strip template: the event-driven sparse render-and-compare tracker, changes against S39
boxed in orange. Banner numbers are read from the main-row / report JSONs when they exist (nothing is typed in).

    python tools/s43/draw_s43_figure.py          # -> docs/assets/s43_sparse_tracker_20261009.png / .svg
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from make_s36_figure_simple import (BLUE, CAP, TEAL, arrow, cube_icon, elbow, events_icon, funnel_icon,   # noqa: E402
                                    grid_icon, hand_skel, node, stack_icon)
from make_s37_figure_simple import ORANGE, canvas, hnode                                                   # noqa: E402


def row(name):
    p = REPO / f"outputs/semkine/{name}_main_row.json"
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    s43, s39 = row("s43n_sp_w200cfg"), row("dt3_dz_l3_w200cfg")
    rep = REPO / "outputs/s43/reports/static_trunk_s43n_sp_s3407_fp32.json"
    asy = REPO / "outputs/s43/async/s43n_sp_s3407_graph_h50_fp32_nomesh.json"
    banner = "橙框 = 相对 S39 的改动：事件逐个写入时间面、任意时刻查询 · 主干只在活跃位点上算 · 更新死区 · 增量 mesh"
    if s43 and s39:
        a, b = s43["two_seed_mean"]["overall"]["mpjpe_ra_mm"], s39["two_seed_mean"]["overall"]["mpjpe_ra_mm"]
        banner += (f"  |  三种子递推 RA {a:.2f}（S39 {b:.2f}）· 实际 MAC {s43['macs_forward_packet'] / 1e9:.3f} G"
                   f"（S39 {s39['macs_forward_packet'] / 1e9:.3f} G）")
    W, Hc = 248, 66
    fig, ax = canvas("S43 稀疏渲染–比对事件跟踪 — 事件与上一步的渲染手部进同一个稀疏卷积网络；状态 51D，每次查询一次稀疏前向",
                     banner, width=W, height=Hc)
    sub = ("LNES(t)[x,y,p] = (t_last − t_0) / w，t_0 = t − w + 1，w = min(200 ms, 段龄)；活跃位点 A0 = {LNES 或渲染非零的像素}，"
           "A_{l+1} = A_l 的 2×2 格占据；主干每层只读写活跃位点，BatchNorm 统计只取活跃位点；"
           "x̂ = prev + fc([mean_{A4} h ‖ Σ_{A4} h / |格|]) + prev_mlp(prev)")
    if rep.exists() and asy.exists():
        r, q = json.loads(rep.read_text()), json.loads(asy.read_text())
        g = r["ms"]["static_graph_auto_bucket"]
        sub += (f"\n部署形态：活跃集按容量三档、整条查询链录成 CUDA graph：主干 {g['global']:.2f} / {g['local']:.2f} ms"
                f"（global / local），整步查询 p50 {q['latency_query_ms']['p50']:.2f} ms；CPU 8 线程稀疏 5.5 ms 对稠密 9.7 ms")
    ax.text(W / 2, Hc - 6.2, sub, fontsize=6.4, color=CAP, ha="center", va="top", zorder=6, linespacing=1.5)

    Y3, Y2, Y1 = 40, 24, 8                 # state row, main chain, event row
    # ---- state branch (top row): prev -> FK -> splat, as in S39
    node(ax, 4, Y2, 17, 10, "prev 51D\nMANO", hand_skel, TEAL, fs=8.2)
    node(ax, 27, Y3, 16, 10, "MANO FK", hand_skel, TEAL, caption="778 顶点（与 S39 相同）")
    node(ax, 47, Y3, 24, 10, "点溅射渲染\nsil ‖ 逆深度", grid_icon, TEAL, fs=8.2, caption="240×180 · 2 面 · 无梯度")
    elbow(ax, [(12.5, Y2 + 10.4), (12.5, Y3 + 5), (26.6, Y3 + 5)])
    arrow(ax, (43.4, Y3 + 5), (46.6, Y3 + 5))
    node(ax, 158, Y3, 20, 10, "prev_mlp\n51 → 64 → 51", None, fs=8.0, caption="事件盲的前态修正（与 S39 相同）")
    elbow(ax, [(12.5, Y3 + 5), (12.5, Y3 + 13), (168, Y3 + 13), (168, Y3 + 10.4)], lw=1.0)

    # ---- event branch (bottom row): asynchronous events -> SAE -> LNES at query time
    node(ax, 27, Y1, 21, 10, "异步事件流\n(x, y, t, p)", events_icon, BLUE, fs=8.2)
    hnode(ax, 52, Y1, 25, 10, "逐像素时间面 SAE\n每事件 O(1) 写入", stack_icon, ORANGE, fs=7.8,
          caption="t_last[x, y, p] ← max(·, t)；0.02–0.2 µs / 事件")
    hnode(ax, 81, Y1, 29, 10, "查询时刻 t：LNES(t)\n= (t_last − t_0) / w", grid_icon, ORANGE, fs=7.6,
          caption="w ≤ 200 ms，裁段首 · 任意 t 可查（50 / 25 / 10 ms）")
    arrow(ax, (48.4, Y1 + 5), (51.6, Y1 + 5))
    arrow(ax, (77.4, Y1 + 5), (80.6, Y1 + 5))

    # ---- main chain (middle row): [LNES | render] -> active sites -> sparse trunk -> readout -> + -> hold
    hnode(ax, 84, Y2, 26, 10, "拼接 4 面 → 活跃位点\nA0 … A4（事件 ∪ 网格投影）", grid_icon, ORANGE, fs=7.4,
          caption="A0 中位 1879 像素（全帧 43200）")
    hnode(ax, 113, Y2, 31, 10, "稀疏 ResNet18-l3\n只在活跃位点 gather–GEMM", cube_icon, ORANGE, fs=7.8,
          caption="stem 7×7/2 · layer1–3 · MaskedBN · CoordConv · 补齐行不进池化")
    hnode(ax, 147, Y2, 25, 10, "读出 [A4 均值 ‖ 面积和]\n→ fc → Δ (51)", funnel_icon, ORANGE, fs=7.4,
          caption="面积和保留表观手大小（深度线索）")
    elbow(ax, [(71.4, Y3 + 5), (97, Y3 + 5), (97, Y2 + 10.4)])      # render -> concat (from above)
    arrow(ax, (97, Y1 + 10.4), (97, Y2 - 0.4))                        # LNES -> concat (from below)
    arrow(ax, (110.4, Y2 + 5), (112.6, Y2 + 5))
    arrow(ax, (144.4, Y2 + 5), (146.6, Y2 + 5))
    ax.add_patch(Circle((178, Y2 + 5), 1.7, facecolor="#ffffff", edgecolor=TEAL, lw=1.4, zorder=5))
    ax.text(178, Y2 + 5, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    arrow(ax, (172.4, Y2 + 5), (176.2, Y2 + 5))
    arrow(ax, (168, Y3 - 0.4), (178, Y2 + 6.8))                       # prev_mlp -> +
    node(ax, 182, Y2, 20, 10, "x̂ = prev + Δ\n零事件包：保持", None, fs=8.0)
    arrow(ax, (179.8, Y2 + 5), (181.6, Y2 + 5))

    # ---- post-processing (right column, downwards): S39-stack filter, dead band, output
    node(ax, 206, Y2, 36, 10, "事件量自适应滤波\n（S39 栈：ramp1000 spec）", funnel_icon, TEAL, fs=7.6,
          caption="增益按最近 50 ms 事件数；与 S39 相同")
    arrow(ax, (202.4, Y2 + 5), (205.6, Y2 + 5))
    hnode(ax, 206, Y1, 36, 10, "更新死区：手指 1°、根 0.5°\n小于阈值的关节保持前态", funnel_icon, ORANGE, fs=7.4,
          caption="local 每步只写 8.2 / 15 个手指关节 · 代价 +0.03 mm")
    arrow(ax, (224, Y2 - 0.4), (224, Y1 + 10.4))
    hnode(ax, 158, Y1, 42, 10, "x_k → 增量 mesh 缓存（输出网格）\n根 = 刚体变换；改动关节 = 子树 LBS", hand_skel, ORANGE, fs=7.2,
          caption="与全量 FK 偏差 ≤ 1e-7 m · 不在查询关键路径上")
    arrow(ax, (205.6, Y1 + 5), (200.4, Y1 + 5))

    # ---- the loop: x_k is the next query's prev
    elbow(ax, [(224, Y1 - 0.4), (224, 2.6), (12.5, 2.6), (12.5, Y2 - 0.4)], lw=1.5)
    ax.text(118, 3.2, "x_k 即下一次查询的 prev（状态只有 51 维；事件在两次查询之间持续写入 SAE，查询可以在任意时刻发起）",
            ha="center", va="bottom", fontsize=6.8, color=TEAL, zorder=6)

    out = REPO / "docs/assets/s43_sparse_tracker_20261009"
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
