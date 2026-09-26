#!/usr/bin/env python3
"""Draw S37 routed readout in the S36 simple horizontal-strip template.

Architecture only: dimensions come from the S37 config and MANO asset.
The event graph and its pooled global feature are separate from FK routing.
Run: python tools/draw_s37_routed_s36_template.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, FancyBboxPatch

from make_s36_figure_simple import (
    BLUE, CAP, PANEL, TEAL, arrow, cube_icon, elbow, events_icon,
    funnel_icon, graph_icon, grid_icon, hand_skel, node, stack_icon,
)


REPO = Path(__file__).resolve().parents[1]


def main():
    cfg = yaml.safe_load((REPO / "configs/semkine/s37_routed_s3407.yaml").read_text())
    mc = cfg["MODEL"]
    assert mc["ENCODER"] == "event_gnn" and mc["ROUTED_READOUT"]
    assert not mc["PREV_RENDER"] and mc["PREVPOS_EMBED"]
    assert mc["ACTIVE_HEAD"] and mc["PREDICT_DELTA"] and mc["ZERO_EVENT_GATE"]
    assert not mc.get("ROOT_COVMAP", False) and not mc.get("EGM_MEMORY", False)
    with np.load(REPO / cfg["MANO"]["NPZ"]) as mano:
        vertices, joints = mano["weights"].shape
    hidden, layers = mc["ENCODER_HIDDEN"], mc["ENCODER_LAYERS"]
    feat = mc["ENCODER_FEAT"]
    evidence = 2 * hidden + 1
    output = mc["OUTPUT_DIM"]

    # Reuse the original S36 canvas, icons, colours and left-to-right arrows.
    # Extra lower space makes the existing root/previous-state branches visible.
    fig, ax = plt.subplots(figsize=(25.5, 4.8))
    ax.set_xlim(0, 321)
    ax.set_ylim(0, 60)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (1, 1), 319, 58, boxstyle="round,pad=0.3,rounding_size=2.5",
        facecolor=PANEL, edgecolor="#e5e7eb", lw=1.0, zorder=0,
    ))
    ax.text(4, 57.3, "S37 路由读出 — 事件图卷积 → MANO 几何路由 → 逐关节证据 → Δ",
            fontsize=8.8, color=CAP, va="top", zorder=6)

    # Upper FK branch: projected mesh vertices supply node-to-joint routing.
    node(ax, 4, 28, 17, 10, f"prev {output}D\nMANO", hand_skel, TEAL, fs=8.2)
    node(ax, 25, 43, 16, 10, "MANO FK", hand_skel, TEAL,
         caption=f"{vertices} 顶点")
    node(ax, 44, 43, 18, 10, "相机投影\n(u, v, z)", grid_icon, TEAL, fs=8.2)
    node(ax, 65, 43, 30, 10, "前表面顶点匹配\n读取 LBS 路由权重", graph_icon, TEAL, fs=8.0)
    elbow(ax, [(12.5, 38.4), (12.5, 48), (24.6, 48)])
    arrow(ax, (41.4, 48), (43.6, 48))
    arrow(ax, (62.4, 48), (64.6, 48))
    elbow(ax, [(95.4, 48), (182, 48), (182, 38.4)])
    ax.text(137, 49.5, f"路由矩阵 A：N × {joints}；距离 ≤ {mc['ROUTE_BAND_PX']:g} px",
            fontsize=7.2, color=TEAL, ha="center", va="bottom", zorder=6)

    # Input events and the state-independent event graph, as in the S36 strip.
    node(ax, 58, 17, 23, 10, "异步事件流\n(x, y, t, p)", events_icon, BLUE, fs=8.2)
    node(ax, 91, 28, 21, 10, "逐事件\ntoken 7 维", stack_icon, TEAL, fs=8.2,
         caption="图节点 = 采样后的事件")
    node(ax, 116, 28, 22, 10, "因果 k-NN\n建图", graph_icon, BLUE, fs=8.2,
         caption=f"前 {mc['ENCODER_WINDOW']} 取 k={mc['ENCODER_K']} · ≤{mc['ENCODER_MAX_NODES']} 点")
    node(ax, 142, 28, 20, 10, f"EdgeConv × {layers}\nH：N × {hidden}",
         cube_icon, BLUE, fs=7.7)
    node(ax, 167, 28, 30, 10, "路由池化\n逐关节证据", funnel_icon, TEAL, fs=8.2,
         caption=f"加权 mean / max / 覆盖率 → {joints} × {evidence}")
    node(ax, 204, 28, 36, 10, f"{joints - 1} 个关节头\n+ root 头", hand_skel, TEAL, fs=8.4,
         caption=f"root 6D + {joints - 1} × 3D")
    node(ax, 247, 28, 16, 10, "合并 → Δ", None, fs=10)
    elbow(ax, [(81.4, 22), (94, 22), (94, 27.6)])
    for start, end in ((112.4, 115.6), (138.4, 141.6), (162.4, 166.6),
                       (197.4, 203.6), (240.4, 246.6)):
        arrow(ax, (start, 33), (end, 33))

    # Routing uses the retained event-node pixels, not graph features as geometry.
    elbow(ax, [(152, 38.4), (152, 41), (80, 41), (80, 42.6)], lw=1.1)
    ax.text(121, 41.6, "采样事件节点的像素坐标", fontsize=6.7,
            color=CAP, ha="center", va="bottom", zorder=6)

    # Global event pooling is a parallel root-only input.
    node(ax, 142, 14, 33, 8, f"全局 mean / max\nMLP → {feat} 维", funnel_icon, TEAL, fs=8.0)
    arrow(ax, (152, 27.6), (152, 22.4))
    elbow(ax, [(175.4, 18), (211, 18), (211, 27.6)])
    ax.text(193, 19.2, "仅送 root", fontsize=7.0, color=TEAL,
            ha="center", va="bottom", zorder=6)
    ax.text(205, 51.5, f"root：全局 {feat} + 全部 {joints} × {evidence} 证据 → 6D",
            fontsize=7.0, color=CAP, va="center", zorder=6)
    ax.text(205, 47.7, f"每个关节头：自身 {evidence} 维证据 + prev 角 3D → 3D",
            fontsize=7.0, color=CAP, va="center", zorder=6)
    ax.text(255, 40.2, f"Δ {output}D；空事件包 Δ = 0", fontsize=7.0,
            color=CAP, ha="center", va="center", zorder=6)

    # Explicit previous-angle and prev-MLP inputs, before the empty-packet gate.
    node(ax, 225, 11, 26, 8, f"prev_mlp\n{output} → 64 → {output}", stack_icon, TEAL, fs=8.0)
    arrow(ax, (238, 6.5), (238, 10.6), lw=1.2)
    elbow(ax, [(251.4, 15), (255, 15), (255, 27.6)], lw=1.2)
    elbow(ax, [(219, 6.5), (219, 22), (231, 22), (231, 27.6)], lw=1.1)
    ax.text(224.5, 20.8, "自身 prev 角", fontsize=6.5,
            color=CAP, ha="center", va="top", zorder=6)

    # Residual state update, preserving the original S36 bottom feedback layout.
    ax.add_patch(Circle((270, 33), 1.7, facecolor="white", edgecolor=TEAL,
                        lw=1.4, zorder=5))
    ax.text(270, 33, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    node(ax, 276, 28, 39, 10, "s(k) = prev + Δ", None, fs=9.0)
    arrow(ax, (263.4, 33), (268.2, 33))
    arrow(ax, (271.8, 33), (275.6, 33))
    elbow(ax, [(12.5, 27.6), (12.5, 6.5), (270, 6.5), (270, 31.2)])
    ax.text(145, 3.3, "prev（本步输出作为下一包 prev，再做 FK 路由；同时提供关节前角与 prev_mlp 输入）",
            ha="center", va="center", fontsize=7.2, color=TEAL, zorder=6)

    target = REPO / "docs/assets/s37_routed_s36_template"
    target.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        path = target.with_suffix(suffix)
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
        print(path.relative_to(REPO))
    plt.close(fig)


if __name__ == "__main__":
    main()
