#!/usr/bin/env python3
"""S37 Mesh Graph in the existing S37 horizontal-strip visual template.

Architecture only. Reuses the existing node, icon, color and connector helpers;
keeps the background and previous-state inputs visible in the compact layout.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
import yaml

from make_s36_figure_simple import (
    BLUE, CAP, PANEL, TEAL, arrow, cube_icon, elbow, events_icon,
    funnel_icon, graph_icon, grid_icon, hand_skel, node, stack_icon,
)
from make_s37_fkgraph_figure import ORANGE, bins_icon, hnode

ROOT = Path(__file__).resolve().parents[1]


def main():
    mc = yaml.safe_load((ROOT / "configs/semkine/s37_meshgraph_s3407.yaml").read_text())["MODEL"]
    assert mc["ENCODER"] == "mesh_graph" and not mc["PREV_RENDER"]
    assert mc["PREVPOS_EMBED"] and mc["PREDICT_DELTA"] and mc["ZERO_EVENT_GATE"]
    hidden, layers = mc["ENCODER_HIDDEN"], mc["ENCODER_LAYERS"]
    obs = 8 if mc.get("MESH_GRAPH_OBS_FLOW", False) else 6

    fig, ax = plt.subplots(figsize=(22.0, 4.5))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.035, top=0.965)
    ax.set_xlim(0, 272)
    ax.set_ylim(0, 56)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 1), 270, 54,
                 boxstyle="round,pad=0.3,rounding_size=2.5",
                 facecolor=PANEL, edgecolor="#e5e7eb", lw=1.0, zorder=0))
    ax.text(4, 52.8, "S37 Mesh Graph — 上一帧 MANO 网格承载当前事件，LBS pooling 汇到 16 个关节",
            fontsize=10, color=CAP, va="top", zorder=6)
    ax.text(268, 52.8, "橙框：网格观测与 LBS 关节聚合", fontsize=9,
            color=ORANGE, ha="right", va="top", zorder=6)

    # Geometry above the main strip, matching the earlier S37 template.
    node(ax, 4, 22, 18, 10, "prev 51D\nMANO", hand_skel, TEAL, fs=9)
    node(ax, 27, 37, 24, 10, "MANO FK\n778 顶点 mesh", hand_skel, TEAL, fs=9)
    hnode(ax, 55, 37, 33, 10, "投影 + 可见性\n最近可见顶点查表", grid_icon, ORANGE, fs=9)
    hnode(ax, 95, 37, 31, 10, "MANO 图拓扑\n网格面 1-ring 边", graph_icon, ORANGE, fs=9)
    elbow(ax, [(13, 32.4), (13, 42), (26.6, 42)])
    arrow(ax, (51.4, 42), (54.6, 42))
    elbow(ax, [(77, 36.6), (77, 32.4)])
    arrow(ax, (105, 36.6), (105, 32.4))

    # Current events and the mesh feature trunk: one left-to-right strip.
    node(ax, 34, 22, 25, 10, "当前事件流\n(x, y, t, p)", events_icon, BLUE, fs=9)
    hnode(ax, 63, 22, 28, 10, f"按顶点归属 + 汇总\n节点观测 779 × {obs}", bins_icon, ORANGE, fs=8.8)
    node(ax, 95, 22, 20, 10, f"嵌入 + 图卷积\nEdgeConv × {layers}", cube_icon, TEAL, fs=8.5)
    node(ax, 119, 22, 25, 10, f"MANO 图特征\n779 × {hidden}", graph_icon, TEAL, fs=9)
    hnode(ax, 148, 22, 21, 10, "LBS pooling\n蒙皮权重聚合", funnel_icon, ORANGE, fs=8.8)
    hnode(ax, 173, 22, 22, 10, f"16 关节特征\n16 × {2 * hidden + 1}", stack_icon, ORANGE, fs=9)
    node(ax, 199, 22, 31, 10, "15 个关节头\n+ root 头", hand_skel, TEAL, fs=9)
    node(ax, 234, 22, 6, 10, "Δ", None, fs=15)
    for left, right in [(59.4, 62.6), (91.4, 94.6), (115.4, 118.6),
                        (144.4, 147.6), (169.4, 172.6), (195.4, 198.6),
                        (230.4, 233.6)]:
        arrow(ax, (left, 27), (right, 27))

    # The background bypass belongs to the root head, not to the LBS pool.
    elbow(ax, [(131.5, 32.4), (131.5, 41.5), (214.5, 41.5), (214.5, 32.4)], lw=1.2)
    ax.text(173, 43.3, "背景节点特征 → root", fontsize=8.5,
            color=TEAL, ha="center", va="bottom", zorder=6)
    ax.text(158.5, 19.1, "仅池化 778 个网格顶点", fontsize=7.8,
            color=CAP, ha="center", va="top", zorder=6)

    # State residual along the bottom; short branches feed the actual decoders.
    ax.add_patch(Circle((245, 27), 1.7, facecolor="white", edgecolor=TEAL, lw=1.5, zorder=5))
    ax.text(245, 27, "+", ha="center", va="center", fontsize=12, color=TEAL, zorder=6)
    node(ax, 250, 22, 17, 10, "当前状态\nprev + Δ", None, fs=9)
    arrow(ax, (240.4, 27), (243.2, 27))
    arrow(ax, (246.8, 27), (249.6, 27))
    elbow(ax, [(13, 21.6), (13, 4.8), (245, 4.8), (245, 25.2)])
    ax.text(99, 6.4, "prev（本步输出作为下一步 prev，再做 FK）", fontsize=8.5,
            color=TEAL, ha="center", va="bottom", zorder=6)

    node(ax, 199, 8.5, 29, 7.2, "prev_mlp", stack_icon, TEAL, fs=8.8)
    arrow(ax, (213.5, 4.8), (213.5, 8.1), lw=1.2)
    elbow(ax, [(228.4, 12.1), (237, 12.1), (237, 21.6)], lw=1.2)
    elbow(ax, [(197, 4.8), (197, 18.2), (207, 18.2), (207, 21.6)], lw=1.1)
    ax.text(204, 19, "自身 prev 角", fontsize=7.5, color=CAP,
            ha="center", va="bottom", zorder=6)
    ax.text(254, 15.7, "Δ 合并头输出与旁路\n空包置零后再加 prev", fontsize=7.8,
            color=CAP, ha="center", va="top", zorder=6)

    out = ROOT / "docs/assets/s37_meshgraph_template_20260923"
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in (".png", ".svg"):
        path = out.with_suffix(ext)
        fig.savefig(path, dpi=220, facecolor="white", bbox_inches="tight")
        print(path.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
