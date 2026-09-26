#!/usr/bin/env python3
"""Draw the trained mesh-memory architecture using the S36 simple template.

Keep the original prev/FK branch, event input, horizontal strip, icons, colours,
and lower pose residual. Expand only the mesh trunk and add the memory loop.
This is an architecture figure, not an experiment result.

    python tools/draw_mesh778_s36_template.py
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
    config = yaml.safe_load(
        (REPO / "configs/semkine/event_guided_mesh_memory_s3407.yaml").read_text()
    )
    model = config["MODEL"]
    assert model["ENCODER"] == "event_guided_mesh" and model["EGM_MEMORY"]
    with np.load(REPO / config["MANO"]["NPZ"]) as mano:
        vertices, joints = mano["weights"].shape
    hidden = model["ENCODER_HIDDEN"]
    layers = model["ENCODER_LAYERS"]
    neighbours = model["ENCODER_K"]
    candidates = model["EGM_CANDIDATES"]

    # Same height, node positions on the left, and units-per-inch as S36.
    # Only the strip grows to make the memory and joint features explicit.
    fig, ax = plt.subplots(figsize=(22.5, 3.6))
    ax.set_xlim(0, 282)
    ax.set_ylim(0, 45)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (1, 1), 280, 43, boxstyle="round,pad=0.3,rounding_size=2.5",
        facecolor=PANEL, edgecolor="#e5e7eb", lw=1.0, zorder=0,
    ))
    ax.text(
        4, 42.3,
        f"Event-guided Mesh — {vertices} 顶点 → 图卷积 → LBS → {joints} 关节特征 → Δ",
        fontsize=8.2, color=CAP, va="top", zorder=6,
    )

    # Preserve the S36 upper geometry branch and lower event branch.
    node(ax, 4, 17, 17, 10, "prev 51D\nMANO", hand_skel, TEAL, fs=8.2)
    node(ax, 25, 30, 16, 10, "MANO FK", hand_skel, TEAL)
    node(ax, 44, 30, 18, 10, f"{vertices} 顶点\n3D mesh", hand_skel, TEAL, fs=8.2)
    node(ax, 65, 30, 21, 10, "投影 + 可见性\n事件归属", grid_icon, TEAL, fs=8.2)
    elbow(ax, [(12.5, 27.4), (12.5, 35), (24.6, 35)])
    arrow(ax, (41.4, 35), (43.6, 35))
    arrow(ax, (62.4, 35), (64.6, 35))
    node(ax, 58, 6, 23, 10, "当前事件包 E(k)\n(x, y, t, p)", events_icon, BLUE, fs=8.2)

    # Full vertex identities persist through graph convolution and memory.
    node(ax, 91, 17, 25, 10, f"{vertices} 顶点图\nEvent2D 引导", graph_icon, BLUE, fs=8.2,
         caption=f"全顶点保留 · {candidates} 候选取 k={neighbours}")
    node(ax, 120, 17, 20, 10, f"图卷积\nEdgeConv × {layers}", cube_icon, BLUE, fs=8.2,
         caption=f"顶点特征 {vertices} × {hidden}")
    node(ax, 144, 17, 23, 10, "逐顶点 GRU\n+ 当前观测门", stack_icon, TEAL, fs=8.0,
         caption=f"门控记忆特征 {vertices} × {hidden}")
    node(ax, 171, 17, 21, 10, f"LBS pooling\n{vertices} → {joints}", funnel_icon, TEAL, fs=8.2,
         caption="固定蒙皮权重，按关节归一化")
    node(ax, 196, 17, 24, 10, f"{joints} 关节特征\n{joints} × {hidden}", hand_skel, TEAL, fs=8.2,
         caption="关节特征向量")
    node(ax, 224, 17, 23, 10, f"MLP → Δ\n{model['OUTPUT_DIM']}D", None, fs=9.2,
         caption=f"root 6D + {joints - 1} × 3D")
    for left, right in ((116.4, 119.6), (140.4, 143.6), (167.4, 170.6),
                        (192.4, 195.6), (220.4, 223.6)):
        arrow(ax, (left, 22), (right, 22))
    elbow(ax, [(86.4, 35), (98, 35), (98, 27.4)])
    elbow(ax, [(81.4, 11), (94, 11), (94, 16.6)])

    # A small upper state box is the only new branch relative to S36.
    node(ax, 144, 31, 23, 9, "上一顶点记忆\nH(k−1)", stack_icon, TEAL, fs=8.0)
    elbow(ax, [(143.6, 35.5), (130, 35.5), (130, 27.4)])
    arrow(ax, (154, 30.6), (154, 27.4))
    elbow(ax, [(162, 27.4), (162, 29), (169, 29),
               (169, 41.5), (155.5, 41.5), (155.5, 40.4)], lw=1.1)
    ax.text(172, 38.7, "无观测顶点：保留局部 H，允许随 MANO 运动",
            fontsize=7.0, color=TEAL, va="center", zorder=6)
    ax.text(172, 34.3, "LBS 输入：where(m(k), H(k), 0)；m(k) 为当前观测",
            fontsize=6.8, color=CAP, va="center", zorder=6)

    # Preserve the exact S36 prev + delta residual convention.
    ax.add_patch(Circle((253, 22), 1.7, facecolor="#ffffff", edgecolor=TEAL,
                        lw=1.4, zorder=5))
    ax.text(253, 22, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    node(ax, 258, 17, 21, 10, "s(k) = prev + Δ", None, fs=8.6)
    arrow(ax, (247.4, 22), (251.2, 22))
    arrow(ax, (254.8, 22), (257.6, 22))
    elbow(ax, [(12.5, 16.6), (12.5, 3.6), (253, 3.6), (253, 20.2)])
    ax.text(166, 4.6, "prev（本步输出作为下一包 prev，再做 FK；整包无前景观测时 Δ = 0）",
            ha="center", va="bottom", fontsize=7.0, color=TEAL, zorder=6)

    output = REPO / "docs/assets/mesh778_s36_template"
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print(path.relative_to(REPO))
    plt.close(fig)


if __name__ == "__main__":
    main()
