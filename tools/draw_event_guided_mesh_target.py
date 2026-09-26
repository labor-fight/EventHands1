#!/usr/bin/env python3
"""Draw full-vertex memory tracking in the S36/S37 horizontal strip style.

The caller carries both MANO pose and local vertex memory across packets. Only
observed vertices write memory; fixed LBS pooling reads memory features gated by
current observations. Memory differences are retained only for diagnostics.
This figure describes implementation, not measured training performance.

    python tools/draw_event_guided_mesh_target.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, FancyBboxPatch

from make_s36_figure_simple import (
    BLUE, CAP, DARK, PANEL, TEAL, arrow, cube_icon, elbow, events_icon,
    funnel_icon, graph_icon, hand_skel, node, stack_icon,
)
from make_s37_fkgraph_figure import ORANGE, hnode


REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs/semkine/event_guided_mesh_memory_s3407.yaml").read_text())
    mc = cfg["MODEL"]
    assert mc["ENCODER"] == "event_guided_mesh" and mc["EGM_MEMORY"]
    with np.load(REPO / cfg["MANO"]["NPZ"]) as mano:
        vertices, joints = mano["weights"].shape
    hidden, layers = mc["ENCODER_HIDDEN"], mc["ENCODER_LAYERS"]
    candidates, neighbours = mc["EGM_CANDIDATES"], mc["ENCODER_K"]
    geometric = mc["EGM_GEOMETRY_NEIGHBORS"]
    delta_dim = mc["OUTPUT_DIM"]

    fig, ax = plt.subplots(figsize=(25.0, 5.0))
    ax.set_xlim(0, 338)
    ax.set_ylim(0, 68)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch(
        (1, 1), 336, 66, boxstyle="round,pad=0.3,rounding_size=2.5",
        facecolor=PANEL, edgecolor="#e5e7eb", lw=1.0, zorder=0,
    ))
    ax.text(4, 65, f"Event2D 引导完整 {vertices} 顶点 · 局部记忆 tracking 已集成",
            fontsize=10.0, color=DARK, va="top", zorder=6)
    ax.text(334, 65, "保留局部 H，允许 MANO 联动 · 精度与训练状态另见运行记录",
            fontsize=8.0, color=ORANGE, ha="right", va="top", zorder=6)

    # Previous pose supplies the 3D geometry used to associate the new packet.
    node(ax, 5, 27, 18, 10, "上一预测\ns(k−1)", hand_skel, TEAL, fs=8.0)
    node(ax, 27, 47, 20, 9, "MANO FK", hand_skel, TEAL, fs=8.0)
    node(ax, 51, 47, 24, 9, f"{vertices} 顶点\n3D mesh", hand_skel, TEAL, fs=8.0)
    elbow(ax, [(14, 37.4), (14, 51.5), (26.6, 51.5)])
    arrow(ax, (47.4, 51.5), (50.6, 51.5))
    node(ax, 46, 16, 29, 10, "随后事件包\nE_k", events_icon, BLUE, fs=8.2)
    hnode(ax, 80, 27, 28, 10, f"{vertices} 节点\n事件引导构图", graph_icon, ORANGE, fs=8.0)
    elbow(ax, [(75.4, 51.5), (94, 51.5), (94, 37.4)])
    elbow(ax, [(75.4, 21), (85, 21), (85, 26.6)])
    ax.text(60, 40, f"{candidates} 个 3D 空间候选", ha="center", va="center",
            fontsize=6.7, color=CAP, zorder=6)
    ax.text(60, 36.5, f"{geometric} 几何 + {neighbours - geometric} 当前观测来源", ha="center", va="center",
            fontsize=6.7, color=CAP, zorder=6)
    ax.text(60, 33, "缺观测 query 仅几何比较；不足补齐", ha="center", va="center",
            fontsize=6.5, color=CAP, zorder=6)

    # Temporary graph context may propagate, but only observations write memory.
    node(ax, 113, 27, 21, 10, f"GraphConv\n× {layers}", cube_icon, BLUE, fs=7.5)
    node(ax, 139, 27, 23, 10, f"本包特征 F\n{vertices} × {hidden}", stack_icon, TEAL, fs=8.0)
    hnode(ax, 167, 27, 28, 10, "逐顶点 GRU\n观测门 → H(k)", stack_icon, ORANGE, fs=8.0)
    node(ax, 200, 27, 26, 10, "本包观测门控\nwhere(m, H(k), 0)", None, fs=7.8)
    hnode(ax, 231, 27, 20, 10, "LBS\npooling", funnel_icon, ORANGE, fs=8.0)
    node(ax, 256, 27, 23, 10, f"关节特征\n{joints} × {hidden}", hand_skel, TEAL, fs=8.0)
    hnode(ax, 284, 27, 18, 10, f"MLP\nΔ {delta_dim}D", None, fs=8.5)
    for start, end in ((108.4, 112.6), (134.4, 138.6), (162.4, 166.6),
                       (195.4, 199.6), (226.4, 230.6), (251.4, 255.6), (279.4, 283.6)):
        arrow(ax, (start, 32), (end, 32))
    ax.text(181, 23.8, "无观测：H(k) = H(k−1)", ha="center", va="center",
            fontsize=7.0, color=ORANGE, zorder=6)
    ax.text(241, 23.8, "固定 LBS 均值，池化门控记忆特征", ha="center", va="center",
            fontsize=6.7, color=CAP, zorder=6)

    # A separate memory loop from the pose/FK loop, with caller-owned identity.
    node(ax, 133, 47, 32, 9, "上一顶点记忆\nH(k−1) / seen", stack_icon, TEAL, fs=8.0)
    elbow(ax, [(143, 46.6), (143, 42), (123.5, 42), (123.5, 37.4)])
    ax.text(115.5, 44.7, "历史作上下文", ha="center", va="center",
            fontsize=6.8, color=TEAL, zorder=6)
    elbow(ax, [(157, 46.6), (157, 42), (175, 42), (175, 37.4)])
    elbow(ax, [(187, 37.4), (187, 59), (149, 59), (149, 56.4)])
    ax.text(190, 56, "H(k) 显式传给下一包", ha="left", va="center",
            fontsize=7.0, color=TEAL, zorder=6)
    ax.text(190, 52, "换序列 / 有效片段：记忆清空", ha="left", va="center",
            fontsize=6.7, color=CAP, zorder=6)
    ax.text(190, 48, "H(k) − H(k−1) 仅诊断，不用于 LBS", ha="left", va="center",
            fontsize=6.7, color=CAP, zorder=6)

    # Heads are gated only by current actual observations, never by retained history.
    elbow(ax, [(101, 26.6), (101, 16), (293, 16), (293, 26.6)])
    ax.text(206, 17.2, "当前观测 m 控制 GRU 写入与 Δ 头；历史 / 图扩张不打开输出门",
            ha="center", va="bottom", fontsize=7.0, color=TEAL, zorder=6)
    elbow(ax, [(181, 16), (181, 26.6)])

    # Pose is reconstructed independently; unobserved vertices follow MANO motion.
    ax.add_patch(Circle((309, 32), 1.7, facecolor="#ffffff", edgecolor=TEAL, lw=1.4, zorder=5))
    ax.text(309, 32, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    node(ax, 317, 27, 17, 10, "更新姿态\ns(k)", None, fs=8.2)
    arrow(ax, (302.4, 32), (307.2, 32))
    arrow(ax, (310.8, 32), (316.6, 32))
    node(ax, 309, 47, 25, 9, f"MANO 重建\n{vertices}V / 21J", hand_skel, TEAL, fs=8.0)
    arrow(ax, (325.5, 37.4), (325.5, 46.6))
    elbow(ax, [(14, 26.6), (14, 10), (309, 10), (309, 30.2)])
    ax.text(198, 11.2, "s(k) = s(k−1) + Δ · 整包无前景观测：姿态与记忆都保持",
            ha="center", va="bottom", fontsize=7.2, color=TEAL, zorder=6)
    elbow(ax, [(325.5, 26.6), (325.5, 3.6), (7, 3.6), (7, 26.6)])
    ax.text(174, 4.8, "姿态 FK 闭环：下一包 prev = s(k) · 无观测顶点可随 root / 关节运动，局部 H 保留",
            ha="center", va="bottom", fontsize=7.2, color=TEAL, zorder=6)

    output = REPO / "docs/assets/event_guided_mesh_target"
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        print("wrote", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
