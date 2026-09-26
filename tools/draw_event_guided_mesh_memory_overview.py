#!/usr/bin/env python3
"""Readable overview of the configured full-MANO vertex-memory tracker.

Creates docs/assets/event_guided_mesh_memory_overview.{svg,png}; no model,
training state, or evaluation artifacts are changed.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

from make_s36_figure_simple import (
    BLUE, CAP, DARK, EDGE, TEAL, cube_icon, events_icon, funnel_icon,
    graph_icon, hand_skel, stack_icon,
)
from make_s37_fkgraph_figure import ORANGE


REPO = Path(__file__).resolve().parents[1]


def arrow(ax, points, *, color=TEAL, dashed=False, lw=2.0):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw,
                linestyle="--" if dashed else "-", solid_capstyle="round", zorder=2)
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=15,
        color=color, lw=lw, shrinkA=0, shrinkB=0, zorder=2,
        linestyle="--" if dashed else "-",
    ))


def box(ax, x, y, w, h, title, subtitle="", *, icon=None, accent=False,
        title_size=16, subtitle_size=13):
    color = ORANGE if accent else TEAL
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.7",
        facecolor="#fff7f1" if accent else "white",
        edgecolor=ORANGE if accent else EDGE, lw=1.8 if accent else 1.4, zorder=3,
    ))
    cx = x + w / 2
    if icon is not None:
        if icon is events_icon:
            icon(ax, cx, y + h - 4.1, 2.0)
        else:
            icon(ax, cx, y + h - 4.1, 2.0, color)
        title_y = y + h - 9.1
        sub_y = y + 3.1
    else:
        title_y = y + h / 2 + (2.2 if subtitle else 0)
        sub_y = y + h / 2 - 3.0
    ax.text(cx, title_y, title, ha="center", va="center", fontsize=title_size,
            color=DARK, weight="medium", zorder=5)
    if subtitle:
        ax.text(cx, sub_y, subtitle, ha="center", va="center",
                fontsize=subtitle_size, color=color if accent else CAP, zorder=5)


def main():
    cfg = yaml.safe_load((REPO / "configs/semkine/event_guided_mesh_memory_s3407.yaml").read_text())
    mc = cfg["MODEL"]
    assert mc["ENCODER"] == "event_guided_mesh" and mc["EGM_MEMORY"]
    with np.load(REPO / cfg["MANO"]["NPZ"]) as mano:
        vertices, joints = mano["weights"].shape
    hidden, layers = mc["ENCODER_HIDDEN"], mc["ENCODER_LAYERS"]
    candidates, neighbours = mc["EGM_CANDIDATES"], mc["ENCODER_K"]
    geometric, delta_dim = mc["EGM_GEOMETRY_NEIGHBORS"], mc["OUTPUT_DIM"]
    assert delta_dim == 6 + (joints - 1) * 3

    fig, ax = plt.subplots(figsize=(23, 9.2))
    ax.set_xlim(-2, 265)
    ax.set_ylim(-1, 108)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(7, 104, "事件引导的完整网格追踪网络", fontsize=23, color=DARK,
            va="center", weight="bold")
    ax.text(258, 104, f"{vertices} 顶点 → 图卷积 → LBS → {joints} 关节特征 → Δ",
            fontsize=16, color=ORANGE, ha="right", va="center")

    # Previous pose gives geometry; the incoming event packet guides association and edges.
    box(ax, 7, 80, 28, 13, "上一姿态", f"s(k−1) · {delta_dim} 维", title_size=16)
    box(ax, 43, 80, 28, 13, "MANO FK", "生成当前几何", title_size=16)
    box(ax, 85, 80, 34, 13, "当前事件包 E(k)", "(x, y, t, p)", title_size=15)
    arrow(ax, [(35.5, 86.5), (42.5, 86.5)])
    arrow(ax, [(57, 79.5), (57, 72), (20, 72), (20, 61.5)])
    arrow(ax, [(102, 79.5), (102, 69), (57, 69), (57, 61.5)])
    ax.text(81, 73, f"投影关联 · {vertices} × 6 观测", fontsize=12.5, color=BLUE,
            ha="center", va="center")

    # The full vertex set survives every graph and memory operation.
    y, h = 42, 19
    specs = [
        (7, 26, f"{vertices} 顶点", f"{vertices} × 3", hand_skel, False),
        (40, 34, "事件引导构图", f"{vertices} 节点 · {neighbours} 邻居", graph_icon, True),
        (81, 27, "图卷积", f"EdgeConv × {layers}", cube_icon, False),
        (115, 38, "局部记忆更新", "GRU + 当前观测门", stack_icon, True),
        (160, 26, "LBS 池化", f"{vertices} → {joints}", funnel_icon, True),
        (193, 30, f"{joints} 关节特征", f"{joints} × {hidden}", hand_skel, False),
        (230, 26, f"Δ · {delta_dim} 维", f"6 + {joints - 1} × 3", None, True),
    ]
    for x, w, title, subtitle, icon, accent in specs:
        box(ax, x, y, w, h, title, subtitle, icon=icon, accent=accent,
            title_size=16, subtitle_size=13)
    for (x, w, *_), (next_x, *_) in zip(specs, specs[1:]):
        arrow(ax, [(x + w + 0.6, 51.5), (next_x - 0.6, 51.5)])

    ax.text(20, 37.5, "保留全部顶点", fontsize=12.5, color=CAP, ha="center")
    ax.text(57, 37.5, f"{candidates} 候选：{geometric} 几何 + {neighbours - geometric} 事件",
            fontsize=11.5, color=CAP, ha="center")
    ax.text(57, 33.5, f"观测嵌入：6 → {hidden}", fontsize=11.5, color=CAP, ha="center")
    ax.text(94.5, 37.5, f"{vertices} × {hidden}", fontsize=12.5, color=CAP, ha="center")
    ax.text(134, 37.5, "输出 m(k) ⊙ H(k)", fontsize=12.5, color=ORANGE, ha="center")
    ax.text(173, 37.5, "固定权重加权均值", fontsize=11.5, color=CAP, ha="center")
    ax.text(208, 37.5, "送入 MLP 解码头", fontsize=11.5, color=CAP, ha="center")

    # Local memory is an explicit second state, separate from MANO pose.
    box(ax, 131, 80, 36, 13, "上一顶点记忆", f"H(k−1) · {vertices} × {hidden}", title_size=16,
        subtitle_size=12.5)
    arrow(ax, [(130.4, 86.5), (124, 86.5), (124, 65), (94.5, 65), (94.5, 61.5)])
    ax.text(111, 68, "上下文", fontsize=11.5, color=TEAL, ha="center", va="center")
    arrow(ax, [(150, 79.5), (150, 61.5)])
    arrow(ax, [(153.6, 56.5), (157, 56.5), (157, 75), (182, 75),
               (182, 96), (149, 96), (149, 93.5)], dashed=True, lw=1.7)
    ax.text(185, 96, "H(k) 传给下一包", fontsize=13, color=TEAL, va="center")
    ax.text(190, 85.5, "有观测 → 更新局部记忆", fontsize=15, color=DARK, va="center")
    ax.text(190, 78.5, "无观测 → H(k) = H(k−1)", fontsize=15, color=ORANGE, va="center")
    ax.text(190, 70.5, "顶点仍随整只手与关节运动", fontsize=13.5, color=CAP, va="center")

    # The decoder predicts MANO parameter increments, followed by an explicit pose add.
    ax.add_patch(Circle((182, 24.5), 2.5, facecolor="white", edgecolor=TEAL, lw=2, zorder=4))
    ax.text(182, 24.5, "+", fontsize=22, color=TEAL, ha="center", va="center", zorder=5)
    arrow(ax, [(243, 41.5), (243, 32.5), (182, 32.5), (182, 27.3)])
    arrow(ax, [(6.5, 86.5), (3.5, 86.5), (3.5, 24.5), (179.2, 24.5)])
    ax.text(88, 27.5, "上一姿态 s(k−1)", color=TEAL, fontsize=14, ha="center")
    box(ax, 199, 18, 57, 13, "当前姿态 s(k)", "s(k) = s(k−1) + Δ(k)", title_size=17,
        subtitle_size=14)
    arrow(ax, [(184.8, 24.5), (198.5, 24.5)])
    arrow(ax, [(227.5, 17.5), (227.5, 10), (0, 10), (0, 96), (21, 96), (21, 93.5)],
          dashed=True, lw=1.7)
    ax.text(101, 12.8, "下一事件包：prev ← s(k) → MANO FK → 更新 778 顶点几何",
            fontsize=13, color=TEAL, ha="center")
    ax.text(7, 2.1, "Δ：全局平移与旋转 6 维 + 15 个关节旋转 × 3 维；空包 Δ = 0。",
            fontsize=13, color=DARK, va="center")
    ax.text(258, 2.1, "m(k)：该顶点在当前包是否分配到事件", fontsize=12.5,
            color=CAP, va="center", ha="right")

    output = REPO / "docs/assets/event_guided_mesh_memory_overview"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.015, right=0.995, bottom=0.015, top=0.99)
    for suffix in (".svg", ".png"):
        path = output.with_suffix(suffix)
        fig.savefig(path, dpi=170, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
