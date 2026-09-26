#!/usr/bin/env python3
"""Draw the current full-778 memory model as a readable S37-style flow strip.

Architecture only; configuration and MANO weights supply all tensor dimensions.
Run from any directory: python tools/draw_mesh_memory_network.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, FancyBboxPatch

from make_s36_figure_simple import (
    BLUE, DARK, TEAL, arrow, cube_icon, elbow, events_icon,
    funnel_icon, graph_icon, stack_icon,
)

ROOT = Path(__file__).resolve().parents[1]
ORANGE = "#ed7311"
MUTED = "#617183"


def box(ax, x, y, w, h, edge="#d5dbe1", fill="white", zorder=3):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.5",
        facecolor=fill, edgecolor=edge, lw=1.4, zorder=zorder,
    ))


def label(ax, x, y, text, size=12, color=DARK, weight="normal", ha="center"):
    ax.text(x, y, text, fontsize=size, color=color, fontweight=weight,
            ha=ha, va="center", zorder=6, linespacing=1.5)


def main():
    cfg = yaml.safe_load((ROOT / "configs/semkine/event_guided_mesh_memory_s3407.yaml").read_text())
    model = cfg["MODEL"]
    assert model["ENCODER"] == "event_guided_mesh" and model["EGM_MEMORY"]
    with np.load(ROOT / cfg["MANO"]["NPZ"]) as data:
        vertices = data["v_template"].copy()
        faces = data["f"].copy()
        weights = data["weights"].copy()
    nv, nj = weights.shape
    hidden = model["ENCODER_HIDDEN"]
    layers = model["ENCODER_LAYERS"]
    candidates = model["EGM_CANDIDATES"]
    neighbors = model["ENCODER_K"]
    geometric = model["EGM_GEOMETRY_NEIGHBORS"]
    delta = model["OUTPUT_DIM"]

    fig, ax = plt.subplots(figsize=(26.8, 9.3))
    fig.subplots_adjust(left=0.012, right=0.988, bottom=0.025, top=0.98)
    ax.set(xlim=(-1, 269), ylim=(0, 94), aspect="equal")
    ax.axis("off")
    box(ax, 0, 1, 267, 92, fill="#f6f8fa", zorder=0)
    label(ax, 8, 88.5, f"{nv} 顶点 → 图卷积 → LBS 池化 → {nj} 关节特征 → Δ",
          size=23, weight="bold", ha="left")
    label(ax, 8, 83.4, "当前实现 · Event2D 引导完整网格构图，局部记忆跨事件包递推",
          size=12, color=MUTED, ha="left")

    # Previous pose provides geometry; the following event packet drives updates.
    for x, w, text, color in (
        (8, 25, "上一预测状态\ns(k−1) · 51D", TEAL),
        (39, 30, "MANO / FK\n生成上一状态网格", TEAL),
        (75, 25, "当前事件包 E(k)\nEvent2D: x, y, t, p", BLUE),
        (106, 35, f"上一顶点记忆 H(k−1)\n{nv} × {hidden}", TEAL),
    ):
        box(ax, x, 69, w, 10, edge=color)
        label(ax, x + w / 2, 74, text, size=11)
    arrow(ax, (33.4, 74), (38.6, 74))
    elbow(ax, [(48, 68.6), (48, 64), (20.5, 64), (20.5, 59.4)])
    elbow(ax, [(83, 68.6), (83, 64), (59, 64), (59, 59.4)])
    elbow(ax, [(110, 68.6), (110, 63), (95, 63), (95, 59.4)])
    arrow(ax, (131, 68.6), (131, 59.4))
    label(ax, 101, 61, "历史上下文", size=9, color=TEAL)
    elbow(ax, [(141.4, 48), (144, 48), (144, 81), (123.5, 81), (123.5, 79.4)])
    label(ax, 158, 75, "记忆递推：H(k) 传给下一包", size=13, color=TEAL, ha="left")
    label(ax, 158, 69.5, "每步使用上一预测的 3D 网格，关联新到达的 2D 事件", size=11, color=MUTED, ha="left")

    # Seven main stages, preserving the full vertex dimension until LBS pooling.
    stages = [(8, 25), (39, 30), (75, 25), (106, 35), (147, 27), (180, 28), (214, 40)]
    for index, (x, w) in enumerate(stages):
        box(ax, x, 37, w, 22, edge=ORANGE if index in (1, 3, 4, 6) else "#cbd5df",
            fill="#fffaf5" if index in (1, 3, 4, 6) else "white")
    for (x, w), (nx, _) in zip(stages, stages[1:]):
        arrow(ax, (x + w + 0.4, 48), (nx - 0.4, 48), lw=1.9)

    # Real MANO template vertices rendered as a tiny mesh, using a PCA projection.
    centered = vertices - vertices.mean(0)
    _, _, axes = np.linalg.svd(centered, full_matrices=False)
    points = centered @ axes[:2].T
    points = points[:, [1, 0]]
    points -= (points.max(0) + points.min(0)) / 2
    points *= 8.5 / np.ptp(points, axis=0).max()
    points += np.array([20.5, 48.1])
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    edges = np.unique(edges, axis=0)
    ax.add_collection(LineCollection(points[edges], colors=TEAL, linewidths=0.23, alpha=0.72, zorder=4))
    ax.scatter(points[:, 0], points[:, 1], s=0.6, c=TEAL, zorder=5)
    label(ax, 20.5, 55.3, f"完整 {nv} 顶点", size=14, weight="bold")
    label(ax, 20.5, 40.9, f"V(k−1): {nv} × 3", size=11)

    label(ax, 54, 55.3, "事件引导构图", size=14, weight="bold")
    graph_icon(ax, 54, 48.5, 3, ORANGE)
    label(ax, 54, 41, f"{nv} 节点 · {neighbors} 邻居 / 点", size=11)
    label(ax, 54, 33, f"投影归属 → 顶点观测 {nv} × 6", size=10, color=MUTED)
    label(ax, 54, 29, f"{candidates} 个 3D 候选 → {geometric} 几何 + {neighbors-geometric} 事件优先", size=9, color=MUTED)

    label(ax, 87.5, 55.3, "图卷积", size=14, weight="bold")
    cube_icon(ax, 87.5, 48.5, 3, BLUE)
    label(ax, 87.5, 41, f"EdgeConv × {layers}", size=12)
    label(ax, 87.5, 33, f"顶点特征 F(k): {nv} × {hidden}", size=10, color=MUTED)

    label(ax, 123.5, 55.3, "顶点记忆更新", size=14, weight="bold")
    label(ax, 123.5, 48.4, "逐顶点 GRU + 观测门", size=12, color=ORANGE)
    label(ax, 123.5, 41.1, f"H(k): {nv} × {hidden}", size=12)
    label(ax, 123.5, 33, "读出：m(k) ⊙ H(k)", size=11, color=ORANGE)
    label(ax, 123.5, 29, "m(k)：本包确实分配到事件", size=9, color=MUTED)

    label(ax, 160.5, 55.3, "LBS 池化", size=14, weight="bold")
    funnel_icon(ax, 160.5, 48.5, 3, ORANGE)
    label(ax, 160.5, 41, "固定蒙皮权重聚合", size=11)
    label(ax, 160.5, 33, f"权重矩阵 {nj} × {nv}", size=10, color=MUTED)

    label(ax, 194, 55.3, f"{nj} 关节特征", size=14, weight="bold")
    stack_icon(ax, 194, 48.5, 3, TEAL)
    label(ax, 194, 41, f"{nj} × {hidden}", size=15, weight="bold", color=TEAL)
    label(ax, 194, 33, "每个关节一份局部特征", size=10, color=MUTED)

    label(ax, 234, 55.3, "MLP 解码 → Δ", size=14, weight="bold")
    label(ax, 234, 48.5, f"{delta}D MANO 参数增量", size=13, color=ORANGE)
    label(ax, 234, 41, f"root 6D + {nj-1} × 3D 关节", size=11)
    label(ax, 234, 33, "输出由当前观测支持门控", size=10, color=MUTED)

    # Residual update and the explicit pose feedback loop.
    ax.add_patch(Circle((198, 16), 1.9, facecolor="white", edgecolor=TEAL, lw=1.7, zorder=5))
    label(ax, 198, 16, "+", size=18, color=TEAL)
    elbow(ax, [(234, 36.6), (234, 24), (198, 24), (198, 18)])
    elbow(ax, [(15, 68.6), (5, 68.6), (5, 25), (189, 25), (189, 16), (196, 16)])
    label(ax, 104, 23.1, "上一姿态 s(k−1)", size=10, color=TEAL)
    box(ax, 211, 11, 43, 10, edge=TEAL)
    label(ax, 232.5, 16, "更新状态 s(k) = s(k−1) + Δ", size=12)
    arrow(ax, (200, 16), (210.6, 16))
    elbow(ax, [(232.5, 10.6), (232.5, 4.3), (2, 4.3), (2, 74), (7.6, 74)])
    label(ax, 120, 6.4, "姿态递推：本步 s(k) 成为下一包的 prev，再经 MANO / FK 重建完整网格", size=10, color=TEAL)

    box(ax, 10, 10, 166, 10, fill="#eaf6f1", edge="#c8e6db")
    label(ax, 15, 16.8, "缺观测顶点：H(k) = H(k−1)；空间位置仍随整只手与关节运动。", size=12, ha="left")
    label(ax, 15, 12.4, "整包无前景观测：Δ = 0，姿态与记忆保持。LBS 池化的是特征，输出 Δ 更新 MANO 参数。", size=10, ha="left", color=MUTED)

    output = ROOT / "docs/assets/mesh_memory_network_778_16_20260923"
    for ext in (".svg", ".png"):
        path = output.with_suffix(ext)
        fig.savefig(path, dpi=160, facecolor="white", bbox_inches="tight")
        print(path.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
