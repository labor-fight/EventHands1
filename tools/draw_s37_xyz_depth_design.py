#!/usr/bin/env python3
"""Draw the proposed, untrained S37 XYZ architecture in the repository flow style.

Outputs PNG plus SVG with editable text; reads no experiment metrics.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/s37_xyz_depth_design_20260929"

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})

BG = "#f5f7fa"
DARK = "#233348"
MUTED = "#728196"
GREEN = "#00a77b"
BLUE = "#377de0"
ORANGE = "#ec641c"
EDGE = "#d4dce6"
W, H = 2440, 1090


def text(ax, x, y, s, fs=13, color=DARK, ha="center", weight="normal", **kw):
    return ax.text(x, y, s, fontsize=fs, color=color, ha=ha, va="center",
                   weight=weight, linespacing=1.45, zorder=8, **kw)


def route(ax, points, color=GREEN, dashed=False, width=1.8):
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=width,
                ls="--" if dashed else "-", zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                mutation_scale=15, lw=width, color=color,
                                linestyle="--" if dashed else "-", zorder=2))


def icon(ax, name, x, y, color):
    if name == "hand":
        for dx, dy in [(-20, -7), (-12, -21), (0, -26), (12, -20), (22, -9)]:
            ax.plot([x, x + dx], [y + 18, y + dy], color=color, lw=1.8, zorder=6)
            for a in [0.5, 1]:
                ax.add_patch(Circle((x + a * dx, y + 18 + a * (dy - 18)),
                                    3.8, color=color, zorder=6))
        ax.add_patch(Circle((x, y + 18), 5, color=color, zorder=6))
    elif name == "event":
        for i, (dx, dy) in enumerate([(-20, -9), (-9, -21), (4, -10), (20, -18),
                                     (-20, 12), (-5, 5), (8, 20), (23, 7)]):
            ax.add_patch(Circle((x + dx, y + dy), 4,
                                color=BLUE if i % 2 else ORANGE, zorder=6))
    elif name == "ray":
        for dy in [-22, 0, 22]:
            ax.plot([x - 22, x + 24], [y, y + dy], color=color, lw=1.7, zorder=6)
        ax.add_patch(Circle((x - 22, y), 4, color=color, zorder=6))
    elif name == "mesh":
        pts = [(-23, 8), (-8, -22), (22, -9), (16, 23), (-1, 3)]
        for a, b in [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]:
            ax.plot([x + pts[a][0], x + pts[b][0]], [y + pts[a][1], y + pts[b][1]],
                    color=color, lw=1.4, zorder=6)
        for dx, dy in pts:
            ax.add_patch(Circle((x + dx, y + dy), 3.3, color=color, zorder=6))
    elif name == "stack":
        for dx, dy in [(-9, -10), (0, 0), (9, 10)]:
            ax.add_patch(plt.Rectangle((x - 21 + dx, y - 10 + dy), 42, 20,
                                       ec=color, fc="white", lw=1.4, zorder=6))
    elif name == "pool":
        ax.add_patch(Polygon([(x - 24, y - 22), (x + 24, y - 22),
                              (x + 5, y + 3), (x + 5, y + 23),
                              (x - 5, y + 23), (x - 5, y + 3)],
                             ec=color, fc="none", lw=1.7, zorder=6))


def box(ax, x, y, w, h, title, body="", kind="old", symbol=None, fs=13):
    c = ORANGE if kind == "new" else EDGE
    fill = "#fff8f2" if kind == "new" else "white"
    if kind == "prior":
        c, fill = MUTED, "#f0f3f6"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=15",
                                ec=c, fc=fill, lw=2 if kind == "new" else 1.5,
                                zorder=4))
    if symbol:
        icon(ax, symbol, x + 38, y + 43, ORANGE if kind == "new" else GREEN)
        tx, align = x + 75, "left"
    else:
        tx, align = x + w / 2, "center"
    text(ax, tx, y + 43, title, fs, ha=align, weight="bold")
    if body:
        text(ax, x + w / 2, y + (80 + h - 16) / 2, body, fs - 1, color=MUTED)


def draw():
    fig, ax = plt.subplots(figsize=(24.4, 10.9), dpi=120)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((5, 5), W - 10, H - 10,
                                boxstyle="round,pad=0,rounding_size=24",
                                fc=BG, ec="#e2e7ed", lw=1.2))
    text(ax, 35, 43, "S37–XYZ 候选网络", 23, ha="left", weight="bold")
    text(ax, 35, 82, "完整 778 顶点 XYZ  +  事件射线关联  +  固定拓扑三维消息传递", 14,
         ha="left", color=MUTED)
    text(ax, W - 36, 42, "设计稿 · 未实现模型 / 未训练", 15, ha="right", color=ORANGE)
    text(ax, W - 36, 81, "橙框：新增或改造     白框：复用 S37     灰框：保留的历史读出", 12,
         ha="right", color=MUTED)

    # History / geometry branch.
    box(ax, 35, 210, 165, 110, "prev 51D", "MANO 状态", symbol="hand")
    box(ax, 240, 210, 165, 110, "MANO FK", "相机系 · 米", symbol="hand")
    box(ax, 445, 200, 270, 130, "完整三维网格", "V_prev : 778 × 3\n保留 X、Y、Z", kind="new", symbol="mesh")
    box(ax, 780, 200, 285, 130, "固定网格拓扑", "面 1-ring；778 顶点全部保留\n边向量含 ΔX、ΔY、ΔZ", kind="new", symbol="mesh")
    box(ax, 1680, 200, 380, 130, "S37 历史读出（沿用）", "各头读取自己的 prev 角\nprev_mlp 的增量仍相加", kind="prior")
    route(ax, [(200, 265), (240, 265)])
    route(ax, [(405, 265), (445, 265)])
    route(ax, [(715, 265), (780, 265)])
    route(ax, [(117, 210), (117, 151), (1860, 151), (1860, 200)], color=MUTED, dashed=True)
    text(ax, 1280, 137, "历史参数旁路（本轮不同时删除）", 11, color=MUTED)

    # Event feature backbone, state-free.
    box(ax, 35, 455, 165, 140, "事件流", "(u, v, t, p)\n本身没有深度 z", symbol="event")
    box(ax, 240, 440, 270, 170, "S37 事件编码", "token 7D → 因果 k-NN\nEvent EdgeConv × 3\n状态无关；N ≤ 2048", symbol="mesh")
    box(ax, 550, 455, 220, 140, "事件特征", "h_i : N × 128\n全局 g_event : 512", symbol="stack")
    route(ax, [(200, 525), (240, 525)], color=BLUE)
    route(ax, [(510, 525), (550, 525)], color=BLUE)

    # New geometry-aware readout.
    box(ax, 820, 415, 255, 220, "三维射线关联", "最近表面 / 轮廓邻域\n关联权重 A；z_prior\n有效性 valid；支持 q\n无匹配 → 全局事件支路", kind="new", symbol="ray")
    box(ax, 1120, 455, 225, 140, "顶点事件特征", "H0 : 778 × 128\n按 A 聚合事件与三维关系", kind="new", symbol="stack", fs=12)
    box(ax, 1390, 445, 240, 160, "三维网格 GNN", "固定 1-ring × 3 层\nXYZ 几何调制事件消息\nH3 : 778 × 128", kind="new", symbol="mesh", fs=12)
    box(ax, 1675, 455, 210, 140, "LBS pooling", "e : 16 × 257\nmean / max / 直接覆盖率", kind="new", symbol="pool", fs=12)
    box(ax, 1930, 435, 245, 180, "S37 增量头", "15 指头：各读自己的 e\nroot：16 个 e + g_event\n加上历史旁路 → Δ51D", symbol="pool", fs=12)
    box(ax, 2220, 455, 180, 140, "新状态 s_k", "s_prev + Δ\nFK → 新的 778 顶点", fs=12)
    route(ax, [(770, 525), (820, 525)], color=BLUE)
    for start, end in [(1075, 1120), (1345, 1390), (1630, 1675), (1885, 1930), (2175, 2220)]:
        route(ax, [(start, 525), (end, 525)])

    route(ax, [(580, 330), (580, 375), (945, 375), (945, 415)])
    text(ax, 765, 362, "完整 XYZ + faces", 11, color=GREEN)
    route(ax, [(1065, 265), (1510, 265), (1510, 445)])
    text(ax, 1290, 247, "固定边 + 随姿态变化的三维边向量", 11, color=GREEN)
    route(ax, [(2000, 330), (2000, 435)], color=MUTED, dashed=True)
    route(ax, [(117, 320), (117, 345), (2195, 345), (2195, 425), (2310, 425), (2310, 455)],
          color=MUTED, dashed=True, width=1.1)
    text(ax, 2295, 407, "+ s_prev", 10, color=MUTED)

    # Inverse camera mapping: no projection of the vertices.
    box(ax, 485, 695, 285, 125, "事件 → 相机射线", "d_i = inv(K_event) [u_i, v_i, 1]^T\n保留时间 t 与极性 p", kind="new", symbol="ray", fs=12)
    route(ax, [(117, 595), (117, 757), (485, 757)], color=BLUE)
    text(ax, 308, 737, "事件像素 + 匹配内参", 11, color=BLUE)
    route(ax, [(770, 757), (945, 757), (945, 635)], color=BLUE)
    text(ax, 881, 736, "射线", 11, color=BLUE)

    # Preserve event fallback to root; it includes unmatched sampled events.
    route(ax, [(660, 595), (660, 657), (2050, 657), (2050, 615)], color=BLUE)
    text(ax, 1510, 641, "全局事件 g_event 直接到 root（包含未匹配事件）", 11, color=BLUE)
    text(ax, 1510, 748, "新增几何支路：几何 × 事件特征\n没有直接观测的顶点保留；可接收邻居推断特征", 13, color=MUTED)
    text(ax, 2135, 743, "空事件包：Δ = 0\n整条更新保持 S37 合同", 12, color=MUTED)

    # Close the state loop explicitly.
    route(ax, [(2310, 595), (2310, 875), (20, 875), (20, 265), (35, 265)],
          color=GREEN, width=1.8)
    text(ax, 1240, 858, "本步新状态成为下一步 prev；每包重新生成完整三维网格", 12, color=GREEN)

    # Three concise notes to keep physical meaning visible in the diagram.
    for x, width, title, body in [
        (35, 750, "01  深度的含义", "X_prior = z_prior · d_i；z_prior 来自历史网格，\n是带有效性标记的假设，不是事件传感器测距。"),
        (810, 775, "02  三维关系真正进入计算", "事件—顶点关系 ΔXYZ 调制聚合；网格边 ΔXYZ 调制消息。\n顶点不压成二维图，不随当前姿态重建 k-NN。"),
        (1610, 790, "03  这是待训练的新结构", "新增模块与事件编码联合训练；暂不增加独立深度头或 GRU。\n首版沿用 S37 包式协议，不宣称逐事件异步执行。"),
    ]:
        ax.add_patch(FancyBboxPatch((x, 925), width, 130,
                                    boxstyle="round,pad=0,rounding_size=14",
                                    fc="white", ec=EDGE, lw=1.0, zorder=4))
        text(ax, x + 22, 954, title, 13, ha="left", weight="bold", color=ORANGE)
        text(ax, x + 22, 1009, body, 12, ha="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        path = OUT.with_suffix("." + ext)
        fig.savefig(path, dpi=120, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    draw()
