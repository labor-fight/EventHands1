#!/usr/bin/env python3
"""Draw the minimal XYZ proposal; no model implementation or measured results."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from draw_s37_xyz_depth_design import BLUE, EDGE, GREEN, MUTED, ORANGE, BG, box, route, text

OUT = Path(__file__).resolve().parents[1] / "docs/assets/s37_xyz_depth_minimal_20260929"


def draw():
    width, height = 2180, 870
    fig, ax = plt.subplots(figsize=(21.8, 8.7), dpi=120)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((5, 5), width - 10, height - 10,
                                boxstyle="round,pad=0,rounding_size=22", fc=BG,
                                ec="#e2e7ed", lw=1.2))
    text(ax, 35, 40, "S37–XYZ 精简建议：一个网格 GNN 主干", 22, ha="left", weight="bold")
    text(ax, 35, 78, "完整三维历史 + 轻量事件编码 → 三维关联 → 网格更新 → 姿态增量", 14,
         ha="left", color=MUTED)
    text(ax, width - 35, 41, "设计稿 · 未实现 / 未训练", 14, ha="right", color=ORANGE)

    box(ax, 35, 195, 165, 110, "prev 51D", "MANO 状态", symbol="hand")
    box(ax, 250, 195, 175, 110, "MANO FK", "相机系 · 米", symbol="hand")
    box(ax, 475, 180, 335, 140, "完整 778 顶点 XYZ", "固定 MANO faces / 1-ring\n保留前后层与三维边向量", kind="new", symbol="mesh")
    route(ax, [(200, 250), (250, 250)])
    route(ax, [(425, 250), (475, 250)])

    box(ax, 35, 440, 165, 140, "事件流", "(u, v, t, p)\n同 S37 采样", symbol="event")
    box(ax, 250, 420, 260, 180, "轻量事件处理", "token7 → 共享 MLP → 128D\n像素 + K → 三维射线\n不建事件 k-NN 图", kind="new", symbol="ray", fs=12)
    box(ax, 560, 420, 275, 180, "三维关联与聚合", "表面 / 轮廓邻域关联\nH0 : 778 × 128\n附 z_prior 与有效性", kind="new", symbol="stack", fs=12)
    box(ax, 885, 420, 260, 180, "唯一图主干", "MeshGNN × 3\n固定拓扑 + ΔXYZ 消息\nH3 : 778 × 128", kind="new", symbol="mesh")
    box(ax, 1195, 440, 235, 140, "LBS 关节聚合", "e : 16 × 257\n均值 / 最大值 / 覆盖率", kind="new", symbol="pool", fs=12)
    box(ax, 1480, 420, 285, 180, "root + 手指小头", "共享网格证据\nroot 另读未匹配 token\n输出 Δ51D；无 prev_mlp", symbol="pool", fs=12)
    box(ax, 1820, 440, 300, 140, "新状态 s_k", "s_k = prev + Δ\n下一步 FK → 完整三维网格", fs=13)
    for start, end in [(200, 250), (510, 560), (835, 885), (1145, 1195), (1430, 1480), (1765, 1820)]:
        route(ax, [(start, 510), (end, 510)], color=BLUE if start < 560 else GREEN)

    route(ax, [(645, 320), (645, 420)])
    text(ax, 682, 365, "XYZ / faces", 11, color=GREEN, ha="left")
    route(ax, [(810, 250), (1015, 250), (1015, 420)])
    text(ax, 1035, 305, "固定边\n三维边向量", 11, color=GREEN, ha="left")

    # A tiny conditional input is distinct from a state-only pose predictor.
    route(ax, [(117, 195), (117, 136), (1622, 136), (1622, 420)], color=MUTED, dashed=True)
    text(ax, 1145, 118, "每个手指头仅附自己的 prev 角（3D 条件值）", 11, color=MUTED)
    route(ax, [(200, 287), (221, 287), (221, 357), (1970, 357), (1970, 440)],
          color=MUTED, dashed=True, width=1.1)
    text(ax, 1895, 342, "残差更新 + prev", 11, color=MUTED)

    # Preserve unmatched evidence without a second global feature encoder.
    route(ax, [(697, 600), (697, 658), (1622, 658), (1622, 600)], color=BLUE)
    text(ax, 1150, 639, "未匹配 / 弱匹配 → 同一事件 MLP 的池化 token（129D，仅到 root）", 11, color=BLUE)

    route(ax, [(1970, 580), (1970, 710), (20, 710), (20, 250), (35, 250)], color=GREEN)
    text(ax, 1080, 693, "本步输出作为下一步 prev；空包 Δ = 0", 12, color=GREEN)

    for x, w, title, body in [
        (35, 670, "删去的计算", "事件 EventGNN、prev_mlp、完整 512D 全局事件读出。"),
        (730, 720, "保留的必要输入", "FK 的三维几何、事件射线、时间属性、未匹配事件。"),
        (1475, 645, "仍需训练验证", "z_prior 是历史假设；精简不代表精度必然更好。"),
    ]:
        ax.add_patch(FancyBboxPatch((x, 752), w, 91,
                                    boxstyle="round,pad=0,rounding_size=13", fc="white",
                                    ec=EDGE, lw=1.0, zorder=4))
        text(ax, x + 20, 778, title, 13, ha="left", color=ORANGE, weight="bold")
        text(ax, x + 20, 817, body, 11.5, ha="left")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "svg"]:
        path = OUT.with_suffix("." + ext)
        fig.savefig(path, dpi=120, facecolor="white")
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    draw()
