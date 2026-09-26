#!/usr/bin/env python3
"""Draw the implemented S37 Mesh Graph, including its state and background bypasses.

Sources: configs/semkine/s37_meshgraph_s3407.yaml; model/model.py
(_mesh_graph_forward, _decode_active, forward_packet); semkine/fk_graph.py
(assign_and_observe, FKGraphEncoder); semkine/mesh_graph.py (lbs_pool_evidence).
This is an architecture drawing, with no experimental metrics.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/s37_meshgraph_current_20260923"
INK = "#172b42"
MUTED = "#536579"
BLUE = "#2873b9"
TEAL = "#167c79"
PURPLE = "#8158ac"
ORANGE = "#b86a21"
BG = "#ffffff"


def main():
    cfg = yaml.safe_load((ROOT / "configs/semkine/s37_meshgraph_s3407.yaml").read_text())
    mc = cfg["MODEL"]
    assert mc["ENCODER"] == "mesh_graph"
    assert mc["PREVPOS_EMBED"] and mc["PREDICT_DELTA"] and mc["ZERO_EVENT_GATE"]
    assert mc["MESH_GRAPH_NODE_ID"] and not mc["MESH_GRAPH_OBS_FLOW"]
    assert mc["ENCODER_HIDDEN"] == 128 and mc["ACTIVE_HIDDEN"] == 64
    assert mc["ENCODER_LAYERS"] == 3

    font = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({"font.family": font_manager.FontProperties(fname=font).get_name(),
                         "svg.fonttype": "path", "axes.unicode_minus": False})
    fig, ax = plt.subplots(figsize=(19.2, 14.2))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1920)
    ax.set_ylim(1420, 0)
    ax.axis("off")

    def text(x, y, s, size=12, color=INK, ha="left", weight="normal"):
        return ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
                       fontweight=weight, linespacing=1.65, zorder=6)

    def box(x, y, w, h, title, body, color=TEAL, fill="#f1f9f8", size=11):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                     boxstyle="round,pad=0,rounding_size=12", linewidth=1.35,
                     edgecolor=color, facecolor=fill, zorder=3))
        text(x + w / 2, y + 28, title, 13, color, "center", "bold")
        if body:
            text(x + w / 2, y + 34 + (h - 34) / 2, body, size, INK, "center")

    def arrow(points, color=TEAL, dashed=False, width=1.6):
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, lw=width, solid_capstyle="round",
                linestyle=(0, (5, 4)) if dashed else "-", zorder=2)
        p, q = points[-2], points[-1]
        dx, dy = q[0] - p[0], q[1] - p[1]
        length = max((dx * dx + dy * dy) ** 0.5, 1)
        start = (q[0] - dx / length * min(16, length),
                 q[1] - dy / length * min(16, length))
        ax.add_patch(FancyArrowPatch(start, q, arrowstyle="-|>", mutation_scale=14,
                                    lw=width, color=color, zorder=4, shrinkA=0, shrinkB=0))

    def plus(x, y, color=PURPLE):
        ax.add_patch(Circle((x, y), 21, edgecolor=color, facecolor=BG, lw=1.8, zorder=4))
        text(x, y - 1, "+", 22, color, "center")

    def section(y, n, title, subtitle):
        ax.plot([60, 1860], [y, y], color="#dce5ee", lw=1)
        text(60, y - 22, n + "  " + title, 14, INK, weight="bold")
        text(1860, y - 22, subtitle, 10, MUTED, "right")

    text(60, 49, "S37 Mesh Graph · 当前实现网络图", 25, weight="bold")
    text(60, 91, "上一帧 MANO 网格承载当前事件 → 图卷积 → 关节读出 → 状态增量 → 递推", 13, MUTED)
    text(1860, 52, "配置：s37_meshgraph_s3407.yaml", 10, MUTED, "right")
    text(1860, 84, "维度省略 batch；紫色 prev 表示同一个上一帧状态", 10, PURPLE, "right")

    section(140, "01", "事件归属与网格编码", "几何关联在 no_grad 下计算；图的连接关系由 MANO 面片固定")
    box(60, 175, 250, 135, "上一帧状态 prev", "51 维 MANO 参数\n根平移 3 + 根旋转 3 + 关节 45", PURPLE, "#f6f0fb", 10)
    box(370, 175, 220, 135, "MANO / FK", "输入：prev + 手形 β\n输出：778 × 3 顶点")
    box(650, 175, 370, 135, "投影 + 可见性 + 归属表", "相机 K → uv、z；背面剔除 + z-buffer\n最近可见顶点：≤ 16 px；否则归背景", TEAL, "#f1f9f8", 10.5)
    box(1110, 175, 310, 135, "固定网格拓扑", "778 个顶点的面片 1-ring 边\n边输入：mesh 类型 one-hot（3 维）", TEAL, "#f1f9f8", 10.5)
    text(1480, 203, "图中共 779 个节点", 14, TEAL, weight="bold")
    text(1480, 249, "778 个网格顶点 + 1 个背景节点\n背景节点无网格边", 12, MUTED)
    arrow([(310, 242), (370, 242)], PURPLE)
    arrow([(590, 242), (650, 242)])

    box(60, 400, 250, 155, "当前事件包 E(t)", "全部事件 (x, y, t, p)\n数量可变", BLUE, "#f0f7ff")
    box(370, 400, 310, 155, "查表归属 + 每节点统计", "O：779 × 6\n数量、偏移 u/v、离散度、时间、极性", BLUE, "#f0f7ff", 10.5)
    box(740, 400, 310, 155, "观测嵌入 + 节点身份", "Linear 6 → 128\n加可学习 Node ID，再 ReLU", TEAL, "#f1f9f8", 11)
    box(1110, 400, 310, 155, "残差 EdgeConv × 3", "h ← h + 邻居消息均值\n每节点始终为 128 维", TEAL, "#f1f9f8", 11)
    box(1480, 400, 370, 155, "节点特征 H：779 × 128", "网格 Hmesh：778 × 128\n背景 hbg：128", TEAL, "#f1f9f8", 12)
    arrow([(310, 477), (370, 477)], BLUE)
    arrow([(680, 477), (740, 477)])
    arrow([(1050, 477), (1110, 477)])
    arrow([(1420, 477), (1480, 477)])
    arrow([(835, 310), (835, 353), (525, 353), (525, 400)])
    text(690, 339, "uv + 顶点归属表", 10, TEAL, "center")
    arrow([(1265, 310), (1265, 400)])
    text(1300, 357, "固定邻接", 10, TEAL)
    text(380, 582, "6 维 = log 数量份额、平均 Δu / Δv、空间 spread、平均时间、平均极性", 10, BLUE)

    section(650, "02", "LBS pooling 与独立预测头", "这一行沿箭头从右往左读；背景特征绕过 pooling")
    box(1330, 705, 520, 145, "固定 LBS pooling：778 顶点 → 16 关节", "权重 W：778 × 16；mean 128 ‖ max 128 ‖ coverage 1\n仅观测到事件的顶点参与池化；输出 E：16 × 257", TEAL, "#f1f9f8", 11)
    arrow([(1710, 555), (1710, 705)])
    text(1735, 672, "Hmesh", 11, TEAL)
    box(1110, 725, 160, 105, "关节证据 E", "16 × 257", TEAL, "#f1f9f8", 13)
    arrow([(1330, 777), (1270, 777)])
    box(660, 690, 350, 120, "root 头：Linear 4240 → 6", "16 × 257 拼平 + 背景 128\n输出：根平移 Δt（3）+ 根旋转 Δr（3）", ORANGE, "#fff6eb", 10.5)
    box(660, 850, 350, 130, "15 个独立关节头", "每头：e[j]（257）+ prev θ[j]（3）\nLinear 260 → 64 → ReLU → 3\n合计：15 × 3 = 45 维", ORANGE, "#fff6eb", 10.5)
    arrow([(1110, 777), (1060, 777), (1060, 749), (1010, 749)])
    arrow([(1060, 777), (1060, 889), (1010, 889)])
    ax.add_patch(Circle((1060, 777), 4, color=TEAL, zorder=5))
    box(1110, 885, 180, 80, "prev θ[j]", "每个关节自身的 3 维", PURPLE, "#f6f0fb", 9.5)
    arrow([(1110, 941), (1010, 941)], PURPLE, dashed=True)
    arrow([(1500, 555), (1500, 606), (835, 606), (835, 690)], BLUE)
    text(1140, 592, "背景旁路 hbg：128 维", 10.5, BLUE, "center")
    box(370, 765, 230, 125, "拼接预测", "root 6 ‖ joints 45\n顺序：[Δt, Δr, Δθ]", ORANGE, "#fff6eb", 11)
    arrow([(660, 750), (632, 750), (632, 794), (600, 794)], ORANGE)
    arrow([(660, 916), (632, 916), (632, 865), (600, 865)], ORANGE)
    box(60, 782, 250, 105, "Δheads", "51 维预测头输出", ORANGE, "#fff6eb", 12)
    arrow([(370, 835), (310, 835)], ORANGE)
    text(1335, 895, "可见性与 has 掩码作用于归属和池化。", 11, MUTED)
    text(1335, 932, "图卷积仍处理全部节点；未观测节点也有 ID 嵌入。", 10.5, MUTED)

    section(1040, "03", "加入状态旁路，再更新姿态", "当前代码在 51 维参数空间直接相加")
    box(60, 1070, 250, 95, "同一 prev", "上一帧完整状态：51 维", PURPLE, "#f6f0fb", 11)
    box(370, 1070, 340, 95, "prev_mlp 旁路", "Linear 51 → 64 → ReLU → 51", PURPLE, "#f6f0fb", 11)
    arrow([(310, 1120), (370, 1120)], PURPLE)
    plus(540, 1260)
    arrow([(185, 887), (30, 887), (30, 1260), (519, 1260)], ORANGE)
    text(335, 1238, "Δheads（51）", 11, ORANGE, "center")
    arrow([(540, 1165), (540, 1239)], PURPLE)
    text(565, 1204, "Δprev（51）", 11, PURPLE)
    box(650, 1210, 260, 105, "空事件包门", "整包无事件 → Δ = 0\n否则保留 Δheads + Δprev", ORANGE, "#fff6eb", 10.5)
    arrow([(561, 1260), (650, 1260)], ORANGE)
    plus(1015, 1260)
    arrow([(910, 1260), (994, 1260)], ORANGE)
    arrow([(310, 1091), (333, 1091), (333, 1053), (1015, 1053), (1015, 1239)], PURPLE, dashed=True)
    text(1040, 1160, "原状态 prev", 11, PURPLE)
    box(1120, 1210, 300, 105, "当前状态 s(t) = prev + Δ", "51 维 MANO 参数", PURPLE, "#f6f0fb", 12)
    arrow([(1036, 1260), (1120, 1260)], PURPLE)
    box(1480, 1210, 370, 105, "输出解码：MANO / FK", "当前关节位置 + 778 顶点 mesh", TEAL, "#f1f9f8", 11)
    arrow([(1420, 1260), (1480, 1260)])
    arrow([(1270, 1315), (1270, 1362), (14, 1362), (14, 242), (60, 242)], PURPLE, dashed=True, width=1.4)
    text(760, 1385, "时间递推：本步 s(t) 在下一事件包中作为 prev，再执行同一网络", 12, PURPLE, "center")
    text(1860, 1390, "按代码核对 · 2026-09-23", 10, MUTED, "right")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".png"):
        path = OUT.with_suffix(suffix)
        fig.savefig(path, dpi=150, facecolor=BG)
        print(path.relative_to(ROOT))
    plt.close(fig)


if __name__ == "__main__":
    main()
