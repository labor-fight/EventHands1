"""Render the XAG F+ architecture directly from the implemented module contract.

Run from the repository root. This script only writes documentation figures.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
FONT = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": FONT.get_name(), "svg.fonttype": "none", "axes.unicode_minus": False})

INK = "#17283F"
MUTED = "#51657C"
BLUE = "#3477A5"
BLUE_BG = "#EDF5FC"
ORANGE = "#CE680E"
ORANGE_BG = "#FFF0DC"
GRAY = "#7D8C9A"
GRAY_BG = "#F4F6F8"
EDGE = "#A4B4C5"


def label(ax, x, y, text, size=13, color=INK, ha="center", va="center", weight="normal", **kw):
    return ax.text(x, y, text, fontsize=size, color=color, ha=ha, va=va,
                   fontproperties=FONT, fontweight=weight, linespacing=1.55, **kw)


def box(ax, x, y, w, h, text, kind="old", size=12.6):
    colors = {"old": (BLUE_BG, BLUE), "new": (ORANGE_BG, ORANGE),
              "neutral": (GRAY_BG, GRAY)}
    bg, ec = colors[kind]
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.12",
                           facecolor=bg, edgecolor=ec, linewidth=1.5, zorder=3)
    ax.add_patch(patch)
    label(ax, x + w / 2, y + h / 2, text, size=size, zorder=4)
    return patch


def arrow(ax, x0, y0, x1, y1, color=GRAY, width=1.5, **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=13,
                                linewidth=width, color=color, shrinkA=3, shrinkB=3,
                                zorder=2, **kw))


def route(ax, points, color=GRAY, width=1.4, arrow_end=True):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=width, zorder=1)
    a, b = points[-2:]
    if arrow_end:
        arrow(ax, *a, *b, color=color, width=width)
    else:
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=width, zorder=1)


def base(width=18, height=6.6, ylim=8.7):
    fig, ax = plt.subplots(figsize=(width, height), dpi=120)
    fig.patch.set_facecolor("white")
    ax.set(xlim=(0, 24), ylim=(0, ylim))
    ax.axis("off")
    fig.subplots_adjust(0.015, 0.02, 0.985, 0.98)
    return fig, ax


def save(fig, stem):
    fig.savefig(HERE / f"{stem}.svg", facecolor="white")
    fig.savefig(HERE / f"{stem}.png", dpi=120, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = base()
    label(ax, .25, 8.32, "增强 GNN（F+）：每个原局部层后，插入一块全局注意力 + FFN", size=20,
          ha="left", weight="bold")
    label(ax, .25, 7.74, "橙色 NEW = 新增可学习模块    蓝色 = 原有模块    所有顶点特征始终为 778 × 128（省略 batch 维）",
          size=12.6, ha="left", color=MUTED)
    rows = [(5.75, "F0｜原 GNN", False), (3.05, "F+｜增强 GNN", True)]
    for y, title, enhanced in rows:
        label(ax, .25, y + 1.02, title, size=16, ha="left", weight="bold")
        box(ax, .25, y - .43, 1.6, .86, "H0\n778 × 128", "neutral", 12)
        last = 1.85
        for i, x in enumerate([2.4, 6.9, 11.4], 1):
            arrow(ax, last, y, x, y)
            box(ax, x, y - .48, 1.75, .96, f"原局部层 {i}\n1-ring", size=12.1)
            last = x + 1.75
            if enhanced:
                nx = x + 2.10
                arrow(ax, last, y, nx, y, color=ORANGE)
                box(ax, nx, y - .57, 1.90, 1.14, f"新增 {i}\n全局注意力\n+ FFN", "new", 11.8)
                last = nx + 1.90
        arrow(ax, last, y, 15.95, y)
        box(ax, 15.95, y - .43, 1.4, .86, "H3\n778 × 128", "neutral", 11.7)
        arrow(ax, 17.35, y, 18.05, y)
    label(ax, 8.65, 1.95, "三块 NEW 各有独立参数；原三层仍保留，支持集 S 只由原局部层逐轮拓展。",
          size=12.8, color=ORANGE)
    box(ax, 18.05, 2.45, 5.50, 4.25,
        "原共同读出（两条路径相同）\n"
        "LBS pooling：16 × 258\n"
        "＋原残差槽：1 × 258\n"
        "分槽压缩：17 × 258 → 17 × 32\n"
        "展平：544；拼接前态：544 + 48\n"
        "原 MLP：592 → 256 → 51\n"
        "输出姿态 / 平移更新 δ", size=12.1)
    box(ax, 19.00, 7.00, 3.60, .56, "固定 LBS：778 × 16", "neutral", 11.8)
    arrow(ax, 20.8, 7.00, 20.8, 6.70)
    box(ax, 17.85, .65, 2.85, .95, "原事件残差支路\n未 / 弱匹配事件 → 1 × 258", size=10.5)
    route(ax, [(19.27, 1.60), (19.27, 2.00), (19.27, 2.45)])
    box(ax, 21.15, .65, 2.55, .95, "原前态 prev48\n48 维", size=11.7)
    route(ax, [(22.42, 1.60), (22.42, 2.00), (22.42, 2.45)])
    label(ax, .25, .76,
          "关系编码 → 顶点聚合 → H0 保留；\nA0F1 只启用 F+，event–网格关联使用原方案 A0。",
          size=12.4, color=MUTED, ha="left")
    save(fig, "overview_horizontal")


def detail():
    fig, ax = base(width=20, height=8.2, ylim=10.0)
    label(ax, .25, 9.63, "一轮放大：原 1-ring 局部消息 → 新增全局注意力残差 → 新增 FFN 残差", size=20,
          ha="left", weight="bold")
    label(ax, .25, 9.10, "三轮都采用同一结构；模块间不共享参数。橙色区域为新增模块。", size=12.8,
          ha="left", color=MUTED)

    # A tinted envelope distinguishes all new computations from the historical layer.
    envelope = FancyBboxPatch((3.25, 2.35), 18.05, 6.05,
                              boxstyle="round,pad=0.03,rounding_size=0.16",
                              facecolor="#FFFAF2", edgecolor="#EDBA82", linewidth=1.3,
                              linestyle="--", zorder=0)
    ax.add_patch(envelope)
    label(ax, 20.95, 8.10, "NEW｜GlobalResidual", size=13.2, color=ORANGE,
          ha="right", weight="bold")

    cy = 5.65
    box(ax, .20, 5.00, 2.45, 1.30, "原 1-ring 局部层\nH → U\n更新支持集 S", size=12.0)
    label(ax, 1.43, 6.83, "原有", size=12.6, color=BLUE)
    box(ax, 3.60, 5.15, 1.55, 1.00, "LN1\nZ = LN1(U)", "new", 11.1)
    box(ax, 7.10, 4.85, 3.25, 1.60,
        "全局 4 头注意力\nQ / K 输入：Z + P\nV 输入：Z\n每头 32 维", "new", 12.0)
    box(ax, 10.95, 5.05, 1.75, 1.20, "残差①\n相加 + mask\n得到 H′", "new", 11.5)
    box(ax, 13.35, 5.15, 1.45, 1.00, "LN2\nLN2(H′)", "new", 11.4)
    box(ax, 15.45, 5.05, 3.05, 1.20, "FFN\n128 → 256 → 128\nGELU", "new", 12.0)
    box(ax, 19.15, 5.05, 1.75, 1.20, "残差②\n相加 + mask\n得到 Hout", "new", 11.4)
    box(ax, 21.70, 5.10, 2.05, 1.10, "输出 Hout\n778 × 128", "neutral", 12.0)
    for x0, x1 in [(2.65, 3.60), (5.15, 7.10), (10.35, 10.95), (12.70, 13.35),
                   (14.80, 15.45), (18.50, 19.15), (20.90, 21.70)]:
        arrow(ax, x0, cy, x1, cy, color=ORANGE if x0 >= 3.0 else GRAY)
    label(ax, 6.10, 6.08, "Z → Q / K", size=11.4, color=ORANGE)

    # LBS is solely the positional/part identity stream for Q and K.
    box(ax, 3.60, 7.15, 2.80, .85, "固定 LBS 权重\n778 × 16", "neutral", 12.0)
    box(ax, 7.10, 7.15, 3.25, .85, "新增 Linear（无 bias）\n16 → 128，得到 P", "new", 12.0)
    arrow(ax, 6.40, 7.57, 7.10, 7.57, color=ORANGE)
    arrow(ax, 8.72, 7.15, 8.72, 6.45, color=ORANGE)
    label(ax, 10.05, 6.87, "P 仅加入 Q / K", size=11.1, color=ORANGE)
    route(ax, [(4.38, 5.15), (4.38, 4.28), (7.75, 4.28), (7.75, 4.85)], color=ORANGE)
    label(ax, 6.05, 4.02, "V = Z（不含 LBS）", size=11.4, color=ORANGE)
    label(ax, 8.72, 3.58, "MHA 内部另有 WQ / WK / WV / WO 投影", size=10.8, color=MUTED)

    # Each residual receives the feature immediately before its own pre-norm block.
    route(ax, [(2.65, cy), (2.98, cy), (2.98, 2.85), (11.83, 2.85), (11.83, 5.05)],
          color=BLUE)
    label(ax, 7.25, 2.56, "跳连①：U（原局部层输出）", size=12.1, color=BLUE)
    route(ax, [(12.70, cy), (13.02, cy), (13.02, 3.78), (20.03, 3.78), (20.03, 5.05)],
          color=ORANGE)
    label(ax, 16.55, 3.43, "跳连②：H′（注意力残差后的特征）", size=11.8, color=ORANGE)

    label(ax, .25, 1.64, "支持集与掩码规则", size=14.8, ha="left", weight="bold")
    label(ax, .25, .96,
          "① 注意力只允许 S > 0 的顶点作为 keys。  ② 两次残差相加后，对 S = 0 的输出清零。\n"
          "③ 全空样本临时开放一个 dummy key 防 NaN，最终仍整行置零。  ④ 新增模块不拓展 S，支持范围由原局部层决定。",
          size=12.5, ha="left", color=MUTED)
    save(fig, "global_residual_detail_horizontal")


if __name__ == "__main__":
    overview()
    detail()
