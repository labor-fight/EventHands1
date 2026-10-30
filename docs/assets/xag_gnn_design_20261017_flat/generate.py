"""Flat, code-faithful XAG F+ explanation figures; documentation only."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
FONT = font_manager.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams.update({"font.family": FONT.get_name(), "svg.fonttype": "none", "axes.unicode_minus": False})
INK, MUTED = "#17304b", "#52677d"
BLUE, BLUE_BG = "#337cac", "#edf5fd"
ORANGE, ORANGE_BG = "#d66c0e", "#fff1df"
GRAY, GRAY_BG = "#8091a2", "#f4f7fa"
TEAL = "#217c80"


def canvas(height):
    fig = plt.figure(figsize=(28, height / 100), dpi=100, facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 2800), ylim=(height, 0))
    ax.axis("off")
    return fig, ax


def text(ax, x, y, value, size=17, color=INK, ha="center", weight="normal", **kw):
    ax.text(x, y, value, fontproperties=FONT, fontsize=size, color=color,
            ha=ha, va="center", fontweight=weight, linespacing=1.42, zorder=5, **kw)


def box(ax, x, y, w, h, value, kind="old", size=17):
    color, bg = {"old": (BLUE, BLUE_BG), "new": (ORANGE, ORANGE_BG), "fixed": (GRAY, GRAY_BG)}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=10",
                              facecolor=bg, edgecolor=color, linewidth=1.6, zorder=3))
    text(ax, x + w / 2, y + h / 2, value, size=size)


def arrow(ax, a, b, color=GRAY, dashed=False):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=15,
                                color=color, linewidth=1.7, linestyle="--" if dashed else "-",
                                shrinkA=2, shrinkB=3, zorder=2))


def route(ax, points, color=GRAY, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.6,
                linestyle="--" if dashed else "-", zorder=1)
    arrow(ax, points[-2], points[-1], color, dashed)


def save(fig, name):
    fig.savefig(HERE / (name + ".svg"), facecolor="white")
    fig.savefig(HERE / (name + ".png"), dpi=140, facecolor="white")
    plt.close(fig)


def overview():
    fig, ax = canvas(400)
    text(ax, 28, 30, "增强 GNN F+｜三次插入全局块，原局部传播与读出接口保留", 23, ha="left", weight="bold")
    text(ax, 28, 70, "蓝色＝原模块   橙色＝新增模块   H0 / H3：778 顶点 × 128 维；省略 batch 维", 16, ha="left", color=MUTED)
    xs = [(350, 150), (530, 220), (790, 150), (970, 220), (1230, 150), (1410, 220)]
    for cy, enhanced in ((141, False), (273, True)):
        text(ax, 28, cy, "F+ 增强" if enhanced else "F0 原版", 19, ha="left", weight="bold")
        box(ax, 180, cy - 34, 125, 68, "H0\n778×128", "fixed", 16)
        end = 305
        for index, (x, width) in enumerate(xs):
            if index % 2:
                if enhanced:
                    arrow(ax, (end, cy), (x, cy), ORANGE)
                    box(ax, x, cy - 45, width, 90,
                        f"新增块 {index // 2 + 1}\nLN·4头注意力·残差\nLN·FFN·残差", "new", 15.5)
                    end = x + width
            else:
                arrow(ax, (end, cy), (x, cy))
                box(ax, x, cy - 34, width, 68, f"原局部层 {index // 2 + 1}\n1-ring", size=16)
                end = x + width
        arrow(ax, (end, cy), (1680, cy))
        box(ax, 1680, cy - 34, 125, 68, "H3\n778×128", "fixed", 16)
        route(ax, [(1805, cy), (1830, cy), (1830, 207), (1860, 207)])
    text(ax, 2270, 121, "原共同读出｜两版结构相同", 18, color=BLUE)
    box(ax, 1860, 162, 290, 90, "LBS池化＋原事件残差槽\n16×258 ＋ 1×258", size=16)
    arrow(ax, (2150, 207), (2190, 207))
    box(ax, 2190, 162, 235, 90, "原逐槽压缩\n17×258 → 544", size=16)
    arrow(ax, (2425, 207), (2465, 207))
    box(ax, 2465, 162, 305, 90, "拼接 prev48 → 原 MLP\n[544＋48]→256→51", size=16)
    text(ax, 2315, 292, "原残差槽绕过GNN；prev48在读出时拼接。", 15, color=MUTED)
    text(ax, 28, 355, "改动目的：局部整理 → 远处支持顶点交流 → 下一轮局部再整理；三块各自独立训练。", 17, ha="left", color=ORANGE)
    text(ax, 28, 385, "A0F1沿用原事件关联；支持集S仍仅由原局部层逐轮扩张。", 15.5, ha="left", color=MUTED)
    save(fig, "overview_flat")


def detail():
    fig, ax = canvas(490)
    text(ax, 28, 29, "一轮展开｜新增算子、准确接线与各自目的", 23, ha="left", weight="bold")
    text(ax, 28, 68, "橙色均为新增；三轮结构相同、参数独立。下图 U / H1 / Hout 始终为 778×128。", 16, ha="left", color=MUTED)
    ax.add_patch(FancyBboxPatch((285, 89), 2285, 352, boxstyle="round,pad=0,rounding_size=12",
                               facecolor="#fffcf7", edgecolor="#e8be8f", linewidth=1.2, linestyle="--", zorder=0))
    cy = 262
    box(ax, 25, 214, 225, 96, "原局部 GNN\n邻边消息＋几何门\n得到 U，更新 S", size=16)
    box(ax, 320, 214, 195, 96, "① 新增 LN1\nZ = LN1(U)", "new", 17)
    box(ax, 710, 196, 515, 132,
        "③ 新增全局四头注意力\nQ / K 输入：Z＋P；V 输入：Z\n4头×32维；输出A（内部Q/K/V投影）", "new", 17)
    box(ax, 1285, 214, 235, 96, "④ 新增残差1＋M\nH1=M⊙(U＋A)", "new", 16.5)
    box(ax, 1580, 214, 170, 96, "⑤ 新增 LN2\nT=LN2(H1)", "new", 16.5)
    box(ax, 1810, 214, 440, 96, "⑥ 新增逐顶点 FFN\nLinear128→256 → GELU\n→ Linear256→128，输出F", "new", 16.5)
    box(ax, 2310, 214, 235, 96, "⑦ 新增残差2＋M\nHout=M⊙(H1＋F)", "new", 15.5)
    box(ax, 2600, 214, 175, 96, "输出 Hout\n→ 下一局部层\n末轮 → 原池化", "fixed", 15.5)
    for a, b in ((250, 320), (515, 710), (1225, 1285), (1520, 1580), (1750, 1810), (2250, 2310), (2545, 2600)):
        arrow(ax, (a, cy), (b, cy), ORANGE if a >= 320 else GRAY)
    box(ax, 320, 106, 290, 62, "固定 LBS：778×16", "fixed", 16)
    box(ax, 650, 106, 500, 62, "② 新增无偏 Linear：16→128，得到 P", "new", 16)
    arrow(ax, (610, 137), (650, 137), ORANGE)
    arrow(ax, (965, 168), (965, 196), ORANGE)
    text(ax, 1220, 137, "只进入 Q/K", 14.5, ORANGE)
    box(ax, 1405, 106, 1075, 62, "新增支持约束 M=1[S>0]：屏蔽无支持 keys；两次残差后清零无支持输出", "new", 15.8)
    route(ax, [(1405, 137), (1320, 137), (1320, 182), (1165, 182), (1165, 196)], TEAL, True)
    arrow(ax, (1480, 168), (1480, 214), TEAL, True)
    arrow(ax, (2460, 168), (2460, 214), TEAL, True)
    text(ax, 416, 344, "目的：统一通道尺度", 14.5, ORANGE)
    text(ax, 965, 344, "目的：跨部位读取已有支持的证据", 15, ORANGE)
    text(ax, 1402, 344, "目的：保留局部特征", 14.5, ORANGE)
    text(ax, 2015, 344, "目的：非线性重组收到的信息", 15, ORANGE)
    text(ax, 2427, 344, "目的：保留融合状态", 14, ORANGE)
    route(ax, [(275, cy), (275, 383), (1402, 383), (1402, 310)], BLUE)
    text(ax, 795, 399, "跳连1：原局部输出 U", 14.5, BLUE)
    route(ax, [(1550, cy), (1550, 419), (2427, 419), (2427, 310)], ORANGE)
    text(ax, 1980, 436, "跳连2：注意力融合后的 H1", 14.5, ORANGE)
    text(ax, 28, 475, "P提示部位、影响读取权重，不直接作为V内容；M只控制支持资格，不表示关联正确或可靠。", 16, ha="left", color=MUTED)
    save(fig, "new_block_flat")


if __name__ == "__main__":
    overview()
    detail()
