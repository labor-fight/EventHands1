#!/usr/bin/env python3
"""S36 `event_gnn`, drawn in the house flow-strip template (the LNES-era style).

One horizontal strip, icon + short label per box, prev loop along the bottom. The detailed
annotated version lives in `tools/make_s36_figure.py`; this one is the poster-style summary.

    python tools/make_s36_figure_simple.py     # -> docs/assets/s36_eventgnn_simple.png

Text is CJK + ASCII only: Noto Sans CJK has no combining diacritics.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon

REPO = Path(__file__).resolve().parents[1]

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TEAL = "#0ca678"      # arrows / loop, matching the LNES figure
BLUE = "#3b7ddd"      # event-side icons
DARK = "#1f2937"      # labels
CAP = "#8b95a1"       # small captions
EDGE = "#d5dbe1"      # box borders
PANEL = "#f6f8fa"


# ------------------------------------------------------------------- icons
def hand_skel(ax, cx, cy, s, c):
    palm = (cx, cy - 0.55 * s)
    for tx, ty in [(-1.0, 0.5), (-0.5, 0.95), (0.0, 1.1), (0.5, 0.95), (1.0, 0.55)]:
        tip = (cx + tx * s, cy + ty * s)
        ax.plot([palm[0], tip[0]], [palm[1], tip[1]], color=c, lw=1.1,
                solid_capstyle="round", zorder=4)
        ax.add_patch(Circle(tip, 0.13 * s, color=c, zorder=5))
        ax.add_patch(Circle(((palm[0] + tip[0]) / 2, (palm[1] + tip[1]) / 2),
                            0.1 * s, color=c, zorder=5))
    ax.add_patch(Circle(palm, 0.2 * s, color=c, zorder=5))


def hand_fill(ax, cx, cy, s, c):
    ax.add_patch(Ellipse((cx, cy - 0.4 * s), 1.5 * s, 1.15 * s, color=c, zorder=4))
    for tx, ty in [(-0.66, 0.42), (-0.34, 0.8), (0.0, 0.92), (0.34, 0.8), (0.68, 0.4)]:
        ax.plot([cx + 0.45 * tx * s, cx + tx * s], [cy - 0.15 * s, cy + ty * s],
                color=c, lw=3.0, solid_capstyle="round", zorder=4)


def grid_icon(ax, cx, cy, s, c):
    for i in range(3):
        for j in range(3):
            ax.add_patch(plt.Rectangle((cx + (i - 1.5) * 0.62 * s, cy + (j - 1.5) * 0.62 * s),
                                       0.5 * s, 0.5 * s, facecolor="none",
                                       edgecolor=c, lw=0.9, zorder=4))


def events_icon(ax, cx, cy, s):
    pts = [(-1.0, 0.2), (-0.55, -0.6), (-0.35, 0.75), (0.05, -0.15), (0.3, 0.7),
           (0.6, -0.65), (0.95, 0.3), (-0.05, -0.85), (0.85, -0.1), (-0.75, 0.9)]
    for i, (px, py) in enumerate(pts):
        ax.add_patch(Circle((cx + px * s, cy + py * s), 0.13 * s,
                            color=BLUE if i % 2 else "#f08c00", zorder=4))


def stack_icon(ax, cx, cy, s, c):
    for i in range(3):
        d = (i - 1) * 0.3 * s
        ax.add_patch(FancyBboxPatch((cx - 0.8 * s + d, cy - 0.35 * s - d), 1.6 * s, 0.7 * s,
                                    boxstyle="round,pad=0.02,rounding_size=0.12",
                                    facecolor="#ffffff", edgecolor=c, lw=1.0, zorder=4 + i))


def graph_icon(ax, cx, cy, s, c):
    nodes = [(-0.9, -0.4), (-0.15, 0.85), (0.75, 0.45), (0.55, -0.75), (-0.2, -0.05)]
    for a, b in [(0, 1), (1, 2), (2, 4), (0, 4), (3, 4), (2, 3)]:
        ax.plot([cx + nodes[a][0] * s, cx + nodes[b][0] * s],
                [cy + nodes[a][1] * s, cy + nodes[b][1] * s], color=c, lw=0.9, zorder=4)
    for px, py in nodes:
        ax.add_patch(Circle((cx + px * s, cy + py * s), 0.16 * s, color=c, zorder=5))


def cube_icon(ax, cx, cy, s, c):
    d = 0.42 * s
    f = [(-0.7 * s, -0.7 * s), (0.7 * s, -0.7 * s), (0.7 * s, 0.7 * s), (-0.7 * s, 0.7 * s)]
    for i in range(4):
        a, b = f[i], f[(i + 1) % 4]
        ax.plot([cx + a[0], cx + b[0]], [cy + a[1], cy + b[1]], color=c, lw=1.0, zorder=4)
        ax.plot([cx + a[0] + d, cx + b[0] + d], [cy + a[1] + d, cy + b[1] + d],
                color=c, lw=1.0, zorder=4)
        ax.plot([cx + a[0], cx + a[0] + d], [cy + a[1], cy + a[1] + d],
                color=c, lw=1.0, zorder=4)


def funnel_icon(ax, cx, cy, s, c):
    ax.add_patch(Polygon([(cx - 0.9 * s, cy + 0.75 * s), (cx + 0.9 * s, cy + 0.75 * s),
                          (cx + 0.15 * s, cy - 0.2 * s), (cx + 0.15 * s, cy - 0.85 * s),
                          (cx - 0.15 * s, cy - 0.85 * s), (cx - 0.15 * s, cy - 0.2 * s)],
                         closed=True, facecolor="none", edgecolor=c, lw=1.1, zorder=4))


# ------------------------------------------------------------------- pieces
def node(ax, x, y, w, h, label, icon=None, icon_color=TEAL, fs=8.6, caption=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.3",
                                facecolor="#ffffff", edgecolor=EDGE, lw=1.2, zorder=3))
    if icon is not None:
        icon(ax, x + 4.0, y + h / 2, 2.1, icon_color) if icon is not events_icon \
            else events_icon(ax, x + 4.0, y + h / 2, 2.1)
        tx = x + 4.0 + (w - 4.0) / 2 + 1.2
    else:
        tx = x + w / 2
    ax.text(tx, y + h / 2, label, ha="center", va="center", fontsize=fs,
            color=DARK, zorder=6)
    if caption:
        ax.text(x + w / 2, y - 1.6, caption, ha="center", va="top",
                fontsize=6.4, color=CAP, zorder=6)


def arrow(ax, p0, p1, head=True, lw=1.5):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>" if head else "-",
                                 mutation_scale=11, lw=lw, color=TEAL,
                                 shrinkA=0, shrinkB=0, zorder=2))


def elbow(ax, pts, lw=1.5):
    for a, b in zip(pts, pts[1:-1]):
        arrow(ax, a, b, head=False, lw=lw)
    arrow(ax, pts[-2], pts[-1], head=True, lw=lw)


def main() -> None:
    mc = yaml.safe_load((REPO / "configs/semkine/s36_eventgnn_s3407.yaml").read_text())["MODEL"]
    K, WIN, MN = mc["ENCODER_K"], mc["ENCODER_WINDOW"], mc["ENCODER_MAX_NODES"]
    FEAT, NL = mc["ENCODER_FEAT"], mc["ENCODER_LAYERS"]

    fig, ax = plt.subplots(figsize=(17.0, 3.6))
    ax.set_xlim(0, 214); ax.set_ylim(0, 45); ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 1), 212, 43, boxstyle="round,pad=0.3,rounding_size=2.5",
                                facecolor=PANEL, edgecolor="#e5e7eb", lw=1.0, zorder=0))
    ax.text(4, 42.3, "S36 EventGNN — 事件流 GNN 递归追踪", fontsize=8.2,
            color=CAP, va="top", zorder=6)

    # 状态支路（上排）
    node(ax, 4, 17, 17, 10, "prev 51D\nMANO", hand_skel, TEAL, fs=8.2)
    node(ax, 25, 30, 16, 10, "MANO FK", hand_skel, TEAL)
    node(ax, 44, 30, 18, 10, "投影\n+ z-buffer", grid_icon, TEAL, fs=8.2)
    node(ax, 65, 30, 21, 10, "silhouette /\n逆深度图", hand_fill, "#1e3a5f", fs=8.2)
    elbow(ax, [(12.5, 27.4), (12.5, 35), (24.6, 35)])
    arrow(ax, (41.4, 35), (43.6, 35))
    arrow(ax, (62.4, 35), (64.6, 35))

    # 事件支路（下排）
    node(ax, 58, 6, 23, 10, "异步事件流\n(x, y, t, p)", events_icon, BLUE, fs=8.2)

    # 主干
    node(ax, 91, 17, 21, 10, "逐事件\n拼 9 维", stack_icon, TEAL, fs=8.2,
         caption="7 维 token ＋ 渲染 2 维")
    node(ax, 115, 17, 20, 10, "因果 k-NN\n建图", graph_icon, BLUE, fs=8.2,
         caption=f"前 {WIN} 里取 k={K} · ≤{MN} 点")
    node(ax, 138, 17, 18, 10, f"EdgeConv\n× {NL}", cube_icon, BLUE, fs=8.2,
         caption="边带 (dx, dy, dt)")
    node(ax, 159, 17, 17, 10, "池化 +\n解码头", funnel_icon, TEAL, fs=8.2,
         caption=f"mean‖max → {FEAT}")
    node(ax, 179, 17, 7, 10, "Δ", None, fs=13, caption="空包 Δ=0")

    elbow(ax, [(86.4, 35), (98, 35), (98, 27.4)])          # 渲染图 -> 拼接
    elbow(ax, [(81.4, 11), (94, 11), (94, 16.6)])          # 事件流 -> 拼接
    arrow(ax, (112.4, 22), (114.6, 22))
    arrow(ax, (135.4, 22), (137.6, 22))
    arrow(ax, (156.4, 22), (158.6, 22))
    arrow(ax, (176.4, 22), (178.6, 22))

    # 求和与输出
    ax.add_patch(Circle((191, 22), 1.7, facecolor="#ffffff", edgecolor=TEAL,
                        lw=1.4, zorder=5))
    ax.text(191, 22, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    node(ax, 196, 17, 15, 10, "x_k = prev + Δ", None, fs=8.6)
    arrow(ax, (186.4, 22), (189.2, 22))
    arrow(ax, (192.8, 22), (195.6, 22))

    # prev 回环（底部走廊）
    elbow(ax, [(12.5, 16.6), (12.5, 3.6), (191, 3.6), (191, 20.2)], lw=1.5)
    ax.text(120, 4.6, "prev（prev_mlp 支路一并加入 Δ；本步输出即下一步的 prev）",
            ha="center", va="bottom", fontsize=7.0, color=TEAL, zorder=6)

    out = REPO / "docs/assets/s36_eventgnn_simple.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
