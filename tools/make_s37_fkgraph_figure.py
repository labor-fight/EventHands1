#!/usr/bin/env python3
"""S37 FK graph, drawn in the S36 flow-strip template with the changes boxed in orange.

Processing modules only: FK, graph, event assignment + per-node summary, embedding, EdgeConv,
decoders, delta, sum. Against S36 there is no event graph, no token, no rendering; the FK of the
previous state is the graph and the events are its observations. Numbers in the banner come from
the main-row JSONs.

    python tools/make_s37_fkgraph_figure.py     # -> docs/assets/s37_fkgraph_simple.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from matplotlib.patches import Circle, FancyBboxPatch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from make_s36_figure_simple import (BLUE, CAP, DARK, PANEL, TEAL, arrow, cube_icon, elbow,   # noqa: E402
                                    events_icon, funnel_icon, graph_icon, hand_skel, node,
                                    stack_icon)

ORANGE = "#e8590c"


def hnode(ax, x, y, w, h, label, icon=None, icon_color=ORANGE, fs=8.2, caption=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.3",
                                facecolor="#fff7f1", edgecolor=ORANGE, lw=1.8, zorder=3))
    if icon is not None:
        if icon is events_icon:
            events_icon(ax, x + 4.0, y + h / 2, 2.1)
        else:
            icon(ax, x + 4.0, y + h / 2, 2.1, icon_color)
        tx = x + 4.0 + (w - 4.0) / 2 + 1.2
    else:
        tx = x + w / 2
    ax.text(tx, y + h / 2, label, ha="center", va="center", fontsize=fs, color=DARK, zorder=6)
    if caption:
        ax.text(x + w / 2, y - 1.6, caption, ha="center", va="top", fontsize=6.4, color=CAP, zorder=6)


def bins_icon(ax, cx, cy, s, c):
    """Events (dots) falling into node bins (bars): assignment + per-node summary."""
    for i, hgt in enumerate((0.5, 1.1, 0.7, 1.4, 0.4)):
        x0 = cx + (i - 2) * 0.42 * s
        ax.add_patch(plt.Rectangle((x0 - 0.15 * s, cy - 0.9 * s), 0.3 * s, hgt * s,
                                   facecolor="none", edgecolor=c, lw=0.9, zorder=4))
    for px, py in [(-0.8, 0.8), (-0.4, 1.0), (0.1, 0.7), (0.5, 1.05), (0.9, 0.6)]:
        ax.add_patch(Circle((cx + px * s, cy + py * s), 0.11 * s, color="#f08c00", zorder=5))


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs/semkine/s37_fkgraph_s3407.yaml").read_text())
    mc = cfg["MODEL"]
    assert mc["ENCODER"] == "fk_graph" and not mc["PREV_RENDER"]
    V, BAND, KM = mc["FK_GRAPH_VERTS"], mc["FK_GRAPH_BAND_PX"], mc["FK_GRAPH_K"]
    NL, HID = mc["ENCODER_LAYERS"], mc["ENCODER_HIDDEN"]

    def ra(name):
        p = REPO / f"outputs/semkine/{name}_main_row.json"
        return json.loads(p.read_text())["two_seed_mean"]["overall"]["mpjpe_ra_mm"] if p.exists() else None

    ra36, ra37 = ra("s36_eventgnn"), ra("s37_fkgraph")

    fig, ax = plt.subplots(figsize=(17.0, 3.6))
    ax.set_xlim(0, 214); ax.set_ylim(0, 45); ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 1), 212, 43, boxstyle="round,pad=0.3,rounding_size=2.5",
                                facecolor=PANEL, edgecolor="#e5e7eb", lw=1.0, zorder=0))
    ax.text(4, 42.3, "S37 FK 图 — prev 的 MANO FK 就是图，事件是它的观测", fontsize=8.2,
            color=CAP, va="top", zorder=6)
    banner = "橙框 = 相对 S36 的改动：无事件图 / token / 渲染，FK 图取代事件图，事件汇成节点观测"
    if ra36 and ra37:
        banner += f"  |  两种子递推 RA {ra36:.2f} → {ra37:.2f} mm（{ra37 - ra36:+.2f}）"
    ax.text(210, 42.3, banner, fontsize=7.0, color=ORANGE, ha="right", va="top", zorder=6)

    # 状态支路（上排）
    node(ax, 4, 17, 17, 10, "prev 51D\nMANO", hand_skel, TEAL, fs=8.2)
    node(ax, 25, 30, 16, 10, "MANO FK", hand_skel, TEAL, caption="778 顶点")
    hnode(ax, 44, 30, 32, 10, f"FK 图：16 关节 + {V} 顶点 + 背景\n边 = 网格 kNN / LBS / 运动学树", graph_icon,
          ORANGE, fs=7.8, caption=f"静止姿态 k={KM} · LBS top-2 · kintree · 边特征 = 边类型 · 固定拓扑")
    elbow(ax, [(12.5, 27.4), (12.5, 35), (24.6, 35)])
    arrow(ax, (41.4, 35), (43.6, 35))

    # 事件支路（下排）
    node(ax, 56, 6, 25, 10, "异步事件流\n(x, y, t, p) 全部事件", events_icon, BLUE, fs=8.0)

    # 主干
    hnode(ax, 91, 17, 25, 10, "就近归属 + 汇总\n每节点 8 维观测", bins_icon, ORANGE, fs=8.0,
          caption=f"d ≤ {BAND:g} px 否则背景 · 数量 / 偏移 / 离散 / 时间 / 极性 / 流速")
    hnode(ax, 119, 17, 20, 10, "观测嵌入 +\n节点身份嵌入", stack_icon, ORANGE, fs=8.0, caption=f"→ {HID} 维")
    hnode(ax, 142, 17, 17, 10, f"EdgeConv\n× {NL}", cube_icon, ORANGE, fs=8.2, caption="在 FK 图上")
    node(ax, 162, 17, 22, 10, "15 指头读关节节点\nroot 读 16 关节+背景", funnel_icon, TEAL, fs=7.6,
         caption="+ 自身 prev 角")
    node(ax, 187, 17, 6, 10, "Δ", None, fs=13, caption="空包 Δ=0")

    elbow(ax, [(76.4, 35), (103, 35), (103, 27.4)])          # FK 图（含各节点像素）-> 归属
    elbow(ax, [(81.4, 11), (97, 11), (97, 16.6)])            # 事件 -> 归属
    arrow(ax, (116.4, 22), (118.6, 22))
    arrow(ax, (139.4, 22), (141.6, 22))
    arrow(ax, (159.4, 22), (161.6, 22))
    arrow(ax, (184.4, 22), (186.6, 22))

    # 求和与输出
    ax.add_patch(Circle((197, 22), 1.7, facecolor="#ffffff", edgecolor=TEAL, lw=1.4, zorder=5))
    ax.text(197, 22, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    node(ax, 200, 17, 12, 10, "x_k =\nprev + Δ", None, fs=8.2)
    arrow(ax, (193.4, 22), (195.2, 22))
    arrow(ax, (198.8, 22), (199.6, 22))

    # prev 回环（底部走廊）
    elbow(ax, [(12.5, 16.6), (12.5, 3.6), (197, 3.6), (197, 20.2)], lw=1.5)
    ax.text(125, 4.4, "prev（prev_mlp 支路一并加入 Δ；本步输出即下一步的 prev）",
            ha="center", va="bottom", fontsize=6.8, color=TEAL, zorder=6)

    out = REPO / "docs/assets/s37_fkgraph_simple.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
