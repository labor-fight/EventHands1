#!/usr/bin/env python3
"""S37 arms in the S36 flow-strip template, changes against S36 boxed in orange.

    python tools/make_s37_figure_simple.py                 # mesh query (the user's design)
    python tools/make_s37_figure_simple.py --arm routed    # the intermediate routed-readout arm

Outputs docs/assets/s37_meshq_simple.png / s37_routed_simple.png. Banner numbers come from the
main-row JSONs when they exist.
"""
from __future__ import annotations

import argparse
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
                                    events_icon, funnel_icon, graph_icon, grid_icon,
                                    hand_skel, node, stack_icon)

ORANGE = "#e8590c"


def hnode(ax, x, y, w, h, label, icon=None, icon_color=ORANGE, fs=8.2, caption=None):
    """`node`, with the orange border that marks a change against S36."""
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


def query_icon(ax, cx, cy, s, c):
    """A query vertex (orange) reading the graph nodes around it inside a kernel."""
    ax.add_patch(Circle((cx, cy), 0.95 * s, facecolor="none", edgecolor=c, lw=0.8,
                        ls=(0, (2, 1.5)), zorder=4))
    for px, py in [(-0.55, 0.45), (0.5, 0.55), (0.6, -0.4), (-0.4, -0.55), (0.05, 0.75)]:
        ax.add_patch(Circle((cx + px * s, cy + py * s), 0.12 * s, color=BLUE, zorder=5))
        ax.plot([cx, cx + px * s], [cy, cy + py * s], color=c, lw=0.7, zorder=4)
    ax.add_patch(Circle((cx, cy), 0.2 * s, facecolor="#f08c00", edgecolor="none", zorder=6))


def ra(name):
    p = REPO / f"outputs/semkine/{name}_main_row.json"
    return json.loads(p.read_text())["two_seed_mean"]["overall"]["mpjpe_ra_mm"] if p.exists() else None


def canvas(title, banner, width=234, height=46):
    fig, ax = plt.subplots(figsize=(width / 12.6, height / 12.4))
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 1), width - 2, height - 2, boxstyle="round,pad=0.3,rounding_size=2.5",
                                facecolor=PANEL, edgecolor="#e5e7eb", lw=1.0, zorder=0))
    ax.text(4, height - 2.7, title, fontsize=8.2, color=CAP, va="top", zorder=6)
    ax.text(width - 4, height - 2.7, banner, fontsize=7.4, color=ORANGE, ha="right", va="top", zorder=6)
    return fig, ax


def draw_meshq(mc):
    K, WIN, MN = mc["ENCODER_K"], mc["ENCODER_WINDOW"], mc["ENCODER_MAX_NODES"]
    NL, HID = mc["ENCODER_LAYERS"], mc["ENCODER_HIDDEN"]
    Q, BAND, SIG = mc["MESH_QUERY_TOKENS"], mc["MESH_QUERY_BAND_PX"], mc["MESH_QUERY_SIGMA_PX"]
    EV = HID + 4 + 3 + 1
    ra36, ra37 = ra("s36_eventgnn"), ra("s37_meshq")
    banner = ("橙框 = 相对 S36 的改动：prev 只以 FK 网格进入（无渲染、无 prev_mlp、头不读 prev 角）· 事件直接建图（无 token）"
              "· 网格读图驱动关节")
    if ra36 and ra37:
        banner += f"  |  两种子递推 RA {ra36:.2f} → {ra37:.2f} mm（{ra37 - ra36:+.2f}）"
    fig, ax = canvas("S37 EventGNN + 网格查询 — prev 的 MANO FK 读事件图的节点与边，驱动关节", banner, height=52)
    ax.text(117, 45.6,
            f"w_qi = softmax(−d²/2·{SIG:g}²)，d ≤ {BAND:g} px，可见查询顶点；v_q = [Σw h_i ‖ Σw g_i ‖ Σw (p_i − u_q) ‖ mass]；"
            f"e_j = Σ_q W[q,j] has_q v_q / Σ_q W[q,j] has_q（{EV} 维）；Δθ_k = MLP(e_k+1)·1[cov_k+1 > 0]",
            fontsize=6.6, color=CAP, ha="center", va="top", zorder=6)

    Y0, Y1, Y2 = 7, 21, 34          # bottom (graph) row, middle (mesh -> joints) row, top (state) row
    # 状态支路（上排）：prev 只走 FK
    node(ax, 4, Y1, 17, 10, "prev 51D\nMANO", hand_skel, TEAL, fs=8.2)
    hnode(ax, 25, Y2, 16, 10, "MANO FK\n（仅此一路）", hand_skel, ORANGE, fs=7.8, caption="778 顶点 · 投影 (u, v, z)")
    hnode(ax, 44, Y2, 26, 10, f"{Q} 个可见查询顶点\n每关节 {Q // 16} 个", grid_icon, ORANGE, fs=7.8,
          caption="按主导蒙皮权重分层 · 3 px z-测试剔除背面")
    elbow(ax, [(12.5, Y1 + 10.4), (12.5, Y2 + 5), (24.6, Y2 + 5)])
    arrow(ax, (41.4, Y2 + 5), (43.6, Y2 + 5))

    # 事件支路（下排）：直接建图
    node(ax, 30, Y0, 23, 10, "异步事件流\n(x, y, t, p)", events_icon, BLUE, fs=8.2)
    hnode(ax, 58, Y0, 21, 10, "直接建图\n因果 k-NN", graph_icon, ORANGE, fs=8.0,
          caption=f"节点 = 事件本身 4 维 · 前 {WIN} 取 k={K} · ≤{MN}")
    node(ax, 82, Y0, 17, 10, f"EdgeConv\n× {NL}", cube_icon, BLUE, fs=8.2, caption="边带 (dx, dy, dt)")
    hnode(ax, 102, Y0, 25, 10, "节点特征 h_i\n边摘要 g_i = mean(dx,dy,dt)", stack_icon, ORANGE, fs=7.4,
          caption="不做池化读出")
    arrow(ax, (53.4, Y0 + 5), (57.6, Y0 + 5))
    arrow(ax, (79.4, Y0 + 5), (81.6, Y0 + 5))
    arrow(ax, (99.4, Y0 + 5), (101.6, Y0 + 5))

    # 网格读图 → 关节
    hnode(ax, 100, Y1, 27, 10, "网格读图：固定核聚合\nv_q = [h, g, 偏移, mass]", query_icon, ORANGE, fs=7.6,
          caption=f"σ = {SIG:g} px，带 {BAND:g} px，无可学习宽度")
    hnode(ax, 130, Y1, 24, 10, "蒙皮权重 → 关节\n证据 e_j，覆盖 cov_j", funnel_icon, ORANGE, fs=7.8,
          caption="只汇看见了事件的顶点")
    hnode(ax, 157, Y1, 28, 10, "关节头 k 读 e_k+1 × 1[cov>0]\nroot 读网格聚合 + 16 证据", funnel_icon, ORANGE, fs=7.2,
          caption="没看见的关节不动 · 无 prev_mlp")
    node(ax, 188, Y1, 6, 10, "Δ", None, fs=13, caption="空包 Δ=0")

    elbow(ax, [(70.4, Y2 + 5), (113, Y2 + 5), (113, Y1 + 10.4)])    # 查询顶点 -> 网格读图
    arrow(ax, (114.5, Y0 + 10.4), (114.5, Y1 - 0.4))                 # h_i, g_i -> 网格读图（自下而上）
    arrow(ax, (127.4, Y1 + 5), (129.6, Y1 + 5))
    arrow(ax, (154.4, Y1 + 5), (156.6, Y1 + 5))
    arrow(ax, (185.4, Y1 + 5), (187.6, Y1 + 5))

    # 求和与输出
    ax.add_patch(Circle((199, Y1 + 5), 1.7, facecolor="#ffffff", edgecolor=TEAL, lw=1.4, zorder=5))
    ax.text(199, Y1 + 5, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    node(ax, 203, Y1, 27, 10, "x_k = prev + Δ", None, fs=8.6)
    arrow(ax, (194.4, Y1 + 5), (197.2, Y1 + 5))
    arrow(ax, (200.8, Y1 + 5), (202.6, Y1 + 5))

    # prev 回环（底部走廊，标签放在下排右侧无框的区域）
    elbow(ax, [(12.5, Y1 - 0.4), (12.5, 2.6), (199, 2.6), (199, Y1 + 3.2)], lw=1.5)
    ax.text(165, 3.2, "prev（本步输出即下一步的 prev；\nprev 不再经 prev_mlp 或关节头进入，只经 FK 网格）",
            ha="center", va="bottom", fontsize=6.8, color=TEAL, zorder=6)
    return fig, "s37_meshq_simple.png"


def draw_routed(mc):
    K, WIN, MN = mc["ENCODER_K"], mc["ENCODER_WINDOW"], mc["ENCODER_MAX_NODES"]
    FEAT, NL, HID = mc["ENCODER_FEAT"], mc["ENCODER_LAYERS"], mc["ENCODER_HIDDEN"]
    BAND = mc["ROUTE_BAND_PX"]
    EV_DIM = 2 * HID + 1
    ra36, ra37 = ra("s36_eventgnn"), ra("s37_routed")
    banner = "橙框 = 相对 S36 的两处改动：PREV_RENDER → false（图不再看 prev）· 全局池化 → FK 路由逐关节读出"
    if ra36 and ra37:
        banner += f"  |  两种子递推 RA {ra36:.2f} → {ra37:.2f} mm（{ra37 - ra36:+.2f}）"
    fig, ax = canvas("S37-routed（中间版本）EventGNN + FK 路由读出 — 状态无关事件图，prev 只在读出侧按 FK 几何路由到关节", banner)
    ax.text(117, 40.0,
            f"a_i = W[v(i)]·1[d_i ≤ {BAND:g} px]，v(i) = {K} 个最近投影顶点里与最近者相距 ≤3 px 的最前者（LBS 行，16 维）；"
            f"e_j = [Σ_i a_ij h_i / Σ_i a_ij ‖ max_i h_i ‖ coverage_j]（{EV_DIM} 维）",
            fontsize=6.6, color=CAP, ha="center", va="top", zorder=6)
    node(ax, 4, 17, 17, 10, "prev 51D\nMANO", hand_skel, TEAL, fs=8.2)
    node(ax, 25, 28, 16, 10, "MANO FK", hand_skel, TEAL, caption="778 顶点")
    node(ax, 44, 28, 17, 10, "仅投影\n(u, v, z)", grid_icon, TEAL, fs=8.2, caption="无 z-buffer · 无渲染")
    hnode(ax, 64, 28, 27, 10, "路由 a_i：前表面\n最近顶点的 LBS 行", query_icon, ORANGE, fs=8.0,
          caption=f"d ≤ {BAND:g} px · ≤{MN} 采样节点 · no_grad")
    elbow(ax, [(12.5, 27.4), (12.5, 33), (24.6, 33)])
    arrow(ax, (41.4, 33), (43.6, 33))
    arrow(ax, (61.4, 33), (63.6, 33))
    node(ax, 56, 5, 23, 10, "异步事件流\n(x, y, t, p)", events_icon, BLUE, fs=8.2)
    hnode(ax, 93, 17, 19, 10, "逐事件\ntoken 7 维", stack_icon, ORANGE, fs=8.2, caption="不再拼渲染通道（S36 为 9 维）")
    node(ax, 115, 17, 19, 10, "因果 k-NN\n建图", graph_icon, BLUE, fs=8.2, caption=f"前 {WIN} 里取 k={K} · ≤{MN} 点")
    node(ax, 137, 17, 16, 10, f"EdgeConv\n× {NL}", cube_icon, BLUE, fs=8.2, caption="边带 (dx, dy, dt)")
    hnode(ax, 156, 17, 22, 10, "逐关节证据\ne_j (16 × %d)" % EV_DIM, stack_icon, ORANGE, fs=8.0,
          caption="节点特征 h_i 按 a_i 加权池化到关节")
    hnode(ax, 181, 17, 23, 10, "15 指头各读 e_k+1\nroot 读池化+16 证据", funnel_icon, ORANGE, fs=7.6,
          caption=f"mean‖max → {FEAT} 只进 root")
    node(ax, 207, 17, 6, 10, "Δ", None, fs=13, caption="空包 Δ=0")
    elbow(ax, [(79.4, 10), (98, 10), (98, 16.6)])
    arrow(ax, (112.4, 22), (114.6, 22))
    arrow(ax, (134.4, 22), (136.6, 22))
    arrow(ax, (153.4, 22), (155.6, 22))
    arrow(ax, (178.4, 22), (180.6, 22))
    arrow(ax, (204.4, 22), (206.6, 22))
    elbow(ax, [(91.4, 33), (167, 33), (167, 27.4)])
    elbow(ax, [(79.4, 12), (88, 12), (88, 27.4)], lw=1.0)
    ax.text(84.5, 20.5, "事件像素\n(u, v)", fontsize=6.2, color=CAP, ha="center", va="center", zorder=6)
    ax.add_patch(Circle((217, 22), 1.7, facecolor="#ffffff", edgecolor=TEAL, lw=1.4, zorder=5))
    ax.text(217, 22, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    node(ax, 220, 17, 12, 10, "x_k =\nprev + Δ", None, fs=8.2)
    arrow(ax, (213.4, 22), (215.2, 22))
    arrow(ax, (218.8, 22), (219.6, 22))
    elbow(ax, [(12.5, 16.6), (12.5, 3.2), (217, 3.2), (217, 20.2)], lw=1.5)
    ax.text(140, 3.9, "prev（prev_mlp 支路一并加入 Δ；关节头各读自己的 prev 角；本步输出即下一步的 prev）",
            ha="center", va="bottom", fontsize=6.8, color=TEAL, zorder=6)
    return fig, "s37_routed_simple.png"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("meshq", "routed"), default="meshq")
    a = ap.parse_args()
    cfg = yaml.safe_load((REPO / f"configs/semkine/s37_{a.arm}_s3407.yaml").read_text())
    fig, name = (draw_meshq if a.arm == "meshq" else draw_routed)(cfg["MODEL"])
    out = REPO / "docs/assets" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
