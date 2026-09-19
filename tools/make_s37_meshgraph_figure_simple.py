#!/usr/bin/env python3
"""S37 mesh graph, drawn in the S36 flow-strip template with the changes against `fk_graph` boxed in orange.

Processing modules only: FK, visibility + nearest-visible-vertex lookup, the mesh graph, event
assignment + per-vertex summary, embedding + EdgeConv, LBS pooling, decoders, delta, sum.
Numbers on the strip come from the config and the main-row JSONs.

    python tools/make_s37_meshgraph_figure_simple.py     # -> docs/assets/s37_meshgraph_simple.png
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

from make_s36_figure_simple import (BLUE, CAP, PANEL, TEAL, arrow, cube_icon, elbow,   # noqa: E402
                                    events_icon, funnel_icon, graph_icon, grid_icon, hand_skel,
                                    node)
from make_s37_fkgraph_figure import ORANGE, bins_icon, hnode                                # noqa: E402


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs/semkine/s37_meshgraph_s3407.yaml").read_text())
    mc = cfg["MODEL"]
    assert mc["ENCODER"] == "mesh_graph" and not mc["PREV_RENDER"]
    BAND, ZTOL = mc["MESH_GRAPH_BAND_PX"], mc["MESH_GRAPH_Z_TOL"]
    NL, HID = mc["ENCODER_LAYERS"], mc["ENCODER_HIDDEN"]
    OBS = 8 if mc.get("MESH_GRAPH_OBS_FLOW", False) else 6

    def row(name):
        p = REPO / f"outputs/semkine/{name}_main_row.json"
        return json.loads(p.read_text()) if p.exists() else None

    r_fk, r_mg = row("s37_fkgraph"), row("s37_meshgraph")

    fig, ax = plt.subplots(figsize=(17.0, 3.6))
    ax.set_xlim(0, 214); ax.set_ylim(0, 45); ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 1), 212, 43, boxstyle="round,pad=0.3,rounding_size=2.5",
                                facecolor=PANEL, edgecolor="#e5e7eb", lw=1.0, zorder=0))
    ax.text(4, 42.3, "S37 网格图 — prev 的整张 MANO mesh 就是图，事件是它的观测，蒙皮权重把图汇到关节",
            fontsize=8.2, color=CAP, va="top", zorder=6)
    banner = "橙框 = 相对 FK 图的改动：778 顶点全作节点 · 面 1-ring 边 · 可见性 + 查表归属 · LBS pooling 取代关节节点"
    if r_fk and r_mg:
        a = r_fk["two_seed_mean"]["overall"]["mpjpe_ra_mm"]
        b = r_mg["two_seed_mean"]["overall"]["mpjpe_ra_mm"]
        s = " / ".join(f"{r_mg['per_seed'][k]['overall']['mpjpe_ra_mm']:.2f}" for k in sorted(r_mg["per_seed"]))
        banner += f"  |  两种子递推 RA {a:.2f} → {b:.2f} mm（{s}）"
    ax.text(210, 42.3, banner, fontsize=7.0, color=ORANGE, ha="right", va="top", zorder=6)

    # 状态支路（上排）
    node(ax, 4, 17, 17, 10, "prev 51D\nMANO", hand_skel, TEAL, fs=8.2)
    node(ax, 25, 30, 16, 10, "MANO FK", hand_skel, TEAL, caption="778 顶点 · 投影 uv, z")
    hnode(ax, 44, 30, 31, 10, "可见性 + 最近可见顶点查表\n背面剔除 · z-buffer · 跳跃泛洪", grid_icon,
          ORANGE, fs=7.6, caption=f"≤ {BAND:g} px 否则背景 · 容差 {ZTOL*100:g} cm · 约 50% 顶点可见 · O(HW)")
    hnode(ax, 108, 30, 30, 10, "网格图：778 顶点 + 背景\n边 = 网格面 1-ring（最大度 8）", graph_icon,
          ORANGE, fs=7.6)
    elbow(ax, [(12.5, 27.4), (12.5, 35), (24.6, 35)])
    arrow(ax, (41.4, 35), (43.6, 35))

    # 事件支路（下排）
    node(ax, 56, 6, 25, 10, "异步事件流\n(x, y, t, p) 全部事件", events_icon, BLUE, fs=8.0)

    # 主干
    hnode(ax, 91, 17, 24, 10, f"按像素查表归属 + 汇总\n每顶点 {OBS} 维观测", bins_icon, ORANGE, fs=7.6,
          caption="数量 / 偏移 / 离散 / 时间 / 极性 · 无流速")
    node(ax, 118, 17, 20, 10, f"嵌入 + EdgeConv\n× {NL}", cube_icon, TEAL, fs=8.0,
         caption=f"身份嵌入 → {HID} · 在网格图上")
    hnode(ax, 141, 17, 21, 10, "LBS pooling\n蒙皮权重 → 16 关节", funnel_icon, ORANGE, fs=7.8,
          caption=f"[mean ‖ max ‖ cov] → {2*HID+1} 维")
    node(ax, 165, 17, 21, 10, "15 指头各读 e_k\nroot 读 16 关节+背景", hand_skel, TEAL, fs=7.4,
         caption="+ 自身 prev 角")
    node(ax, 189, 17, 5, 10, "Δ", None, fs=13, caption="空包 Δ=0")

    elbow(ax, [(75.4, 35), (103, 35), (103, 27.4)])          # 查找表（每像素 -> 顶点）-> 归属
    arrow(ax, (123, 29.6), (123, 27.4))                       # 网格图 -> EdgeConv
    elbow(ax, [(81.4, 11), (93.5, 11), (93.5, 16.6)])        # 事件 -> 归属
    arrow(ax, (115.4, 22), (117.6, 22))
    arrow(ax, (138.4, 22), (140.6, 22))
    arrow(ax, (162.4, 22), (164.6, 22))
    arrow(ax, (186.4, 22), (188.6, 22))

    # 求和与输出
    ax.add_patch(Circle((197.5, 22), 1.7, facecolor="#ffffff", edgecolor=TEAL, lw=1.4, zorder=5))
    ax.text(197.5, 22, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    node(ax, 200.5, 17, 11, 10, "x_k =\nprev + Δ", None, fs=8.0)
    arrow(ax, (194.4, 22), (195.7, 22))
    arrow(ax, (199.3, 22), (200.1, 22))

    # prev 回环（底部走廊）
    elbow(ax, [(12.5, 16.6), (12.5, 3.6), (197.5, 3.6), (197.5, 20.2)], lw=1.5)
    ax.text(125, 4.4, "prev（prev_mlp 支路一并加入 Δ；本步输出即下一步的 prev，再做 FK）",
            ha="center", va="bottom", fontsize=6.8, color=TEAL, zorder=6)

    out = REPO / "docs/assets/s37_meshgraph_simple.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
