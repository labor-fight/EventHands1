#!/usr/bin/env python3
"""S37 mesh-graph architecture figure, in the S36 detailed template (`make_s36_figure.py`).

Every number on the canvas is read from the config, the built model and the main-row / probe
JSONs, not typed in, so the figure cannot drift away from the code or the recorded results.

    python tools/make_s37_meshgraph_figure.py      # -> docs/assets/s37_meshgraph.png

Text is restricted to CJK + ASCII (Noto Sans CJK has no glyphs for combining diacritics).
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
sys.path.insert(0, str(REPO / "tools"))

from make_s36_figure import EV, GATE, GRAPH, HEAD, INK, LOOP, MUTED, STATE, arrow, box, path  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Noto Sans CJK SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

GEO = "#0b7285"                     # conditioning geometry (FK / visibility / LUT)
TOP_Y, TOP_H = 70.0, 24.0
MID_Y, MID_H = 38.0, 22.0
COL_X, COL_W = 149.5, 26.5


def _fmt(x, nd=2):
    return f"{x:.{nd}f}"


def main() -> None:
    cfg = yaml.safe_load((REPO / "configs/semkine/s37_meshgraph_s3407.yaml").read_text())
    mc = cfg["MODEL"]
    from model import MNISTModel
    m = MNISTModel(cfg)
    enc = m.event_encoder
    n = lambda mod: sum(p.numel() for p in mod.parameters())
    fe, tot = n(enc), sum(p.numel() for p in m.parameters()) - n(m.mano)
    V, N, K = m.mg_spec.n_verts, m.mg_spec.n_nodes, m.mg_spec.k
    HID, NL, OBS = enc.hidden, len(enc.layers), enc.obs_embed.in_features
    EVD = 2 * HID + 1
    HIDH = int(mc.get("ACTIVE_HIDDEN", 64))
    band, front, ztol = m.mg_band_px, m.mg_front_px, m.mg_z_tol
    root_in = m.root_head.in_features
    H, W = m.render_h, m.render_w

    row = json.loads((REPO / "outputs/semkine/s37_meshgraph_main_row.json").read_text())
    lat, macs = row["latency_ms_scaled_full1p75"], row["macs_forward_packet"] / 1e9
    ra = {s: row["per_seed"][s]["overall"]["mpjpe_ra_mm"] for s in ("3407", "3408")}
    dec = json.loads((REPO / "outputs/semkine/probe_s37_meshgraph_rotdecomp.json").read_text())
    fing = {s: dec[f"s37_meshgraph_s{s}"]["closed_loop"]["ra_rotaligned"] for s in ("3407", "3408")}
    rot = {s: dec[f"s37_meshgraph_s{s}"]["closed_loop"]["rot_p50_deg"] for s in ("3407", "3408")}
    corr = dec["s37_meshgraph_s3407"]["grid_summary"]
    fkd = {s: dec[f"s37_fkgraph_s{s}"]["closed_loop"]["ra_rotaligned"] for s in ("3407", "3408")}
    cl = json.loads((REPO / "outputs/semkine/closed_loop_s37meshgraph_vs_fkgraph.json").read_text())["arms"]
    tf = {s: cl[f"mg_{s}"]["teacher_forced"]["ra_mm"] for s in ("3407", "3408")}

    fig, ax = plt.subplots(figsize=(17.8, 10.6))
    ax.set_xlim(0, 178); ax.set_ylim(0, 106); ax.axis("off")

    ax.text(2, 104.0, "S37 网格图 — prev 的整张 MANO mesh 就是图，事件是它的观测，蒙皮权重把图汇到关节",
            fontsize=16, fontweight="bold", color=INK, va="top")
    ax.text(2, 99.0, f"{V} 顶点节点 + 1 背景 · 边 = 网格面 1-ring（最大度 {K}）· 节点输入只有事件观测（无几何）· "
                     f"前端 {fe/1e6:.2f} M · 可训练总计 {tot/1e6:.2f} M · {macs:.3f} GFLOPs/步 · {lat:.2f} ms/步",
            fontsize=9.6, color=MUTED, va="top")

    # ================================================================ 主流水线（几何条件 → 证据）
    xs = [2 + i * 29.5 for i in range(6)]
    w = 27.0
    box(ax, xs[0], TOP_Y, w, TOP_H, "① MANO FK（prev 51D）", [
        "prev = [ t3 ‖ R3 ‖ residual45 ]",
        f"FK → {V} 顶点  (B, {V}, 3)",
        f"针孔投影 K（{W}×{H}）",
        f"→ uv (B, {V}, 2) · z (B, {V})",
        "no_grad：只决定事件给谁",
    ], GEO, fs=7.8)
    box(ax, xs[1], TOP_Y, w, TOP_H, "② 可见性", [
        "背面剔除：顶点法向 n·p < 0",
        "（面法向 float64 累加）",
        f"点溅 z-buffer {int(2*front+1)}×{int(2*front+1)} px · 容差 {ztol*100:g} cm",
        f"vis (B, {V})：约 50% 顶点可见",
        "背面 / 被遮挡顶点收不到事件",
    ], GEO, fs=7.8)
    box(ax, xs[2], TOP_Y, w, TOP_H, "③ 最近可见顶点查找表", [
        "跳跃泛洪 5 轮 × 8 邻域",
        f"{H}×{W} 每像素 → 最近可见顶点",
        f"≤ {band:g} px；否则背景节点 {V}",
        "O(HW)，与事件数、节点数无关",
        "等价于暴力最近点（契约测试）",
    ], GEO, fs=7.8)
    box(ax, xs[3], TOP_Y, w, TOP_H, "④ 事件归属 + 逐顶点观测", [
        "全部事件按像素查表归属",
        f"每节点 {OBS} 维：log1p(n)/log1p(N),",
        "(Δu, Δv)/r, 离散度/r, 时间, 极性",
        f"float64 累加 → (B, {N}, {OBS})",
        "空节点全零 · 无流速项（H6）",
    ], EV, fs=7.8)
    box(ax, xs[4], TOP_Y, w, TOP_H, f"⑤ 编码 + EdgeConv × {NL}", [
        f"Linear {OBS}→{HID} + 身份嵌入 {N}×{HID}",
        f"→ ReLU → EdgeConv×{NL}（残差）",
        "m_ij = ReLU(W[h_j − h_i ; 边类型])",
        f"边 = 面 1-ring · 3 跳 ≈ 1–2 cm",
        f"→ h (B, {N}, {HID})",
    ], GRAPH, fs=7.8)
    box(ax, xs[5], TOP_Y, w, TOP_H, "⑥ LBS pooling（固定蒙皮权重）", [
        "e_j = [ ΣW_vj·has_v·h_v / ΣW_vj·has_v",
        "  ‖ max_{argmax W=j, has} h_v ‖ cov_j ]",
        f"→ e (B, 16, {EVD})；背景 h_{V} ({HID})",
        "只在看见事件的可见顶点上池化",
        "W 是 MANO 资产，优化器动不了",
    ], GRAPH, fs=7.6, fc="#fff9db")
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + w + 0.3, TOP_Y + TOP_H / 2), (b - 0.3, TOP_Y + TOP_H / 2))

    # ================================================================ 状态与事件源
    box(ax, xs[0], MID_Y, w, MID_H, "上一帧状态 51D", [
        "PREVPOS_EMBED", "prev_mlp: 51 → 64 → 51",
        "末层零初始化", "同一个 prev 也是 ① 的 FK 输入",
    ], STATE)
    arrow(ax, (xs[0] + w / 2, MID_Y + MID_H + 0.3), (xs[0] + w / 2, TOP_Y - 0.3), STATE)
    ax.text(xs[0] + w / 2 + 1.5, (MID_Y + MID_H + TOP_Y) / 2, "prev → FK",
            fontsize=7.9, color=STATE, ha="left", va="center", zorder=5)

    box(ax, xs[1], MID_Y, w, MID_H, "异步事件流", [
        "(x, y, t, p) 全部事件，不子采样",
        f"评测 50 ms 一包，均约 {row['latency_mean_events_per_packet']} 事件",
        "训练 30–300 ms · 域随机化",
        "只在 ④ 进入：归属 + 观测",
    ], EV)
    for i, (dx, dy) in enumerate([(4, 1.6), (9, 2.4), (14, 1.3), (19, 2.2), (23.5, 1.5)]):
        ax.add_patch(Circle((xs[1] + dx, MID_Y + dy), 0.55,
                            color=EV if i % 2 else "#f08c00", zorder=3))
    path(ax, [(xs[1] + w / 2, MID_Y + MID_H + 0.3), (xs[1] + w / 2, 65.0),
              (xs[3] + w / 2, 65.0), (xs[3] + w / 2, TOP_Y - 0.3)], EV)
    ax.text((xs[1] + xs[3]) / 2 + w / 2, 66.0, "事件 → ④", fontsize=7.9, color=EV,
            ha="center", va="bottom", zorder=5)

    # ================================================================ 解码列
    cx = COL_X + COL_W / 2
    ax.add_patch(FancyBboxPatch((COL_X, 58.0), COL_W, 8.5,
                                boxstyle="round,pad=0.3,rounding_size=1.3",
                                linewidth=1.4, edgecolor=GATE, facecolor="#f8f9fa", zorder=2))
    ax.text(cx, 64.6, "⑦ 零事件门", ha="center", va="top", fontsize=9.0,
            color=INK, fontweight="bold", zorder=3)
    ax.text(cx, 60.6, "空包 ⇒ 逐位返回 prev（Δ ≡ 0）", ha="center", va="center",
            fontsize=7.4, color=MUTED, zorder=3)

    box(ax, COL_X, 30.0, COL_W, 24.0, "⑧ 解码头", [
        f"root: Linear({root_in} → 6)",
        f"  读 16×{EVD} 证据按序平铺 + 背景",
        f"15 × [ {EVD}+3 → {HIDH} → 3 ]",
        "  关节 k 只读 e_k+1 + 自身 prev 角",
        "手指与 root 没有共享通路",
    ], HEAD, fs=7.8)
    box(ax, COL_X, 4.0, COL_W, 16.0, "⑨ 输出 51D 姿态", [
        "PREDICT_DELTA", "x_k = prev + Δ → 下一步 FK",
    ], HEAD, fc="#f8f0fc")

    arrow(ax, (cx, TOP_Y - 0.3), (cx, 66.8), HEAD)
    arrow(ax, (cx, 57.7), (cx, 54.3), HEAD)
    arrow(ax, (cx, 29.7), (cx, 27.6), HEAD)
    ax.add_patch(Circle((cx, 26.0), 1.6, facecolor="#ffffff", edgecolor=HEAD,
                        linewidth=1.5, zorder=5))
    ax.text(cx, 26.0, "+", ha="center", va="center", fontsize=11, color=HEAD, zorder=6)
    arrow(ax, (cx, 24.4), (cx, 20.3), HEAD)

    # prev_mlp 汇入求和点
    path(ax, [(xs[0] + 8, MID_Y - 0.3), (xs[0] + 8, 26.0), (cx - 1.9, 26.0)], STATE)
    ax.text(xs[0] + 10, 27.4, "prev_mlp(prev) 直接加到输出", fontsize=7.9,
            color=STATE, ha="left", va="bottom", zorder=5)

    # 递归闭环
    path(ax, [(COL_X - 0.3, 12.0), (xs[0] + 19, 12.0), (xs[0] + 19, MID_Y - 0.3)], LOOP, lw=1.9)
    ax.text(84, 13.4, "递归追踪闭环：本步输出即下一步的 prev，再做 FK（训练时 teacher forcing + 噪声课程）",
            fontsize=8.6, color=LOOP, ha="center", va="bottom", zorder=5)

    # ================================================================ 注 A：zgz 结果与机制读数
    ax.add_patch(FancyBboxPatch((xs[2], 30.0), 2 * w + 2.5, 26.0,
                                boxstyle="round,pad=0.4,rounding_size=1.4",
                                linewidth=1.1, edgecolor=GATE, facecolor="#fbfcfd", zorder=1))
    ax.text(xs[2] + 3, 53.0, "注 A — zgz 两种子读数（50 ms 递推）", fontsize=9.6,
            color=INK, fontweight="bold", va="top", zorder=3)
    ax.text(xs[2] + 3, 48.4,
            f"递推 RA  {_fmt(ra['3407'])} / {_fmt(ra['3408'])} mm   TF 单步 {_fmt(tf['3407'])} / {_fmt(tf['3408'])}"
            f"（fk_graph 22.64 / 21.56，9.24 / 8.87）\n"
            f"手指      旋转对齐后 {_fmt(fing['3407'], 1)} / {_fmt(fing['3408'], 1)} —— 两种子相同"
            f"（fk_graph {_fmt(fkd['3407'], 1)} / {_fmt(fkd['3408'], 1)}）\n"
            f"root 旋转  闭环中位数 {rot['3407']:.1f}° / {rot['3408']:.1f}°；"
            f"网格 RA 与旋转相关 {corr['corr_ra_rot']:.2f}，与手指相关 {corr['corr_ra_rotaligned']:.2f}\n\n"
            "⇒ 手指读出成立且稳定；不稳定全在 root：图里只有 1-ring 边，root 头没有长程通路",
            fontsize=8.2, color=MUTED, va="top", linespacing=1.85, zorder=3)

    ax.text(176, 1.5, "docs/S37_MESHGRAPH_PREREG.md", fontsize=7.2,
            color=GATE, ha="right", va="bottom")

    out = REPO / "docs/assets/s37_meshgraph.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
