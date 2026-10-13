#!/usr/bin/env python3
"""S39 (dense LNES + render tracker and its inference stack) in the S36 flow-strip template, changes against the
EventHands-Track row of the main table (`track_render51_dr_sem`: render + SO3/FK + LBS + DomRand) boxed in orange.

    python tools/make_s39_figure_simple.py                   # S39: W200 + adaptive filter + evidence gate ramp1000
    python tools/make_s39_figure_simple.py --arm w200filt    # the same stack without the evidence gate

Outputs docs/assets/s39_dense_simple.png / s39_w200filt_simple.png. The network comes from configs/dt/dt_dz_l3.yaml,
the filter gains from the DT2 spec, cost and RA from the main-row JSONs (RA only for an arm that has one). W, the count
rule and the ramp are evaluation arguments, not config keys (docs/EXPERIMENT_RECORDS.md#dt3-verdict).
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

from make_s36_figure_simple import (BLUE, CAP, PANEL, TEAL, arrow, cube_icon, elbow, events_icon,   # noqa: E402
                                    funnel_icon, grid_icon, hand_fill, hand_skel, node, stack_icon)
from make_s37_figure_simple import ORANGE, hnode   # noqa: E402

WINDOW_MS = 200         # evidence window, clipped to the segment start (evalx --window-ms 200 --clip-run-start)
RAMP_N = 1000           # evidence gate ramp1000: every gain times min(1, n / RAMP_N)
SPEC = REPO / "outputs/dt2/filter_sweep/recommended_spec.json"
S39_ROW = REPO / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json"
# the old row's cost: the sibling EventHands repo's main table (its semkine main row went in the 2026-10-07 cleanup)
OLD_TABLE = REPO.parent / "EventHands/outputs/hand_data51/main_table.json"
OLD_PROV = "dr_sem_rep2"


def smooth_icon(ax, cx, cy, s, c):
    """A noisy track (grey) and its filtered version (colour)."""
    xs = [cx + (i / 8 - 0.5) * 2.0 * s for i in range(9)]
    noisy = [0.0, 0.75, -0.25, 0.9, 0.05, 1.1, 0.3, 1.25, 0.7]
    smooth = [0.15, 0.25, 0.32, 0.42, 0.5, 0.6, 0.68, 0.78, 0.85]
    ax.plot(xs, [cy - 0.55 * s + v * 0.9 * s for v in noisy], color="#adb5bd", lw=0.8, zorder=4)
    ax.plot(xs, [cy - 0.55 * s + v * 0.9 * s for v in smooth], color=c, lw=1.5, zorder=5)


def canvas(title, banner, note, width=240, height=48):
    fig, ax = plt.subplots(figsize=(width / 12.6, height / 12.4))
    ax.set_xlim(0, width); ax.set_ylim(0, height); ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 1), width - 2, height - 2, boxstyle="round,pad=0.3,rounding_size=2.5",
                                facecolor=PANEL, edgecolor="#e5e7eb", lw=1.0, zorder=0))
    ax.text(4, height - 2.7, title, fontsize=8.2, color=CAP, va="top", zorder=6)
    ax.text(width - 4, height - 2.7, banner, fontsize=7.2, color=ORANGE, ha="right", va="top", zorder=6)
    ax.text(4, height - 6.4, note, fontsize=6.6, color=CAP, va="top", zorder=6)
    return fig, ax


def gain_lines(spec, gate):
    nodes = spec["a_root"]
    ns = " / ".join(f"{n / 1000:g}k" if n >= 1000 else f"{n:g}" for n, _ in nodes)
    gs = " / ".join(f"{g:g}" for _, g in nodes)
    lines = ["增益 a(n)（按 log10 n 插值）", f"根 {gs} @ n = {ns}", f"手指 {spec['a_rest']:g} · 平移 {spec['a_trans']:g}"]
    if gate:
        lines.append(f"证据门：再 × min(1, n / {RAMP_N})")
    return "\n".join(lines)


def draw(arm):
    cfg = yaml.safe_load((REPO / "configs/dt/dt_dz_l3.yaml").read_text())
    mc, lc, tc = cfg["MODEL"], cfg["LOSS"], cfg["TRAIN"]
    rs, RH, RW = mc["RENDER_SCALE"], mc["RENDER_H"], mc["RENDER_W"]
    chans = " + ".join(mc.get("RENDER_CHANNELS", ["sil", "inv"]))
    spec = json.loads(SPEC.read_text())
    s39 = json.loads(S39_ROW.read_text())
    old = next(r for r in json.loads(OLD_TABLE.read_text())["rows"] if OLD_PROV in r["prov"])
    cost = (f"{s39['params_total'] / 1e6:.2f} M · {s39['macs_forward_packet'] / 1e9:.3f} G MAC/步"
            f"（旧 {old['params'] / 1e6:.2f} M · {old['macs'] / 1e9:.3f} G）")
    gate = arm == "s39"
    if gate:
        m = s39["two_seed_mean"]
        title = f"S39 — LNES + 渲染前态 CNN 递归跟踪（dt_dz_l3 + W={WINDOW_MS} + 自适应滤波 + 证据门 ramp{RAMP_N}）"
        result = (f"{s39['n_seeds']} 种子 RA {m['overall']['mpjpe_ra_mm']:.2f} mm"
                  f"（local {m['local']['mpjpe_ra_mm']:.2f} / global {m['global']['mpjpe_ra_mm']:.2f}）· ")
        name = "s39_dense_simple.png"
    else:
        title = f"dt_dz_l3 + W={WINDOW_MS} + 自适应滤波（S39 去掉证据门）— LNES + 渲染前态 CNN 递归跟踪"
        result = "无主行，不报精度 · 网络同 S39："
        name = "s39_w200filt_simple.png"
    banner = f"橙框 = 相对 EventHands-Track 行（dr_sem）的改动  |  {result}{cost}"
    note = (f"训练侧（图中不画）：损失 SO3 + 平移 + FK（{lc['ROT_WEIGHT']:g} / {lc['TRANS_WEIGHT']:g} / "
            f"{lc['FK_WEIGHT']:g}，不变）· DomRand 尺度改为 {cfg['AUG']['DOMRAND']['SCALE_MODE']} 语义（旧 focal）· "
            f"统一配方：随机初始化、有效 batch {tc['BATCH_SIZE_PER_GPU'] * tc['ACCUMULATE_GRAD_BATCHES']}、"
            f"{tc['LR_SCHEDULE']}、{tc['MAX_STEPS']} 步、取最后一步（旧：热启动、batch 2048、常数学习率、3000 步、zgz 选点）")
    fig, ax = canvas(title, banner, note)

    # 状态支路（上排）
    node(ax, 4, 17, 17, 10, "prev 51D\nMANO", hand_skel, TEAL, fs=8.2)
    node(ax, 25, 28.5, 16, 10, "MANO FK", hand_skel, TEAL, caption="序列 GT β · LBS")
    node(ax, 44, 28.5, 18, 10, "投影\n+ z-buffer", grid_icon, TEAL, fs=8.2, caption=f"K × {rs:g} → {RH} × {RW}")
    hnode(ax, 65, 28.5, 22, 10, "silhouette /\n逆深度图", hand_fill, "#1e3a5f", fs=8.2,
          caption=f"{chans}（旧行为语义轮廓 semsil）")
    elbow(ax, [(12.5, 27.4), (12.5, 33.5), (24.6, 33.5)])
    arrow(ax, (41.4, 33.5), (43.6, 33.5))
    arrow(ax, (62.4, 33.5), (64.6, 33.5))

    # 事件支路（下排）
    node(ax, 26, 5, 23, 10, "异步事件流\n(x, y, t, p)", events_icon, BLUE, fs=8.2)
    hnode(ax, 53, 5, 27, 10, f"LNES 2 通道\nW = {WINDOW_MS} ms", grid_icon, BLUE, fs=8.2)
    ax.text(66.5, 16.0, "每 50 ms 出一步 · 窗口裁到段首（旧 W = 50）", ha="center", va="bottom",
            fontsize=6.4, color=CAP, zorder=6)
    arrow(ax, (49.4, 10), (52.6, 10))

    # 主干
    node(ax, 89, 17, 22, 10, "拼 4 通道\n+ conv1 4→3", stack_icon, TEAL, fs=8.2,
         caption=f"LNES 2 + 渲染 2 · {RH} × {RW}")
    hnode(ax, 114, 17, 25, 10, "ResNet18-L3\n去掉 layer4", cube_icon, ORANGE, fs=8.2,
          caption="stem 64 · 64 · 128 · 256（旧 + 512）")
    node(ax, 142, 17, 18, 10, "GAP +\nfc 256→51", funnel_icon, TEAL, fs=8.2)
    node(ax, 163, 17, 7, 10, "Δ", None, fs=13, caption="空包 Δ=0")
    elbow(ax, [(87.4, 33.5), (103, 33.5), (103, 27.4)])     # 渲染图 -> 拼接
    elbow(ax, [(80.4, 10), (85, 10), (85, 22), (88.6, 22)])  # LNES -> 拼接（从左侧进，让开下方说明）
    arrow(ax, (111.4, 22), (113.6, 22))
    arrow(ax, (139.4, 22), (141.6, 22))
    arrow(ax, (160.4, 22), (162.6, 22))

    # 求和 z = prev + Δ，再经自适应滤波
    ax.add_patch(Circle((174.5, 22), 1.7, facecolor="#ffffff", edgecolor=TEAL, lw=1.4, zorder=5))
    ax.text(174.5, 22, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    arrow(ax, (170.4, 22), (172.8, 22))
    arrow(ax, (176.2, 22), (179.6, 22))
    ax.text(177.9, 22.6, "z", ha="center", va="bottom", fontsize=7.4, color=TEAL, zorder=6)
    hnode(ax, 180, 17, 30, 10, "自适应滤波\nx = prev ⊕ a(n)·(z ⊖ prev)\n根 SO(3) 测地 · 指/平移 线性",
          smooth_icon, ORANGE, fs=7.0)
    node(ax, 214, 17, 22, 10, "x_k\n滤波后状态", hand_skel, TEAL, fs=8.2)
    arrow(ax, (210.4, 22), (213.6, 22))

    # 增益：本包最新 50 ms 的事件数 n 查表
    hnode(ax, 180, 4, 30, 9.5, gain_lines(spec, gate), None, fs=6.6)
    hnode(ax, 214, 4, 22, 9.5, "n：最新 50 ms\n原始事件数", events_icon, BLUE, fs=7.6)
    arrow(ax, (213.6, 8.75), (210.4, 8.75))
    arrow(ax, (195, 13.9), (195, 16.6))

    # prev 回环（底部走廊）
    elbow(ax, [(12.5, 16.6), (12.5, 3.2), (174.5, 3.2), (174.5, 20.2)], lw=1.5)
    ax.text(128, 4.0, "prev（prev_mlp 51→64→51 支路一并加入 Δ；滤波后的 x_k 即下一步的 prev，渲染、prev_mlp、滤波都以它为准）",
            ha="center", va="bottom", fontsize=6.8, color=TEAL, zorder=6)
    return fig, name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("s39", "w200filt"), default="s39")
    a = ap.parse_args()
    fig, name = draw(a.arm)
    out = REPO / "docs/assets" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
