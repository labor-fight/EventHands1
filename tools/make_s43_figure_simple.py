#!/usr/bin/env python3
"""S43 (sparse render-and-compare tracker, `s43n_sp`) in the S39 flow-strip template, under the W50 raw protocol of the
main table ("S43: sparse render-and-compare event tracking (W50 raw protocol, same as S37)"). Changes against the S39
dense network (`configs/dt/dt_dz_l3.yaml`) boxed in orange.

    python tools/make_s43_figure_simple.py

Outputs docs/assets/s43_sparse_simple.png. The network comes from configs/s43/s43n_sp.yaml and model/sparse_trunk.py
(`sparse_resnet18_l3` = SparseResNet defaults: widths 64 / 128 / 256, CoordConv, mean_area readout), RA and cost from the
W50 raw main row (`evalx eval --controls --tf --perturb`, no filter), the active-set size from the W50 MAC report
(docs/EXPERIMENT_RECORDS.md#s43-prereg).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import yaml
from matplotlib.patches import Circle

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from make_s36_figure_simple import (BLUE, CAP, TEAL, arrow, cube_icon, elbow, events_icon, funnel_icon,   # noqa: E402
                                    grid_icon, hand_fill, hand_skel, node, stack_icon)
from make_s37_figure_simple import ORANGE, hnode   # noqa: E402
from make_s39_figure_simple import canvas   # noqa: E402

CFG = REPO / "configs/s43/s43n_sp.yaml"
S43_ROW = REPO / "outputs/semkine/s43n_sp_tf_pert_main_row.json"        # W50 raw protocol (eval_variant tf_pert)
S39_ROW = REPO / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json"      # S39 dense cost
MACS_W50 = REPO / "outputs/s43/reports/macs_w50_s43n_sp_s3407.json"     # active-set size A0 on the zgz W50 steps
WIDTHS = (64, 128, 256)  # SparseResNet defaults (model/sparse_trunk.py), the `sparse_resnet18_l3` entry of backbones.py


def half(n):
    """Grid size of a stride-2 level (ceil, as `cell_mask` and every stride-2 layer of the trunk)."""
    return -(-n // 2)


def draw():
    cfg = yaml.safe_load(CFG.read_text())
    mc, lc, tc, dr = cfg["MODEL"], cfg["LOSS"], cfg["TRAIN"], cfg["AUG"]["DOMRAND"]
    rs, RH, RW = mc["RENDER_SCALE"], mc["RENDER_H"], mc["RENDER_W"]
    win = cfg["EVAL"]["WINDOW_MS"]
    chans = " + ".join(mc.get("RENDER_CHANNELS", ["sil", "inv"]))
    h4, w4 = RH, RW
    for _ in range(4):
        h4, w4 = half(h4), half(w4)
    fc_in = 2 * WIDTHS[-1]                                                  # mean_area: mean || sum / (H4 W4)
    s43 = json.loads(S43_ROW.read_text())
    s39 = json.loads(S39_ROW.read_text())
    a0 = json.loads(MACS_W50.read_text())["A0_median"]
    m = s43["two_seed_mean"]                                                # the N-seed mean (key name kept)
    cost = (f"{s43['params_total'] / 1e6:.2f} M · {s43['macs_forward_packet'] / 1e9:.3f} G MAC/步"
            f"（S39 {s39['params_total'] / 1e6:.2f} M · {s39['macs_forward_packet'] / 1e9:.3f} G）")
    title = f"S43 — 稀疏渲染–比对事件跟踪（s43n_sp · W{win} 原始协议，同 S37 口径）"
    banner = (f"橙框 = 相对 S39 稠密网络（dt_dz_l3）的改动  |  {s43['n_seeds']} 种子 RA {m['overall']['mpjpe_ra_mm']:.2f} mm"
              f"（local {m['local']['mpjpe_ra_mm']:.2f} / global {m['global']['mpjpe_ra_mm']:.2f}）· {cost}")
    note = (f"训练侧（图中不画）：损失 SO3 + 平移 + FK（{lc['ROT_WEIGHT']:g} / {lc['TRANS_WEIGHT']:g} / {lc['FK_WEIGHT']:g}，同 S39）· "
            f"DomRand 尺度 {dr['SCALE_MODE']} · 热像素注入 {dr['HOT_PIXEL_RATE']:g}（S39 有；稀疏活跃集必需，偏离 3）· "
            f"窗口 {cfg['DATA']['WINDOW_MIN']}–{cfg['DATA']['WINDOW_MAX']} ms · 有效 batch "
            f"{tc['BATCH_SIZE_PER_GPU'] * tc['ACCUMULATE_GRAD_BATCHES']}、{tc['LR_SCHEDULE']}、{tc['MAX_STEPS']} 步、取最后一步 · "
            f"评测：W = {win}、无自适应滤波 / 证据门（evalx eval --controls --tf --perturb）")
    fig, ax = canvas(title, banner, note)

    # 状态支路（上排）：与 S39 相同
    node(ax, 4, 17, 17, 10, "prev 51D\nMANO", hand_skel, TEAL, fs=8.2)
    node(ax, 25, 28.5, 16, 10, "MANO FK", hand_skel, TEAL, caption="序列 GT β · LBS")
    node(ax, 44, 28.5, 18, 10, "投影\n+ z-buffer", grid_icon, TEAL, fs=8.2, caption=f"K × {rs:g} → {RH} × {RW}")
    node(ax, 65, 28.5, 22, 10, "silhouette /\n逆深度图", hand_fill, "#1e3a5f", fs=8.2, caption=f"{chans}（同 S39）")
    elbow(ax, [(12.5, 27.4), (12.5, 33.5), (24.6, 33.5)])
    arrow(ax, (41.4, 33.5), (43.6, 33.5))
    arrow(ax, (62.4, 33.5), (64.6, 33.5))

    # 事件支路（下排）
    node(ax, 26, 5, 23, 10, "异步事件流\n(x, y, t, p)", events_icon, BLUE, fs=8.2)
    node(ax, 53, 5, 27, 10, f"LNES 2 通道\nW = {win} ms", grid_icon, BLUE, fs=8.2)
    ax.text(66.5, 16.0, f"每 50 ms 出一步 · W{win} 原始协议（S39 栈为 W = 200）", ha="center", va="bottom",
            fontsize=6.4, color=CAP, zorder=6)
    arrow(ax, (49.4, 10), (52.6, 10))

    # 活跃位点（上排右）：事件 ∪ 前态网格投影，逐级 2x2 格占据
    hnode(ax, 110, 28.5, 30, 10, f"活跃位点 m0 = 事件 ∪ 网格投影\nA1…A4：逐级 2×2 格占据\nzgz W{win} 中位 {a0:.0f} px"
          f"（{a0 / (RH * RW):.1%}）", None, fs=6.8)

    # 主干
    node(ax, 89, 17, 22, 10, "拼 4 通道\n+ conv1 4→3", stack_icon, TEAL, fs=8.2,
         caption="LNES 2 + 渲染 2 · 输出 × m0")
    hnode(ax, 115, 17, 32, 10, "稀疏 ResNet18-L3\n只算活跃位点", cube_icon, ORANGE, fs=8.2,
          caption=f"CoordConv · stem {WIDTHS[0]} · {' · '.join(map(str, WIDTHS))} · MaskedBN")
    hnode(ax, 151, 17, 24, 10, f"均值 ‖ 面积和\n+ fc {fc_in}→{mc['OUTPUT_DIM']}", funnel_icon, ORANGE, fs=8.2,
          caption=f"A4 {h4}×{w4}：mean ‖ sum / {h4 * w4}")
    node(ax, 179, 17, 7, 10, "Δ", None, fs=13, caption="空包 Δ=0")
    elbow(ax, [(87.4, 33.5), (97, 33.5), (97, 27.4)])                 # 渲染图 -> 拼接
    elbow(ax, [(80.4, 10), (85, 10), (85, 22), (88.6, 22)])           # LNES -> 拼接（从左侧进，让开下方说明）
    elbow(ax, [(106, 27.4), (106, 31.5), (109.6, 31.5)])              # 拼接输入 -> 活跃位点
    elbow(ax, [(140.4, 33.5), (143.5, 33.5), (143.5, 27.4)])          # 活跃位点 -> 稀疏主干（各级掩码）
    arrow(ax, (111.4, 22), (114.6, 22))
    arrow(ax, (147.4, 22), (150.6, 22))
    arrow(ax, (175.4, 22), (178.6, 22))

    # 求和 x = prev + Δ，W50 原始协议不滤波
    ax.add_patch(Circle((192, 22), 1.7, facecolor="#ffffff", edgecolor=TEAL, lw=1.4, zorder=5))
    ax.text(192, 22, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)
    arrow(ax, (186.4, 22), (190.3, 22))
    arrow(ax, (193.7, 22), (199.6, 22))
    node(ax, 200, 17, 30, 10, "x_k = prev + Δ\n下一步的 prev", hand_skel, TEAL, fs=8.2, caption="不经滤波，直接回送")

    # 推理形态（下排右，说明文字）
    ax.text(215, 8.6, "推理：SparseExecutor 只在活跃位点\ngather → GEMM（BN 折叠）\n训练：掩码稠密 cuDNN，同一函数",
            ha="center", va="center", fontsize=6.4, color=CAP, zorder=6, linespacing=1.35)

    # prev 回环（底部走廊）
    elbow(ax, [(12.5, 16.6), (12.5, 3.2), (192, 3.2), (192, 20.2)], lw=1.5)
    ax.text(132, 4.0, "prev（prev_mlp 51→64→51 支路一并加入 Δ；x_k 不经滤波即下一步的 prev，渲染、活跃位点、prev_mlp 都以它为准）",
            ha="center", va="bottom", fontsize=6.8, color=TEAL, zorder=6)
    return fig, "s43_sparse_simple.png"


def main() -> None:
    fig, name = draw()
    out = REPO / "docs/assets" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main()
