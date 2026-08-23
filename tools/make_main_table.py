#!/usr/bin/env python3
"""Render the consolidated main results table as markdown and PNG.

Accuracy is read straight out of the evaluation JSONs wherever those still exist, so
the table cannot drift from the artifacts. Cost columns (latency / MACs / params) are
re-measured in one interleaved session on the current machine, which is the only way
the rows become comparable: the historical numbers in docs were taken across several
sessions and disagree by up to 10% on the render rows.

Only methods that still exist in the tree are listed. KSGN, domain randomization and
the delta-trust inference flag were removed by the rollback to EventHands-AbsRender, so
their rows are gone from here; their history stays in docs sections 9 to 11.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import textwrap
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model"))

from config import load_config  # noqa: E402
from model import MNISTModel  # noqa: E402

OUT = ROOT / "outputs" / "hand_data51"
FULL_PUBLISHED_MS = 1.75  # the anchor the published latency column is expressed on

# (label, cost config, accuracy source, note key)
#   accuracy source: ("json", relative eval dir) reads the recursive metrics;
#                    ("lit", 4 values) keeps a published frame-wise/absolute reading.
ROWS = [
    dict(label="EventHands-PCA6", cost="eventhands_abs_pca12.yaml",
         acc=("lit", (30.00, 10.99, 23.58, 8.15)), ra=None, note="a",
         prov="abs_pca12 · eval_abs"),
    dict(label="EventHands-Full", cost="eventhands_abs_full51.yaml",
         acc=("lit", (28.56, 11.78, 24.01, 9.16)), ra=("json", "abs_full51/eval"),
         note="a,b", prov="abs_full51 · eval_abs / eval"),
    dict(label="EventHands-Track (Δ)", cost="eventhands_track_delta51.yaml",
         acc=("json", "track_delta51/eval_last"), ra=("json", "track_delta51/eval_last"),
         note="c", prov="track_delta51 · last.ckpt"),
    dict(label="EventHands-AbsRender (4ch abs)", cost="eventhands_abs_render51.yaml",
         acc=("json", "abs_render51/eval_step3000"),
         ra=("json", "abs_render51/eval_step3000"),
         prov="abs_render51 · step3000"),
    dict(label="EventHands-Track (render)",
         cost="eventhands_track_render51_ch_both.yaml",
         acc=("json", "track_render51_ch_both_rep2/eval_step1500"),
         ra=("json", "track_render51_ch_both_rep2/eval_step1500"), note="d,e",
         prov="…ch_both_rep2 · step1500"),
    dict(label="EventHands-Track (render, sil only)",
         cost="eventhands_track_render51_ch_sil.yaml",
         acc=("json", "track_render51_ch_sil/eval_step1500"),
         ra=("json", "track_render51_ch_sil/eval_step1500"), note="e",
         prov="…ch_sil · step1500"),
    dict(label="EventHands-Track (render, inv only)",
         cost="eventhands_track_render51_ch_inv.yaml",
         acc=("json", "track_render51_ch_inv_rep2/eval_step1000"),
         ra=("json", "track_render51_ch_inv_rep2/eval_step1000"), note="e",
         prov="…ch_inv_rep2 · step1000"),
    dict(label="EventHands-Track (render, 语义 mask)",
         cost="eventhands_track_render51_ch_semsil.yaml",
         acc=("json", "track_render51_ch_semsil/eval_step1500"),
         ra=("json", "track_render51_ch_semsil/eval_step1500"), note="e,h",
         prov="…ch_semsil · step1500"),
    dict(label="EventHands-Track (render, SO3+FK loss)",
         cost="eventhands_track_render51_so3fk.yaml",
         acc=("json", "track_render51_so3fk_rep2/eval_step2500"),
         ra=("json", "track_render51_so3fk_rep2/eval_step2500"), note="e,f,g",
         prov="…so3fk_rep2 · step2500"),
    dict(label="EventHands-Track (render, 语义 mask + SO3+FK loss)",
         cost="eventhands_track_render51_sem.yaml",
         acc=("json", "track_render51_sem_rep2/eval_step2500"),
         ra=("json", "track_render51_sem_rep2/eval_step2500"), note="e,h",
         prov="…sem_rep2 · step2500"),
]

# Row indices that start a new block; the renderer draws a heavier rule above each.
# Blocks: absolute baselines | open-loop delta | render mechanism | channel ablation
# (incl. the semantic mask, whose control is ch_both) | loss ablation and the same
# semantic mask on top of it.
GROUP_ENDS = {2, 3, 5, 8}


def read_eval(rel):
    p = OUT / rel / "track_metrics_step50.json"
    m = json.loads(p.read_text())
    per = {r["seq"]: r for r in m["per_sequence"]}
    return {
        "ra": m["overall"]["mpjpe_ra_mm"],
        "acc": (per["zgz_local"]["mpjpe_ra_mm"], per["zgz_global"]["mpjpe_ra_mm"],
                per["zgz_local"]["mpvpe_ra_mm"], per["zgz_global"]["mpvpe_ra_mm"]),
    }


@torch.no_grad()
def measure(cfg_names, iters=300, warmup=300, rounds=8):
    """Batch-1 latency (min over rounds), MACs and params, one model at a time.

    Holding every model resident and timing them round-robin looks like it should
    cancel drift, but it inflates the cheap rows much more than the expensive ones
    (Full +26% vs Track +10% when 8 MANO layers are co-resident) and so corrupts the
    ratio the scaled column depends on. Measuring in isolation is reproducible to
    about 1.5% across independent processes on this machine, where the SM clock is
    pinned, so isolation beats interleaving here.
    """
    try:
        from thop import profile
    except ImportError:
        profile = None
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {}
    x = torch.rand(1, 180, 240, 2, device=dev)
    prev = torch.randn(1, 51, device=dev) * 0.02
    prev[:, 2] += 0.4
    for c in cfg_names:
        m = MNISTModel(load_config(ROOT / "configs" / c)).eval()
        n_params = sum(p.numel() for p in m.parameters())
        macs = None
        if profile is not None:
            # thop mutates and re-devices what it is handed, so profile a detached copy.
            trunk = copy.deepcopy(torch.nn.Sequential(m.conv1, m.rn))
            macs, _ = profile(trunk, inputs=(
                torch.zeros(1, m.conv1.weight.shape[1], 180, 240),), verbose=False)
            del trunk
        m = m.to(dev)
        rec = {"params": n_params, "macs": macs, "raw_ms": None, "raw_spread_ms": None}
        if dev.type == "cuda":
            for _ in range(warmup):
                m(x, prev)
            torch.cuda.synchronize()
            samples = []
            for _ in range(rounds):
                t0 = time.perf_counter()
                for _ in range(iters):
                    m(x, prev)
                torch.cuda.synchronize()
                samples.append((time.perf_counter() - t0) / iters * 1000.0)
            rec["raw_ms"] = min(samples)
            rec["raw_spread_ms"] = max(samples) - min(samples)
        out[c] = rec
        del m
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return out


def build(rounds=32):
    cfgs = sorted({r["cost"] for r in ROWS})
    cost = measure(cfgs, rounds=rounds)
    anchor = cost["eventhands_abs_full51.yaml"]["raw_ms"]
    for v in cost.values():
        v["ms"] = (None if anchor in (None, 0) or v["raw_ms"] is None
                   else v["raw_ms"] / anchor * FULL_PUBLISHED_MS)
    rows = []
    for r in ROWS:
        c = cost[r["cost"]]
        acc = r["acc"][1] if r["acc"][0] == "lit" else read_eval(r["acc"][1])["acc"]
        if r.get("ra") is None:
            ra = None
        elif r["ra"][0] == "lit":
            ra = r["ra"][1]
        else:
            ra = read_eval(r["ra"][1])["ra"]
        rows.append({"label": r["label"], "prov": r.get("prov", ""),
                     "acc": acc, "ra": ra, "ms": c["ms"],
                     "macs": c["macs"], "params": c["params"],
                     "note": r.get("note", ""), "bold": r.get("bold", False),
                     "cost_config": r["cost"], "raw_ms": c["raw_ms"],
                     "raw_spread_ms": c.get("raw_spread_ms")})
    return rows, anchor


FOOTNOTES = [
    "a  PCA6 / Full 的四列精度为逐帧绝对评测（eval_abs.py，更密采样）；其余各行为递推协议"
    "（step=50 ms，每段 valid run 开头 GT+噪声初始化，之后 prev←pred，不再喂姿态 GT）。",
    "b  Full 的 RA 取自同一权重的递推评测（abs_full51/eval），故 RA 列全表同协议可比；"
    "该协议下 Full 的 local/global 为 29.12 / 12.03。PCA6 无递推评测，RA 留空。",
    "c  Track(Δ) 取 last.ckpt（与历史表格一致）；其 eval_best 更差（RA 93.38）。",
    "d  本行取的是 Track (render) 的独立重训练（2 副本 × 6 测点，见 e 的口径），不是历史上那次原始 run"
    "（seed 0、2 卡×1024、4 个测点）。历史发表值 19.26 是那 4 个测点里的最小值，与本表其余各行的保守口径"
    "不可比：按同一口径重算，原始 run 与本行的 RA 分布均值是 20.63 vs 20.70（差 0.07 mm，而单次重训的 "
    "sd 约 1.2–1.3 mm），即两者测的是同一个网络、同一个水平；若都取各自最小值，本行是 18.73、反而优于 "
    "19.26。故合并为一行，避免用选点规则制造出不存在的差距。见 §12.4。",
    "e  全表递推各行统一口径：每副本取最优步、每臂取更差副本（单副本的行即其最优步）。"
    "§12 三个通道臂之间差异 ≤0.5 mm，低于该轮 1.1 mm 的分辨率，不可区分。见 §12。",
    "f  SO3+FK 行与 Track (render) 行同配方（单卡 2048、3000 步、每 500 步存盘、仅 LOSS 块不同），"
    "故可直接比大小。预注册门槛为 ΔRA ≤ −1.1 mm，实测池化 −0.78 ± 0.61 mm，判定为打平。见 §13。",
    "g  本表四列精度与 RA 列均为腕对齐口径，看不到该行的绝对定位退化（非对齐 MPJPE 池化均值 "
    "63.7 → 83.6 mm）：新 loss 的 L_trans 走米制 SmoothL1 二次区、L_FK 又是 root-relative，"
    "总目标里几乎没有约束绝对平移的项。改 loss 不改变推理图，故成本三列与 Track (render) 行同值。见 §13.4。",
    "h  「语义 mask」把 LBS 蒙皮权重导出的径向骨架码写进 silhouette 的取值（掩码内 0.35+0.65·码），"
    "掩码形状、通道数、参数量、FLOPs 全部不变，深度通道保持米制。同一改动在两种 loss 基线上各测一次，"
    "各与自己的对照只差 MODEL.RENDER_CHANNELS 一个键：叠在 51D MSE 上（对照 Track (render)）池化 "
    "−0.44 ± 0.48 mm，叠在 SO3+FK 上（对照 SO3+FK 行）池化 +1.21 ± 0.97 mm。门槛 ΔRA ≤ −1.1 mm，"
    "**两臂均未过槛**，但符号相反：语义在原始 loss 下微弱有利且把副本方差压到全表最低（sd 0.96 vs 对照 1.28），"
    "在新 loss 下则明显不利且方差最大（sd 2.97）。交互约 1.65 mm、约 1.5σ，不可声称但足以说明"
    "「语义 mask 无用」这个结论只在 SO3+FK 基线上成立。见 §14.3。",
    "注  KSGN（§9）、域随机化（§10-11）与 δ-trust 推理开关的代码均已随回退到 EventHands-AbsRender 删除；"
    "LBS 关节驱动输入通道（原 §13）因设置有问题已撤回；迭代 render-and-compare（原 §14 的 it2 / sem_it2 臂）"
    "因闭环漂移放大已撤回。三者的代码、config 与产物均已删除，故均不在本表内。",
    "*  Latency：batch=1，同机逐个模型单独驻留测量（32 轮×300 次取最小），再按与 Full 的实测"
    "比值对齐到 Full=1.75 ms。轮数是必要的：光栅化的 scatter 路径依赖数据，8 轮取最小尚未触底，"
    "两次独立 8 轮会话在渲染行上相差达 8%（0.4 ms）；32 轮下两次独立会话的偏差 ≤2.9%、"
    "多数行 ≤1%，且各行标定值回到历史发表值（±0.03 ms）。即便如此，各渲染行之间"
    "（4.4–4.9 ms）的差仍不具判别力，只有渲染行与非渲染行之间的约 2.6 倍差距是实的。"
    "FLOPs/step 为 thop 的 MAC 计数（未乘 2，以 CNN 主干为主，MANO 点撒渲染未计入）。"
    "每行的权重来源见 main_table.json 的 prov 字段。",
]

HEADERS = ["Model", "权重来源\n(run · ckpt)", "MPJPE\nlocal", "MPJPE\nglobal",
           "MPVPE\nlocal", "MPVPE\nglobal", "RA-MPJPE\n(递推)", "Latency",
           "FLOPs/step", "Params"]


def cells(r):
    return [
        r["label"] + (f" [{r['note']}]" if r["note"] else ""),
        r.get("prov", ""),
        f"{r['acc'][0]:.2f}", f"{r['acc'][1]:.2f}",
        f"{r['acc'][2]:.2f}", f"{r['acc'][3]:.2f}",
        "—" if r["ra"] is None else f"{r['ra']:.2f}",
        "n/a" if r["ms"] is None else f"{r['ms']:.2f} ms",
        "n/a" if r["macs"] is None else f"{r['macs']/1e9:.3f} G",
        f"{r['params']/1e6:.2f} M",
    ]


def to_markdown(rows):
    out = ["| " + " | ".join(h.replace("\n", " ") for h in HEADERS) + " |",
           "|---" * len(HEADERS) + "|"]
    for r in rows:
        c = cells(r)
        if r["bold"]:
            c = [f"**{v}**" for v in c]
        out.append("| " + " | ".join(c) + " |")
    out.append("")
    out += [f"- {t}" for t in FOOTNOTES]
    return "\n".join(out) + "\n"


def to_png(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt

    for cand in ("Noto Sans CJK JP", "Noto Sans CJK SC", "WenQuanYi Zen Hei",
                 "Source Han Sans CN", "SimHei", "Microsoft YaHei"):
        if any(f.name == cand for f in fm.fontManager.ttflist):
            plt.rcParams["font.family"] = cand
            break
    plt.rcParams["axes.unicode_minus"] = False

    # Footnotes are wrapped explicitly so the layout can reserve the right height;
    # matplotlib's own wrap=True gives no way to ask how many lines it produced.
    note_lines = [ln for t in FOOTNOTES for ln in textwrap.wrap(t, width=104,
                                                                subsequent_indent="   ")]
    # The model column carries the longest label ("... 语义 mask + SO3+FK loss [e,h]"),
    # so it gets the slack the five-character accuracy columns do not need.
    widths = [0.290, 0.171, 0.058, 0.062, 0.058, 0.062, 0.080, 0.072, 0.080, 0.065]
    n = len(rows)
    row_h, head_h, note_h = 0.30, 0.52, 0.215
    fig_w = 13.2
    fig_h = head_h + n * row_h + note_h * len(note_lines) + 0.75
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    table_top = 1.0
    table_bottom = table_top - (head_h + n * row_h) / fig_h
    y_head = table_top - head_h / fig_h
    dy = row_h / fig_h

    xs = [0.0]
    for w in widths:
        xs.append(xs[-1] + w)

    ax.add_patch(plt.Rectangle((0, y_head), xs[-1], table_top - y_head,
                               facecolor="#eef2f7", edgecolor="none", zorder=0))
    for i, r in enumerate(rows):
        if i % 2 == 1:
            ax.add_patch(plt.Rectangle((0, y_head - (i + 1) * dy), xs[-1], dy,
                                       facecolor="#fafbfc", edgecolor="none", zorder=0))
        if r["bold"]:
            ax.add_patch(plt.Rectangle((0, y_head - (i + 1) * dy), xs[-1], dy,
                                       facecolor="#fff6e0", edgecolor="none", zorder=0))

    for i, h in enumerate(HEADERS):
        left = i <= 1
        ax.text(xs[i] + 0.006 if left else xs[i + 1] - 0.006,
                y_head + (table_top - y_head) / 2, h,
                ha="left" if left else "right", va="center",
                fontsize=8.6 if i == 1 else 9.5, fontweight="bold", color="#1b2733")

    for i, r in enumerate(rows):
        y = y_head - (i + 0.5) * dy
        for j, v in enumerate(cells(r)):
            left = j <= 1
            ax.text(xs[j] + 0.006 if left else xs[j + 1] - 0.006, y, v,
                    ha="left" if left else "right", va="center",
                    fontsize=8.9 if j == 0 else (7.9 if j == 1 else 9.5),
                    fontweight="bold" if r["bold"] else "normal",
                    color="#5c6b7a" if j == 1 else "#1b2733")

    ax.plot([0, xs[-1]], [table_top, table_top], color="#1b2733", lw=1.4)
    ax.plot([0, xs[-1]], [y_head, y_head], color="#1b2733", lw=1.0)
    ax.plot([0, xs[-1]], [table_bottom, table_bottom], color="#1b2733", lw=1.4)
    for i in range(1, n):
        y = y_head - i * dy
        group = i in GROUP_ENDS
        ax.plot([0, xs[-1]], [y, y], color="#33414f" if group else "#dde3ea",
                lw=1.3 if group else 0.5)

    y = table_bottom - 0.42 / fig_h
    for ln in note_lines:
        ax.text(0.0, y, ln, ha="left", va="top", fontsize=7.6, color="#465565")
        y -= note_h / fig_h

    fig.subplots_adjust(left=0.012, right=0.995, top=0.985, bottom=0.01)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", default="outputs/hand_data51/main_table.png")
    ap.add_argument("--md", default="outputs/hand_data51/main_table.md")
    ap.add_argument("--reuse", action="store_true",
                    help="re-render from the saved JSON instead of re-measuring, so "
                         "cosmetic edits do not silently move the numbers")
    ap.add_argument("--rounds", type=int, default=32,
                    help="timing rounds per model. The reported figure is the min over "
                         "rounds, and on the render rows 8 rounds does not reach the "
                         "floor: two independent 8-round sessions disagreed by up to "
                         "8%% (0.4 ms), against ~1%% on the render-free rows, because "
                         "the rasterizer's scatter path is data dependent")
    args = ap.parse_args()

    if args.reuse:
        saved = json.loads((ROOT / args.md).with_suffix(".json").read_text())
        anchor = saved["anchor_full_raw_ms"]
        # Only the measured columns are reused; the row set, its order and the metadata
        # come from ROWS, so deleting a row here actually removes it from the artifact
        # instead of leaving the saved copy in place.
        by_label = {r["label"]: r for r in saved["rows"]}
        missing = [r["label"] for r in ROWS if r["label"] not in by_label]
        if missing:
            raise SystemExit(f"--reuse has no saved measurement for: {missing}")
        rows = []
        for m in ROWS:
            row = by_label[m["label"]]
            row["prov"] = m.get("prov", row.get("prov", ""))
            row["note"] = m.get("note", row.get("note", ""))
            row["cost_config"] = m["cost"]
            rows.append(row)
    else:
        rows, anchor = build(rounds=args.rounds)
    md = to_markdown(rows)
    (ROOT / args.md).write_text(md)
    (ROOT / args.md).with_suffix(".json").write_text(json.dumps(
        {"anchor_full_raw_ms": anchor, "rows": rows}, ensure_ascii=False, indent=2))
    to_png(rows, ROOT / args.png)
    print(md)
    print("Full raw latency anchor: %.3f ms" % anchor)
    print("wrote", ROOT / args.png)


if __name__ == "__main__":
    main()
