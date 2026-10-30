#!/usr/bin/env python3
"""Draw the S43 W50 method as an editable, publication-sized vector figure.

All geometry and maps are illustrative, not measured event data.  The hand is
generated from the real MANO asset; its vertex projections remain points rather
than a filled triangle silhouette.  No model weights, datasets, or GPU are used.

python tools/s43/draw_s43_publication.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
import numpy as np

from figure_geometry import load_hand_geometry

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO / "docs/assets/s43_w50_publication_20261019"
INK = "#233344"
MUTED = "#667486"
EDGE = "#D9E1E9"
BLUE = "#356F9F"
ORANGE = "#CF7755"
GREEN = "#287D6C"
PURPLE = "#766197"
ARROW = "#65758A"

WORDS = {
    "zh": {
        "title": "S43｜稀疏渲染–比对手部跟踪",
        "sub": "事件观测与前态投影在同一像素空间融合，预测 51 维状态修正量",
        "a": "观测与前态投影", "b": "稀疏特征提取与读出", "c": "残差更新与状态递推",
        "prev": "前态", "bypass": "前态旁路", "mesh": "MANO 网格", "meshsub": "778 顶点 · 固定 β、K",
        "render": "顶点点投影", "rendersub": "占据图 / 逆深度图",
        "events": "50 ms 事件", "lnes": "双极性 LNES",
        "cat": "四通道拼接", "union": "活跃集：LNES ∪ 前态投影",
        "adapt": "3×3 适配 4→3 + 坐标通道",
        "cnn": "稀疏 ResNet18-L3", "cnnsub": "每阶段 2 个残差块 · MaskedBN",
        "bn": "活跃位置归一化（MaskedBN）",
        "pool": "活跃均值  ∥  面积归一化和",
        "poolsub": "256 + 256 → FC 512→51",
        "pmlp": "前态修正 MLP", "update": "状态更新 / 空事件保持",
        "nonempty": "有事件", "empty": "空窗口",
        "state": "当前状态", "feedback": "当前状态直接作为下一步前态 · 每 50 ms 更新一次",
        "foot": "51 维 = 平移 3 + 根轴角 3 + 手指轴角 45。网格与事件图均为方法示意。",
        "detail": "活跃区域逐级下采样", "train": "训练：掩码稠密  /  推理：活跃位置稀疏计算",
    },
    "en": {
        "title": "S43 | Sparse render-and-compare hand tracking",
        "sub": "Fuse event evidence and the projected previous hand to estimate a 51-D state correction",
        "a": "Event evidence and prior", "b": "Sparse encoder and readout", "c": "Recurrent state update",
        "prev": "Previous state", "bypass": "Previous-state bypass", "mesh": "MANO mesh", "meshsub": "778 vertices",
        "render": "Vertex projection", "rendersub": "Occupancy / inv. depth",
        "events": "50-ms events", "lnes": "Two-polarity LNES",
        "cat": "4-channel fusion", "union": "Support = LNES ∪ projection",
        "adapt": "3×3 adapter 4→3 + coordinates",
        "cnn": "Sparse ResNet18-L3", "cnnsub": "2 blocks / stage · MaskedBN",
        "bn": "Active-site normalization (MaskedBN)",
        "pool": "Active mean  ∥  area-scaled sum",
        "poolsub": "256 + 256 → FC 512→51",
        "pmlp": "State correction MLP", "update": "Residual update + empty hold",
        "nonempty": "nonempty", "empty": "empty",
        "state": "State", "feedback": "Direct state feedback · one update every 50 ms",
        "foot": "51-D = translation 3 + root axis-angle 3 + finger axis-angles 45. Geometry and events are schematic.",
        "detail": "Hierarchical active support", "train": "Training: masked dense  /  inference: active-site sparse",
    },
}


def label(ax, x, y, text, size=7.2, color=INK, ha="center", weight="normal", **kw):
    return ax.text(x, y, text, fontsize=size, color=color, ha=ha, va="center",
                   fontweight=weight, zorder=12, **kw)


def box(ax, x, y, w, h, face="white", edge=EDGE, lw=0.65, radius=1.2, z=3):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                      fc=face, ec=edge, lw=lw, zorder=z)
    ax.add_patch(p)
    return p


def path(ax, points, color=ARROW, lw=0.85, head=True, z=7):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], c=color, lw=lw, zorder=z,
                solid_capstyle="round")
    a, b = points[-2:]
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>" if head else "-",
                               mutation_scale=6.4, lw=lw, color=color,
                               shrinkA=0, shrinkB=0, zorder=z))


def mesh(ax, g, x, y, w, h):
    xy = g["xy"] * [w, h] + [x, y]
    depth = g["depth"][g["faces"]].mean(1)
    order = np.argsort(depth)
    u = (depth - depth.min()) / np.ptp(depth)
    cmap = mcolors.LinearSegmentedColormap.from_list("hand", ["#E4EDEB", "#89B6AA", "#347F6E"])
    coll = PolyCollection(xy[g["faces"]][order], facecolors=cmap(0.12 + 0.7*u[order]),
                          edgecolors="#548E80", linewidths=0.12, zorder=6)
    ax.add_collection(coll)


def frame(ax, x, y, w, h):
    box(ax, x, y, w, h, face="#FFFFFF", edge="#D1DCE5", radius=0.65, lw=0.6, z=4)


def projected(ax, g, x, y, w, h, kind="occupancy"):
    frame(ax, x, y, w, h)
    pts = (g["xy"]-.5) * (min(w,h)-1.5) + [x+w/2,y+h/2]
    if kind == "depth":
        z = g["depth"]
        color = (z-z.min()) / np.ptp(z)
        cmap = mcolors.LinearSegmentedColormap.from_list("depth", ["#ABCFC2", "#246D63"])
        ax.scatter(*pts.T, s=0.55, c=color, cmap=cmap, linewidths=0, zorder=6)
    else:
        ax.scatter(*pts.T, s=0.45, c=GREEN, alpha=0.8, linewidths=0, zorder=6)


def event_data(g):
    rng = np.random.default_rng(43)
    j = g["joints_xy"]
    chunks = []
    for a, b in g["skeleton_edges"]:
        t = rng.uniform(0, 1, 14)[:, None]
        p = j[a]*(1-t) + j[b]*t + rng.normal(0, 0.017, (len(t), 2))
        chunks.append(p)
    pts = np.concatenate(chunks)
    # Shift event evidence slightly relative to the prior to illustrate motion.
    pts += [0.025, 0.01]
    return np.clip(pts, 0.05, 0.95), rng.integers(0, 2, len(pts)), rng.uniform(0.25, 1, len(pts))


def event_map(ax, data, x, y, w, h, polarity=None, stacked=False):
    pts, pol, times = data
    if stacked:
        frame(ax, x, y, w, h)
    if polarity is not None:
        keep = pol == polarity
        pts, pol, times = pts[keep], pol[keep], times[keep]
    pts = (pts-.5) * (min(w,h)-1) + [x+w/2,y+h/2]
    colors = np.array([mcolors.to_rgba(BLUE), mcolors.to_rgba(ORANGE)])[pol]
    colors[:, 3] = 0.4 + times*0.6
    ax.scatter(*pts.T, s=1.15 if polarity is None else 0.9, c=colors,
               edgecolors="none", zorder=6)


def support_cells(g, data, n=14):
    m = np.zeros((n, n), dtype=bool)
    for arr in (g["xy"], data[0]):
        p = np.clip((arr*n).astype(int), 0, n-1)
        m[p[:, 1], p[:, 0]] = True
    return m


def channel_support(points, n=10):
    m = np.zeros((n,n), dtype=bool)
    p = np.clip((points*n).astype(int),0,n-1)
    m[p[:,1],p[:,0]] = True
    return m


def support_map(ax, mask, x, y, w, h, color=PURPLE, grid=False):
    nr, nc = mask.shape
    frame(ax, x, y, w, h)
    cw, ch = w/nc, h/nr
    for r in range(nr):
        for c in range(nc):
            if mask[r, c]:
                ax.add_patch(Rectangle((x+c*cw+cw*0.12, y+r*ch+ch*0.12), cw*0.76, ch*0.76,
                                       fc=color, ec="none", alpha=0.68, zorder=6))
            elif grid:
                ax.add_patch(Rectangle((x+c*cw, y+r*ch), cw, ch,
                                       fc="none", ec="#EDEAF2", lw=0.2, zorder=5))


def feature_block(ax, x, y, w, h, depth, n, active):
    front = np.array([[x,y], [x+w,y], [x+w,y+h], [x,y+h]])
    top = np.array([[x,y+h], [x+depth,y+h+depth*0.6],
                    [x+w+depth,y+h+depth*0.6], [x+w,y+h]])
    side = np.array([[x+w,y], [x+w+depth,y+depth*0.6],
                     [x+w+depth,y+h+depth*0.6], [x+w,y+h]])
    ax.add_patch(Polygon(front, fc="#F5F2F9", ec=PURPLE, lw=0.65, zorder=5))
    ax.add_patch(Polygon(top, fc="#EBE5F3", ec=PURPLE, lw=0.65, zorder=5))
    ax.add_patch(Polygon(side, fc="#D7CDE7", ec=PURPLE, lw=0.65, zorder=5))
    for r, c in np.argwhere(active):
        if r < n and c < n:
            ax.add_patch(Rectangle((x+(c+.18)*w/n, y+(r+.18)*h/n), .64*w/n, .64*h/n,
                                   fc=PURPLE, ec="none", alpha=.75, zorder=6))


def make_figure(lang, g):
    t = WORDS[lang]
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"] if lang == "zh" else ["DejaVu Sans", "Noto Sans CJK JP"],
                         "mathtext.fontset": "dejavusans", "axes.unicode_minus": False,
                         "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(190/25.4, 108/25.4), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 190), ylim=(0, 108), aspect="equal")
    ax.axis("off")
    label(ax, 3, 103, t["title"], size=11.4, ha="left", weight="bold")
    label(ax, 3, 97.3, t["sub"], size=7.2, color=MUTED, ha="left")
    box(ax, 174.5, 99.3, 12, 6, face="#EEF3F7", edge="none", radius=1.1)
    label(ax, 180.5, 102.3, "W50", size=8, weight="bold", color=BLUE)
    ax.plot([3, 187], [93.7, 93.7], color=EDGE, lw=.65)

    for x, w, color, face, letter, key in [
        (3,60,BLUE,"#F5F8FC","a","a"),
        (67,61,PURPLE,"#F8F6FB","b","b"),
        (132,55,GREEN,"#F4F9F7","c","c")]:
        box(ax, x, 20, w, 65, face=face, edge="none", radius=2, z=0)
        label(ax, x, 89.1, f"({letter})", size=8.6, weight="bold", ha="left", color=color)
        label(ax, x+7, 89.1, t[key], size=8.0 if lang=="en" else 8.5, ha="left", weight="bold")

    # Previous state and two bypasses: correction MLP and the additive residual.
    box(ax, 8, 76, 48, 7, face="white", edge="#B7CDC5", radius=1)
    label(ax, 32, 79.5, t["prev"]+r"  $\mathbf{x}_{k-1}\in\mathbb{R}^{51}$", size=7.7, color=GREEN)
    path(ax, [(56,79.5),(182.5,79.5),(182.5,48),(178,48)], color=GREEN, lw=.85)
    label(ax, 98, 81.8, t["bypass"], size=6.3, color=GREEN,
          bbox=dict(fc="white", ec="none", pad=0.6))
    ax.add_patch(Circle((156,79.5), .5, fc=GREEN, ec="none", zorder=8))
    path(ax, [(156,79.5),(156,74.5)], color=GREEN)
    path(ax, [(18,76),(18,70.5)], color=GREEN)

    # Panel a: actual MANO geometry and point-projected prior maps.
    mesh(ax, g, 7.3, 50, 21, 21)
    label(ax, 18, 47.8, t["mesh"], size=7.1, color=GREEN, weight="bold")
    label(ax, 18, 44.1, t["meshsub"], size=5.8, color=MUTED)
    path(ax, [(28,60),(33.7,60)], color=GREEN)
    projected(ax, g, 36.2, 54.1, 19.5, 14.6, "depth")
    projected(ax, g, 33.4, 51.3, 19.5, 14.6, "occupancy")
    label(ax, 45, 47.8, t["render"], size=7.1, color=GREEN, weight="bold")
    label(ax, 45, 44.1, t["rendersub"], size=5.8, color=MUTED)

    data = event_data(g)
    event_map(ax, data, 7.8, 24.4, 19.3, 13.5)
    label(ax, 18, 39.9, t["events"], size=7.1, color=BLUE, weight="bold",
          bbox=dict(fc="#F5F8FC",ec="none",pad=.6))
    label(ax, 18, 22.6, r"$(x,y,t,p)$", size=6.4, color=MUTED)
    path(ax, [(28,31.3),(33.7,31.3)], color=BLUE)
    event_map(ax, data, 36.2, 26.2, 19.5, 12.5, polarity=1, stacked=True)
    event_map(ax, data, 33.4, 23.4, 19.5, 12.5, polarity=0, stacked=True)
    label(ax, 45, 39.9, t["lnes"], size=7.1, color=BLUE, weight="bold",
          bbox=dict(fc="#F5F8FC",ec="none",pad=.6))
    label(ax, 45, 22.0, r"$180\times240\times2$", size=6.2, color=MUTED)

    # Merge the two pixel-aligned inputs. Two distinct arrows terminate at the fusion stack.
    path(ax, [(54,61),(60.4,61),(60.4,57.5),(70,57.5)], color=GREEN)
    path(ax, [(54,31.3),(60.4,31.3),(60.4,49.5),(70,49.5)], color=BLUE)

    # Panel b: four channels, input support, sparse hierarchy and faithful dual pooling.
    label(ax, 97.5, 72.4, t["union"], size=7.1, color=PURPLE, weight="bold")
    label(ax, 97.5, 67.4, t["adapt"], size=6.2, color=MUTED)
    channel_masks = [channel_support(data[0][data[1]==0]), channel_support(data[0][data[1]==1]),
                     channel_support(g["xy"]),channel_support(g["xy"])]
    for i, col in reversed(list(enumerate([BLUE,ORANGE,GREEN,"#589483"]))):
        support_map(ax, channel_masks[i], 70+i*.9, 45+i*.9, 12.5, 15.5, color=col)
    label(ax, 78, 41.7, t["cat"], size=6.3, color=PURPLE)
    path(ax, [(86,53.8),(90,53.8)], color=PURPLE)
    feature_block(ax,91,46.3,6.4,12.5,2.2,8,support_cells(g,data,8))
    path(ax, [(100.1,53.8),(102,53.8)], color=PURPLE, lw=.7)
    feature_block(ax,102.4,48.7,5.5,9.6,2.6,6,support_cells(g,data,6))
    path(ax, [(111.2,53.8),(113.2,53.8)], color=PURPLE, lw=.7)
    feature_block(ax,113.6,50.1,4.6,6.8,3.0,4,support_cells(g,data,4))
    for x, width in [(95.3,64),(106.0,128),(117.5,256)]:
        label(ax, x, 43, str(width), size=6.4, color=PURPLE)
    label(ax, 106.4, 62.9, t["cnn"], size=7.0, color=PURPLE, weight="bold")
    label(ax, 97.5, 38, t["cnnsub"], size=5.8, color=MUTED)
    box(ax, 71, 23, 53, 11.5, face="white", edge="#D4CBE1", radius=1)
    label(ax, 97.5, 30.5, t["pool"], size=6.6 if lang=="en" else 7.0, color=PURPLE, weight="bold")
    label(ax, 97.5, 26.1, t["poolsub"], size=6.8, color=MUTED)
    path(ax, [(121.2,53.8),(124.5,53.8),(124.5,36),(117,36),(117,34.5)], color=PURPLE, lw=.75)

    # Panel c: add both learned corrections, gate the whole delta, then add prev.
    box(ax, 138, 65, 40, 9.5, face="white", edge="#B7CDC5", radius=1)
    label(ax, 158, 71.5, t["pmlp"], size=7.0, color=GREEN, weight="bold")
    label(ax, 158, 67.8, "51 → 64 → 51", size=6.6, color=MUTED)
    path(ax, [(151,65),(151,62.1)], color=GREEN)
    label(ax, 158.4, 62.8, r"$\Delta_{\mathrm{MLP}}$", size=7.0, color=GREEN)
    ax.add_patch(Circle((151,59.8), 2.0, fc="white", ec=GREEN, lw=.9, zorder=9))
    label(ax, 151, 59.8, "+", size=10, color=GREEN)
    path(ax, [(124,28.7),(130,28.7),(130,59.8),(149,59.8)], color=PURPLE)
    label(ax, 138.3, 62.5, r"$\Delta_{\mathrm{CNN}}$", size=7.0, color=PURPLE)
    path(ax, [(151,57.8),(151,54.6)], color=GREEN)
    label(ax, 155, 55.6, r"$\Delta$", size=7.2, color=GREEN)
    box(ax, 138, 36, 40, 18.5, face="white", edge="#AFC9BF", radius=1.2, lw=.8)
    label(ax, 158, 51.2, t["update"], size=6.5 if lang=="en" else 7.0, color=GREEN, weight="bold")
    label(ax, 158, 45.6, r"$\mathbf{x}_k=\mathbf{x}_{k-1}+g_k\,\Delta$", size=8.0, color=INK)
    label(ax, 158, 40.0, r"$g_k=\mathbf{1}[\|L_k\|_1>0]$", size=7.3, color=MUTED)
    path(ax, [(158,36),(158,31.5)], color=GREEN)
    box(ax, 138, 23, 40, 8.5, face="#E8F2EE", edge="none", radius=1)
    label(ax, 158, 27.25, t["state"]+r"  $\mathbf{x}_k\in\mathbb{R}^{51}$", size=7.5, color=GREEN, weight="bold")

    # One unobstructed recurrent edge, kept outside all three panels.
    path(ax, [(158,23),(158,12),(0.8,12),(0.8,79.5),(8,79.5)], color=GREEN, lw=.95)
    label(ax, 87, 12, t["feedback"], size=7.0, color=GREEN,
          bbox=dict(fc="white",ec="none",pad=1.5))
    label(ax, 3, 5.8, t["foot"], size=6.0, color=MUTED, ha="left")
    return fig


CAPTIONS = {
    "zh": "S43 的 W50 原始递推流程。每 50 ms 将当前窗口事件编码为双极性 LNES；前态经 MANO 正向运动学生成 778 个顶点，并按相机内参点投影及深度缓冲生成占据图和逆深度图。四通道输入的非零支持并集定义 A0，后续四级支持通过 2×2 格占据构建。3×3 适配卷积之后加入坐标通道，经稀疏 ResNet18-L3 和仅统计活跃位置的 MaskedBN 提取特征。末层活跃均值与特征总和除以整层网格面积拼接，得到 512 维读出，经全连接预测 51 维修正量；前态 MLP 的修正量在零 LNES 门之前相加，随后与前态相加得到当前状态并直接反馈。根与手指旋转使用轴角参数相加。训练采用掩码稠密实现，推理采用活跃位置稀疏执行。形状 β 与相机内参 K 在序列内固定；本配置 β 来自序列真值。所有网格、事件点及活跃图均为方法示意，事件图不是实验样例。图中不包含滤波或更新死区。",
    "en": "S43 under the W50 raw recursive protocol. Every 50 ms, events in the current window are encoded as a two-polarity LNES. The previous 51-D state is decoded by MANO forward kinematics; 778 vertices are point-projected with a depth buffer to obtain occupancy and inverse-depth maps. The nonzero support of the four-channel input defines A0, and four subsequent support levels are constructed by 2×2 cell occupancy. A 3×3 input adapter, coordinate channels, and a sparse ResNet18-L3 with active-site MaskedBN encode the input. Concatenating the active-site mean and the feature sum divided by the full final-grid area gives a 512-D representation, which is mapped to a 51-D correction. A previous-state MLP supplies an additional correction; the entire correction is gated to zero when the LNES is empty, then added to the previous state. The resulting state is fed directly to the next step. Root and finger rotations use additive axis-angle parameters. Training uses a masked dense implementation and inference uses active-site sparse execution. Sequence shape β and camera intrinsics K are fixed; this configuration uses sequence-ground-truth β. Mesh, event, and support drawings are illustrative rather than measured examples. No filtering or update deadband is included.",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dpi", type=int, default=600)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    g = load_hand_geometry(REPO)
    manifest = {"protocol": "W50 raw; 50-ms window and update interval", "canvas_mm": [190,108],
                "raster_dpi": args.dpi, "illustration_note": g["illustration_note"],
                "config": "configs/s43/s43n_sp.yaml", "files": []}
    for lang in ("zh", "en"):
        fig = make_figure(lang, g)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        # Meaningful layout guard: no textual content may escape the canvas.
        for txt in fig.axes[0].texts:
            bounds = txt.get_window_extent(renderer)
            if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.width or bounds.y1 > fig.bbox.height:
                raise RuntimeError(f"Text outside canvas: {txt.get_text()!r}")
        base = args.out_dir / f"s43_w50_{lang}"
        for suffix in ("svg", "pdf", "png"):
            p = base.with_suffix("."+suffix)
            fig.savefig(p, dpi=args.dpi, facecolor="white", metadata={"Creator": "S43 W50 figure generator"})
            manifest["files"].append({"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
            print(p)
        fig.savefig(base.with_suffix(".tiff"), dpi=args.dpi, facecolor="white",
                    pil_kwargs={"compression": "tiff_lzw"})
        manifest["files"].append({"path": base.with_suffix(".tiff").name})
        (args.out_dir/f"caption_{lang}.txt").write_text(CAPTIONS[lang]+"\n", encoding="utf-8")
        plt.close(fig)
    (args.out_dir/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
