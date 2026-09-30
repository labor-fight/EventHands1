#!/usr/bin/env python3
"""Render the post-DEBUG XYZ candidate and its explicit data-flow contract.

This draws a proposal, not a trained model. No model or experiment is executed.
Nodes/edges are shared by the drawing, JSON manifest, and structural validation.
"""
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/s37_xyz_recommended_20260929"
WIDTH, HEIGHT = 2630, 1050
BG, INK, MUTED = "#f5f7fa", "#233348", "#6d7d91"
GREEN, BLUE, ORANGE, BORDER = "#009f78", "#377de0", "#e96621", "#d3dce7"

plt.rcParams.update({
    "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
})


@dataclass(frozen=True)
class Node:
    key: str
    title: str
    body: str
    rect: tuple
    learned: bool = False
    body_size: float = 11.5


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    payload: str
    points: tuple
    kind: str = "feature"
    delayed: bool = False


NODES = [
    Node("prev", "历史状态 prev", "s_prev：51D\n平移 3 + 旋转参数 48", (65, 190, 200, 120)),
    Node("events", "当前事件包 + 内参 K", "(u, v, t, p)\n事件没有实测深度 z", (65, 440, 200, 140), body_size=11),
    Node("prepare", "几何与事件准备", "FK(s_prev; β) → 778×3 XYZ\n(u,v,K) → 相机射线 d\n射线—表面关联 → A、q\nz_prior、valid、ΔXYZ\n无几何记录：valid=0",
         (325, 335, 300, 300)),
    Node("relation", "① 共享关系编码 φ", "时间 t / 极性 p 与三维关系\n先联合编码，再聚合\n匹配 / 未匹配共用权重",
         (685, 405, 285, 220), True),
    Node("scatter", "顶点聚合", "按 qA 写入 778 顶点\nmean / max / mass\n无观测顶点仍保留",
         (1025, 420, 205, 190), body_size=11),
    Node("mesh", "② 一个 MeshGNN", "固定面 1-ring\nΔXYZ 调制消息传递\n区分直接 / 传播支持",
         (1290, 405, 265, 220), True),
    Node("lbs", "LBS 关节池化", "固定 LBS 权重 W\n16 槽保留关节顺序\nmean / max / 支持统计",
         (1615, 420, 240, 190), body_size=11),
    Node("readout", "③ 一个共同读出", "16 个关节槽 + 未匹配槽 b\n+ 原始历史旋转参数\n内部可用分组 / 稀疏连接\n只输出一份 Δs：51D",
         (1915, 405, 310, 220), True),
    Node("update", "状态更新 → s_k", "s_k = s_prev + Δs\n空事件包原样返回 prev",
         (2335, 430, 230, 170), body_size=11),
    Node("unmatched", "未匹配 / 弱匹配的剩余摘要 b", "共享 φ 的 valid=0 记录\n按剩余权重汇总 mean / max / mass",
         (1040, 705, 355, 140), body_size=11),
]

# Payloads describe data, not gradients. The delayed edge alone closes recurrence.
# Association weights and packet metadata are carried alongside learned features.
EDGES = [
    Edge("prev", "prepare", "完整 s_prev；用于 MANO FK", ((265, 270), (295, 270), (295, 385), (325, 385)), "state"),
    Edge("events", "prepare", "事件 (u,v,t,p)、K、事件掩码与有效数量 N", ((265, 515), (325, 515)), "event"),
    Edge("prepare", "relation", "匹配及剩余记录：事件 token、d、ΔXYZ、z_prior、valid；A,q,N 随记录传递", ((625, 515), (685, 515))),
    Edge("relation", "scatter", "匹配关系特征 φ_ij；直接分配权重 w_ij=q_i*A_ij", ((970, 515), (1025, 515))),
    Edge("scatter", "mesh", "H0；直接事件质量及支持标记（不删除无观测顶点）", ((1230, 515), (1290, 515))),
    Edge("prepare", "mesh", "V_prev 的完整 XYZ 与固定 faces；由此构造固定边和随姿态变化的 ΔXYZ", ((565, 335), (565, 295), (1425, 295), (1425, 405)), "geometry"),
    Edge("mesh", "lbs", "H_L；传播有效性，以及透传的直接事件质量/支持", ((1555, 515), (1615, 515))),
    Edge("lbs", "readout", "16 个有序关节槽：mean、max、直接支持统计、有效性", ((1855, 515), (1915, 515))),
    Edge("relation", "unmatched", "同一 φ 编码的无几何记录；剩余权重 r_i=1-sum_j w_ij；有效事件数量 N", ((830, 625), (830, 775), (1040, 775))),
    Edge("unmatched", "readout", "剩余事件摘要 b（第 17 槽）；mass/validity；透传包计数 N", ((1395, 775), (2145, 775), (2145, 625))),
    Edge("prev", "readout", "prev[3:51] 原始旋转参数，直接作为加性轴角残差的坐标条件；没有独立 prev_mlp", ((265, 220), (2060, 220), (2060, 405)), "state"),
    Edge("readout", "update", "唯一 51D 增量 Δs；透传有效事件数量 N 给确定性的空包门", ((2225, 515), (2335, 515))),
    Edge("prev", "update", "完整 s_prev，用于残差加法和空包保持", ((165, 190), (165, 145), (2450, 145), (2450, 430)), "state"),
    Edge("update", "prev", "s_k 成为下一事件包的 prev；延迟一步，不反馈到本包", ((2450, 600), (2450, 910), (35, 910), (35, 270), (65, 270)), "feedback", True),
]


def on_boundary(point, node):
    px, py = point
    x, y, w, h = node.rect
    return ((px in (x, x + w) and y <= py <= y + h)
            or (py in (y, y + h) and x <= px <= x + w))


def enters_rect(a, b, rect):
    x, y, w, h = rect
    if a[0] == b[0]:
        return x < a[0] < x + w and max(min(a[1], b[1]), y) < min(max(a[1], b[1]), y + h)
    return y < a[1] < y + h and max(min(a[0], b[0]), x) < min(max(a[0], b[0]), x + w)


def validate():
    nodes = {n.key: n for n in NODES}
    assert len(nodes) == len(NODES)
    assert {n.key for n in NODES if n.learned} == {"relation", "mesh", "readout"}
    graph = {key: [] for key in nodes}
    indegree = dict.fromkeys(nodes, 0)
    segments = []
    for idx, edge in enumerate(EDGES):
        assert on_boundary(edge.points[0], nodes[edge.source]), edge
        assert on_boundary(edge.points[-1], nodes[edge.target]), edge
        for a, b in zip(edge.points, edge.points[1:]):
            assert a != b and (a[0] == b[0] or a[1] == b[1]), edge
            for node in NODES:
                assert not enters_rect(a, b, node.rect), (edge, node.key)
            segments.append((idx, a, b))
        if not edge.delayed:
            graph[edge.source].append(edge.target)
            indegree[edge.target] += 1
    queue = deque(k for k in nodes if not indegree[k])
    order = []
    while queue:
        key = queue.popleft()
        order.append(key)
        for target in graph[key]:
            indegree[target] -= 1
            if not indegree[target]:
                queue.append(target)
    assert len(order) == len(nodes), "Unexpected same-packet cycle"
    assert [(e.source, e.target) for e in EDGES if e.delayed] == [("update", "prev")]
    assert {e.source for e in EDGES if e.target == "readout"} == {"lbs", "unmatched", "prev"}
    assert {e.target for e in EDGES if e.source == "unmatched"} == {"readout"}
    # Fail on crossings and overlapping line segments, so no false junctions appear.
    for pos, (idx, a, b) in enumerate(segments):
        for jdx, c, d in segments[pos + 1:]:
            if idx == jdx:
                continue
            av, cv = a[0] == b[0], c[0] == d[0]
            if av != cv:
                v1, v2, h1, h2 = (a, b, c, d) if av else (c, d, a, b)
                cross = (min(h1[0], h2[0]) <= v1[0] <= max(h1[0], h2[0])
                         and min(v1[1], v2[1]) <= h1[1] <= max(v1[1], v2[1]))
            elif av:
                cross = a[0] == c[0] and max(min(a[1], b[1]), min(c[1], d[1])) <= min(max(a[1], b[1]), max(c[1], d[1]))
            else:
                cross = a[1] == c[1] and max(min(a[0], b[0]), min(c[0], d[0])) <= min(max(a[0], b[0]), max(c[0], d[0]))
            assert not cross, ("Ambiguous crossing", EDGES[idx], EDGES[jdx])
    return {"single_packet_topological_order": order,
            "arrows_attach_to_box_boundaries": True,
            "no_edges_cross_boxes_or_each_other": True,
            "learned_modules": ["relation", "mesh", "readout"]}


def text(ax, x, y, value, size=12, color=INK, align="center", bold=False, **kwargs):
    return ax.text(x, y, value, ha=align, va="center", fontsize=size,
                   color=color, weight="bold" if bold else "normal",
                   linespacing=1.5, zorder=6, **kwargs)


def draw_edge(ax, edge):
    color = {"feature": GREEN, "geometry": BLUE, "event": BLUE,
             "state": MUTED, "feedback": GREEN}[edge.kind]
    dashed = edge.kind in {"state", "feedback"}
    width = 1.5 if edge.kind == "state" else 2.0
    for a, b in zip(edge.points[:-2], edge.points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=width,
                linestyle="--" if dashed else "-", zorder=2)
    ax.add_patch(FancyArrowPatch(edge.points[-2], edge.points[-1],
                                arrowstyle="-|>", mutation_scale=16,
                                color=color, lw=width,
                                linestyle="--" if dashed else "-", zorder=2))


def draw_node(ax, node):
    x, y, w, h = node.rect
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0,rounding_size=14",
                                facecolor="#fff7ef" if node.learned else "white",
                                edgecolor=ORANGE if node.learned else BORDER,
                                lw=2.1 if node.learned else 1.4, zorder=3))
    title = text(ax, x + w / 2, y + 34, node.title, 13.5,
                 color=ORANGE if node.learned else INK, bold=True)
    body = text(ax, x + w / 2, y + 65 + (h - 76) / 2,
                node.body, node.body_size, color=INK)
    return [(node, title), (node, body)]


def main():
    verification = validate()
    fig, ax = plt.subplots(figsize=(WIDTH / 100, HEIGHT / 100), dpi=140)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((5, 5), WIDTH - 10, HEIGHT - 10,
                                boxstyle="round,pad=0,rounding_size=24",
                                fc=BG, ec="#e2e7ed", lw=1.2))
    text(ax, 35, 41, "S37–XYZ：推荐保留的学习结构", 22, align="left", bold=True)
    text(ax, 35, 82, "共享关系编码 → 一个 MeshGNN → 一个共同读出；完整保留历史 778 顶点的三维几何", 13, color=MUTED, align="left")
    text(ax, WIDTH - 35, 41, "DEBUG 后设计稿 · 未实现 / 未训练", 14, color=ORANGE, align="right")
    text(ax, WIDTH - 35, 82, "橙框：学习模块   白框：确定性操作   灰虚线：历史状态输入", 11.5, color=MUTED, align="right")

    for edge in EDGES:
        draw_edge(ax, edge)
    text_bounds = []
    for node in NODES:
        text_bounds.extend(draw_node(ax, node))

    text(ax, 1270, 124, "s_prev 直送状态更新：与唯一的 Δs 相加", 12, color=MUTED)
    text(ax, 1210, 198, "原始旋转参数 prev[3:51]：沿用加性轴角更新时，直接作为读出的坐标条件", 12, color=MUTED)
    text(ax, 1060, 273, "完整 V_prev (778×3) + 固定 faces → 网格几何条件", 12, color=BLUE)
    for x, value in [(997, "φ"), (1260, "H0"), (1585, "H_L"), (1885, "e"), (2280, "Δs")]:
        text(ax, x, 494, value, 11, color=GREEN)
    text(ax, 475, 659, "完整三维几何；不做网格投影", 11, color=BLUE)
    text(ax, 925, 756, "剩余记录", 11, color=GREEN)
    text(ax, 1770, 752, "b：第 17 个摘要槽 → 同一个读出", 12, color=GREEN)
    text(ax, 1770, 809, "绕过 MeshGNN；没有独立姿态预测头", 11.5, color=MUTED)
    text(ax, 80, 718, "保留三维关系", 14, align="left", bold=True, color=BLUE)
    text(ax, 80, 778, "z_prior 来自历史网格，并带 valid。\n无匹配不伪造深度，信息进入剩余摘要 b。\n几何边保留 ΔX、ΔY、ΔZ。", 12, align="left", color=MUTED)
    text(ax, 1315, 887, "跨包反馈：s_k 仅作为下一包的 prev（延迟一步）", 12.5, color=GREEN)

    ax.plot([40, WIDTH - 40], [956, 956], color=BORDER, lw=1)
    text(ax, 50, 984, "学习输出只有一份 Δs；mean / max / 支持统计是池化内容，不是多套预测分支。", 12, align="left")
    text(ax, 50, 1020, "本候选不设独立 EventGNN、深度预测头或 prev_mlp；其精度与容量仍需后续训练验证。", 11.5, align="left", color=MUTED)
    text(ax, WIDTH - 50, 999, "箭头表示前向数据；不表示梯度。\n事件有效数量 N 随包传递至空包门。", 11, color=MUTED, align="right")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for node, artist in text_bounds:
        box = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        x, y, w, h = node.rect
        assert box.x0 >= x + 5 and box.x1 <= x + w - 5, (node.key, artist.get_text(), box)
        assert min(box.y0, box.y1) >= y + 5 and max(box.y0, box.y1) <= y + h - 5, (node.key, artist.get_text(), box)
    verification["node_text_inside_boxes"] = True
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = OUT.with_suffix("." + ext)
        fig.savefig(path, dpi=140, facecolor="white")
        print(path)
    manifest = {
        "status": "unimplemented_untrained_design",
        "basis": "docs/S37_XYZ_CANDIDATE_20260929.md",
        "state_contract": "S37 additive 51D state; raw prev[3:51] conditions the common readout",
        "nodes": [asdict(n) for n in NODES],
        "edges": [asdict(e) for e in EDGES],
        "diagram_verification": verification,
    }
    OUT.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(OUT.with_suffix(".json"))
    plt.close(fig)


if __name__ == "__main__":
    main()
