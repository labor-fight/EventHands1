#!/usr/bin/env python3
"""The accepted XYZ design in the user's S37 horizontal-strip template.

Only presentation changes. Reuses the original template's icons, nodes and arrows.
No experiment metrics, model changes, or training.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

from make_s36_figure_simple import (
    BLUE, CAP, DARK, EDGE, PANEL, TEAL, cube_icon, elbow, events_icon,
    funnel_icon, graph_icon, hand_skel, node, stack_icon,
)
from make_s37_figure_simple import ORANGE, hnode, query_icon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets/s37_xyz_recommended_template_20260929"
plt.rcParams["svg.fonttype"] = "none"

# x, y, width, height, two-line label, icon, icon colour, learned
BOXES = {
    "prev": (4, 21, 17, 10, "prev 51D\nMANO", hand_skel, TEAL, False),
    "fk": (25, 34.8, 16, 10, "MANO FK", hand_skel, TEAL, False),
    "xyz": (44, 34.8, 18, 10, "778 顶点\n(X, Y, Z)", graph_icon, TEAL, False),
    "associate": (65, 34.8, 24, 10, "三维射线关联\nz_prior / valid", query_icon, BLUE, False),
    "events": (56, 21, 23, 10, "事件 (u,v,t,p)\n相机内参 K", events_icon, BLUE, False),
    "relation": (93, 21, 21, 10, "共享关系编码\n事件 + 三维关系", stack_icon, ORANGE, True),
    "scatter": (117, 21, 18, 10, "顶点聚合\n778 个节点", stack_icon, BLUE, False),
    "mesh": (138, 21, 20, 10, "MeshGNN\n固定 1-ring", cube_icon, BLUE, True),
    "lbs": (161, 21, 21, 10, "LBS 池化\n16 个关节槽", stack_icon, BLUE, False),
    "readout": (185, 21, 22, 10, "共同读出\n16 槽 + b", funnel_icon, ORANGE, True),
    "delta": (210, 21, 6, 10, "Δ", None, DARK, False),
    "output": (224, 21, 12, 10, "x_k =\nprev + Δ", None, DARK, False),
    "unmatched": (139, 8.5, 25, 7, "剩余摘要 b", stack_icon, BLUE, False),
}

# Expanded deterministic preparation/update, same accepted learning graph.
# Paths stop 0.4 units from nominal rectangles, matching template's 0.35 padding.
EDGES = [
    ("prev", "fk", [(16, 31.4), (16, 39.8), (24.6, 39.8)]),
    ("fk", "xyz", [(41.4, 39.8), (43.6, 39.8)]),
    ("xyz", "associate", [(62.4, 39.8), (64.6, 39.8)]),
    ("events", "associate", [(76, 31.4), (76, 34.4)]),
    ("events", "relation", [(79.4, 26), (92.6, 26)]),
    ("associate", "relation", [(89.4, 39.8), (103.5, 39.8), (103.5, 31.4)]),
    ("relation", "scatter", [(114.4, 26), (116.6, 26)]),
    ("scatter", "mesh", [(135.4, 26), (137.6, 26)]),
    ("xyz", "mesh", [(53, 45.2), (53, 47.3), (148, 47.3), (148, 31.4)]),
    ("mesh", "lbs", [(158.4, 26), (160.6, 26)]),
    ("lbs", "readout", [(182.4, 26), (184.6, 26)]),
    ("relation", "unmatched", [(103.5, 20.6), (103.5, 12), (138.6, 12)]),
    ("unmatched", "readout", [(164.4, 12), (196, 12), (196, 20.6)]),
    ("prev", "readout", [(9, 31.4), (9, 51.3), (196, 51.3), (196, 31.4)]),
    ("readout", "delta", [(207.4, 26), (209.6, 26)]),
    ("delta", "sum", [(216.4, 26), (218.2, 26)]),
    ("prev", "sum", [(12.5, 20.6), (12.5, 5.5), (220, 5.5), (220, 24.2)]),
    ("sum", "output", [(221.8, 26), (223.6, 26)]),
    ("output", "prev", [(230, 20.6), (230, 1.8), (2.2, 1.8), (2.2, 26), (3.6, 26)]),
]


def validate():
    """Check the template routing and retained semantic dependencies."""
    from draw_s37_xyz_recommended import enters_rect

    segments = []
    for idx, (source, target, path) in enumerate(EDGES):
        for a, b in zip(path, path[1:]):
            assert a != b and (a[0] == b[0] or a[1] == b[1])
            for key, box in BOXES.items():
                x, y, w, h = box[:4]
                assert not enters_rect(a, b, (x - 0.35, y - 0.35, w + 0.7, h + 0.7)), (source, target, key)
            segments.append((idx, a, b))
    for i, (idx, a, b) in enumerate(segments):
        for jdx, c, d in segments[i + 1:]:
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
            assert not cross, (EDGES[idx], EDGES[jdx])
    assert {key for key, box in BOXES.items() if box[-1]} == {"relation", "mesh", "readout"}
    assert {a for a, b, _ in EDGES if b == "readout"} == {"lbs", "unmatched", "prev"}
    assert {b for a, b, _ in EDGES if a == "unmatched"} == {"readout"}
    # Compare against the accepted diagram after contracting display-only boxes.
    from draw_s37_xyz_recommended import EDGES as ACCEPTED
    group = {"fk": "prepare", "xyz": "prepare", "associate": "prepare",
             "delta": "update", "sum": "update", "output": "update"}
    contracted = {(group.get(a, a), group.get(b, b)) for a, b, _ in EDGES
                  if group.get(a, a) != group.get(b, b)}
    # Event tokens are passed through preparation in the detailed view, and are
    # explicitly drawn as a direct input in this expanded preparation view.
    contracted.remove(("events", "relation"))
    assert contracted == {(e.source, e.target) for e in ACCEPTED}
    return {"no_line_crossings": True, "no_lines_through_boxes": True,
            "matches_accepted_functional_graph": True}


def main():
    checks = validate()
    fig, ax = plt.subplots(figsize=(240 / 14, 58 / 14))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set(xlim=(0, 240), ylim=(0, 58))
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.8, 0.7), 238.4, 56.6,
                                boxstyle="round,pad=0.3,rounding_size=2.5",
                                facecolor=PANEL, edgecolor="#e5e7eb", lw=1.0, zorder=0))
    ax.text(4, 55.3, "S37–XYZ  完整 778 顶点三维几何 + 事件关系编码",
            fontsize=8.3, color=CAP, va="center", zorder=6)
    ax.text(236, 55.3, "橙框：3 个学习模块  |  无网格投影  |  设计稿 · 未训练",
            fontsize=7.6, color=ORANGE, ha="right", va="center", zorder=6)
    for _, _, path in EDGES:
        elbow(ax, path, lw=1.45)

    labels = []
    for key, box in BOXES.items():
        x, y, w, h, label, icon, colour, learned = box
        draw = hnode if learned else node
        draw(ax, x, y, w, h, label, icon, colour, fs=13 if key == "delta" else 8.1)
        labels.append((box, ax.texts[-1]))
    ax.add_patch(Circle((220, 26), 1.7, facecolor="white", edgecolor=TEAL, lw=1.4, zorder=5))
    ax.text(220, 26, "+", ha="center", va="center", fontsize=11, color=TEAL, zorder=6)

    for x, y, value, size in [
        (152, 52.6, "prev 旋转参数", 6.7),
        (119, 48.6, "XYZ / 三维边", 6.7),
        (43, 6.8, "prev", 6.7),
        (122, 3.0, "下一包 prev", 6.4),
    ]:
        ax.text(x, y, value, ha="center", va="center", fontsize=size, color=CAP, zorder=6)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for box, artist in labels:
        x, y, w, h = box[:4]
        bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        assert bounds.x0 >= x + 0.2 and bounds.x1 <= x + w - 0.2, (artist.get_text(), bounds)
        assert bounds.y0 >= y + 0.2 and bounds.y1 <= y + h - 0.2, (artist.get_text(), bounds)
    checks["labels_fit_boxes"] = True

    OUT.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = OUT.with_suffix("." + ext)
        fig.savefig(path, dpi=200, facecolor="white")
        print(path)
    OUT.with_suffix(".json").write_text(json.dumps({
        "status": "untrained_design_presentation_revision",
        "detailed_diagram": "s37_xyz_recommended_20260929.json",
        "template": "tools/make_s37_figure_simple.py",
        "arrows": [{"source": a, "target": b, "path": p,
                    "delayed": (a, b) == ("output", "prev")} for a, b, p in EDGES],
        "verification": checks,
    }, ensure_ascii=False, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
