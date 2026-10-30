"""Expanded horizontal XAG F+ drawing; documentation assets only."""
from pathlib import Path
import importlib.util

HERE = Path(__file__).resolve().parent
HELPERS = HERE.parent / "xag_gnn_design_20261017" / "generate.py"
spec = importlib.util.spec_from_file_location("xag_figure_helpers", HELPERS)
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
box, label, arrow, route = G.box, G.label, G.arrow, G.route


def generate():
    fig, ax = G.base(width=24, height=13.4, ylim=15.2)
    label(ax, .25, 14.72, "增强 GNN（F+）：保留局部传播，新增跨部位交流与特征重组", size=23,
          ha="left", weight="bold")
    label(ax, .25, 14.11, "蓝色＝原模块    橙色＝新增模块    特征接口始终为 778 个顶点 × 128 维；省略 batch 维",
          size=14.2, ha="left", color=G.MUTED)

    # Keep the approved two-row overview and the common readout on the right.
    for y, name, enhanced in ((11.80, "F0｜原 GNN", False), (9.10, "F+｜增强 GNN", True)):
        label(ax, .25, y + 1.02, name, size=18, ha="left", weight="bold")
        box(ax, .25, y - .43, 1.35, .86, "H0\n778 × 128", "neutral", 12.1)
        last = 1.60
        for i, x in enumerate((2.00, 6.65, 11.30), 1):
            arrow(ax, last, y, x, y)
            box(ax, x, y - .49, 1.65, .98, f"原局部层 {i}\n1-ring", size=12.7)
            last = x + 1.65
            if enhanced:
                nx = x + 1.95
                arrow(ax, last, y, nx, y, color=G.ORANGE)
                box(ax, nx, y - .77, 2.35, 1.54,
                    f"新增块 {i}\nLN＋4头全局注意力\nLN＋FFN：128→256→128\n两条残差＋支持掩码",
                    "new", 11.7)
                last = nx + 2.35
        arrow(ax, last, y, 16.00, y)
        box(ax, 16.00, y - .43, 1.25, .86, "H3\n778 × 128", "neutral", 12.0)
        arrow(ax, 17.25, y, 18.02, y)

    box(ax, 18.02, 8.23, 5.60, 4.20,
        "原共同读出（F0 / F+ 相同结构）\n"
        "LBS pooling：16 × 258\n＋原事件残差槽：1 × 258\n"
        "逐槽压缩：17 × 258 → 17 × 32\n"
        "展平 544；拼接前态 48\n"
        "原 MLP：592 → 256 → 51\n输出状态增量 delta", size=12.8)
    box(ax, 19.10, 12.89, 3.40, .60, "固定 LBS：778 × 16", "neutral", 12.6)
    arrow(ax, 20.80, 12.89, 20.80, 12.43)
    box(ax, 17.98, 6.96, 2.82, .79, "原事件残差支路\n未 / 弱匹配事件", size=11.1)
    arrow(ax, 19.39, 7.75, 19.39, 8.23)
    box(ax, 21.12, 6.96, 2.50, .79, "原前态 prev48\n48 维", size=11.8)
    arrow(ax, 22.37, 7.75, 22.37, 8.23)
    label(ax, .25, 7.63,
          "三个新增块各有独立参数；局部层仍负责扩张支持 S。\n"
          "A0F1 的事件关联使用 A0；原关系编码与顶点聚合仍生成同形状 H0。",
          size=13.2, color=G.MUTED, ha="left")

    # Expand one representative block, labeling the purpose of every operation.
    ax.plot([.25, 23.65], [6.56, 6.56], color=G.EDGE, lw=1)
    label(ax, .25, 6.13, "放大任意一个橙色块：新增 ①—⑦ 的做法与目的", size=18,
          ha="left", weight="bold", color=G.ORANGE)
    cy = 3.87
    box(ax, .23, cy - .46, 1.68, .92, "局部输出 U\n778 × 128", "neutral", 12.0)
    box(ax, 2.28, cy - .47, 1.48, .94, "① LN1\nZ＝LN1(U)", "new", 12.2)
    box(ax, 4.35, cy - .68, 3.15, 1.36,
        "③ 4头全局注意力\nQ/K 输入：Z＋P\nV 输入：Z；每头32维", "new", 12.8)
    box(ax, 8.00, cy - .55, 2.32, 1.10, "④ 残差＋mask\nH1＝M × (U＋A)", "new", 12.0)
    box(ax, 10.88, cy - .47, 1.50, .94, "⑤ LN2\nLN2(H1)", "new", 12.3)
    box(ax, 12.98, cy - .60, 3.10, 1.20,
        "⑥ 逐顶点 FFN\n128 → 256 → GELU → 128", "new", 12.4)
    box(ax, 16.63, cy - .55, 2.85, 1.10,
        "⑦ 残差＋mask\nHout＝M × (H1＋F)", "new", 12.3)
    box(ax, 20.10, cy - .46, 3.50, .92,
        "Hout：778 × 128\n送入下一原局部层 / 读出", "neutral", 12.3)
    for left, right in ((1.91, 2.28), (3.76, 4.35), (7.50, 8.00), (10.32, 10.88),
                        (12.38, 12.98), (16.08, 16.63), (19.48, 20.10)):
        arrow(ax, left, cy, right, cy, color=G.ORANGE)

    box(ax, 3.80, 5.02, 4.25, .68,
        "② 固定 LBS → 新 Linear(16→128) → P", "new", 12.3)
    arrow(ax, 5.92, 5.02, 5.92, 4.55, color=G.ORANGE)
    label(ax, 8.35, 5.33, "目的：提示部位归属，只加入 Q/K 输入", size=13.2,
          color=G.ORANGE, ha="left")

    # Two exact residual paths, kept clear of the main operation chain.
    route(ax, [(1.91, cy), (2.09, cy), (2.09, 1.88), (9.16, 1.88), (9.16, cy - .55)], color=G.BLUE)
    label(ax, 5.56, 1.65, "保留原局部特征 U", size=12.4, color=G.BLUE)
    route(ax, [(10.32, cy), (10.60, cy), (10.60, 2.00), (18.05, 2.00), (18.05, cy - .55)], color=G.ORANGE)
    label(ax, 14.28, 1.77, "保留注意力融合特征 H1", size=12.2, color=G.ORANGE)

    # Short labels directly under the corresponding modifications.
    for x, text in ((3.02, "统一通道尺度"), (5.92, "跨已有支持顶点读信息"),
                    (9.16, "在局部表示上加补充"), (11.63, "整理融合后的尺度"),
                    (14.53, "逐点非线性重组通道"), (18.05, "在融合表示上再加工")):
        label(ax, x, 4.72 if x in (9.16, 18.05) else 2.74,
              text, size=11.6, color=G.MUTED)

    label(ax, .25, .91,
          "M＝1[S>0]：只读取有支持顶点；无支持输出清零。掩码不扩大支持，也不判断证据是否正确。",
          size=13.3, ha="left", color=G.MUTED)
    label(ax, .25, .35,
          "目的：在槽压缩前提前融合远处证据。①⑤便于优化，②辅助选取读取关系，③跨点交流，⑥通道重组，④⑦保留直通路径。",
          size=12.9, ha="left", color=G.INK)

    fig.savefig(HERE / "overview_expanded_horizontal.svg", facecolor="white")
    fig.savefig(HERE / "overview_expanded_horizontal.png", dpi=130, facecolor="white")
    G.plt.close(fig)


if __name__ == "__main__":
    generate()
