#!/usr/bin/env python3
"""PFO module contribution / cost table (the goal's "模块贡献 / 成本表"): one row per stage of the pfasrdz chain with its
parameters, executed MACs, synchronised stage latency and the accuracy change when the stage is removed or replaced,
separating (a) frozen-weight test-time interventions (phase 1 D3: co-adaptation, "what the trained net uses") from
(b) retrained arms (phase 2a / 2r / 2b / 2c: the deployable contribution). CPU only, read-only on the readings.

    python -m tools.pfo.modtable [--cost TAG] [--out FILE]

Cost columns come from tools/pfo/cost.py's reading of the given tag (default: the phase-1 pfasrdz fast / default-flag
reading; the final block's `final__pfasrdz__fast__default` is used when it exists). Router MACs of the slim routers
are the registered analytic figures (docs/EXPERIMENT_RECORDS.md#pfo-p2r-prereg-20261103, #pfo-p2r2-prereg-20261103,
#pfo-p3d-prereg-20261103); the executed MACs of the deployed router come from the cost reading.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo.analyze import fmt, rep, row5  # noqa: E402


def sfmt(v):
    return f"{float(v):+.2f}"
from tools.pfo.arms import run_name  # noqa: E402
from tools.pfo.common import OUT, read  # noqa: E402

A = OUT / "analysis"
COST_DEFAULT = ("final__pfasrdz__fast__default", "pfasrdz__fast__default")
S43_COST = ("final__s43_s3407__exec__default", "s43_s3407__exec__default")
ROUTER_MACS_M = {"sr1": 180.7, "t0p": 1200.0, "r_s2d": 74.5, "r_s2dkd": 74.5, "r_s2d_nh": 74.5, "r_s2dw": 82.0,
                 "r_wide": 162.0, "r_s2dns": None}
ROUTER_PARAMS = {"sr1": 137028, "r_s2d": 137252, "r_s2dkd": 137252, "r_s2d_nh": 137252, "r_s2dw": 246356,
                 "r_wide": 1194452}

# stage -> (parameter groups, MAC groups, breakdown timers)
STAGES = [
    ("事件表示（token 4 通道图 + 记录）", (), (), ("tokens",)),
    ("MANO FK（前态网格；S43 也算）", (), ("mano_fk",), ("mano_fk",)),
    ("SRNet 路由器：事件图 + CNN / FPN", ("router.stem", "router.stages", "router.lateral", "router.smooth"),
     ("router.trunk(dense)",), ("router_image", "router_cnn")),
    ("SRNet 头 + 模板路由解码", ("router.head", "router.part", "router.coord", "router.onhand", "router.patch"),
     ("router",), ("route_decode",)),
    ("PFA 关联（放到路由面，取前态深度）", (), (), ("pfa",)),
    ("事件—顶点关系编码 φ（13 维 → 64）", ("xyz.rel",), ("xyz.rel",), ("phi",)),
    ("逐顶点聚合（均值 / 最大 / 质量）+ vin + 边特征", ("xyz.vin", "xyz.edge"), ("xyz.vin", "xyz.edge"), ("vin", "edge")),
    ("MeshGNN（3 层消息传递）", ("xyz.layers",), ("xyz.layers",), ("gnn",)),
    ("LBS 槽池化", ("xyz.slot_proj",), ("xyz.slot_proj",), ("slot_pool",)),
    ("读出 + 51 维状态增量", ("xyz.fc1", "xyz.fc2"), ("xyz.fc1", "xyz.fc2"), ("readout",)),
]


def first_cost(tags):
    for t in tags:
        p = OUT / "costs" / t / "report.json"
        if p.exists():
            return t, read(p)
    return None, None


def d3(tag):
    """Phase-1 frozen-weight intervention: (dE1 local, dE1 global, dE1 RA, dE3 RA) or None."""
    p1 = read(A / "phase1.json")
    e = p1["d3"].get(tag)
    if not e:
        return None
    de1 = e.get("d_e1") or (None, None, None)     # None: the E1 reading diverged (agg_max_only, failure preserved)
    return tuple(de1) + (e.get("d_e3"),)


def mean5(arm, seeds):
    rows = [row5(rep(run_name(arm, s), "e1")) for s in seeds]
    rows = [r for r in rows if r]
    return (np.mean(rows, 0), len(rows)) if rows else (None, 0)


def swap(trunk, router):
    r = rep(f"{trunk}+{router}", "e1")
    r3 = rep(f"{trunk}+{router}", "e3")
    return (row5(r)[4] if r else None, r3["overall"]["mpjpe_ra_mm"] if r3 else None)


def ladder(tag):
    r = rep(tag, "e1")
    return row5(r)[4] if r else None


def table(cost_tag=None):
    tag, c = first_cost((cost_tag,) if cost_tag else COST_DEFAULT)
    stag, s = first_cost(S43_COST)
    pg, mm, bd = c["params_by_group"], c["macs_by_module_mean"], c["breakdown_ms_mean"]
    fk = c["macs_mano_fk"]
    total_macs = c["macs_layers"]["median"]
    rows, js = [], {"cost_reading": tag, "s43_cost_reading": stag, "stages": []}
    for name, pgs, mgs, tms in STAGES:
        params = sum(pg.get(g, 0) for g in pgs)
        macs = sum((fk if g == "mano_fk" else mm.get(g, 0.0)) for g in mgs)
        ms = sum(bd.get(t, 0.0) for t in tms)
        rows.append((name, params, macs, ms))
        js["stages"].append({"stage": name, "params": params, "macs": macs, "latency_ms": ms})
    base = row5(rep("pfasrdz", "e1"))
    p1 = read(A / "phase1.json")
    lines = ["## 模块贡献 / 成本表（pfasrdz，lyq_local 600 包；成本读数 `%s`）" % tag, "",
             "成本：参数 / 逐包执行 MACs 中位（层计数；MANO FK 为解析值）/ 同步分段时延均值（含计时开销）。"
             "精度：冻结权重的测试时干预（第 1 阶段 D3，共适应）与重训臂（第 2 阶段，可部署贡献），ΔE1 RA 为对 pfasrdz（或对照臂）的差，正 = 变差。",
             "",
             "| 阶段 | 参数 | MACs | 分段时延 | 冻结干预 ΔE1 L / G / RA（ΔE3 RA） | 重训 / 零训练替换（可部署贡献） |",
             "|---|---|---|---|---|---|"]

    def iv(*tags):
        out = []
        for t in tags:
            d = d3(f"pfasrdz__iv-{t}")
            g = lambda v: "发散" if v is None else fmt(v)  # noqa: E731
            out.append(f"{t} {g(d[0])} / {g(d[1])} / {g(d[2])}（{g(d[3])}）" if d else f"{t} –")
        return "；".join(out)

    # retrained / zero-training evidence per stage
    b3, nb = mean5("b", (3407, 3408, 3409))
    nomp3, nn = mean5("nomp", (3407, 3408, 3409))
    self2, ns = mean5("self", (3407, 3408))
    rew2, nr = mean5("rewire", (3407, 3408))
    nohot1, _ = mean5("nohot", (3407,))
    gnn_txt = "–"
    if b3 is not None and nomp3 is not None:
        gnn_txt = (f"重训去掉消息传递 nomp − b（{nn} / {nb} 种子）：RA {sfmt(nomp3[4] - b3[4])}，local "
                   f"{sfmt(nomp3[0] - b3[0])}，global {sfmt(nomp3[1] - b3[1])}")
        if self2 is not None and rew2 is not None:
            gnn_txt += (f"；同参数无邻居 self / 乱边 rewire（2 种子）RA {fmt(self2[4])} / {fmt(rew2[4])}，"
                        f"b 2 种子 {fmt(np.mean([row5(rep(run_name('b', s), 'e1'))[4] for s in (3407, 3408)]))}")
    rt = {}
    for r in ("r_s2d", "r_s2dkd", "r_s2dw", "r_wide", "r_s2d_nh"):
        e1, e3 = swap("pfasrdz", r)
        if e1 is not None:
            rt[r] = (e1, e3)
    router_txt = "零训练换路由器（pfasrdz 主干，E1 RA / E3 RA；sr1 %s / %s）：" % (fmt(base[4]), fmt(p1["pfasrdz"]["e3"][4]))
    router_txt += "；".join(f"{r}（{ROUTER_MACS_M[r]:.0f} M）{fmt(v[0])} / {fmt(v[1])}" for r, v in rt.items()) or "–"
    lad = {k: ladder(f"pfasrdz__{k}") for k in ("label", "t0p", "label_loc", "label_part", "router_shuffle")}
    route_txt = ("理想路由阶梯（E1 RA）：标签 %s、T0P %s、只修定位 %s、只修部位 %s、包内打乱 %s" %
                 tuple(fmt(lad[k]) for k in ("label", "t0p", "label_loc", "label_part", "router_shuffle")))
    evidence = {
        0: (iv("tok_xy_mean", "tok_time_mean", "tok_pol_zero"),
            f"训练不注入热像素 nohot（种子 3407，对 b）：RA {sfmt(nohot1[4] - row5(rep(run_name('b', 3407), 'e1'))[4])}"
            if nohot1 is not None else "–"),
        1: ("–", "无可去除（前态网格是关联与池化的几何）；S43 的 render_prev 同样含 FK"),
        2: ("–", router_txt),
        3: ("–", route_txt + "；τ 0.3–0.9 平坦"),
        4: ("–", "S45-SM 第 1 轮：PFA 优于射线关联 face_route（两种路由器下），`#s45sm-r1-results-20261031`；第 2k 显式几何更新 KILL"),
        5: (iv("dxyz_zero", "dxyz_shuffle", "zn_const"), "位移通道只在闭环纠错时承重（单步 −0.3）；深度通道 +0.42"),
        6: (iv("agg_mean_only", "agg_max_only", "agg_mass_zero"), "均值与最大值两路都承重；质量通道可去（−0.48）"),
        7: (iv("gnn_skip", "gnn_rewire"), gnn_txt),
        8: ("–", "无单独对照（槽池化 = 读出的输入接口）"),
        9: (iv("prev_fing_mean", "resid_zero"), "读出强依赖前态；残差槽置零 local −1.38 / global +0.79"),
    }
    for i, (name, params, macs, ms) in enumerate(rows):
        a, b = evidence[i]
        lines.append(f"| {name} | {params:,} | {macs / 1e6:.1f} M | {ms:.2f} ms | {a} | {b} |")
    lines += ["", f"合计：参数 {c['params_total']:,}；执行 MACs 中位 {total_macs / 1e6:.1f} M（+ MANO FK {fk / 1e6:.2f} M）；"
              f"一步 {bd.get('step_total', float('nan')):.2f} ms（主行口径 {c['latency_row']['scaled_ms']:.2f} ms，"
              f"端到端 p50 / p99 {c['latency_e2e_ms']['p50']:.2f} / {c['latency_e2e_ms']['p99']:.2f} ms）；"
              f"一步峰值显存 {c['memory_peak_step_mib']:.1f} MiB（权重 {c['memory_weights_mib']:.1f} MiB）。"]
    if s is not None:
        sb = s["breakdown_ms_mean"]
        lines += [f"S43 稀疏执行器（`{stag}`）：参数 {s['params_total']:,}；执行 MACs 中位 {s['macs_layers']['median'] / 1e6:.1f} M"
                  f"（掩码稠密 {s['macs_alt_s43']['median'] / 1e6:.1f} M）；分段 render_prev {sb.get('render_prev', 0):.2f}、"
                  f"sparse_trunk {sb.get('sparse_trunk', 0):.2f}、prev_mlp {sb.get('prev_mlp', 0):.2f}、一步 {sb.get('step_total', 0):.2f} ms；"
                  f"端到端 p50 / p99 {s['latency_e2e_ms']['p50']:.2f} / {s['latency_e2e_ms']['p99']:.2f} ms；"
                  f"峰值显存 {s['memory_peak_step_mib']:.1f} MiB。"]
    lines += ["", "### 按收益与成本排序（读法）", "",
              "1. MeshGNN：MACs 最大的一块（%.0f %% 的执行 MACs、%.2f ms），重训去掉后 RA 不变 → 成本高、贡献小，去掉。" %
              (100 * mm.get("xyz.layers", 0) / total_macs, bd.get("gnn", 0)),
              "2. 路由器稠密主干：第二大（%.0f %%），像素重排 stem 的 r_s2d 质量等价、MACs 减半以上；蒸馏 / 加宽在网络层面更好但伪标签指标更差。" %
              (100 * mm.get("router.trunk(dense)", 0) / total_macs),
              "3. φ / vin：MACs 第三（%.0f %%），干预显示位移与深度通道承重，像素 / 时间 / 极性列可忽略；未做结构替换。" %
              (100 * (mm.get("xyz.rel", 0) + mm.get("xyz.vin", 0)) / total_macs),
              "4. MANO FK + token：约 %.1f ms 时延、几乎无 MACs，两种网络共有，不是可省项。" % (bd.get("mano_fk", 0) + bd.get("tokens", 0)),
              "5. 槽池化 / 读出：参数最多（%.0f %%）但 MACs 可忽略；读出依赖前态是闭环保留率的所在。" %
              (100 * (pg.get("xyz.slot_proj", 0) + pg.get("xyz.fc1", 0) + pg.get("xyz.fc2", 0)) / c["params_total"])]
    js["rows"] = [{"stage": n, "params": p_, "macs": m, "latency_ms": t, "frozen": evidence[i][0], "retrained": evidence[i][1]}
                  for i, (n, p_, m, t) in enumerate(rows)]
    return lines, js


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--cost")
    p.add_argument("--out")
    a = p.parse_args(argv)
    lines, js = table(a.cost)
    text = "\n".join(lines) + "\n"
    out = Path(a.out) if a.out else A / "modules.md"
    out.write_text(text)
    out.with_suffix(".json").write_text(json.dumps(js, indent=1, ensure_ascii=False) + "\n")
    print(text)
    print("wrote", out)


if __name__ == "__main__":
    main()
