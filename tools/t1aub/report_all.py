#!/usr/bin/env python3
"""T1A-UB final deliverable tables (CPU only; descriptive: compiles the frozen rounds' readings and summaries, decides nothing).

    python tools/t1aub/report_all.py [--out FILE]     -> outputs/t1aub/20261102/final_tables.md (default) + .json

Sections: the nine-column main table (every measured row, by block); the upper-bound ladder (label-assisted diagnostic
bounds / same inputs over budget / in-budget deployable / baseline) with single-step, in-sample, correct-state keeping,
wrong-state recovery and retention; the probes (event-level part accuracy, slot-level information); costs (parameters,
MACs, mean and tail latency, stage breakdown); the written decisions of every round. Missing pieces are shown as '-'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402
from tools.t1aub import decide as D  # noqa: E402
from tools.t1aub import r2, r5  # noqa: E402

BASELINE_ROW = "| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |"
HEADER = ["| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |",
          "|---|---|---|---|---|---|---|---|---|"]


def kind(net):
    """baseline | label (label-assisted diagnostic) | over (same inputs, over budget) | deployable."""
    if net in ("t1a", "t1arep"):
        return "baseline"
    if r5.is_r5(net):
        return "label"
    arms = C.registry()["arms"]
    if net in arms:
        s = arms[net]
        if s.get("gtfeat") or (s.get("route") or {}).get("router") == "gt":
            return "label"
        if (s.get("route") or {}).get("router") == "t0p":
            return "over"
        return "deployable"
    return "deployable"


def networks():
    reg, reg2, reg5 = C.registry(), r2.registry(), r5.registry()
    out = ["t1a"]
    out += [a for a in reg["arms"] if "formal" in C.phases(a)]
    out += list(reg["cells"]) + list(reg2["cells"]) + list(reg5["cells"])
    return out


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception:  # noqa: BLE001 (a reading not produced yet)
        return None


def rows_of(net):
    """[(block, row_metrics, cost)] of every main row of a network (T1a: one per block, `t1aref_<B>`)."""
    out = []
    names = sorted(p.name for p in (C.OUT / "evaluations").glob("t1aref_*")) if net == "t1a" else [net]
    for name in names:
        f = C.OUT / "evaluations" / name / "row_step6000/report.json"
        if not f.is_file():
            continue
        r = C.read(f)
        row = C.read(r["main_row"]["path"])
        out.append((r["block"], D.row_metrics(row), r["cost"]))
    return out


def readings(net):
    z = safe(D.zgz, net)
    ins = safe(D.insample, net)
    return {"zgz": z, "insample": {s: D.med(v) for s, v in ins.items()} if ins else None,
            "fixed_point": safe(D.fixed_point, net), "single_step": safe(D.single_step, net),
            "retention": safe(D.retention, net), "rows": rows_of(net), "kind": kind(net)}


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def main_table(R):
    lines = HEADER + [BASELINE_ROW]
    by_block = {}
    for n, r in R.items():
        for block, rm, cost in r["rows"]:
            by_block.setdefault(block, []).append((n, rm, cost))
    for block in sorted(by_block, reverse=True):
        for n, rm, cost in by_block[block]:
            label = ("S45-T1a（基线，同块重测）" if n == "t1a" else f"T1A-UB {n}") + f" [{block}]"
            lines.append(D.fmt_row(label, rm, cost))
    return lines


def ladder(R):
    order = {"label": 0, "over": 1, "deployable": 2, "baseline": 3}
    names = {"label": "标签辅助诊断上界", "over": "相同输入、超预算（T0P 路由器 1.6 M）", "deployable": "预算内可部署（结构）",
             "baseline": "基线"}
    lines = ["| 类别 | 网络 | zgz E1@6000 local / global | E3 local / global | 样本内 5000 / 5500 / 6000 | C（均值起点） "
             "| GT 起点 RA@1 → @20 | 闭环起点 RA@1 → @20 | 保留率 T10 / R5 / Fs |", "|---|---|---|---|---|---|---|---|---|"]
    for n, r in sorted(R.items(), key=lambda kv: (order[kv[1]["kind"]], kv[0])):
        z, ins, fp, ret = r["zgz"], r["insample"], r["fixed_point"], r["retention"]
        e1 = f"{f2(z['e1'][6000]['local'])} / {f2(z['e1'][6000]['global'])}" if z else "-"
        e3 = f"{f2(z['e3']['local'])} / {f2(z['e3']['global'])}" if z else "-"
        s = " / ".join(f2(ins[k]) for k in C.STEPS) if ins else "-"
        c = f2(fp["C"]) if fp else "-"
        g = f"{f2(fp['gt']['ra1'])} → {f2(fp['gt']['ra20'])}" if fp else "-"
        lp = f"{f2(fp['loop']['ra1'])} → {f2(fp['loop']['ra20'])}" if fp else "-"
        rt = f"{f2(ret['T10'])} / {f2(ret['R5'])} / {f2(ret['Fs'])}" if ret else "-"
        lines.append(f"| {names[r['kind']]} | {n} | {e1} | {e3} | {s} | {c} | {g} | {lp} | {rt} |")
    return lines


def costs(R):
    lines = ["| 网络 [块] | 参数 模型 + 路由器 = 合计 | MACs | 时延均值 | 尾延迟 p50 / p90 / p99 / max | 阶段（scaled ms/包） |",
             "|---|---|---|---|---|---|"]
    for n, r in R.items():
        for block, _, cost in r["rows"]:
            t = cost["tail_scaled_ms"]["per_packet_min_over_passes_ms"]
            name = f"t1aref_{block}" if n == "t1a" else n
            bd = C.OUT / "evaluations" / name / "breakdown/report.json"
            st = "-"
            if bd.is_file():
                b = C.read(bd)
                if b.get("block") == block:
                    st = ", ".join(f"{k} {v:.2f}" for k, v in b["stages_scaled_ms_per_packet"].items())
            lines.append(f"| {n} [{block}] | {cost['params_model']:,} + {cost['params_router']:,} = {cost['params_total']:,} "
                         f"| {cost['macs_total'] / 1e9:.3f} G | {cost['latency_ms']:.2f} ms | {t['p50']:.2f} / {t['p90']:.2f} / "
                         f"{t['p99']:.2f} / {t['max']:.2f} ms | {st} |")
    return lines


def training_costs():
    """GPU-hours of every formal training of this study (two GPUs x wall clock, from the shared driver's status)."""
    import datetime as dt
    status = C.read(ROOT / "outputs/s45rt/20261031/status_r2.json")["jobs"]
    lines = ["| 训练 | 开始（UTC） | 墙钟 | GPU·h（2 卡） | 声明内存 |", "|---|---|---|---|---|"]
    for key, v in sorted(status.items(), key=lambda kv: kv[1].get("started_at_utc") or ""):
        if not (key.startswith("ub:") and key.split(":")[2] == "formal" and v.get("status") == "EXIT_0"):
            continue
        t0, t1 = (dt.datetime.fromisoformat(v[k]) for k in ("started_at_utc", "ended_at_utc"))
        h = (t1 - t0).total_seconds() / 3600
        lines.append(f"| {key.split(':')[-1]} | {v['started_at_utc'][:16]} | {h:.2f} h | {2 * h:.1f} | {v.get('mem_gib')} GiB |")
    return lines


def probes():
    lines = ["| 探针 | 对象 | ylf_local / ylf_global / zgz_local / zgz_global |", "|---|---|---|"]
    for d in sorted((C.OUT / "probes").glob("*/summary.json")):
        s = C.read(d)
        ev = s.get("events") or {}
        for head, v in ([("router", ev)] if "zgz_local" in ev else [(k, ev[k]) for k in ("linear", "mlp") if k in ev]):
            vals = " / ".join(f"{v[q]['part_accuracy']:.3f}" for q in ("ylf_local", "ylf_global", "zgz_local", "zgz_global") if q in v)
            lines.append(f"| 事件级部位准确率（{head}） | {d.parent.name} | {vals} |")
        f4 = ((s.get("slots") or {}).get("f4") or {}).get("folds") or {}
        if f4:
            a, b = f4.get("zs+prior"), f4.get("prior")
            lines.append(f"| 槽级 f4 RMS（[槽 + 前态] / [前态]，样本内中位；zgz_local） | {d.parent.name} | "
                         f"{a['median_rms']:.3f} / {b['median_rms']:.3f}；{a['zgz_local_rms']:.3f} / {b['zgz_local_rms']:.3f} |")
    return lines


def decisions():
    out = {}
    for name in ("summary_r1", "summary_r2", "summary_r4", "summary_r4_amend1", "summary_final"):
        f = C.OUT / f"{name}.json"
        if f.is_file():
            s = C.read(f)
            keep = {k: s[k] for k in ("layers", "candidates", "next", "arbitration", "best_deployable", "adopted", "paths",
                                      "verdict", "mean_delta_row", "half_spread_delta_row", "gap") if k in s}
            if "mechanisms" in s:
                keep["mechanisms"] = {m: {k: v[k] for k in ("network", "class", "budget_ok", "row_minus_base",
                                                            "insample_gain_vs_base", "C")} for m, v in s["mechanisms"].items()}
            out[name] = keep
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(C.OUT / "final_tables.md"))
    a = p.parse_args(argv)
    r5.patch_common()
    R = {n: r for n in networks() for r in [readings(n)] if r["zgz"] or r["rows"]}
    dec = decisions()
    md = ["# T1A-UB 汇总表（描述；由 tools/t1aub/report_all.py 从冻结读数生成）", "", "## 主表（九列，按测量块）", "",
          *main_table(R), "", "## 上限阶梯与状态能力", "", *ladder(R), "", "## 成本", "", *costs(R), "", "## 训练成本", "", *training_costs(), "", "## 探针", "",
          *probes(), "", "## 写定判读", "", "```", json.dumps(dec, indent=1, ensure_ascii=False, default=str), "```", ""]
    Path(a.out).write_text("\n".join(md))
    Path(a.out).with_suffix(".json").write_text(json.dumps({"networks": R, "decisions": dec, "computed_at_utc": C.now()},
                                                           indent=1, ensure_ascii=False, default=str) + "\n")
    print(a.out)


if __name__ == "__main__":
    main()
