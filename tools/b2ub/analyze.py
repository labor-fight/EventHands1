#!/usr/bin/env python3
"""B2UB written readouts (docs/EXPERIMENT_RECORDS.md#b2ub-prereg-20261104), computed once after every reading; CPU only,
read-only on the readings (outputs/b2ub/20261104 and PFO2's B2 readings), descriptive throughout.

    python -m tools.b2ub.analyze all
        -> analysis/{u1,u2,u3,bound}.md, decisions/summary.json, and the main rows
           outputs/semkine/b2ub_{b2label,b2gt_label,b2gt_router}_main_row.json (tools/report_table.py's format)

  U1  per test-time route: E1 five, E3, zgz_local retention R15 / F15 / T10 / T30 (median rho), C (label);
      shares s_r = (RA_deployed - RA_r) / (RA_deployed - RA_label) per seed; "TEST_ROUTE_SUFFICES" when the three-seed
      means of all five are below S43's engine re-computation, else "TEST_ROUTE_INSUFFICIENT"; positive-gap sum G
  U2  per routing (deployed, label) and clamp set: RA overall / local / global; shares
      s_S = (RA_none - RA_S) / (RA_none - RA_TRF) per seed; interactions s_XY - s_X - s_Y; the single block with the
      largest mean share
  U3  B2-gt (seed 3407): label rows (E1 five, E3, retention, C, in-sample tl_e1 median, E1 at 5000 / 5500) and the
      deployed-router rows (exposure bias); "BOUND_BELOW_S43" when all five are below S43, else "BOUND_NOT_BELOW_S43";
      differences to pfagtdz (old recipe); the clamp ladder under the label routes
  gap decomposition (seed 3407): B2 deployed -> B2 + label (test-time routes) -> B2-gt + label (training-time routes)
      -> B2-gt + label E3 (previous state ideal too), in mm per step and as shares of the total
Main-row cost columns: B2's deployable form measured in EVF's final block (the same architecture; label routes are not
deployable, the rows are diagnostic).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.b2ub.common import (B2_COST, COLS, OUT, PFAGTDZ, PFO2_OUT, S43_ENGINE, SEEDS, TRAIN_LOCAL, U3_SEED, now,  # noqa: E402
                               read, require, save)
from tools.b2ub.engine import CLAMPS  # noqa: E402

MIXES = ("label_loc", "label_part", "label_onhand")
U1_ROUTES = ("deployed", "label") + MIXES
NOISE_RA = 0.8                       # single-seed noise floor of RA (#pfo2-prereg-20261103)
SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")


def mech():
    from tools.evf import mech as M
    M.ROOTS["b2ub"] = OUT
    return M


def u1_tag(seed, route):
    return ("pfo2", f"pfo2_fgs_s{seed}") if route == "deployed" else ("b2ub", f"b2_s{seed}__{route}")


def report(root, tag, task):
    base = PFO2_OUT if root == "pfo2" else OUT
    p = base / "readings" / tag / task / "report.json"
    return read(p) if p.exists() else None


def five_of(r):
    s = r["per_sequence"]
    return [s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"], s["zgz_local"]["mpvpe_ra_mm"],
            s["zgz_global"]["mpvpe_ra_mm"], r["overall"]["mpjpe_ra_mm"]]


def ra3(r):
    """RA overall, zgz_local, zgz_global."""
    s = r["per_sequence"]
    return [r["overall"]["mpjpe_ra_mm"], s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"]]


def gap(five):
    return float(sum(max(0.0, x - y) for x, y in zip(five, S43_ENGINE)))


def tl_median(root, tag):
    r = report(root, tag, "tl_e1")
    return None if r is None else float(np.median([r["per_sequence"][s]["mpjpe_ra_mm"] for s in TRAIN_LOCAL]))


def fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


# --------------------------------------------------------------------------- U1
def u1():
    M = mech()
    rows = {}
    for route in U1_ROUTES:
        per = {}
        for seed in SEEDS:
            root, tag = u1_tag(seed, route)
            per[seed] = M.row(root, tag)
        rows[route] = per
    out = {"routes": {}, "shares": {}}
    for route, per in rows.items():
        five = [mean([per[s]["five"][k] for s in SEEDS]) for k in range(5)]
        out["routes"][route] = {
            "five_mean": five, "ra_per_seed": {s: per[s]["five"][4] for s in SEEDS},
            "e3_mean": mean([per[s].get("e3") for s in SEEDS]),
            "amp_mean": mean([per[s]["amp"][0] for s in SEEDS if "amp" in per[s]]),
            "retention_mean": {k: mean([per[s]["retention"][k] for s in SEEDS if "retention" in per[s]])
                               for k in ("R15", "F15", "T10", "T30")},
            "C_mean": mean([per[s].get("C") for s in SEEDS]), "G": gap(five),
            "all_below_s43": all(x < y for x, y in zip(five, S43_ENGINE))}
    dep, lab = rows["deployed"], rows["label"]
    for route in MIXES:
        out["shares"][route] = {s: (dep[s]["five"][4] - rows[route][s]["five"][4]) / (dep[s]["five"][4] - lab[s]["five"][4])
                                if dep[s]["five"][4] != lab[s]["five"][4] else None for s in SEEDS}
        out["shares"][route]["mean"] = mean([out["shares"][route][s] for s in SEEDS])
    out["label_gain_per_seed"] = {s: dep[s]["five"][4] - lab[s]["five"][4] for s in SEEDS}
    out["verdict"] = "TEST_ROUTE_SUFFICES" if out["routes"]["label"]["all_below_s43"] else "TEST_ROUTE_INSUFFICIENT"
    names = {"deployed": "B2 部署路由 r_s2d", "label": "B2 + 标签路由", "label_loc": "B2 + 只修面 / 重心（label_loc）",
             "label_part": "B2 + 只修部位（label_part）", "label_onhand": "B2 + 只修手上判定（label_onhand）"}
    L = ["## U1 测试时理想路由（B2 三种子，零训练；三种子均值）", "",
         f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}；判读：**{out['verdict']}**（B2 + 标签 G = "
         f"{fmt(out['routes']['label']['G'])} mm）", "",
         "| 路由 | E1 五项 | RA 逐种子 | G | E3 | 放大 | local R15 / F15 / T10 / T30 | C | 份额 s_r |", "|---|---|---|---|---|---|---|---|---|"]
    for route in U1_ROUTES:
        r = out["routes"][route]
        ret = " / ".join(fmt(r["retention_mean"][k]) for k in ("R15", "F15", "T10", "T30"))
        share = fmt(out["shares"][route]["mean"]) if route in MIXES else ("1" if route == "label" else "0")
        L.append(f"| {names[route]} | {' / '.join(fmt(x) for x in r['five_mean'])} | "
                 f"{' / '.join(fmt(r['ra_per_seed'][s]) for s in SEEDS)} | {fmt(r['G'])} | {fmt(r['e3_mean'])} | "
                 f"{fmt(r['amp_mean'])} | {ret} | {fmt(r['C_mean'], 3)} | {share} |")
    L += ["", "份额 s_r = (RA 部署 − RA 混合) / (RA 部署 − RA 标签)，逐种子算后取均值；标签收益逐种子 "
          + " / ".join(fmt(out["label_gain_per_seed"][s]) for s in SEEDS) + " mm。单种子噪声底约 0.8 mm RA。", ""]
    return out, "\n".join(L)


# --------------------------------------------------------------------------- U2
def clamp_tag(net, seed, routing, clamp):
    lab = "__label" if routing == "label" else ""
    return f"{net}_s{seed}{lab}__clamp-{clamp}"


def u2_block(net, seeds, routing):
    per = {}
    for seed in seeds:
        per[seed] = {c: ra3(report("b2ub", clamp_tag(net, seed, routing, c), "e1c")) for c in CLAMPS}
    shares = {}
    for c in CLAMPS:
        shares[c] = {}
        for j, part in enumerate(("overall", "local", "global")):
            vals = []
            for seed in seeds:
                v = per[seed]
                den = v["none"][j] - v["TRF"][j]
                vals.append((v["none"][j] - v[c][j]) / den if den != 0 else None)
            shares[c][part] = {"per_seed": vals, "mean": mean(vals)}
    def less(xy, part):
        v = [shares[c][part]["mean"] for c in (xy, xy[0], xy[1])]
        return None if None in v else v[0] - v[1] - v[2]

    def largest(part):
        known = [c for c in ("T", "R", "F") if shares[c][part]["mean"] is not None]
        return max(known, key=lambda c: shares[c][part]["mean"]) if known else None
    inter = {xy: {part: less(xy, part) for part in ("overall", "local", "global")} for xy in ("TR", "TF", "RF")}
    main_block = {part: largest(part) for part in ("overall", "local", "global")}
    ra_mean = {c: [mean([per[s][c][j] for s in seeds]) for j in range(3)] for c in CLAMPS}
    return {"ra_per_seed": per, "ra_mean": ra_mean, "shares": shares, "interactions": inter, "largest_single": main_block}


def u2_table(title, blk):
    L = [f"### {title}", "", "| 钳位 S | RA 全 / local / global（均值） | 份额 s_S 全 / local / global |", "|---|---|---|"]
    for c in CLAMPS:
        sh = " / ".join(fmt(blk["shares"][c][p]["mean"]) for p in ("overall", "local", "global"))
        L.append(f"| {c} | {' / '.join(fmt(x) for x in blk['ra_mean'][c])} | {sh} |")
    it = blk["interactions"]
    L += ["", "交互（s_XY − s_X − s_Y，全 / local / global）：" + "；".join(
        f"{xy} {' / '.join(fmt(it[xy][p]) for p in ('overall', 'local', 'global'))}" for xy in ("TR", "TF", "RF")),
          "份额最大的单一块：" + " / ".join(f"{p} {blk['largest_single'][p]}" for p in ("overall", "local", "global")), ""]
    return "\n".join(L)


def u2():
    out = {"deployed": u2_block("b2", SEEDS, "deployed"), "label": u2_block("b2", SEEDS, "label")}
    L = ["## U2 闭环逐自由度钳位（B2 三种子，零训练）", "",
         "s_S = (RA_none − RA_S) / (RA_none − RA_TRF)，逐种子；none = 自身闭环 E1，TRF = 每段首个端点以外前态全为 GT。", "",
         u2_table("部署路由 r_s2d", out["deployed"]), u2_table("标签路由", out["label"])]
    return out, "\n".join(L)


# --------------------------------------------------------------------------- U3
def u3():
    M = mech()
    lab = M.row("b2ub", f"b2gt_s{U3_SEED}__label")
    rout = M.row("b2ub", f"b2gt_s{U3_SEED}")
    b2 = M.row("pfo2", f"pfo2_fgs_s{U3_SEED}")
    late = {st: report("b2ub", f"b2gt_s{U3_SEED}@{st}__label", "e1") for st in (5000, 5500)}
    out = {"label": lab, "router": rout, "b2_3407": b2,
           "late_ra": {st: (r["overall"]["mpjpe_ra_mm"] if r else None) for st, r in late.items()},
           "tl_e1_median": {"b2gt_label": tl_median("b2ub", f"b2gt_s{U3_SEED}__label"),
                            "b2_deployed": tl_median("pfo2", f"pfo2_fgs_s{U3_SEED}")},
           "verdict": "BOUND_BELOW_S43" if all(x < y for x, y in zip(lab["five"], S43_ENGINE)) else "BOUND_NOT_BELOW_S43",
           "G": gap(lab["five"]), "minus_s43": [x - y for x, y in zip(lab["five"], S43_ENGINE)],
           "minus_pfagtdz": [x - y for x, y in zip(lab["five"], PFAGTDZ)],
           "exposure_ra": rout["five"][4] - b2["five"][4],
           "clamps": u2_block("b2gt", (U3_SEED,), "label")}
    L = ["## U3 B2-gt：标签路由训练（种子 3407，单种子描述）", "",
         f"判读：**{out['verdict']}**（G = {fmt(out['G'])} mm）；对 S43 五项差 {' / '.join(fmt(x) for x in out['minus_s43'])}；"
         f"对 pfagtdz（旧配方）{' / '.join(fmt(x) for x in out['minus_pfagtdz'])}；暴露偏差（B2-gt + r_s2d − B2 3407）RA "
         f"{fmt(out['exposure_ra'])} mm", "",
         "| 网络 | E1 五项 | E3 | 放大 | local R15 / F15 / T10 / T30 | C | 样本内 tl_e1 中位 | E1 RA @5000 / 5500 |",
         "|---|---|---|---|---|---|---|---|"]
    for name, r, tl, lt in (("B2-gt + 标签路由", lab, out["tl_e1_median"]["b2gt_label"], out["late_ra"]),
                            ("B2-gt + r_s2d（暴露偏差）", rout, None, None),
                            ("B2 3407（部署，参照）", b2, out["tl_e1_median"]["b2_deployed"], None)):
        ret = " / ".join(fmt(r["retention"][k]) for k in ("R15", "F15", "T10", "T30")) if "retention" in r else "–"
        late_s = " / ".join(fmt(lt[st]) for st in (5000, 5500)) if lt else "–"
        L.append(f"| {name} | {' / '.join(fmt(x) for x in r['five'])} | {fmt(r.get('e3'))} | "
                 f"{fmt(r['amp'][0] if 'amp' in r else None)} | {ret} | {fmt(r.get('C'), 3)} | {fmt(tl)} | {late_s} |")
    L += ["", u2_table("B2-gt 标签路由下的钳位（单种子）", out["clamps"])]
    return out, "\n".join(L)


# --------------------------------------------------------------------------- gap decomposition and the bound table
def decomposition(o1, o3):
    a = o3["b2_3407"]
    b = mech().row("b2ub", f"b2_s{U3_SEED}__label")
    c = o3["label"]
    steps = [("B2 部署（r_s2d）", a["five"][4], a["five"]), ("B2 + 标签路由（测试时）", b["five"][4], b["five"]),
             ("B2-gt + 标签路由（训练时也理想）", c["five"][4], c["five"]), ("B2-gt + 标签路由 E3（前态也理想）", c["e3"], None)]
    known = all(v is not None for _, v, _ in steps)
    total = steps[0][1] - steps[-1][1] if known else None
    parts = [{"from": steps[i][0], "to": steps[i + 1][0], "mm": steps[i][1] - steps[i + 1][1] if known else None,
              "share": (steps[i][1] - steps[i + 1][1]) / total if known and total else None} for i in range(3)]
    return {"seed": U3_SEED, "steps": [{"name": n, "ra": v, "five": f} for n, v, f in steps], "parts": parts,
            "total_mm": total}


def bound_md(o1, o2, o3, dec):
    L = ["## 上限表（标签辅助诊断上界与同部署输入的实测参照分开列；都不是理论上限）", "",
         "| 层 | 网络 / 读法 | 种子 | E1 五项 | RA | E3 |", "|---|---|---|---|---|---|",
         f"| 参照 | S43（引擎三种子） | 3 | {' / '.join(fmt(x) for x in S43_ENGINE)} | {fmt(S43_ENGINE[4])} | – |",
         f"| 同部署输入实测 | B2 部署 r_s2d | 3 | {' / '.join(fmt(x) for x in o1['routes']['deployed']['five_mean'])} | "
         f"{fmt(o1['routes']['deployed']['five_mean'][4])} | {fmt(o1['routes']['deployed']['e3_mean'])} |",
         f"| 前态理想（闭环全钳位） | B2 部署，TRF 钳位 | 3 | – | {fmt(o2['deployed']['ra_mean']['TRF'][0])} | – |",
         f"| 测试时路由理想 | B2 + 标签路由 | 3 | {' / '.join(fmt(x) for x in o1['routes']['label']['five_mean'])} | "
         f"{fmt(o1['routes']['label']['five_mean'][4])} | {fmt(o1['routes']['label']['e3_mean'])} |",
         f"| 训练与测试路由理想 | B2-gt + 标签路由 | 1 | {' / '.join(fmt(x) for x in o3['label']['five'])} | "
         f"{fmt(o3['label']['five'][4])} | {fmt(o3['label'].get('e3'))} |",
         f"| 旧配方参照 | pfagtdz（S45-SM，标签路由训练 + 测试） | 1 | {' / '.join(fmt(x) for x in PFAGTDZ)} | {fmt(PFAGTDZ[4])} | – |",
         "", f"差距分解（种子 {dec['seed']}，RA，mm）：" + "；".join(
             f"{p['from']} → {p['to']} {fmt(p['mm'])}（{fmt(p['share'])}）" for p in dec["parts"])
         + f"；合计 {fmt(dec['total_mm'])}。单种子噪声底约 {NOISE_RA} mm RA。", ""]
    return "\n".join(L)


# --------------------------------------------------------------------------- main rows
def main_row(name, items, label):
    c = read(B2_COST)
    require(c["policy"] == "default", "the table's cost comes from the PyTorch-default-flag reading")
    per = {}
    for i, (root, tag) in enumerate(items):
        r = report(root, tag, "e1")
        s = r["per_sequence"]
        per[str(i)] = {"reading": f"{root}:{tag}", "checkpoint": r["identity"]["checkpoint"],
                       "local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
                       "overall": {k: r["overall"][k] for k in KEYS}}
    m = {b: {k: float(np.mean([per[s][b][k] for s in per])) for k in KEYS} for b in ("local", "global", "overall")}
    out = {"arm": name, "label": label, "split": "val_core", "n_seeds": len(items), "two_seed_mean": m, "per_seed": per,
           "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": str(B2_COST),
           "written_utc": now(),
           "note": "B2UB diagnostic row: PFO's engine E1 (deterministic flags); the cost columns are B2's deployable form "
                   "(the same architecture) from EVF's final block; label routes are not deployable"}
    p = SEMKINE / f"b2ub_{name}_main_row.json"
    save(p, out)
    return p


def all_():
    o1, t1 = u1()
    o2, t2 = u2()
    o3, t3 = u3()
    dec = decomposition(o1, o3)
    tb = bound_md(o1, o2, o3, dec)
    for name, text in (("u1", t1), ("u2", t2), ("u3", t3), ("bound", tb)):
        p = OUT / "analysis" / f"{name}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        require(not p.exists(), "written once: " + str(p))
        p.write_text(text + "\n")
    rows = [main_row("b2label", [("b2ub", f"b2_s{s}__label") for s in SEEDS], "B2 + 标签路由（测试时，三种子，不可部署）"),
            main_row("b2gt_label", [("b2ub", f"b2gt_s{U3_SEED}__label")], "B2-gt：标签路由训练 + 测试（种子 3407，不可部署）"),
            main_row("b2gt_router", [("b2ub", f"b2gt_s{U3_SEED}")], "B2-gt + r_s2d（标签路由训练、部署路由测试，种子 3407）")]
    summary = {"u1": {k: o1[k] for k in ("verdict", "label_gain_per_seed", "shares")},
               "u1_routes": o1["routes"], "u2": {k: {"ra_mean": v["ra_mean"], "shares": v["shares"],
                                                     "interactions": v["interactions"], "largest_single": v["largest_single"]}
                                                 for k, v in o2.items()},
               "u3": {k: o3[k] for k in ("verdict", "G", "minus_s43", "minus_pfagtdz", "exposure_ra", "late_ra", "tl_e1_median")},
               "u3_rows": {k: {kk: o3[k].get(kk) for kk in ("five", "e3", "amp", "retention", "C")} for k in
                           ("label", "router", "b2_3407")},
               "u3_clamps": {"ra_mean": o3["clamps"]["ra_mean"], "shares": o3["clamps"]["shares"]},
               "decomposition": dec, "main_rows": [str(p) for p in rows], "s43_engine": list(S43_ENGINE),
               "pfagtdz": list(PFAGTDZ), "noise_ra": NOISE_RA, "written_at_utc": now()}
    save(OUT / "decisions" / "summary.json", summary, exclusive=True)
    print("\n\n".join((t1, t2, t3, tb)))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("all",))
    p.parse_args(argv)
    all_()


if __name__ == "__main__":
    main()
