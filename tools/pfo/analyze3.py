#!/usr/bin/env python3
"""PFO final-phase readings -> tables and the written rules (docs/EXPERIMENT_RECORDS.md#pfo-p3-prereg-20261103,
#pfo-p3b-prereg-20261103, #pfo-p3c-prereg-20261103, #pfo-p3d-prereg-20261103); CPU only, read-only on the readings.

    python -m tools.pfo.analyze3 arms [--out FILE]    tools/pfo/analyze2.py's arm table over every registered arm
                                                      (2a + tools/pfo/arms3.py)
    python -m tools.pfo.analyze3 checks [--tags T,...] [--out FILE]
        the closed-loop checks of #pfo-prereg-20261103 per reading tag: E1 / E3 RA and their ratio (amplification),
        correct-state hold (GT start, same packet 1 -> 20 times), recovery (mean-hand start C; injected-error retention),
        drift (E1 RA by elapsed time in the segment; zgz_local by step in the segment)
    python -m tools.pfo.analyze3 rules [--out FILE]   the written rules on what has been read; a rule whose readings are
                                                      incomplete is reported as pending:
        criterion  F against S43's engine 3-seed means (five metrics, count below, sum of the positive gaps)
        loo        leave-one-out F - m (f1hot): 2-seed paired mean E1 RA of F better by >= 0.3 mm, both seeds the same way
        addon      every F + m arm (p3b, p3c): more of the five metrics below S43 than F, or as many with the positive-gap sum
                   lower by >= 0.3 mm; neither seed's RA worse than F's same seed by more than 0.3 mm
        screen     2b / 2c arms against B seed 3407: E1 RA better by >= 0.5 mm, or local R15 and F15 retention both lower
                   by >= 0.10 with E1 RA not worse than B + 0.3 mm
        swap       zero-training deployed-router replacement on F's three seeds (#pfo-p3d-prereg-20261103)
        router     equivalence (#pfo-p2r-prereg-20261103) and 'quality better' (#pfo-p2r2-prereg-20261103)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
import numpy as np  # noqa: E402

from tools.pfo import analyze2 as A2  # noqa: E402
from tools.pfo.analyze import R, drift, f1_c, fmt, pert_rho, rep, row5  # noqa: E402
from tools.pfo.arms import run_name  # noqa: E402
from tools.pfo.common import OUT, read  # noqa: E402

SEEDS = (3407, 3408, 3409)
PAIRED = (3407, 3408)
COLS = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")
FINAL = "f1"
LOO = {"f1hot": "nohot"}
ADDON_SECTIONS = ("pfo-p3b-prereg-20261103", "pfo-p3c-prereg-20261103")
SCREENED = ("noise", "bank", "t0ptrain", "noprev", "pos", "nohot")
ROUTERS = ("r_s2d", "r_s2dkd", "r_wide", "r_s2d_nh")


def arm_names():
    from tools.pfo.arms import ARMS
    from tools.pfo.arms3 import ARMS3
    return list(ARMS) + [a for a in ARMS3 if a not in ARMS]


def e1(arm, seed):
    return row5(rep(run_name(arm, seed), "e1"))


def s43():
    rows = [row5(rep(f"s43_s{s}", "e1")) for s in SEEDS]
    return np.mean(rows, 0) if all(rows) else None


def criterion(m, ref):
    below = [bool(a < b) for a, b in zip(m, ref)]
    return {"mean": [float(v) for v in m], "below": below, "count": int(sum(below)),
            "gap_sum": float(sum(max(0.0, a - b) for a, b in zip(m, ref)))}


def rule_criterion(ref):
    per = {s: e1(FINAL, s) for s in SEEDS}
    per = {s: v for s, v in per.items() if v}
    if not per:
        return {"status": "pending"}
    res = criterion(np.mean(list(per.values()), 0), ref)
    res["seeds"] = sorted(per)
    res["per_seed_below"] = [int(sum(per[s][i] < ref[i] for s in per)) for i in range(5)]
    res["status"] = "complete" if len(per) == 3 else "partial"
    res["met"] = res["count"] == 5 if len(per) == 3 else None
    res["strong"] = (res["met"] and all(3 * n >= 2 * len(per) for n in res["per_seed_below"])) if len(per) == 3 else None
    return res


def rule_loo():
    out = {}
    for arm, mech in LOO.items():
        d = {s: (e1(FINAL, s), e1(arm, s)) for s in PAIRED}
        if not all(a and b for a, b in d.values()):
            out[arm] = {"mechanism": mech, "status": "pending"}
            continue
        diff = [d[s][0][4] - d[s][1][4] for s in PAIRED]        # F - (F - m): negative = the mechanism helps
        out[arm] = {"mechanism": mech, "status": "complete", "ra_diff_per_seed": diff, "ra_diff_mean": float(np.mean(diff)),
                    "five_diff_mean": (np.mean([d[s][0] for s in PAIRED], 0) - np.mean([d[s][1] for s in PAIRED], 0)).tolist(),
                    "keep": bool(np.mean(diff) <= -0.3 and all(x < 0 for x in diff))}
    return out


def addon_arms():
    from tools.pfo.arms3 import ARMS3
    return [a for a, s in ARMS3.items() if s.get("section") in ADDON_SECTIONS]


def rule_addon(ref):
    out = {}
    for arm in addon_arms():
        d = {s: (e1(FINAL, s), e1(arm, s)) for s in PAIRED}
        if not all(a and b for a, b in d.values()):
            out[arm] = {"status": "pending"}
            continue
        f = criterion(np.mean([d[s][0] for s in PAIRED], 0), ref)
        m = criterion(np.mean([d[s][1] for s in PAIRED], 0), ref)
        ra = [d[s][1][4] - d[s][0][4] for s in PAIRED]
        better = m["count"] > f["count"] or (m["count"] == f["count"] and m["gap_sum"] <= f["gap_sum"] - 0.3)
        out[arm] = {"status": "complete", "F": f, "addon": m, "ra_diff_per_seed": ra,
                    "adopt": bool(better and all(x <= 0.3 for x in ra))}
    return out


def rule_swap(ref, arm=FINAL):
    """#pfo-p3d-prereg-20261103: router X replaces the arm's deployed router when, over its three seeds, X gives more of
    the five metrics below S43 (or as many with the positive-gap sum lower by >= 0.3 mm) and no seed's RA worse by more
    than 0.3 mm; the budget check (X's MACs) is applied with the final cost block."""
    base = {s: e1(arm, s) for s in SEEDS}
    out = {}
    for x in ("r_s2dkd", "r_s2dw", "r_wide", "r_s2d_nh"):
        sw = {s: row5(rep(f"{run_name(arm, s)}+{x}", "e1")) for s in SEEDS}
        if not all(base.values()) or not all(sw.values()):
            out[x] = {"status": "pending", "seeds_read": [s for s in SEEDS if sw[s]]}
            continue
        f = criterion(np.mean(list(base.values()), 0), ref)
        m = criterion(np.mean(list(sw.values()), 0), ref)
        ra = [sw[s][4] - base[s][4] for s in SEEDS]
        better = m["count"] > f["count"] or (m["count"] == f["count"] and m["gap_sum"] <= f["gap_sum"] - 0.3)
        out[x] = {"status": "complete", "base": f, "swap": m, "ra_diff_per_seed": ra,
                  "replace": bool(better and all(v <= 0.3 for v in ra))}
    return out


def rule_screen():
    b = e1("b", 3407)
    rb = A2.retention(run_name("b", 3407))
    out = {}
    for arm in SCREENED:
        a = e1(arm, 3407)
        if a is None or b is None:
            out[arm] = {"status": "pending"}
            continue
        r = A2.retention(run_name(arm, 3407))
        d_ra = a[4] - b[4]
        d_r15 = None if r is None or rb is None else r["R15"][0] - rb["R15"][0]
        d_f15 = None if r is None or rb is None else r["F15"][0] - rb["F15"][0]
        ret_ok = d_r15 is not None and d_r15 <= -0.10 and d_f15 <= -0.10 and d_ra <= 0.3
        out[arm] = {"status": "complete" if r is not None else "partial", "e1": list(a), "d_ra": d_ra,
                    "d_local_R15": d_r15, "d_local_F15": d_f15, "promote": bool(d_ra <= -0.5 or ret_ok)}
    return out


def rule_router():
    base = read(OUT / "routers_read" / "sr1.json")["per_sequence"]
    p_e1, p_e3 = row5(rep("pfasrdz", "e1")), row5(rep("pfasrdz", "e3"))
    t_sr1 = row5(rep("pfat0pdz_cos__sr1", "e1"))
    out = {}
    for name in ROUTERS:
        q = OUT / "routers_read" / f"{name}.json"
        sw1, sw3 = row5(rep(f"pfasrdz+{name}", "e1")), row5(rep(f"pfasrdz+{name}", "e3"))
        tw1 = row5(rep(f"pfat0pdz_cos+{name}", "e1"))
        fw1, fw3 = row5(rep(f"{run_name(FINAL, 3407)}+{name}", "e1")), row5(rep(f"{run_name(FINAL, 3407)}+{name}", "e3"))
        res = {"swap_F3407": {"e1": fw1, "e3": fw3}}
        if q.exists():
            s = read(q)["per_sequence"]
            res["quality"] = {k: {m: s[k][m] for m in ("part_acc", "point_err_mm_median", "offhand_in_mesh")}
                              for k in ("zgz_local", "zgz_global", "ylf_local")}
            res["quality_ok"] = all(s[k]["part_acc"] >= base[k]["part_acc"] - 0.02
                                    and s[k]["point_err_mm_median"] <= base[k]["point_err_mm_median"] + 1.0
                                    and s[k]["offhand_in_mesh"] <= base[k]["offhand_in_mesh"] + 0.02
                                    for k in ("zgz_local", "zgz_global"))
        if sw1 and sw3:
            res["swap_pfasrdz"] = {"e1": sw1, "e3": sw3}
            if "quality_ok" in res:
                res["equivalent"] = bool(res["quality_ok"] and sw1[4] <= p_e1[4] + 0.5 and sw3[4] <= p_e3[4] + 0.3)
        if q.exists() and sw1 and tw1:
            res["swap_pfat0pdz_cos"] = {"e1": tw1}
            res["quality_better"] = bool(s["zgz_local"]["point_err_mm_median"] <= base["zgz_local"]["point_err_mm_median"] - 1.0
                                         and sw1[4] < p_e1[4] and tw1[4] < t_sr1[4])
        out[name] = res
    return out


def rules():
    ref = s43()
    res = {"s43": ref.tolist(), "criterion": rule_criterion(ref), "loo": rule_loo(), "addon": rule_addon(ref),
           "swap": rule_swap(ref), "screen": rule_screen(), "router": rule_router()}
    c = res["criterion"]
    lines = ["## 写定规则（读到的部分）", "", f"S43 引擎三种子均值：{' / '.join(fmt(v) for v in ref)}", ""]
    if c["status"] == "pending":
        lines.append("- F 判据：未读")
    else:
        lines.append(f"- F（种子 {c['seeds']}）均值 {' / '.join(fmt(v) for v in c['mean'])}：低于 S43 {c['count']} / 5 项，"
                     f"正差距和 {c['gap_sum']:.2f} mm；逐项低于 S43 的种子数 {c['per_seed_below']}；"
                     f"达标 {c['met']}，强达标 {c['strong']}")
    for arm, r in res["loo"].items():
        lines.append(f"- 逐一去掉 {arm}（去 {r['mechanism']}）：" + (
            "未读" if r["status"] == "pending" else
            f"F − ({arm}) RA {', '.join(f'{x:+.2f}' for x in r['ra_diff_per_seed'])}，均值 {r['ra_diff_mean']:+.2f} → "
            f"{'保留' if r['keep'] else '不满足保留条件'}；五项均值差 {' / '.join(f'{x:+.2f}' for x in r['five_diff_mean'])}"))
    for arm, r in res["addon"].items():
        lines.append(f"- 附加臂 {arm}：" + (
            "未读" if r["status"] == "pending" else
            f"F 两种子 {r['F']['count']} 项 / 正差距 {r['F']['gap_sum']:.2f}；{arm} {r['addon']['count']} 项 / "
            f"{r['addon']['gap_sum']:.2f}（{' / '.join(fmt(v) for v in r['addon']['mean'])}）；RA 对 F "
            f"{', '.join(f'{x:+.2f}' for x in r['ra_diff_per_seed'])} → {'采纳' if r['adopt'] else '不采纳'}"))
    for x, r in res["swap"].items():
        lines.append(f"- 部署路由器换成 {x}（F 三种子）：" + (
            f"未读齐（已读种子 {r['seeds_read']}）" if r["status"] == "pending" else
            f"F {r['base']['count']} 项 / 正差距 {r['base']['gap_sum']:.2f}；换上 {r['swap']['count']} 项 / "
            f"{r['swap']['gap_sum']:.2f}（{' / '.join(fmt(v) for v in r['swap']['mean'])}）；RA 逐种子 "
            f"{', '.join(f'{v:+.2f}' for v in r['ra_diff_per_seed'])} → {'替换（再查成本）' if r['replace'] else '不替换'}"))
    for arm, r in res["screen"].items():
        if r["status"] == "pending":
            lines.append(f"- 筛选 {arm}：未读")
            continue
        rr = ("–" if r["d_local_R15"] is None else f"{r['d_local_R15']:+.3f} / {r['d_local_F15']:+.3f}")
        lines.append(f"- 筛选 {arm}（对 B 3407）：RA {r['d_ra']:+.2f}，local R15 / F15 {rr} → "
                     f"{'晋级' if r['promote'] else '不晋级'}")
    for name, r in res["router"].items():
        parts = []
        if "quality" in r:
            ql = r["quality"]["zgz_local"]
            parts.append(f"zgz_local 部位 {ql['part_acc']:.3f} / 点误差 {ql['point_err_mm_median']:.1f} / 手外 "
                         f"{ql['offhand_in_mesh']:.3f}")
        if "swap_pfasrdz" in r:
            parts.append(f"pfasrdz 换上 E1 RA {r['swap_pfasrdz']['e1'][4]:.2f}、E3 RA {r['swap_pfasrdz']['e3'][4]:.2f}")
        if r["swap_F3407"]["e1"]:
            parts.append(f"F 3407 换上 E1 {' / '.join(fmt(v) for v in r['swap_F3407']['e1'])}")
        if "equivalent" in r:
            parts.append(f"等价 {r['equivalent']}")
        if "quality_better" in r:
            parts.append(f"更好 {r['quality_better']}")
        lines.append(f"- 路由器 {name}：" + ("；".join(parts) if parts else "未读"))
    return lines, res


def arms():
    A2.arm_names = arm_names
    return A2.arms()


PERT = ("T10", "T30", "R5", "R15", "F05", "F15", "N")
CHECK_TAGS = tuple(f"s43_s{s}" for s in SEEDS) + ("pfasrdz",) + tuple(run_name(a, s) for a in ("b", FINAL) for s in SEEDS)


def step_profile(tag):
    p = R / tag / "e1" / "states.npz"
    if not p.exists():
        return None
    z = np.load(p)
    step = (z["zgz_local/elapsed_ms"] + 1) // 50
    ra = z["zgz_local/mpjpe_ra_mm"]
    return [float(ra[step == k].mean()) for k in range(1, 9)] + [float(ra[step >= 9].mean())]


def checks(tags=CHECK_TAGS):
    lines = ["## 闭环检查（zgz；local / global）", "",
             "| 网络 | E1 RA | E3 RA | 放大 E1 / E3 | GT 起点同包 1 → 20 次 RA | 均值手型起点 C | 自身闭环态 1 → 20 次 |",
             "|---|---|---|---|---|---|---|"]
    rho_lines = ["", "注入误差的一步保留率 ρ 中位（local / global；0 = 一步纠正，1 = 原样保留）：", "",
                 "| 网络 | " + " | ".join(PERT) + " |", "|---|" + "---|" * len(PERT)]
    drift_lines = ["", "漂移：E1 RA 按段内经过时间（0–1 / 1–2 / 2–4 / 4–8 / ≥ 8 s）与 zgz_local 段内第 1–8 步、≥ 第 9 步：", "",
                   "| 网络 | zgz_local 经过时间 | zgz_global 经过时间 | zgz_local 第 1–8 步 / ≥ 9 |", "|---|---|---|---|"]
    js = {}
    for tag in tags:
        a, b = rep(tag, "e1"), rep(tag, "e3")
        if a is None:
            continue
        s1, s3 = a["per_sequence"], (b or {}).get("per_sequence")
        ra1 = [s1[s]["mpjpe_ra_mm"] for s in ("zgz_local", "zgz_global")]
        ra3 = None if s3 is None else [s3[s]["mpjpe_ra_mm"] for s in ("zgz_local", "zgz_global")]
        f = f1_c(tag)
        rho = pert_rho(tag)
        d = drift(tag)
        prof = step_profile(tag)
        js[tag] = {"e1": ra1, "e3": ra3, "f1": f, "rho": rho, "drift": d, "profile": prof}
        amp = "–" if ra3 is None else f"{ra1[0] / ra3[0]:.2f} / {ra1[1] / ra3[1]:.2f}"
        gt = "–" if f is None else (f"{f['zgz_local']['gt1']:.2f} → {f['zgz_local']['gt20']:.2f} / "
                                     f"{f['zgz_global']['gt1']:.2f} → {f['zgz_global']['gt20']:.2f}")
        loop = "–" if f is None else (f"{f['zgz_local']['loop1']:.2f} → {f['zgz_local']['loop20']:.2f} / "
                                       f"{f['zgz_global']['loop1']:.2f} → {f['zgz_global']['loop20']:.2f}")
        lines.append(f"| {tag} | {ra1[0]:.2f} / {ra1[1]:.2f} | {'–' if ra3 is None else f'{ra3[0]:.2f} / {ra3[1]:.2f}'} | "
                     f"{amp} | {gt} | {fmt(f and f['pooled_C'], 3)} | {loop} |")
        if rho is not None:
            rho_lines.append(f"| {tag} | " + " | ".join(f"{rho['zgz_local'][k][0]:.2f} / {rho['zgz_global'][k][0]:.2f}"
                                                         for k in PERT) + " |")
        if d is not None:
            cell = lambda s: " / ".join("–" if v is None else f"{v:.1f}" for _, _, v, _ in d[s])  # noqa: E731
            drift_lines.append(f"| {tag} | {cell('zgz_local')} | {cell('zgz_global')} | "
                               f"{' / '.join(f'{v:.1f}' for v in prof[:8])} / {prof[8]:.1f} |")
    return lines + rho_lines + drift_lines, js


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("arms", "rules", "checks"))
    p.add_argument("--tags")
    p.add_argument("--out")
    a = p.parse_args(argv)
    if a.what == "checks":
        lines, js = checks(tuple(a.tags.split(",")) if a.tags else CHECK_TAGS)
    else:
        lines, js = {"arms": arms, "rules": rules}[a.what]()
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)
        Path(a.out).with_suffix(".json").write_text(json.dumps(js, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()
