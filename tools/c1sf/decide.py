#!/usr/bin/env python3
"""C1SF readouts and pre-registered classifications (thresholds read from the frozen protocol).

  python tools/c1sf/decide.py step1   # step 1 only (after the zero-training readings)
  python tools/c1sf/decide.py all     # step 1 + the three arms + the main table

Writes outputs/c1sf/20261029/{step1.json, decision.json, report.md}; never trains, selects or re-scores.
"""
from __future__ import annotations

import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.c1sf.common import FORMAL_ARMS, OUT, TRAIN_BASE, XCL_C1_EVALUATIONS, now, read, run_name, save, verify_protocol  # noqa: E402

DOMAINS = ("local", "global")
LABELS = {"dz": "C1SF dz（seed3407）：C1 + depth 尺度增强", "zr": "C1SF zr（seed3407）：C1 + zn 相对前态根深度",
          "so3": "C1SF so3（seed3407）：C1 + SO(3)+平移+FK 损失"}


def report(run, mode, step=6000, intervention="none", repeat=1):
    f = OUT / "evaluations" / run / f"{mode}_step{step}_repeat{repeat}_{intervention}" / "report.json"
    if not f.is_file():
        return None
    r = read(f)
    return r if r.get("status") == "COMPLETE" else None


def ra(r, domain):
    return None if r is None else r["metrics"][domain]["mpjpe_ra_mm"]


def step1(protocol):
    g = protocol["gates"]["step1"]
    zgz = report("c1", "e3")
    out = {"computed_at_utc": now(), "domains": {}, "per_recording": {}}
    reads = {i: report("c1", "train_e3", intervention=i) for i in ("none", "geometry_zero", "h3_zero")}
    if zgz is None or reads["none"] is None:
        out["status"] = "MISSING_READINGS"
        save(OUT / "step1.json", out)
        return out
    seq = {i: {r["seq"]: r for r in rep["per_sequence"]} for i, rep in reads.items() if rep is not None}
    for name in TRAIN_BASE:
        base = seq["none"][name]
        out["per_recording"][name] = {
            "n_endpoints": base["n_frames"], "c1_e3_ra": base["mpjpe_ra_mm"], "identity_ra": base["identity_mpjpe_ra_mm"],
            "delta_vs_identity": base["mpjpe_ra_mm"] - base["identity_mpjpe_ra_mm"],
            **{f"{i}_minus_none": seq[i][name]["mpjpe_ra_mm"] - base["mpjpe_ra_mm"] for i in ("geometry_zero", "h3_zero") if i in seq}}
    for domain in DOMAINS:
        names = [n for n in TRAIN_BASE if n.endswith("_" + domain)]
        deltas = [out["per_recording"][n]["delta_vs_identity"] for n in names]
        zseq = "zgz_" + domain
        z_c1 = next(r["mpjpe_ra_mm"] for r in zgz["per_sequence"] if r["seq"] == zseq)
        z_id = zgz["identity"][zseq]["mpjpe_ra_mm"]
        dz = z_c1 - z_id
        dt = statistics.median(deltas)
        if dz <= 0:
            verdict = "zgz 上 C1 不劣于恒等，不适用"
        elif dt <= g["subject_bias_max_ratio"] * dz:
            verdict = "以受试者偏差为主"
        elif dt >= g["structural_min_ratio"] * dz:
            verdict = "结构本身有害（训练受试者上同样有害）"
        else:
            verdict = "混合"
        out["domains"][domain] = {
            "zgz_c1_e3_ra": z_c1, "zgz_identity_ra": z_id, "zgz_delta": dz,
            "train_median_delta": dt, "train_deltas_positive": sum(d > 0 for d in deltas), "n_recordings": len(deltas),
            "ratio": dt / dz if dz > 0 else None, "verdict": verdict,
            "train_median_geometry_zero_minus_none": statistics.median(
                out["per_recording"][n].get("geometry_zero_minus_none") for n in names) if "geometry_zero" in seq else None,
            "train_median_h3_zero_minus_none": statistics.median(
                out["per_recording"][n].get("h3_zero_minus_none") for n in names) if "h3_zero" in seq else None}
    out["status"] = "COMPLETE"
    save(OUT / "step1.json", out)
    return out


def c1_reference(mode, step=6000, intervention="none"):
    """C1 readings: current-code re-evaluation where run (pairing-checked equal to XCL), else the XCL trace."""
    if mode == "e2":
        mode = "e1"  # C1's own trace is the common previous state: E2 of C1 is its E1 (XCL convention)
    mine = report("c1", mode, step, intervention)
    if mine is not None:
        return mine
    f = XCL_C1_EVALUATIONS / f"{mode}_step{step}_repeat1_{intervention}" / "report.json"
    return read(f) if f.is_file() else None


def arms(protocol):
    from tools.xcl.report import direction_support
    from tools.xcl.evaluate import compare_traces
    g = protocol["gates"]
    result = {}
    for arm in FORMAL_ARMS:
        if not (OUT / "receipts" / f"formal_{arm}.json").is_file():
            result[arm] = {"status": "NO_VERIFIED_TRAINING"}
            continue
        entry = {"status": "COMPLETE", "readouts": {}, "support": {}}
        for domain in DOMAINS:
            d = {}
            for mode in ("e1", "e2", "e3"):
                a, c = ra(report(arm, mode), domain), ra(c1_reference(mode), domain)
                d[mode] = {"arm": a, "c1": c, "delta": None if a is None or c is None else a - c}
            late = {}
            for step in (5000, 5500, 6000):
                a, c = ra(report(arm, "e1", step), domain), ra(c1_reference("e1", step), domain)
                late[str(step)] = None if a is None or c is None else a - c
            entry["readouts"][domain] = {"three": d, "late_e1_delta": late}
            entry["support"][domain] = direction_support([d[m]["delta"] for m in ("e1", "e2", "e3")],
                                                         [late[s] for s in ("5000", "5500", "6000")],
                                                         g["xcl_rule"]["threshold_mm"][domain])
        a1, a2 = (OUT / "evaluations" / arm / f"e1_step6000_repeat{r}_none/states.npz" for r in (1, 2))
        try:
            entry["e1_fresh_repeat"] = compare_traces(a1, a2)["status"]
        except Exception as exc:  # recorded, never hidden
            entry["e1_fresh_repeat"] = f"FAIL: {exc}"
        e3 = ra(report(arm, "e3"), "local")
        geo = ra(report(arm, "e3", intervention="geometry_zero"), "local")
        h3 = ra(report(arm, "e3", intervention="h3_zero"), "local")
        entry["mechanism_local_e3"] = {"e3": e3, "geometry_zero_minus_none": None if geo is None or e3 is None else geo - e3,
                                       "h3_zero_minus_none": None if h3 is None or e3 is None else h3 - e3}
        w = report(arm, "e1")
        entry["e1_wrist_abs_mm"] = None if w is None else {k: w["metrics"][k]["wrist_mm"] for k in DOMAINS}
        result[arm] = entry
    c1_e3 = ra(c1_reference("e3"), "local")
    c1_geo = ra(c1_reference("e3", intervention="geometry_zero"), "local")
    m = g["mechanism"]
    rows = {}
    for arm in ("dz", "zr"):
        e = result.get(arm, {})
        mech = e.get("mechanism_local_e3") or {}
        if mech.get("e3") is None or mech.get("geometry_zero_minus_none") is None or c1_e3 is None:
            rows[arm] = None
            continue
        rows[arm] = {"e3_change_vs_c1": mech["e3"] - c1_e3, "geometry_zero_minus_none": mech["geometry_zero_minus_none"]}
    if any(v is None for v in rows.values()):
        verdict = "缺读数"
    elif any(v["e3_change_vs_c1"] <= -m["ood_e3_improvement_mm"] and v["geometry_zero_minus_none"] >= -m["ood_geometry_harm_max_mm"]
             for v in rows.values()):
        verdict = "越界解释得到支持"
    elif all(v["geometry_zero_minus_none"] <= -m["echo_geometry_harm_min_mm"] and abs(v["e3_change_vs_c1"]) < m["ood_e3_improvement_mm"]
             for v in rows.values()):
        verdict = "回声解释得到支持"
    else:
        verdict = "未定"
    return result, {"c1_e3_local": c1_e3, "c1_geometry_zero_minus_none": None if c1_geo is None or c1_e3 is None else c1_geo - c1_e3,
                    "arms": rows, "verdict": verdict}


def main_table():
    from tools import report_table as original
    paths = {"dt3_dz_l3_w200cfg": ROOT / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json"}
    labels = {"c1sf_c1": "XYZ-C1 基线（xyz_c1_stable_s3407，C1SF 同环境重测）"}
    for run in ("c1", *FORMAL_ARMS):
        prefix = "c1sf_c1" if run == "c1" else run_name(run).removesuffix("_s3407")
        f = OUT / "evaluations" / run / "row_step6000_repeat1_none" / f"{prefix}_main_row_step6000.json"
        if f.is_file():
            paths[prefix] = f
            labels.setdefault(prefix, LABELS.get(run, prefix))

    class Rows:
        def __truediv__(self, filename):
            return paths[str(filename).removesuffix("_main_row.json")]
    old = original.OUT
    original.OUT = Rows()
    try:
        lines = [original.HEADER, original.RULE, original.BASELINE,
                 original.row("dt3_dz_l3_w200cfg", "S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *")]
        lines += [original.row(p, labels[p]) for p in paths if p != "dt3_dz_l3_w200cfg"]
    finally:
        original.OUT = old
    return "\n".join(lines)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    protocol = verify_protocol()
    s1 = step1(protocol)
    if which == "step1":
        print(json.dumps(s1, indent=2, ensure_ascii=False))
        return
    per_arm, mechanism = arms(protocol)
    table = main_table()
    decision = {"computed_at_utc": now(), "step1": s1, "arms": per_arm, "mechanism": mechanism,
                "single_seed_descriptive_only": True, "s39_s43_c1_unchanged": True}
    save(OUT / "decision.json", decision)
    lines = ["# C1SF 结果（自动生成，单 seed3407 描述性）", "", table, "",
             "## 第一步（零训练，冻结 C1，GT 前态单步）", ""]
    for domain, d in s1.get("domains", {}).items():
        lines.append(f"- {domain}：zgz C1 {d['zgz_c1_e3_ra']:.2f} / 恒等 {d['zgz_identity_ra']:.2f}（差 {d['zgz_delta']:+.2f}）；"
                     f"训练受试者差的中位 {d['train_median_delta']:+.2f}（{d['train_deltas_positive']}/{d['n_recordings']} 条 > 0）→ {d['verdict']}")
    lines += ["", "## 三臂对 C1（E1 / E2 / E3 差，mm；负 = 更好）", ""]
    for arm, e in per_arm.items():
        if e.get("status") != "COMPLETE":
            lines.append(f"- {arm}：{e.get('status')}")
            continue
        for domain in DOMAINS:
            t = e["readouts"][domain]["three"]
            lines.append(f"- {arm} {domain}：" + " / ".join(
                "—" if t[m]["delta"] is None else f"{t[m]['delta']:+.2f}" for m in ("e1", "e2", "e3"))
                + f" → {e['support'][domain]}")
    lines += ["", f"## 机制（越界 vs 回声）：{mechanism['verdict']}", ""]
    (OUT / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
