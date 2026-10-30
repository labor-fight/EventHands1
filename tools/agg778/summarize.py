#!/usr/bin/env python3
"""AGG778 gates and verdict (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024), CPU only.

    python tools/agg778/summarize.py b        Stage-B gates (zgz, report only) + selection-subject factor effects
                                              -> outputs/agg778/b_summary.json (the Stage-D selection input)
    python tools/agg778/summarize.py final    Stage A / B / C / D verdict + the nine-column table
                                              -> outputs/agg778/verdict.json, outputs/agg778/main_table.md

All contrasts are paired (same endpoints, same anchors), averaged over the three probe seeds, with 95 % block-bootstrap
intervals (20-endpoint blocks within runs, 2000 resamples, seed 20261024).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402

B_TASK = C.OUT / "b_task"
B_VERT = C.OUT / "b_vertex"
SEEDS = (0, 1, 2)
#: pre-registered thresholds
TASK_MAT = {"local": 1.5, "global": 1.0}
TASK_NEG = {"local": 0.75, "global": 0.5}
VERT_MAT, VERT_NEG, NULL_MAX = 0.05, 0.02, 0.05
IDENTITY_E3_LOCAL = 4.41
#: XCL strict E3 of the frozen modules (outputs/xcl/20261019/evaluations/*/e3_step6000_repeat1_none/report.json)
XCL_E3 = {"A0": {"local": 15.3511, "global": 4.8169}, "A3": {"local": 8.7779, "global": 3.7395}}
P4_TOL = 0.02
FACTORS = {"matpt": ("V5", "V6"), "mass": ("V5", "V7"), "fanout": ("V5", "V8"), "vin": ("V3", "V4")}
TASK_CONTRASTS = {"A3": {"G_learn": ("T2", "T5"), "G_rep": ("T4", "T5"), "G_vin": ("T2", "T3"),
                         "G_rec": ("T5", "T6"), "G_total": ("T2", "T6"), "event_info": ("T0", "T2")},
                  "A0": {"G_vin": ("T2", "T3"), "G_total": ("T2", "T6"), "event_info": ("T0", "T2")}}


def load_task(cond, readout):
    z = np.load(B_TASK / cond / f"eval_{readout}.npz", allow_pickle=False)
    names = json.loads((B_TASK / cond / "result.json").read_text())["names"]
    return z, names


def task_contrast(z, names, a, b, domain, metric="mpjpe_ra_mm"):
    """Paired per-endpoint difference (arm a - arm b), seed-averaged, within one zgz / selection domain."""
    seqs = z["seq"]
    sel = np.array([("_" + domain) in s for s in seqs])
    idx = {n: i for i, n in enumerate(names)}
    M = z[metric]
    da = np.stack([M[idx[f"{a}/{s}"], sel] for s in SEEDS])
    db = np.stack([M[idx[f"{b}/{s}"], sel] for s in SEEDS])
    diff = (da - db).mean(0)
    blocks = z["si"][sel] * 10 ** 6 + z["rid"][sel] * 10 ** 3 + z["blk"][sel]
    ci = C.block_bootstrap({"d": diff}, blocks)["d"]
    per_seed = (da - db).mean(1)
    return {"mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "per_seed": per_seed.tolist(),
            "a_mean": float(da.mean()), "b_mean": float(db.mean()), "n": int(sel.sum())}


def classify_task(c, domain, noise=None):
    mat, neg = TASK_MAT[domain], TASK_NEG[domain]
    same = all(np.sign(c["per_seed"]) == np.sign(c["mean"])) and c["mean"] != 0
    excl = c["lo"] > 0 or c["hi"] < 0
    co = noise is None or np.sign(noise["mean"]) == np.sign(c["mean"])
    if abs(c["mean"]) >= mat and excl and same and co:
        return "material"
    if c["lo"] > -neg and c["hi"] < neg:
        return "negligible"
    return "undetermined"


def task_arm_mean(z, names, arm, domain, metric="mpjpe_ra_mm"):
    sel = np.array([("_" + domain) in s for s in z["seq"]])
    idx = {n: i for i, n in enumerate(names)}
    return float(np.mean([z[metric][idx[f"{arm}/{s}"], sel].mean() for s in SEEDS]))


def load_vert(cond, split, regime):
    return dict(np.load(B_VERT / cond / f"stats_{split}_{regime}.npz", allow_pickle=False))


def vert_corr(st, arm, readout, key, blocks_sel=None):
    """Seed-averaged correlation with a block-bootstrap interval; per-seed values."""
    mats = [st[f"{arm}/{readout}/{s}#{key}"] for s in SEEDS]
    per_seed = [float(C.corr_from_stats(m.sum(0))) for m in mats]
    stack = np.stack(mats, 1)                                                    # (blocks, seeds, 6)
    point, lo, hi = C.bootstrap_stats(stack, lambda s: np.mean(C.corr_from_stats(s)))
    return {"mean": float(point), "lo": float(lo), "hi": float(hi), "per_seed": per_seed}


def vert_contrast(st, a, b, readout, key):
    ma = np.stack([st[f"{a}/{readout}/{s}#{key}"] for s in SEEDS], 1)
    mb = np.stack([st[f"{b}/{readout}/{s}#{key}"] for s in SEEDS], 1)
    stack = np.stack([ma, mb], 1)                                                # (blocks, 2, seeds, 6)
    fn = lambda s: float(np.mean(C.corr_from_stats(s[1]) - C.corr_from_stats(s[0])))  # noqa: E731  (b - a)
    point, lo, hi = C.bootstrap_stats(stack, fn)
    per_seed = [float(C.corr_from_stats(mb[:, i].sum(0)) - C.corr_from_stats(ma[:, i].sum(0))) for i in range(3)]
    return {"mean": float(point), "lo": float(lo), "hi": float(hi), "per_seed": per_seed}


def classify_vert(c):
    same = all(np.sign(c["per_seed"]) == np.sign(c["mean"])) and c["mean"] != 0
    if abs(c["mean"]) >= VERT_MAT and (c["lo"] > 0 or c["hi"] < 0) and same:
        return "material"
    if c["lo"] > -VERT_NEG and c["hi"] < VERT_NEG:
        return "negligible"
    return "undetermined"


KEY = "own|con|normal"


def stage_b():
    out = {"task": {}, "vertex": {}, "validity": {}, "validity_values": {}, "selection_factors": {}}
    # ---------------- B-task (zgz report)
    for cond in ("A0", "A3"):
        e3, names = load_task(cond, "zgz_e3")
        nz, _ = load_task(cond, "zgz_noise")
        res = {}
        for cname, (a, b) in TASK_CONTRASTS[cond].items():
            for dom in ("local", "global"):
                c = task_contrast(e3, names, a, b, dom)
                cn = task_contrast(nz, names, a, b, dom)
                extra = {}
                for ro in ("zgz_e2", "zgz_e1"):
                    if (B_TASK / cond / f"eval_{ro}.npz").exists():
                        zz, _ = load_task(cond, ro)
                        extra[ro] = task_contrast(zz, names, a, b, dom)
                res[f"{cname}/{dom}"] = {"e3": c, "noise": cn, **extra, "class": classify_task(c, dom, cn)}
        arms_e3 = {arm: {dom: task_arm_mean(e3, names, arm, dom) for dom in ("local", "global")}
                   for arm in sorted({n.split("/")[0] for n in names})}
        res["arm_means_e3"] = arms_e3
        # frozen module's own single step from the same anchors (validity: reproduces XCL E3)
        fro = {dom: float(e3["mpjpe_ra_mm"][-1, np.array([("_" + dom) in s for s in e3["seq"]])].mean())
               for dom in ("local", "global")}
        res["frozen_own_e3"] = fro
        mono = [arms_e3[a]["local"] for a in ("T6", "T6s3", "T6s6", "T6s12")]
        c12 = task_contrast(e3, names, "T6s12", "T6", "local")
        xcl = XCL_E3[cond]
        out["validity"][f"task/{cond}"] = {
            "ceiling_below_identity": arms_e3["T6"]["local"] <= IDENTITY_E3_LOCAL,
            "dose_response_monotone": bool(np.all(np.diff(mono) > 0)),
            "dose_response_resolved": c12["lo"] > 0,
            "harness_reproduces_xcl_e3": all(abs(fro[d] - xcl[d]) <= 0.01 for d in ("local", "global"))}
        out["validity_values"][f"task/{cond}"] = {"dose_response_12mm": c12, "T6_ladder_local": mono,
                                                   "frozen_own_e3": fro, "xcl_e3": xcl}
        out["task"][cond] = res
    # ---------------- B-vertex (zgz report)
    for cond in ("A0", "A3"):
        res = {}
        for regime in ("e3", "noise", "loop"):
            st = load_vert(cond, "zgz", regime)
            r = {}
            for f, (a, b) in FACTORS.items():
                for ro in ("mlp", "gnn"):
                    c = vert_contrast(st, a, b, ro, KEY)
                    r[f"{f}/{ro}"] = {**c, "class": classify_vert(c)}
            for ro in ("mlp", "gnn"):
                if f"V9a/{ro}/0#{KEY}" in st:
                    c = vert_contrast(st, "V3", "V9a", ro, KEY)
                    r[f"ceiling_gap_V9a_vs_V3/{ro}"] = {**c, "class": classify_vert(c)}
            c = vert_contrast(st, "V9a", "V9b", "mlp", KEY)
            r["event_budget_V9b_vs_V9a/mlp"] = {**c, "class": classify_vert(c)}
            c = vert_contrast(st, "V4", "V5", "mlp", KEY)
            r["phi_retrain_V5_vs_V4/mlp"] = {**c, "class": classify_vert(c)}
            r["arms"] = {f"{a}/{ro}": vert_corr(st, a, ro, KEY) for a in ("V0", "V1", "V2", "V3", "V4", "V5", "V6", "V7",
                                                                        "V8", "V9a", "V9b", "V10a", "V10b")
                         for ro in ("mlp", "gnn") if f"{a}/{ro}/0#{KEY}" in st}
            res[regime] = r
            if regime in ("e3", "noise"):
                out["validity"][f"vertex/{cond}/{regime}"] = {
                    "null_counts": abs(res[regime]["arms"]["V0/mlp"]["mean"]) <= NULL_MAX,
                    "null_donor": abs(res[regime]["arms"]["V2/mlp"]["mean"]) <= NULL_MAX}
        out["vertex"][cond] = res
    p4 = json.loads((B_VERT / "p4repro" / "result.json").read_text())["p4repro"]
    out["validity"]["p4repro"] = {s: abs(p4[s]["diff"]) <= P4_TOL for s in C.ZGZ}
    out["validity_values"]["p4repro"] = p4
    # ---------------- selection-subject factor effects (the only input of the Stage-D arm choice)
    for f, (a, b) in FACTORS.items():
        cells = []
        for cond in ("A0", "A3"):
            for regime in ("e3", "noise"):
                st = load_vert(cond, "sel", regime)
                for ro in ("mlp", "gnn"):
                    cells.append(vert_contrast(st, a, b, ro, KEY))
        eff = float(np.mean([c["mean"] for c in cells]))
        excl = sum(c["lo"] > 0 for c in cells)
        out["selection_factors"][f] = {"effect": eff, "cells": cells, "cells_ci_above_zero": excl,
                                       "material": eff >= VERT_MAT and excl >= len(cells) // 2 + 1}
    out["valid"] = all(all(bool(v) for v in d.values()) for d in out["validity"].values())
    C.save_json(C.OUT / "b_summary.json", out, exclusive=False)
    return out


def main_table(arms):
    from tools import report_table as original
    rows = {"dt3_dz_l3_w200cfg": C.REPO / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json",
            "xyz_c1_baseline": C.REPO / "outputs/semkine/xyz_c1_baseline_main_row.json",
            "xcl_a3f0": C.REPO / "outputs/xcl/20261019/evaluations/xcl_a3f0_s3407/row_step6000_repeat1_none/"
                                 "xcl_a3f0_main_row_step6000.json"}
    for arm in arms:
        rows[arm.removesuffix("_s3407")] = C.REPO / "outputs/semkine" / f"{arm.removesuffix('_s3407')}_main_row.json"

    class Rows:
        def __truediv__(self, filename):
            return rows[str(filename).removesuffix("_main_row.json")]
    old = original.OUT
    original.OUT = Rows()
    try:
        lines = [original.HEADER, original.RULE, original.BASELINE,
                 original.row("dt3_dz_l3_w200cfg", "S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *"),
                 original.row("xyz_c1_baseline", "XYZ C1（seed3407）"),
                 original.row("xcl_a3f0", "XCL A3F0（seed3407） [oracle]")]
        for arm in arms:
            v, c = arm.split("_")[1], arm.split("_")[2]
            lines.append(original.row(arm.removesuffix("_s3407"),
                                      f"AGG778 {v} on {'C1' if c == 'a0' else 'A3F0'}（seed3407）" +
                                      (" [oracle]" if c == "a3" else "")))
    finally:
        original.OUT = old
    return "\n".join(lines)


def xcl_report(cond, name):
    from tools.agg778.d_common import CONTROLS
    r = json.loads((C.REPO / CONTROLS[cond]["xcl_eval"] / name / "report.json").read_text())
    return r["metrics"]


def arm_report(arm, name):
    r = json.loads((C.REPO / "outputs/agg778/d/evaluations" / arm / name / "report.json").read_text())
    return r["metrics"]


def d_gate(arm):
    """XCL single-seed descriptive rule for one arm against its paired control (arm - control, mm)."""
    cond = arm.split("_")[2]
    keys = {"e1": "e1_step6000_repeat1_none", "e2": "e2_step6000_repeat1_none", "e3": "e3_step6000_repeat1_none",
            "e5_5000": "e1_step5000_repeat1_none", "e5_5500": "e1_step5500_repeat1_none"}
    mine = {k: arm_report(arm, v) for k, v in keys.items()}
    ctrl = {k: xcl_report(cond, v) for k, v in keys.items()}
    h3z = xcl_report(cond, "e3_step6000_repeat1_h3_zero")
    out = {}
    for metric, dom, thr in (("mpjpe_ra_mm", "local", 5.0), ("mpjpe_ra_mm", "global", 8.0), ("wrist_mm", "local", 10.0),
                             ("wrist_mm", "global", 10.0)):
        d = {k: mine[k][dom][metric] - ctrl[k][dom][metric] for k in keys}
        sgn = np.sign(d["e1"])
        late = [d["e5_5000"], d["e5_5500"], d["e1"]]
        support = (abs(d["e1"]) >= thr and np.sign(d["e2"]) == sgn and np.sign(d["e3"]) == sgn
                   and sum(np.sign(x) == sgn for x in late) >= 2)
        out[f"{metric}/{dom}"] = {"delta": d, "threshold": thr, "class": "direction_supported" if support else "noise_band",
                                  "direction": "better" if sgn < 0 else "worse"}
    e3_local = mine["e3"]["local"]["mpjpe_ra_mm"]
    out["h3_zero_guard"] = {"arm_e3_local": e3_local, "control_h3_zero_e3_local": h3z["local"]["mpjpe_ra_mm"],
                            "only_less_event_reliance": bool(e3_local - ctrl["e3"]["local"]["mpjpe_ra_mm"] < 0
                                                             and e3_local >= h3z["local"]["mpjpe_ra_mm"])}
    return out


def final():
    a = json.loads((C.OUT / "a" / "result.json").read_text())
    b = json.loads((C.OUT / "b_summary.json").read_text())
    c_path = C.OUT / "c" / "result.json"
    c = json.loads(c_path.read_text()) if c_path.exists() else None
    sel_path = C.OUT / "d" / "selection.json"
    d = {}
    if sel_path.exists():
        for arm in json.loads(sel_path.read_text())["arms"]:
            d[arm] = d_gate(arm)
    v = {"A_pass": a["pass"], "B_valid": b["valid"], "C_rate_flag": None if c is None else c.get("rate_flag"), "D": d}
    t3 = b["task"]["A3"]
    v["H_impl"] = "defect" if not a["pass"] else "correct"
    v["H_learn"] = {dom: t3[f"G_learn/{dom}"]["class"] for dom in ("local", "global")}
    v["H_rep"] = {dom: t3[f"G_rep/{dom}"]["class"] for dom in ("local", "global")}
    v["H_rec"] = {dom: t3[f"G_rec/{dom}"]["class"] for dom in ("local", "global")}
    v["H_vin"] = {cond: {dom: b["task"][cond][f"G_vin/{dom}"]["class"] for dom in ("local", "global")} for cond in ("A0", "A3")}
    v["factors_zgz"] = {cond: {reg: {k: x["class"] for k, x in b["vertex"][cond][reg].items() if "class" in x}
                               for reg in b["vertex"][cond]} for cond in ("A0", "A3")}
    C.save_json(C.OUT / "verdict.json", v, exclusive=False)
    table = main_table(list(d))
    (C.OUT / "main_table.md").write_text(table + "\n")
    return v, table


if __name__ == "__main__":
    if sys.argv[1:] == ["final"]:
        v, table = final()
        print(table)
        print(json.dumps({k: v[k] for k in v if k != "D"}, indent=1, ensure_ascii=False))
    elif sys.argv[1:] == ["b"]:
        r = stage_b()
        print(json.dumps({"valid": r["valid"], "selection_factors": {k: {kk: vv for kk, vv in v.items() if kk != "cells"}
                                                                     for k, v in r["selection_factors"].items()}},
                         indent=1))
    else:
        raise SystemExit("usage: summarize.py b | final")
