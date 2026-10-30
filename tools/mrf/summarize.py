#!/usr/bin/env python3
"""MRF gates and summaries (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024), CPU only.

    python tools/mrf/summarize.py r2gate    AGG778 R2 validity per condition -> which MRF B conditions may be read
    python tools/mrf/summarize.py b         validity, contrasts, classes, D triggers -> outputs/mrf/b/summary.json
    python tools/mrf/summarize.py bvertex   per-round probe readouts -> outputs/mrf/b_vertex/summary.json
    python tools/mrf/summarize.py c         interventions -> outputs/mrf/c/summary.json
    python tools/mrf/summarize.py final     D gates, the main table, the verdict -> outputs/mrf/{verdict.json, main_table.md}

Contrasts are paired per endpoint (arm a - arm b, mm E3 RA; positive = a worse), averaged over the probe seeds, with a
95 % block bootstrap (20-endpoint blocks within runs, 2000 resamples). "Material": |mean| >= 1.5 mm (local) / 1.0 mm
(global), interval excludes 0, every seed the same sign (one-seed arms cannot be material). "Negligible": interval
inside +-0.75 / +-0.5 mm and every seed's difference inside that band. D triggers read the selection subjects only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.agg778 import common as C  # noqa: E402
from tools.mrf import common as MC  # noqa: E402

B_DIR = MC.OUT / "b"
MAT = {"local": 1.5, "global": 1.0}
NEG = {"local": 0.75, "global": 0.5}
XCL_E3 = {"A0": {"local": 15.3511, "global": 4.8169}, "A3": {"local": 8.7779, "global": 3.7395}}
DOMS = ("local", "global")
CONDS = (("A3", "e3"), ("A0", "e3"), ("A3", "noise"), ("A0", "noise"))
V5_G = ("N0d", "L3np", "K12np", "Gg", "Gl")
PLAN = {
    "H-U": [("M.L3np", "M.N0d")],
    "total": [("M.L3", "M.N0d")],
    "H-P": [("M.L3", "M.L3np")],
    "H-S": [("M.K12np", "M.L3np"), ("M.Gg", "M.Gl")],
    "desc": [("M.K12", "M.L3"), ("M.Gg", "M.L3"), ("M.K12", "M.K12np"), ("M.N0", "M.N0d"), ("M.Dfix", "M.N0d"),
             ("M.Dfix", "M.L3"), ("M.Lg0", "M.L3")],
    "H-K": [("H.L3", "H.N0d"), ("H.L3np", "H.N0d"), ("H.L3", "H.L3np")],
    "H-C": [(f"{x}.{a}", f"{x}.{b}") for x in ("T6s12", "T6out", "T6ap")
            for a, b in (("L3", "N0d"), ("K12np", "L3"), ("Gg", "L3"))],
    "noise": [("M.L3", "M.N0d"), ("M.L3np", "M.N0d"), ("M.L3", "M.L3np"), ("M.Gg", "M.L3")],
}


# ------------------------------------------------------------------------------------------------ helpers
def load(cond, regime, split):
    """All arms of one (condition, regime) merged over its process groups: {arm: (seeds, n)} mm, plus the meta."""
    arms, meta, own = {}, None, None
    for d in sorted(B_DIR.glob(f"probe/{cond}_{regime}_*")):
        f = d / f"eval_{split}.npz"
        if not f.exists():
            continue
        z = dict(np.load(f, allow_pickle=False))
        names = json.loads((d / "result.json").read_text())["names"]
        m = {k: z[k] for k in ("seq", "si", "k", "rid", "blk")}
        if meta is None:
            meta, own = {**m, "anchor": z["anchor"], "target": z["target"]}, z["mpjpe_ra_mm"][-1]
        elif not all(np.array_equal(meta[k], m[k]) for k in m):
            raise RuntimeError(f"endpoint order differs across groups: {d}")
        for j, n in enumerate(names):
            a, s = n.split("/")
            if int(s) in arms.get(a, {}):
                raise RuntimeError(f"arm {n} appears in two groups")
            arms.setdefault(a, {})[int(s)] = z["mpjpe_ra_mm"][j]
    if meta is None:
        raise FileNotFoundError(f"no probe outputs for {cond} {regime} {split}")
    return {a: np.stack([v[s] for s in sorted(v)]) for a, v in arms.items()}, meta, own


def dom_mask(meta, dom):
    return np.array([("_" + dom) in s for s in meta["seq"]])


def blocks(meta, sel):
    return meta["si"][sel] * 10 ** 6 + meta["rid"][sel] * 10 ** 3 + meta["blk"][sel]


def contrast(arms, meta, a, b, dom):
    sel = dom_mask(meta, dom)
    if not sel.any():
        return None
    da, db = arms[a][:, sel], arms[b][:, sel]
    per_seed = (da - db).mean(1) if len(da) == len(db) else np.array([da.mean() - db.mean()])
    diff = da.mean(0) - db.mean(0)
    ci = C.block_bootstrap({"d": diff}, blocks(meta, sel))["d"]
    return {"mean": ci["mean"], "lo": ci["lo"], "hi": ci["hi"], "per_seed": [float(x) for x in per_seed],
            "a": float(da.mean()), "b": float(db.mean()), "n": int(sel.sum()), "seeds": [len(da), len(db)]}


def classify(c, dom):
    if c is None:
        return "absent"
    seeds = np.array(c["per_seed"])
    if (abs(c["mean"]) >= MAT[dom] and (c["lo"] > 0 or c["hi"] < 0) and min(c["seeds"]) > 1
            and np.all(np.sign(seeds) == np.sign(c["mean"]))):
        return "material"
    if c["lo"] > -NEG[dom] and c["hi"] < NEG[dom] and np.all(np.abs(seeds) < NEG[dom]):
        return "negligible"
    return "undetermined"


def arm_mean(arms, meta, a, dom):
    sel = dom_mask(meta, dom)
    return float(arms[a][:, sel].mean()) if sel.any() else float("nan")


def clean(x):
    """NaN (an absent domain) -> null for JSON."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean(v) for v in x]
    return None if isinstance(x, float) and x != x else x


# ------------------------------------------------------------------------------------------------ R2 precondition
def r2gate():
    """AGG778 R2 validity of the same condition (V2 and, under E3, V3); R2 has no A0 / noise condition."""
    s = json.loads((C.OUT / "r2" / "summary.json").read_text())["validity"]
    out = {}
    for cond, regime in CONDS:
        v = s.get(f"{cond}/{regime}")
        if v is None:
            out[f"{cond}/{regime}"] = {"r2_condition": False, "may_read": True, "note": "no R2 condition"}
            continue
        ok = bool(v.get("V2_calibration")) and (regime != "e3" or bool(v.get("V3_ceiling")))
        out[f"{cond}/{regime}"] = {"r2_condition": True, "r2_validity": v, "may_read": ok}
    C.save_json(MC.OUT / "r2gate.json", out, exclusive=False)
    return out


# ------------------------------------------------------------------------------------------------ Stage B
def stage_b():
    from tools.agg778.r2_summarize import identity_ra
    gate = json.loads((MC.OUT / "r2gate.json").read_text())
    S = {"r2gate": gate, "validity": {}, "v5": {}, "contrasts": {}, "classes": {}, "arm_means": {}, "not_read": []}
    data = {}
    for cond, regime in CONDS:
        if not gate[f"{cond}/{regime}"]["may_read"]:
            S["not_read"].append(f"{cond}/{regime}")
            continue
        for split in ("zgz", "sel"):
            arms, meta, own = load(cond, regime, split)
            meta["identity"] = identity_ra(meta)
            arms["identity"] = meta["identity"][None]
            data[(cond, regime, split)] = (arms, meta, own)
            S["arm_means"][f"{cond}/{regime}/{split}"] = {a: {d: arm_mean(arms, meta, a, d) for d in DOMS} for a in arms}
    # ---------------- validity
    for cond, regime in CONDS:
        if (cond, regime, "zgz") not in data:
            continue
        az, mz, own = data[(cond, regime, "zgz")]
        asel, msel, _ = data[(cond, regime, "sel")]
        v = {"V2_calibration": all(abs(arm_mean(a, m, "T0s.L3", d) - arm_mean(a, m, "identity", d)) <= 0.1
                                   for a, m in ((az, mz), (asel, msel)) for d in DOMS)}
        if regime == "e3":
            v["V1_harness"] = all(abs(float(own[dom_mask(mz, d)].mean()) - XCL_E3[cond][d]) <= 0.01 for d in DOMS)
            v["V3_ceiling"] = all(arm_mean(az, mz, "T6.L3", d) <= arm_mean(az, mz, "identity", d) for d in DOMS)
            dose = contrast(az, mz, "T6s12.L3", "T6.L3", "local")
            v["V4_dose"] = dose is not None and dose["lo"] > 0
            S["contrasts"][f"{cond}/{regime}|V4|T6s12.L3-T6.L3|zgz/local"] = dose
            v5 = {}
            for g in V5_G:
                ok = True
                for split in ("zgz", "sel"):
                    a_, m_, _ = data[(cond, regime, split)]
                    for d in DOMS:
                        c = contrast(a_, m_, f"T6.{g}", "T6.L3", d)
                        S["contrasts"][f"{cond}/{regime}|V5|T6.{g}-T6.L3|{split}/{d}"] = c
                        ok &= classify(c, d) == "negligible"
                v5[g] = bool(ok)
            S["v5"][cond] = v5
        S["validity"][f"{cond}/{regime}"] = {**v, "valid": all(v.values())}
    valid = {k: v["valid"] for k, v in S["validity"].items()}
    # ---------------- contrasts
    for hyp, rows in PLAN.items():
        for cond, regime in CONDS:
            if (regime == "noise") != (hyp == "noise") or (cond, regime, "zgz") not in data:
                continue
            for a, b in rows:
                for split in ("zgz", "sel"):
                    arms, meta, _ = data[(cond, regime, split)]
                    if a not in arms or b not in arms:
                        continue
                    for d in DOMS:
                        c = contrast(arms, meta, a, b, d)
                        key = f"{hyp}|{cond}/{regime}|{a}-{b}|{split}/{d}"
                        S["contrasts"][key] = c
                        S["classes"][key] = classify(c, d)
    S["d_triggers"] = d_triggers(S, valid)
    S = clean(S)
    C.save_json(B_DIR / "summary.json", S, exclusive=False)
    return S


def d_triggers(S, valid):
    """Pre-registered Stage-D arms, from the selection subjects only. `self3` on the C1 recipe always; `noprop` when the
    inpainting effect M.L3 - M.L3np is material (either sign) with the same sign under the noise regime; `ring4np` when
    M.K12np - M.L3np is a material gain; `globalloc` when M.Gg - M.Gl is a material gain. A trigger needs its condition
    valid and V5 for the G variants it compares; the C1 recipe wins when both conditions trigger; at most two optional
    arms, ranked by the selection-subject |effect| (local first)."""
    cls, con = S["classes"], S["contrasts"]

    def material(hyp, cond, regime, a, b, sign=None):
        hits = []
        for d in DOMS:
            key = f"{hyp}|{cond}/{regime}|{a}-{b}|sel/{d}"
            if cls.get(key) == "material" and (sign is None or np.sign(con[key]["mean"]) == sign):
                hits.append((d, con[key]["mean"]))
        return hits

    detail, cand = {}, []
    for arm, (hyp, a, b, sign, gs) in {"noprop": ("H-P", "M.L3", "M.L3np", None, ("L3np",)),
                                       "ring4np": ("H-S", "M.K12np", "M.L3np", -1, ("K12np", "L3np")),
                                       "globalloc": ("H-S", "M.Gg", "M.Gl", -1, ("Gg", "Gl"))}.items():
        per_cond = {}
        for cond in ("A0", "A3"):
            ok = valid.get(f"{cond}/e3", False) and all(S["v5"].get(cond, {}).get(g, False) for g in gs)
            hits = material(hyp, cond, "e3", a, b, sign) if ok else []
            if arm == "noprop" and hits:
                if not valid.get(f"{cond}/noise", False):
                    hits = []
                else:
                    hits = [(d, m) for d, m in hits
                            if np.sign((con.get(f"noise|{cond}/noise|M.L3-M.L3np|sel/{d}") or {"mean": 0})["mean"])
                            == np.sign(m)]
            per_cond[cond] = hits
        detail[arm] = per_cond
        recipe = "a0" if per_cond["A0"] else ("a3" if per_cond["A3"] else None)
        if recipe:
            hits = per_cond[recipe.upper()]
            eff = max((abs(m) / MAT[d], d == "local") for d, m in hits)
            cand.append((eff, f"mrf_{arm}_{recipe}_s3407"))
    cand.sort(reverse=True)
    arms = ["mrf_self3_a0_s3407"] + [n for _, n in cand[:2]]
    return {"arms": arms, "detail": detail, "ranked_optional": [n for _, n in cand], "valid": valid}


# ------------------------------------------------------------------------------------------------ B-vertex
def stage_bvertex(n_boot=2000, seed=20261024):
    out = {}
    corr, r2 = C.corr_from_stats, C.r2_from_stats
    for cond in ("A0", "A3"):
        d = MC.OUT / "b_vertex" / cond
        if not (d / "result.json").exists():
            continue
        r = json.loads((d / "result.json").read_text())
        for split in ("zgz", "sel"):
            z = dict(np.load(d / f"stats_{split}.npz"))
            nb = len(z["blocks"])
            rng = np.random.default_rng(seed)
            draws = rng.integers(0, nb, size=(n_boot, nb))
            for ro in r["readouts"]:
                res = {}
                for key in r["keys"]:
                    fn = corr if key.endswith("normal") else r2

                    def st_of(f):
                        return np.stack([z[f"{f}/{ro}/{s}#{key}"] for s in r["seeds"]], 1)      # (nb, S, 6)
                    per = {}
                    for f in r["feats"]:
                        st = st_of(f)
                        boots = np.array([np.mean(fn(st[dw].sum(0))) for dw in draws])
                        per[f] = {"point": float(np.mean(fn(st.sum(0)))), "lo": float(np.percentile(boots, 2.5)),
                                  "hi": float(np.percentile(boots, 97.5)), "seeds": fn(st.sum(0)).tolist()}

                    def gain(a, b):
                        sa, sb = st_of(a), st_of(b)
                        boots = np.array([np.mean(fn(sa[dw].sum(0))) - np.mean(fn(sb[dw].sum(0))) for dw in draws])
                        pt = float(np.mean(fn(sa.sum(0))) - np.mean(fn(sb.sum(0))))
                        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
                        cl = "material" if (abs(pt) >= 0.05 and (lo > 0 or hi < 0)) else \
                            ("negligible" if lo > -0.02 and hi < 0.02 else "undetermined")
                        return {"point": pt, "lo": lo, "hi": hi, "class": cl}
                    per["gain_H3-H0"] = gain("H3", "H0")
                    per["gain_H1-H0"] = gain("H1", "H0")
                    per["gain_H3-H0near"] = gain("H3", "H0near")
                    per["gain_H3-H0smooth"] = gain("H3", "H0smooth")
                    res[key] = per
                out[f"{cond}/{split}/{ro}"] = clean(res)
    C.save_json(MC.OUT / "b_vertex" / "summary.json", out, exclusive=False)
    return out


# ------------------------------------------------------------------------------------------------ Stage C
def stage_c():
    out = {}
    for d in sorted((MC.OUT / "c").glob("*/result.json")):
        out[d.parent.name] = json.loads(d.read_text())
    C.save_json(MC.OUT / "c" / "summary.json", out, exclusive=False)
    return out


# ------------------------------------------------------------------------------------------------ Stage D and final
def xcl_metrics(run, name):
    return json.loads((C.REPO / "outputs/xcl/20261019/evaluations" / run / name / "report.json").read_text())["metrics"]


def arm_metrics(arm, name):
    return json.loads((MC.OUT / "d" / "evaluations" / arm / name / "report.json").read_text())["metrics"]


def d_gate(arm, control_run):
    """XCL single-seed descriptive rule against one control (arm - control, mm), with the H3=0 guard."""
    keys = {"e1": "e1_step6000_repeat1_none", "e2": "e2_step6000_repeat1_none", "e3": "e3_step6000_repeat1_none",
            "e5_5000": "e1_step5000_repeat1_none", "e5_5500": "e1_step5500_repeat1_none"}
    mine = {k: arm_metrics(arm, v) for k, v in keys.items()}
    ctrl = {k: xcl_metrics(control_run, v) for k, v in keys.items()}
    out = {"control": control_run}
    for metric, dom, thr in (("mpjpe_ra_mm", "local", 5.0), ("mpjpe_ra_mm", "global", 8.0), ("wrist_mm", "local", 10.0),
                             ("wrist_mm", "global", 10.0)):
        d = {k: mine[k][dom][metric] - ctrl[k][dom][metric] for k in keys}
        sgn = np.sign(d["e1"])
        late = [d["e5_5000"], d["e5_5500"], d["e1"]]
        support = (abs(d["e1"]) >= thr and np.sign(d["e2"]) == sgn and np.sign(d["e3"]) == sgn
                   and sum(np.sign(x) == sgn for x in late) >= 2)
        out[f"{metric}/{dom}"] = {"delta": d, "threshold": thr,
                                  "class": "direction_supported" if support else "noise_band",
                                  "direction": "better" if sgn < 0 else "worse"}
    h3z = xcl_metrics(control_run, "e3_step6000_repeat1_h3_zero")
    e3l = mine["e3"]["local"]["mpjpe_ra_mm"]
    out["h3_zero_guard"] = {"arm_e3_local": e3l, "control_h3_zero_e3_local": h3z["local"]["mpjpe_ra_mm"],
                            "only_less_event_reliance": bool(e3l < ctrl["e3"]["local"]["mpjpe_ra_mm"]
                                                             and e3l >= h3z["local"]["mpjpe_ra_mm"])}
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
    label = {"self3": "去掉跨顶点消息与外推（self3）", "noprop": "不外推（noprop）", "ring4np": "4 跳球、不外推（ring4np）",
             "globalloc": "局部注意力对照（global_loc）"}
    old = original.OUT
    original.OUT = Rows()
    try:
        lines = [original.HEADER, original.RULE, original.BASELINE,
                 original.row("dt3_dz_l3_w200cfg", "S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *"),
                 original.row("xyz_c1_baseline", "XYZ C1（seed3407）")]
        if any("_a3_" in a for a in arms):
            lines.append(original.row("xcl_a3f0", "XCL A3F0（seed3407） [oracle]"))
        for arm in arms:
            v, c = arm.split("_")[1], arm.split("_")[2]
            lines.append(original.row(arm.removesuffix("_s3407"), f"MRF {label[v]} on {'C1' if c == 'a0' else 'A3F0'}"
                                      f"（seed3407）" + (" [oracle]" if c == "a3" else "")))
    finally:
        original.OUT = old
    return "\n".join(lines)


def final():
    from tools.agg778.d_common import CONTROLS
    a = json.loads((MC.OUT / "a" / "result.json").read_text())
    b = json.loads((B_DIR / "summary.json").read_text())
    sel = json.loads((MC.OUT / "d" / "selection.json").read_text())
    d = {}
    for arm in sel["arms"]:
        ctrl = CONTROLS[arm.split("_")[2]]["xcl_eval"].rstrip("/").split("/")[-1]
        d[arm] = {"vs_control": d_gate(arm, ctrl)}
        if "globalloc_a0" in arm:
            d[arm]["vs_fplus"] = d_gate(arm, "xag_paired_a0f1_s3407")
    v = {"A_pass": a["pass"], "A_gates": a["gates"], "B_validity": b["validity"], "B_v5": b["v5"],
         "B_not_read": b["not_read"], "D": d}
    v["classes_zgz"] = {k: c for k, c in b["classes"].items() if "|zgz/" in k}
    v["classes_sel"] = {k: c for k, c in b["classes"].items() if "|sel/" in k}
    bv = MC.OUT / "b_vertex" / "summary.json"
    if bv.exists():
        s = json.loads(bv.read_text())
        v["B_vertex_gains"] = {k: {key: {g: s[k][key][g]["class"] for g in s[k][key] if g.startswith("gain")}
                                   for key in s[k]} for k in s}
    C.save_json(MC.OUT / "verdict.json", v, exclusive=False)
    table = main_table(sel["arms"])
    (MC.OUT / "main_table.md").write_text(table + "\n")
    return v, table


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "r2gate":
        print(json.dumps(r2gate(), indent=1))
    elif cmd == "b":
        r = stage_b()
        print(json.dumps({"validity": r["validity"], "v5": r["v5"], "not_read": r["not_read"],
                          "d_triggers": r["d_triggers"]}, indent=1))
    elif cmd == "bvertex":
        stage_bvertex()
        print("ok")
    elif cmd == "c":
        stage_c()
        print("ok")
    elif cmd == "final":
        v, table = final()
        print(table)
        print(json.dumps({k: v[k] for k in ("A_pass", "B_validity", "B_v5")}, indent=1, ensure_ascii=False))
    else:
        raise SystemExit("usage: summarize.py r2gate | b | bvertex | c | final")
