#!/usr/bin/env python3
"""TPD stage ROL readouts and gates (docs/EXPERIMENT_RECORDS.md#tpd-rol-prereg-20261105); CPU only, read-only on the
readings (this study's and PFO2's B2); descriptive except the written rules.

    python -m tools.tpd.analyze_rol plan            preflight step times / peak memory -> decisions/rol_plan.json (rol5's
                                                    formal run is allowed when its median preflight step is <= 8 s)
    python -m tools.tpd.analyze_rol all             seed 3407 -> analysis/rol.md, decisions/rol_summary.json, main rows
    python -m tools.tpd.analyze_rol all2            seed 3408 -> analysis/rol_s2.md, decisions/rol_s2_summary.json
The conditional jobs (rol5; the seed-3408 runs) are wrapped by tools/tpd/maybe.py, which runs them or exits 0 with a
skip record according to these files.

Written rules (RA = overall recursive MPJPE, E3 = single step, mm; R15 / F15 = zgz_local median retention; seed 3407):
  base      B2 3407, unless |RA(rol1) - RA(B2)| > NOISE_RA (0.8): then rol1 (the fine-tune itself moved the baseline)
  screen    (RA <= RA(base) - 0.3, or R15 and F15 both <= base's - 0.05) and E3 <= E3(base) + 0.3 -> SCREEN_PASS
  choose    among rol3 / rol5 / rol3d with SCREEN_PASS (against the base) and RA below rol1's, the lowest RA (ties in that
            order) trains seed 3408 from B2 3408; rol1 seed 3408 trains with it when the base is rol1. No arm chosen:
            CLOSED_LOOP_NOT_MOVED (rol1 passing alone: FINETUNE_ONLY)
  adopt     seed 3408: the chosen arm passes the same screen against B2 3408 (rol1 3408 when rebased) -> ADOPT (the
            rollout becomes the training method of every later arm; a from-scratch 6000-step version goes to the
            synthesis stage), else NOT_CONFIRMED
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import arms_rol as A  # noqa: E402
from tools.tpd.common import B2_COST, NOISE_RA, OUT, PFO2_OUT, S43_ENGINE, b2_tag, now, read, require, save  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
ROLLOUT_ARMS = ("rol3", "rol5", "rol3d")
ROL5_MAX_STEP_S = 8.0
MEM_MAX_GIB = 42.0
SUMMARY = OUT / "decisions" / "rol_summary.json"
SUMMARY2 = OUT / "decisions" / "rol_s2_summary.json"


# --------------------------------------------------------------------------- rules (pure)
def base_for(ra_b2, ra_rol1):
    return "rol1" if abs(ra_rol1 - ra_b2) > NOISE_RA else "b2"


def screen(row, base):
    better_ra = row["ra"] <= base["ra"] - 0.3
    better_ret = row["R15"] <= base["R15"] - 0.05 and row["F15"] <= base["F15"] - 0.05
    e3_ok = row["e3"] <= base["e3"] + 0.3
    return {"verdict": "SCREEN_PASS" if (better_ra or better_ret) and e3_ok else "SCREEN_FAIL",
            "better_ra": bool(better_ra), "better_retention": bool(better_ret), "e3_ok": bool(e3_ok),
            "d_ra": row["ra"] - base["ra"], "d_e3": row["e3"] - base["e3"], "d_R15": row["R15"] - base["R15"],
            "d_F15": row["F15"] - base["F15"]}


def choose(verdicts, ra):
    ok = [a for a in ROLLOUT_ARMS if a in verdicts and verdicts[a]["verdict"] == "SCREEN_PASS" and ra[a] < ra["rol1"]]
    return min(ok, key=lambda a: (ra[a], ROLLOUT_ARMS.index(a))) if ok else None


def gate_rol5(plan):
    return plan["step_s_median"] <= ROL5_MAX_STEP_S


# --------------------------------------------------------------------------- readings
def mech():
    from tools.evf import mech as M
    M.ROOTS.update(tpd=OUT, pfo2=PFO2_OUT)
    return M


def compact(r):
    return {"ra": r["five"][4], "e3": r["e3"], "R15": r["retention"]["R15"], "F15": r["retention"]["F15"],
            "amp": r["amp"][0], "C": r.get("C"), "five": r["five"]}


def fmt(v, d=2):
    return "–" if v is None else f"{v:.{d}f}"


def plan():
    rows = {}
    for arm in A.ARMS:
        d = A.run_dir(arm, 3407, "preflight")
        t = [read(d / f"rol_timing_rank{r}.json") for r in (0, 1)]
        steps = [x["s"] for x in t[0]["steps"][5:]]                       # past the first steps (allocator warm-up)
        rows[arm] = {"step_s_median": statistics.median(steps), "steps": len(t[0]["steps"]),
                     "max_allocated_gib": max(x["max_allocated_gib"] for x in t),
                     "max_reserved_gib": max(x["max_reserved_gib"] for x in t)}
    for arm, r in rows.items():
        r["formal_h_projected"] = r["step_s_median"] * A.FORMAL["MAX_STEPS"] / 3600
        r["memory_ok"] = r["max_reserved_gib"] <= MEM_MAX_GIB
    out = {"rows": rows, "rol5_allowed": gate_rol5(rows["rol5"]), "rol5_max_step_s": ROL5_MAX_STEP_S,
           "mem_max_gib": MEM_MAX_GIB, "written_at_utc": now()}
    save(OUT / "decisions" / "rol_plan.json", out)
    print(json.dumps(out, indent=1))
    return out


def tag(arm, seed):
    return A.run_name(arm, seed)


def main_row(arm, seed, label):
    c = read(B2_COST)
    require(c["policy"] == "default", "the table's cost comes from the PyTorch-default-flag reading")
    t = tag(arm, seed)
    r = read(OUT / "readings" / t / "e1" / "report.json")
    s = r["per_sequence"]
    per = {"0": {"reading": f"tpd:{t}", "checkpoint": r["identity"]["checkpoint"],
                 "local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
                 "overall": {k: r["overall"][k] for k in KEYS}}}
    m = {b: dict(per["0"][b]) for b in ("local", "global", "overall")}
    out = {"arm": t, "label": label, "split": "val_core", "n_seeds": 1, "two_seed_mean": m, "per_seed": per,
           "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": str(B2_COST),
           "written_utc": now(),
           "note": f"TPD ROL row (seed {seed}, PFO's engine E1, deterministic flags); the network is B2's (same "
                   "architecture and deployed router), so the cost columns are B2's deployable form from EVF's final block"}
    p = SEMKINE / f"{t}_main_row.json"
    save(p, out)
    return str(p)


NAMES = {"b2": "B2（PFO2 fgs，参照）", "rol1": "rol1：K = 1 微调（对照）", "rol3d": "rol3d：K = 3，前态截断梯度",
         "rol3": "rol3：K = 3，梯度穿过前态", "rol5": "rol5：K = 5，梯度穿过前态"}


def table(rows, title):
    L = [title, "", f"S43（引擎三种子）{' / '.join(fmt(x) for x in S43_ENGINE)}；单种子噪声底约 {NOISE_RA} mm RA。", "",
         "| 行 | E1 五项 | E3 | 放大 | local R15 / F15 / T10 / T30 | C |", "|---|---|---|---|---|---|"]
    for k, r in rows.items():
        ret = " / ".join(fmt(r["retention"][q]) for q in ("R15", "F15", "T10", "T30")) if "retention" in r else "–"
        L.append(f"| {NAMES.get(k.split('@')[0], k)} | {' / '.join(fmt(x) for x in r['five'])} | {fmt(r.get('e3'))} | "
                 f"{fmt(r['amp'][0]) if 'amp' in r else '–'} | {ret} | {fmt(r.get('C'), 3)} |")
    return L


def all_():
    M = mech()
    rows = {"b2": M.row("pfo2", b2_tag(3407))}
    rol5 = read(OUT / "decisions" / "rol_plan.json")["rol5_allowed"]
    for arm in A.ARMS:
        if arm != "rol5" or rol5:
            rows[arm] = M.row("tpd", tag(arm, 3407))
            require(all(k in rows[arm] for k in ("five", "e3", "retention", "C")), "readings missing: " + arm)
    c = {k: compact(v) for k, v in rows.items()}
    base_name = base_for(c["b2"]["ra"], c["rol1"]["ra"])
    base = c[base_name]
    verdicts = {a: screen(c[a], base) for a in c if a not in ("b2", base_name)}
    if base_name == "rol1":
        verdicts["rol1"] = {"verdict": "BASE", **screen(c["rol1"], c["b2"])}
    ra = {k: v["ra"] for k, v in c.items()}
    chosen = choose(verdicts, ra)
    rol1_vs_b2 = screen(c["rol1"], c["b2"])
    outcome = ("CHOSEN" if chosen else "FINETUNE_ONLY" if rol1_vs_b2["verdict"] == "SCREEN_PASS"
               else "CLOSED_LOOP_NOT_MOVED")
    mech_rol3 = {"amp_below_1p8": c["rol3"]["amp"] < 1.8, "C_below_0p7": (c["rol3"]["C"] or 9) < 0.7} if "rol3" in c else {}
    L = table(rows, "## TPD ROL 闭环展开训练（种子 3407；PFO 引擎；描述 + 写定规则）")
    L += ["", f"基准：**{base_name}**（RA(rol1) − RA(B2) = {fmt(ra['rol1'] - ra['b2'])}，噪声底 {NOISE_RA}）。"]
    for a, v in verdicts.items():
        L.append(f"- {a}：**{v['verdict']}**（ΔRA {fmt(v['d_ra'])}，ΔE3 {fmt(v['d_e3'])}，ΔR15 {fmt(v['d_R15'])}，"
                 f"ΔF15 {fmt(v['d_F15'])}）")
    L += [f"- 选中补种子 3408：**{chosen or '无'}**（结果 {outcome}）；rol1 对 B2：{rol1_vs_b2['verdict']}（描述）。",
          f"- rol3 − rol3d = {fmt(ra.get('rol3', float('nan')) - ra.get('rol3d', float('nan')))}（梯度穿过前态的价值，描述）；"
          f"rol3 机制预期：{mech_rol3}。", "", "单种子；标签不进任何行。"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "rol.md").write_text(text)
    main_rows = [main_row(a, 3407, f"TPD {a}：{NAMES[a].split('：')[1]}（种子 3407，可部署）") for a in A.ARMS
                 if a in rows]
    summary = {"rows": rows, "compact": c, "base": base_name, "verdicts": verdicts, "chosen": chosen,
               "train_rol1_s3408": bool(chosen and base_name == "rol1"), "outcome": outcome,
               "rol1_vs_b2": rol1_vs_b2, "mechanism_rol3": mech_rol3, "noise_ra": NOISE_RA,
               "s43_engine": list(S43_ENGINE), "main_rows": main_rows, "written_at_utc": now()}
    save(SUMMARY, summary, exclusive=True)
    print(text)
    return summary


def all2():
    s = read(SUMMARY)
    if not s["chosen"]:
        save(SUMMARY2, {"arm": None, "verdict": "NONE_CHOSEN", "outcome": s["outcome"], "written_at_utc": now()},
             exclusive=True)
        print("no arm chosen for seed 3408:", s["outcome"])
        return
    M = mech()
    arm = s["chosen"]
    rows = {"b2@3408": M.row("pfo2", b2_tag(3408)), f"{arm}@3408": M.row("tpd", tag(arm, 3408))}
    if s["base"] == "rol1":
        rows["rol1@3408"] = M.row("tpd", tag("rol1", 3408))
    c = {k: compact(v) for k, v in rows.items()}
    base = c["rol1@3408" if s["base"] == "rol1" else "b2@3408"]
    v = screen(c[f"{arm}@3408"], base)
    verdict = "ADOPT" if v["verdict"] == "SCREEN_PASS" else "NOT_CONFIRMED"
    L = table(rows, "## TPD ROL 第二种子（3408；写定规则）")
    L += ["", f"{arm} 种子 3408 对 {'rol1' if s['base'] == 'rol1' else 'B2'} 3408：{v['verdict']}（ΔRA {fmt(v['d_ra'])}，"
          f"ΔE3 {fmt(v['d_e3'])}，ΔR15 {fmt(v['d_R15'])}，ΔF15 {fmt(v['d_F15'])}）→ **{verdict}**。"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis" / "rol_s2.md").write_text(text)
    main = main_row(arm, 3408, f"TPD {arm}：{NAMES[arm].split('：')[1]}（种子 3408，可部署）")
    save(SUMMARY2, {"arm": arm, "rows": rows, "compact": c, "screen": v, "verdict": verdict, "main_row": main,
                    "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    for verb in ("plan", "all", "all2"):
        sub.add_parser(verb)
    a = p.parse_args(argv)
    {"plan": plan, "all": all_, "all2": all2}[a.verb]()


if __name__ == "__main__":
    main()
