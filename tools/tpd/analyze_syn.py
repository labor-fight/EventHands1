#!/usr/bin/env python3
"""TPD synthesis readouts and gates (docs/EXPERIMENT_RECORDS.md#tpd-syn-prereg-20261105); CPU only, read-only on the readings.

    python -m tools.tpd.analyze_syn screen   decisions/syn_screen.json (once): seed 3407, the chosen rollout arm X against c0
    python -m tools.tpd.analyze_syn keep     decisions/syn_keep.json (once): seed 3408, the same rule -> the final arm
    python -m tools.tpd.analyze_syn final    decisions/syn_final.json, analysis/syn.md, main rows (once): the final gate

Written rules (RA = overall recursive MPJPE, E3 = single step, R15 / F15 = zgz_local median retention; every arm trained
from scratch in the causal form with L1's training routes and read in the causal form with the deployed router r_s2d):
  screen  X passes against c0 when (RA <= RA(c0) - 0.3, or R15 and F15 both <= c0's - 0.05) and E3 <= E3(c0) + 0.3, and
          ROL was adopted (decisions/rol_s2_summary.json verdict ADOPT); then X and c0 train seed 3408
  keep    seed 3408: X passes against c0 by the same rule -> the final arm is X (it trains seed 3409); otherwise none
          (the synthesis has no item left: stop point)
  final   the final arm's three seeds: the three-seed means of MPJPE-L / MPJPE-G / MPVPE-L / MPVPE-G / RA all below S43's
          engine values (12.81 / 8.02 / 10.77 / 6.64 / 10.24) and every seed's RA below 11.0 -> FINAL_PASS, else
          FINAL_FAIL (reported with the gap per metric); never the number of metrics below S43
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import arms_final as A  # noqa: E402
from tools.tpd.analyze_rol import compact, screen as screen_rule  # noqa: E402
from tools.tpd.common import B2_COST, OUT, S43_ENGINE, now, read, require, save  # noqa: E402

SEMKINE = ROOT / "outputs/semkine"
KEYS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")
RA_SEED_MAX = 11.0


def mech():
    from tools.evf import mech as M
    M.ROOTS.update(tpd=OUT)
    return M


def chosen_arm():
    c = read(OUT / "decisions" / "rol_summary.json")["chosen"]
    return A.OF_ROL[c] if c else None


def final_rule(rows_by_seed):
    five = [sum(r["five"][i] for r in rows_by_seed.values()) / len(rows_by_seed) for i in range(5)]
    below = [five[i] < S43_ENGINE[i] for i in range(5)]
    seeds_ok = {s: r["five"][4] < RA_SEED_MAX for s, r in rows_by_seed.items()}
    ok = all(below) and all(seeds_ok.values())
    return {"verdict": "FINAL_PASS" if ok else "FINAL_FAIL", "mean_five": five, "below_s43": below,
            "gap_to_s43": [five[i] - S43_ENGINE[i] for i in range(5)], "seed_ra_below_11": seeds_ok}


def screen():
    out = OUT / "decisions" / "syn_screen.json"
    require(not out.exists(), "written once")
    x = chosen_arm()
    require(x is not None, "no ROL arm chosen")
    M = mech()
    rx, r0 = M.row("tpd", A.run_name(x, 3407)), M.row("tpd", A.run_name("c0", 3407))
    v = screen_rule(compact(rx), compact(r0))
    adopt = read(OUT / "decisions" / "rol_s2_summary.json")["verdict"] == "ADOPT"
    go = v["verdict"] == "SCREEN_PASS" and adopt
    save(out, {"arm": x, "screen": v, "rol_adopt": adopt, "seed3408_arms": [x, "c0"] if go else [],
               "rows": {x: rx, "c0": r0}, "written_at_utc": now()}, exclusive=True)
    print(x, v["verdict"], "ROL adopt", adopt, "-> seed 3408" if go else "-> stop")


def keep():
    out = OUT / "decisions" / "syn_keep.json"
    require(not out.exists(), "written once")
    s = read(OUT / "decisions" / "syn_screen.json")
    x = s["arm"]
    final = None
    v = None
    rows = {}
    if s["seed3408_arms"]:
        M = mech()
        rows = {x: M.row("tpd", A.run_name(x, 3408)), "c0": M.row("tpd", A.run_name("c0", 3408))}
        v = screen_rule(compact(rows[x]), compact(rows["c0"]))
        final = x if v["verdict"] == "SCREEN_PASS" else None
    save(out, {"arm": x, "screen_3408": v, "final_arm": final, "rows": rows, "written_at_utc": now()}, exclusive=True)
    print("final arm:", final)


def main_row(arm, label):
    c = read(B2_COST)
    per, ms = {}, []
    for i, seed in enumerate(A.SEEDS):
        t = A.run_name(arm, seed)
        r = read(OUT / "readings" / t / "e1" / "report.json")
        s = r["per_sequence"]
        per[str(i)] = {"reading": f"tpd:{t}", "checkpoint": r["identity"]["checkpoint"],
                       "local": {k: s["zgz_local"][k] for k in KEYS}, "global": {k: s["zgz_global"][k] for k in KEYS},
                       "overall": {k: r["overall"][k] for k in KEYS}}
    mean = {b: {k: sum(per[i][b][k] for i in per) / len(per) for k in KEYS} for b in ("local", "global", "overall")}
    out = {"arm": f"tpd_{arm}", "label": label, "split": "val_core", "n_seeds": len(per), "two_seed_mean": mean,
           "per_seed": per, "latency_ms_raw": c["latency_row"]["raw_ms"], "latency_anchor_ms": c["latency_row"]["anchor_ms"],
           "latency_ms_scaled_full1p75": c["latency_row"]["scaled_ms"], "macs_forward_packet": c["macs_layers"]["median"],
           "params_total": c["params_total"], "latency_e2e_ms": c["latency_e2e_ms"], "cost_reading": str(B2_COST),
           "written_utc": now(),
           "note": "TPD final configuration (three seeds, PFO's engine E1 in the causal deployment form, deterministic "
                   "flags); cost columns are B2's deployable form until the final cost block replaces them"}
    p = SEMKINE / f"tpd_{arm}_main_row.json"
    save(p, out)
    return str(p)


def final():
    out = OUT / "decisions" / "syn_final.json"
    require(not out.exists(), "written once")
    k = read(OUT / "decisions" / "syn_keep.json")
    arm = k["final_arm"]
    require(arm is not None, "no final arm")
    M = mech()
    rows = {s: M.row("tpd", A.run_name(arm, s)) for s in A.SEEDS}
    verdict = final_rule(rows)
    names = ("MPJPE-L", "MPJPE-G", "MPVPE-L", "MPVPE-G", "RA")
    L = [f"## TPD 最终配置 {arm}（三种子；因果部署形式；写定的最终门）", "",
         "| 种子 | " + " | ".join(names) + " | E3 | 放大 | R15 / F15 | C |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s, r in rows.items():
        L.append(f"| {s} | " + " | ".join(f"{x:.2f}" for x in r["five"]) + f" | {r['e3']:.2f} | {r['amp'][0]:.2f} | "
                 f"{r['retention']['R15']:.2f} / {r['retention']['F15']:.2f} | {r['C']:.3f} |")
    L.append("| 均值 | " + " | ".join(f"{x:.2f}" for x in verdict["mean_five"]) + " | | | | |")
    L.append("| S43 | " + " | ".join(f"{x:.2f}" for x in S43_ENGINE) + " | | | | |")
    L += ["", f"判读：**{verdict['verdict']}**；对 S43 差 " + " / ".join(f"{g:+.2f}" for g in verdict["gap_to_s43"]) +
          f"；每种子 RA < 11.0：{verdict['seed_ra_below_11']}。"]
    text = "\n".join(L) + "\n"
    (OUT / "analysis" / "syn.md").write_text(text)
    row = main_row(arm, f"TPD 最终配置 {arm}（三种子，因果部署形式）")
    save(out, {"arm": arm, "rows": rows, **verdict, "main_row": row, "written_at_utc": now()}, exclusive=True)
    print(text)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("screen", "keep", "final"))
    a = p.parse_args(argv)
    {"screen": screen, "keep": keep, "final": final}[a.verb]()


if __name__ == "__main__":
    main()
