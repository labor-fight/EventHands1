#!/usr/bin/env python3
"""T1A-UB round-2 summary (CPU only; once, after every round-2 reading and the round's main rows; single seed, descriptive).

    python tools/t1aub/decide2.py --round 2 --block B     -> outputs/t1aub/20261102/summary_r2.json, report_r2.md

Written rules of docs/EXPERIMENT_RECORDS.md#t1aub-r2-prereg-20261102. Per trunk, every arbitrated cell against the trunk's
part-first cell (the router alone): in-sample tl_e1 at the trunk's steps 5000 / 5500 / 6000, C, zgz (XCL single-seed rule,
descriptive); the class ARB_HELPS / ARB_NULL / ARB_HURTS / MIXED; the budget (latency against T1a re-measured in block B);
then the best deployable network over rounds 1-2 and the written next step. Uses tools/t1aub/decide.py's readings unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402
from tools.t1aub import decide as D  # noqa: E402
from tools.t1aub import r2  # noqa: E402


def classify(gains, c_var, c_pf):
    v = list(gains.values())
    if all(x >= D.TH["insample"] for x in v) and c_var <= D.TH["C"]:
        return "ARB_HELPS"
    if all(x <= -D.TH["insample"] for x in v) or (c_var > D.TH["C"] >= c_pf):
        return "ARB_HURTS"
    if all(abs(x) < D.TH["insample"] for x in v) and c_var <= D.TH["C"]:
        return "ARB_NULL"
    return "MIXED"


def round2(block):
    reg = r2.registry()
    cells = {c: s for c, s in reg["cells"].items() if s["round"] == 2}
    R = {c: D.readings(c, block) for c in cells}
    R["t1a"] = D.readings("t1a", block)
    ref_lat = R["t1a"]["row"][1]["latency_ms"]
    out = {"round": 2, "block": block, "t1aref_latency_ms": ref_lat, "cells": {}, "arbitration": {}}
    for c in cells:
        row, cost = R[c]["row"]
        ok, parts = D.budget(cost, ref_lat)
        out["cells"][c] = {"spec": cells[c], "row": D.row_metrics(row),
                           "cost": {k: cost[k] for k in ("params_total", "macs_total", "latency_ms")},
                           "tail_ms": cost["tail_scaled_ms"]["per_packet_min_over_passes_ms"], "budget_ok": ok,
                           "budget_parts": parts, "zgz": R[c]["zgz"],
                           "insample_median": {s: D.med(R[c]["insample"][s]) for s in C.STEPS},
                           "C": R[c]["fixed_point"]["C"], "keep_correct_gt_ra20": R[c]["fixed_point"]["gt"]["ra20"],
                           "single_step": R[c]["single_step"], "retention": R[c]["retention"],
                           "vs_t1a_insample_detected": D.insample_detected(R["t1a"], R[c])}
    for trunk in sorted({s["trunk"] for s in cells.values()}):
        group = {c: s for c, s in cells.items() if s["trunk"] == trunk}
        pf = [c for c, s in group.items() if s["arb"]["mode"] == "point" and s["arb"].get("sigma_px") is None]
        C.require(len(pf) == 1, "one part-first reference per trunk")
        pf = pf[0]
        res = {"reference": pf}
        for c in group:
            if c == pf:
                continue
            gains = D.diffs(R[c], R[pf])
            res[c] = {"insample_gain_vs_pf": gains, "C": R[c]["fixed_point"]["C"], "C_pf": R[pf]["fixed_point"]["C"],
                      "zgz_support_local_vs_pf": D.zgz_support(R[pf], R[c], "local")[0],
                      "class": classify(gains, R[c]["fixed_point"]["C"], R[pf]["fixed_point"]["C"])}
        out["arbitration"][trunk] = res
    s1 = C.read(C.OUT / "summary_r1.json")
    pool = {n: {"ra": v["row"]["ra"], "C": v["C"], "budget_ok": v["budget_ok"], "round": 1}
            for n, v in s1["networks"].items() if n in ("t0ps16_srdwa", "t0pnoeg20_srdwa")}
    pool.update({n: {"ra": v["row"]["ra"], "C": v["C"], "budget_ok": v["budget_ok"], "round": 2}
                 for n, v in out["cells"].items()})
    ok = {n: v for n, v in pool.items() if v["budget_ok"] and v["C"] <= D.TH["C"]}
    best = min(ok, key=lambda n: (ok[n]["ra"], ok[n]["C"])) if ok else None
    helps = [(t, c) for t, g in out["arbitration"].items() for c, v in g.items() if c != "reference" and v["class"] == "ARB_HELPS"]
    over = [c for c, v in out["cells"].items() if not v["budget_ok"]]
    out["best_deployable"] = best
    out["pool"] = pool
    out["next"] = {"arbitration_helps": helps,
                   "train_with_arbitration": bool(helps),
                   "engineering_for_over_budget_helpers": [c for t, c in helps if c in over],
                   "drop_arbitration": not helps}
    out["computed_at_utc"] = C.now()
    return out


def report(s):
    lines = ["| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |",
             "|---|---|---|---|---|---|---|---|---|",
             "| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |"]
    for c, v in s["cells"].items():
        lines.append(D.fmt_row(f"T1A-UB r2 {c}", v["row"], v["cost"]))
    lines += ["", "| 单元 | 样本内 5000 / 5500 / 6000 | C | GT 起点 RA@20 | E3 local / global | 预算 |", "|---|---|---|---|---|---|"]
    for c, v in s["cells"].items():
        ins = " / ".join(f"{v['insample_median'][st]:.2f}" for st in C.STEPS)
        e3 = v["zgz"]["e3"]
        lines.append(f"| {c} | {ins} | {v['C']:.2f} | {v['keep_correct_gt_ra20']:.2f} | {e3['local']:.2f} / {e3['global']:.2f} "
                     f"| {'OK' if v['budget_ok'] else 'over: ' + ','.join(k for k, ok in v['budget_parts'].items() if not ok)} |")
    lines += ["", "```", __import__("json").dumps({"arbitration": s["arbitration"], "best_deployable": s["best_deployable"],
                                                     "next": s["next"]}, indent=1, ensure_ascii=False), "```"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--block", required=True)
    a = p.parse_args(argv)
    C.require(a.round == 2, "this tool decides round 2")
    r2.patch_common()
    C.verify_protocol()
    s = round2(a.block)
    C.save(C.OUT / "summary_r2.json", s, exclusive=True)
    md = C.OUT / "report_r2.md"
    C.require(not md.exists(), "report exists")
    md.write_text(report(s))
    print(md)


if __name__ == "__main__":
    main()
