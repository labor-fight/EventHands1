#!/usr/bin/env python3
"""T1A-UB round-6 summary (CPU only; once, after every round-6 reading and the block's main rows; single seed, descriptive).

    python tools/t1aub/decide6.py --block B6 --match-block B4     -> outputs/t1aub/20261102/summary_r6.json, report_r6.md

Written rules of docs/EXPERIMENT_RECORDS.md#t1aub-r6-prereg-20261102. The router-location-feature arm `rf6` (candidate B's
trunk; srdwkda's predicted surface location as an input feature; S45's own ray association) against b4match (the same trunk
and router at training and test, the router's information entering by PFA instead): in-sample tl_e1 at 5000 / 5500 /
6000, C, the main-row RA and the XCL single-seed zgz rule (descriptive). Classes FEAT_BETTER / FEAT_NULL / FEAT_WORSE /
MIXED; the budget against T1a re-measured in block B6. Also (descriptive): rf6 against candidate B (t0pnoeg20_srdwa, block
b1) and the share of gtf's in-sample gain over T1a that survives the router, (T1a - rf6) / (T1a - gtf) per checkpoint.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402
from tools.t1aub import decide as D  # noqa: E402

ARM, MATCH, BASE, ORACLE = "rf6", "b4match", "t0pnoeg20_srdwa", "gtf"


def classify(gains, c_arm, c_ref):
    v = list(gains.values())
    if all(x >= D.TH["insample"] for x in v) and c_arm <= D.TH["C"]:
        return "FEAT_BETTER"
    if all(x <= -D.TH["insample"] for x in v) or (c_arm > D.TH["C"] >= c_ref):
        return "FEAT_WORSE"
    if all(abs(x) < D.TH["insample"] for x in v) and c_arm <= D.TH["C"]:
        return "FEAT_NULL"
    return "MIXED"


def round6(block, match_block):
    R = {ARM: D.readings(ARM, block), MATCH: D.readings(MATCH, match_block), BASE: D.readings(BASE, "b1"),
         "t1a": D.readings("t1a", block)}
    gtf = {"insample": D.insample(ORACLE)}
    ref_lat = R["t1a"]["row"][1]["latency_ms"]
    row, cost = R[ARM]["row"]
    rm = D.row_metrics(row)
    ok, parts = D.budget(cost, ref_lat)
    out = {"round": 6, "block": block, "t1aref_latency_ms": ref_lat, "arm": ARM, "row": rm,
           "cost": {k: cost[k] for k in ("params_total", "params_model", "params_router", "macs_total", "latency_ms")},
           "tail_ms": cost["tail_scaled_ms"]["per_packet_min_over_passes_ms"], "budget_ok": ok, "budget_parts": parts,
           "C": R[ARM]["fixed_point"]["C"], "keep_correct_gt_ra20": R[ARM]["fixed_point"]["gt"]["ra20"],
           "zgz": R[ARM]["zgz"], "insample_median": {s: D.med(R[ARM]["insample"][s]) for s in C.STEPS}}
    for ref in (MATCH, BASE):
        gains = D.diffs(R[ARM], R[ref])
        rr = D.row_metrics(R[ref]["row"][0])
        out["vs_" + ref] = {"insample_gain": gains, "C_ref": R[ref]["fixed_point"]["C"],
                            "row_minus_ref": {k: rm[k] - rr[k] for k in ("local", "global", "ra")},
                            "zgz_support_local": D.zgz_support(R[ref], R[ARM], "local")[0],
                            "zgz_support_global": D.zgz_support(R[ref], R[ARM], "global")[0]}
    m = out["vs_" + MATCH]
    out["class"] = classify(m["insample_gain"], out["C"], m["C_ref"])
    t1 = {s: D.med(R["t1a"]["insample"][s]) for s in C.STEPS}
    out["router_survival_of_gtf_gain"] = {
        s: (t1[s] - out["insample_median"][s]) / (t1[s] - D.med(gtf["insample"][s])) for s in C.STEPS}
    out["next"] = {"location_as_feature_preferred": out["class"] == "FEAT_BETTER" or bool(
        m["zgz_support_local"] and m["row_minus_ref"]["local"] < 0 and out["C"] <= D.TH["C"]
        and not all(g <= -D.TH["insample"] for g in m["insample_gain"].values())),
        "final_configuration_unchanged": True}
    out["computed_at_utc"] = C.now()
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--block", required=True)
    p.add_argument("--match-block", required=True)
    a = p.parse_args(argv)
    from tools.t1aub import r6
    r6.patch_common()
    C.verify_protocol()
    s = round6(a.block, a.match_block)
    C.save(C.OUT / "summary_r6.json", s, exclusive=True)
    m = s["vs_" + MATCH]
    lines = [*D.report({"networks": {}, "layers": {}, "candidates": {}, "next": {}}).splitlines()[:3],
             D.fmt_row(f"T1A-UB {ARM}", s["row"], s["cost"]), "",
             f"class vs {MATCH}: {s['class']} | in-sample gain 5000 / 5500 / 6000: "
             + " / ".join(f"{m['insample_gain'][st]:+.2f}" for st in C.STEPS)
             + f" | C {s['C']:.2f} (ref {m['C_ref']:.2f}) | row - ref local / global / RA: "
             + " / ".join(f"{m['row_minus_ref'][k]:+.2f}" for k in ("local", "global", "ra"))
             + f" | budget {'OK' if s['budget_ok'] else 'over'}",
             "gtf gain surviving the router: " + " / ".join(f"{s['router_survival_of_gtf_gain'][st]:.2f}" for st in C.STEPS),
             f"next: {s['next']}"]
    md = C.OUT / "report_r6.md"
    C.require(not md.exists(), "report exists")
    md.write_text("\n".join(lines) + "\n")
    print(md)


if __name__ == "__main__":
    main()
