#!/usr/bin/env python3
"""T1A-UB round-4 summary (CPU only; once, after every round-4 reading and the block-b4 main rows; single seed, descriptive).

    python tools/t1aub/decide4.py --block B4 --ref-block B1     -> outputs/t1aub/20261102/summary_r4.json, report_r4.md

Written rules of docs/EXPERIMENT_RECORDS.md#t1aub-r4-prereg-20261102: every training-side mechanism's in-budget network
(b4cos_srdwa, b4n2_srdwa, b4dz_srdwa, b4match) against candidate B (t0pnoeg20_srdwa, read in its own block B1): in-sample
tl_e1 at steps 5000 / 5500 / 6000, C, the main-row RA; HELPS / HURTS / NULL / MIXED; zgz descriptive (XCL single-seed rule);
budget against T1a re-measured in block B4; the mechanisms adopted for the final combination (HELPS only). Also reads the
T1a paired replicate (t1arep) row of block B4 (descriptive). Uses tools/t1aub/decide.py's readings unchanged.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402
from tools.t1aub import decide as D  # noqa: E402

BASE = "t0pnoeg20_srdwa"
MECHANISMS = {"cos": "b4cos_srdwa", "n2": "b4n2_srdwa", "dz": "b4dz_srdwa", "match": "b4match"}


def classify(gains, c_mech, c_base, ra_mech, ra_base):
    v = list(gains.values())
    if all(x >= D.TH["insample"] for x in v) and c_mech <= D.TH["C"] and ra_mech - ra_base <= 0.5:
        return "HELPS"
    if all(x <= -D.TH["insample"] for x in v) or (c_mech > D.TH["C"] >= c_base):
        return "HURTS"
    if all(abs(x) < D.TH["insample"] for x in v) and c_mech <= D.TH["C"]:
        return "NULL"
    return "MIXED"


def round4(block, ref_block):
    R = {BASE: D.readings(BASE, ref_block)}
    for net in MECHANISMS.values():
        R[net] = D.readings(net, block)
    R["t1a"] = D.readings("t1a", block)
    ref_lat = R["t1a"]["row"][1]["latency_ms"]
    base_row = D.row_metrics(R[BASE]["row"][0])
    out = {"round": 4, "block": block, "ref_block": ref_block, "t1aref_latency_ms": ref_lat, "base": BASE,
           "base_row": base_row, "base_C": R[BASE]["fixed_point"]["C"], "mechanisms": {}}
    adopted = []
    for mech, net in MECHANISMS.items():
        row, cost = R[net]["row"]
        rm = D.row_metrics(row)
        ok, parts = D.budget(cost, ref_lat)
        gains = D.diffs(R[net], R[BASE])
        cls = classify(gains, R[net]["fixed_point"]["C"], R[BASE]["fixed_point"]["C"], rm["ra"], base_row["ra"])
        out["mechanisms"][mech] = {
            "network": net, "row": rm, "row_minus_base": {k: rm[k] - base_row[k] for k in ("local", "global", "ra")},
            "cost": {k: cost[k] for k in ("params_total", "macs_total", "latency_ms")}, "budget_ok": ok, "budget_parts": parts,
            "insample_gain_vs_base": gains, "C": R[net]["fixed_point"]["C"],
            "keep_correct_gt_ra20": R[net]["fixed_point"]["gt"]["ra20"], "zgz": R[net]["zgz"],
            "zgz_support_local_vs_base": D.zgz_support(R[BASE], R[net], "local")[0], "class": cls}
        if cls == "HELPS" and ok:
            adopted.append(mech)
    out["adopted"] = adopted
    try:
        rep = C.read(C.OUT / "evaluations" / "t1arep" / "row_step6000/report.json")
        out["t1arep_row"] = D.row_metrics(C.read(rep["main_row"]["path"]))
    except FileNotFoundError:
        out["t1arep_row"] = None
    out["computed_at_utc"] = C.now()
    return out


def report(s):
    lines = ["| 网络结构 | MPJPE-local | MPJPE-global | MPVPE-local | MPVPE-global | RA-MPJPE(递推) | Latency | FLOPs/step | Params |",
             "|---|---|---|---|---|---|---|---|---|",
             "| EventHands-PCA6 (baseline) | 30 | 10.99 | 23.58 | 8.15 | - | 1.76 ms | 1.653 G | 11.18 M |"]
    for mech, v in s["mechanisms"].items():
        lines.append(D.fmt_row(f"T1A-UB r4 {v['network']}（{mech}）", v["row"], v["cost"]))
    lines += ["", "```", __import__("json").dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("zgz",)}
                                                     for k, v in s["mechanisms"].items()}, indent=1, ensure_ascii=False),
              "adopted: " + ", ".join(s["adopted"]), "```"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--block", required=True)
    p.add_argument("--ref-block", required=True)
    a = p.parse_args(argv)
    C.verify_protocol()
    s = round4(a.block, a.ref_block)
    C.save(C.OUT / "summary_r4.json", s, exclusive=True)
    md = C.OUT / "report_r4.md"
    C.require(not md.exists(), "report exists")
    md.write_text(report(s))
    print(md)


if __name__ == "__main__":
    main()
