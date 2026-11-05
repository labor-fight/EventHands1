#!/usr/bin/env python3
"""ACU written decisions (docs/EXPERIMENT_RECORDS.md#acu-s1-prereg-20261104); CPU, read-only on the readings, run once.

    python -m tools.acu.decide s1a       stage 1a (CA screens, seed 3407) -> decisions/s1a.json, analysis/s1a.md

Rules (written before any ACU training):
  a1 (B2-gt + CA, label identity at training and test) against B2-gt (label / label), E1 RA:
     IFACE_KEEPS_INFO   if a1 <= B2-gt + 0.7 mm   (the contour-normal records keep what the exact points gave)
     IFACE_LOSES_INFO   otherwise
  a2 (B2 + CA, r_s2d identity at training and test) against B2 seed 3407 (r_s2d / r_s2d):
     CA_EFFECTIVE       if E1 RA <= B2 - 0.5 mm, or (zgz_local R15 and F15 retention medians both <= B2's - 0.05 and E3 RA
                        <= B2's + 0.3 mm)
     CA_NOT_EFFECTIVE   otherwise
Descriptive (never a rule): the five E1 metrics, E3, amplification E1 / E3, f1 (GT 1 -> 20 feeds, mean-pose start ->
20), the per-step decomposition of tools/acu/diag_update.py (motion gain, injection, low-motion change, closed-loop
correction a), a1 with the deployed router at test (exposure; B2-gt with r_s2d is RA 15.44).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu.common import (B2GT_LABEL_READ, B2GT_ROUTER_READ, B2_READ, OUT, S43_ENGINE, now, read,  # noqa: E402
                              require, save)

RULES = {"a1_margin_mm": 0.7, "a2_ra_gain_mm": 0.5, "a2_retention_drop": 0.05, "a2_e3_tolerance_mm": 0.3}


def reading(tag_dir, task):
    p = Path(tag_dir) / task / "report.json"
    require(p.exists(), "missing reading " + str(p))
    return read(p)


def summary(tag_dir):
    e1, e3, pt = (reading(tag_dir, t) for t in ("e1", "e3", "pert"))
    s = e1["per_sequence"]
    five = [s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"], s["zgz_local"]["mpvpe_ra_mm"],
            s["zgz_global"]["mpvpe_ra_mm"], e1["overall"]["mpjpe_ra_mm"]]
    loc = pt["per_sequence"]["zgz_local"]
    out = {"five": five, "e3": e3["overall"]["mpjpe_ra_mm"], "amp": five[4] / e3["overall"]["mpjpe_ra_mm"],
           "R15": loc["R15"]["rho_median"], "F15": loc["F15"]["rho_median"], "T30": loc["T30"]["rho_median"]}
    f1p = Path(tag_dir) / "f1" / "report.json"
    if f1p.exists():
        f = read(f1p)["per_sequence"]
        for k in ("gt_iter1", "gt_iter20", "mean_iter20", "mean_start"):
            out[k] = {sq: f[sq][k] for sq in ("zgz_local", "zgz_global")}
    return out


def s1a():
    out_json, out_md = OUT / "decisions" / "s1a.json", OUT / "analysis" / "s1a.md"
    require(not out_json.exists(), "decided once")
    A1L = OUT / "readings" / "acu_a1_s3407__label"
    A1R = OUT / "readings" / "acu_a1_s3407"
    A2 = OUT / "readings" / "acu_a2_s3407"
    rows = {"a1_label": summary(A1L), "a1_router": summary(A1R), "a2_router": summary(A2),
            "b2gt_label": summary(B2GT_LABEL_READ), "b2gt_router": summary(B2GT_ROUTER_READ), "b2_router": summary(B2_READ)}
    a1 = rows["a1_label"]["five"][4] <= rows["b2gt_label"]["five"][4] + RULES["a1_margin_mm"]
    b2, a2 = rows["b2_router"], rows["a2_router"]
    a2_ra = a2["five"][4] <= b2["five"][4] - RULES["a2_ra_gain_mm"]
    a2_mech = (a2["R15"] <= b2["R15"] - RULES["a2_retention_drop"] and a2["F15"] <= b2["F15"] - RULES["a2_retention_drop"]
               and a2["e3"] <= b2["e3"] + RULES["a2_e3_tolerance_mm"])
    dec = {"a1": "IFACE_KEEPS_INFO" if a1 else "IFACE_LOSES_INFO",
           "a2": "CA_EFFECTIVE" if (a2_ra or a2_mech) else "CA_NOT_EFFECTIVE",
           "a2_by_ra": bool(a2_ra), "a2_by_mechanism": bool(a2_mech)}
    from tools.acu import diag_update as DU
    DU.READ_ROOTS["acu"] = OUT / "readings"
    DU.NETS.clear()
    DU.NETS.update({"a1 label": ("acu", "acu_a1_s3407__label"), "a2 r_s2d": ("acu", "acu_a2_s3407"),
                    "B2-gt label": ("b2ub", "b2gt_s3407__label"), "B2 3407": ("pfo2", "pfo2_fgs_s3407")})
    decomposition = DU.run(OUT / "analysis" / "s1a_decomposition")
    save(out_json, {"rules": RULES, "decisions": dec, "rows": rows, "decided_at_utc": now()}, exclusive=True)
    L = ["# ACU stage 1a: CA screens (seed 3407)", "", f"Decisions: a1 {dec['a1']}, a2 {dec['a2']} "
         f"(by RA {dec['a2_by_ra']}, by mechanism {dec['a2_by_mechanism']}).", "",
         "| row | MPJPE-L | MPJPE-G | MPVPE-L | MPVPE-G | RA | E3 | amp | R15 | F15 | T30 |", "|---|---|---|---|---|---|---|---|---|---|---|",
         "| S43 (engine, 3 seeds) | " + " | ".join(f"{x:.2f}" for x in S43_ENGINE) + " | – | – | – | – | – |"]
    for k, r in rows.items():
        L.append(f"| {k} | " + " | ".join(f"{x:.2f}" for x in r["five"]) +
                 f" | {r['e3']:.2f} | {r['amp']:.2f} | {r['R15']:.2f} | {r['F15']:.2f} | {r['T30']:.2f} |")
    L += ["", "Per-step decomposition: analysis/s1a_decomposition.md (tools/acu/diag_update.py definitions)."]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return dec, decomposition


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("s1a",))
    a = p.parse_args(argv)
    s1a()


if __name__ == "__main__":
    main()
