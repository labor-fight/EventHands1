#!/usr/bin/env python3
"""ACU stage-1b written decision (docs/EXPERIMENT_RECORDS.md#acu-s1b-prereg-20261104); CPU, read-only, run once.

    python -m tools.acu.decide_b1 s1b     -> decisions/s1b.json, analysis/s1b.md

Rule (written before any b1 training): b1 (B2 + B1 consistency training) against B2 seed 3407, both with the deployed
r_s2d router:
  B1_EFFECTIVE       if E1 RA <= B2 - 0.5 mm, or (zgz_local R15 and F15 retention medians both <= B2's - 0.05 and E3 RA
                     <= B2's + 0.3 mm)
  B1_NOT_EFFECTIVE   otherwise
Descriptive: the five E1 metrics, E3, amplification, f1, and the per-step decomposition of tools/acu/diag_update.py (B1's
intended mechanism is a smaller unjustified change at a correct prior: E3 injection and the low-motion change).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu.common import B2_READ, OUT, S43_ENGINE, now, require, save  # noqa: E402
from tools.acu.decide import RULES, summary  # noqa: E402


def s1b():
    out_json, out_md = OUT / "decisions" / "s1b.json", OUT / "analysis" / "s1b.md"
    require(not out_json.exists(), "decided once")
    rows = {"b1_router": summary(OUT / "readings" / "acu_b1_s3407"), "b2_router": summary(B2_READ)}
    b1, b2 = rows["b1_router"], rows["b2_router"]
    by_ra = b1["five"][4] <= b2["five"][4] - RULES["a2_ra_gain_mm"]
    by_mech = (b1["R15"] <= b2["R15"] - RULES["a2_retention_drop"] and b1["F15"] <= b2["F15"] - RULES["a2_retention_drop"]
               and b1["e3"] <= b2["e3"] + RULES["a2_e3_tolerance_mm"])
    dec = {"b1": "B1_EFFECTIVE" if (by_ra or by_mech) else "B1_NOT_EFFECTIVE", "by_ra": bool(by_ra),
           "by_mechanism": bool(by_mech)}
    from tools.acu import diag_update as DU
    DU.READ_ROOTS["acu"] = OUT / "readings"
    DU.NETS.clear()
    DU.NETS.update({"b1 r_s2d": ("acu", "acu_b1_s3407"), "B2 3407": ("pfo2", "pfo2_fgs_s3407")})
    DU.run(OUT / "analysis" / "s1b_decomposition")
    save(out_json, {"rules": RULES, "decisions": dec, "rows": rows, "decided_at_utc": now()}, exclusive=True)
    L = ["# ACU stage 1b: B1 consistency training (seed 3407)", "", f"Decision: {dec['b1']} (by RA {dec['by_ra']}, "
         f"by mechanism {dec['by_mechanism']}).", "",
         "| row | MPJPE-L | MPJPE-G | MPVPE-L | MPVPE-G | RA | E3 | amp | R15 | F15 | T30 |", "|---|---|---|---|---|---|---|---|---|---|---|",
         "| S43 (engine, 3 seeds) | " + " | ".join(f"{x:.2f}" for x in S43_ENGINE) + " | – | – | – | – | – |"]
    for k, r in rows.items():
        L.append(f"| {k} | " + " | ".join(f"{x:.2f}" for x in r["five"]) +
                 f" | {r['e3']:.2f} | {r['amp']:.2f} | {r['R15']:.2f} | {r['F15']:.2f} | {r['T30']:.2f} |")
    L += ["", "Per-step decomposition: analysis/s1b_decomposition.md."]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return dec


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("s1b",))
    p.parse_args(argv)
    s1b()


if __name__ == "__main__":
    main()
