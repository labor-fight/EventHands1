#!/usr/bin/env python3
"""ACU stage-2 gate and written decision (docs/EXPERIMENT_RECORDS.md#acu-s2-prereg-20261104); CPU, read-only, run once.

    python -m tools.acu.decide_s2 gate        exit 0 if s1c decided PR_OK, exit 3 (closed by rule) if PR_WEAK
    python -m tools.acu.decide_s2 s2          the 2x2 (seed 3407) -> decisions/s2.json, analysis/s2.md

Rules (written before any stage-2 training):
  effect of a factor (A: observation pfa -> ca; B: update orig -> b1), judged in both levels of the other factor:
    EFFECTIVE   if both paired differences give E1 RA <= -0.5 mm, or both give (zgz_local R15 and F15 retention medians
                <= -0.05 and E3 RA <= +0.3 mm)
    MIXED       if exactly one of the two paired differences qualifies
    NONE        otherwise
  winner = the cell with the lowest E1 RA among the cells that keep every EFFECTIVE / MIXED factor's qualifying level
  (c00 when both factors are NONE); the winner goes to seeds 3408 / 3409 and to the MeshGNN check (stage 3).
Descriptive: the five E1 metrics, E3, amplification, retention, f1, the interaction (c11 - c10) - (c01 - c00), the
per-step decomposition (tools/acu/diag_update.py), S43's engine numbers as the target reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu import arms_s2 as A  # noqa: E402
from tools.acu.common import OUT, S43_ENGINE, now, read, require, save  # noqa: E402
from tools.acu.decide import RULES, summary  # noqa: E402

SEED = 3407


def gate():
    d = read(OUT / "decisions" / "s1c.json")["decision"]
    print("s1c", d, flush=True)
    sys.exit(0 if d == "PR_OK" else 3)


def qualifies(new, base):
    by_ra = new["five"][4] <= base["five"][4] - RULES["a2_ra_gain_mm"]
    by_mech = (new["R15"] <= base["R15"] - RULES["a2_retention_drop"] and new["F15"] <= base["F15"] - RULES["a2_retention_drop"]
               and new["e3"] <= base["e3"] + RULES["a2_e3_tolerance_mm"])
    return bool(by_ra or by_mech)


def s2():
    out_json, out_md = OUT / "decisions" / "s2.json", OUT / "analysis" / "s2.md"
    require(not out_json.exists(), "decided once")
    rows = {c: summary(OUT / "readings" / A.run_name(c, SEED)) for c in A.CELLS}
    q = {"A|orig": qualifies(rows["c10"], rows["c00"]), "A|b1": qualifies(rows["c11"], rows["c01"]),
         "B|orig": qualifies(rows["c01"], rows["c00"]), "B|ca": qualifies(rows["c11"], rows["c10"])}

    def verdict(a, b):
        return "EFFECTIVE" if (a and b) else ("MIXED" if (a or b) else "NONE")
    dec = {"A": verdict(q["A|orig"], q["A|b1"]), "B": verdict(q["B|orig"], q["B|ca"]), "paired": q}
    allowed = [c for c in A.CELLS if (dec["A"] == "NONE" or A.CELLS[c][0] == "ca") and
               (dec["B"] == "NONE" or A.CELLS[c][1] == "b1")]
    dec["winner"] = min(allowed, key=lambda c: rows[c]["five"][4]) if allowed else "c00"
    dec["interaction_ra_mm"] = (rows["c11"]["five"][4] - rows["c10"]["five"][4]) - (rows["c01"]["five"][4] - rows["c00"]["five"][4])
    from tools.acu import diag_update as DU
    DU.READ_ROOTS["acu"] = OUT / "readings"
    DU.NETS.clear()
    DU.NETS.update({c: ("acu", A.run_name(c, SEED)) for c in A.CELLS})
    DU.run(OUT / "analysis" / "s2_decomposition")
    save(out_json, {"rules": RULES, "decisions": dec, "rows": rows, "decided_at_utc": now()}, exclusive=True)
    L = ["# ACU stage 2: the asynchronous 2x2 (seed 3407, PR identities, causal pipeline)", "",
         f"A (pfa -> ca): {dec['A']}; B (orig -> b1): {dec['B']}; paired {q}; winner {dec['winner']}; "
         f"interaction RA {dec['interaction_ra_mm']:+.2f} mm.", "",
         "| cell | obs / update | MPJPE-L | MPJPE-G | MPVPE-L | MPVPE-G | RA | E3 | amp | R15 | F15 | T30 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|",
         "| S43 (engine, 3 seeds) | – | " + " | ".join(f"{x:.2f}" for x in S43_ENGINE) + " | – | – | – | – | – |"]
    for c, r in rows.items():
        L.append(f"| {c} | {'/'.join(A.CELLS[c])} | " + " | ".join(f"{x:.2f}" for x in r["five"]) +
                 f" | {r['e3']:.2f} | {r['amp']:.2f} | {r['R15']:.2f} | {r['F15']:.2f} | {r['T30']:.2f} |")
    L += ["", "Per-step decomposition: analysis/s2_decomposition.md."]
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("gate", "s2"))
    a = p.parse_args(argv)
    gate() if a.stage == "gate" else s2()


if __name__ == "__main__":
    main()
