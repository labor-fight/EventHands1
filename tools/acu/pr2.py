#!/usr/bin/env python3
"""ACU PR revision PR2 (docs/EXPERIMENT_RECORDS.md#acu-pr2-prereg-20261104): small verbs outside the namespace.

    python -m tools.acu.pr2 gate0      exit 0 iff the main s1c decision (PR) is PR_WEAK, else exit 3 (every PR2 job is
                                       closed by rule: stage 2 runs with PR)
    python -m tools.acu.pr2 table      analysis/pr2_router.md: PR, PR2 and r_s2d identity accuracy side by side
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.acu.common import OUT, read  # noqa: E402

Q = OUT.with_name(OUT.name + "q")
SEQS = ("zgz_local", "zgz_global", "ylf_local")


def gate0():
    d = read(OUT / "decisions" / "s1c.json")["decision"]
    print("main s1c", d, flush=True)
    sys.exit(0 if d == "PR_WEAK" else 3)


def table():
    L = ["# ACU PR revision: identity accuracy of PR, PR2 and r_s2d (causal window nodes; r_s2d with its whole-packet image)",
         "", "| recording | router | packets | part acc | finger-group acc | on-hand recall | point err med mm |"
         " routed: part acc / err mm |", "|---|---|---|---|---|---|---|---|"]
    for seq in SEQS:
        rows = []
        for tag, root in (("PR", OUT), ("PR2", Q)):
            p = root / "readings" / "router" / f"pr_{seq}.json"
            if p.exists():
                r = read(p)
                rows.append((tag, r["packets"], r["pr"]))
                ref = (r["packets"], r["r_s2d"])
        if not rows:
            continue
        for tag, n, x in rows + [("r_s2d", *ref)]:
            L.append(f"| {seq} | {tag} | {n} | {x['part_acc']:.3f} | {x['group_acc']:.3f} | {x['onhand_recall']:.3f} | "
                     f"{x['err_med_mm']:.2f} | {x['routed_part_acc']:.3f} / {x['routed_err_med_mm']:.2f} |")
    for tag, root in (("PR", OUT), ("PR2", Q)):
        p = root / "decisions" / "s1c.json"
        if p.exists():
            d = read(p)
            L.append(f"\n{tag}: {d['decision']} (equivalence {d['equivalence']}, per recording {d['per_recording']})")
    out = OUT / "analysis" / "pr2_router.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", choices=("gate0", "table"))
    a = p.parse_args(argv)
    {"gate0": gate0, "table": table}[a.verb]()


if __name__ == "__main__":
    main()
