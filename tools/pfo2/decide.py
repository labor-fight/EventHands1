#!/usr/bin/env python3
"""PFO2 written rules (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103); CPU only, read-only on the readings.

    python -m tools.pfo2.decide mp
        f1mp (F + the native three-round MeshGNN) against F, seeds 3407 / 3408 paired, E1 overall RA at step 6000:
        both seeds better by >= 0.3 mm -> the base's MeshGNN form is gshare, otherwise l1. Writes decisions/s0_mp.json
        after re-verifying the stage-0 manifest.
    python -m tools.pfo2.decide gate --rule mp --open l1|gshare
        exit 0 when the written decision is the given form (the dependent chain may start), exit 3 when it is the other
        form (the chain is closed by the rule; the driver records this as a preserved non-zero exit and skips the
        dependents), any other exit = a tool error
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pfo2.common import OUT, PFO_OUT, now, read, require, save  # noqa: E402
from tools.pfo2.freeze import verify  # noqa: E402

GATE_CLOSED = 3
MP_SEEDS = (3407, 3408)
MP_MARGIN = 0.3


def report(tag, task="e1", root=None):
    p = (root or OUT) / "readings" / tag / task / "report.json"
    return read(p) if p.exists() else None


def row5(r):
    """MPJPE-L, MPJPE-G, MPVPE-L, MPVPE-G, overall RA (tools/pfo/analyze.py `row5`)."""
    if r is None:
        return None
    s = r["per_sequence"]
    return [s["zgz_local"]["mpjpe_ra_mm"], s["zgz_global"]["mpjpe_ra_mm"], s["zgz_local"]["mpvpe_ra_mm"],
            s["zgz_global"]["mpvpe_ra_mm"], r["overall"]["mpjpe_ra_mm"]]


def f_row(seed, task="e1"):
    """F (PFO arm f1) at step 6000, read from PFO's frozen readings."""
    return row5(report(f"pfo_f1_s{seed}", task, PFO_OUT))


def decide_mp():
    verify("s0")
    rows = {}
    for s in MP_SEEDS:
        f, m = f_row(s), row5(report(f"pfo2_f1mp_s{s}"))
        require(f is not None and m is not None, f"missing E1 reading for seed {s}")
        rows[s] = {"F": f, "f1mp": m, "ra_gain": f[4] - m[4]}
    both = all(rows[s]["ra_gain"] >= MP_MARGIN for s in MP_SEEDS)
    out = {"rule": "both seeds: RA(F) - RA(f1mp) >= 0.3 mm -> gshare, else l1", "seeds": rows,
           "decision": "gshare" if both else "l1", "decided_at_utc": now()}
    save(OUT / "decisions" / "s0_mp.json", out, exclusive=True)
    print(out["decision"], {s: round(rows[s]["ra_gain"], 3) for s in MP_SEEDS}, flush=True)
    return out


def gate(rule, want):
    d = read(OUT / "decisions" / f"s0_{rule}.json")
    print("decision", d["decision"], "gate for", want, "->", "OPEN" if d["decision"] == want else "CLOSED", flush=True)
    if d["decision"] != want:
        sys.exit(GATE_CLOSED)


def main(argv=None):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    sub.add_parser("mp")
    g = sub.add_parser("gate")
    g.add_argument("--rule", required=True, choices=("mp",))
    g.add_argument("--open", required=True, choices=("l1", "gshare"))
    a = p.parse_args(argv)
    if a.verb == "mp":
        decide_mp()
    else:
        gate(a.rule, a.open)


if __name__ == "__main__":
    main()
