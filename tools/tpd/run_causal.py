#!/usr/bin/env python3
"""TPD causal (asynchronous) deployment form: readings and checks (docs/EXPERIMENT_RECORDS.md#tpd-causal-prereg-20261105),
through PFO's frozen engine with the output root redirected to outputs/tpd/20261105 and semkine/tpd_causal.py installed
(the trunk's nodes by the causal stride, its tokens with the causal "never fired" fill; the router packet-based, exempt).

    python -m tools.tpd.run_causal read --net b2 --task T      B2 seed 3407 (PFO2 fgs, accepted step 6000) in the causal
                                                               form, zero training; tag b2_s3407__causal
    python -m tools.tpd.run_causal decide                      decisions/causal.json (once): the written rule
Rule: d = RA(b2_s3407__causal) - RA(B2 3407, PFO2's published e1). d <= 0.15 -> TEST_ONLY (later trunks may train with
B2's sampler and are deployed in the causal form, the cost reported); d > 0.15 -> TRAIN_CAUSAL (the final configuration
trains in the causal form).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd.common import OUT, PFO2_OUT, b2_checkpoint, b2_spec, now, read, require, save  # noqa: E402

TASKS = ("e1", "e3", "pert", "f1")
LIMIT = 0.15


def reading(task):
    from semkine import tpd_causal as TC
    from tools.pfo import engine as E
    from tools.tpd import run_rol as RR
    require(task in TASKS, "task")
    b2_checkpoint(3407)
    sp = b2_spec(3407)
    RR.prepare(sp)
    TC.install()
    TC.engine_hooks()
    E.OUT = OUT
    out = E.run("b2_s3407__causal", task, None, None, (), str(sp))
    save(Path(out) / "causal.json", {"causal": True, "cap": TC.CAP, "packet_ms": TC.PACKET_MS, "written_utc": now()},
         exclusive=True)


def rule(d):
    return "TEST_ONLY" if d <= LIMIT else "TRAIN_CAUSAL"


def decide():
    out = OUT / "decisions" / "causal.json"
    require(not out.exists(), "decided once")
    a = read(OUT / "readings" / "b2_s3407__causal" / "e1" / "report.json")
    b = read(PFO2_OUT / "readings" / "pfo2_fgs_s3407" / "e1" / "report.json")
    ra_a, ra_b = a["overall"]["mpjpe_ra_mm"], b["overall"]["mpjpe_ra_mm"]
    per = {s: {"causal": a["per_sequence"][s]["mpjpe_ra_mm"], "b2": b["per_sequence"][s]["mpjpe_ra_mm"]}
           for s in ("zgz_local", "zgz_global")}
    d = ra_a - ra_b
    save(out, {"ra_causal": ra_a, "ra_b2": ra_b, "d": d, "per_sequence": per, "verdict": rule(d), "limit": LIMIT,
               "decided_utc": now()}, exclusive=True)
    print(rule(d), round(d, 3), per)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)
    s = sub.add_parser("read")
    s.add_argument("--net", required=True, choices=("b2",))
    s.add_argument("--task", required=True, choices=TASKS)
    sub.add_parser("decide")
    a = p.parse_args(argv)
    if a.verb == "read":
        return reading(a.task)
    return decide()


if __name__ == "__main__":
    main()
