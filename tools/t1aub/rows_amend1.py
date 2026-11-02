#!/usr/bin/env python3
"""T1A-UB engineering amendment 1 (docs/EXPERIMENT_RECORDS.md#t1aub-amend1-rows-20261102): re-register the main-row blocks
b1 / b2 / b4 and the round decisions that the 07:28 UTC failure of `ub:r1:row:b1:t0p` skipped (CPU only; writes a jobs file).

    rows_amend1.py jobs --out FILE      every SKIPPED / FAILED block job and decision of rounds 1, 2 and 4, re-keyed
                                        (`:row:` -> `:row2:`, `:bd:` -> `:bd2:`, decisions + `:r2`), commands unchanged
                                        except tools/t1aub/evaluate(2).py -> evaluate7.py and breakdown.py ->
                                        breakdown7.py, dependencies on re-keyed jobs remapped; block b1's first row also
                                        waits for the acceptance of the queued trainings b4dz and b4match (a
                                        training still running when it becomes ready is waited for, not depended on)
    rows_amend1.py archive              moves block b1's attempt-1 row folders (t1aref_b1, t1arp_b1, t0p) to
                                        outputs/t1aub/20261102/rows_attempt1/b1/<net>/row_step6000 (b1 is re-measured
                                        whole, in one exclusive session, under its own name)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.t1aub import common as C  # noqa: E402

QUEUE = ROOT / "outputs/s45rt/20261031/queue_r2.json"
STATUS = ROOT / "outputs/s45rt/20261031/status_r2.json"
NOTE = "T1A-UB amendment 1 (docs/EXPERIMENT_RECORDS.md#t1aub-amend1-rows-20261102)"
HOLD_B1 = ["ub:r4:acc:formal:b4dz", "ub:r4:acc:formal:b4match"]
ATTEMPT1_B1 = ("t1aref_b1", "t1arp_b1", "t0p")
WRAPPERS = {"tools/t1aub/evaluate.py": "tools/t1aub/evaluate7.py", "tools/t1aub/evaluate2.py": "tools/t1aub/evaluate7.py",
            "tools/t1aub/breakdown.py": "tools/t1aub/breakdown7.py"}


def rekey(key):
    if ":row:" in key:
        return key.replace(":row:", ":row2:")
    if ":bd:" in key:
        return key.replace(":bd:", ":bd2:")
    return key + ":r2"


def jobs():
    q = json.loads(QUEUE.read_text())
    queue = q["jobs"] if isinstance(q, dict) else q
    status = json.loads(STATUS.read_text())["jobs"]
    attempt1 = {f"ub:r1:row:b1:{n}" for n in ATTEMPT1_B1}
    redo = [j for j in queue if j["key"].startswith(("ub:r1:", "ub:r2:", "ub:r4:"))
            and (status.get(j["key"], {}).get("status") in ("SKIPPED_DEP_FAILED", "FAILED_PRESERVED")
                 or j["key"] in attempt1)]
    C.require(all(any(t in j["key"] for t in (":row:", ":bd:", "decide")) for j in redo),
              "only block rows, breakdowns and decisions were skipped")
    old = {j["key"] for j in redo}
    out = []
    for j in redo:
        cmd = [WRAPPERS.get(c, c) for c in j["command"]]
        deps = [rekey(d) if d in old else d for d in j.get("deps", [])]
        if j["key"] == "ub:r1:row:b1:t1aref_b1":
            deps += HOLD_B1
        out.append({"key": rekey(j["key"]), "kind": j["kind"], "mem_gib": j.get("mem_gib"), "note": NOTE,
                    "deps": deps, "command": cmd})
    C.require({rekey(k) for k in attempt1} <= {j["key"] for j in out}, "block b1 is re-registered whole")
    return out


def archive():
    dst_root = C.OUT / "rows_attempt1" / "b1"
    moved = []
    for net in ATTEMPT1_B1:
        src = C.OUT / "evaluations" / net / "row_step6000"
        C.require(src.is_dir(), "missing attempt-1 row: " + str(src))
        dst = dst_root / net / "row_step6000"
        C.require(not dst.exists(), "archive exists: " + str(dst))
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append({"net": net, "from": str(src), "to": str(dst)})
    C.save(dst_root / "archive.json", {"amendment": "t1aub-amend1-rows-20261102", "moved": moved, "at_utc": C.now()},
           exclusive=True)
    return moved


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="what", required=True)
    j = sub.add_parser("jobs")
    j.add_argument("--out", required=True)
    sub.add_parser("archive")
    a = p.parse_args(argv)
    if a.what == "jobs":
        js = jobs()
        Path(a.out).write_text(json.dumps(js, indent=1) + "\n")
        print(len(js), "jobs ->", a.out)
    else:
        print(json.dumps(archive(), indent=1))


if __name__ == "__main__":
    main()
