#!/usr/bin/env python3
"""TPD conditional jobs (docs/EXPERIMENT_RECORDS.md#tpd-rol-prereg-20261105): run a registered command only when the
written rule opens it, otherwise exit 0 with a skip record (so the jobs after it, which also cover the open case, still run).

    python -m tools.tpd.maybe <condition> -- <command ...>
      rol5         decisions/rol_plan.json: rol5's formal run is allowed
      s2:<arm>     decisions/rol_summary.json: <arm> trains seed 3408 (the chosen arm; rol1 when the base was rebased)
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.tpd.common import OUT, now, read, require, save  # noqa: E402


def is_open(condition):
    if condition == "rol5":
        return bool(read(OUT / "decisions" / "rol_plan.json")["rol5_allowed"])
    if condition.startswith("s2:"):
        arm = condition[3:]
        s = read(OUT / "decisions" / "rol_summary.json")
        return s["chosen"] == arm or (arm == "rol1" and bool(s["train_rol1_s3408"]))
    raise ValueError("unknown condition " + condition)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require(len(argv) >= 3 and argv[1] == "--", "usage: maybe.py <condition> -- <command ...>")
    condition, command = argv[0], argv[2:]
    if is_open(condition):
        print(json.dumps({"condition": condition, "open": True, "command": command}), flush=True)
        os.execvp(command[0], command)
    key = hashlib.sha256("\0".join(command).encode()).hexdigest()[:12]
    save(OUT / "skipped" / f"{condition.replace(':', '_')}__{key}.json",
         {"condition": condition, "open": False, "command": command, "at_utc": now()})
    print(json.dumps({"condition": condition, "open": False, "skipped": key}), flush=True)


if __name__ == "__main__":
    main()
