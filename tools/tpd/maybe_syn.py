#!/usr/bin/env python3
"""TPD synthesis conditional jobs (docs/EXPERIMENT_RECORDS.md#tpd-syn-prereg-20261105): run a registered command only
when the written rules open it, otherwise exit 0 with a skip record.

    python -m tools.tpd.maybe_syn <condition> -- <command ...>
      chosen:<rol_arm>   decisions/rol_summary.json: ROL chose <rol_arm> for seed 3408 (its synthesis arm runs seed 3407)
      any_chosen         decisions/rol_summary.json: ROL chose an arm (the synthesis base c0 runs seed 3407)
      s2:<arm>           decisions/syn_screen.json: <arm> runs seed 3408 (the rollout arm passed against c0, and c0 itself)
      s3:<arm>           decisions/syn_keep.json: <arm> is the final configuration (it runs seed 3409)
      s3:any             decisions/syn_keep.json: there is a final configuration
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


def is_open(cond):
    """A condition whose decision file was never written (an earlier gate closed the path) is closed."""
    files = {"any_chosen": "rol_summary.json", "chosen": "rol_summary.json", "s2": "syn_screen.json", "s3": "syn_keep.json"}
    if not (OUT / "decisions" / files[cond.split(":")[0]]).exists():
        return False
    if cond == "any_chosen":
        return read(OUT / "decisions" / "rol_summary.json")["chosen"] is not None
    if cond.startswith("chosen:"):
        return read(OUT / "decisions" / "rol_summary.json")["chosen"] == cond[7:]
    if cond.startswith("s2:"):
        return cond[3:] in read(OUT / "decisions" / "syn_screen.json")["seed3408_arms"]
    if cond == "s3:any":
        return read(OUT / "decisions" / "syn_keep.json")["final_arm"] is not None
    if cond.startswith("s3:"):
        return read(OUT / "decisions" / "syn_keep.json")["final_arm"] == cond[3:]
    raise ValueError("unknown condition " + cond)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require(len(argv) >= 3 and argv[1] == "--", "usage: maybe_syn.py <condition> -- <command ...>")
    cond, command = argv[0], argv[2:]
    if is_open(cond):
        print(json.dumps({"condition": cond, "open": True}), flush=True)
        os.execvp(command[0], command)
    key = hashlib.sha256("\0".join(command).encode()).hexdigest()[:12]
    save(OUT / "skipped" / f"syn_{cond.replace(':', '_')}__{key}.json", {"condition": cond, "open": False,
                                                                        "command": command, "at_utc": now()})
    print(json.dumps({"condition": cond, "open": False}), flush=True)


if __name__ == "__main__":
    main()
