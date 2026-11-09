#!/usr/bin/env python3
"""TPD LC stage conditional jobs (docs/EXPERIMENT_RECORDS.md#tpd-lc-prereg-20261105): run a registered command only when
the written identity screen opens it, otherwise exit 0 with a skip record.

    python -m tools.tpd.maybe_lc swap:<router> -- <command ...>    decisions/lc_screen.json: <router> passed the screen
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


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    require(len(argv) >= 3 and argv[1] == "--" and argv[0].startswith("swap:"), "usage: maybe_lc.py swap:<router> -- <cmd>")
    router, command = argv[0][5:], argv[2:]
    ok = bool(read(OUT / "decisions" / "lc_screen.json")["screen"][router]["swap"])
    if ok:
        print(json.dumps({"condition": argv[0], "open": True}), flush=True)
        os.execvp(command[0], command)
    key = hashlib.sha256("\0".join(command).encode()).hexdigest()[:12]
    save(OUT / "skipped" / f"lc_{router}__{key}.json", {"condition": argv[0], "open": False, "command": command,
                                                       "at_utc": now()})
    print(json.dumps({"condition": argv[0], "open": False}), flush=True)


if __name__ == "__main__":
    main()
