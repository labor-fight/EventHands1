#!/usr/bin/env python3
"""Freeze the NOPART files (docs/EXPERIMENT_RECORDS.md#nopart-prereg-20261113): sha256 of each, written once.

    python tools/nopart/freeze.py write|verify
tools/rfsu/sim.py (RFSU's file, with the hook) is recorded, not required unchanged: RFSU owns it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "outputs/nopart/20261113/manifests/r1.json"
FILES = ["semkine/nopart.py", "tools/nopart/router_sim.py", "tools/nopart/sim_ext.py", "tools/nopart/router_capture.py",
         "tools/nopart/router_compare.py", "tools/nopart/table.py", "tools/nopart/freeze.py", "tests/test_nopart.py",
         "outputs/nopart/20261113/preflight.sh"]
RECORDED = ["tools/rfsu/sim.py"]


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def main():
    verb = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if verb == "write":
        if MANIFEST.exists():
            raise SystemExit("written once: " + str(MANIFEST))
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text(json.dumps({"files": {f: sha(f) for f in FILES}, "recorded": {f: sha(f) for f in RECORDED},
                                        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1) + "\n")
        print("frozen", len(FILES), "files")
    else:
        m = json.loads(MANIFEST.read_text())
        bad = [f for f, h in m["files"].items() if sha(f) != h]
        print("verify", "OK" if not bad else "CHANGED: " + ", ".join(bad))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
