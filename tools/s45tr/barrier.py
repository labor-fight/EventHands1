#!/usr/bin/env python3
"""S45-TR barrier: exit 0 once every training (pair) job of the driver queue is terminal, whatever its outcome.

The serial main rows (exclusive GPU 4, latency) depend on it, so they never become ready while a training could
still start: an exclusive job that is ready blocks every new GPU job in tools/s45tr/chain.py.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.s45tr.common import OUT, read  # noqa: E402

TERMINAL = ("EXIT_0", "FAILED_PRESERVED", "SKIPPED_DEP_FAILED")

while True:
    q, s = read(OUT / "queue.json"), read(OUT / "status.json")
    pairs = [j["key"] for j in q["jobs"] if j["kind"] == "pair"]
    if q.get("closed") or all(s["jobs"].get(k, {}).get("status") in TERMINAL for k in pairs) and \
            not any(j["key"].startswith("preflight:n4") or j["key"].startswith("formal:n4") for j in q["jobs"]
                    if s["jobs"].get(j["key"], {}).get("status") not in TERMINAL):
        if all(s["jobs"].get(k, {}).get("status") in TERMINAL for k in pairs):
            print("all trainings terminal", flush=True)
            break
    time.sleep(60)
