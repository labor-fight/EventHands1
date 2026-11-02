#!/usr/bin/env python3
"""S45-RT barrier: exit 0 once every job of the driver queue except the main rows, the decision and the barrier
itself is terminal, whatever its outcome. The main rows (exclusive GPU 4, latency) and the decision depend on it, so
latency is measured with nothing else running and the decision reads only finished readings.
"""
from __future__ import annotations

from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.s45rt.common import OUT, read  # noqa: E402

TERMINAL = ("EXIT_0", "FAILED_PRESERVED", "SKIPPED_DEP_FAILED")


def waiting():
    q, s = read(OUT / "queue.json"), read(OUT / "status.json")
    keys = [j["key"] for j in q["jobs"] if not j["key"].startswith(("row:", "final", "barrier:"))]
    return [k for k in keys if s["jobs"].get(k, {}).get("status") not in TERMINAL]


while True:
    left = waiting()
    if not left:
        print("all registered readings terminal", flush=True)
        break
    time.sleep(60)
