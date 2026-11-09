#!/usr/bin/env python3
"""Wait for a job of another TPD driver state directory (docs/EXPERIMENT_RECORDS.md#tpd-amend3-20261105): exit 0 when it
is EXIT_0, exit 1 when it failed or was skipped (or the wait exceeds --hours).

    python -m tools.tpd.waitfor --driver driver3 --job syn:train:c0:s3407 [--hours 6]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT)]

from tools.tpd.common import OUT  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--driver", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--hours", type=float, default=6.0)
    a = p.parse_args()
    deadline = time.monotonic() + a.hours * 3600
    status = OUT / a.driver / "status.json"
    while time.monotonic() < deadline:
        try:
            st = json.loads(status.read_text())["jobs"].get(a.job, {}).get("status")
        except (OSError, ValueError):
            st = None
        if st == "EXIT_0":
            print(a.driver, a.job, "EXIT_0", flush=True)
            return
        if st in ("FAILED_PRESERVED", "SKIPPED_DEP_FAILED"):
            print(a.driver, a.job, st, flush=True)
            sys.exit(1)
        time.sleep(60)
    print("timeout", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
