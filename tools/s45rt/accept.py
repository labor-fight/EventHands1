#!/usr/bin/env python3
"""Engineering acceptance of one S45-RT T1p training (tools/s45rt/plan2.py verify_run); CPU only, never scores.

Writes receipts/<phase>_<arm>.json (exclusive); a failed check records the failure and exits non-zero.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.s45rt.common import OUT, PROTOCOL, ROOT, now, require, save, sha, verify_protocol  # noqa: E402
from tools.s45rt.plan2 import ARMS, verify_run  # noqa: E402


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--phase", required=True, choices=("preflight", "formal"))
    a = p.parse_args()
    verify_protocol()
    try:
        receipt = verify_run(a.arm, a.phase)
    except RuntimeError as exc:
        receipt = {"status": "FAIL", "arm": a.arm, "phase": a.phase, "error": str(exc)}
    receipt.update(protocol_sha256=sha(PROTOCOL), wrapper_sha256=sha(ROOT / "tools/s45rt/train.py"),
                   verified_at_utc=now(), engineering_only_no_accuracy_result=True)
    save(OUT / "receipts" / f"{a.phase}_{a.arm}.json", receipt, exclusive=True)
    require(receipt["status"] == "PASS", "acceptance failed: " + a.phase + " " + a.arm)
    print("PASS", a.phase, a.arm)


if __name__ == "__main__":
    main()
