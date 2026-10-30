#!/usr/bin/env python3
"""Engineering acceptance of one S45-TR training (tools/s45tr/common.py verify_run); CPU only, never scores.

Writes receipts/<phase>_<arm>.json (exclusive). The s45replay preflight must replay S45's recorded steps 1-3 bit for
bit; otherwise the receipt records the failure and the process exits non-zero, which stops every formal training.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.s45tr.common import ARMS, OUT, PROTOCOL, ROOT, now, require, save, sha, verify_protocol, verify_run  # noqa: E402


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--arm", required=True, choices=tuple(ARMS))
    p.add_argument("--phase", required=True, choices=("preflight", "formal"))
    a = p.parse_args()
    verify_protocol()
    receipt = verify_run(a.arm, a.phase)
    receipt.update(protocol_sha256=sha(PROTOCOL), wrapper_sha256=sha(ROOT / "tools/s45tr/train.py"),
                   verified_at_utc=now(), engineering_only_no_accuracy_result=True)
    if a.arm == "s45replay" and not receipt["s45_code_replay_exact"]:
        receipt["status"] = "FAIL"
    save(OUT / "receipts" / f"{a.phase}_{a.arm}.json", receipt, exclusive=True)
    require(receipt["status"] == "PASS", "acceptance failed: " + a.phase + " " + a.arm)
    print("PASS", a.phase, a.arm)


if __name__ == "__main__":
    main()
