#!/usr/bin/env python3
"""S45-PX amendment 1 (docs/EXPERIMENT_RECORDS.md#s45px-amend1-20261030): formal-training entry.

Runs the frozen tools/s45px/train.py unchanged except its formal binding check, which reads the amendment-1 preflight
receipts (receipts_r1/, corrected departure step) in place of the superseded ones. Recipe validation, fresh claim,
hook, strict contract and audit are train.py's; both DDP ranks start from this file.
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45px import train as T  # noqa: E402  (strict environment via tools/xyz_c1_repro/train.py)
from tools.s45px.common import (ARMS, OUT, PROTOCOL, S45TR, config_frozen, read, require, sha,  # noqa: E402
                                verify_protocol)

MANIFEST = OUT / "manifests" / "amend1.json"
AMENDED = ("tools/s45px/train_r1.py", "tools/s45px/accept_r1.py", "tools/s45px/run_r2.py")


def check_binding(options, config):
    from tools.s45tr import common as TR
    verify_protocol()
    config_frozen(config)
    files = read(MANIFEST)["file_sha256"]
    for name in AMENDED:
        require(sha(ROOT / name) == files[name], "amendment-1 file changed: " + name)
    for arm in ARMS:
        pre = read(OUT / "receipts_r1" / f"preflight_{arm}.json")
        require(pre["status"] == "PASS" and pre["protocol_sha256"] == sha(PROTOCOL)
                and pre["s45_steps_1_2_exact_and_departs_at_step3"] is True, "Preflight not accepted (amendment 1): " + arm)
        require(pre["wrapper_sha256"] == sha(T.__file__), "Training wrapper changed since preflight")
    TR.verify_protocol()
    replay = read(S45TR / "receipts" / "preflight_s45replay.json")
    require(replay["status"] == "PASS" and replay["s45_code_replay_exact"] is True,
            "S45-TR P0 (current code replays S45 training exactly) not on record")


T.check_binding = check_binding

if __name__ == "__main__":
    T.main()
