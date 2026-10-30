#!/usr/bin/env python3
"""Register an engineering amendment to the frozen K-star protocol (outputs/kstar/protocol.json).

Only for tool defects that do not touch a threshold or a reading: the old and new SHA-256 of the file are recorded in
`amendments`, and `files` is updated so `common.verify_protocol()` accepts the corrected file. The ledger entry that
mirrors the amendment is written by hand.

    python tools/kstar/amend.py <repo-relative file> "<reason>"
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kstar import common as KC  # noqa: E402


def main():
    rel, reason = sys.argv[1], sys.argv[2]
    proto = json.loads(KC.PROTOCOL.read_text())
    if rel not in proto["files"]:
        raise SystemExit(f"{rel} is not a frozen file")
    old, new = proto["files"][rel], KC.sha(KC.REPO / rel)
    if old == new:
        raise SystemExit(f"{rel} unchanged")
    proto.setdefault("amendments", []).append(
        {"at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "file": rel, "old_sha256": old, "new_sha256": new,
         "reason": reason, "kind": "engineering (no threshold / reading change)"})
    proto["files"][rel] = new
    KC.save_json(KC.PROTOCOL, proto, exclusive=False)
    print(json.dumps(proto["amendments"][-1], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
