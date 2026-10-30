#!/usr/bin/env python3
"""Register an amendment to the frozen MV2 protocol (outputs/mv2/protocol.json).

The old and new SHA-256 of the changed file are recorded in `amendments` with the reason and the kind (an engineering fix
that touches no threshold or reading, or a prereg §5 tool-defect classification that turns a gate descriptive), and
`files` is updated so `common.verify_protocol()` accepts the corrected file. The ledger entry is written by hand.

    python tools/mv2/amend.py <repo-relative file> "<kind>" "<reason>"
    python tools/mv2/amend.py add "<kind>" "<reason>" <new file> [<new file> ...]
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mv2 import common as C  # noqa: E402


def add(rels, kind, reason):
    """Register new files (a conditional arm's configuration / code, registered after its gate and before training)."""
    proto = json.loads(C.PROTOCOL.read_text())
    have = [r for r in rels if r in proto["files"]]
    if have:
        raise SystemExit(f"already frozen: {have}")
    new = {r: C.sha(C.REPO / r) for r in rels}
    proto.setdefault("amendments", []).append(
        {"at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "added_files": new, "kind": kind, "reason": reason})
    proto["files"].update(new)
    C.save_json(C.PROTOCOL, proto, exclusive=False)
    print(json.dumps(proto["amendments"][-1], indent=1, ensure_ascii=False))


def main():
    if sys.argv[1] == "add":                    # amend.py add "<kind>" "<reason>" file [file ...]
        return add(sys.argv[4:], sys.argv[2], sys.argv[3])
    rel, kind, reason = sys.argv[1], sys.argv[2], sys.argv[3]
    proto = json.loads(C.PROTOCOL.read_text())
    if rel not in proto["files"]:
        raise SystemExit(f"{rel} is not a frozen file")
    old, new = proto["files"][rel], C.sha(C.REPO / rel)
    if old == new:
        raise SystemExit(f"{rel} unchanged")
    proto.setdefault("amendments", []).append(
        {"at_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "file": rel, "old_sha256": old, "new_sha256": new,
         "kind": kind, "reason": reason})
    proto["files"][rel] = new
    C.save_json(C.PROTOCOL, proto, exclusive=False)
    print(json.dumps(proto["amendments"][-1], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
