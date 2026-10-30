#!/usr/bin/env python3
"""Register an amendment to the frozen OBS protocol (outputs/obs/protocol.json).

    python tools/obs/amend.py files "<kind>" "<reason>" <repo-relative file> [...]   # changed or new files: record old/new SHA
    python tools/obs/amend.py prereg "<anchor>" "<end marker>" "<reason>"          # a further pre-registration block's digest

`files` updates `files` so `common.verify_protocol()` accepts the registered state; `prereg` stores the SHA-256 of the
ledger text between `<a id="<anchor>"></a>` and the end marker under `extra_preregs`, which `verify_extra()` checks.
The ledger entry that mirrors the amendment is written by hand.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obs import common as C  # noqa: E402


def block(anchor: str, end: str) -> str:
    doc = (C.REPO / "docs" / "EXPERIMENT_RECORDS.md").read_text(encoding="utf-8")
    m = re.search(re.escape(f'<a id="{anchor}"></a>') + r"(.*?)" + re.escape(end), doc, re.S)
    if not m:
        raise SystemExit(f"block {anchor} ... {end} not found")
    return m.group(1)


def main():
    kind = sys.argv[1]
    proto = json.loads(C.PROTOCOL.read_text())
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    if kind == "files":
        k, reason, rels = sys.argv[2], sys.argv[3], sys.argv[4:]
        rec = {"at_utc": now, "kind": k, "reason": reason, "files": {}}
        for r in rels:
            new = C.sha(C.REPO / r)
            rec["files"][r] = {"old_sha256": proto["files"].get(r), "new_sha256": new}
            proto["files"][r] = new
        proto.setdefault("amendments", []).append(rec)
    elif kind == "prereg":
        anchor, end, reason = sys.argv[2], sys.argv[3], sys.argv[4]
        digest = hashlib.sha256(block(anchor, end).encode()).hexdigest()
        proto.setdefault("extra_preregs", {})[anchor] = {"end": end, "sha256": digest, "at_utc": now}
        proto.setdefault("amendments", []).append({"at_utc": now, "kind": "pre-registration block",
                                                   "anchor": anchor, "sha256": digest, "reason": reason})
    else:
        raise SystemExit("usage: amend.py files|prereg ...")
    C.save_json(C.PROTOCOL, proto, exclusive=False)
    print(json.dumps(proto["amendments"][-1], indent=1, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    main()
