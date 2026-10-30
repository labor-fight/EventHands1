#!/usr/bin/env python3
"""Freeze KA step 1 (docs/EXPERIMENT_RECORDS.md#ka-prereg-20261028) into outputs/kin_atlas/protocol.json: the digest of
the pre-registration text, SHA-256 of the KA tools / test and the modules they import, the split manifest, and the
(size, mtime_ns) of the 18 training base recordings' data files. Refuses to overwrite; amendments go into `amendments`."""
from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.kin_atlas import common as C  # noqa: E402

DEPS = ["tests/test_kin_atlas.py", "tools/s43_br/geometry.py", "tools/s43_br/__init__.py", "model/mano_layer.py",
        "assets/mano_right.npz"]


def main():
    if C.PROTOCOL.exists():
        raise SystemExit("outputs/kin_atlas/protocol.json exists")
    txt = C.prereg_text()
    if txt is None:
        raise SystemExit("pre-registration block #ka-prereg-20261028 ... KA-PREREG:END not found in the ledger")
    files = sorted({str(p.relative_to(C.REPO)) for p in (C.REPO / "tools" / "kin_atlas").glob("*.py")} | set(DEPS))
    data = {}
    for seq in C.base_seqs():
        base = C.DATA_ROOT / C._legacy_dirs()[seq]
        for f in sorted(base.glob(f"{seq}_*")) + [base / f"{seq}.meta"]:
            if f.exists():
                st = f.stat()
                data[str(f)] = [st.st_size, st.st_mtime_ns]
    git = lambda *a: subprocess.run(["git", *a], cwd=C.REPO, capture_output=True, text=True).stdout  # noqa: E731
    proto = {"study": "KA step 1: kinematic-atlas equivariance check (2026-10-28)",
             "anchor": "docs/EXPERIMENT_RECORDS.md#ka-prereg-20261028",
             "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "prereg_sha256": hashlib.sha256(txt.encode()).hexdigest(),
             "files": {f: C.sha(C.REPO / f) for f in files},
             "splits": {str(C.DATA_ROOT / "splits_semkine.json"): C.sha(C.DATA_ROOT / "splits_semkine.json")},
             "data_identity": data,
             "git": {"head": git("rev-parse", "HEAD").strip(),
                     "status_sha256": hashlib.sha256(git("status", "--short").encode()).hexdigest()},
             "amendments": []}
    C.save_json(C.PROTOCOL, proto)
    print(f"frozen {len(files)} files, {len(data)} data files, prereg {proto['prereg_sha256'][:8]}")


if __name__ == "__main__":
    main()
