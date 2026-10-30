#!/usr/bin/env python3
"""Freeze the K-star protocol (docs/EXPERIMENT_RECORDS.md#kstar-prereg-20261025) before any reading.

Writes outputs/kstar/protocol.json: SHA-256 of every probe script, every imported frozen dependency, the input
artefacts (S39 archives, frozen XVM heads, splits) and the pre-registration section text itself; the git HEAD and a
digest of `git status --porcelain` (the work tree is not committed); and the identity (size, mtime_ns) of every data
file the probes read (FIT / SELECT / zgz sequences and their XAG source caches). Refuses to overwrite.

    python tools/kstar/freeze.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kstar import common as KC  # noqa: E402

FILES = [
    "tools/kstar/__init__.py", "tools/kstar/common.py", "tools/kstar/freeze.py", "tools/kstar/aa_audit.py",
    "tools/kstar/fusion_ceiling.py", "tools/kstar/head_probe.py", "tools/kstar/corr_rstar.py", "tools/kstar/decide.py",
    "tools/xvm/common.py", "tools/xvm/p4_data.py", "tools/xvm/p4_train.py", "tools/xvm/p5_solve.py",
    "tools/xvm/ladder.py", "tools/s44/evflow.py", "tools/xcl/evaluate.py", "tools/xdl/geometry.py",
    "semkine/xvm.py", "semkine/eval_track.py", "semkine/mrf.py", "semkine/lie.py", "semkine/xyz_mesh.py",
    "model/mano_layer.py", "assets/mano_right.npz",
    "outputs/s44/p10/s39_s3407_control.npz", "outputs/s44/p10/s39_s3408_control.npz",
    "outputs/s44/p10/s39_s3409_control.npz", "outputs/xvm/s2/p4/A_m600/heads.pt",
]
SPLITS = Path("/data1/lyq/code/mesh/EventHands/data/hand_data51/splits_semkine.json")
ANCHOR = "kstar-prereg-20261025"


def prereg_text():
    doc = (KC.REPO / "docs" / "EXPERIMENT_RECORDS.md").read_text()
    m = re.search(rf'<a id="{ANCHOR}"></a>(.*?)<!-- KSTAR-PREREG:END -->', doc, re.S)
    if not m:
        raise SystemExit(f"pre-registration section #{ANCHOR} (ending with <!-- KSTAR-PREREG:END -->) not found")
    return m.group(1)


def data_identity():
    from xvm import common as XC
    from s44 import evflow as EF
    seqs = XC.subject_seqs(KC.FIT_SUBJ + KC.SELECT_SUBJ) + list(XC.zgz_seqs())
    out = {}
    for name, d in seqs:
        for f in sorted((EF.DATA_ROOT / d).glob(f"{name}[_.]*")):
            st = f.stat()
            out[str(f)] = [st.st_size, st.st_mtime_ns]
        src = EF.SOURCE_ROOT / name
        for f in sorted(src.glob("*")) if src.exists() else []:
            st = f.stat()
            out[str(f)] = [st.st_size, st.st_mtime_ns]
    return out


def git_state():
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=KC.REPO, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=KC.REPO, capture_output=True, text=True).stdout
    return {"head": head, "status_porcelain_sha256": hashlib.sha256(status.encode()).hexdigest(),
            "status_entries": len(status.splitlines()), "committed": False}


def main():
    if KC.PROTOCOL.exists():
        raise SystemExit("outputs/kstar/protocol.json exists; amendments go into a new 'amendments' entry")
    text = prereg_text()
    proto = {"study": "K-star (K-Battle 2026-10-25)", "anchor": f"docs/EXPERIMENT_RECORDS.md#{ANCHOR}",
             "prereg_sha256": hashlib.sha256(text.encode()).hexdigest(),
             "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
             "files": {f: KC.sha(KC.REPO / f) for f in FILES},
             "splits": {str(SPLITS): KC.sha(SPLITS)}, "git": git_state(), "data_identity": data_identity(),
             "amendments": []}
    KC.OUT.mkdir(parents=True, exist_ok=True)
    KC.save_json(KC.PROTOCOL, proto)
    print(json.dumps({"files": len(proto["files"]), "data_files": len(proto["data_identity"]),
                      "prereg_sha256": proto["prereg_sha256"]}))


if __name__ == "__main__":
    main()
