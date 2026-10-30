#!/usr/bin/env python3
"""Freeze the MV2 study (docs/EXPERIMENT_RECORDS.md#mv2-prereg-20261026) into outputs/mv2/protocol.json.

Records: the SHA-256 of every MV2 tool / config / test and of the frozen modules they import, the S39 checkpoints the
probes and the GPU stop rule read, the split manifest, the digest of the pre-registration text, the git state and the
(size, mtime_ns) of every data file the probes read. Refuses to overwrite: amendments go into `amendments`.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mv2 import common as C  # noqa: E402

DEPS = ["model/model.py", "model/backbones.py", "model/mano_layer.py", "model/pose_repr.py", "model/config.py",
        "semkine/eval_track.py", "semkine/events.py", "semkine/dataset.py", "semkine/train.py", "semkine/aura.py",
        "semkine/xyz_mesh.py", "semkine/sba.py", "semkine/sba_emission.py", "semkine/kssf.py", "semkine/jacobian.py",
        "semkine/lie.py", "semkine/slr.py", "tools/tracking/evalx.py", "tools/dt/filter_eval.py",
        "tools/trk6/carry_probe.py", "tools/kstar/head_probe.py", "tools/kstar/common.py", "tools/xvm/common.py",
        "tools/xvm/p4_data.py", "tools/s44/evflow.py", "tools/report_table.py", "tools/make_s36_row.py",
        "configs/dt/dt_dz_l3.yaml", "tests/test_slr.py", "assets/mano_right.npz"]


def files():
    out = sorted({str(p.relative_to(C.REPO)) for p in (C.REPO / "tools" / "mv2").glob("*.py")}
                 | {str(p.relative_to(C.REPO)) for p in (C.REPO / "configs" / "mv2").glob("*.yaml")}
                 | {str(p.relative_to(C.REPO)) for p in (C.REPO / "tools" / "sba").glob("*.py")}
                 | set(DEPS))
    return [f for f in out if (C.REPO / f).exists()]


def checkpoints():
    cks = [C.s39_ckpt(s, 6000) for s in C.SEEDS] + [C.s39_ckpt(s, 2000) for s in C.SEEDS]
    return {str(p): C.sha(p) for p in cks}


def data_identity():
    out = {}
    for seq in C.base_seqs(C.FIT7 + C.SELECT):
        d = C.DATA_ROOT / C.seq_dir(seq)
        for f in sorted(d.glob(f"{seq}_*")) + sorted(d.glob(f"{seq}.meta")):
            st = f.stat()
            out[str(f)] = [st.st_size, st.st_mtime_ns]
    return out


def git_state():
    def run(*a):
        return subprocess.run(["git", *a], cwd=C.REPO, capture_output=True, text=True).stdout
    return {"head": run("rev-parse", "HEAD").strip(),
            "status_sha256": hashlib.sha256(run("status", "--short").encode()).hexdigest(),
            "committed": False}


def main():
    if C.PROTOCOL.exists():
        raise SystemExit("outputs/mv2/protocol.json exists; amendments go into a new 'amendments' entry")
    txt = C.prereg_text()
    if txt is None:
        raise SystemExit("pre-registration block #mv2-prereg-20261026 ... MV2-PREREG:END not found in the ledger")
    proto = {"study": "MV2 ten-scheme v2.1 (2026-10-26)", "anchor": "docs/EXPERIMENT_RECORDS.md#mv2-prereg-20261026",
             "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "prereg_sha256": hashlib.sha256(txt.encode()).hexdigest(),
             "files": {f: C.sha(C.REPO / f) for f in files()},
             "checkpoints": checkpoints(),
             "splits": {str(C.DATA_ROOT / "splits_semkine.json"): C.sha(C.DATA_ROOT / "splits_semkine.json")},
             "data_identity": data_identity(), "git": git_state(), "amendments": []}
    C.save_json(C.PROTOCOL, proto)
    print(f"frozen {len(proto['files'])} files, {len(proto['checkpoints'])} checkpoints, "
          f"{len(proto['data_identity'])} data files")


if __name__ == "__main__":
    main()
