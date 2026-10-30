#!/usr/bin/env python3
"""Freeze the OBS study (docs/EXPERIMENT_RECORDS.md#obs-prereg-20261026) into outputs/obs/protocol.json.

Records the digest of the pre-registration text, the SHA-256 of every OBS tool / test and of the modules they import,
the S39 step-6000 checkpoints, the split manifest, the stage-0 sigma* file and the git state. Refuses to overwrite.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from obs import common as C  # noqa: E402

DEPS = ["tools/mv2/common.py", "model/model.py", "model/backbones.py", "model/mano_layer.py", "model/pose_repr.py",
        "model/config.py", "semkine/eval_track.py", "semkine/events.py", "semkine/dataset.py", "semkine/anchored.py",
        "tools/tracking/evalx.py", "tools/dt/filter_eval.py", "tools/prepare_hand_data.py", "tools/extract_subms.py",
        "outputs/dt3/gate_specs/ramp1000.json", "outputs/sba/p0v/result.json", "assets/mano_right.npz",
        "tests/test_obs.py"] + [f"outputs/semkine/dt_dz_l3_s{s}/config_resolved.yaml" for s in (3407, 3408, 3409)] + [
        f"outputs/dt3/gate_eval/ramp1000_w200/dt_dz_l3_s{s}/af9f41a0_rn0.015-0.03-0.06-0.09-0.172185-0.261517-0.404576"
        f"-0.5-0.8_fn0.04-0.08-0.16-0.24-0.4-0.56-0.8_tn0.025-0.05-0.1-0.15-0.25-0.35-0.5_w200_clip.json"
        for s in (3407, 3408, 3409)]


def files():
    out = sorted({str(p.relative_to(C.REPO)) for p in (C.REPO / "tools" / "obs").glob("*.py")} | set(DEPS))
    return [f for f in out if (C.REPO / f).exists()]


def main():
    if C.PROTOCOL.exists():
        raise SystemExit("outputs/obs/protocol.json exists; amendments go into 'amendments'")
    txt = C.prereg_text()
    if txt is None:
        raise SystemExit("pre-registration block #obs-prereg-20261026 ... OBS-PREREG:END not found in the ledger")
    cks = {str(C.MC.s39_ckpt(s, 6000)): C.sha(C.MC.s39_ckpt(s, 6000)) for s in C.SEEDS}
    git = lambda *a: subprocess.run(["git", *a], cwd=C.REPO, capture_output=True, text=True).stdout  # noqa: E731
    v2e = Path("/data1/lyq/data/hand_data/v2e")
    proto = {"study": "OBS: event-label time alignment audit + individual calibration upper bound (2026-10-26)",
             "anchor": "docs/EXPERIMENT_RECORDS.md#obs-prereg-20261026",
             "frozen_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
             "prereg_sha256": hashlib.sha256(txt.encode()).hexdigest(),
             "files": {f: C.sha(C.REPO / f) for f in files()}, "checkpoints": cks,
             "splits": {str(C.DATA_ROOT / "splits_semkine.json"): C.sha(C.DATA_ROOT / "splits_semkine.json")},
             "v2e_source_read": {str(p): C.sha(p) for p in (v2e / "v2e.py", v2e / "v2ecore" / "emulator.py",
                                                          v2e / "scripts" / "batch_bag_v2e.py") if p.exists()},
             "git": {"head": git("rev-parse", "HEAD").strip(),
                     "status_sha256": hashlib.sha256(git("status", "--short").encode()).hexdigest(), "committed": False},
             "amendments": []}
    C.save_json(C.PROTOCOL, proto)
    print(f"frozen {len(proto['files'])} files, {len(cks)} checkpoints")


if __name__ == "__main__":
    main()
