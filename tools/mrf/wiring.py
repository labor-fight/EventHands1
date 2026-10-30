#!/usr/bin/env python3
"""MRF `MODEL.XYZ_RF` wiring of model/model.py (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024).

model/model.py is frozen by the running AGG778 R2 protocol (its Stage D re-verifies the SHA at every launch), so the
wiring is not applied while R2 can still run. It is fixed now as three exact insertions (the XYZ_AGG pattern), and the
MRF protocol records the SHA before and after; Stage D applies it only once R2 is complete:

    python tools/mrf/wiring.py check      the current file is the frozen pre-wiring file; print both SHA-256
    python tools/mrf/wiring.py write DIR  write the wired copy to DIR/model.py (tests of the wiring on a copy)
    python tools/mrf/wiring.py apply      replace model/model.py by the wired text (pre / post SHA verified)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PY = ROOT / "model" / "model.py"

INSERTIONS = (
    ('            # AGG778 (docs/EXPERIMENT_RECORDS.md#agg778-prereg-20261024): one aggregation variant, opt-in\n'
     '            "XYZ_AGG",\n',
     '            # MRF (docs/EXPERIMENT_RECORDS.md#mrf-prereg-20261024): one receptive-field (MeshGNN) variant, opt-in\n'
     '            "XYZ_RF",\n'),
    ('        if not self.xyz_mesh and "XYZ_AGG" in model_cfg:\n'
     '            raise ValueError("MODEL.XYZ_AGG belongs to ENCODER=xyz_mesh")\n',
     '        if not self.xyz_mesh and "XYZ_RF" in model_cfg:\n'
     '            raise ValueError("MODEL.XYZ_RF belongs to ENCODER=xyz_mesh")\n'),
    ('                xyz_cls, agg_kw = AggXYZMeshNet, {"agg": self.xyz_agg}\n',
     '            # MRF: absent / legacy builds the plain XYZMeshNet (bit for bit); a variant builds its subclass\n'
     '            self.xyz_rf = str(model_cfg.get("XYZ_RF", "legacy"))\n'
     '            if self.xyz_rf != "legacy":\n'
     '                from semkine.mrf import RF_MODES, MRFXYZMeshNet\n'
     '                if self.xyz_rf not in RF_MODES:\n'
     '                    raise ValueError(f"MODEL.XYZ_RF must be one of {RF_MODES}, got {self.xyz_rf!r}")\n'
     '                if self.xyz_agg != "legacy" or self.xyz_assoc_prev != "self" or self.xag_gnn_mode != "legacy":\n'
     '                    raise ValueError("MODEL.XYZ_RF is defined on the C1 recipe (ASSOC_PREV self, legacy GNN / AGG)")\n'
     '                xyz_cls, agg_kw = MRFXYZMeshNet, {"rf": self.xyz_rf}\n'),
)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def wired(text: str) -> str:
    """Each anchor occurs exactly once; its insertion goes right after it."""
    for anchor, add in INSERTIONS:
        if text.count(anchor) != 1:
            raise RuntimeError(f"wiring anchor not unique in model/model.py: {anchor.splitlines()[0]!r}")
        text = text.replace(anchor, anchor + add)
    return text


def shas() -> dict:
    text = MODEL_PY.read_text()
    return {"pre": sha_text(text), "post": sha_text(wired(text))}


def apply(expected: dict):
    text = MODEL_PY.read_text()
    if sha_text(text) == expected["post"]:
        return "already wired"
    if sha_text(text) != expected["pre"]:
        raise RuntimeError("model/model.py is neither the frozen pre-wiring nor the wired file")
    new = wired(text)
    if sha_text(new) != expected["post"]:
        raise RuntimeError("wired text differs from the frozen post-wiring SHA")
    tmp = MODEL_PY.with_suffix(".py.mrf_tmp")
    tmp.write_text(new)
    tmp.replace(MODEL_PY)
    return "wired"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        print(json.dumps(shas(), indent=1))
    elif cmd == "write":
        dst = Path(sys.argv[2]) / "model.py"
        dst.write_text(wired(MODEL_PY.read_text()))
        print(dst)
    elif cmd == "apply":
        p = json.loads((ROOT / "outputs" / "mrf" / "protocol.json").read_text())
        print(apply(p["model_wiring"]))
    else:
        raise SystemExit("usage: wiring.py check | write DIR | apply")
