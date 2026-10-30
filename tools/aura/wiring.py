#!/usr/bin/env python3
"""AURA `MODEL.XYZ_AURA` wiring of model/model.py (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024).

model/model.py is frozen by AGG778 R2 (and later re-frozen by MRF's own wiring), so this wiring is fixed now as four
exact insertions at anchors that neither R2 nor MRF touch, and applied only once both studies are complete
(`tools/aura/d_stage.py apply-wiring`). It applies to either accepted pre-state (R2's file, or R2's file with the MRF
wiring); the protocol records both pre / post SHA pairs:

    1. "XYZ_AURA" in the allowed MODEL keys (after "XYZ_SIGN")
    2. MODEL.XYZ_AURA belongs to ENCODER=xyz_mesh (after the XYZ_RESIDUAL guard)
    3. the variant and its class (before `self.xyz = xyz_cls(`): `shell` builds `semkine.aura.AuraXYZMeshNet`
    4. the association hook (before the AGG778 fan-out line of `_xyz_mesh_forward`): every other variant replaces the
       association by `semkine.aura.apply_aura(...)` under no_grad

    python tools/aura/wiring.py check      both pre / post SHA-256 pairs (pre from the current file)
    python tools/aura/wiring.py write DIR  write the wired copy of the current file to DIR/model.py
    python tools/aura/wiring.py apply      replace model/model.py by the wired text (pre / post SHA verified)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_PY = ROOT / "model" / "model.py"

INSERTIONS = (
    ("after", '            "XYZ_SIGN",\n',
     '            # AURA (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024): one association variant, opt-in\n'
     '            "XYZ_AURA",\n'),
    ("after",
     '        if not self.xyz_mesh and any(k in model_cfg for k in ("XYZ_RESIDUAL_MODE", "XYZ_RESIDUAL_SEED")):\n'
     '            raise ValueError("MODEL.XYZ_RESIDUAL_MODE / XYZ_RESIDUAL_SEED belong to ENCODER=xyz_mesh")\n',
     '        if not self.xyz_mesh and "XYZ_AURA" in model_cfg:\n'
     '            raise ValueError("MODEL.XYZ_AURA belongs to ENCODER=xyz_mesh")\n'),
    ("before", '            self.xyz = xyz_cls(\n',
     '            # AURA: absent / legacy leaves the C1 association and class (bit for bit); `shell` builds its subclass,\n'
     '            # every other variant replaces the association in `_xyz_mesh_forward`\n'
     '            self.xyz_aura = str(model_cfg.get("XYZ_AURA", "legacy"))\n'
     '            if self.xyz_aura != "legacy":\n'
     '                from semkine.aura import AURA_MODES, AuraXYZMeshNet\n'
     '                if self.xyz_aura not in AURA_MODES:\n'
     '                    raise ValueError(f"MODEL.XYZ_AURA must be one of {AURA_MODES}, got {self.xyz_aura!r}")\n'
     '                if (self.xyz_agg != "legacy" or getattr(self, "xyz_rf", "legacy") != "legacy"\n'
     '                        or self.xyz_assoc_prev != "self" or self.xyz_assoc_mode != "hit_first"\n'
     '                        or self.xag_gnn_mode != "legacy" or self.xag_assoc_mode != "base"):\n'
     '                    raise ValueError("MODEL.XYZ_AURA is defined on the C1 recipe (hit_first / self association, "\n'
     '                                     "no XYZ_AGG / XYZ_RF / XAG)")\n'
     '                if self.xyz_aura == "shell":\n'
     '                    xyz_cls, agg_kw = AuraXYZMeshNet, {}\n'),
    ("before",
     '        # AGG778 fan-out binning reads the association band (ray units); every other path passes nothing new\n',
     '        # AURA (docs/EXPERIMENT_RECORDS.md#aura-prereg-20261024): an association variant replaces C1\'s, no_grad\n'
     '        if getattr(self, "xyz_aura", "legacy") not in ("legacy", "shell"):\n'
     '            from semkine.aura import apply_aura\n'
     '            with torch.no_grad():\n'
     '                assoc = apply_aura(self.xyz_aura, assoc, d, mask, verts_assoc, self.mano.f, self.xyz.vert_faces,\n'
     '                                   (fx, fy, cx, cy), batch, self.xyz.lbs_hard, k=self.xyz_k)\n'),
)


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def wired(text: str) -> str:
    """Each anchor occurs exactly once; its insertion goes right after / before it."""
    for where, anchor, add in INSERTIONS:
        if text.count(anchor) != 1:
            raise RuntimeError(f"AURA wiring anchor not unique in model/model.py: {anchor.splitlines()[0]!r}")
        text = text.replace(anchor, anchor + add if where == "after" else add + anchor)
    return text


def states(text_r2: str) -> dict:
    """Both accepted pre-states (R2's file, R2's file + the MRF wiring) and their AURA-wired post-states."""
    sys.path.insert(0, str(ROOT))
    from tools.mrf import wiring as MW
    text_mrf = MW.wired(text_r2)
    return {"pre": {"r2": sha_text(text_r2), "mrf": sha_text(text_mrf)},
            "post": {"r2": sha_text(wired(text_r2)), "mrf": sha_text(wired(text_mrf))}}


def apply(expected: dict) -> str:
    text = MODEL_PY.read_text()
    have = sha_text(text)
    if have in expected["post"].values():
        return "already wired"
    state = next((k for k, v in expected["pre"].items() if v == have), None)
    if state is None:
        raise RuntimeError("model/model.py is neither an accepted pre-wiring state nor an AURA-wired file")
    new = wired(text)
    if sha_text(new) != expected["post"][state]:
        raise RuntimeError("wired text differs from the frozen post-wiring SHA")
    tmp = MODEL_PY.with_suffix(".py.aura_tmp")
    tmp.write_text(new)
    tmp.replace(MODEL_PY)
    return f"wired from the {state} state"


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "check":
        print(json.dumps({"current": sha_text(MODEL_PY.read_text()), **states(MODEL_PY.read_text())}, indent=1))
    elif cmd == "write":
        dst = Path(sys.argv[2]) / "model.py"
        dst.write_text(wired(MODEL_PY.read_text()))
        print(dst)
    elif cmd == "apply":
        p = json.loads((ROOT / "outputs" / "aura" / "protocol.json").read_text())
        print(apply(p["model_wiring"]))
    else:
        raise SystemExit("usage: wiring.py check | write DIR | apply")
