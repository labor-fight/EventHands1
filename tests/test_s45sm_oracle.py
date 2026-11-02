"""CPU tests: S45-SM label-route PFA (semkine/s45sm_oracle.py, tools/s45sm/oracle.py)."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import s45sm as SM  # noqa: E402
from semkine import s45sm_fast as FAST  # noqa: E402
from semkine import s45sm_oracle as O  # noqa: E402
from semkine import xag as XAG  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from tools.s45sm import configs as K  # noqa: E402
from tools.s45sm import paths as P  # noqa: E402
from tools.s45sm import registry as REG  # noqa: E402

NPZ = ROOT / "assets/mano_right.npz"


def test_install_restores_and_keeps_native_sources():
    state = O.install()
    try:
        assert state == O.STATE and O.installed() == O.STATE
        assert XAG.sample_sources is SM._NATIVE["sample_sources"]
        assert XAG.route_association is FAST.pfa_association and XM.associate_rays is FAST.empty_association
        with pytest.raises(RuntimeError):
            O.install()
    finally:
        O.uninstall()
    assert O.installed() is None
    assert XAG.route_association is SM._NATIVE["route_association"]
    assert XM.associate_rays is SM._NATIVE["associate_rays"]


def test_label_routes_become_pfa_records():
    with np.load(NPZ, allow_pickle=False) as z:
        verts = torch.tensor(np.asarray(z["v_template"], np.float32))
        faces = torch.tensor(np.asarray(z["f"], np.int64))
        lbs = torch.tensor(np.asarray(z["weights"], np.float32))
    g = torch.Generator().manual_seed(2)
    B, N = 2, 64
    V = verts.unsqueeze(0).repeat(B, 1, 1)
    V[..., 2] += 0.5
    mask = torch.rand(B, N, generator=g) > 0.2
    face = torch.randint(0, faces.shape[0], (B, N), generator=g)
    bary = torch.rand(B, N, 3, generator=g) + 0.05
    valid = torch.rand(B, N, generator=g) > 0.3
    d = torch.cat([torch.rand(B, N, 2, generator=g) * 0.2 - 0.1, torch.ones(B, N, 1)], -1)
    vf = XM.vertex_face_table(faces, V.shape[1])
    band = torch.full((B,), 0.02)
    O.install()
    try:
        empty = XM.associate_rays(d, mask, V, faces, vf, band, band / 2, 0.01)
        out, audit = XAG.route_association("face_route", empty, d, mask, V, faces, vf, lbs, band, band / 2, 0.01,
                                           source_part=torch.zeros_like(face), source_face=face, source_bary=bary,
                                           source_valid=valid)
    finally:
        O.uninstall()
    e = valid & mask
    assert torch.equal(audit["xag_eligible"], e) and torch.equal(out["q"], e.float())
    assert torch.equal(out["vid"][e][:, :3], faces[face[e]]) and bool(out["valid"][e].all()) and not out["valid"][~e].any()
    w = bary / bary.sum(-1, keepdim=True)
    point = (w.unsqueeze(-1) * V[torch.arange(B).view(B, 1, 1), faces[face]]).sum(-2)
    assert torch.allclose(out["z"][e], point[..., 2][e])


def test_oracle_config_is_t1ddz_with_identity(monkeypatch):
    monkeypatch.setattr(REG, "oracles", lambda: {"pfagtdz": {"base": "t1ddz", "assoc": "pfa", "round": 5}})
    base = yaml.safe_load(P.T1DDZ_CONFIG.read_text())
    for phase in ("preflight", "formal"):
        cfg = K.expected_raw_oracle("pfagtdz", phase, base)
        assert cfg["S45SM"] == {"ARM": "pfagtdz", "ROUTER": "gt", "ASSOC": "pfa", "BASE": "s45at_t1ddz"}
        assert "S45AT" not in cfg and cfg["MODEL"]["XAG_ASSOC"] == "face_route" and cfg["DATA"]["XAG_SOURCE_ROOT"]
        ref = dict(base)
        ref.pop("S45AT")
        got = dict(cfg)
        got.pop("S45SM")
        for k in ("TRAIN", "EVAL"):
            got[k] = {kk: v for kk, v in got[k].items() if kk not in ("RUN_NAME", "OUTPUT_DIR", "MAX_STEPS", "VAL_CHECK_INTERVAL")}
            ref[k] = {kk: v for kk, v in ref[k].items() if kk not in ("RUN_NAME", "OUTPUT_DIR", "MAX_STEPS", "VAL_CHECK_INTERVAL")}
        assert got == ref
        assert cfg["TRAIN"]["RUN_NAME"] == "s45sm_pfagtdz_s3407" + ("_preflight" if phase == "preflight" else "")
