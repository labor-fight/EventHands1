"""T1A-UB round 6 (tools/t1aub/r6.py): router-location features with S45's own association. CPU only, nothing written."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.t1aub import common as C  # noqa: E402
from tools.t1aub import r6  # noqa: E402

SPEC = {"eg": False, "readout": {"slot_dim": 20, "hidden": 256}, "gtfeat": True,
        "route": {"router": "srdwkda", "assoc": "face_route", "onhand_tau": 0.5}, "note": "test"}


@pytest.fixture
def rf(monkeypatch):
    native = C.registry

    def merged():
        out = native()
        if "zz6" not in out["arms"]:
            out["arms"]["zz6"] = {**SPEC, "round": 99}
            out["rounds"][99] = "test"
        return out
    monkeypatch.setattr(C, "registry", merged)
    return "zz6"


def test_config(rf):
    cfg = r6.expected_raw(rf, "formal")
    b = cfg["T1AUB"]
    assert b["GTFEAT"] is True and b["EG"] is False and b["ROUTE"]["ASSOC"] == "face_route"
    assert b["ROUTE"]["ROUTER"] == "srdwkda" and b["READOUT"] == {"SLOT_DIM": 20, "HIDDEN": 256}
    assert cfg["MODEL"]["XAG_ASSOC"] == "face_route" and "XAG_SOURCE_ROOT" not in cfg["DATA"]
    ref = C.expected_raw("t0pnoeg20", "formal")
    diff = {k for k in cfg if cfg[k] != ref[k]}
    assert diff == {"TRAIN", "EVAL", "T1AUB"}
    assert {k for k in cfg["TRAIN"] if cfg["TRAIN"][k] != ref["TRAIN"][k]} == {"RUN_NAME", "OUTPUT_DIR"}
    pre = r6.expected_raw(rf, "preflight")
    assert pre["TRAIN"]["MAX_STEPS"] == 50
    with pytest.raises(RuntimeError):
        C.expected_raw(rf, "formal")          # the frozen generator refuses the combination


def _forward(rf, fast):
    import numpy as np
    import torch
    import pytorch_lightning as pl
    from semkine import eval_track as ET, s45sm, s45sm_fast, t1aub as M, xyz_mesh as XM, xag as XAG
    from semkine.dataset import sequences_for_split, splits_manifest
    from model import MNISTModel
    cfg = r6.expected_raw(rf, "formal")
    native_assoc = XM.associate_rays
    state = r6.install_for_config(cfg, evaluation=True, fast=fast)
    try:
        assert state["route"]["assoc"] == "face_route" and state["rfeat"] is True
        assert XAG.route_association is M.passthrough_route and XM.associate_rays is native_assoc
        pl.seed_everything(3407)
        model = MNISTModel(cfg).eval()
        assert model.xyz.t1aub_gtfeat and not model.xyz.t1aub_eg
        root = Path(cfg["DATA"]["ROOT"])
        seqs = dict(sequences_for_split(root, "val_core", splits_manifest(cfg)))
        events, offsets, aux, pos51 = ET.load_sequence(root, seqs["zgz_local"], "zgz_local")
        betas = torch.tensor(aux["betas"], dtype=torch.float32).view(1, -1)
        K = torch.tensor(aux["camera_K"], dtype=torch.float32).view(1, 3, 3)
        a = int(np.asarray(aux["valid_runs_ms"], dtype=np.int64).reshape(-1, 2)[0][0])
        prev = torch.from_numpy(pos51[a + 99].copy()).view(1, -1).float()
        ev5 = ET._window_events(events, offsets, None, a + 149, 50)
        pk = ET.make_eval_packet(ev5, prev, betas, K, 50, "cpu")
        seen = {}
        native_gt = M.gt_features

        def spy(net, mask):
            out = native_gt(net, mask)
            seen["valid"] = int(out[..., 3].sum())
            seen["n"] = int(mask.sum())
            return out
        M.gt_features = spy
        with torch.no_grad():
            torch.manual_seed(1)
            model.xyz.fc2.weight.normal_(0, 0.01)
            out = model.forward_packet(pk)
        M.gt_features = native_gt
        assert torch.isfinite(out).all() and seen["valid"] > 0.3 * seen["n"]
        return out
    finally:
        M.uninstall()
        (s45sm_fast if fast else s45sm).uninstall()


def test_forward_router_features_and_fast_path(rf):
    import torch
    slow = _forward(rf, fast=False)
    fast = _forward(rf, fast=True)
    assert torch.allclose(slow, fast, atol=1e-5, rtol=1e-4)
