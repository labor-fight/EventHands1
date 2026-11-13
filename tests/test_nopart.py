"""ESMNet without the part loss (docs/EXPERIMENT_RECORDS.md#nopart-prereg-20261113). CPU only:
    CUDA_VISIBLE_DEVICES="" python -m pytest -q tests/test_nopart.py
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "model"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

X2 = {"stem": "s2d", "trunk": [32, 64, 96, 128], "fpn": 64, "head": 128}


def _net():
    from semkine import pfo_router as PR
    from tools.s45sm import router_train as S
    cfg = S._data({"dz": True})
    torch.manual_seed(0)
    return PR.build(cfg["MANO"]["NPZ"], **X2)


def _out(net, n=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    tmpl = net.tmpl.float()
    lo, hi = tmpl.min(0).values - 0.01, tmpl.max(0).values + 0.01
    pt = lo + (hi - lo) * torch.rand(1, n, 3, generator=g)
    part = torch.randn(1, n, 16, generator=g)
    return {"coord": pt / 0.01, "part": part, "onhand": torch.zeros(1, n), "mask": torch.ones(1, n, dtype=torch.bool)}


def test_nopart_checkpoint_decodes_with_ci_alone(tmp_path):
    from semkine import nopart as NP
    from semkine import pfo_router as PR
    from semkine import s45sm_fast as FAST
    from semkine.s45tr_t0 import project_bary
    net = _net()
    PR.save_checkpoint(net, tmp_path / "np.ckpt", nopart=True)
    PR.save_checkpoint(net, tmp_path / "x2.ckpt")
    NP.enable()
    np_net, obj = PR.load(tmp_path / "np.ckpt")
    x2_net, _ = PR.load(tmp_path / "x2.ckpt")
    assert obj["nopart"] and getattr(np_net, "nopart", False) and not getattr(x2_net, "nopart", False)
    out = _out(np_net)
    sel = out["mask"]
    point = out["coord"].float() * 0.01
    want = torch.cdist(point[0], np_net.centroid.float()).argmin(-1).unsqueeze(0)
    for face, bary in (np_net.route(out, sel), FAST.route(np_net, "sr", out, sel)):
        assert torch.equal(face, want)
        assert torch.allclose(bary, project_bary(point, np_net.tmpl[np_net.faces[want]]))
    # a checkpoint without the flag keeps the part-restricted decode
    face_x2, _ = x2_net.route(out, sel)
    part = out["part"].argmax(-1)
    assert torch.equal(x2_net.face_part[face_x2], part)
    assert not torch.equal(face_x2, want)


def test_zero_part_weight_removes_the_part_gradient():
    from semkine import s45sm as SM
    net = _net()
    n = 64
    out = _out(net, n)
    out = {k: (v.clone().requires_grad_(True) if v.dtype.is_floating_point else v) for k, v in out.items()}
    tgt = {"valid": torch.ones(1, n, dtype=torch.bool), "part": torch.randint(0, 16, (1, n)),
           "point": torch.rand(1, n, 3) * 0.05}
    saved = dict(SM.WEIGHTS)
    try:
        SM.WEIGHTS["part"] = 0.0
        total, parts = SM.loss(net, out, tgt)
        assert torch.allclose(total, parts["coord"] + 0.5 * parts["onhand"])
        total.backward()
        assert out["part"].grad is not None and float(out["part"].grad.abs().max()) == 0.0
        assert float(out["coord"].grad.abs().max()) > 0.0
    finally:
        SM.WEIGHTS.clear()
        SM.WEIGHTS.update(saved)


def test_extension_keeps_every_existing_arm_and_adds_two():
    import importlib
    S = importlib.import_module("tools.rfsu.sim")
    from tools.nopart import sim_ext as E
    if getattr(S, "NOPART_ARMS_ADDED", False):
        pytest.skip("tools/rfsu/sim.py already loads the extension; see test_hooked_module_matches_native")
    native = {}
    for arm in S.ARMS:
        for seed in S.SEEDS:
            for phase in S.PHASES:
                native[(arm, seed, phase)] = copy.deepcopy(S.build_config(arm, seed, phase))
    E.extend(S)
    for key, cfg in native.items():
        assert S.build_config(*key) == cfg, key
    assert set(S.ARMS) - {k[0] for k in native} == {"rfsrnp", "rfsrnp_rol5"}
    _check_new_arms(S)


def _check_new_arms(S):
    from tools.nopart import sim_ext as E
    for seed in S.SEEDS:
        for new, ref in (("rfsrnp", "rfsr"), ("rfsrnp_rol5", "rfsr_rol5")):
            a, b = S.build_config(new, seed), S.build_config(ref, seed)
            assert a["PFO"]["ROUTER_CHECKPOINT"] == str(E.ROUTER) and a["RFS"]["ROUTER"] == "r_s2dx2np"
            for c in (a, b):
                c["PFO"].pop("ROUTER_CHECKPOINT"), c["RFS"].pop("ROUTER"), c["PFO"].pop("ARM"), c["PFO"].pop("BASE_ARM", None)
                c["TRAIN"].pop("OUTPUT_DIR"), c["TRAIN"].pop("RUN_NAME"), c["EVAL"].pop("OUTPUT_DIR")
                c["MODEL"].pop("INIT_FROM", None)
            assert a == b, (new, seed)


def test_hooked_module_matches_native():
    """With the hook in tools/rfsu/sim.py: the module carries the two arms and their configs differ from rfsr's only
    in the router checkpoint, its label, the arm and run names."""
    import importlib
    S = importlib.import_module("tools.rfsu.sim")
    if not getattr(S, "NOPART_ARMS_ADDED", False):
        pytest.skip("the hook is not installed yet")
    _check_new_arms(S)
