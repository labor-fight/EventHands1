"""IDP CPU tests (docs/EXPERIMENT_RECORDS.md#idp-prereg-20261105). Run with CUDA_VISIBLE_DEVICES="".

- The three arms' configs differ from B2's (PFO2 fgs, seed 3407) only in the registered keys; the router block is B2's.
- semkine/idp_feat.py: construction draws B2's parameters (equal state on the common keys, the feature map at zero); the
  forward equals PFOMeshNet4("gshare") exactly while the map is zero and departs once it is not; the finger geometry is
  ~1 with small flexion on the fronto-parallel template and drops when the fingers are tilted at the camera; records
  take their vertex's finger.
- The label_part training hook installs the engine's mix on top of B2's hooks (and is undone); on a synthetic packet the
  mix keeps the router's on-hand set and takes the label's part where a label exists.
- The job list is well formed; the written rules on synthetic numbers.
"""
from __future__ import annotations

from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.idp import arms as A  # noqa: E402

BASE_KEYS = {"EVAL.OUTPUT_DIR", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "PFO.ARM", "PFO.BASE_ARM", "PFO.TRAIN_ROUTES",
             "PFO.FEAT", "PFO.SECTION", "PFO.STUDY"}


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + k + "."))
        else:
            out[pre + k] = v
    return out


@pytest.mark.parametrize("arm", tuple(A.ARMS))
@pytest.mark.parametrize("phase", A.PHASES)
def test_configs_differ_from_b2_only_in_registered_keys(arm, phase):
    from tools.pfo2 import arms as P2
    a, b = flat(A.build_config(arm, 3407, phase)), flat(P2.build_config("fgs", 3407, phase))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    expect = BASE_KEYS | ({"DATA.XAG_SOURCE_ROOT"} if A.ARMS[arm]["train_routes"] == "label_part" else set())
    assert diff == expect
    assert a["PFO.ROUTER"] == "r_s2d" and a["PFO.ROUTER_KIND"] == "pfr" and a["PFO.TRUNK"] == "gshare"
    assert a["PFO.FEAT"] == A.ARMS[arm]["feat"] and a["PFO.TRAIN_ROUTES"] == A.ARMS[arm]["train_routes"]


def _mano():
    with np.load(ROOT / "assets/mano_right.npz") as z:
        return (torch.tensor(np.asarray(z["weights"], np.float32)), torch.tensor(np.asarray(z["f"], np.int64)),
                torch.tensor(np.asarray(z["v_template"], np.float32)))


def _build(install, uninstall):
    from semkine import xyz_mesh as XM
    w, f, _ = _mano()
    install()
    try:
        torch.manual_seed(0)
        net = XM.XYZMeshNet(w, f).eval()
    finally:
        uninstall()
    torch.manual_seed(1)
    for p in net.fc2.parameters():
        torch.nn.init.normal_(p, std=0.01)
    return net


def _nets():
    from semkine import idp_feat as IF
    from semkine import pfo_trunk4 as T4
    base = _build(lambda: T4.install("gshare"), T4.uninstall)
    mine = _build(IF.install, IF.uninstall)
    return base, mine


def test_feat_trunk_equals_gshare_while_the_map_is_zero():
    from tests.test_pfo import _net_and_inputs
    base, mine = _nets()
    sb, sm = base.state_dict(), mine.state_dict()
    assert set(sm) - set(sb) == {"fore_in.weight"}
    assert all(torch.equal(sb[k], sm[k]) for k in sb)
    assert torch.count_nonzero(sm["fore_in.weight"]) == 0
    _, args = _net_and_inputs()
    with torch.no_grad():
        a, b = base(*args)[0], mine(*args)[0]
        assert torch.equal(a, b)
        torch.manual_seed(2)
        torch.nn.init.normal_(mine.fore_in.weight, std=0.1)
        assert not torch.equal(a, mine(*args)[0])


def _joints(net, verts):
    return torch.cat([torch.einsum("ji,bik->bjk", net.j_reg, verts), verts[:, net.tip_ids]], 1)[:, net.op21]


def test_finger_geometry_on_template_and_tilted():
    _, mine = _nets()
    _, _, v = _mano()
    verts = (v + torch.tensor([0.0, 0.0, 0.45])).unsqueeze(0)
    geo = mine.finger_geometry(verts)
    assert geo.shape == (1, 6, 2)
    assert (geo[0, :4, 0] >= 0.85).all() and (geo[0, :4, 0] <= 1.05).all(), geo[0, :, 0]
    assert (geo[0, :4, 1] <= 30.0).all(), geo[0, :, 1]
    assert geo[0, 5, 0] >= 0.85 and geo[0, 5, 1] == 0
    J = _joints(mine, v.unsqueeze(0))[0]
    d = J[8] - J[5]
    d = d / d.norm()
    axis = torch.cross(d, torch.tensor([0.0, 0.0, 1.0]), dim=0)
    axis = axis / axis.norm()
    th = torch.deg2rad(torch.tensor(70.0))
    K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = torch.eye(3) + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)
    c = v.mean(0, keepdim=True)
    tilted = ((v - c) @ R.T + c + torch.tensor([0.0, 0.0, 0.45])).unsqueeze(0)
    geo_t = mine.finger_geometry(tilted)
    assert geo_t[0, 0, 0] <= 0.45 and (geo_t[0, :4, 0] <= 0.8).all(), geo_t[0, :, 0]   # the index points at the camera
    assert float((geo[0, :4, 0] - geo_t[0, :4, 0]).mean()) >= 0.3
    assert torch.allclose(geo_t[0, :, 1], geo[0, :, 1], atol=1e-3)            # flexion is pose-only, not view-dependent


def test_record_features_follow_the_vertex_finger():
    _, mine = _nets()
    _, _, v = _mano()
    verts = (v + torch.tensor([0.0, 0.0, 0.45])).unsqueeze(0)
    geo = mine.finger_geometry(verts)
    vrec = torch.nonzero((mine.lbs_hard >= 1) & (mine.lbs_hard <= 3)).flatten()[:5]        # index-finger vertices
    f = mine.record_features(geo, torch.zeros_like(vrec), vrec)
    expect = torch.stack([(geo[0, 0, 0] - 0.8) / 0.2, geo[0, 0, 1] / 90.0]).expand(5, 2)
    assert torch.allclose(f, expect)
    palm = torch.nonzero(mine.lbs_hard == 0).flatten()[:3]
    fp = mine.record_features(geo, torch.zeros_like(palm), palm)
    assert torch.allclose(fp[:, 0], ((geo[0, 5, 0] - 0.8) / 0.2).expand(3)) and (fp[:, 1] == 0).all()


def _synthetic_packet(net, n=400, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(60, 180, (n,), generator=g).float()
    y = torch.randint(40, 140, (n,), generator=g).float()
    t = torch.sort(torch.rand(n, generator=g) * 0.05).values
    pol = torch.randint(0, 2, (n,), generator=g).float()
    events = torch.stack([torch.zeros(n), x, y, t, pol], -1)
    part = torch.randint(0, 16, (n,), generator=g)
    face = torch.stack([net.part_faces[p, torch.randint(0, int(net.part_faces_valid[p].sum()), (1,), generator=g)]
                        for p in part]).flatten()
    bary = torch.rand(n, 3, generator=g)
    bary = bary / bary.sum(-1, keepdim=True)
    valid = torch.rand(n, generator=g) < 0.5
    return types.SimpleNamespace(events=events, ptr=torch.tensor([0, n]), delta_t_s=torch.tensor([0.05]),
                                 xag_source_index=torch.where(valid, torch.arange(n), torch.full((n,), -1)),
                                 xag_source_ms=torch.where(valid, torch.arange(n), torch.full((n,), -1)),
                                 xag_part=torch.where(valid, part, torch.full((n,), -1)),
                                 xag_face=torch.where(valid, face, torch.full((n,), -1)),
                                 xag_bary=bary, xag_valid=valid)


def test_label_part_hook_and_mix_on_a_synthetic_packet():
    from semkine import s45sm as SM
    from semkine import xag as XAG
    from semkine import xyz_mesh as XM
    from tools.idp.train import install_routes
    from tools.pfo import nets as N
    p = A.build_config("l1", 3407, "formal")["PFO"]
    state = install_routes(p)
    try:
        assert XAG.sample_sources is N.mixed_sources and N._MIX["mode"] == "label_part"
        assert XAG.route_association is SM.pfa_association and XM.associate_rays is SM.empty_association
        assert state["mix"] == "label_part" and state["router"] == "pfr" and state["assoc"] == "pfa"
        net = SM._net(torch.device("cpu"))
        batch = _synthetic_packet(net)
        with torch.no_grad():
            src, mask = XM.stride_sample(batch.ptr, SM.MAX_NODES)
            out = XAG.sample_sources(batch, src, mask)
            from semkine import s45sm_fast as FAST
            r = FAST.router_forward(net, batch.events, batch.ptr, batch.delta_t_s, src, mask)
        r_sel = mask & (torch.sigmoid(r["onhand"].float()) >= 0.5)
        assert torch.equal(out["xag_valid"], r_sel)                              # the router's on-hand set
        lab = SM._NATIVE["sample_sources"](batch, src, mask)
        both = out["xag_valid"] & lab["xag_valid"]
        assert both.any()
        assert torch.equal(out["xag_part"][both], lab["xag_part"][both])        # the label's part
        only = out["xag_valid"] & ~lab["xag_valid"]
        if only.any():
            assert torch.equal(out["xag_part"][only], r["part"].float().argmax(-1)[only])
        assert torch.equal(net.face_part[out["xag_face"]], out["xag_part"])
    finally:
        SM.uninstall()
        N._MIX.update(mode=None, gen=None)
    assert XAG.sample_sources is SM._NATIVE["sample_sources"]


def test_jobs_well_formed(tmp_path):
    from tools.idp import jobs as J
    jobs = J.r1()
    seen = set()
    for j in jobs:
        assert all(d in seen for d in j["deps"]), j["key"]
        seen.add(j["key"])
    final = [j for j in jobs if j["key"] == "final:analyze"][0]
    reads = {j["key"] for j in jobs if j["key"].startswith("read:")}
    assert reads and reads <= set(final["deps"]) and "k:check" in final["deps"]
    mem = {j["key"]: j.get("mem_gib") for j in jobs if j["kind"] == "pair"}
    assert mem["train:l1"] == mem["train:l2"] == 260 and mem["train:l3"] == 130
    for arm in A.ARMS:
        assert {"k:check", f"pre:accept:{arm}"} <= set([j for j in jobs if j["key"] == f"train:{arm}"][0]["deps"])


def test_written_rules():
    from tools.idp.analyze import rules
    ret_b2 = {"R15": 0.448, "F15": 0.409}
    v = rules(11.47, 11.2, 10.5, 11.0, 5.59, 5.6, ret_b2, {"R15": 0.45, "F15": 0.41})
    assert v["L1"] == "POINT_ERROR_LIMITS" and v["L2"] == "FEAT_HELPS" and v["L3"] == "SCREEN_PASS"
    v = rules(11.47, 7.9, 7.6, 11.3, 5.59, 5.6, ret_b2, {"R15": 0.39, "F15": 0.35})
    assert v["L1"] == "IDENTITY_IS_LEVER" and v["L2"] == "FEAT_NULL" and v["L3"] == "SCREEN_PASS"
    v = rules(11.47, 9.5, 9.2, 11.0, 5.59, 6.0, ret_b2, {"R15": 0.45, "F15": 0.41})
    assert v["L1"] == "MIXED" and v["L3"] == "SCREEN_FAIL"
