"""CPU tests: semkine/s45sm_soft.py (`spfa`) = PFA's records with q = the router's route probability."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import s45sm as SM  # noqa: E402
from semkine import s45sm_fast as FAST  # noqa: E402
from semkine import s45sm_soft as SOFT  # noqa: E402
from semkine import xag as XAG  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from semkine.events import EV_BATCH, EV_P, EV_T, EV_X, EV_Y  # noqa: E402

NPZ = ROOT / "assets/mano_right.npz"


def packets(seed=0, n_per=(900, 0, 400, 1500)):
    g = torch.Generator().manual_seed(seed)
    rows, ptr = [], [0]
    for b, n in enumerate(n_per):
        ev = torch.zeros(n, 5)
        ev[:, EV_BATCH] = b
        ev[:, EV_X] = torch.randint(80, 140, (n,), generator=g).float()
        ev[:, EV_Y] = torch.randint(60, 110, (n,), generator=g).float()
        ev[:, EV_T] = torch.sort(torch.rand(n, generator=g) * 0.05).values
        ev[:, EV_P] = (torch.rand(n, generator=g) > 0.5).float()
        rows.append(ev)
        ptr.append(ptr[-1] + n)
    return torch.cat(rows), torch.tensor(ptr), torch.full((len(n_per),), 0.05)


class Batch:
    def __init__(self, events, ptr, dt):
        self.events, self.ptr, self.delta_t_s = events, ptr, dt


def mesh(B):
    with np.load(NPZ, allow_pickle=False) as z:
        verts = torch.tensor(np.asarray(z["v_template"], np.float32))
        faces = torch.tensor(np.asarray(z["f"], np.int64))
        lbs = torch.tensor(np.asarray(z["weights"], np.float32))
    V = verts.unsqueeze(0).repeat(B, 1, 1)
    V[..., 2] += 0.45
    return V, faces, lbs


@pytest.fixture()
def sr_ckpt(tmp_path):
    torch.manual_seed(3)
    net = SM.build(NPZ)
    with torch.no_grad():
        net.onhand.bias.fill_(0.3)                                               # most nodes routed at tau 0.5
    path = tmp_path / "sr.ckpt"
    SM.save_checkpoint(net, path, step=0)
    return path


def run(module, ckpt, assoc, events, ptr, dt):
    batch = Batch(events, ptr, dt)
    src, mask = XM.stride_sample(ptr, SM.MAX_NODES)
    B = int(ptr.numel() - 1)
    V, faces, lbs = mesh(B)
    d = torch.cat([torch.rand(B, src.shape[1], 2) * 0.2 - 0.1, torch.ones(B, src.shape[1], 1)], -1)
    vf = XM.vertex_face_table(faces, V.shape[1])
    band = torch.full((B,), 0.02)
    try:
        module.install("sr", ckpt, assoc, 0.5)
        s = XAG.sample_sources(batch, src, mask)
        empty = XM.associate_rays(d, mask, V, faces, vf, band, band / 2, 0.01)
        out, audit = XAG.route_association("face_route", empty, d, mask, V, faces, vf, lbs, band, band / 2, 0.01,
                                           source_part=s["xag_part"], source_face=s["xag_face"],
                                           source_bary=s["xag_bary"], source_valid=s["xag_valid"])
        net = SM._net(src.device)
        conf = SOFT.route_confidence("sr", FAST.router_forward(net, events, ptr, dt, src, mask))
    finally:
        module.uninstall()
    return s, out, audit, conf


def test_spfa_is_pfa_with_route_probability(sr_ckpt):
    events, ptr, dt = packets(2)
    s1, pfa, a1, _ = run(FAST, sr_ckpt, "pfa", events, ptr, dt)
    s2, spfa, a2, conf = run(SOFT, sr_ckpt, "spfa", events, ptr, dt)
    for k in s1:
        assert torch.equal(s1[k], s2[k]), k
    for k in a1:
        assert torch.equal(a1[k], a2[k]), k
    for k in ("vid", "A", "z", "valid", "hit"):
        assert torch.equal(pfa[k], spfa[k]) and pfa[k].dtype == spfa[k].dtype, k
    e = a2["xag_eligible"]
    assert e.any() and (~e).any()
    assert torch.equal(spfa["q"][e], conf[e].to(spfa["q"].dtype))
    assert torch.equal(spfa["q"][~e], torch.zeros_like(spfa["q"][~e]))
    assert bool((spfa["q"][e] > 0).all() and (spfa["q"][e] <= 1).all() and (spfa["q"][e] < 1).any())
    assert torch.equal(pfa["q"], e.to(pfa["q"].dtype))


def test_hooks_restored_and_state(sr_ckpt):
    state = SOFT.install("sr", sr_ckpt, "spfa", 0.5)
    try:
        assert state["assoc"] == "spfa" and state["fast"] is True and SOFT.installed()["assoc"] == "spfa"
        assert XAG.sample_sources is SOFT.predicted_sources and XAG.route_association is SOFT.spfa_association
        assert XM.associate_rays is FAST.empty_association
    finally:
        SOFT.uninstall()
    assert SOFT.installed() is None and SM.installed() is None
    assert XAG.sample_sources is SM._NATIVE["sample_sources"]
    assert XAG.route_association is SM._NATIVE["route_association"]
    assert XM.associate_rays is SM._NATIVE["associate_rays"]
    with pytest.raises(ValueError):
        SOFT.install("sr", sr_ckpt, "pfa", 0.5)


def test_stash_belongs_to_routes(sr_ckpt):
    events, ptr, dt = packets(4)
    batch = Batch(events, ptr, dt)
    src, mask = XM.stride_sample(ptr, SM.MAX_NODES)
    B = int(ptr.numel() - 1)
    V, faces, lbs = mesh(B)
    d = torch.cat([torch.zeros(B, src.shape[1], 2), torch.ones(B, src.shape[1], 1)], -1)
    vf = XM.vertex_face_table(faces, V.shape[1])
    band = torch.full((B,), 0.02)
    SOFT.install("sr", sr_ckpt, "spfa", 0.5)
    try:
        s = XAG.sample_sources(batch, src, mask)
        empty = XM.associate_rays(d, mask, V, faces, vf, band, band / 2, 0.01)
        kw = dict(source_part=s["xag_part"], source_bary=s["xag_bary"], source_valid=s["xag_valid"])
        with pytest.raises(RuntimeError):
            XAG.route_association("face_route", empty, d, mask, V, faces, vf, lbs, band, band / 2, 0.01,
                                  source_face=s["xag_face"].clone(), **kw)
        s = XAG.sample_sources(batch, src, mask)
        kw = dict(source_part=s["xag_part"], source_bary=s["xag_bary"], source_valid=s["xag_valid"])
        XAG.route_association("face_route", empty, d, mask, V, faces, vf, lbs, band, band / 2, 0.01,
                              source_face=s["xag_face"], **kw)
        with pytest.raises(RuntimeError):                                       # the stash is used once
            XAG.route_association("face_route", empty, d, mask, V, faces, vf, lbs, band, band / 2, 0.01,
                                  source_face=s["xag_face"], **kw)
    finally:
        SOFT.uninstall()
