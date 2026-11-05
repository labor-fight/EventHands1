"""EVF second model registry (semkine/evf_model2.py, semkine/evf_graph.py): CPU tests."""
from __future__ import annotations

import pytest
import torch

from tests.test_evf import base_cfg, packets, randomize_readout, states


def build(cls, seed=0, **evf):
    torch.manual_seed(seed)
    return cls(base_cfg(**evf)).eval()


def model1(seed=0, **evf):
    from semkine.evf_model import EVFModel
    return build(EVFModel, seed, **evf)


def model2(seed=0, **evf):
    from semkine.evf_model2 import EVFModel2
    return build(EVFModel2, seed, **evf)


@pytest.mark.parametrize("evf", [{}, {"VF": False}, {"LBS": True}, {"PREV48": False}, {"ITERS": 2}])
def test_round1_settings_are_the_round1_model(evf):
    a, b = model1(seed=3, **evf), model2(seed=3, **evf)
    sa, sb = a.state_dict(), b.state_dict()
    assert list(sa) == list(sb) and all(torch.equal(sa[k], sb[k]) for k in sa)
    randomize_readout(a)
    randomize_readout(b)
    prev = states(4, seed=4)
    batch = packets(a, prev, seed=5)
    with torch.no_grad():
        assert torch.equal(a.forward_packet(batch), b.forward_packet(batch))


@pytest.mark.parametrize("evf", [{"NET": "graph", "HIDDEN": 32}, {"VFMODE": "concat"}, {"READOUT": "surface"}])
def test_variants_identity_gate_and_gradients(evf):
    m = model2(seed=6, **evf)
    prev = states(4, seed=7)
    batch = packets(m, prev, seed=8)
    with torch.no_grad():
        out = m.forward_packet(batch)
    assert out.shape == (4, 51) and torch.equal(out, prev)
    randomize_readout(m)
    m.train()
    out = m.forward_packet(batch)
    assert torch.isfinite(out).all() and torch.equal(out[2].detach(), prev[2])
    loss, _ = m._compute_loss(out, batch.target + 0.01, batch.betas)
    loss.backward()
    missing = [n for n, p in m.named_parameters() if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    assert not missing, missing
    zero = [n for n, p in m.named_parameters() if p.grad is not None and float(p.grad.abs().sum()) == 0.0]
    assert not zero, zero


def test_graph_levels_are_nested_and_local():
    from semkine import evf_nodes as EN
    from semkine.evf_graph import CELLS, build_levels
    from semkine.evf_render import Frame
    m = model2(seed=9, NET="graph", HIDDEN=32)
    prev = states(2, seed=10)
    batch = packets(m, prev, seed=11, n_per=(1500, 300), dts=(0.05, 0.05))
    fx, fy, cx, cy = m._intrinsics(batch.camera_K)
    with torch.no_grad():
        verts, joints = m._fk(prev, batch.betas)
        fr = Frame(verts, joints, m.mano.f, fx, fy, cx, cy, 180, 240)
        src, mask, _ = EN.select_nodes(batch.events, batch.ptr, batch.delta_t_s, 180, 240)
        lv = build_levels(batch.events, src, mask, batch.delta_t_s, fr, m.evf_template, 180, 240)
    n_prev = mask.sum(1)
    for k, lvl in enumerate(lv):
        n = lvl["mask"].sum(1)
        assert bool((n <= n_prev).all()) and bool((n > 0).all())
        live = lvl["parent"] >= 0
        assert torch.equal(live.sum(1), n_prev)                       # every live child has a parent
        assert int(lvl["parent"].max()) < lvl["mask"].shape[1]
        ok = lvl["nbr"] >= 0
        assert bool((lvl["edge"][..., :2].abs()[ok] <= 2.0 + 1e-5).all())  # neighbours within one cell (centroids)
        n_prev = n
    assert len(lv) == len(CELLS)
