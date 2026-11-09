"""TPD final-configuration pieces, CPU tests (semkine/tpd_final.py). Run with CUDA_VISIBLE_DEVICES="" in a process of its
own (routing / trunk / geometry hooks are module patches).

- The final rollout samples equal ROL's samples (same draws) and carry n_prev: packet k >= 1 = packet k-1's event count,
  packet 0 without augmentation = the raw count of the previous 50 ms; with the XAG root the sources follow the events.
- The plain samples equal B2's and carry n_prev; collate gives CausalPacketBatch / RolloutBatch that survive .to().
- The model: the non-causal form's delta equals ROL's (and B2's) on real packets; the causal form takes the causal nodes
  (j % k == 0 from n_prev) and still trains (gradient reaches every weight on a rollout batch and a plain batch).
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import tpd_final as TF  # noqa: E402
from semkine import tpd_rol as R  # noqa: E402
from tools.tpd import arms_rol as A  # noqa: E402


@pytest.fixture(scope="module")
def base():
    from semkine.dataset import build_dataset
    cfg = A.build_config("rol3d", 3407, "formal")
    comps = np.load(cfg["MANO"]["NPZ"])["hands_components"].astype(np.float32)
    return cfg, comps, build_dataset


def test_rollout_samples(base):
    cfg, comps, build = base
    rol = R.upgrade(build(cfg, "train", comps, train=True), 3)
    fin = TF.upgrade_rollout(build(cfg, "train", comps, train=True), 3)
    for idx in (5, 123456, 2000000):
        a, b = rol[idx], fin[idx]
        for i in range(3):
            assert np.array_equal(a.packets[i].events, b.packets[i].events)
            assert np.array_equal(a.packets[i].prev_state, b.packets[i].prev_state)
            assert np.array_equal(a.packets[i].target, b.packets[i].target)
        assert b.packets[1].meta["n_prev"] == b.packets[0].n_events and b.packets[2].meta["n_prev"] == b.packets[1].n_events
    plain = copy.deepcopy(cfg)
    plain["AUG"]["DOMRAND"]["ENABLED"] = False
    fz = TF.upgrade_rollout(build(plain, "train", comps, train=True), 2)
    item = fz[777]
    si, ends, _, _ = fz.ends(777)
    h = fz._handle(si)
    s0 = ends[0] - 49
    assert item.packets[0].meta["n_prev"] == int(h.offsets[s0]) - int(h.offsets[s0 - 50])
    xcfg = copy.deepcopy(cfg)
    from tools.pfo.common import XAG_SOURCE_ROOT
    xcfg["DATA"]["XAG_SOURCE_ROOT"] = str(XAG_SOURCE_ROOT)
    fx = TF.upgrade_rollout(build(xcfg, "train", comps, train=True), 3)
    it = fx[123456]
    assert all(p.xag_face is not None and len(p.xag_face) == p.n_events for p in it.packets)
    assert np.array_equal(it.packets[1].events, fin[123456].packets[1].events)
    batch = TF.collate_final([fx[i] for i in (1, 2)])
    assert isinstance(batch, R.RolloutBatch) and batch.packets[0].n_prev.shape == (2,)
    moved = batch.to("cpu")
    assert torch.equal(moved.packets[2].n_prev, batch.packets[2].n_prev)


def test_plain_samples(base):
    cfg, comps, build = base
    ref = build(cfg, "train", comps, train=True)
    fp = TF.upgrade_plain(build(cfg, "train", comps, train=True))
    a, b = ref[4321], fp[4321]
    assert np.array_equal(a.events, b.events) and np.array_equal(a.prev_state, b.prev_state) and b.meta["n_prev"] >= 0
    ev = TF.upgrade_plain(build(cfg, "val_core", comps, train=False))
    p = ev[1000]
    h = ev._handle(int(ev.index[1000][0]))
    s = p.meta["end_idx"] - 49
    assert p.meta["n_prev"] == int(h.offsets[s]) - int(h.offsets[max(s - 50, 0)])
    bb = TF.collate_final([p, ev[1001]])
    assert isinstance(bb, TF.CausalPacketBatch) and bb.to("cpu").n_prev.tolist() == bb.n_prev.tolist()


@pytest.fixture(scope="module")
def model_and_batches(base):
    from semkine import pfo_router as PR
    from semkine import pfo_trunk4 as T4
    from semkine import s45sm as SM
    cfg, comps, build = base
    p = cfg["PFO"]
    PR.register()
    SM.install(p["ROUTER_KIND"], p["ROUTER_CHECKPOINT"], p["ASSOC"], float(p["ONHAND_TAU"]))
    T4.install("gshare")
    R.install_grad_geometry()
    try:
        fin = TF.upgrade_rollout(build(cfg, "train", comps, train=True), 3)
        rb = TF.collate_final([fin[i] for i in (11, 222222, 333333)])
        pl = TF.upgrade_plain(build(cfg, "train", comps, train=True))
        pb = TF.collate_final([pl[i] for i in (7, 70000, 700000)])
        yield cfg, rb, pb
    finally:
        R.uninstall_grad_geometry()
        T4.uninstall()
        SM.uninstall()
        PR.unregister()


def _model(cfg, causal):
    c = copy.deepcopy(cfg)
    c["FINAL"] = {"CAUSAL": causal}
    torch.manual_seed(0)
    m = TF.final_model_class()(c)
    m.init_from_abs_checkpoint(c["MODEL"]["INIT_FROM"])
    return m.eval()


def test_noncausal_equals_rol(model_and_batches):
    cfg, rb, _ = model_and_batches
    m = _model(cfg, False)
    rol = R.rol_model_class()(cfg)
    rol.init_from_abs_checkpoint(cfg["MODEL"]["INIT_FROM"])
    rol.eval()
    pk = rb.packets[1]
    with torch.no_grad():
        s, nodes = m._routes(pk)
        a = m._xyz_mesh_forward(pk, pk.prev_state, s, nodes)
        b = rol._xyz_mesh_forward(pk, pk.prev_state, rol._routes(pk, rol.xyz_max_nodes))
    assert torch.equal(a, b)


def test_causal_nodes_and_training(model_and_batches, tmp_path):
    """The causal form routes at its own nodes: semkine/s45sm_fast.py's deployed routes (B2's training hook re-samples
    the router's nodes and refuses)."""
    from semkine import s45sm_fast as FAST
    from semkine import tpd_causal as TC
    from semkine import xag as XAG
    cfg, rb, pb = model_and_batches
    native = XAG.sample_sources
    XAG.sample_sources = FAST.predicted_sources
    try:
        _causal(cfg, rb, pb, tmp_path, TC)
    finally:
        XAG.sample_sources = native


def _causal(cfg, rb, pb, tmp_path, TC):
    m = _model(cfg, True)
    m.cfg["TRAIN"]["OUTPUT_DIR"] = str(tmp_path)
    pk = rb.packets[2]
    src, mask = m._nodes(pk)
    k = TC.k_from_prev(pk.n_prev)
    for b in range(pk.batch_size):
        sel = src[b][mask[b]] - pk.ptr[b]
        assert torch.all(sel % k[b] == 0) and int(mask[b].sum()) <= 2048
    m.train()
    m.zero_grad(set_to_none=True)
    out = m.training_step(rb, 0)
    assert out.requires_grad
    out.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    m.zero_grad(set_to_none=True)
    out2 = m.training_step(pb, 0)
    out2.backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
