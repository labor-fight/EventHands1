"""PFO tools (docs/EXPERIMENT_RECORDS.md#pfo-prereg-20261103): CPU tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]


def _net_and_inputs(seed=0):
    from semkine import xyz_mesh as XM
    with np.load(ROOT / "assets/mano_right.npz") as z:
        w = torch.tensor(np.asarray(z["weights"], np.float32))
        f = torch.tensor(np.asarray(z["f"], np.int64))
        v = torch.tensor(np.asarray(z["v_template"], np.float32))
    torch.manual_seed(seed)
    net = XM.XYZMeshNet(w, f).eval()
    for p in net.fc2.parameters():
        torch.nn.init.normal_(p, std=0.01)
    B, N = 2, 64
    tok = torch.rand(B, N, 7)
    d = torch.cat([torch.randn(B, N, 2) * 0.05, torch.ones(B, N, 1)], -1)
    mask = torch.rand(B, N) > 0.2
    verts = (v + torch.tensor([0.0, 0.0, 0.45])).unsqueeze(0).repeat(B, 1, 1)
    vid = torch.randint(0, v.shape[0], (B, N, 4))
    A = torch.rand(B, N, 4)
    A[..., 3] = 0
    A = A / A.sum(-1, keepdim=True)
    q = (torch.rand(B, N) > 0.3).float()
    assoc = {"vid": vid, "A": A, "q": q, "z": torch.full((B, N), 0.45) + torch.rand(B, N) * 0.01,
             "valid": q > 0, "hit": q > 0}
    prev48 = torch.randn(B, 48) * 0.1
    return net, (tok, d, mask, verts, assoc, prev48)


def test_diag_forward_equals_native_without_interventions():
    from semkine import pfo_diag as D
    net, args = _net_and_inputs()
    with torch.no_grad():
        ref = net(*args)[0]
        D.install()
        try:
            D.set_interventions(())
            D.set_capture(True)
            out = net(*args)[0]
            caps = D.captured()
        finally:
            D.uninstall()
    assert torch.equal(ref, out)
    assert caps and caps[0]["slots"].shape == (2, 17, 258)


def test_diag_interventions_change_output():
    from semkine import pfo_diag as D
    net, args = _net_and_inputs(1)
    with torch.no_grad():
        ref = net(*args)[0]
        D.install()
        try:
            for name in ("dxyz_zero", "gnn_skip", "gnn_rewire", "resid_zero", "agg_max_only"):
                D.set_interventions((name,))
                assert not torch.equal(net(*args)[0], ref), name
        finally:
            D.uninstall()
    assert torch.equal(net(*args)[0], ref)


def test_phase1_jobs_unique_and_ordered():
    from tools.pfo.jobs import phase1
    jobs = phase1()
    keys = [j["key"] for j in jobs]
    assert len(keys) == len(set(keys))
    pos = {k: i for i, k in enumerate(keys)}
    for j in jobs:
        assert all(pos[d] < pos[j["key"]] for d in j["deps"])
    assert sum(j["kind"] == "exclusive" for j in jobs) == 9
