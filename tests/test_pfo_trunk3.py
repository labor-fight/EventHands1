"""PFO cheaper message passing (semkine/pfo_trunk3.py): CPU tests."""
from __future__ import annotations

import numpy as np
import torch

from tests.test_pfo import ROOT, _net_and_inputs


def _build(variant):
    from semkine import pfo_trunk3 as T
    from semkine import xyz_mesh as XM
    with np.load(ROOT / "assets/mano_right.npz") as z:
        w = torch.tensor(np.asarray(z["weights"], np.float32))
        f = torch.tensor(np.asarray(z["f"], np.int64))
    T.install(variant)
    try:
        torch.manual_seed(0)
        return XM.XYZMeshNet(w, f).eval()
    finally:
        T.uninstall()


def test_params_and_shared_initial_values():
    from semkine import xyz_mesh as XM
    with np.load(ROOT / "assets/mano_right.npz") as z:
        w = torch.tensor(np.asarray(z["weights"], np.float32))
        f = torch.tensor(np.asarray(z["f"], np.int64))
    torch.manual_seed(0)
    base = XM.XYZMeshNet(w, f).eval()
    nb = sum(p.numel() for p in base.parameters())
    l1, gs = _build("l1"), _build("gshare")
    assert nb - sum(p.numel() for p in l1.parameters()) == 2 * 36992
    assert nb - sum(p.numel() for p in gs.parameters()) == 2 * (32 * 128 + 128)
    assert torch.equal(gs.gate_shared.weight, base.layers[0].gate.weight)
    assert torch.equal(gs.layers[2].msg.weight, base.layers[2].msg.weight)
    _, args = _net_and_inputs()
    with torch.no_grad():
        for net in (l1, gs):
            out = net(*args)[0]
            assert torch.isfinite(out).all()
        # gshare equals the base when every round's gate equals round 1's
        for k in (1, 2):
            base.layers[k].gate.load_state_dict(base.layers[0].gate.state_dict())
        for p in (base.fc2.weight, base.fc2.bias):
            torch.nn.init.normal_(p, std=0.01)
        gs.fc2.load_state_dict(base.fc2.state_dict())
        assert torch.allclose(gs(*args)[0], base(*args)[0], atol=1e-6)
