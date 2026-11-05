"""PFO trunk variants, second set (docs/EXPERIMENT_RECORDS.md#pfo-p2b-prereg-20261103): CPU tests."""
from __future__ import annotations

import numpy as np
import torch

from tests.test_pfo import ROOT, _net_and_inputs


def _build(variant, seed=0):
    from semkine import pfo_trunk2 as T
    from semkine import xyz_mesh as XM
    with np.load(ROOT / "assets/mano_right.npz") as z:
        w = torch.tensor(np.asarray(z["weights"], np.float32))
        f = torch.tensor(np.asarray(z["f"], np.int64))
    T.install(variant)
    try:
        torch.manual_seed(seed)
        net = XM.XYZMeshNet(w, f).eval()
    finally:
        T.uninstall()
    for p in net.fc2.parameters():                                      # the same draws as tests/test_pfo.py
        torch.nn.init.normal_(p, std=0.01)
    return net


def test_base2_and_zero_init_variants_equal_native():
    native, args = _net_and_inputs()
    with torch.no_grad():
        ref = native(*args)[0]
        for v in ("base2", "pos"):
            net = _build(v)
            assert torch.allclose(net(*args)[0], ref, atol=0, rtol=0) or torch.equal(net(*args)[0], ref), v


def test_shared_parameters_start_equal_and_variants_differ():
    base = _build("base2")
    for v in ("pos", "hframe", "noprev"):
        net = _build(v)
        own = dict(net.named_parameters())
        for name, p in base.named_parameters():
            assert torch.equal(p, own[name]), (v, name)
    _, args = _net_and_inputs()
    with torch.no_grad():
        ref = base(*args)[0]
        assert not torch.equal(_build("hframe")(*args)[0], ref)
        assert not torch.equal(_build("noprev")(*args)[0], ref)
        net = _build("pos")
        torch.nn.init.normal_(net.pos_in.weight, std=0.1)
        assert not torch.equal(net(*args)[0], ref)
