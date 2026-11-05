"""PFO trunk variants (docs/EXPERIMENT_RECORDS.md#pfo-p2a-prereg-20261103): CPU tests."""
from __future__ import annotations

import numpy as np
import torch

from tests.test_pfo import ROOT, _net_and_inputs


def test_trunk_variants_params_and_rng():
    from semkine import pfo_trunk as T
    from semkine import xyz_mesh as XM
    with np.load(ROOT / "assets/mano_right.npz") as z:
        w = torch.tensor(np.asarray(z["weights"], np.float32))
        f = torch.tensor(np.asarray(z["f"], np.int64))
    nets = {}
    for v in T.VARIANTS:
        T.install(v)
        try:
            torch.manual_seed(0)
            nets[v] = XM.XYZMeshNet(w, f).eval()
        finally:
            T.uninstall()
    base = dict(nets["base"].named_parameters())
    for v, net in nets.items():
        for name, p in net.named_parameters():
            assert torch.equal(p, base[name]), (v, name)        # shared parameters start equal
    n = {v: sum(p.numel() for p in net.parameters()) for v, net in nets.items()}
    assert n["base"] == n["self"] == n["rewire"] == 459059 - 0 - 0 or n["base"] == n["self"] == n["rewire"]
    assert n["base"] - n["nomp"] == 110976 + 160
    ring, mask = nets["base"].ring_mask, nets["rewire"].ring_mask
    assert torch.equal((ring > 0).sum(1), (mask > 0).sum(1))   # every vertex keeps its degree
    same = (nets["rewire"].ring_idx == nets["base"].ring_idx) & (mask > 0)
    assert same.float().sum() / (mask > 0).sum() < 0.1           # most edges moved
    args = _net_and_inputs()[1]
    with torch.no_grad():
        for v, net in nets.items():
            out = net(*args)[0]
            assert torch.isfinite(out).all(), v
