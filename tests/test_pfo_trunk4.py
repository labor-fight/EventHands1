"""PFO composable trunk (semkine/pfo_trunk4.py) equals every single-change variant: CPU tests."""
from __future__ import annotations

import numpy as np
import torch

from tests.test_pfo import ROOT, _net_and_inputs


def _mano():
    with np.load(ROOT / "assets/mano_right.npz") as z:
        return torch.tensor(np.asarray(z["weights"], np.float32)), torch.tensor(np.asarray(z["f"], np.int64))


def _build(module, variant):
    from semkine import xyz_mesh as XM
    w, f = _mano()
    module.install(variant)
    try:
        torch.manual_seed(0)
        net = XM.XYZMeshNet(w, f).eval()
    finally:
        module.uninstall()
    torch.manual_seed(1)
    for p in net.fc2.parameters():
        torch.nn.init.normal_(p, std=0.01)
    if hasattr(net, "pos_in"):
        torch.nn.init.normal_(net.pos_in.weight, std=0.05)
    return net


def test_trunk4_matches_single_variants():
    from semkine import pfo_trunk as T1, pfo_trunk2 as T2, pfo_trunk3 as T3, pfo_trunk4 as T4
    _, args = _net_and_inputs()
    pairs = [(T1, "base", "full"), (T1, "nomp", "nomp"), (T1, "self", "self"), (T1, "rewire", "rewire"),
             (T2, "pos", "pos"), (T2, "hframe", "hframe"), (T2, "noprev", "noprev"), (T3, "l1", "l1"),
             (T3, "gshare", "gshare")]
    with torch.no_grad():
        for mod, single, comp in pairs:
            a, b = _build(mod, single), _build(T4, comp)
            assert dict((k, v.shape) for k, v in a.state_dict().items()) == dict((k, v.shape) for k, v in b.state_dict().items()), comp
            b.load_state_dict(a.state_dict())
            assert torch.allclose(a(*args)[0], b(*args)[0], atol=1e-6, rtol=0), comp


def test_trunk4_combinations_build_and_run():
    from semkine import pfo_trunk4 as T4
    _, args = _net_and_inputs()
    with torch.no_grad():
        for v in ("nomp+pos", "nomp+noprev", "gshare+pos+noprev", "l1+hframe"):
            out = _build(T4, v)(*args)[0]
            assert torch.isfinite(out).all(), v
