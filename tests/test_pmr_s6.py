"""PMR stage S6, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s6-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- semkine/pmr_trunk7.py: d1hub builds inside MNISTModel, keeps D1's shared parameters at one seed, starts as D1's function
  (zero adapters), every parameter takes part in the backward, the trained adapters change the increment, and the bypass
  audit holds with the messages zeroed.
- The S6 job list; the written decision on its rule.
"""
from __future__ import annotations

from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import pmr_trunk as PM  # noqa: E402
from semkine import pmr_trunk7 as P7  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from tests.test_pmr import _inputs, _jac, _mano, _net, _redraw_zero_layers  # noqa: E402


def _hub(seed=1):
    mano = _mano()
    torch.manual_seed(seed)
    return type("T_hub", (P7.PMRHubNet,), {"variant": "r"})(mano.weights, mano.f)


def test_builds_inside_the_model():
    from model import MNISTModel
    from tools.pmr import arms7 as A7
    cfg = A7.build_config("d1hub", 3407, "preflight")
    cfg["_config_path"] = "test"
    P7.install(cfg["PMR"]["VARIANT"])
    try:
        torch.manual_seed(0)
        model = MNISTModel(cfg)
    finally:
        XM.XYZMeshNet = PM._NATIVE["cls"]
    assert isinstance(model.xyz, P7.PMRHubNet)


def test_starts_as_d1_with_its_shared_parameters():
    _, _, args = _inputs()
    d1, hub = _net("r").train(), _hub().train()
    a, b = d1.state_dict(), hub.state_dict()
    extra = [k for k in b if k not in a]
    assert extra and all(k.split(".")[0] in ("p2h_a", "p2h_b", "h2p_a", "h2p_b") for k in extra)
    assert all(torch.equal(a[k], b[k]) for k in a)
    _redraw_zero_layers(d1)
    _redraw_zero_layers(hub)
    with torch.no_grad():
        da, _, _ = d1(*args)
        db, _, _ = hub(*args)
    assert torch.allclose(da, db, atol=1e-6) and float(da.abs().max()) > 0
    n_extra = sum(p.numel() for k, p in hub.named_parameters() if k.split(".")[0] in ("p2h_a", "p2h_b", "h2p_a", "h2p_b"))
    assert 100_000 < n_extra < 130_000


def test_backward_and_adapters_matter():
    _, _, args = _inputs()
    hub = _hub().train()
    _redraw_zero_layers(hub)
    delta, _, _ = hub(*args)
    delta.square().sum().backward()
    assert not [k for k, p in hub.named_parameters() if p.grad is None]
    assert float(hub.p2h_b[0].weight.grad.abs().sum()) > 0 and float(hub.h2p_b[0].weight.grad.abs().sum()) > 0
    with torch.no_grad():
        before, _, _ = hub(*args)
        for m in list(hub.p2h_b) + list(hub.h2p_b):
            torch.nn.init.normal_(m.weight, std=0.05)
        after, _, _ = hub(*args)
    assert not torch.allclose(before, after, atol=1e-4)


def test_bypass_audit_with_messages_zeroed():
    hub = _hub().train()
    _redraw_zero_layers(hub)
    for m in list(hub.p2h_b) + list(hub.h2p_b):
        torch.nn.init.normal_(m.weight, std=0.05)
    _, _, args = _inputs()
    with torch.no_grad():
        n = hub.node_inputs(*args)
    G = _jac(hub, n["x_part"].float(), n["x_hub"].float(), messages=False)
    for j in range(1, 16):
        rows = G[6 + 3 * (j - 1): 9 + 3 * (j - 1)]
        assert torch.count_nonzero(rows[:, [i for i in range(17) if i != j]]) == 0


def test_jobs_and_decision():
    from tools.pmr import analyze9 as A9
    from tools.pmr import jobs12 as J12
    J = J12.s6("s6")
    assert sum(j["kind"] == "pair" for j in J) == 2 and J[-1]["key"] == "s6:decide" and "s4c:joint" in J[-1]["deps"]
    assert all("train7.py" in " ".join(j["command"]) for j in J if j["kind"] == "pair")

    def row(L, G, ra, e3):
        return {"five": [L, G, 0.0, 0.0, ra], "e3": e3}
    rows = {"b2": row(14.8, 8.6, 11.47, 5.6), "d1r": row(12.7, 9.96, 11.22, 5.9), "d1n": row(16.5, 11.8, 13.97, 4.7),
            "d1m": row(17.0, 8.7, 12.58, 6.5)}
    assert A9.select({**rows, "d1hub": row(12.6, 9.3, 10.9, 5.8)})["verdict"] == "ADOPT_hub"
    assert A9.select({**rows, "d1hub": row(12.6, 9.7, 11.1, 5.8)})["verdict"] == "HUB_NOT_ADOPTED"
