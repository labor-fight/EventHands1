"""PMR stage S4b, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s4b-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- semkine/pmr_trunk5.py: the three variants build inside MNISTModel, start with a zero increment, keep D1's shared
  parameters at one seed (plus lev_in, zero), and every parameter takes part in the backward; the lever phi reads is the
  routed point about the prior joint of the event's dominant part in that part's frame / L_LEV (zero off-hand); with
  lev_in at zero the node inputs equal D1's exactly; the bypass audit holds with the messages zeroed.
- The S4b job list and the joint S4 + S4b selection on synthetic numbers.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import pmr_trunk as PM  # noqa: E402
from semkine import pmr_trunk5 as P5  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from tests.test_pmr import _inputs, _jac, _mano, _net, _redraw_zero_layers  # noqa: E402


def _pos(variant, seed=1):
    mano = _mano()
    torch.manual_seed(seed)
    return type(f"T_{variant}", (P5.PMRPosNet,), {"variant": P5.BASE[variant], "refine": variant})(mano.weights, mano.f)


@pytest.mark.parametrize("arm", ("d1pos", "d1posn", "d1posm"))
def test_builds_inside_the_model(arm):
    from model import MNISTModel
    from tools.pmr import arms5 as A5
    cfg = A5.build_config(arm, 3407, "preflight")
    cfg["_config_path"] = "test"
    P5.install(cfg["PMR"]["VARIANT"])
    try:
        torch.manual_seed(0)
        model = MNISTModel(cfg)
    finally:
        XM.XYZMeshNet = PM._NATIVE["cls"]
    assert isinstance(model.xyz, P5.PMRPosNet)


@pytest.mark.parametrize("variant", P5.VARIANTS)
def test_zero_start_shared_parameters_backward(variant):
    _, _, args = _inputs()
    net = _pos(variant).train()
    delta, _, _ = net(*args)
    assert torch.count_nonzero(delta) == 0
    r, mine = _net(P5.BASE[variant]).state_dict(), net.state_dict()
    assert set(mine) - set(r) == {"lev_in.weight"} and torch.count_nonzero(mine["lev_in.weight"]) == 0
    assert all(torch.equal(r[k], mine[k]) for k in set(r) & set(mine))
    delta.square().sum().backward()
    assert not [k for k, p in net.named_parameters() if p.grad is None]


def test_lever_and_equality_with_d1_at_zero_map():
    _, _, args = _inputs()
    net = _pos("pos").eval()
    ref = _net("r").eval()
    with torch.no_grad():
        n, m = net.node_inputs(*args), ref.node_inputs(*args)
    assert torch.equal(n["x_part"], m["x_part"]) and torch.equal(n["x_hub"], m["x_hub"])
    b, i = 0, int((n["m"][0] > 0).nonzero()[3])
    p = int(n["omega"][b, i].argmax())
    want = (n["P"][b, i] - n["J"][b, p]) @ n["F"][b, p] / PM.L_LEV
    assert torch.allclose(n["lev"][b, i].double(), want, atol=1e-6)
    off = n["m"] <= 0
    assert torch.count_nonzero(n["lev"][off]) == 0


def test_bypass_audit_with_messages_zeroed():
    net = _pos("pos").train()
    _redraw_zero_layers(net)
    _, _, args = _inputs()
    with torch.no_grad():
        n = net.node_inputs(*args)
    G = _jac(net, n["x_part"].float(), n["x_hub"].float(), messages=False)
    for j in range(1, 16):
        rows = G[6 + 3 * (j - 1): 9 + 3 * (j - 1)]
        assert torch.count_nonzero(rows[:, [i for i in range(17) if i != j]]) == 0


def test_job_list_and_joint_selection():
    from tools.pmr import analyze6b as B
    from tools.pmr import jobs9 as J9
    J = J9.s4b("s4b")
    assert sum(j["kind"] == "pair" for j in J) == 6 and J[-1]["key"] == "s4b:joint"
    assert "s4:read:d1hp:e1" in J[-1]["deps"]

    def row(L, G, ra, e3):
        return {"five": [L, G, 0.0, 0.0, ra], "e3": e3}
    rows = {"b2": row(14.8, 8.6, 11.47, 5.6), "d1r": row(12.7, 9.96, 11.22, 5.9), "d1n": row(16.5, 11.8, 13.97, 4.7),
            "d1m": row(17.0, 8.7, 12.58, 6.5), "d1dual": row(12.6, 9.7, 10.9, 5.8), "d1dualn": row(16.4, 11.5, 13.8, 4.8),
            "d1dualm": row(16.9, 8.8, 12.4, 6.3), "d1hp": row(12.8, 9.7, 11.3, 5.9), "d1pos": row(12.3, 9.1, 10.6, 5.5),
            "d1posn": row(16.0, 11.0, 13.5, 4.9), "d1posm": row(16.2, 8.6, 11.9, 6.0)}
    s = B.select(rows)
    assert s["verdict"] == "REFINE_pos" and set(s["candidates"]) == {"d1dual", "d1hp", "d1pos"}
