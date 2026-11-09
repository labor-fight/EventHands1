"""PMR stage S4c, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s4c-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- semkine/pmr_trunk6.py: both gate variants build inside MNISTModel, start with a zero increment, keep D1's shared
  parameters at one seed (the geometry embedding is new), and every parameter takes part in the backward; ghand's gates do
  not change under a global rotation of the prior hand (joints and root frame rotated together) while D1's do; gview's
  read no direction; the bypass audit holds with the messages zeroed.
- The S4c job list; the joint selection over five candidates; the S5 tooling takes the S4c arms.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import pmr_trunk as PM  # noqa: E402
from semkine import pmr_trunk6 as P6  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from tests.test_pmr import _inputs, _jac, _mano, _net, _redraw_zero_layers  # noqa: E402


def _gate(kind, seed=1):
    mano = _mano()
    torch.manual_seed(seed)
    return type(f"T_{kind}", (P6.PMRGateNet,), {"variant": "r", "gate_kind": kind})(mano.weights, mano.f)


@pytest.mark.parametrize("arm", ("d1ghand", "d1gview"))
def test_builds_inside_the_model(arm):
    from model import MNISTModel
    from tools.pmr import arms6 as A6
    cfg = A6.build_config(arm, 3407, "preflight")
    cfg["_config_path"] = "test"
    P6.install(cfg["PMR"]["VARIANT"])
    try:
        torch.manual_seed(0)
        model = MNISTModel(cfg)
    finally:
        XM.XYZMeshNet = PM._NATIVE["cls"]
    assert isinstance(model.xyz, P6.PMRGateNet)


@pytest.mark.parametrize("kind", P6.VARIANTS)
def test_zero_start_shared_parameters_backward(kind):
    _, _, args = _inputs()
    net = _gate(kind).train()
    delta, _, _ = net(*args)
    assert torch.count_nonzero(delta) == 0
    r, mine = _net("r").state_dict(), net.state_dict()
    assert all(torch.equal(r[k], mine[k]) for k in set(r) & set(mine) if not k.startswith("edge_emb."))
    assert mine["edge_emb.weight"].shape[1] == P6.GEO_IN[kind]
    delta.square().sum().backward()
    assert not [k for k, p in net.named_parameters() if p.grad is None]


def test_ghand_gates_are_invariant_to_the_global_pose():
    torch.manual_seed(3)
    B = 2
    J = torch.randn(B, 16, 3) * 0.03 + torch.tensor([0.0, 0.0, 0.5])
    prev48 = torch.randn(B, 48) * 0.2
    mano = _mano()
    F = PM.node_frames(prev48, mano.hands_mean)
    Q = PM.rodrigues(torch.tensor([[0.3, -0.5, 0.8]])).expand(B, 3, 3)       # a global rotation of the hand
    J2 = torch.einsum("bij,bkj->bki", Q, J - J[:, :1]) + J[:, :1]
    F2 = Q.unsqueeze(1) @ F
    net = _gate("ghand").eval()
    with torch.no_grad():
        assert torch.allclose(net.gates_from(J, F), net.gates_from(J2, F2), atol=1e-5)
        d1 = _net("r").eval()
        assert not torch.allclose(d1.gates(J), d1.gates(J2), atol=1e-3)


def test_gview_reads_no_direction():
    J = torch.randn(1, 16, 3) * 0.03 + torch.tensor([0.0, 0.0, 0.5])
    F = torch.eye(3).expand(1, 16, 3, 3)
    g = P6.gate_features("gview", J, F, torch.tensor([0, 1]), torch.tensor([1, 2]), torch.tensor([0, 0]))
    assert g.shape[-1] == P6.GEO_IN["gview"]


@pytest.mark.parametrize("kind", P6.VARIANTS)
def test_bypass_audit_with_messages_zeroed(kind):
    net = _gate(kind).train()
    _redraw_zero_layers(net)
    _, _, args = _inputs()
    with torch.no_grad():
        n = net.node_inputs(*args)
    G = _jac(net, n["x_part"].float(), n["x_hub"].float(), messages=False)
    for j in range(1, 16):
        rows = G[6 + 3 * (j - 1): 9 + 3 * (j - 1)]
        assert torch.count_nonzero(rows[:, [i for i in range(17) if i != j]]) == 0


def test_jobs_joint_selection_and_s5_tooling():
    from tools.pmr import analyze6c as C
    from tools.pmr import jobs10 as J10
    from tools.pmr import jobs8 as J8
    J = J10.s4c("s4c")
    assert sum(j["kind"] == "pair" for j in J) == 4 and J[-1]["key"] == "s4c:joint"
    assert "s4:read:d1hp:e1" in J[-1]["deps"] and "s4b:read:d1pos:e1" in J[-1]["deps"]

    def row(L, G, ra, e3):
        return {"five": [L, G, 0.0, 0.0, ra], "e3": e3}
    rows = {"b2": row(14.8, 8.6, 11.47, 5.6), "d1r": row(12.7, 9.96, 11.22, 5.9), "d1n": row(16.5, 11.8, 13.97, 4.7),
            "d1m": row(17.0, 8.7, 12.58, 6.5), "d1dual": row(12.6, 9.7, 10.9, 5.8), "d1dualn": row(16.4, 11.5, 13.8, 4.8),
            "d1dualm": row(16.9, 8.8, 12.4, 6.3), "d1hp": row(12.8, 9.7, 11.3, 5.9), "d1pos": row(12.3, 9.6, 10.6, 5.5),
            "d1posn": row(16.0, 11.0, 13.5, 4.9), "d1posm": row(16.2, 8.6, 11.9, 6.0), "d1ghand": row(12.5, 9.0, 10.7, 5.8),
            "d1gview": row(13.5, 9.3, 11.4, 5.9)}
    s = C.select(rows)
    assert s["verdict"] == "REFINE_ghand" and set(s["candidates"]) == {"d1dual", "d1hp", "d1pos", "d1ghand", "d1gview"}
    J5 = J8.s5("d1ghand", "s5")
    first = [j for j in J5 if j["key"] == "s5:rtrain:d1ghand_rol5:s3407"][0]
    assert "s4c:accept:d1ghand" in first["deps"]
    assert any("train6.py" in " ".join(j["command"]) for j in J5)
