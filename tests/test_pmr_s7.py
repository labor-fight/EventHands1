"""PMR stage S7, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s7-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- semkine/pmr_trunk8.py: both combinations build inside MNISTModel, keep D1's shared parameters at one seed (ghand's
  gate embedding is new), start with a zero increment, every parameter takes part in the backward; ghand + hp gates read
  hp's gate types and stay invariant to the hand's global pose; the bypass audit holds with the messages zeroed.
- The S7 job list; the written decision on its rule.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import pmr_trunk as PM  # noqa: E402
from semkine import pmr_trunk8 as P8  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from tests.test_pmr import _inputs, _jac, _mano, _net, _redraw_zero_layers  # noqa: E402


def _combo(variant, seed=1):
    mano = _mano()
    torch.manual_seed(seed)
    return P8.CLASSES[variant](mano.weights, mano.f)


@pytest.mark.parametrize("arm", ("d1ghandhp", "d1ghandhub"))
def test_builds_inside_the_model(arm):
    from model import MNISTModel
    from tools.pmr import arms8 as A8
    cfg = A8.build_config(arm, 3407, "preflight")
    cfg["_config_path"] = "test"
    P8.install(cfg["PMR"]["VARIANT"])
    try:
        torch.manual_seed(0)
        model = MNISTModel(cfg)
    finally:
        XM.XYZMeshNet = PM._NATIVE["cls"]
    assert isinstance(model.xyz, P8.CLASSES[arm[2:]])


@pytest.mark.parametrize("variant", P8.VARIANTS)
def test_zero_start_shared_parameters_backward(variant):
    _, _, args = _inputs()
    net = _combo(variant).train()
    delta, _, _ = net(*args)
    assert torch.count_nonzero(delta) == 0
    r, mine = _net("r").state_dict(), net.state_dict()
    assert all(torch.equal(r[k], mine[k]) for k in set(r) & set(mine) if not k.startswith("edge_emb."))
    _redraw_zero_layers(net)
    delta, _, _ = net(*args)
    delta.square().sum().backward()
    assert not [k for k, p in net.named_parameters() if p.grad is None]


def test_ghandhp_gates_read_hp_types_and_ignore_the_global_pose():
    torch.manual_seed(3)
    B = 2
    J = torch.randn(B, 16, 3) * 0.03 + torch.tensor([0.0, 0.0, 0.5])
    prev48 = torch.randn(B, 48) * 0.2
    F = PM.node_frames(prev48, _mano().hands_mean)
    Q = PM.rodrigues(torch.tensor([[0.3, -0.5, 0.8]])).expand(B, 3, 3)
    J2 = torch.einsum("bij,bkj->bki", Q, J - J[:, :1]) + J[:, :1]
    F2 = Q.unsqueeze(1) @ F
    net = _combo("ghandhp").eval()
    assert int(net.e_type.max()) == 5 and int(net.e_type_gate.max()) == 3
    with torch.no_grad():
        assert torch.allclose(net.gates_from(J, F), net.gates_from(J2, F2), atol=1e-5)


@pytest.mark.parametrize("variant", P8.VARIANTS)
def test_bypass_audit_with_messages_zeroed(variant):
    net = _combo(variant).train()
    _redraw_zero_layers(net)
    _, _, args = _inputs()
    with torch.no_grad():
        n = net.node_inputs(*args)
    G = _jac(net, n["x_part"].float(), n["x_hub"].float(), messages=False)
    for j in range(1, 16):
        rows = G[6 + 3 * (j - 1): 9 + 3 * (j - 1)]
        assert torch.count_nonzero(rows[:, [i for i in range(17) if i != j]]) == 0


def test_jobs_and_decision():
    from tools.pmr import analyze10 as A10
    from tools.pmr import jobs13 as J13
    J = J13.s7(["d1ghandhp", "d1ghandhub"], "s7")
    assert sum(j["kind"] == "pair" for j in J) == 4 and J[-1]["key"] == "s7:decide"
    assert all("train8.py" in " ".join(j["command"]) for j in J if j["kind"] == "pair")

    def row(L, G, ra, e3):
        return {"five": [L, G, 0.0, 0.0, ra], "e3": e3}
    rows = {"b2": row(14.8, 8.6, 11.47, 5.6), "d1n": row(16.5, 11.8, 13.97, 4.7), "d1m": row(17.0, 8.7, 12.58, 6.5),
            "d1ghand": row(12.91, 8.44, 10.52, 6.0)}
    s = A10.decide({**rows, "d1ghandhp": row(12.9, 8.0, 10.1, 6.0), "d1ghandhub": row(13.5, 7.9, 10.0, 6.0)},
                   ["d1ghandhp", "d1ghandhub"])
    assert s["verdict"] == "REPLACE_ghandhp"                          # hub's local is worse than the guard allows
    assert A10.decide({**rows, "d1ghandhp": row(12.9, 8.3, 10.4, 6.0)}, ["d1ghandhp"])["verdict"] == "KEEP_ghand"
