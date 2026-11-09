"""PMR stage S4, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s4-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- semkine/pmr_trunk4.py: every refinement builds inside MNISTModel with its registered config, starts with a zero
  increment, keeps D1's shared parameters at one seed, and every parameter takes part in the backward; dual's extra
  statistics are the camera-frame moments of each part about its prior joint; hp's palm <-> hand edges are retyped and
  read the palm with weight 1 at the hand node; the structural bypass audit holds for dual / hp with the messages zeroed.
- tools/pmr/arms4.py: configs differ from D1's R only in the PMR block and names; the job list is well formed; the
  written selection on synthetic numbers.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import pmr_trunk as PM  # noqa: E402
from semkine import pmr_trunk4 as P4  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from tests.test_pmr import _inputs, _mano, _net, _redraw_zero_layers  # noqa: E402


def _ref(variant, seed=1):
    mano = _mano()
    torch.manual_seed(seed)
    return type(f"T_{variant}", (P4.PMRRefineNet,), {"variant": P4.BASE[variant], "refine": variant})(mano.weights, mano.f)


@pytest.mark.parametrize("arm", ("d1dual", "d1dualn", "d1dualm", "d1hp"))
def test_builds_inside_the_model(arm):
    from model import MNISTModel
    from tools.pmr import arms4 as A4
    cfg = A4.build_config(arm, 3407, "preflight")
    cfg["_config_path"] = "test"
    P4.install(cfg["PMR"]["VARIANT"])
    try:
        torch.manual_seed(0)
        model = MNISTModel(cfg)
    finally:
        XM.XYZMeshNet = PM._NATIVE["cls"]
    assert isinstance(model.xyz, P4.PMRRefineNet)


@pytest.mark.parametrize("variant", P4.VARIANTS)
def test_zero_start_shared_parameters_and_backward(variant):
    _, _, args = _inputs()
    net = _ref(variant).train()
    delta, _, _ = net(*args)
    assert torch.count_nonzero(delta) == 0
    r, mine = _net(P4.BASE[variant]).state_dict(), net.state_dict()
    assert all(torch.equal(r[k], mine[k]) for k in set(r) & set(mine))
    delta.square().sum().backward()
    assert not [k for k, p in net.named_parameters() if p.grad is None]


def test_dual_extra_statistics_are_camera_frame_moments():
    _, _, args = _inputs()
    net = _ref("dual").eval()
    with torch.no_grad():
        n = net.node_inputs(*args)
    B = n["J"].shape[0]
    eye = torch.eye(3, dtype=n["J"].dtype).expand(B, 16, 3, 3)
    _, ref = PM.moments(n["omega"], n["P"], n["res"], n["J"], eye)
    assert torch.allclose(n["x_extra"], ref) and n["x_extra"].shape[-1] == P4.EXTRA_STATS


def test_hp_palm_edges_are_retyped():
    net = _ref("hp")
    E = P4.hp_edges()
    palm_in = [k for s, t, k in E if s == PM.HUB and t == 0]
    palm_out = [k for s, t, k in E if s == 0 and t == PM.HUB]
    assert palm_in == [4] and palm_out == [5]
    S5 = net.e_scatter5 if hasattr(net, "e_scatter5") else getattr(net, "e_scatter5")
    assert S5.shape == (1, PM.N_NODES) and float(S5[0, PM.HUB]) == 1.0
    S3 = getattr(net, "e_scatter3")
    assert S3.shape[0] == 15 and torch.allclose(S3[:, PM.HUB], torch.full((15,), 1 / 15))
    assert all(len(rd.msg) == 6 for rd in net.rounds)
    assert torch.equal(net.e_type_gate[net.e_type == 4], torch.full((1,), 2)) and torch.equal(
        net.e_type_gate[net.e_type == 5], torch.full((1,), 3))


@pytest.mark.parametrize("variant", ("dual", "hp"))
def test_bypass_audit_with_messages_zeroed(variant):
    from tests.test_pmr import _jac
    net = _ref(variant).train()
    _redraw_zero_layers(net)
    _, _, args = _inputs()
    with torch.no_grad():
        n = net.node_inputs(*args)
    xp, xh = n["x_part"].float(), n["x_hub"].float()
    if variant == "dual":
        xe = n["x_extra"].float().requires_grad_(True)
        xp_ = xp.clone().requires_grad_(True)
        h0 = net.encode(xp_, xh, xe)
        out = net.readout(net.propagate(h0, None, messages=False))
        for j in range(1, 16):
            g = torch.autograd.grad(out[:, 6 + 3 * (j - 1): 9 + 3 * (j - 1)].sum(), (xp_, xe), retain_graph=True)
            others = [i for i in range(16) if i != j]
            assert torch.count_nonzero(g[0][:, others]) == 0 and torch.count_nonzero(g[1][:, others]) == 0
    else:
        G = _jac(net, xp, xh, messages=False)
        for j in range(1, 16):
            rows = G[6 + 3 * (j - 1): 9 + 3 * (j - 1)]
            assert torch.count_nonzero(rows[:, [i for i in range(17) if i != j]]) == 0


@pytest.mark.parametrize("arm", ("d1dual", "d1dualn", "d1dualm", "d1hp"))
def test_configs_differ_from_r_only_in_registered_keys(arm):
    from tools.pmr import arms as A
    from tools.pmr import arms4 as A4

    def flat(d, pre=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, pre + k + ".") if isinstance(v, dict) else {pre + k: v})
        return out
    a, b = flat(A4.build_config(arm, 3407, "formal")), flat(A.build_config("d1r", 3407, "formal"))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert diff <= {"EVAL.OUTPUT_DIR", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "PFO.ARM", "PFO.SECTION", "PMR.DESIGN",
                    "PMR.VARIANT", "PMR.MODULE", "PMR.BASE_ARM"}


def test_job_list_and_selection_rule():
    from tools.pmr import analyze6 as A6
    from tools.pmr import jobs7 as J7
    J = J7.s4("s4")
    assert sum(j["kind"] == "pair" for j in J) == 8 and J[-1]["key"] == "s4:select"

    def row(L, G, ra, e3):
        return {"five": [L, G, 0.0, 0.0, ra], "e3": e3}
    rows = {"b2": row(14.8, 8.6, 11.47, 5.6), "d1r": row(12.7, 9.96, 11.22, 5.9), "d1n": row(16.5, 11.8, 13.97, 4.7),
            "d1m": row(17.0, 8.7, 12.58, 6.5), "d1dual": row(12.6, 9.2, 10.9, 5.8), "d1dualn": row(16.4, 11.5, 13.8, 4.8),
            "d1dualm": row(16.9, 8.8, 12.4, 6.3), "d1hp": row(12.8, 9.7, 11.3, 5.9)}
    s = A6.select(rows)
    assert s["verdict"] == "REFINE_dual" and s["candidates"]["d1hp"]["adopted"] is False
    rows["d1dual"] = row(12.6, 9.7, 10.9, 5.8)
    assert A6.select(rows)["verdict"] == "NO_REFINEMENT"
