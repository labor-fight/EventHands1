"""PMR design D2, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s9-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- semkine/pmr_trunk9.py: d1hubfull builds inside MNISTModel, keeps D1's shared parameters at one seed, starts as D1's
  function (zero maps), every parameter takes part in the backward, the trained maps change the increment, and the bypass
  audit holds with the messages zeroed.
- tools/pmr/arms_any4.py takes d1hubfull; its rol5 config differs from it exactly as D1's rol5 configs differ from D1.
- The D2 job list is well formed (rollouts on the direct pairs only); the verdict reads the S5 rule with D1's controls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.environ.setdefault("PMR_DRIVER_DIR", "driver4")

from semkine import pmr_trunk as PM  # noqa: E402
from semkine import pmr_trunk9 as P9  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from tests.test_pmr import _inputs, _jac, _mano, _net, _redraw_zero_layers  # noqa: E402


def _hubfull(seed=1):
    mano = _mano()
    torch.manual_seed(seed)
    return type("T_hubfull", (P9.PMRHubFullNet,), {"variant": "r"})(mano.weights, mano.f)


def test_builds_inside_the_model():
    from model import MNISTModel
    from tools.pmr import arms9 as A9
    cfg = A9.build_config("d1hubfull", 3407, "preflight")
    cfg["_config_path"] = "test"
    P9.install(cfg["PMR"]["VARIANT"])
    try:
        torch.manual_seed(0)
        model = MNISTModel(cfg)
    finally:
        XM.XYZMeshNet = PM._NATIVE["cls"]
    assert isinstance(model.xyz, P9.PMRHubFullNet)


def test_starts_as_d1_and_trains():
    _, _, args = _inputs()
    d1, net = _net("r").train(), _hubfull().train()
    a, b = d1.state_dict(), net.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a) and {k.split(".")[0] for k in b if k not in a} == {"p2h_w"}
    assert 750_000 < sum(p.numel() for k, p in net.named_parameters() if k.startswith("p2h_w")) < 820_000
    _redraw_zero_layers(d1)
    _redraw_zero_layers(net)
    with torch.no_grad():
        da, _, _ = d1(*args)
        db, _, _ = net(*args)
    assert torch.allclose(da, db, atol=1e-6) and float(da.abs().max()) > 0
    delta, _, _ = net(*args)
    delta.square().sum().backward()
    assert not [k for k, p in net.named_parameters() if p.grad is None]
    assert float(net.p2h_w[0].weight.grad.abs().sum()) > 0
    with torch.no_grad():
        for m in net.p2h_w:
            torch.nn.init.normal_(m.weight, std=0.05)
        dc, _, _ = net(*args)
    assert not torch.allclose(db, dc, atol=1e-4)


def test_bypass_audit_with_messages_zeroed():
    net = _hubfull().train()
    _redraw_zero_layers(net)
    for m in net.p2h_w:
        torch.nn.init.normal_(m.weight, std=0.05)
    _, _, args = _inputs()
    with torch.no_grad():
        n = net.node_inputs(*args)
    G = _jac(net, n["x_part"].float(), n["x_hub"].float(), messages=False)
    for j in range(1, 16):
        rows = G[6 + 3 * (j - 1): 9 + 3 * (j - 1)]
        assert torch.count_nonzero(rows[:, [i for i in range(17) if i != j]]) == 0


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, pre + k + ".") if isinstance(v, dict) else {pre + k: v})
    return out


def test_registry_rollout_jobs_and_verdict():
    from tools.pmr import arms_any4  # noqa: F401
    from tools.pmr import arms as A
    from tools.pmr import arms_any2 as U
    from tools.pmr import arms_rol as R1
    from tools.pmr import arms_rol2 as R2
    from tools.pmr import jobs16 as J16
    assert "d1hubfull" in U.arms() and U.DESIGNS["hubfull"][0] == "semkine.pmr_trunk9"
    a, b = flat(R2.build_config("d1hubfull_rol5", 3408, "formal")), flat(U.build_config("d1hubfull", 3408, "formal"))
    c, e = flat(R1.build_config("d1r_rol5", 3408, "formal")), flat(A.build_config("d1r", 3408, "formal"))
    diff = lambda x, y: {k for k in set(x) | set(y) if x.get(k, "<absent>") != y.get(k, "<absent>")}  # noqa: E731
    assert diff(a, b) == diff(c, e)
    J = J16.s9("s9")
    own = {j["key"] for j in J}
    assert all(d in own for j in J for d in j["deps"])
    rol = [j for j in J if "train_rol4.py" in " ".join(j["command"])]
    assert len(rol) == 4 and all(j["pairs"] == [[4, 5], [6, 7]] for j in rol)
    assert sum("train9.py" in " ".join(j["command"]) for j in J) == 4
    assert J[-1]["command"][2:] == ["tools.pmr.analyze12", "s5", "--arm", "d1hubfull"]
    from tools.pmr import analyze12  # noqa: F401
    from tools.pmr import analyze7 as A7
    assert A7.NOPROP["d1hubfull"] == "d1n" and A7.MLP["d1hubfull"] == "d1m"
