"""PMR engineering amendment 3 (docs/EXPERIMENT_RECORDS.md#pmr-amend3-20261109), CPU. Run with CUDA_VISIBLE_DEVICES="".

- Every registered arm (S1 designs, the ladder rungs, the rounds rungs) builds inside MNISTModel with its config (the
  check that would have caught d1l1 / d1l2: semkine/xag.py pins ENCODER_LAYERS with the PFA routes on).
- The rounds rungs keep R's first k rounds: their parameters equal R's at one seed (the dropped rounds' parameters
  absent), the initial increment is zero, and every parameter takes part in the backward.
- A rounds rung's config differs from D1's R config only in the PMR block and names / directories / identity keys.
"""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from semkine import pmr_trunk as PM  # noqa: E402
from semkine import pmr_trunk3 as P3  # noqa: E402
from semkine import xyz_mesh as XM  # noqa: E402
from tests.test_pmr import _inputs, _mano, _net  # noqa: E402

REGISTRIES = ("tools.pmr.arms", "tools.pmr.arms2", "tools.pmr.arms3")


def _all_arms():
    out = []
    for reg in REGISTRIES:
        mod = importlib.import_module(reg)
        out += [(reg, a) for a in mod.arms()]
    return out


@pytest.mark.parametrize("reg,arm", _all_arms())
def test_every_registered_arm_builds_inside_the_model(reg, arm):
    from model import MNISTModel
    mod = importlib.import_module(reg)
    if reg == "tools.pmr.arms2" and arm in ("d1l1", "d1l2"):
        pytest.skip("replaced by d1l1b / d1l2b (amendment 3): ENCODER_LAYERS is pinned by semkine/xag.py")
    cfg = mod.build_config(arm, 3407, "preflight")
    cfg["_config_path"] = "test"
    trunk = importlib.import_module(cfg["PMR"]["MODULE"])
    trunk.install(cfg["PMR"]["VARIANT"])
    try:
        torch.manual_seed(0)
        model = MNISTModel(cfg)
    finally:
        XM.XYZMeshNet = PM._NATIVE["cls"]
    assert isinstance(model.xyz, PM.PMRNet)


@pytest.mark.parametrize("variant,k", (("r1", 1), ("r2", 2)))
def test_rounds_rung_keeps_rs_first_rounds(variant, k):
    mano = _mano()
    torch.manual_seed(1)
    net = type("T", (P3.PMRRoundsNet,), {"variant": "r", "keep_rounds": k})(mano.weights, mano.f)
    assert len(net.rounds) == k
    r, mine = _net("r").state_dict(), net.state_dict()
    assert set(mine) <= set(r) and all(torch.equal(r[key], mine[key]) for key in mine)
    assert {key for key in set(r) - set(mine)} == {key for key in r if key.startswith(tuple(f"rounds.{i}." for i in range(k, 3)))}
    _, _, args = _inputs()
    net.train()
    delta, _, _ = net(*args)
    assert torch.count_nonzero(delta) == 0
    delta.square().sum().backward()
    assert not [key for key, p in net.named_parameters() if p.grad is None]


@pytest.mark.parametrize("arm", ("d1l1b", "d1l2b"))
def test_rounds_rung_configs_differ_from_r_only_in_registered_keys(arm):
    from tools.pmr import arms as A
    from tools.pmr import arms3 as A3

    def flat(d, pre=""):
        out = {}
        for key, v in d.items():
            out.update(flat(v, pre + key + ".") if isinstance(v, dict) else {pre + key: v})
        return out
    for phase in ("preflight", "formal"):
        a, b = flat(A3.build_config(arm, 3407, phase)), flat(A.build_config("d1r", 3407, phase))
        diff = {key for key in set(a) | set(b) if a.get(key, "<absent>") != b.get(key, "<absent>")}
        assert diff <= {"EVAL.OUTPUT_DIR", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "PFO.ARM", "PFO.SECTION", "PMR.DESIGN",
                        "PMR.VARIANT", "PMR.MODULE", "PMR.BASE_ARM"}
        assert a["MODEL.ENCODER_LAYERS"] == 3
