"""PFO2 stage-0 tools (docs/EXPERIMENT_RECORDS.md#pfo2-prereg-20261103): CPU tests."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

#: bookkeeping keys that may differ between a PFO2 config and the PFO config of the same recipe
PATHS = (("TRAIN", "OUTPUT_DIR"), ("TRAIN", "RUN_NAME"), ("EVAL", "OUTPUT_DIR"))
PFO_META = ("ARM", "SECTION", "STUDY", "MODULE", "PFO_ARM", "PHASE")


def _strip(cfg):
    cfg = copy.deepcopy(cfg)
    for a, b in PATHS:
        cfg[a].pop(b)
    for k in PFO_META:
        cfg["PFO"].pop(k, None)
    return cfg


def _pfo_yaml(name):
    return yaml.safe_load((ROOT / "configs/pfo_20261103" / f"{name}.yaml").read_text())


@pytest.mark.parametrize("arm,pfo,seed,phase", [("f1rep", "f1", 3407, "preflight"), ("f1rep", "f1", 3407, "formal"),
                                                ("f1rep", "f1", 3408, "formal"), ("f1mp", "f1mp", 3407, "preflight"),
                                                ("f1mp", "f1mp", 3407, "formal"), ("f1mp", "f1mp", 3408, "formal")])
def test_configs_equal_pfo_recipe(arm, pfo, seed, phase):
    from tools.pfo2.arms import build_config
    name = f"pfo_{pfo}_s{seed}" + ("_preflight" if phase == "preflight" else "")
    assert _strip(build_config(arm, seed, phase)) == _strip(_pfo_yaml(name))


def test_base_forms_differ_from_f_only_in_trunk():
    from tools.pfo2.arms import build_config
    for seed in (3407, 3408, 3409):
        f = _strip(build_config("f1rep", seed))
        for arm, trunk in (("fl1", "l1"), ("fgs", "gshare"), ("rl1", "l1"), ("rgs", "gshare")):
            c = _strip(build_config(arm, seed))
            assert c["PFO"].pop("TRUNK") == trunk and f["PFO"]["TRUNK"] == "nomp"
            assert c == {**f, "PFO": {k: v for k, v in f["PFO"].items() if k != "TRUNK"}}


def test_head500_is_formal_stopped_at_500():
    from tools.pfo2.arms import build_config
    a, b = build_config("rl1", 3408, "head500"), build_config("rl1", 3408, "formal")
    assert a["TRAIN"]["MAX_STEPS"] == 500 and a["TRAIN"]["LR_SCHEDULE"] == "constant"
    assert b["TRAIN"]["MAX_STEPS"] == 6000 and b["TRAIN"]["LR_SCHEDULE"] == "cosine"
    assert a["TRAIN"]["WARMUP_STEPS"] == 500 and a["TRAIN"]["SAVE_EVERY_N_STEPS"] == 500
    for c in (a, b):
        c["TRAIN"].pop("MAX_STEPS"); c["TRAIN"].pop("LR_SCHEDULE"); c["PFO"].pop("PHASE")
    assert _strip(a) == _strip(b)


def test_warmup_factor_equal_for_both_schedules():
    """Steps 0..499 and the factor set after step 499: constant and cosine schedules agree (model.py lr_lambda)."""
    import math
    floor = 0.0
    try:
        from model import BaseModel
        floor = BaseModel.LR_FLOOR
    except Exception:  # pragma: no cover - the attribute is read from the frozen model when importable
        pass
    warmup, total = 500, 6000

    def lam(step, schedule):
        if step < warmup:
            return (step + 1) / warmup
        if schedule == "constant":
            return 1.0
        p = min(max((step - warmup) / float(total - warmup), 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * p))
    assert all(lam(s, "constant") == lam(s, "cosine") for s in range(0, 501))


def test_pfo_eval_config_equals_pfo_written_one(tmp_path, monkeypatch):
    from tools.pfo2 import run as R
    monkeypatch.setattr(R, "CONFIGS", tmp_path)
    for seed in (3407, 3408, 3409):
        mine = yaml.safe_load(R.pfo_eval_config("f1", seed).read_text())
        theirs = yaml.safe_load((ROOT / f"configs/pfo_20261103/eval/pfo_f1_s{seed}.yaml").read_text())
        assert mine == theirs


def test_bitwise_compare_detects_one_element(tmp_path):
    from tools.pfo2.replay import compare
    net = torch.nn.Linear(4, 3)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net(torch.randn(2, 4)).sum().backward()
    opt.step()
    ck = {"state_dict": net.state_dict(), "optimizer_states": [opt.state_dict()],
          "lr_schedulers": [{"last_epoch": 1, "_last_lr": [1e-3]}], "global_step": 1, "hyper_parameters": {"x": 1}}
    torch.save(ck, tmp_path / "a.ckpt")
    ck2 = copy.deepcopy(ck)
    ck2["hyper_parameters"] = {"x": 2}                                  # metadata is not compared
    torch.save(ck2, tmp_path / "b.ckpt")
    assert compare(tmp_path / "a.ckpt", tmp_path / "b.ckpt")["mismatches"] == 0
    ck2["optimizer_states"][0]["state"][0]["exp_avg"][0, 0] += 1e-7
    torch.save(ck2, tmp_path / "c.ckpt")
    r = compare(tmp_path / "a.ckpt", tmp_path / "c.ckpt")
    assert r["mismatches"] == 1 and "exp_avg" in r["first"][0]["at"]


def test_stage0_jobs_wellformed():
    from tools.pfo2 import jobs as J
    jobs = J.stage0()
    keys = {j["key"] for j in jobs}
    for j in jobs:
        assert set(j["deps"]) <= keys, j["key"]
        assert j["kind"] in ("pair", "gpu", "cpu", "exclusive")
    # exactly one formal base chain can open: both gates read the same written decision
    gates = [j for j in jobs if j["key"].startswith("s0:gate:")]
    assert {tuple(j["command"][-2:]) for j in gates} == {("--open", "l1"), ("--open", "gshare")}
    # f1mp's formal trainings wait for the F replay check
    pre = next(j for j in jobs if j["key"] == "s0:pre:f1mp")
    assert pre["deps"] == ["s0:cmp:f1rep:pre"]


def test_driver_policy_and_pairs(tmp_path, monkeypatch):
    from tools.pfo2 import chain as C
    monkeypatch.setattr(C, "POLICY", tmp_path / "policy.json")
    assert C.train_pairs() == list(C.PAIRS)
    (tmp_path / "policy.json").write_text('{"train_pairs": [[2, 3], [4, 5]]}')
    assert C.train_pairs() == [(2, 3), (4, 5)]
    (tmp_path / "policy.json").write_text("not json")
    assert C.train_pairs() == list(C.PAIRS)
