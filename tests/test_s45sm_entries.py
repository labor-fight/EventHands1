"""CPU tests: the GPU entries of the consolidated S45-SM package (evaluate, row, readouts) run end to end with the XCL
evaluator replaced by a recorder: what they resolve, install, record in launch.json / report.json and the swap receipts."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.s45sm import evaluate as E  # noqa: E402
from tools.s45sm import paths as P  # noqa: E402
from tools.s45sm import protocol as PR  # noqa: E402
from tools.s45sm import readouts as RO  # noqa: E402
from tools.s45sm import registry as R  # noqa: E402
from tools.s45sm import routing as RT  # noqa: E402


@pytest.fixture()
def rig(monkeypatch, tmp_path):
    from semkine import s45sm, s45sm_fast, s45sm_oracle, s45sm_soft
    from tools.s45tr import readouts as TRR
    from tools.xcl import evaluate as XE
    seen = {"trace": [], "row": []}

    def trace(run, checkpoint, cfg, output, mode, canonical, intervention):
        seen["trace"].append({"run": run, "checkpoint": Path(checkpoint).name, "mode": mode, "canonical": canonical,
                              "routing": RT.installed_routing(), "oracle": s45sm_oracle.installed()})
        return {"metrics": {"local": {"mpjpe_ra_mm": 1.0}}}

    def row(run, checkpoint, cfg, output, report, step):
        seen["row"].append((run["name"], step))
        return {"path": "p", "sha256": "s"}
    monkeypatch.setattr(XE, "configure_environment", lambda: None)
    monkeypatch.setattr(XE, "set_policy", lambda: None)
    monkeypatch.setattr(XE, "trace", trace)
    monkeypatch.setattr(XE, "make_main_row", row)
    monkeypatch.setattr(E, "OUT", tmp_path)
    monkeypatch.setattr(RO, "OUT", tmp_path)
    monkeypatch.setattr(RT, "swap_receipt_path", lambda net, entry, what: tmp_path / f"{net}_{RT.ENTRY_LABEL[entry]}_{what}.json")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")

    def run_task(net, task, limit=None, device_name="cuda", out_root=None):
        model, cfg, checkpoint = TRR.load(net, "cpu")
        seen.setdefault("readouts", []).append((net, task, Path(checkpoint).name, RT.installed_routing(), s45sm_oracle.installed()))
        target = Path(out_root) / net / task
        target.mkdir(parents=True)
        (target / "states.npz").write_bytes(b"x")
    monkeypatch.setattr(TRR, "run_task", run_task)
    yield seen, tmp_path
    for module in (s45sm, s45sm_fast, s45sm_soft, s45sm_oracle):
        module.uninstall()
    RT.unregister_kinds()


def reset():
    from semkine import s45sm, s45sm_fast, s45sm_oracle, s45sm_soft
    for module in (s45sm, s45sm_fast, s45sm_soft, s45sm_oracle):
        module.uninstall()


def ev(*args):
    try:
        return E.evaluate(*args)
    finally:
        reset()


def row(*args):
    try:
        return E.row_fast(*args)
    finally:
        reset()


def rd(*args):
    try:
        return RO.readout(*args)
    finally:
        reset()


def launch(tmp, net, name):
    return json.loads((tmp / "evaluations" / net / name / "launch.json").read_text())


def test_round1_arm_evaluation_and_slow_row(rig):
    seen, tmp = rig
    ev("pfat0pdz", "e1", 5500)
    L = launch(tmp, "pfat0pdz", "e1_step5500")
    assert L["step"] == 5500 and L["routing"]["router"] == "t0p" and "hook_modules_sha256" not in L and "oracle_module_sha256" not in L
    assert L["router_checkpoint_sha256"] == P.sha(R.router_ready("t0p")) and L["code_identity_sha256"] == PR.code_identity()
    assert seen["trace"][0]["checkpoint"] == "s45sm_pfat0pdz_s3407-step=5500.ckpt" and seen["trace"][0]["canonical"] is None
    assert seen["trace"][0]["routing"]["fast"] is False
    ev("pfat0pdz", "e2", 6000)
    assert seen["trace"][1]["canonical"] == P.S45_E1_TRACE
    ev("pfat0pdz", "row", 6000)
    assert seen["row"] == [("s45sm_pfat0pdz_s3407", 6000)]
    report = json.loads((tmp / "evaluations/pfat0pdz/row_step6000/report.json").read_text())
    assert report["main_row"] == {"path": "p", "sha256": "s"} and report["no_training_or_selection"] is True
    with pytest.raises(RuntimeError, match="Preserve existing"):
        ev("pfat0pdz", "e2", 6000)
    with pytest.raises(RuntimeError, match="late checkpoints"):
        ev("pfat0pdz", "e3", 5000)


def test_later_arms_have_no_slow_row_and_record_their_hook_modules(rig):
    seen, tmp = rig
    with pytest.raises(RuntimeError, match="latency path"):
        ev("pfat0pdz_cos", "row", 6000)
    ev("pfat0pdz_cos", "e3", 6000)
    L = launch(tmp, "pfat0pdz_cos", "e3_step6000")
    assert set(L["hook_modules_sha256"]) == {"s45sm", "s45sm_fast", "s45sm_soft"} and "fast" not in L["routing"]
    assert seen["trace"][0]["routing"]["fast"] is False                   # a constant-path arm runs the original routes
    with pytest.raises(RuntimeError, match="unregistered"):
        ev("no_such_arm", "e1", 6000)


def test_the_s45_reference_and_the_oracle(rig):
    seen, tmp = rig
    ev("s45ref", "row", 6000)
    L = launch(tmp, "s45ref", "row_step6000")
    assert L["routing"] is None and L["router_checkpoint_sha256"] is None and seen["row"] == [("s45sm_s45ref", 6000)]
    assert seen["trace"][0]["routing"] is None and seen["trace"][0]["oracle"] is None
    with pytest.raises(RuntimeError, match="unregistered"):
        ev("s45ref", "e1", 6000)
    ev("pfagtdz", "e3", 6000)
    L = launch(tmp, "pfagtdz", "e3_step6000")
    assert L["routing"]["router"] == "gt" and "oracle_module_sha256" in L and "router_checkpoint_sha256" not in L
    assert seen["trace"][1]["oracle"] == L["routing"]
    ev("pfagtdz", "row", 6000)
    assert seen["row"][-1] == ("s45sm_pfagtdz_s3407", 6000)


def test_a_cell_evaluates_its_trunk_checkpoint_and_writes_the_swap_receipt(rig):
    seen, tmp = rig
    ev("xt0_srdwa", "e1", 5000)
    L = launch(tmp, "xt0_srdwa", "e1_step5000")
    assert seen["trace"][0]["checkpoint"] == "s45sm_pfat0pdz_s3407-step=5000.ckpt" and L["routing"]["router"] == "dw"
    receipt = json.loads((tmp / "xt0_srdwa_evaluate2_e1_step5000.json").read_text())
    assert receipt["trunk"] == "pfat0pdz" and receipt["trunk_routes"] == "t0p" and receipt["router"] == "srdwkda" and receipt["step"] == 5000
    assert receipt["checkpoint_sha256"] == L["checkpoint_sha256"] and receipt["config_sha256"] == L["config_sha256"]
    ev("pfat0pdz", "e3", 6000)
    assert not list(tmp.glob("pfat0pdz_*"))                                 # a trained arm writes no swap receipt


def test_the_latency_path_rows(rig):
    seen, tmp = rig
    row("pfat0pdz", "r99")
    L = launch(tmp, "pfat0pdz", "row_fast_step6000")
    assert L["mode"] == "row_fast" and L["block"] == "r99" and L["routing"]["fast"] is True and "code_identity_sha256" in L
    assert seen["row"] == [("s45sm_fast_pfat0pdz_s3407", 6000)]
    row("s45ref_r99", "r99")
    assert launch(tmp, "s45ref_r99", "row_fast_step6000")["routing"] is None and seen["row"][-1] == ("s45sm_fast_s45ref_r99", 6000)
    with pytest.raises(RuntimeError, match="unregistered"):
        row("s45ref_r98", "r99")
    with pytest.raises(RuntimeError, match="block tag"):
        row("pfat0pdz", "not a tag")
    row("xt0_srdwa", "r99")
    assert json.loads((tmp / "xt0_srdwa_evaluate_fast_row_fast.json").read_text())["step"] == 6000
    with pytest.raises(RuntimeError, match="unregistered"):
        row("pfagtdz", "r99")                                         # the oracle's row is the slow one


def test_a_failure_is_preserved(rig, monkeypatch):
    seen, tmp = rig
    from tools.xcl import evaluate as XE
    monkeypatch.setattr(XE, "trace", lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError):
        ev("pfat0pdz_cos", "e1", 6000)
    failure = json.loads((tmp / "evaluations/pfat0pdz_cos/e1_step6000/failure.json").read_text())
    assert failure["status"] == "FAILED_PRESERVED" and failure["error"] == "ValueError: boom"
    assert not (tmp / "evaluations/pfat0pdz_cos/e1_step6000/report.json").exists()


def test_one_gpu_among_2_to_7_is_required(rig, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="one GPU"):
        ev("pfat0pdz", "e1", 6000)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    with pytest.raises(RuntimeError, match="one GPU"):
        row("pfat0pdz", "r99")


def test_readouts_resolve_their_network(rig):
    seen, tmp = rig
    rd("pfat0pdz_cos_5500", "tl_e1")
    net, task, ckpt, routing, oracle = seen["readouts"][0]
    assert (net, task, ckpt) == ("pfat0pdz_cos_5500", "tl_e1", "s45sm_pfat0pdz_cos_s3407-step=5500.ckpt") and routing["fast"] is False
    rd("xt0_srdwa", "zl_e3")
    assert seen["readouts"][1][2] == "s45sm_pfat0pdz_s3407-step=6000.ckpt"
    assert json.loads((tmp / "xt0_srdwa_readouts2_zl_e3.json").read_text())["router"] == "srdwkda"
    rd("pfagtdz", "tl_f1")
    assert seen["readouts"][2][4] is not None and seen["readouts"][2][3] is None
    with pytest.raises(RuntimeError, match="not registered"):
        rd("pfat0pdz_5000", "tl_e3")                                 # late steps: tl_e1 only
    with pytest.raises(RuntimeError, match="unregistered"):
        rd("nothing", "tl_e1")
    assert RO.parse("a_b_5000") == ("a_b", 5000) and RO.parse("a_b") == ("a_b", 6000) and RO.parse("x_5500") == ("x", 5500)


# --------------------------------------------------------------------------- the trainer's audit callbacks (need B, so a subprocess)
def test_train_audit_classes_and_the_virtual_audit_module(tmp_path):
    import os
    import subprocess
    code = f"""
import json, sys, types
from pathlib import Path
sys.path[:0] = [{str(ROOT)!r}, {str(ROOT / 'model')!r}]
from tools.xyz_c1_repro import train as B
from tools.s45sm import train as T
from tools.s45sm.registry import config_path
import yaml
X, A, O = T.audit_classes(B)
T.register_audit_module(X)
import tools.xag.train_audit as TA
assert TA.XAGTrainAudit is X and TA.write is T.write
assert issubclass(A, B.ReproAudit) and issubclass(O, B.ReproAudit) and A.runtime_record is not O.runtime_record
out = Path({str(tmp_path)!r})
trainer = types.SimpleNamespace(global_rank=0)
for arm in ("pfat0pdz_cos", "pfagtdz"):
    cfg = yaml.safe_load(config_path(arm, "formal").read_text())
    cfg["_config_path"] = str(config_path(arm, "formal"))
    d = out / arm
    d.mkdir()
    X(d).on_fit_start(trainer, types.SimpleNamespace(cfg=cfg))
    rec = json.loads((d / "xag_binding.json").read_text())
    print(arm, sorted(rec))
print(sorted(set(T.VALIDATION)))
"""
    env = {**os.environ, "PYTHONHASHSEED": "3407", "CUDA_VISIBLE_DEVICES": "", "S45SM_ROOT_ALIAS": os.environ.get("S45SM_ROOT_ALIAS", "")}
    done = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, env=env, timeout=600, cwd=str(ROOT))
    assert done.returncode == 0, done.stderr[-1500:]
    lines = done.stdout.strip().splitlines()
    assert "routing" in lines[-3] and "xag_source_build_summary_sha256" not in lines[-3]
    assert "xag_source_build_summary_sha256" in lines[-2] and "routing" not in lines[-2]
