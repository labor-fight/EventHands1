"""CPU tests: router-kind registration, hooks and network resolution of the consolidated S45-SM package (ported from the
round 3 hook tests and the round 7 / 9 / 11 swap-cell tests, which patched `resolve` through a wrapper)."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.s45sm import paths as P  # noqa: E402
from tools.s45sm import registry as R  # noqa: E402
from tools.s45sm import routing as RT  # noqa: E402
from tools.s45sm import summary as S  # noqa: E402

READY = ("srdwkd", "srpkdh", "srdwkdh")


@pytest.fixture(autouse=True)
def clean_kinds():
    yield
    RT.unregister_kinds()


def test_hook_module_choice():
    from semkine import s45sm, s45sm_fast, s45sm_soft
    assert RT.hook_module({"ASSOC": "spfa", "FAST": True}) is s45sm_soft
    with pytest.raises(RuntimeError):
        RT.hook_module({"ASSOC": "spfa"})
    assert RT.hook_module({"ASSOC": "pfa", "FAST": True}) is s45sm_fast
    assert RT.hook_module({"ASSOC": "pfa"}, force_fast=True) is s45sm_fast
    assert RT.hook_module({"ASSOC": "pfa"}) is s45sm


def test_install_eval_hook_spfa_and_forced_fast():
    from semkine import s45sm_fast, s45sm_soft
    from semkine import xag as XAG
    cfg = yaml.safe_load(R.config_path("spfasrdz", "formal").read_text())
    state = RT.install_eval_hook(cfg)
    try:
        assert state["assoc"] == "spfa" and state["fast"] is True
        assert RT.installed_routing()["assoc"] == "spfa" and RT.installed_routing()["fast"] is True
        assert XAG.route_association is s45sm_soft.spfa_association
    finally:
        s45sm_soft.uninstall()
    assert RT.installed_routing() is None
    cfg = yaml.safe_load(R.config_path("pfasrdz", "formal").read_text())
    state = RT.install_hook(cfg, force_fast=True)
    try:
        assert state["assoc"] == "pfa" and state["fast"] is True and RT.installed_routing()["fast"] is True
    finally:
        s45sm_fast.uninstall()


def test_install_hook_refuses_a_foreign_checkpoint_path():
    cfg = yaml.safe_load(R.config_path("pfasrdz", "formal").read_text())
    cfg["S45SM"]["ROUTER_CHECKPOINT"] = str(R.router_checkpoint("srkd"))
    with pytest.raises(RuntimeError, match="checkpoint path differs"):
        RT.install_hook(cfg)
    assert RT.installed_routing() is None


def test_register_kinds_by_need():
    from semkine import s45sm as SM
    natives = SM.ROUTERS
    assert RT.kinds_for("t0p") == () and RT.kinds_for("sr") == () and RT.kinds_for("srp") == ("srp",)
    assert RT.kinds_for("dwh") == ("dw", "hier") and RT.kinds_for("srph", cell=True) == RT.ALL_KINDS
    with pytest.raises(RuntimeError):
        RT.kinds_for("nonexistent")
    assert RT.register_for_net("pfat0pdz") == () and SM.ROUTERS == natives          # a T0P arm registers nothing
    assert RT.register_for_net("pfagtdz") == ()                                      # nor does the oracle
    assert RT.register_for_net("xt0_srph") == RT.ALL_KINDS and {"srp", "dw", "srph", "dwh", "t0ph"} <= set(SM.ROUTERS)
    RT.register_kinds()                                                              # idempotent
    RT.unregister_kinds()
    assert SM.ROUTERS == natives
    RT.register_kinds(("srp",))
    assert SM.ROUTERS == natives + ("srp",)


def test_cells_resolve_to_their_trunks_and_register_every_kind(monkeypatch):
    frozen = []
    monkeypatch.setattr(RT, "config_frozen", lambda config: frozen.append(Path(config)))
    from semkine import s45sm as SM
    RT.register_for_net("xgt_srdwah")
    assert {"srp", "dw", "srph", "dwh", "t0ph"} <= set(SM.ROUTERS)
    ckpt, config = RT.resolve("xgt_srdwah", 5000)
    assert ckpt == R.trunk_checkpoint("pfagtdz", 5000) and config == R.config_path("xgt_srdwah", "formal")
    ckpt, _ = RT.resolve("xt0_srdw", 6000)
    assert ckpt == R.trunk_checkpoint("pfat0pdz", 6000) and ckpt.parent.name == "s45sm_pfat0pdz_s3407"
    ckpt, _ = RT.resolve("xgt_srp", 5500)                                            # a round-7 cell (key `trunk`)
    assert ckpt == R.trunk_checkpoint("pfagtdz", 5500)
    assert frozen == [R.config_path("xgt_srdwah", "formal"), R.config_path("xt0_srdw", "formal"), R.config_path("xgt_srp", "formal")]
    ckpt, config = RT.resolve("pfasrdz", 6000)                                       # a trained arm: no config_frozen call
    assert ckpt.name == "s45sm_pfasrdz_s3407-step=6000.ckpt" and config == R.config_path("pfasrdz", "formal") and len(frozen) == 3
    ckpt, config = RT.resolve("pfagtdz", 6000)
    assert ckpt.parent.name == "s45sm_pfagtdz_s3407" and config == R.config_path("pfagtdz", "formal")
    with pytest.raises(RuntimeError):
        R.trunk_checkpoint("pfat0pdz", 4000)
    with pytest.raises(FileNotFoundError):
        RT.resolve("pfakdbank", 6000)                                               # never trained: no formal receipt


def test_trunk_checkpoints_share_one_architecture():
    shapes = {}
    for trunk in ("pfasrdz", "pfat0pdz", "pfagtdz"):
        paths = {step: R.trunk_checkpoint(trunk, step) for step in R.STEPS}
        assert all(p.is_file() for p in paths.values())
        sd = torch.load(str(paths[6000]), map_location="cpu")["state_dict"]
        shapes[trunk] = {k: tuple(v.shape) for k, v in sd.items() if torch.is_tensor(v)}
    assert shapes["pfasrdz"] == shapes["pfat0pdz"] == shapes["pfagtdz"]
    assert R.trunk_checkpoint("pfasrbank", 6000).is_file()               # any network with a verified formal training
    with pytest.raises(FileNotFoundError):
        R.trunk_checkpoint("pfakdbank", 6000)                            # never trained: no receipt
    with pytest.raises(RuntimeError):
        R.trunk_checkpoint("pfagtdz", 4000)


def test_swap_receipts_rederive():
    """Every swap receipt on disk is what the resolver derives today (checkpoint SHA, config SHA, trunk, routes, router)."""
    entry = {"evaluate2": "evaluate", "readouts2": "readouts", "evaluate_fast": "row"}
    files = sorted((P.OUT / "swap_receipts").glob("*.json"))
    assert len(files) >= 394
    for path in files:
        rec = json.loads(path.read_text())
        what = path.stem[len(rec["net"]) + 1 + len(rec["entry"]) + 1:]
        expected = {k: v for k, v in rec.items() if k != "completed_at_utc"}
        assert RT.swap_receipt(rec["net"], entry[rec["entry"]], rec["step"]) == expected, path.name
        assert RT.swap_receipt_path(rec["net"], entry[rec["entry"]], what).name == path.name


def test_cell_net_and_labels():
    assert RT.cell_net("xt0_srdwa") and RT.cell_net("xt0_srdwa_5000") and not RT.cell_net("pfasrdz")
    assert not RT.cell_net("pfagtdz") and not RT.cell_net("s45ref_r14")
    assert RT.ENTRY_LABEL == {"evaluate": "evaluate2", "readouts": "readouts2", "row": "evaluate_fast"}
    assert RT.swap_receipt_path("xt0_srdwa", "row", "row_fast").name == "xt0_srdwa_evaluate_fast_row_fast.json"


def test_router_cost_of_the_ready_kinds():
    RT.register_kinds()
    for name in READY:
        spec = R.router_spec(name)
        cost = S.router_cost({"router": spec["kind"], "checkpoint": str(R.router_ready(name))})
        assert cost["params"] == int(spec["params"]) and cost["macs"] > 0


def test_bank_ready_and_receipts():
    assert R.bank_ready("pfasrdz") == R.bank_dir("pfasrdz")
    with pytest.raises(FileNotFoundError):
        R.bank_ready("pfat0pdz")


def test_router_ready_checks_completion(tmp_path, monkeypatch):
    assert R.router_ready("t0p") == R.router_checkpoint("t0p")
    monkeypatch.setattr(R, "ROUTERS_DIR", tmp_path)
    (tmp_path / "sr1").mkdir()
    (tmp_path / "sr1" / "sr1_final.ckpt").write_bytes(b"x")
    (tmp_path / "sr1" / "sr1_final.json").write_text(json.dumps({"status": "COMPLETE", "phase": "formal", "checkpoint_sha256": "0" * 64}))
    with pytest.raises(RuntimeError, match="not complete"):
        R.router_ready("sr1")
