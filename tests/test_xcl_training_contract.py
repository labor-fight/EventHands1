"""CPU checks for strict C1 recipe gates and real shared-initialization evidence.

Run with PYTHONHASHSEED=3407, as required by the reused native C1 wrapper.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model"))


@pytest.fixture
def training():
    from tools.xcl import train
    return train


def recipe(training, cell, phase="formal"):
    cfg = training.B.load_config(training.CANONICAL)
    assoc, gnn, geom, _ = training.CELLS[cell]
    cfg["MODEL"].update(XAG_ASSOC=assoc, XAG_GNN=gnn, XYZ_GEOM=geom)
    cfg["DATA"]["XAG_SOURCE_ROOT"] = str(ROOT / "outputs/xag/sources")
    cfg["XCL"] = {"ENABLED": True, "CELL": cell}
    name = f"xcl_{cell}_s3407"
    base = training.OUT / "runs" / name
    if phase == "preflight":
        name += "_preflight50"
        base = training.OUT / "preflight" / "runs" / name
        cfg["TRAIN"].update(MAX_STEPS=50, VAL_CHECK_INTERVAL=25)
    cfg["TRAIN"].update(RUN_NAME=name, OUTPUT_DIR=str(base))
    cfg["EVAL"]["OUTPUT_DIR"] = str(base / "eval")
    return cfg


@pytest.mark.parametrize("cell", ["a2f0", "a3f0", "c0", "a0fod", "a0foo"])
@pytest.mark.parametrize("phase", ["formal", "preflight"])
def test_registered_recipe_accepted_without_mutation(training, cell, phase):
    cfg = recipe(training, cell, phase)
    before = copy.deepcopy(cfg)
    training.validate_recipe(cfg, cell, phase)
    assert cfg == before


@pytest.mark.parametrize("section,key,value", [
    ("TRAIN", "NUM_WORKERS", 0), ("TRAIN", "PREFETCH_FACTOR", 2),
    ("TRAIN", "BATCH_SIZE_PER_GPU", 256), ("TRAIN", "MAX_STEPS", 1000),
    ("TRAIN", "PRECISION", "32"), ("TRAIN", "LR", .001),
    ("TRAIN", "VAL_CHECK_INTERVAL", 1000), ("TRAIN", "DEVICES", 1),
    ("MODEL", "XYZ_GEOM", False), ("MODEL", "XAG_GNN", "global_residual"),
    ("MODEL", "XYZ_ASSOC_PREV", "target"), ("DATA", "SPLITS_MANIFEST", "/tmp/other_split.json"),
    ("XCL", "CELL", "c0"),
])
def test_unregistered_contract_changes_rejected(training, section, key, value):
    cfg = recipe(training, "a2f0")
    cfg[section][key] = value
    with pytest.raises(RuntimeError, match="Unregistered"):
        training.validate_recipe(cfg, "a2f0", "formal")


@pytest.mark.parametrize("cell", ["a2f0", "a3f0", "c0", "a0fod", "a0foo"])
def test_entire_shared_model_initialization_matches_real_canonical_evidence(training, cell):
    cfg = recipe(training, cell)
    training.B.pl.seed_everything(3407, workers=True)
    model = training.B.ORIGINAL.MNISTModel(cfg=cfg)
    initial = training.B.normalized(model.state_dict())
    ref_path = ROOT / "outputs/xyz_c1_repro/20261018/runs/xyz_c1_stable_s3407/state_rank0_step000000.json"
    if not ref_path.is_file():
        pytest.skip("external canonical full-training evidence is absent")
    reference = training.read(ref_path)["model"]
    assert len(reference) == 37
    assert all(initial[k] == identity for k, identity in reference.items())
    extra = set(initial) - set(reference)
    assert extra == ({"xyz.xag_oracle_projection.weight", "xyz.xag_oracle_projection.bias"}
                     if cell in ("a0fod", "a0foo") else set())
    if cell in ("a0fod", "a0foo"):
        assert all(name.startswith(("xyz.slot_proj.", "xyz.fc1.", "xyz.fc2.", "xyz.xag_oracle_projection."))
                   for name, parameter in model.named_parameters() if parameter.requires_grad)
    else:
        assert all(parameter.requires_grad for parameter in model.parameters())


def test_formal_output_cannot_reuse_failed_paired_a2f0_directory(training):
    cfg = recipe(training, "a2f0")
    cfg["TRAIN"]["OUTPUT_DIR"] = str(ROOT / "outputs/xag_paired/20261018/runs/xag_paired_a2f0_s3407")
    with pytest.raises(RuntimeError, match="Unregistered"):
        training.validate_recipe(cfg, "a2f0", "formal")


def test_depth_readiness_requires_all_74_published_complete_sidecars(tmp_path, monkeypatch):
    from tools.xcl import execute
    names = [f"seq{i:02}" for i in range(74)]
    monkeypatch.setattr(execute, "ROOT", tmp_path)
    monkeypatch.setattr(execute, "read", lambda _: {"sequences": names})
    base = tmp_path / "outputs/xag/sources"
    for name in names[:-1]:
        directory = base / name
        directory.mkdir(parents=True)
        (directory / "depth_metadata.json").write_text(json.dumps({"complete": True}))
    assert not execute.depth_ready(), "73 published sequences cannot authorize A3"
    last = base / names[-1]
    last.mkdir()
    (last / "depth.npy").write_bytes(b"partial")
    assert not execute.depth_ready(), "an allocated partial array is not completion"
    metadata = last / "depth_metadata.json"
    metadata.write_text(json.dumps({"complete": False}))
    assert not execute.depth_ready()
    metadata.write_text(json.dumps({"complete": True}))
    assert execute.depth_ready()


@pytest.mark.parametrize("existing", ["directory", "dangling_symlink"])
def test_fresh_claim_preserves_existing_output(training, tmp_path, monkeypatch, existing):
    for name in ("XCL_CLAIM_NONCE", "XCL_CLAIM_PARENT", "LOCAL_RANK"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name, raising=False)
    output = tmp_path / "run"
    if existing == "directory":
        output.mkdir()
        (output / "keep").write_text("prior failure")
    else:
        output.symlink_to(tmp_path / "absent", target_is_directory=True)
    config, protocol = tmp_path / "cfg.yaml", tmp_path / "protocol.json"
    config.write_text("test")
    protocol.write_text("{}")
    cfg = {"_config_path": str(config), "TRAIN": {"OUTPUT_DIR": str(output)}}
    options = SimpleNamespace(cell="a2f0", phase="preflight", protocol=str(protocol))
    with pytest.raises(RuntimeError, match="Fresh-only output already exists"):
        training.fresh_claim(options, cfg)
    if existing == "directory":
        assert (output / "keep").read_text() == "prior failure"
    else:
        assert output.is_symlink()


def test_native_child_claim_cannot_be_reused_by_an_ordinary_second_launch(training, tmp_path, monkeypatch):
    for name in ("XCL_CLAIM_NONCE", "XCL_CLAIM_PARENT", "LOCAL_RANK"):
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name, raising=False)
    config, protocol = tmp_path / "cfg.yaml", tmp_path / "protocol.json"
    config.write_text("test")
    protocol.write_text("{}")
    output = tmp_path / "run"
    cfg = {"_config_path": str(config), "TRAIN": {"OUTPUT_DIR": str(output)}}
    options = SimpleNamespace(cell="a2f0", phase="preflight", protocol=str(protocol))
    training.fresh_claim(options, cfg)
    before = (output / ".fresh_claim.json").read_bytes()
    with pytest.raises(RuntimeError, match="direct native DDP child"):
        training.fresh_claim(options, cfg)
    assert (output / ".fresh_claim.json").read_bytes() == before
