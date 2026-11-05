"""PFO2 amendment 1 (docs/EXPERIMENT_RECORDS.md#pfo2-amend1-20261103): CPU tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]


def test_lightning_default_resets_and_policy_keeps_deterministic_algorithms():
    import pytorch_lightning as pl
    from tools.pfo2 import train_det as TD
    native = pl.Trainer
    try:
        torch.use_deterministic_algorithms(True)
        pl.Trainer(accelerator="cpu", max_steps=1, logger=False, enable_checkpointing=False)
        assert not torch.are_deterministic_algorithms_enabled()          # the PFO gap
        TD.install_policy()
        torch.use_deterministic_algorithms(True)
        pl.Trainer(accelerator="cpu", max_steps=1, logger=False, enable_checkpointing=False)
        assert torch.are_deterministic_algorithms_enabled()
    finally:
        pl.Trainer = native
        torch.use_deterministic_algorithms(False)


def test_equiv_distance():
    from tools.pfo2.equiv import dist
    a = {"x": torch.ones(4, dtype=torch.float64), "y": torch.zeros(2, dtype=torch.float64)}
    b = {"x": torch.full((4,), 2.0, dtype=torch.float64), "y": torch.zeros(2, dtype=torch.float64)}
    assert dist(a, a) == 0.0
    assert abs(dist(a, b) - 0.5) < 1e-12


def test_amend1_jobs():
    from tools.pfo2 import jobs_a1 as JA
    jobs = JA.amend1()
    keys = {j["key"] for j in jobs}
    for j in jobs:
        assert set(j["deps"]) <= keys, j["key"]
    by = {j["key"]: j for j in jobs}
    assert by["a1:pre:f1mp"]["deps"] == ["a1:equiv:f1rep"]
    assert "tools/pfo2/train.py" in by["a1:train:f1mp:3407"]["command"][3]          # F's procedure
    for k in ("a1:train:fl1:3409", "a1:train:fgs:3407", "a1:pre:rl1", "a1:head:rgs:3408"):
        assert "tools/pfo2/train_det.py" in by[k]["command"][3], k
    assert by["a1:head:rl1:3407"]["pairs"] == [[6, 7]]
    assert set(by["a1:train:f1rep:3407"]["deps"]) == {f"a1:head:{r}:{s}" for r in ("rl1", "rgs") for s in (3407, 3408, 3409)}


def test_deterministic_record_required(tmp_path, monkeypatch):
    from tools.pfo2 import run_a1 as RA
    monkeypatch.setattr(RA.A, "run_dir", lambda arm, seed, phase="formal": tmp_path)
    with pytest.raises(FileNotFoundError):
        RA.deterministic_record("fl1", 3407)
    (tmp_path / "pfo2_det.json").write_text('{"trainer_deterministic": true, "deterministic_algorithms": false, '
                                            '"entry_sha256": "x"}')
    with pytest.raises(RuntimeError):
        RA.deterministic_record("fl1", 3407)
