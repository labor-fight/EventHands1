"""S45-AT (docs/EXPERIMENT_RECORDS.md#s45at-prereg-20261031): one change per arm, the training-split hook and the w200
window patch; CPU only."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
from tools.s45at import common as C  # noqa: E402


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out


@pytest.mark.parametrize("arm", tuple(C.ARMS))
def test_one_change(arm):
    base = yaml.safe_load(C.base_config(arm).read_text())
    for phase in ("preflight", "formal"):
        cfg = flat(C.expected_raw(arm, phase, base))
        ref = flat({k: v for k, v in base.items() if k not in ("C1SF", "S45TR")})
        changed = {k for k in set(cfg) | set(ref) if cfg.get(k) != ref.get(k)}
        bookkeeping = {"TRAIN.RUN_NAME", "TRAIN.OUTPUT_DIR", "EVAL.OUTPUT_DIR"} | (
            {"TRAIN.MAX_STEPS", "TRAIN.VAL_CHECK_INTERVAL"} if phase == "preflight" else set())
        study = {k for k in changed if k.startswith("S45AT.")}
        assert changed - bookkeeping - study == set(C.ARMS[arm]["change"]), arm
        assert cfg["S45AT.ARM"] == arm and cfg["S45AT.EXCLUDE_SUBJECT"] == C.ARMS[arm].get("exclude_subject")
        assert cfg["SEED"] == 3407


def test_training_split_hook(monkeypatch):
    import semkine.dataset as D
    fake = [("ylf_local", "d"), ("ylf_global_v2", "d"), ("ch_local", "d"), ("zgz_local", "d")]
    monkeypatch.setattr(D, "sequences_for_split", lambda root, split, manifest=None: list(fake))
    assert C.install_train_hook({"S45AT": {"EXCLUDE_SUBJECT": None}}) is None
    assert C.install_train_hook({"S45AT": {"EXCLUDE_SUBJECT": "ylf"}}) == "ylf"
    assert [s for s, _ in D.sequences_for_split("r", "train")] == ["ch_local", "zgz_local"]
    assert len(D.sequences_for_split("r", "val_core")) == 4


def test_w200_window_patch(monkeypatch):
    from semkine import eval_track as ET
    calls = []
    runs = np.array([[1000, 1400], [2000, 2300]])
    monkeypatch.setattr(ET, "load_sequence", lambda root, legacy, seq: (None, None, {"valid_runs_ms": runs}, None))
    monkeypatch.setattr(ET, "_window_events", lambda events, offsets, tsub, end, w: calls.append(("ev", end, w)) or w)
    monkeypatch.setattr(ET, "make_eval_packet", lambda ev5, prev, b, k, window_ms, device: calls.append(("pk", window_ms)) or window_ms)
    assert C.install_window_patch({"EVAL": {"WINDOW_MS": 50}}) is None
    assert C.install_window_patch({"EVAL": {"WINDOW_MS": 200}}) == 200
    ET.load_sequence("r", "d", "s")
    for end, expect in ((1049, 50), (1149, 150), (1199, 200), (1399, 200), (2049, 50), (2249, 200)):
        assert ET._window_events(None, None, None, end, 50) == expect
        assert ET.make_eval_packet(None, None, None, None, 50, "cpu") == expect
    assert ET._window_events(None, None, None, 1399, 30) == 30          # explicit non-50 windows pass through
