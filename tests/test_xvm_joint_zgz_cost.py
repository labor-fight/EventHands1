"""Synthetic staging checks for the cost-only training fixture."""
import json

import numpy as np
import pytest
import torch

from tools.xvm_joint.zgz_cost import prepare_latency_cases


@pytest.mark.parametrize("mode,expected_windows", [("w50", [50]*4), ("s39", [50, 100, 150, 200])])
def test_cost_fixture_is_training_only_beta_zero_and_growing_clip(tmp_path, monkeypatch, mode, expected_windows):
    from semkine import eval_track as ET
    manifest = tmp_path/"splits_semkine.json"
    manifest.write_text(json.dumps({"train": {"trials": ["lyq_local"]}}))
    class Aux(dict):
        def __getitem__(self, key):
            if key == "betas":raise AssertionError("GT beta must not be accessed")
            return super().__getitem__(key)
    aux = Aux(camera_K=np.eye(3, dtype=np.float32), valid_runs_ms=np.array([[100, 301]]))
    events = np.zeros((400, 3), np.int32)
    offsets = np.arange(401, dtype=np.int64)
    poses = np.zeros((400, 51), np.float32)
    poses[:, 2] = np.arange(400)*1e-6+.6
    calls = []
    def load_sequence(root, directory, recording):
        calls.append((directory, recording))
        return events, offsets, aux, poses
    monkeypatch.setattr(ET, "load_sequence", load_sequence)
    protocol = {"data_root": str(tmp_path), "split": {"path": str(manifest)},
                "resources": {"timing_recording": "lyq_local", "max_packets": 4},
                "modes": {"w50": {"window_ms": 50, "clip_run_start": False},
                          "s39": {"window_ms": 200, "clip_run_start": True}}}
    rows, fixture = prepare_latency_cases(protocol, mode, "cpu")
    assert calls == [("train", "lyq_local")]
    assert [int(row.batch.t_end_us.item()/1000) for row in rows] == expected_windows
    assert [len(row.batch.events) for row in rows] == expected_windows
    assert [row.reset for row in rows] == [True, False, False, False]
    assert all(row.run_start_ms == 100 for row in rows)
    assert all(torch.count_nonzero(row.batch.betas) == 0 for row in rows)
    assert all(torch.count_nonzero(row.batch.target) == 0 for row in rows)
    assert all(torch.equal(row.batch.prev_state, torch.from_numpy(poses[100:101])) for row in rows)
    assert fixture["initialization"] == "original GT run-start pose, no noise, timing only"
    assert fixture["packets"] == 4
    assert "no zgz" in fixture["source_data_scope"]
