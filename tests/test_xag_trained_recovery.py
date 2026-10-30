"""Own-state recurrence, exact reference identity and failure denominators."""
import json

import numpy as np
import pytest
import torch

from tools.xag import trained_recovery as R


def test_each_trajectory_reuses_own_previous_prediction_never_common_or_gt():
    initial = np.zeros((3, 51), np.float32)
    initial[:, 0] = [1., 10., 100.]
    seen = []
    def predict(row, incoming, trajectory, step):
        seen.append((trajectory, step, incoming[0]))
        incoming[0] = 2 * incoming[0] + row
        return incoming
    out, failures = R.own_states(initial, [1, 2, 3], predict)
    assert not failures
    np.testing.assert_array_equal(out[:, 0], initial)
    np.testing.assert_array_equal(out[:, :, 0], [[1, 3, 8, 19], [10, 21, 44, 91], [100, 201, 404, 811]])
    assert seen[6:] == [(0, 2, 8), (1, 2, 44), (2, 2, 404)]


def test_failure_retains_every_unexecuted_endpoint_and_other_trajectories():
    called = []
    def predict(row, incoming, trajectory, step):
        called.append((trajectory, step))
        if trajectory == 1 and step == 1:
            return np.full(51, np.nan, np.float32)
        return incoming + 1
    out, failures = R.own_states(np.zeros((2, 51)), [0, 1, 2, 3], predict)
    assert out.shape == (2, 5, 51)
    assert np.isfinite(out[0]).all()
    assert np.isfinite(out[1, :2]).all() and np.isnan(out[1, 2:]).all()
    assert len(failures) == 3
    assert failures[0]["reason"] == "FloatingPointError"
    assert [f["step"] for f in failures] == [1, 2, 3]
    assert (1, 2) not in called and (1, 3) not in called


def test_source_preparation_and_oom_are_infrastructure_not_model_failure():
    def bad_source(row, step):
        raise ValueError("wrong source timebase")
    with pytest.raises(ValueError, match="timebase"):
        R.own_states(np.zeros((1, 51)), [0], lambda *a: None, prepare=bad_source)
    def oom(*args):
        raise torch.cuda.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(torch.cuda.OutOfMemoryError):
        R.own_states(np.zeros((1, 51)), [0], oom)


def test_whole_run_sharding_preserves_canonical_timebase(monkeypatch):
    monkeypatch.setattr(R, "COUNTS", {"zgz_global": 3, "zgz_local": 1})
    rows = [{"run_id": 0, "previous_end_ms": 10, "end_ms": 59},
            {"run_id": 0, "previous_end_ms": 59, "end_ms": 109},
            {"run_id": 2, "previous_end_ms": 300, "end_ms": 349}]
    bank = {"sequences": {
        "zgz_global": {"valid_runs_ms": [[10, 111], [150, 151], [300, 350]], "rows": rows},
        "zgz_local": {"valid_runs_ms": [[30, 80]], "rows": [{"run_id": 0, "previous_end_ms": 30, "end_ms": 79}]}}}
    whole = R.complete_runs(bank)
    assert [(a, b) for a, b, _, _ in whole] == [("zgz_global", 0), ("zgz_global", 2), ("zgz_local", 0)]
    shards = [[entry for i, entry in enumerate(whole) if i % 2 == shard] for shard in range(2)]
    assert sum(len(row[3]) for shard in shards for row in shard) == 4
    rows[1]["previous_end_ms"] = 60
    with pytest.raises(ValueError, match="timebase"):
        R.complete_runs(bank)


def test_reference_uses_exact_fourteen_initials_and_rejects_partial(tmp_path, monkeypatch):
    path = tmp_path / "source.npz"
    specs = [{"id": str(i)} for i in range(14)]
    monkeypatch.setattr(R.X, "rollout_specs", lambda *a: specs)
    monkeypatch.setattr(R.X, "source_hashes", lambda: {"code": "fixed"})
    initial = np.arange(14 * 51, dtype=np.float32).reshape(14, 51)
    baseline = np.stack([initial, initial + 1], 1)
    truth = np.zeros((2, 51), np.float32)
    identity = {"phase": "rollout", "protocol_sha256": "protocol", "limit": None,
                "checkpoint_sha256": "ckpt", "config_sha256": "config", "source_manifest_sha256": "sources",
                "source_hashes": {"code": "fixed"}, "sequence": "zgz_local", "run_id": 1}
    R.X.save_npz(path, {"initial51": initial, "base_own51": baseline, "truth51": truth, "times_ms": np.array([100, 149])},
                 {"identity": identity, "specs": specs, "physical_checks": {"passed": True}})
    p = {"checkpoints": {"3407": {"checkpoint_sha256": "ckpt", "config_sha256": "config"}}}
    args = (path, p, "protocol", {"source_manifest_sha256": "sources"}, 3407, "zgz_local", 1, 100, [{"end_ms": 149}], truth)
    recovered, _, _, _ = R.reference(*args)
    np.testing.assert_array_equal(recovered, initial)
    audit_path = path.with_suffix(".json")
    audit = json.loads(audit_path.read_text()); audit["identity"]["limit"] = 1
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="identity"):
        R.reference(*args)


def test_summary_no_survivor_means_when_any_endpoint_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "source_hashes", lambda: {"code": "fixed"})
    name = "xag_a0f0_s3407"
    path = tmp_path / "trained_recovery" / name / "zgz_local_run0000.npz"
    own = np.zeros((14, 3, 51), np.float32); own[1, 2] = np.nan
    error = np.ones((14, 3), np.float32); error[1, 2] = np.nan
    identity = {"protocol_sha256": "protocol", "run_name": name, "limit": None,
                "source_hashes": {"code": "fixed"}, "phase": "trained_own_recovery",
                "sequence": "zgz_local", "run_id": 0}
    specs = [{"block": str(i)} for i in range(14)]
    monkeypatch.setattr(R.X, "rollout_specs", lambda *a: specs)
    R.X.save_npz(path, {"own51": own, "error_root_deg": error, "initial51": own[:, 0], "times_ms": np.array([0, 49, 99])},
                 {"identity": identity, "expected_endpoints": 28, "failed_endpoints": 1, "specs": specs,
                  "physical_checks": {"passed": True}})
    entries = [("zgz_local", 0, 0, [{"end_ms": 49}, {"end_ms": 99}])]
    result = R.summarize_one(name, tmp_path, {}, "protocol", entries)
    assert result["status"] == "COMPLETE_WITH_FAILURES" and result["failed_endpoints"] == 1
    saved = R.X.json_value(result)
    assert saved["groups"]["zgz_local|1||"]["endpoint_mean"]["root_deg"] is None
    assert saved["groups"]["zgz_local|0||"]["endpoint_mean"]["root_deg"] == 1
