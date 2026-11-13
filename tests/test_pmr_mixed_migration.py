"""The mixed-pair migration changes scheduling without changing experiments."""
import copy

import pytest

from tools.pmr import jobs18 as J


def source():
    return {"jobs": [
        {"key": "freeze:old", "kind": "cpu", "deps": [], "command": ["old-freeze"]},
        {"key": "s9x:train:d1hubfull:s3407", "kind": "pair", "deps": ["freeze:old"], "command": ["train"]},
        {"key": "s9x:rtrain:d1hubfull_rol5:s3407", "kind": "pair", "deps": ["s9x:train:d1hubfull:s3407"],
         "command": ["rol5"], "pairs": [[5, 7]]},
        {"key": "restricted", "kind": "pair", "deps": [], "command": ["other"], "pairs": [[0, 1]]},
    ]}, {"status": "DRAINED", "jobs": {
        "freeze:old": {"status": "EXIT_0"},
        "s9x:train:d1hubfull:s3407": {"status": "PENDING"},
        "s9x:rtrain:d1hubfull_rol5:s3407": {"status": "PENDING"},
        "restricted": {"status": "PENDING"},
    }}


def test_migration_preserves_commands_and_records_only_rol5_pair_change():
    queue, status = source()
    before = copy.deepcopy(queue)
    jobs, changes = J.plan(queue, status)
    assert queue == before
    assert jobs[0]["key"] == J.FREEZE_KEY
    assert jobs[1]["deps"] == [J.FREEZE_KEY]
    assert jobs[2]["deps"] == [J.FREEZE_KEY, "s9x:train:d1hubfull:s3407"]
    assert "pairs" not in jobs[2]
    assert jobs[3]["pairs"] == [[0, 1]]
    assert [job["command"] for job in jobs[1:]] == [["train"], ["rol5"], ["other"]]
    assert len(changes) == 1 and changes[0]["before"] == [[5, 7]]


def test_migration_rejects_undrained_or_running_source():
    queue, status = source()
    status["status"] = "RUNNING"
    with pytest.raises(RuntimeError, match="finish draining"):
        J.plan(queue, status)
    status["status"] = "DRAINED"
    status["jobs"]["s9x:train:d1hubfull:s3407"]["status"] = "RUNNING"
    with pytest.raises(RuntimeError, match="running jobs"):
        J.plan(queue, status)


def test_migration_rejects_unexpected_rol5_restriction():
    queue, status = source()
    queue["jobs"][2]["pairs"] = [[4, 5]]
    with pytest.raises(RuntimeError, match="unexpected rol5"):
        J.plan(queue, status)
