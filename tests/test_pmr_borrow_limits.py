"""CPU verification of borrowed PMR descriptor limits and mixed GPU leases."""
from __future__ import annotations

import json
import os
from pathlib import Path
import resource
import subprocess
import sys
from unittest.mock import Mock

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.gpu_borrow.monitor import GPU, Process  # noqa: E402
from tools.gpu_borrow.supervisor import BorrowRunner, select_gpus  # noqa: E402
from tools.pmr import borrow_job, borrow_job2  # noqa: E402


def cpu_subprocess(code):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                          check=True, capture_output=True, text=True, timeout=20)


def test_real_process_raises_soft_1024_and_preserves_hard():
    result = cpu_subprocess(
        "import json,resource; from tools.pmr.borrow_job2 import raise_nofile_limit; "
        "soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE); "
        "resource.setrlimit(resource.RLIMIT_NOFILE,(1024,hard)); "
        "print(json.dumps({'hard_before':hard,'after':raise_nofile_limit()}))"
    )
    row = json.loads(result.stdout)
    soft, hard = row["after"]
    assert hard == row["hard_before"]
    assert soft == (65536 if hard == resource.RLIM_INFINITY else min(65536, hard))


@pytest.mark.parametrize("limits, expected", [
    ((1024, 524288), (65536, 524288)),
    ((100000, 524288), (100000, 524288)),
    ((1024, 16384), (16384, 16384)),
    ((1024, resource.RLIM_INFINITY), (65536, resource.RLIM_INFINITY)),
    ((resource.RLIM_INFINITY, resource.RLIM_INFINITY),
     (resource.RLIM_INFINITY, resource.RLIM_INFINITY)),
])
def test_limit_bounds_never_lower_existing_limits(monkeypatch, limits, expected):
    current = [limits]

    def set_limit(which, value):
        assert which == resource.RLIMIT_NOFILE
        current[0] = value

    setter = Mock(side_effect=set_limit)
    monkeypatch.setattr(borrow_job2.resource, "getrlimit", lambda which: current[0])
    monkeypatch.setattr(borrow_job2.resource, "setrlimit", setter)
    assert borrow_job2.raise_nofile_limit() == expected
    assert current[0][1] == limits[1]
    assert setter.call_count == int(limits != expected)


def test_failed_or_ineffective_limit_raise_has_clear_error(monkeypatch):
    monkeypatch.setattr(borrow_job2.resource, "getrlimit", lambda which: (1024, 524288))
    monkeypatch.setattr(borrow_job2.resource, "setrlimit", Mock(side_effect=PermissionError("denied")))
    with pytest.raises(RuntimeError, match="cannot raise RLIMIT_NOFILE.*1024.*65536"):
        borrow_job2.raise_nofile_limit()
    monkeypatch.setattr(borrow_job2.resource, "setrlimit", Mock())
    with pytest.raises(RuntimeError, match="insufficient RLIMIT_NOFILE"):
        borrow_job2.raise_nofile_limit()


def test_main_forwards_original_arguments_after_preparing_limits(monkeypatch):
    events = []
    args = ["--key", "s9x:rol5", "--", "python", "train.py", "literal $HOME", "two words"]
    monkeypatch.setattr(borrow_job2, "raise_nofile_limit", lambda: events.append("limits"))

    def original_main(argv):
        assert argv is args
        events.append("original")
        return 17

    monkeypatch.setattr(borrow_job, "main", original_main)
    assert borrow_job2.main(args) == 17
    assert events == ["limits", "original"]


def test_exec_and_preemption_still_use_original_attempt_archive(tmp_path):
    out = tmp_path / "runs" / "pmr_test"
    out.mkdir(parents=True)
    (out / "present_before_first_start.txt").write_text("keep on first start")
    cfg = tmp_path / "train.yaml"
    cfg.write_text(yaml.safe_dump({"TRAIN": {"OUTPUT_DIR": str(out)}}))
    attempts = tmp_path / "attempts.json"
    logs = tmp_path / "logs"
    logs.mkdir()
    archive = tmp_path / "preempted"
    training = (
        "import json,pathlib,resource; "
        f"out=pathlib.Path({str(out)!r}); out.mkdir(parents=True,exist_ok=True); "
        "(out/'limits.json').write_text(json.dumps(resource.getrlimit(resource.RLIMIT_NOFILE)))"
    )
    args = ["--key", "test:rol5", "--attempts", str(attempts), "--log-dir", str(logs),
            "--", sys.executable, "-c", training, "--config", str(cfg)]
    code = (
        "from pathlib import Path; from tools.pmr import borrow_job,borrow_job2; "
        f"borrow_job.PREEMPTED=Path({str(archive)!r}); borrow_job2.main({args!r})"
    )
    cpu_subprocess(code)
    assert (out / "present_before_first_start.txt").read_text() == "keep on first start"
    first_limits = json.loads((out / "limits.json").read_text())
    assert first_limits[0] >= min(65536, first_limits[1])
    cpu_subprocess(code)
    prior = archive / out.name / "attempt1"
    assert (prior / "present_before_first_start.txt").exists()
    assert not (out / "present_before_first_start.txt").exists()
    assert (out / "limits.json").exists()
    records = json.loads(attempts.read_text())["attempts"]
    assert len(records) == 2
    assert records[0]["archived_previous"] is None
    assert records[1]["archived_previous"] == str(prior)
    assert {p.name for p in logs.iterdir()} == {"test_rol5.attempt1.log", "test_rol5.attempt2.log"}


def test_mixed_pair_waits_600_seconds_and_yields_whole_job(tmp_path):
    teacher = os.getuid() + 10000
    cards = {index: GPU(index, f"GPU-test-{index}", 14, 0, ()) for index in (2, 5)}
    monitor = Mock()
    monitor.snapshot.side_effect = lambda uuids=None: cards
    now = [0.0]
    runner = BorrowRunner(
        monitor=monitor, gpus=select_gpus(cards, "2,5"), priority_uid=teacher,
        command=["/usr/bin/true"], resume_command=["/usr/bin/true"], cwd=tmp_path,
        state_dir=tmp_path, run_dir=tmp_path / "run", lock_root=tmp_path,
        clock=lambda: now[0],
    )
    starts, stops = [], []

    def start():
        starts.append(tuple(g.index for g in runner.gpus))
        runner.child = object()
        runner.phase = "RUNNING"

    def stop(reason, **kwargs):
        stops.append(reason)
        runner.child = None
        runner.locks.release()
        runner.idle_since.clear()

    runner.start_child = start
    runner.stop_child = stop
    runner.peek_exit = lambda: None
    runner.owns = lambda process: False
    try:
        runner.tick()
        now[0] = 599.9
        runner.tick()
        assert not starts
        now[0] = 600
        runner.tick()
        assert starts == [(2, 5)]
        assert set(runner.locks.uuids) == {"GPU-test-2", "GPU-test-5"}
        cards[2] = GPU(2, "GPU-test-2", 14, 0, (Process(123, teacher),))
        runner.tick()
        assert runner.child is None and runner.phase == "WAITING"
        assert stops and not runner.locks.files
    finally:
        runner.close()
