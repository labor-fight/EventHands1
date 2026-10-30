"""Reject duplicate scheduling even when an active child's parent has exited."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def environment():
    value = dict(os.environ)
    value.pop("XAG_SCHEDULER_LOCK_FDS", None)
    return value


def contender(path):
    return subprocess.run([sys.executable, "-c",
        "import sys; from tools.xag.process_lock import acquire; acquire(sys.argv[1])", str(path)],
        cwd=ROOT, env=environment(), capture_output=True, text=True)


def test_child_keeps_lock_after_owner_exit(tmp_path):
    path = tmp_path / "scheduler.lock"
    script = """
import subprocess, sys
from tools.xag.process_lock import acquire, inherited_fds
acquire(sys.argv[1])
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'],
    pass_fds=inherited_fds(), start_new_session=True,
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(child.pid, flush=True)
"""
    owner = subprocess.run([sys.executable, "-c", script, str(path)], cwd=ROOT,
                           env=environment(), capture_output=True, text=True, check=True)
    pid = int(owner.stdout.strip())
    try:
        blocked = contender(path)
        assert blocked.returncode != 0
        assert "already active" in blocked.stderr
        assert f'"pid": {pid}' not in path.read_text()
    finally:
        os.kill(pid, signal.SIGTERM)
    deadline = time.monotonic() + 5
    while True:
        released = contender(path)
        if released.returncode == 0:
            break
        assert time.monotonic() < deadline, released.stderr
        time.sleep(.05)


def test_invalid_inherited_descriptor_fails_closed(tmp_path):
    env = environment()
    env["XAG_SCHEDULER_LOCK_FDS"] = "999999"
    result = subprocess.run([sys.executable, "-c",
        "from tools.xag.process_lock import inherited_fds; inherited_fds()"],
        cwd=ROOT, env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Bad file descriptor" in result.stderr


def test_rank_without_lock_fd_is_detected_by_study_tag(tmp_path):
    env = environment()
    env.update(XAG_STUDY_ROOT=str(tmp_path.resolve()), XAG_ACTIVE_JOB="simulated_rank1")
    rank = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                            env=env, close_fds=True, start_new_session=True)
    try:
        result = subprocess.run([sys.executable, "-c",
            "import sys; from tools.xag.process_lock import assert_no_orphan_jobs; assert_no_orphan_jobs(sys.argv[1])",
            str(tmp_path)], cwd=ROOT, env=environment(), capture_output=True, text=True)
        assert result.returncode != 0
        assert "simulated_rank1" in result.stderr
        assert str(rank.pid) in result.stderr
    finally:
        rank.terminate()
        rank.wait(timeout=5)
