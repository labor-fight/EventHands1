"""PMR driver 2 on borrowed GPUs, CPU (docs/EXPERIMENT_RECORDS.md#pmr-amend4-20261109). Run with CUDA_VISIBLE_DEVICES="".

- tools/pmr/borrow_job.py records every start and archives a stopped attempt's run (never a run present at the first start).
- tools/pmr/chain2.py: trainings go to an idle borrowed pair first, wrapped in `gpu-borrow run`, and never to a borrowed
  pair that lmx or a peer driver holds; evaluations only use GPUs 4-7; a direct pair is kept for a training that has no
  borrowed pair; a borrowed job whose command has not run for REHOME_S moves to a free direct pair.
- tools/pmr/migrate.py moves pending jobs unchanged, mapping dependencies on running jobs to wait jobs.
- Every training still pending in driver 1 names a config with TRAIN.OUTPUT_DIR (what borrow_job.py archives).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.environ.setdefault("PMR_DRIVER_DIR", "driver2")

from tools.pmr import borrow_job as B  # noqa: E402
from tools.pmr import chain2 as C2  # noqa: E402
from tools.pmr import migrate as M  # noqa: E402

C = C2.C


@pytest.fixture
def driver(tmp_path, monkeypatch):
    d = tmp_path / "driver2"
    for name, value in {"D2": d, "QUEUE": d / "queue.json", "LOCK": d / "queue.lock", "STATUS": d / "status.json",
                        "LOGS": d / "logs", "STOP": d / "STOP", "DRAIN": d / "DRAIN", "POLICY": d / "policy.json"}.items():
        monkeypatch.setattr(C, name, value)
    (d / "logs").mkdir(parents=True)
    monkeypatch.setattr(C2, "PEERS", [tmp_path / "driver1"])
    monkeypatch.setattr(C, "gpu_used_gib", lambda g: 0.0)
    monkeypatch.setattr(C, "mem_available_gib", lambda: 400.0)
    calls = []

    class Proc:
        def __init__(self, cmd, cwd=None, env=None, stdout=None, stderr=None, start_new_session=False):
            self.pid, self.returncode = os.getpgrp(), None
            calls.append({"cmd": cmd, "env": env})

        def poll(self):
            return None

        def wait(self):
            return 0
    monkeypatch.setattr(C.subprocess, "Popen", Proc)
    monkeypatch.setattr(C2.subprocess, "Popen", Proc)
    return d, calls


def job(key, kind="pair", deps=()):
    return {"key": key, "kind": kind, "command": ["python", "train.py", "--config", "c.yaml"], "deps": list(deps),
            "note": "", "mem_gib": C.DEFAULT_MEM[kind]}


def test_borrow_job_archives_only_its_own_stopped_attempt(tmp_path, monkeypatch):
    out = tmp_path / "runs" / "pmr_x_s3407"
    cfg = tmp_path / "x.yaml"
    cfg.write_text(yaml.safe_dump({"TRAIN": {"OUTPUT_DIR": str(out)}}))
    monkeypatch.setattr(B, "PREEMPTED", tmp_path / "preempted")
    rec = tmp_path / "attempts.json"
    cmd = ["python", "train.py", "--config", str(cfg)]
    assert B.begin("k", rec, cmd) == 1
    out.mkdir(parents=True)
    (out / "step=500.ckpt").write_text("partial")
    assert B.begin("k", rec, cmd) == 2
    assert not out.exists() and (tmp_path / "preempted" / "pmr_x_s3407" / "attempt1" / "step=500.ckpt").exists()
    assert B.begin("k", rec, cmd) == 3                                            # killed before creating its run
    r = json.loads(rec.read_text())
    assert [a["archived_previous"] is not None for a in r["attempts"]] == [False, True, False]
    rec2 = tmp_path / "attempts2.json"
    out.mkdir()
    assert B.begin("k2", rec2, cmd) == 1 and out.exists()                         # present at the first start: left alone
    with pytest.raises(RuntimeError):
        B.begin("other", rec, cmd)


def test_trainings_prefer_an_idle_borrowed_pair_and_are_wrapped(driver, monkeypatch):
    d, calls = driver
    monkeypatch.setattr(C2, "probe", lambda: {0: True, 1: True, 2: False, 3: False})
    C.add([job("t1"), job("t2"), job("e1", "gpu"), job("e2", "gpu")])
    ch = C2.Chain()
    ch.schedule(C.load_queue())
    st = ch.state["jobs"]
    assert st["t1"]["gpus"] == [0, 1] and st["t1"]["borrowed"] and st["t2"]["status"] == "PENDING"
    assert st["e1"]["gpus"] == [6] and st["e2"]["gpus"] == [6]        # (4, 5) kept for t2 (no idle borrowed pair left)
    first = calls[0]["cmd"]
    assert first[:4] == [C2.GPU_BORROW, "run", "--gpus", "0,1"] and first[4] == "--state-dir"
    assert Path(first[5]) == d / "borrow" / "t1" and oct(Path(first[5]).stat().st_mode & 0o777) == "0o700"
    i = first.index("--", 6)
    assert first[i + 1:i + 3] == [C2.PY, C2.BORROW_JOB] and first[-4:] == job("t1")["command"]
    ch.last_pair_start = time.monotonic() - C.MIN_PAIR_GAP_S - 1
    ch.schedule(C.load_queue())
    assert st["t2"]["gpus"] == [4, 5] and not st["t2"].get("borrowed") and calls[-1]["cmd"] == job("t2")["command"]
    assert calls[-1]["env"]["CUDA_VISIBLE_DEVICES"] == "4,5"


def test_peers_and_lmx_hold_their_gpus(driver, tmp_path, monkeypatch):
    d, calls = driver
    monkeypatch.setattr(C2, "probe", lambda: {0: False, 1: True, 2: True, 3: True})       # lmx on GPU 0
    peer = tmp_path / "driver1"
    peer.mkdir()
    (peer / "status.json").write_text(json.dumps({"status": "RUNNING", "jobs": {
        "p": {"status": "RUNNING", "kind": "pair", "pid": os.getpgrp(), "gpus": [2, 3]},
        "q": {"status": "RUNNING", "kind": "pair", "pid": os.getpgrp(), "gpus": [4, 5]}}}))
    C.add([job("t1"), job("e1", "gpu")])
    ch = C2.Chain()
    ch.schedule(C.load_queue())
    st = ch.state["jobs"]
    assert st["t1"]["gpus"] == [6, 7] and st["e1"]["status"] == "PENDING"
    assert all(c["cmd"][0] != C2.GPU_BORROW for c in calls)


def test_evaluations_never_use_borrowed_gpus(driver, monkeypatch):
    d, calls = driver
    monkeypatch.setattr(C2, "probe", lambda: {g: True for g in range(4)})
    C.add([job(f"e{i}", "gpu") for i in range(10)])
    ch = C2.Chain()
    ch.schedule(C.load_queue())
    used = [v["gpus"][0] for v in ch.state["jobs"].values() if v["status"] == "RUNNING"]
    assert sorted(used) == [4, 4, 5, 5, 6, 6, 7, 7]


def test_rehome_after_a_long_wait(driver, monkeypatch):
    d, calls = driver
    monkeypatch.setattr(C2, "probe", lambda: {0: True, 1: True, 2: False, 3: False})
    killed = []
    monkeypatch.setattr(C2.os, "killpg", lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(C, "live_group", lambda pid: False)
    C.add([job("t1")])
    ch = C2.Chain()
    ch.schedule(C.load_queue())
    assert ch.state["jobs"]["t1"]["borrowed"]
    run = d / "borrow" / "t1" / "runs" / "x"
    run.mkdir(parents=True)
    (run / "state.json").write_text(json.dumps({"phase": "RUNNING"}))
    ch.waiting["t1"] = time.monotonic() - C2.REHOME_S - 1
    ch.last_pair_start = 0.0
    ch.schedule(C.load_queue())
    assert ch.state["jobs"]["t1"]["borrowed"] and not killed                     # running: never moved
    (run / "state.json").write_text(json.dumps({"phase": "WAITING"}))
    ch.schedule(C.load_queue())
    assert ch.state["jobs"]["t1"]["borrowed"] and not killed                     # the wait restarts after running
    ch.waiting["t1"] = time.monotonic() - C2.REHOME_S - 1
    ch.schedule(C.load_queue())
    st = ch.state["jobs"]["t1"]
    assert killed and not st["borrowed"] and st["gpus"] == [4, 5] and st["rehomed_from"] == [0, 1]
    assert calls[-1]["cmd"][:2] == [C2.PY, C2.BORROW_JOB] and calls[-1]["env"]["CUDA_VISIBLE_DEVICES"] == "4,5"


def test_migration_plan():
    q = {"jobs": [job("a"), job("b", "cpu", ["a"]), job("c"), job("d", "cpu", ["c", "x"]), job("e", "gpu", ["d", "b"])]}
    s = {"jobs": {"a": {"status": "RUNNING"}, "c": {"status": "PENDING"}, "x": {"status": "EXIT_0"},
                  "b": {"status": "PENDING"}}}
    waits, moved = M.plan(q, s, "driver1")
    assert [w["key"] for w in waits] == ["d2:wait:a"] and waits[0]["command"][-4:] == ["--driver", "driver1", "--key", "a"]
    assert {m["key"]: m["deps"] for m in moved} == {"b": ["d2:wait:a"], "c": [], "d": ["c"], "e": ["d", "b"]}
    with pytest.raises(RuntimeError):
        M.plan({"jobs": [job("f", "cpu", ["z"])]}, {"jobs": {"z": {"status": "FAILED_PRESERVED"}}}, "driver1")


def test_pending_trainings_name_an_output_dir():
    q = json.loads((ROOT / "outputs/pmr/20261109/driver1/queue.json").read_text())
    for j in q["jobs"]:
        if j["kind"] == "pair":
            assert str(B.output_dir(j["command"])).startswith(str(ROOT / "outputs/pmr/20261109/runs/"))
