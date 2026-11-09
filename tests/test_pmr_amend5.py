"""PMR engineering amendment 5, CPU (docs/EXPERIMENT_RECORDS.md#pmr-amend5-20261109). Run with CUDA_VISIBLE_DEVICES="".

- A stopped wrapper that has exited but is not reaped (a zombie) no longer reads as alive: driver 3 reaps it, re-homes the
  job, and its emergency cleanup returns at once.
- Driver 3 prefers direct pairs and re-homes after 900 s.
- tools/pmr/migrate2.py re-queues pending and stopped jobs unchanged and archives a stopped training's output directory.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.environ.setdefault("PMR_DRIVER_DIR", "driver3")

from tools.pmr import chain3 as C3  # noqa: E402
from tools.pmr import migrate2 as M2  # noqa: E402

C = C3.C


def zombie():
    p = subprocess.Popen(["sleep", "0.05"], start_new_session=True)
    time.sleep(0.5)
    assert C.live_group(p.pid)                                      # exited, not reaped: the group reads as alive
    return p


@pytest.fixture
def driver(tmp_path, monkeypatch):
    d = tmp_path / "driver3"
    for name, value in {"D2": d, "QUEUE": d / "queue.json", "LOCK": d / "queue.lock", "STATUS": d / "status.json",
                        "LOGS": d / "logs", "STOP": d / "STOP", "DRAIN": d / "DRAIN", "POLICY": d / "policy.json"}.items():
        monkeypatch.setattr(C, name, value)
    (d / "logs").mkdir(parents=True)
    return d


def test_stop_group_reaps_a_zombie():
    p = zombie()
    t = time.monotonic()
    C3.stop_group(p)
    assert time.monotonic() - t < 5 and p.returncode is not None and not C.live_group(p.pid)


def test_rehome_of_an_exited_wrapper(driver, monkeypatch):
    calls = []

    class Proc:
        def __init__(self, cmd, **kw):
            self.pid, self.returncode = os.getpgrp(), None
            calls.append((cmd, kw["env"]))
    ch = C3.Chain()
    key = "s4b:pre:d1pos"
    ch.children[key] = zombie()
    ch.state["jobs"][key] = {"status": "RUNNING", "kind": "pair", "gpus": [2, 3], "borrowed": True}
    monkeypatch.setattr(C3.subprocess, "Popen", Proc)
    job = {"key": key, "kind": "pair", "command": ["python", "train.py", "--config", "c.yaml"], "deps": []}
    ch.rehome(job, (4, 5))
    st = ch.state["jobs"][key]
    assert st["gpus"] == [4, 5] and st["rehomed_from"] == [2, 3] and not st["borrowed"]
    assert calls[0][0][:2] == [C3.PY, C3.C2.BORROW_JOB] and calls[0][1]["CUDA_VISIBLE_DEVICES"] == "4,5"


def test_cleanup_returns_at_once_with_zombies(driver):
    ch = C3.Chain()
    ch.children["a"] = zombie()
    t = time.monotonic()
    ch.cleanup()
    assert time.monotonic() - t < 5


def test_order_and_rehome_time():
    assert C.PAIRS == C3.C2.DIRECT + C3.C2.BORROWED and C3.C2.REHOME_S == 900


def test_migration_plan_and_archive(tmp_path):
    def job(key, deps=(), config=None):
        cmd = ["python", "train.py"] + (["--config", str(config)] if config else [])
        return {"key": key, "kind": "pair" if config else "cpu", "command": cmd, "deps": list(deps), "note": "",
                "mem_gib": 4}
    out = tmp_path / "runs" / "pmr_x_s3407"
    out.mkdir(parents=True)
    cfg = tmp_path / "x.yaml"
    cfg.write_text(yaml.safe_dump({"TRAIN": {"OUTPUT_DIR": str(out)}}))
    q = {"jobs": [job("done"), job("t", ["done"], cfg), job("acc", ["t"]), job("p", ["acc", "done"])]}
    s = {"jobs": {"done": {"status": "EXIT_0"}, "t": {"status": "RUNNING"}, "acc": {"status": "PENDING"}}}
    moved, redo = M2.plan(q, s)
    assert redo == ["t"] and {m["key"]: m["deps"] for m in moved} == {"t": [], "acc": ["t"], "p": ["acc"]}
    dest = M2.archive(q["jobs"][1], tmp_path / "formal_failed" / "driver2_stop")
    assert dest and not out.exists() and Path(dest).name == "pmr_x_s3407"
    assert M2.archive(q["jobs"][2], tmp_path / "ff") is None
    with pytest.raises(RuntimeError):
        M2.plan({"jobs": [job("z", ["f"])]}, {"jobs": {"f": {"status": "FAILED_PRESERVED"}}})
