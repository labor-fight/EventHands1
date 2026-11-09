"""PMR engineering amendment 7, CPU (docs/EXPERIMENT_RECORDS.md#pmr-amend7-20261109). Run with CUDA_VISIBLE_DEVICES="".

- Driver 4 defers a ready cost reading: it never stops the dispatch of trainings and readings, starts on GPU 4 only when no
  job runs and none is ready, and nothing else starts while it runs.
- tools/pmr/migrate3.py: driver 3's pending jobs move unchanged with `d4:wait:` jobs for the running ones.
"""
from __future__ import annotations

import os
from pathlib import Path
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.environ.setdefault("PMR_DRIVER_DIR", "driver4")

from tools.pmr import chain4 as C4  # noqa: E402
from tools.pmr import migrate3 as M3  # noqa: E402

C, C2 = C4.C, C4.C2


@pytest.fixture
def driver(tmp_path, monkeypatch):
    d = tmp_path / "driver4"
    for name, value in {"D2": d, "QUEUE": d / "queue.json", "LOCK": d / "queue.lock", "STATUS": d / "status.json",
                        "LOGS": d / "logs", "STOP": d / "STOP", "DRAIN": d / "DRAIN", "POLICY": d / "policy.json"}.items():
        monkeypatch.setattr(C, name, value)
    (d / "logs").mkdir(parents=True)
    monkeypatch.setattr(C2, "PEERS", [])
    monkeypatch.setattr(C2, "probe", lambda: {})
    monkeypatch.setattr(C, "gpu_used_gib", lambda g: 0.0)
    monkeypatch.setattr(C, "mem_available_gib", lambda: 400.0)

    class Proc:
        def __init__(self, cmd, **kw):
            self.pid, self.returncode = os.getpgrp(), None

        def poll(self):
            return None
    monkeypatch.setattr(C.subprocess, "Popen", Proc)
    return d


def job(key, kind, deps=()):
    return {"key": key, "kind": kind, "command": ["python", "x.py"], "deps": list(deps), "note": "",
            "mem_gib": C.DEFAULT_MEM[kind]}


def test_cost_reading_waits_without_blocking(driver):
    C.add([job("cost", "exclusive"), job("t1", "pair"), job("t2", "pair")])
    ch = C4.Chain()
    ch.schedule(C.load_queue())
    st = ch.state["jobs"]
    assert st["t1"]["status"] == "RUNNING" and st["cost"]["status"] == "PENDING"
    ch.last_pair_start = time.monotonic() - C.MIN_PAIR_GAP_S - 1
    ch.schedule(C.load_queue())
    assert st["t2"]["status"] == "RUNNING" and st["cost"]["status"] == "PENDING"     # driver 3 would have held t2
    for k in ("t1", "t2"):
        ch.children.pop(k)
        st[k]["status"] = "EXIT_0"
    ch.schedule(C.load_queue())
    assert st["cost"]["status"] == "RUNNING" and st["cost"]["gpus"] == [C.EXCLUSIVE_GPU]


def test_nothing_starts_beside_a_running_cost_reading(driver):
    C.add([job("cost", "exclusive"), job("e1", "gpu", ["cost"]), job("e2", "gpu")])
    ch = C4.Chain()
    ch.state["jobs"]["e2"] = {"status": "EXIT_0", "kind": "gpu", "deps": []}
    ch.schedule(C.load_queue())
    assert ch.state["jobs"]["cost"]["status"] == "RUNNING"
    C.add([job("e3", "gpu"), job("t3", "pair")])
    ch.schedule(C.load_queue())
    assert ch.state["jobs"]["e3"]["status"] == "PENDING" and ch.state["jobs"]["t3"]["status"] == "PENDING"


def test_migration_waits_use_the_driver4_prefix():
    q = {"jobs": [job("a", "pair"), job("b", "cpu", ["a"])]}
    s = {"jobs": {"a": {"status": "RUNNING"}}}
    waits, moved = M3.plan(q, s, "driver3")
    assert [w["key"] for w in waits] == ["d4:wait:a"] and moved[0]["deps"] == ["d4:wait:a"]
    assert waits[0]["command"][-4:] == ["--driver", "driver3", "--key", "a"]
