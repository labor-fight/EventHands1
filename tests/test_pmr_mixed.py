"""CPU checks for mixed borrowed/direct PMR training pairs.

Run this file in its own process because PMR driver modules configure the
shared scheduler modules when imported: CUDA_VISIBLE_DEVICES="" python -m
pytest tests/test_pmr_mixed.py
"""
from itertools import combinations
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.pmr import chain6 as C6  # noqa: E402

C = C6.C
C2 = C6.C2


def job(key, kind="pair", deps=(), pairs=None):
    result = {
        "key": key, "kind": kind,
        "command": ["python", "train.py", "--config", "c.yaml"],
        "deps": list(deps), "note": "", "mem_gib": C.DEFAULT_MEM[kind],
    }
    if pairs is not None:
        result["pairs"] = pairs
    return result


@pytest.fixture
def driver(tmp_path, monkeypatch):
    d = tmp_path / "driver6"
    for name, value in {
        "D2": d, "QUEUE": d / "queue.json", "LOCK": d / "queue.lock", "STATUS": d / "status.json",
        "LOGS": d / "logs", "STOP": d / "STOP", "DRAIN": d / "DRAIN", "POLICY": d / "policy.json",
    }.items():
        monkeypatch.setattr(C, name, value)
    (d / "logs").mkdir(parents=True)
    monkeypatch.setattr(C2, "PEERS", [])
    monkeypatch.setattr(C6, "cmp_jobs", lambda: {})
    monkeypatch.setattr(C, "mem_available_gib", lambda: 400.0)
    used = {g: 0.0 for g in C6.GPU_POOL}
    idle = {g: False for g in C6.GPU_POOL}
    monkeypatch.setattr(C, "gpu_used_gib", lambda g: used[g])
    monkeypatch.setattr(C2, "probe", lambda: idle.copy())
    calls = []

    class Proc:
        def __init__(self, cmd, cwd=None, env=None, stdout=None, stderr=None, start_new_session=False):
            self.pid, self.returncode = 123450 + len(calls), None
            calls.append({"cmd": cmd, "env": env})

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

    monkeypatch.setattr(C.subprocess, "Popen", Proc)
    return SimpleNamespace(path=d, calls=calls, idle=idle, used=used)


def only_mixed_free(driver):
    driver.idle.update({2: True, 5: True})
    driver.used[7] = 14.0


def test_card_pool_and_wrapper():
    assert C.D2.name == "driver6" and C2.PEERS == []
    assert C.PAIRS[0] == (5, 7)
    assert set(C.PAIRS) == set(combinations((0, 1, 2, 5, 7), 2))
    assert C2.DIRECT == ((5, 7),)
    assert all(any(g < 4 for g in p) for p in C2.BORROWED)
    assert C.PHYSICAL_GPUS == C2.EVAL_GPUS == (5, 7)
    assert C2.BORROW_JOB == str(ROOT / "tools/pmr/borrow_job2.py")


def test_only_gpu2_and_gpu5_free_dispatches_whole_mixed_pair(driver):
    only_mixed_free(driver)
    C.add([job("mixed")])
    chain = C6.Chain()
    chain.schedule(C.load_queue())
    state = chain.state["jobs"]["mixed"]
    assert state["gpus"] == [2, 5] and state["borrowed"]
    cmd = driver.calls[0]["cmd"]
    assert cmd[:4] == [C2.GPU_BORROW, "run", "--gpus", "2,5"]
    assert C2.BORROW_JOB in cmd
    assert driver.calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "2,5"


def test_busy_borrowed_card_blocks_mixed_pair(driver):
    only_mixed_free(driver)
    driver.idle[2] = False
    C.add([job("mixed")])
    chain = C6.Chain()
    chain.schedule(C.load_queue())
    assert chain.state["jobs"]["mixed"]["status"] == "PENDING"
    assert not driver.calls


def test_busy_direct_card_blocks_mixed_pair(driver):
    only_mixed_free(driver)
    driver.idle[5] = False
    C.add([job("mixed")])
    chain = C6.Chain()
    chain.schedule(C.load_queue())
    assert chain.state["jobs"]["mixed"]["status"] == "PENDING"
    assert not driver.calls


def test_pairs_do_not_overlap(driver):
    only_mixed_free(driver)
    C.add([job("first"), job("second")])
    chain = C6.Chain()
    queue = C.load_queue()
    chain.schedule(queue)
    chain.last_pair_start = time.monotonic() - C.MIN_PAIR_GAP_S - 1
    chain.schedule(queue)
    assert chain.state["jobs"]["first"]["gpus"] == [2, 5]
    assert chain.state["jobs"]["second"]["status"] == "PENDING"
    assert len(driver.calls) == 1


def test_peer_use_prevents_mixed_dispatch(driver, monkeypatch):
    only_mixed_free(driver)
    monkeypatch.setattr(C2, "peer_jobs", lambda: [("pair", (2, 5))])
    C.add([job("mixed")])
    chain = C6.Chain()
    chain.schedule(C.load_queue())
    assert chain.state["jobs"]["mixed"]["status"] == "PENDING"
    assert not driver.calls


def test_direct_pair_is_preferred(driver):
    driver.idle.update({g: True for g in C6.GPU_POOL})
    C.add([job("direct")])
    chain = C6.Chain()
    chain.schedule(C.load_queue())
    assert chain.state["jobs"]["direct"]["gpus"] == [5, 7]
    assert not chain.state["jobs"]["direct"].get("borrowed")
    assert driver.calls[0]["cmd"] == job("direct")["command"]


def test_job_pair_restriction_is_honored(driver):
    only_mixed_free(driver)
    C.add([job("direct_only", pairs=[[5, 7]])])
    chain = C6.Chain()
    chain.schedule(C.load_queue())
    assert chain.state["jobs"]["direct_only"]["status"] == "PENDING"
    assert not driver.calls


def test_readings_stay_on_direct_cards(driver):
    C.add([job(f"e{i}", "gpu") for i in range(6)])
    chain = C6.Chain()
    chain.schedule(C.load_queue())
    used = [v["gpus"][0] for v in chain.state["jobs"].values() if v["status"] == "RUNNING"]
    assert sorted(used) == [5, 5, 7, 7]
    assert all(call["cmd"][0] != C2.GPU_BORROW for call in driver.calls)


def test_cmp_cost_hold_still_applies(driver, monkeypatch):
    busy = {"train:fme:s3407": {"state": "RUNNING"}, "cost:epx": {"state": "PENDING"}}
    assert not C6.cmp_cost_block(busy)
    assert C6.cmp_cost_block({**busy, "train:fme:s3407": {"state": "DONE"}})
    assert C6.cmp_cost_block({**busy, "cost:epx": {"state": "RUNNING"}})
    assert not C6.cmp_cost_block({"cost:epx": {"state": "DONE"}})
    monkeypatch.setattr(C6, "cmp_jobs", lambda: {"cost:epx": {"state": "RUNNING"}})
    C.add([job("training"), job("reading", "gpu"), job("cpu", "cpu")])
    chain = C6.Chain()
    chain.schedule(C.load_queue())
    assert not driver.calls
    assert all(v["status"] == "PENDING" for v in chain.state["jobs"].values())
    assert chain.state["held_for_cmp_cost_utc"]


def test_waiting_mixed_job_rehomes_to_free_direct_pair(driver, monkeypatch):
    only_mixed_free(driver)
    C.add([job("mixed")])
    chain = C6.Chain()
    queue = C.load_queue()
    chain.schedule(queue)
    driver.used[7] = 0.0
    chain.waiting["mixed"] = time.monotonic() - C2.REHOME_S - 1
    chain.last_pair_start = 0.0
    stopped = []
    monkeypatch.setattr(C6.C4.C3, "stop_group", lambda p: stopped.append(p.pid))
    monkeypatch.setattr(C, "live_group", lambda pid: False)
    chain.schedule(queue)
    state = chain.state["jobs"]["mixed"]
    assert stopped and state["gpus"] == [5, 7] and not state["borrowed"]
    assert state["rehomed_from"] == [2, 5]
    assert driver.calls[-1]["cmd"][:2] == [C2.PY, C2.BORROW_JOB]
    assert driver.calls[-1]["env"]["CUDA_VISIBLE_DEVICES"] == "5,7"


@pytest.mark.parametrize("block", ["running", "peer", "restricted"])
def test_mixed_rehome_preserves_other_constraints(driver, monkeypatch, block):
    only_mixed_free(driver)
    pairs = [[2, 5]] if block == "restricted" else None
    C.add([job("mixed", pairs=pairs)])
    chain = C6.Chain()
    queue = C.load_queue()
    chain.schedule(queue)
    driver.used[7] = 0.0
    chain.waiting["mixed"] = time.monotonic() - C2.REHOME_S - 1
    chain.last_pair_start = 0.0
    if block == "running":
        monkeypatch.setattr(C2, "borrow_phase", lambda key: "RUNNING")
    elif block == "peer":
        monkeypatch.setattr(C2, "peer_jobs", lambda: [("gpu", (7,))])
    stopped = []
    monkeypatch.setattr(C6.C4.C3, "stop_group", lambda p: stopped.append(p.pid))
    chain.schedule(queue)
    assert not stopped and chain.state["jobs"]["mixed"]["gpus"] == [2, 5]
    assert len(driver.calls) == 1


def test_probe_checks_every_candidate_and_rejects_foreign_compute(monkeypatch):
    calls = []
    gpus = [
        {"index": 0, "used_mib": 10, "utilization": 0, "processes": []},
        {"index": 1, "used_mib": 10, "utilization": 0,
         "processes": [{"uid": 2000, "kind": "C"}]},
        {"index": 2, "used_mib": 10, "utilization": 0,
         "processes": [{"uid": 9999, "kind": "G"}]},
        {"index": 5, "used_mib": 256, "utilization": 5,
         "processes": [{"uid": 2000, "kind": "G"}]},
        {"index": 7, "used_mib": 257, "utilization": 0, "processes": []},
    ]

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=json.dumps({"gpus": gpus}))

    monkeypatch.setattr(C6.subprocess, "run", run)
    monkeypatch.setattr(C6.pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=9999))
    assert C6.probe() == {0: True, 1: False, 2: False, 5: True, 7: False}
    assert calls[0] == [C2.GPU_BORROW, "probe", "--gpus", "0,1,2,5,7"]
    monkeypatch.setattr(C6.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="invalid"))
    assert C6.probe() == {}
