"""PMR stage S3c job lists and report, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s3c-prereg-20261109). Run with
CUDA_VISIBLE_DEVICES="".

- D1's S3c readings need no dependency; each final form's list waits on S5's acceptances and verdict, keys never collide
  with D1's, and every dependency resolves within D1's list, S5's list for that arm or itself.
- The cost block comes after S5's verdict and every reading, the fast rows after the reference rows, the tables after both.
- tools/pmr/cost5.py rebinds tools/pmr/run.py to the union registry; tools/pmr/analyze8.py reads a cost report.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.environ.setdefault("PMR_DRIVER_DIR", "driver2")

from tools.pmr import jobs11 as J  # noqa: E402


def test_d1_readings():
    d = J.d1()
    assert len(d) == 3 + 12 + 3 and all(not j["deps"] for j in d) and all(j["key"].startswith("s3c:d1r:") for j in d)
    assert {j["kind"] for j in d} == {"cpu", "gpu"}
    assert any(j["command"][2:4] == ["tools.pmr.erh", "run"] for j in d)


@pytest.mark.parametrize("arm", ("d1dual", "d1hp", "d1pos", "d1ghand", "d1gview"))
def test_final_form_lists(arm):
    from tools.pmr import jobs8 as J8
    f = J.final(arm)
    s5 = {j["key"] for j in J8.s5(arm, "s5")}
    d1 = {j["key"] for j in J.d1()}
    own = {j["key"] for j in f}
    assert not own & d1 and not own & s5
    assert all(dep in s5 or dep in d1 or dep in own for j in f for dep in j["deps"])
    excl = [j for j in f if j["kind"] == "exclusive"]
    assert len(excl) == 4 + 4 + 2 and all(j["command"][2] == "tools.pmr.cost5" for j in excl)
    reads = {j["key"] for j in f if j["kind"] != "exclusive" and not j["key"].startswith("s3c:table")} | d1
    for j in excl:
        assert "--fast" in j["command"] or ("s5:verdict" in j["deps"] and reads <= set(j["deps"]))
    rows = {j["key"] for j in excl if "--fast" not in j["command"]}
    assert all(rows <= set(j["deps"]) for j in excl if "--fast" in j["command"])
    tables = [j for j in f if j["key"].startswith("s3c:table:")]
    assert [t["key"] for t in tables] == ["s3c:table:d1r", f"s3c:table:{arm}"]
    assert all(rows <= set(t["deps"]) and not any("fast" in d for d in t["deps"]) for t in tables)


def test_cost5_rebinds_the_registry():
    from tools.pmr import arms_any2 as U
    from tools.pmr import cost5  # noqa: F401
    from tools.pmr import run as R
    assert R.A is U


def test_report_reads_a_cost(tmp_path, monkeypatch):
    from tools.pmr import analyze8 as A8
    d = tmp_path / "costs" / "final__s43_s3407__exec__default"
    d.mkdir(parents=True)
    (d / "report.json").write_text(json.dumps({"params_total": 10, "macs_layers": {"median": 100.0}, "macs_mano_fk": 5,
                                               "latency_row": {"raw_ms": 2.0}, "latency_e2e_ms": {"p50": 3.0, "p99": 4.0}}))
    monkeypatch.setattr(A8, "OUT", tmp_path)
    assert A8.cost("s43_s3407", "exec", "default") == {"params": 10, "macs": 105.0, "row_ms": 2.0, "e2e_p50": 3.0,
                                                        "e2e_p99": 4.0}
    assert A8.cost("s43_s3407", "exec", "det") is None
