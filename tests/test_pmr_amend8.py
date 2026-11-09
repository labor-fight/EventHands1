"""PMR engineering amendment 8, CPU (docs/EXPERIMENT_RECORDS.md#pmr-amend8-20261109). Run with CUDA_VISIBLE_DEVICES="".

- The S8 repair list re-runs the failed d1hub rol5 fine-tunes under :r2 on the direct pairs only, keeps the surviving
  seed's readings, and every dependency resolves within the list or driver 4's queue.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.environ.setdefault("PMR_DRIVER_DIR", "driver4")


def test_s8_repair_jobs():
    from tools.pmr import jobs15 as J15
    J = J15.s8r2([3407, 3408])
    queued = {j["key"] for j in json.loads((ROOT / "outputs/pmr/20261109/driver4/queue.json").read_text())["jobs"]}
    own = {j["key"] for j in J}
    assert not own & queued and all(d in own or d in queued for j in J for d in j["deps"])
    pairs = [j for j in J if j["kind"] == "pair"]
    assert [j["key"] for j in pairs] == ["s8:rtrain:d1hub_rol5:s3407:r2", "s8:rtrain:d1hub_rol5:s3408:r2"]
    assert all(j["pairs"] == [[4, 5], [6, 7]] and "train_rol3.py" in " ".join(j["command"]) for j in pairs)
    cost = [j for j in J if j["kind"] == "exclusive"][0]
    assert "s8:rread:d1hub_rol5:s3409:e1" in cost["deps"] and "s8:rread:d1hub_rol5:s3407:e1:r2" in cost["deps"]
    assert J[-1]["key"] == "s8:verdict:r2" and J[-1]["deps"] == [cost["key"]]
