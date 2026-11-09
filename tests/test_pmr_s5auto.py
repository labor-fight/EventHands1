"""PMR S5 auto-registration, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s5-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- tools/pmr/s5auto.py turns each possible written joint selection into the S5 job list of the selected refinement, keeps
  every rollout fine-tune on the direct pairs, and registers nothing for NO_REFINEMENT; the list's dependencies on S4 / S4b
  / S4c acceptances exist in driver 2's queue.
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

from tools.pmr import s5auto as S  # noqa: E402


@pytest.mark.parametrize("x", ("dual", "hp", "pos", "ghand", "gview"))
def test_each_selection_gives_its_s5_jobs(x):
    arm, jobs = S.plan({"verdict": f"REFINE_{x}"})
    assert arm == f"d1{x}" and jobs[0]["key"] == "s5:freeze" and jobs[-1]["key"] == "s5:verdict"
    rol = [j for j in jobs if "train_rol2.py" in " ".join(j["command"])]
    assert rol and all(j["pairs"] == [[4, 5], [6, 7]] for j in rol)
    assert all("pairs" not in j for j in jobs if j not in rol)
    assert {f"s5:rtrain:d1{x}_rol5:s{s}" for s in (3407, 3408, 3409)} <= {j["key"] for j in jobs}
    queued = {j["key"] for j in json.loads((ROOT / "outputs/pmr/20261109/driver2/queue.json").read_text())["jobs"]}
    own = {j["key"] for j in jobs}
    assert all(d in queued or d in own for j in jobs for d in j["deps"])


def test_no_refinement_registers_nothing():
    assert S.plan({"verdict": "NO_REFINEMENT"}) == (None, [])
    with pytest.raises(RuntimeError):
        S.plan({"verdict": "SOMETHING_ELSE"})
