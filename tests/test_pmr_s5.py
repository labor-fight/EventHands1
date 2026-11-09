"""PMR stage S5 tooling, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s5-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- tools/pmr/arms_any.py dispatches every registered arm to its registry; tools/pmr/arms_rol2.py builds a rollout config of
  an S4 refinement that differs from it exactly as D1's rollout configs differ from D1 (TPD's rol5 keys + prefetch 1).
- The S5 job list for either refinement is well formed (controls only for dual; the base acceptance of seed 3407 is S4's).
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, pre + k + ".") if isinstance(v, dict) else {pre + k: v})
    return out


def test_union_registry_dispatches():
    from tools.pmr import arms_any as U
    a = U.arms()
    assert {"d1r", "d1n", "d1m", "d1g0", "d1l1b", "d1dual", "d1hp"} <= set(a)
    assert U.parse("d1hp") == ("hp", "hp") and U.DESIGNS["hp"][0] == "semkine.pmr_trunk4" and U.DESIGNS["d1"][0] == "semkine.pmr_trunk"


@pytest.mark.parametrize("base", ("d1dual", "d1hp", "d1pos"))
def test_rollout_config_of_a_refinement(base):
    from tools.pmr import arms_any as U
    from tools.pmr import arms_rol as R1
    from tools.pmr import arms_rol2 as R2
    from tools.pmr import arms as A
    a, b = flat(R2.build_config(f"{base}_rol5", 3408, "formal")), flat(U.build_config(base, 3408, "formal"))
    d_new = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    c, e = flat(R1.build_config("d1r_rol5", 3408, "formal")), flat(A.build_config("d1r", 3408, "formal"))
    d_ref = {k for k in set(c) | set(e) if c.get(k, "<absent>") != e.get(k, "<absent>")}
    assert d_new == d_ref
    assert a["MODEL.INIT_FROM"].endswith(f"runs/pmr_{base}_s3408/pmr_{base}_s3408-step=6000.ckpt")


@pytest.mark.parametrize("arm", ("d1dual", "d1hp", "d1pos"))
def test_s5_job_list(arm):
    from tools.pmr import jobs8 as J8
    J = J8.s5(arm, "s5")
    keys = [j["key"] for j in J]
    scratch = [j for j in J if j["kind"] == "pair" and ("train4.py" in " ".join(j["command"]) or "train5.py" in " ".join(j["command"]))]
    assert len(scratch) == (2 if arm == "d1hp" else 4)
    rol = [j for j in J if j["kind"] == "pair" and "train_rol2.py" in " ".join(j["command"])]
    assert len(rol) == (1 + 3 if arm == "d1hp" else 3 + 5)
    assert keys[-1] == "s5:verdict" and keys[-2] == f"s5:cost:{arm}"
    first = [j for j in J if j["key"] == f"s5:rtrain:{arm}_rol5:s3407"][0]
    assert f"{'s4b' if arm == 'd1pos' else 's4'}:accept:{arm}" in first["deps"]
