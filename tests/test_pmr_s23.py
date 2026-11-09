"""PMR stages S2 / S3 tooling, CPU (docs/EXPERIMENT_RECORDS.md#pmr-prereg-20261109, "过筛之后"). Run with
CUDA_VISIBLE_DEVICES="".

- S2: the job list trains R and R-noprop at seeds 3408 / 3409 through the registered entry (with NCCL_P2P_DISABLE=1),
  reads e1 / e3 / pert / f1 and ends in the written confirmation; the confirmation rule on synthetic gaps.
- S3: a rollout arm's config differs from its base PMR arm's config (same seed and phase) exactly in the keys TPD's rol5
  changed on B2's config (schedule, warm start, ROL block, names); the warm start is the base arm's step-6000 checkpoint.
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


def test_s2_job_list_is_well_formed():
    from tools.pmr import jobs3 as J3
    J = J3.s2("d1", "s2d1")
    keys = [j["key"] for j in J]
    trains = [j for j in J if j["kind"] == "pair"]
    assert len(trains) == 4 and all("NCCL_P2P_DISABLE=1" in j["command"] for j in trains)
    assert {j["note"] for j in trains} == {f"{a} formal s{s}" for a in ("d1r", "d1n") for s in (3408, 3409)}
    assert keys[0] == "s2:d1:freeze" and keys[-1] == "s2:d1:confirm"
    assert sum(j["kind"] == "gpu" for j in J) == 16
    assert all(d in keys[:i] for i, j in enumerate(J) for d in j["deps"])


def test_s2_rule_on_synthetic_gaps():
    from tools.pmr.analyze2 import rule
    assert rule({3407: 1.2, 3408: 0.4, 3409: 0.9}) == "MODULE_ESTABLISHED"
    assert rule({3407: 1.2, 3408: -0.1, 3409: 0.9}) == "MODULE_NOT_ESTABLISHED"
    with pytest.raises(Exception):
        rule({3407: 1.0, 3408: 1.0})


@pytest.mark.parametrize("phase", ("preflight", "formal"))
def test_rollout_config_differs_from_its_base_in_the_rol5_keys(phase, monkeypatch):
    from tools.pmr import arms as A
    from tools.pmr import arms_rol as R
    from tools.tpd import arms_rol as T
    from tools.pfo2 import arms as P2

    a, b = flat(R.build_config("d1r_rol5", 3407, phase)), flat(A.build_config("d1r", 3407, phase))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    monkeypatch.setattr(T, "read", lambda p: {"checkpoint": "/b2.ckpt"})
    monkeypatch.setattr(T, "b2_spec", lambda seed: "/b2spec")
    t, f = flat(T.build_config("rol5", 3407, phase)), flat(P2.build_config("fgs", 3407, phase))
    tdiff = {k for k in set(t) | set(f) if t.get(k, "<absent>") != f.get(k, "<absent>")}
    assert diff == (tdiff - {"PFO.SECTION", "PFO.STUDY"}) | {"TRAIN.PREFETCH_FACTOR"} and "PFO.BASE_ARM" in diff
    assert a["TRAIN.PREFETCH_FACTOR"] == 1
    assert a["MODEL.INIT_FROM"].endswith("runs/pmr_d1r_s3407/pmr_d1r_s3407-step=6000.ckpt") and a["ROL.K"] == 5
    assert a["ROL.GRAD"] is True
    assert a["PMR.VARIANT"] == "r" and a["PFO.TRUNK"] == "pmr" and a["TRAIN.LR_SCHEDULE"] == "cosine"


def test_s3a_job_list_is_well_formed():
    from tools.pmr import jobs4 as J4
    J = J4.s3a("d1", "s3ad1")
    keys = [j["key"] for j in J]
    trains = [j for j in J if j["kind"] == "pair"]
    assert len(trains) == 8 and all(j["mem_gib"] == J4.ROL_MEM and "NCCL_P2P_DISABLE=1" in j["command"] for j in trains)
    formal = {j["key"]: j for j in trains if ":train:" in j["key"]}
    assert set(formal) == {f"s3a:d1:train:d1r_rol5:s{s}" for s in (3407, 3408, 3409)} | {
        "s3a:d1:train:d1n_rol5:s3407", "s3a:d1:train:d1m_rol5:s3407"}
    assert "s2:d1:accept:d1r:s3408" in formal["s3a:d1:train:d1r_rol5:s3408"]["deps"]
    assert "s1:d1:accept:d1r" in formal["s3a:d1:train:d1r_rol5:s3407"]["deps"]
    assert keys[-1] == "s3a:d1:verdict" and sum(j["kind"] == "gpu" for j in J) == 20


def test_s3a_done_rule():
    from tools.pmr.analyze3 import done
    assert done([12.5, 7.9, 10.5, 6.5, 10.0]) == "DONE"
    assert done([12.9, 7.9, 10.5, 6.5, 10.0]) == "NOT_DONE"
