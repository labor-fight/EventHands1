"""PMR driver 5 and the D2 single-seed screen (docs/EXPERIMENT_RECORDS.md#pmr-amend9-20261110,
#pmr-d2-screen-prereg-20261110). CPU only: CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_pmr_amend9.py
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]


def test_driver5_cards_and_cmp_hold():
    from tools.pmr import chain5 as C5
    assert C5.C.D2.name == "driver5" and C5.C2.PEERS == []
    assert C5.C.PAIRS == ((5, 7), (0, 1)) and C5.C2.DIRECT == ((5, 7),) and C5.C2.BORROWED == ((0, 1),)
    assert C5.C2.EVAL_GPUS == (5, 7) and C5.C.PHYSICAL_GPUS == (5, 7)
    assert not ({3, 4, 6} & {g for p in C5.C.PAIRS for g in p})
    busy = {"train:fme:s3407": {"state": "RUNNING"}, "cost:epx": {"state": "PENDING"}, "rows:a6": {"state": "PENDING"}}
    assert not C5.cmp_cost_block(busy)
    assert C5.cmp_cost_block({**busy, "train:fme:s3407": {"state": "DONE"}})
    assert C5.cmp_cost_block({**busy, "cost:epx": {"state": "RUNNING"}})
    assert not C5.cmp_cost_block({"train:fme:s3407": {"state": "DONE"}, "cost:epx": {"state": "DONE"},
                                  "rows:a6": {"state": "PENDING"}})
    assert not C5.cmp_cost_block({})


def test_screen_jobs_are_s9_commands_at_seed_3407():
    from tools.pmr import jobs16 as J16
    from tools.pmr import jobs17 as J17
    jobs = J17.s9x()
    keys = [j["key"] for j in jobs]
    assert len(keys) == len(set(keys))
    src = {j["key"]: j for j in J16.s9("s9")}
    seen = set()
    for j in jobs:
        assert all(d in seen for d in j["deps"]), j["key"]
        seen.add(j["key"])
        assert "3408" not in " ".join(j["command"]) and "3409" not in " ".join(j["command"])
        s = "s9" + j["key"][len("s9x"):]
        if s in src:
            assert j["command"] == src[s]["command"] and j["kind"] == src[s]["kind"]
        if j["kind"] == "pair" and "rol5" in j["key"]:
            assert j["pairs"] == [[5, 7]]
    order = {k: i for i, k in enumerate(keys)}
    assert order["s9x:read:d1hubfull:s3407:e1"] < order["s9x:rpre:d1hubfull_rol5"]
    assert "s9x:raccept:d1hubfull_rol5:s3407" in next(j for j in jobs if j["key"] == "s9x:read:d1hubfull:s3407:pert")["deps"]
    assert not any(j["kind"] == "exclusive" or "preflight.yaml" in " ".join(j["command"]) and "rol5" not in j["key"]
                   for j in jobs)
    assert jobs[-1]["key"] == "s9x:screen" and len(jobs[-1]["deps"]) == 10


def test_rule():
    from tools.pmr import analyze13 as A
    d1 = {"five": [11.69, 8.90, 9.59, 7.56, 10.20], "yb": 4.64, "rootG": 8.42}

    def d2(dl=0.0, dg=0.0, dra=0.0, yb=4.64, root=8.42):
        return {"five": [11.69 + dl, 8.90 + dg, 9.59, 7.56, 10.20 + dra], "yb": yb, "rootG": root}
    assert A.rule(d1, d2())["verdict"] == "D2_NULL"
    assert A.rule(d1, d2(dg=-0.6, dra=-0.31, yb=3.6, root=8.0))["verdict"] == "D2_SCREEN_PASS"
    assert A.rule(d1, d2(dg=-0.6, dra=-0.31, yb=-3.6, root=8.0))["verdict"] == "D2_SCREEN_PASS"
    assert A.rule(d1, d2(dg=-0.6, dra=-0.31, yb=-6.0, root=8.0))["verdict"] == "D2_METRIC_ONLY"
    assert A.rule(d1, d2(dg=-0.6, dra=-0.29, yb=3.6, root=8.0))["verdict"] == "D2_MECH_ONLY"
    assert A.rule(d1, d2(dl=0.6, dg=-0.6, dra=-0.31, yb=3.6, root=8.0))["verdict"] == "D2_MECH_ONLY"
    assert A.rule(d1, d2(dg=-0.6, dra=-0.31, yb=3.7, root=8.0))["verdict"] == "D2_METRIC_ONLY"
    assert A.rule(d1, d2(dg=-0.6, dra=-0.31, yb=3.6, root=8.1))["verdict"] == "D2_METRIC_ONLY"
