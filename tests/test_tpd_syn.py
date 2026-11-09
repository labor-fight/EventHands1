"""TPD synthesis CPU tests (docs/EXPERIMENT_RECORDS.md#tpd-syn-prereg-20261105): configs differ from B2's only in the
registered keys; the job list is well formed; the conditions close when their decision file is missing; the final rule."""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import arms_final as A  # noqa: E402

KEYS = {"EVAL.OUTPUT_DIR", "TRAIN.OUTPUT_DIR", "TRAIN.RUN_NAME", "DATA.XAG_SOURCE_ROOT", "TRAIN.NUM_WORKERS", "ROL.K",
        "ROL.GRAD", "ROL.CHECKPOINT", "ROL.WEIGHTS", "ROL.PACKET_MS", "FINAL.CAUSAL", "FINAL.ROUTES", "FINAL.ROLLOUT",
        "PFO.ARM", "PFO.BASE_ARM", "PFO.SECTION", "PFO.STUDY"}


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, pre + k + ".") if isinstance(v, dict) else {pre + k: v})
    return out


@pytest.mark.parametrize("arm", tuple(A.ARMS))
@pytest.mark.parametrize("seed", A.SEEDS)
def test_configs(arm, seed):
    from tools.pfo2 import arms as P2
    a, b = flat(A.build_config(arm, seed)), flat(P2.build_config("fgs", seed))
    diff = {k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>")}
    assert diff <= KEYS
    assert a["TRAIN.MAX_STEPS"] == 6000 and a["TRAIN.LR"] == 0.004 and a["FINAL.CAUSAL"] is True
    assert a["FINAL.ROUTES"] == "label_part" and a["DATA.XAG_SOURCE_ROOT"]
    assert "MODEL.INIT_FROM" not in a
    assert (a["TRAIN.NUM_WORKERS"] == 12) == (A.ARMS[arm]["k"] > 1)


def test_jobs_conditions_and_rule(tmp_path, monkeypatch):
    from tools.tpd import analyze_syn as AS
    from tools.tpd import jobs_syn as J
    from tools.tpd import maybe_syn as MS
    jobs = J.stage()
    keys = [j["key"] for j in jobs]
    assert len(keys) == len(set(keys)) and all(set(j) <= {"key", "kind", "command", "deps", "note", "mem_gib"} for j in jobs)
    monkeypatch.setattr(MS, "OUT", tmp_path)
    for cond in ("any_chosen", "chosen:rol3d", "s2:c0", "s3:any", "s3:c3d"):
        assert MS.is_open(cond) is False
    row = lambda five: {"five": five}  # noqa: E731
    ok = AS.final_rule({3407: row([12.0, 7.9, 10.0, 6.5, 10.0]), 3408: row([12.5, 7.8, 10.5, 6.6, 10.1]),
                        3409: row([12.6, 7.9, 10.6, 6.5, 10.3])})
    assert ok["verdict"] == "FINAL_PASS"
    bad = AS.final_rule({3407: row([12.0, 7.9, 10.0, 6.5, 9.0]), 3408: row([12.5, 7.8, 10.5, 6.6, 9.1]),
                         3409: row([12.6, 7.9, 10.6, 6.5, 11.2])})
    assert bad["verdict"] == "FINAL_FAIL" and bad["seed_ra_below_11"][3409] is False
