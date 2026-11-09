"""PMR stage S8 tooling, CPU (docs/EXPERIMENT_RECORDS.md#pmr-s8-prereg-20261109). Run with CUDA_VISIBLE_DEVICES="".

- tools/pmr/arms_any3.py extends the union registry with the S6 / S7 arms; the S5 rollout registry then builds an S6 arm's
  rol5 config that differs from the arm exactly as D1's rol5 configs differ from D1, warm-started from the arm's own
  accepted checkpoint.
- The S8 job list for d1hub is well formed: dependencies resolve within the list or driver 3's queue, rollouts use the
  S8 entry on any pair, the verdict reads the S5 rule with D1's controls.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
os.environ.setdefault("PMR_DRIVER_DIR", "driver3")

from tools.pmr import arms_any3  # noqa: E402,F401
from tools.pmr import arms_any2 as U  # noqa: E402


def flat(d, pre=""):
    out = {}
    for k, v in d.items():
        out.update(flat(v, pre + k + ".") if isinstance(v, dict) else {pre + k: v})
    return out


def test_union_takes_the_s6_s7_arms():
    a = U.arms()
    assert {"d1hub", "d1ghandhp", "d1ghandhub", "d1ghand", "d1r"} <= set(a)
    assert U.DESIGNS["hub"][0] == "semkine.pmr_trunk7" and U.DESIGNS["ghandhp"][0] == "semkine.pmr_trunk8"


def test_rollout_config_of_d1hub():
    from tools.pmr import arms as A
    from tools.pmr import arms_rol as R1
    from tools.pmr import arms_rol2 as R2
    a, b = flat(R2.build_config("d1hub_rol5", 3408, "formal")), flat(U.build_config("d1hub", 3408, "formal"))
    c, e = flat(R1.build_config("d1r_rol5", 3408, "formal")), flat(A.build_config("d1r", 3408, "formal"))
    diff = lambda x, y: {k for k in set(x) | set(y) if x.get(k, "<absent>") != y.get(k, "<absent>")}  # noqa: E731
    assert diff(a, b) == diff(c, e) and "TRAIN.PREFETCH_FACTOR" in diff(a, b)
    assert a["MODEL.INIT_FROM"].endswith("pmr_d1hub_s3408/pmr_d1hub_s3408-step=6000.ckpt")
    assert R2.init_sha("d1hub", 3407)


def test_s8_jobs():
    from tools.pmr import jobs14 as J14
    J = J14.s8("d1hub", "s8")
    queued = {j["key"] for j in json.loads((ROOT / "outputs/pmr/20261109/driver3/queue.json").read_text())["jobs"]}
    own = {j["key"] for j in J}
    assert not own & queued and all(d in own or d in queued for j in J for d in j["deps"])
    assert "s6:accept:d1hub" in {d for j in J for d in j["deps"]}
    rol = [j for j in J if "train_rol3.py" in " ".join(j["command"])]
    assert len(rol) == 4 and all("pairs" not in j for j in J)
    assert J[-1]["key"] == "s8:verdict" and J[-1]["command"][2:] == ["tools.pmr.analyze11", "s5", "--arm", "d1hub"]


def test_verdict_takes_d1hub():
    from tools.pmr import analyze11  # noqa: F401
    from tools.pmr import analyze7 as A7
    assert A7.NOPROP["d1hub"] == "d1n" and A7.MLP["d1hub"] == "d1m"
