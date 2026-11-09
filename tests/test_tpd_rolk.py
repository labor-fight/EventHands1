"""TPD stage ROL-k registry (tools/tpd/ns_rolk.py, tools/tpd/run_rolk.py). CPU only; run in its own process (the
namespace patches the ROL / synthesis registries)."""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import copy  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]

from tools.tpd import ns_rolk  # noqa: E402

ns_rolk.enter()
from semkine import tpd_final as TF  # noqa: E402
from tools.tpd import arms_final as A  # noqa: E402
from tools.tpd import arms_rol as AR  # noqa: E402
from tools.tpd import run_rolk as RK  # noqa: E402


def test_config_is_rol5_plus_the_causal_final_block_and_names():
    for seed in (3407, 3408, 3409):
        for phase in ("preflight", "formal"):
            mine, rol5 = A.build_config("rolk", seed, phase), AR.build_config("rol5", seed, phase)
            assert mine["FINAL"] == {"CAUSAL": True, "ROUTES": "router", "ROLLOUT": True}
            assert mine["MODEL"]["INIT_FROM"] == rol5["MODEL"]["INIT_FROM"] and str(seed) in mine["MODEL"]["INIT_FROM"]
            assert mine["ROL"]["K"] == 5 and mine["ROL"]["GRAD"] and mine["TRAIN"]["PREFETCH_FACTOR"] == 1
            assert not mine["DATA"].get("XAG_SOURCE_ROOT")
            a, b = copy.deepcopy(mine), copy.deepcopy(rol5)
            a.pop("FINAL")
            for c in (a, b):
                c["TRAIN"].pop("OUTPUT_DIR"); c["TRAIN"].pop("RUN_NAME"); c["EVAL"].pop("OUTPUT_DIR")
                c["PFO"].pop("ARM"); c["PFO"].pop("SECTION")
            assert a == b


def test_model_class_has_rol_floor_and_the_final_model():
    cls = TF.final_model_class()
    assert cls.LR_FLOOR == 0.0 and cls.__mro__[1].__name__ == "FinalModel" and cls is TF.final_model_class()


def test_registry_paths_and_follow_up_jobs():
    assert A.ARMS["rolk"]["causal"] and A.ARMS["rolk"]["routes"] == "router" and A.ARMS["rolk"]["k"] == 5
    assert A.config_path("rolk", 3407).parent.name == "rolk" and A.run_name("rolk", 3408) == "tpd_rolk_s3408"
    jobs = RK.follow_up_jobs()
    keys = [j["key"] for j in jobs]
    assert keys[0] == "rolk:train:s3408" and jobs[0]["deps"] == ["rolk:screen"] and keys[-1] == "rolk:table"
    assert set(jobs[-1]["deps"]) == {k for k in keys if ":read:" in k} and len(jobs[-1]["deps"]) == 8
    assert all(j["command"][3].endswith("tools/tpd/train_rolk.py") and j["command"][1].startswith("PYTORCH_CUDA_ALLOC_CONF")
               for j in jobs if j["kind"] == "pair")
    assert RK.RF.GRID["formal"] == (500, 1000, 1500)
