"""Protocol guardrails: reject a plausible-looking but incomplete result."""
import copy
import json
from pathlib import Path
import pytest
import yaml

ROOT=Path(__file__).resolve().parents[1]


def test_all_eighteen_configs_only_change_declared_factors_and_identity():
    p=json.loads((ROOT/"outputs/xag/protocol.json").read_text())
    assert len(p["runs"])==18
    base=yaml.safe_load((ROOT/"configs/xa/xa_c1_s3407.yaml").read_text())
    keys={(r["cell"],r["seed"],r["repeat"]) for r in p["runs"]}
    assert len(keys)==18
    assert p["resources"]["train_prefetch_factor"]==1
    for r in p["runs"]:
        cfg=yaml.safe_load((ROOT/r["config"]).read_text())
        assert cfg["TRAIN"]["BATCH_SIZE_PER_GPU"]==512 and cfg["TRAIN"]["DEVICES"]==2
        assert cfg["TRAIN"]["MAX_STEPS"]==6000
        assert cfg["TRAIN"]["PREFETCH_FACTOR"]==p["resources"]["train_prefetch_factor"]
        assert Path(cfg["DATA"]["SPLITS_MANIFEST"]).name=="splits_semkine.json"
        for key in ("XAG_ASSOC","XAG_GNN"): cfg["MODEL"].pop(key)
        for key in ("XAG_SOURCE_ROOT","SPLITS_MANIFEST"): cfg["DATA"].pop(key)
        cfg["SEED"]=base["SEED"]
        for key in ("OUTPUT_DIR","RUN_NAME"): cfg["TRAIN"][key]=base["TRAIN"][key]
        cfg["TRAIN"]["PREFETCH_FACTOR"]=base["TRAIN"]["PREFETCH_FACTOR"]
        cfg["EVAL"]=base["EVAL"]
        assert cfg==base


def _row():
    metric={k:1. for k in ("mpjpe_ra_mm","mpvpe_ra_mm","mpjpe_abs_mm","mpvpe_abs_mm")}
    return {"per_seed":{str(s):{"step":6000,"n_frames":2590,"sequences":["zgz_global","zgz_local"],
            **{d:copy.deepcopy(metric) for d in ("local","global","overall")}} for s in (3407,3408)},
            "two_seed_mean":{d:copy.deepcopy(metric) for d in ("local","global","overall")},
            "protocol":{"split":"val_core","subject":None,"manifest":"splits_semkine.json","step_ms":50,
                        "step_policy":"step6000","sequences":["zgz_global","zgz_local"]},
            "step_policy":"step6000","latency_ms_scaled_full1p75":1.,"macs_forward_packet":1.,"params_total":1.}


@pytest.mark.parametrize("case",["short","selected","missing_seed","nonfinite","wrong_subject","wrong_mean","wrong_window","bad_cost"])
def test_complete_matrix_guard_rejects_invalid_evidence(case):
    from tools.xag.finalize import validate_row
    row=_row(); s=row["per_seed"]["3407"]
    if case=="short": s["n_frames"]=2589
    if case=="selected": s["step"]=5500
    if case=="missing_seed": row["per_seed"].pop("3408")
    if case=="nonfinite": s["local"]["mpjpe_ra_mm"]=float("nan")
    if case=="wrong_subject": s["sequences"]=["ly_local","ly_global"]
    if case=="wrong_mean": row["two_seed_mean"]["local"]["mpjpe_ra_mm"]=0.5
    if case=="wrong_window": row["protocol"]["step_ms"]=200
    if case=="bad_cost": row["macs_forward_packet"]=float("nan")
    with pytest.raises(ValueError): validate_row(row,(3407,3408))


def test_complete_row_passes():
    from tools.xag.finalize import validate_row
    validate_row(_row(),(3407,3408))


def test_registered_config_uses_resolved_paths_under_legacy_model_import(monkeypatch):
    import sys, types
    from tools.xag.run import registered_config
    monkeypatch.setitem(sys.modules,"model",types.ModuleType("model"))
    p=json.loads((ROOT/"outputs/xag/protocol.json").read_text())
    cfg=registered_config(p["runs"][0])
    assert "_config_path" not in cfg
    for group,key in (("DATA","ROOT"),("MANO","NPZ"),("TRAIN","OUTPUT_DIR"),("EVAL","OUTPUT_DIR")):
        assert Path(cfg[group][key]).is_absolute()
