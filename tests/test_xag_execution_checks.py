import json
import pytest
from tools.xag.execution_checks import check_common_outputs
from tools.xag.prepare import sha
from tools.xag import execution_checks as checks


def artifact(tmp_path, reason):
    data=tmp_path/"zgz_local_run0000.npz"
    data.write_bytes(b"Raw trajectory bytes: full semantic audit is separate")
    record={"cache_sha256":sha(data),"failures":[{"reason":reason,"step":0,"row":0}]}
    data.with_suffix(".json").write_text(json.dumps(record))


@pytest.mark.parametrize("reason",["OutOfMemoryError","RuntimeError","IndexError","ValueError"])
def test_hidden_program_failure_stops_next_model(tmp_path,reason):
    artifact(tmp_path,reason)
    with pytest.raises(RuntimeError,match="implementation failure"):
        check_common_outputs(tmp_path,expected_files=1)


@pytest.mark.parametrize("reason",["FloatingPointError","shared_reference_failed"])
def test_recorded_numerical_failure_continues_to_full_audit(tmp_path,reason):
    artifact(tmp_path,reason)
    assert check_common_outputs(tmp_path,expected_files=1)["recorded_numerical_failures"]==1


def test_partial_common_run_cannot_start_own_evaluation(tmp_path):
    artifact(tmp_path,"FloatingPointError")
    with pytest.raises(RuntimeError,match="Incomplete"):
        check_common_outputs(tmp_path,expected_files=42)


@pytest.mark.parametrize("case,expected",[("planned",True),("complete",True),("unplanned",False),
                                         ("dead",False),("resource_drift",False),("driver_drift",False)])
def test_only_live_registered_handoff_accepts_temporary_interrupt(tmp_path,monkeypatch,case,expected):
    driver=tmp_path/"orchestrate_frozen.py";driver.write_text("verified driver")
    resources=tmp_path/"fstar_resources_protocol.json"
    resources.write_text(json.dumps({"driver_path":str(driver),"driver_sha256":sha(driver)}))
    status={"resources_sha256":sha(resources),"expected_driver_sha256":sha(driver),"handoff_pid":100,
            "new_driver_pid":200,"status":"HANDOFF_IN_PROGRESS"}
    if case=="complete":status["status"]="HANDOFF_COMPLETE_HOLD_RELEASED"
    if case=="unplanned":status["status"]="WAITING_COMPLETE_PROBES"
    (tmp_path/"fstar_handoff_status.json").write_text(json.dumps(status))
    if case=="resource_drift":resources.write_text(resources.read_text()+"\n")
    if case=="driver_drift":driver.write_text("changed driver")
    monkeypatch.setattr(checks,"_running",lambda *args:case!="dead")
    assert checks.controlled_frozen_handoff(tmp_path) is expected
