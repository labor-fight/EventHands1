"""Scope reduction must never dispatch a second seed or invent reliability."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.xag_single_seed import diagnostics as D
from tools.xag_single_seed import execute as E
from tools.xag_single_seed import report as R


def registered():
    root = Path(__file__).resolve().parents[1]
    return json.loads((root / "outputs/xag/protocol.json").read_text())


def six_scope():
    return {"active_runs": [r for r in registered()["runs"] if r["seed"] == 3407 and not r["repeat"]]}


def test_six_run_plan_reuses_two_and_preserves_live_training():
    scope = six_scope()
    handoff = {"adopted": [{"name": "xag_a1f0_s3407", "pid": 123, "physical_gpus": [4, 5]}]}
    journal = {"xag_a0f0_s3407": {"status": "TRAINED"}, "xag_a0f1_s3407": {"status": "TRAINED"},
               "xag_a0f0_s3408": {"status": "TRAINED"}, "xag_a0f0r_s3407": {"status": "TRAINED"}}
    complete, retained, pending = E.training_plan(scope, handoff, journal)
    assert [r["name"] for r in complete] == ["xag_a0f0_s3407", "xag_a0f1_s3407"]
    assert retained[0][0]["name"] == "xag_a1f0_s3407"
    assert [(r["name"], pair) for r, pair in pending] == [
        ("xag_a1f1_s3407", [0, 1]), ("xag_a2f0_s3407", [2, 3]), ("xag_a2f1_s3407", [6, 7])]
    assert all(r["seed"] == 3407 and not r["repeat"] for r in complete + [r for r, _ in retained + pending])


@pytest.mark.parametrize("mutation", ["second_seed", "repeat", "missing", "extra", "duplicate"])
def test_protocol_refuses_any_unselected_or_missing_run(mutation):
    old = registered()
    scope = six_scope()
    if mutation == "second_seed":
        scope["active_runs"][0] = next(r for r in old["runs"] if r["name"] == "xag_a0f0_s3408")
    elif mutation == "repeat":
        scope["active_runs"][0] = next(r for r in old["runs"] if r["name"] == "xag_a0f0r_s3407")
    elif mutation == "missing":
        scope["active_runs"].pop()
    elif mutation == "extra":
        scope["active_runs"].append(old["runs"][1])
    else:
        scope["active_runs"][1] = scope["active_runs"][0]
    with pytest.raises(RuntimeError, match="exactly the six"):
        E.validate_protocol(scope, old)


def fake_rows():
    values = {"a0f0": 20., "a0f1": 18., "a1f0": 16., "a1f1": 12., "a2f0": 15., "a2f1": 9.}
    return {cell: {"per_seed": {"3407": {domain: {metric: value for metric in R.F.METRICS}
                                                     for domain in R.F.DOMAINS}}}
            for cell, value in values.items()}


def test_single_seed_arithmetic_does_not_invent_noise_or_gate():
    rows = fake_rows()
    result = R.descriptive_effects(rows)
    for domains in result.values():
        for effect in domains.values():
            assert effect["gnn_at_base_mm"] == 2.
            assert effect["associations"]["1"]["association_at_legacy_mm"] == 4.
            assert effect["associations"]["1"]["interaction_mm"] == 2.
            assert effect["associations"]["2"]["interaction_mm"] == 4.
            assert effect["tau_mm"] is None and effect["gnn_gate"] is None
            assert effect["cross_seed_sign_agreement"] is None
            assert effect["gate_status"] == "NOT_ASSESSED_SINGLE_SEED_NO_REPEATS"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1., True])
def test_invalid_or_nonfinite_main_row_cannot_enter_effects(bad):
    rows = fake_rows()
    rows["a0f0"]["per_seed"]["3407"]["local"]["mpjpe_ra_mm"] = bad
    with pytest.raises(ValueError, match="invalid metrics"):
        R.descriptive_effects(rows)


def test_extra_seed_and_missing_cell_are_rejected_in_main_effects():
    rows = fake_rows()
    rows["a0f0"]["per_seed"]["3408"] = deepcopy(rows["a0f0"]["per_seed"]["3407"])
    with pytest.raises(ValueError, match="exactly seed3407"):
        R.descriptive_effects(rows)
    rows = fake_rows()
    rows.pop("a2f1")
    with pytest.raises(ValueError, match="every one"):
        R.descriptive_effects(rows)


def test_diagnostic_effects_retain_nonfinite_and_missing_cells():
    vals = {c: rows["per_seed"]["3407"]["local"]["mpjpe_ra_mm"] for c, rows in fake_rows().items()}
    result = D.descriptive_effect(vals)
    assert result["gnn_at_base_mm"] == 2.
    assert result["associations"]["2"]["interaction_mm"] == 4.
    assert result["reliability_gate_pass"] is None
    assert result["repeat_noise_tau_mm"] is None
    vals["a0f0"] = None
    assert D.descriptive_effect(vals)["status"] == "UNDEFINED_NUMERICAL_FAILURE"
    vals.pop("a0f0")
    assert D.descriptive_effect(vals)["status"] == "UNDEFINED_INCOMPLETE"


def test_scope_declares_all_six_cells_as_original_registered_configs():
    scope = json.loads((R.SCOPE / "protocol.json").read_text())
    originals = registered()
    assert len(scope["excluded_runs"]) == 12
    assert len(R.scope_runs()) == len(D.selected_runs()) == 6
    E.validate_protocol(scope, originals)


def test_incomplete_diagnostic_denominator_cannot_be_used_as_survivor_mean():
    class Audit:
        declared = {r["name"]: r for r in six_scope()["active_runs"]}
        means = {( "common", name, "clean", "local"): {
                    "mpjpe_ra_mm": {"n": 1204, "mean": 20.}}
                 for name in declared}

    audit = Audit()
    audit.means[("common", "xag_a2f1_s3407", "clean", "local")]["mpjpe_ra_mm"] = {
        "n": 1203, "mean": 1.}
    result = D.summarize_diagnostic_effects(audit)
    effect = result["common"]["clean"]["local"]["mpjpe_ra_mm"]
    assert effect["status"] == "UNDEFINED_INCOMPLETE"
    assert effect["missing_cells"] == ["a2f1"]
