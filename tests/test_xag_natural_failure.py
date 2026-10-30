"""Only completed, intact natural evaluations may become numerical evidence."""
import copy
import json
from pathlib import Path
import sys

import pytest

from tools.xag import finalize as F
from tools.xag import run as RUN


def row(seeds=(3407, 3408), bad=None):
    metrics = {key: 1. for key in F.METRICS}
    value = {"per_seed": {}, "two_seed_mean": {}, "step_policy": "step6000",
             "protocol": {"split": "val_core", "subject": None, "manifest": "splits_semkine.json",
                          "step_ms": 50, "step_policy": "step6000", "sequences": ["zgz_global", "zgz_local"]},
             "latency_ms_scaled_full1p75": 1., "macs_forward_packet": 1., "params_total": 1.}
    for seed in seeds:
        record = {"step": 6000, "n_frames": 2590, "sequences": ["zgz_global", "zgz_local"],
                  "local": metrics.copy(), "global": metrics.copy()}
        if bad is not None and seed == seeds[0]:
            record["local"]["mpjpe_ra_mm"] = bad
        record["overall"] = {key: sum(record[d][key] * F.COUNTS[d] for d in ("local", "global")) / 2590
                             for key in F.METRICS}
        record["per_seq"] = {"zgz_" + d: record[d].copy() for d in ("local", "global")}
        value["per_seed"][str(seed)] = record
    value["two_seed_mean"] = {d: {key: sum(value["per_seed"][str(s)][d][key] for s in seeds) / len(seeds)
                                  for key in F.METRICS} for d in F.DOMAINS}
    return value


def environment(tmp_path, monkeypatch):
    output = tmp_path / "outputs/xag"
    output.mkdir(parents=True)
    (tmp_path / "outputs/semkine").mkdir(parents=True)
    for name in ("protocol.json", "source_manifest.json", "frozen_sources.json"):
        (output / name).write_text("{}")
    monkeypatch.setattr(F, "ROOT", tmp_path)
    monkeypatch.setattr(F, "OUT", output)
    monkeypatch.setattr(RUN, "ROOT", tmp_path)
    monkeypatch.setattr(RUN, "OUT", output)
    monkeypatch.setattr(F, "validate_identity", lambda *a: None)
    return output


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_only_propagated_complete_nonfinite_summary_can_be_numerical(bad):
    actual = F.validate_row(row(bad=bad), (3407, 3408), allow_numerical=True)
    assert actual["status"] == "COMPLETE_WITH_NUMERICAL_FAILURES"
    assert actual["expected_endpoints"] == 5180
    assert actual["per_seed_expected_endpoints"] == {"3407": 2590, "3408": 2590}
    assert actual["failed_endpoint_count"] is None  # aggregate output cannot identify which frames failed
    assert len(actual["nonfinite_metrics"]) == 2  # affected local and full mean
    with pytest.raises(ValueError, match="nonfinite"):
        F.validate_row(row(bad=bad), (3407, 3408))


@pytest.mark.parametrize("mutation", ["short", "shape", "sequence", "mean", "overall", "cost", "negative"])
def test_allowing_numerical_never_masks_structural_or_denominator_failure(mutation):
    value = row(bad=float("nan"))
    record = value["per_seed"]["3407"]
    if mutation == "short": record["n_frames"] = 2589
    elif mutation == "shape": record["per_seq"] = {"zgz_local": record["per_seq"]["zgz_local"]}
    elif mutation == "sequence": record["sequences"].append("zgz_local")
    elif mutation == "mean": value["two_seed_mean"]["local"]["mpjpe_ra_mm"] = 1.
    elif mutation == "overall": record["overall"]["mpjpe_ra_mm"] = 1.
    elif mutation == "cost": value["macs_forward_packet"] = float("nan")
    elif mutation == "negative": record["global"]["mpjpe_abs_mm"] = float("-inf")
    with pytest.raises((ValueError, KeyError, TypeError)):
        F.validate_row(value, (3407, 3408), allow_numerical=True)


def test_successful_row_continues_with_receipt_and_retains_original_nan(tmp_path, monkeypatch):
    output = environment(tmp_path, monkeypatch)
    prefix = "xag_a1f0"
    path = tmp_path / f"outputs/semkine/{prefix}_main_row_step6000.json"
    content = json.dumps(row(bad=float("nan")))
    calls = []
    def evaluate(*args):
        calls.append(args)
        path.write_text(content)
    journals = []
    monkeypatch.setattr(RUN, "run_command", evaluate)
    monkeypatch.setattr(RUN, "journal", lambda *a, **kw: journals.append(kw))
    RUN.row_one(prefix, False, 0)
    RUN.row_one(prefix, False, 1)  # restart reuses only the hash-bound successful result
    assert len(calls) == 1
    assert path.read_text() == content
    assert all(j["status"] == "EVALUATED_WITH_NUMERICAL_FAILURES" for j in journals)
    receipt = json.loads((output / f"main_row_audits/{prefix}.json").read_text())
    assert receipt["evaluation_process_exit_code"] == 0
    assert receipt["validation"]["expected_endpoints"] == 5180
    path.write_text(content + "\n")
    with pytest.raises(ValueError, match="receipt"):
        RUN.row_one(prefix, False, 0)


@pytest.mark.parametrize("error", [RuntimeError("CUDA out of memory"), ValueError("wrong model shape")])
def test_failed_evaluator_cannot_publish_numerical_receipt_even_if_it_left_a_row(tmp_path, monkeypatch, error):
    output = environment(tmp_path, monkeypatch)
    prefix = "xag_a0f0"
    path = tmp_path / f"outputs/semkine/{prefix}_main_row_step6000.json"
    def fail(*args):
        path.write_text(json.dumps(row(bad=float("nan"))))
        raise error
    monkeypatch.setattr(RUN, "run_command", fail)
    with pytest.raises(type(error), match=str(error)):
        RUN.row_one(prefix, False, 0)
    assert not (output / f"main_row_audits/{prefix}.json").exists()
    with pytest.raises(FileNotFoundError, match="receipt"):
        RUN.row_one(prefix, False, 0)


def complete_rows(tmp_path, numerical):
    for cell in F.CELLS:
        for repeat in (False, True):
            prefix = "xag_" + cell + ("r" if repeat else "")
            seeds = (3407,) if repeat else (3407, 3408)
            value = row(seeds, float("nan") if numerical and cell == "a1f1" and not repeat else None)
            path = tmp_path / f"outputs/semkine/{prefix}_main_row_step6000.json"
            path.write_text(json.dumps(value))
            validation = F.validate_row(value, seeds, allow_numerical=True)
            F.row_receipt(path, prefix, validation, publish=True)


def test_finalizer_retains_all_twelve_rows_without_emitting_finite_table(tmp_path, monkeypatch):
    output = environment(tmp_path, monkeypatch)
    complete_rows(tmp_path, numerical=True)
    from tools.xag import prepare
    monkeypatch.setattr(prepare, "verify", lambda: None)
    verified = []
    monkeypatch.setattr(prepare, "verify_sources", lambda full: verified.append(full))
    monkeypatch.setattr(sys, "argv", ["finalize.py", "--report"])
    monkeypatch.setattr(F.subprocess, "check_output", lambda *a, **kw: pytest.fail("Nonfinite matrix generated a finite table"))
    F.main()  # returns normally so the coordinator can execute common/own stages
    result = json.loads((output / "decision.json").read_text())
    assert result["status"] == "COMPLETE_WITH_NUMERICAL_FAILURES"
    assert len(result["source_rows"]) == len(result["row_audits"]) == 12
    assert result["numerical_failure_rows"] == ["xag_a1f1"]
    assert result["effects"] is None and result["final_bottleneck_conclusion"] is None
    assert not result["canonical_table_eligible"]
    assert not (output / "main_table.md").exists()
    assert verified == [True]


def test_matrix_audit_classifies_natural_numerical_failure_without_implementation_failure(tmp_path, monkeypatch):
    from tools.xag import assess_matrix as M
    environment(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "ROOT", tmp_path)
    complete_rows(tmp_path, numerical=True)
    audit = M.MatrixAudit.__new__(M.MatrixAudit)
    audit.missing = []; audit.errors = []; audit.numerical = []
    actual = audit.main_rows()
    assert actual["effects"] is None and not actual["canonical_table_eligible"]
    assert len(actual["source_rows"]) == len(actual["row_audits"]) == 12
    assert not audit.errors and not audit.missing
    assert M.final_status(audit.missing, audit.errors, audit.numerical) == "COMPLETE_WITH_NUMERICAL_FAILURES"
    receipt = tmp_path / "outputs/xag/main_row_audits/xag_a1f1.json"
    receipt.unlink()
    other = M.MatrixAudit.__new__(M.MatrixAudit)
    other.missing = []; other.errors = []; other.numerical = []
    other.main_rows()
    assert M.final_status(other.missing, other.errors, other.numerical) == "IMPLEMENTATION_FAIL"


def test_finite_natural_matrix_still_calculates_effects_and_bad_values_cannot_enter_effects(tmp_path, monkeypatch):
    from tools.xag import assess_matrix as M
    environment(tmp_path, monkeypatch)
    monkeypatch.setattr(M, "ROOT", tmp_path)
    complete_rows(tmp_path, numerical=False)
    audit = M.MatrixAudit.__new__(M.MatrixAudit)
    audit.missing = []; audit.errors = []; audit.numerical = []
    actual = audit.main_rows()
    assert actual["canonical_table_eligible"] and actual["effects"] is not None
    assert not audit.errors and not audit.missing and not audit.numerical
    rows = {c: row() for c in F.CELLS}; repeats = {c: row((3407,)) for c in F.CELLS}
    rows["a2f0"] = row(bad=float("nan"))
    with pytest.raises(ValueError, match="nonfinite natural"):
        F.effects(rows, repeats)
