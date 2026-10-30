"""Missingness, attribution arithmetic and exact matrix artifact identities."""
import copy
import json

import numpy as np
import pytest

from tools.xag import assess_matrix as M
from tools.xag import finalize as F


def test_effects_match_registered_formula_including_interaction_sign_and_repeat_floor():
    values = {"a0f0": [20., 22.], "a0f1": [17., 18.], "a1f0": [15., 16.],
              "a1f1": [10., 10.], "a2f0": [21., 20.], "a2f1": [19., 17.]}
    repeats = {c: v[0] + .3 for c, v in values.items()}; repeats["a2f1"] += .4
    actual = M.factor_effect(values, repeats)
    rows, repeat_rows = {}, {}
    for cell in M.CELLS:
        rows[cell] = {"per_seed": {str(seed): {dom: {metric: values[cell][j] for metric in M.METRICS}
                           for dom in M.DOMAINS} for j, seed in enumerate((3407, 3408))}}
        repeat_rows[cell] = {"per_seed": {"3407": {dom: {metric: repeats[cell] for metric in M.METRICS} for dom in M.DOMAINS}}}
    reference = F.effects(rows, repeat_rows)["local"]["mpjpe_ra_mm"]
    assert actual["tau_mm"] == pytest.approx(1.4)
    assert actual["tau_mm"] == reference["tau_mm"]
    assert actual["gnn_at_base"]["per_seed_mm"] == reference["gnn_at_base"]
    for a in ("1", "2"):
        for key in ("association_at_legacy", "association_at_global", "gnn_at_oracle", "interaction"):
            assert actual["associations"][a][key]["per_seed_mm"] == reference["associations"][a][key]
    assert actual["associations"]["1"]["interaction"]["per_seed_mm"] == [2., 2.]
    assert actual["associations"]["1"]["coupled_above_noise"]


def test_effects_never_drop_failed_seed_or_repeat_and_keep_seed_counterexample():
    values = {c: [10., 10.] for c in M.CELLS}; repeats = {c: 10. for c in M.CELLS}
    values["a0f1"] = [8., 11.]
    actual = M.factor_effect(values, repeats)
    assert actual["gnn_at_base"]["seed_sign_disagreement"]
    assert not actual["gnn_at_base"]["gain_gate"]
    repeats["a2f1"] = None
    assert M.factor_effect(values, repeats)["status"] == "UNDEFINED_NUMERICAL_FAILURE"
    with pytest.raises(ValueError, match="six-cell"):
        M.factor_effect({k: v for k, v in values.items() if k != "a1f1"}, repeats)


def test_status_never_claims_complete_with_missing_or_implementation_failure():
    assert M.final_status(["missing"], [], []) == "INCOMPLETE"
    assert M.final_status(["missing"], ["bad_identity"], []) == "IMPLEMENTATION_FAIL"
    assert M.final_status([], [], ["nan_endpoint"]) == "COMPLETE_WITH_NUMERICAL_FAILURES"
    assert M.final_status([], [], []) == "COMPLETE"


def test_common_failure_ledger_keeps_gt_condition_and_separates_shared_reference():
    pred = np.zeros((15, 3, 51)); incoming = pred.copy()
    incoming[2, 1] = np.nan; pred[2, 1] = np.nan; pred[14, 2] = np.nan
    failures = [{"row": 2, "step": 1, "reason": "shared_reference_failed"},
                {"row": 14, "step": 2, "reason": "FloatingPointError"}]
    result = M.check_failure_ledger(pred, failures, "common", incoming)
    assert result == {"expected_endpoints": 45, "failed_endpoints": 2,
                      "shared_reference_failed_endpoints": 1, "new_model_failed_endpoints": 1}
    with pytest.raises(ValueError, match="ledger"):
        M.check_failure_ledger(pred, failures[:1], "common", incoming)
    with pytest.raises(ValueError, match="Duplicate"):
        M.check_failure_ledger(pred, failures + failures[:1], "common", incoming)
    wrong = copy.deepcopy(failures); wrong[0]["reason"] = "FloatingPointError"
    with pytest.raises(ValueError, match="attributed"):
        M.check_failure_ledger(pred, wrong, "common", incoming)
    wrong = copy.deepcopy(failures); wrong[1]["reason"] = "unexecuted_after_prior_failure"
    with pytest.raises(ValueError, match="own-loop"):
        M.check_failure_ledger(pred, wrong, "common", incoming)


def test_own_failure_cannot_reset_or_be_hidden_as_infrastructure_nan():
    pred = np.zeros((2, 4, 51)); pred[1, 1:] = np.nan
    failures = [{"trajectory": 1, "step": k, "reason": "FloatingPointError" if k == 1 else "unexecuted_after_prior_failure"}
                for k in (1, 2, 3)]
    result = M.check_failure_ledger(pred, failures, "own")
    assert result["expected_endpoints"] == 8 and result["failed_endpoints"] == 3
    wrong = copy.deepcopy(failures); wrong[0]["reason"] = "unexecuted_after_prior_failure"
    with pytest.raises(ValueError, match="First own"):
        M.check_failure_ledger(pred, wrong, "own")
    pred[1, 3] = 0
    with pytest.raises(ValueError, match="restarted"):
        M.check_failure_ledger(pred, failures[:2], "own")
    pred[1, 3] = np.nan; failures[0]["reason"] = "RuntimeError"
    with pytest.raises(ValueError, match="implementation"):
        M.check_failure_ledger(pred, failures, "own")


def test_curve_has_eligible_horizon_denominator_and_no_survivor_mean():
    result = M.curve_summary([(np.array([100, 149, 199]), np.array([1., 2., 3.])),
                             (np.array([400, 449]), np.array([1., np.nan]))])
    assert result["elapsed_ms"] == [0, 49, 99]
    assert result["n_expected"] == [2, 2, 1]
    assert result["n_failed"] == [0, 1, 0]
    assert result["mean"] == [1., None, 3.]
    assert M.summary([1., np.nan]) == {"n": 2, "n_failed": 1, "mean": None}
    compact = M.curve_summary([(np.array([0, 49, 99, 149]), np.array([1., np.nan, 2., 3.]))], milestones=(0, 49))
    assert compact["elapsed_ms"] == [0, 49, 149] and compact["full_curve_timepoints"] == 4
    assert compact["mean"] == [1., None, 3.]  # include final, never select by score/finite status


@pytest.mark.parametrize("mutation", ["wrong_config", "wrong_checkpoint", "wrong_source", "wrong_common_reference", "partial", "wrong_spec"])
def test_artifact_rejects_wrong_provenance_and_common_reference(tmp_path, mutation):
    audit = M.MatrixAudit.__new__(M.MatrixAudit)
    audit.ph = "diagnostic_protocol"; audit.training_protocol_sha = "training_protocol"
    audit.code = {"common": {"code": "stable"}}
    audit.source_id = {"source_manifest_sha256": "source_sha"}
    expected = {"checkpoint": str(tmp_path / "step6000.ckpt"), "checkpoint_sha256": "checkpoint_sha",
                "config": str(tmp_path / "config.yaml"), "config_sha256": "config_sha",
                "registered_config_sha256": "registered_config_sha"}
    audit.run_records = {"run": expected}
    ident = {**expected, "phase": "trained_common_replay", "limit": None, "global_step": 6000,
             "model_class": "MNISTModel", "protocol_sha256": audit.ph,
             "training_protocol_sha256": audit.training_protocol_sha, "source_hashes": audit.code["common"],
             "source_manifest_sha256": "source_sha", "sequence": "zgz_local", "run_id": 0,
             "reference_sha256": "same_incoming"}
    baseline = {"sha256": "same_incoming", "specs": [{"block": "clean"}]}
    path = tmp_path / "raw.npz"; np.savez(path, x=np.zeros(1))
    meta = {"identity": ident, "cache_sha256": M.digest(path), "specs": baseline["specs"]}
    path.with_suffix(".json").write_text(json.dumps(meta))
    audit.artifact(path, "common", "run", "zgz_local", 0, baseline)
    if mutation == "wrong_config": meta["identity"]["config_sha256"] = "different"
    elif mutation == "wrong_checkpoint": meta["identity"]["checkpoint_sha256"] = "different"
    elif mutation == "wrong_source": meta["identity"]["source_manifest_sha256"] = "different"
    elif mutation == "wrong_common_reference": meta["identity"]["reference_sha256"] = "different"
    elif mutation == "partial": meta["identity"]["limit"] = 2
    elif mutation == "wrong_spec": meta["specs"] = [{"block": "natural"}]
    path.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        audit.artifact(path, "common", "run", "zgz_local", 0, baseline)


@pytest.mark.parametrize("mutation", ["valid", "incoming", "metric", "shape"])
def test_raw_common_states_are_recomputed_with_real_mano_and_exact_incoming(tmp_path, mutation):
    import torch
    from mano_layer import ManoLayer
    from tools.xdl import diagnostics as D
    old_threads = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        mano = ManoLayer(str(M.ROOT / "assets/mano_right.npz"), add_mean=False).eval()
        mano.fast_chain = True
        truth = np.zeros((3, 51), np.float32); truth[:, 2] = .5
        baseline = {"states": np.repeat(truth[None], 14, 0), "truth": truth,
                    "times": np.array([0, 49, 99]), "specs": [{"block": str(i)} for i in range(14)]}
        pred = baseline["states"][:, 1:].copy(); pred[0, :, 2] += .005
        gt = truth[1:].copy(); beta = torch.zeros(1, 10)
        all_pred = np.concatenate((pred, gt[None]), 0)
        with torch.no_grad():
            error = D.endpoint(mano, all_pred, truth[None, 1:], beta, torch.device("cpu"), mano.hands_mean.numpy())[0]
        arrays = {"common51": pred, "gt51": gt, "reference51": baseline["states"].copy(),
                  "truth51": truth, "times_ms": baseline["times"],
                  **{"common_" + key: value[:14].copy() for key, value in error.items()},
                  **{"gt_" + key: value[14].copy() for key, value in error.items()}}
        if mutation == "incoming": arrays["reference51"][7, 0, 0] += .01
        elif mutation == "metric": arrays["common_mpjpe_abs_mm"][0, 0] += 2.
        elif mutation == "shape": arrays["common51"] = pred[:13]
        path = tmp_path / "sample.npz"; np.savez(path, **arrays)
        audit = M.MatrixAudit.__new__(M.MatrixAudit)
        audit.data = {"zgz_local": {"aux": {"betas": np.zeros(10)}, "gt": np.repeat(truth[:1], 100, 0)}}
        audit.mano = lambda: mano
        audit.artifact = lambda *args: ({"expected_common_endpoints": 28, "expected_gt_endpoints": 2, "failures": []}, {})
        invoke = lambda: audit.one("run", "zgz_local", 0, 0,
                                  [{"previous_end_ms": 0}, {"previous_end_ms": 49}], "common", path, baseline)
        if mutation != "valid":
            with pytest.raises(ValueError): invoke()
        else:
            record, result = invoke()
            assert record["expected_endpoints"] == 30 and record["failed_endpoints"] == 0
            assert result["errors"]["mpjpe_abs_mm"].shape == (15, 2)
            assert result["errors"]["mpjpe_abs_mm"][0].mean() == pytest.approx(5., abs=.001)
            assert result["errors"]["mpjpe_abs_mm"][14].mean() == 0
    finally:
        torch.set_num_threads(old_threads)
