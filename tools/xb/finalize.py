#!/usr/bin/env python3
"""Apply the frozen ABCD rules to completed artifacts; never infer missing gates.

This is analysis-only postprocessing. It neither changes frozen candidates nor
starts training/evaluation. The displayed primary table is read verbatim from
TABLE_full.md; diagnostic numbers are kept in the verdict artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

REPO = Path(__file__).resolve().parents[2]
SECTION = "XYZ-RESIDUAL-ABCD-20261011"
SEEDS = (3407, 3408, 3409)
ARMS = "ABCD"
CONDITIONS = ("natural", "clean", "small", "large")
NUMERIC_ANNOTATIONS = frozenset(("numeric_valid", "numeric_status", "first_nonfinite", "invalid_numeric_count",
                                 "invalid_numeric_paths", "invalid_known_count", "invalid_known_paths"))
INVARIANTS = ("common_slots_equal", "residual_mass_equal", "residual_count_equal", "metadata_equal",
              "tok_equal", "residual_weights_equal", "permutation_preserves_stratum",
              "d_common_slots_equal", "d_metadata_equal", "d_mass_equal", "d_count_equal", "d_content_zero")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def json_safe(value):
    """Preserve an explicit nonfinite marker instead of dropping a failed seed."""
    if isinstance(value, float) and not math.isfinite(value):
        return f"NONFINITE:{value}"
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def get(value, *keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def has(value, *keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return False
        value = value[key]
    return True


def measured_failure(value):
    return isinstance(value, dict) and (value.get("numeric_valid") is False or value.get("numeric_status") == "FAIL")


def mandatory_numeric_failure(value):
    """An added, nonconfirmatory r1 control cannot become a new victory gate."""
    if not measured_failure(value):
        return False
    paths = []
    for key in ("invalid_numeric_paths", "invalid_known_paths"):
        if isinstance(value.get(key), list):
            paths.extend(path for path in value[key] if isinstance(path, str))
    if not paths and isinstance(value.get("first_nonfinite"), str):
        paths.append(value["first_nonfinite"])
    return not paths or not all("supplementary_r1_exchange" in path for path in paths)


def subtree_numeric_failure(value):
    """Inspect only an original required subtree, never its propagated parent flags."""
    if isinstance(value, dict):
        return measured_failure(value) or any(subtree_numeric_failure(item) for key, item in value.items()
                                              if key not in NUMERIC_ANNOTATIONS)
    if isinstance(value, list):
        return any(subtree_numeric_failure(item) for item in value)
    return isinstance(value, float) and not math.isfinite(value)


def gate(checks, missing=(), details=None):
    """Missing required values always keep a gate pending, even with partial failures."""
    absent = list(missing)
    failures = [name for name, passed in checks.items() if passed is False]
    unknown = [name for name, passed in checks.items() if passed is None]
    absent += unknown
    return {"status": "pending" if absent else "fail" if failures else "pass",
            "missing": sorted(set(absent)), "violations": failures, "checks": checks,
            "details": details or {}}


def read(path, artifacts):
    if not path.exists():
        artifacts[str(path)] = {"present": False}
        return None
    artifacts[str(path)] = {"present": True, "sha256": digest(path)}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        artifacts[str(path)]["read_error"] = str(error)
        return None


def preregistration(out, artifacts):
    manifest = read(out / "frozen_manifest.json", artifacts)
    if manifest is None:
        return gate({}, ["frozen_manifest.json"])
    text = get(manifest, "prereg_text")
    registered = get(manifest, "prereg_sha256")
    if not isinstance(text, str) or not isinstance(registered, str):
        return gate({}, ["frozen preregistration text/SHA256"])
    docs = REPO / "docs/EXPERIMENT_RECORDS.md"
    source = docs.read_text()
    begin, end = (f"<!-- EXPERIMENT_RECORD:{SECTION}:{kind} -->" for kind in ("BEGIN", "END"))
    current = source.split(begin, 1)[1].split(end, 1)[0] if source.count(begin) == source.count(end) == 1 else None
    return gate({"correct_study": manifest.get("study") == SECTION,
                 "frozen_text_digest": hashlib.sha256(text.encode()).hexdigest() == registered,
                 "live_rule_section_unchanged": current == text}, details={"prereg_sha256": registered})


def primary_gates(source):
    missing, checks = [], {}
    if not isinstance(source, dict):
        pending = gate({}, ["gates.json"])
        return {name: pending for name in ("utility", "binding_removal", "D_explains_B", "D_utility")}
    paired, thresholds, noise = source.get("paired_gain_mm"), source.get("threshold_mm"), source.get("same_seed_repeat_abs_difference_mm")
    invalid = []
    if mandatory_numeric_failure(source):
        invalid.append("recorded numeric failure")
    for name in ("paired_gain_mm", "mean_gain_mm", "threshold_mm", "same_seed_repeat_abs_difference_mm", "binding_gain_removed_fraction"):
        def inspect(value, location):
            if isinstance(value, dict):
                for key, item in value.items():
                    inspect(item, location + "/" + str(key))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    inspect(item, location + "/" + str(index))
            elif isinstance(value, (float, int)) and not isinstance(value, bool) and not math.isfinite(value):
                invalid.append(location)
            elif value is None and name != "binding_gain_removed_fraction":
                invalid.append(location)
        if name in source:
            inspect(source[name], name)
    if invalid:
        failed = gate({"all_required_primary_gate_values_finite": False}, details={"nonfinite_fields": invalid,
                      "interpretation": "no seed is discarded; numerical failure forbids primary efficacy or causal adoption"})
        return {name: failed for name in ("utility", "binding_removal", "D_explains_B", "D_utility")}
    parsed = {}
    for arm in "bcd":
        parsed[arm] = {}
        for scope in ("overall", "local", "global"):
            values = get(paired, arm, scope)
            if not isinstance(values, list) or len(values) != 3 or not all(numeric(value) for value in values):
                missing.append(f"paired_gain_mm/{arm}/{scope}/three seeds")
            else:
                parsed[arm][scope] = values
                recorded = get(source, "mean_gain_mm", arm, scope)
                if not numeric(recorded):
                    missing.append(f"mean_gain_mm/{arm}/{scope}")
                else:
                    checks[f"recorded_mean_{arm}_{scope}"] = math.isclose(recorded, mean(values), rel_tol=1e-10, abs_tol=1e-10)
    for scope in ("overall", "local", "global"):
        threshold, delta = get(thresholds, scope), get(noise, scope)
        if not numeric(threshold) or not numeric(delta):
            missing.append(f"noise and threshold/{scope}")
        else:
            checks[f"threshold_{scope}"] = delta >= 0 and math.isclose(threshold, max(1.1, 2 * delta), rel_tol=1e-10, abs_tol=1e-10)
    checks["three_registered_seeds"] = source.get("seeds") == list(SEEDS) if "seeds" in source else None
    details = {"threshold_mm": thresholds, "same_seed_repeat_abs_difference_mm": noise,
               "paired_gain_mm": paired, "mean_gain_mm": source.get("mean_gain_mm")}
    if missing:
        pending = gate(checks, missing, details)
        return {name: pending for name in ("utility", "binding_removal", "D_explains_B", "D_utility")}
    def efficacy(arm):
        return (all(value > 0 for value in parsed[arm]["overall"]) and mean(parsed[arm]["overall"]) > thresholds["overall"]
                and all(mean(parsed[arm][scope]) >= -thresholds[scope] for scope in ("local", "global")))
    b_gain = mean(parsed["b"]["overall"])
    removal = (b_gain - mean(parsed["c"]["overall"])) / b_gain if b_gain > 0 else None
    recorded_removal = source.get("binding_gain_removed_fraction")
    utility = {**checks, "three_seed_B_improvement": all(value > 0 for value in parsed["b"]["overall"]),
               "mean_B_improvement_exceeds_noise": b_gain > thresholds["overall"],
               "local_global_degradation_within_noise": all(mean(parsed["b"][scope]) >= -thresholds[scope] for scope in ("local", "global")),
               "recorded_efficacy_matches": source.get("B_efficacy_gate") == efficacy("b") if isinstance(source.get("B_efficacy_gate"), bool) else None}
    binding = {**checks, "B_gain_positive": b_gain > 0, "half_gain_removed": removal is not None and removal >= .5,
               "recorded_removal_matches": ((recorded_removal is None and removal is None) or
                    (numeric(recorded_removal) and removal is not None and math.isclose(recorded_removal, removal, rel_tol=1e-10, abs_tol=1e-10)))
                    if "binding_gain_removed_fraction" in source else None,
               "recorded_binding_matches": source.get("binding_at_least_half_gate") == (removal is not None and removal >= .5)
                    if isinstance(source.get("binding_at_least_half_gate"), bool) else None}
    occupancy = all(abs(mean(parsed["b"][scope]) - mean(parsed["d"][scope])) <= thresholds[scope]
                    for scope in ("overall", "local", "global"))
    occupancy_checks = {**checks, "all_columns_D_within_noise_of_B": occupancy,
                        "recorded_occupancy_matches": source.get("D_explains_B_within_noise") == occupancy
                             if isinstance(source.get("D_explains_B_within_noise"), bool) else None}
    return {"utility": gate(utility, details=details), "binding_removal": gate(binding, details={**details, "removed_fraction": removal}),
            "D_explains_B": gate(occupancy_checks, details={"D_minus_B_error_mm": {scope: mean(parsed["b"][scope]) - mean(parsed["d"][scope])
                                                                                 for scope in ("overall", "local", "global")}, "threshold_mm": thresholds}),
            "D_utility": gate({**checks, "D_satisfies_same_utility_rule": efficacy("d")}, details=details)}


def common_replay(payload):
    missing, checks, by_arm = [], {}, {}
    if not isinstance(payload, dict):
        return gate({}, ["recovery artifact"])
    protocol = payload.get("protocol", {})
    for key, value in (("window_ms", 50), ("step_ms", 50), ("steps", 10)):
        checks[f"protocol_{key}"] = protocol[key] == value if key in protocol else None
    selected = protocol.get("selected_trace_indices")
    registered_selected = {seq: selected[seq] for seq in ("zgz_local", "zgz_global") if isinstance(selected, dict) and seq in selected}
    checks["48_preselected_clips"] = (isinstance(selected, dict) and set(selected) - NUMERIC_ANNOTATIONS == {"zgz_local", "zgz_global"}
                                      and all(isinstance(values, list) and len(values) == 24 and len(set(values)) == 24 for values in registered_selected.values())) \
                                      if selected is not None else None
    for arm in ARMS:
        rows = get(payload, "arms", arm, "rows")
        if not isinstance(rows, list):
            missing.append(f"{arm}/rows")
            continue
        checks[f"{arm}_192_rows"] = len(rows) == 192
        mapping = {}
        for index, row in enumerate(rows):
            required = ("seq", "clip_id", "condition", "run_id", "first_end_ms", "previous_end_ms", "n_raw_events", "initial_state_sha256")
            if any(key not in row for key in required):
                missing.append(f"{arm}/row{index}/clip or initial SHA identity")
                continue
            key = (row["seq"], row["clip_id"], row["condition"])
            if key in mapping:
                checks[f"{arm}_duplicate_row{index}"] = False
            if row["condition"] not in CONDITIONS or row["seq"] not in ("zgz_local", "zgz_global"):
                checks[f"{arm}_registered_row{index}"] = False
            checks[f"{arm}_row{index}_10_windows"] = isinstance(row["n_raw_events"], list) and len(row["n_raw_events"]) == 10
            checks[f"{arm}_row{index}_initial_sha"] = isinstance(row["initial_state_sha256"], str) and len(row["initial_state_sha256"]) == 64
            for metric in ("mpjpe_ra_mm", "mpjpe_abs_mm"):
                values = row.get(metric)
                # Null scores retain a measured failure, while their frame identity
                # and count still participate in the common-clip contract.
                checks[f"{arm}_row{index}_{metric}_11_states"] = isinstance(values, list) and len(values) == 11 \
                                                                  if values is not None else None
            mapping[key] = {name: row[name] for name in required if name not in ("seq", "clip_id", "condition")}
        by_arm[arm] = mapping
        for scope, count in (("all", 48), ("zgz_local", 24), ("zgz_global", 24)):
            for condition in CONDITIONS:
                summary = get(payload, "arms", arm, "summary", scope, condition)
                checks[f"{arm}_{scope}_{condition}_nclips"] = summary.get("n_clips") == count if isinstance(summary, dict) else None
    if len(by_arm) == 4:
        expected = {(seq, clip, condition) for seq in ("zgz_local", "zgz_global") for clip in range(24) for condition in CONDITIONS}
        checks["ABCD_same_registered_clip_set"] = all(set(mapping) == expected for mapping in by_arm.values())
        checks["ABCD_same_clip_and_initial_state_by_condition"] = all(mapping == by_arm["A"] for mapping in by_arm.values())
    identity = {"selected_trace_indices": registered_selected, "A_identities": [dict(key=list(key), **values) for key, values in sorted(by_arm.get("A", {}).items())]}
    identity_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return gate(checks, missing, {"common_identity_sha256": identity_sha, "selected_trace_indices": registered_selected})


def contract(payload):
    missing, checks, details = [], {}, {}
    if not isinstance(payload, dict):
        return gate({}, ["recovery artifact/control_audit"])
    for arm in ARMS:
        audit = get(payload, "arms", arm, "control_audit")
        if not isinstance(audit, dict):
            if mandatory_numeric_failure(get(payload, "arms", arm)):
                checks[f"{arm}/measured_control_export_failure"] = False
            else:
                missing.append(f"{arm}/control_audit")
            continue
        if measured_failure(audit):
            checks[f"{arm}/measured_control_numerics"] = False
            details[arm] = {"numeric_failure": True, "first_nonfinite": audit.get("first_nonfinite"),
                           "invalid_numeric_paths": audit.get("invalid_numeric_paths"), "invalid_known_paths": audit.get("invalid_known_paths"),
                           "scope": "measured diagnostic failure, not missing and not proof that C caused it"}
            continue
        checks[f"{arm}_audit_applicable"] = audit.get("applicable") if isinstance(audit.get("applicable"), bool) else None
        rows = audit.get("rows")
        if not isinstance(rows, list) or not rows:
            missing.append(f"{arm}/nonempty audit rows")
            continue
        checks[f"{arm}_recorded_invariants"] = get(audit, "summary", "all_common_invariants_pass") \
                                                if isinstance(get(audit, "summary", "all_common_invariants_pass"), bool) else None
        for index, row in enumerate(rows):
            for name in INVARIANTS:
                checks[f"{arm}/row{index}/{name}"] = row.get(name) if isinstance(row.get(name), bool) else False if name in row else None
            for name in ("residual_eligible_fraction", "residual_eligible_swap_fraction", "residual_r1_quality_diff", "a_r1_content_max_abs_change"):
                if not numeric(row.get(name)):
                    if name in row or measured_failure(row):
                        checks[f"{arm}/row{index}/{name}/measured_finite"] = False
                    else:
                        missing.append(f"{arm}/row{index}/{name}")
            if numeric(row.get("residual_eligible_fraction")) and numeric(row.get("residual_eligible_swap_fraction")):
                checks[f"{arm}/row{index}/eligible_changed_at_least_half"] = row["residual_eligible_fraction"] <= 0 or row["residual_eligible_swap_fraction"] >= .5
            if numeric(row.get("residual_r1_quality_diff")):
                checks[f"{arm}/row{index}/r1_quality_exactly_preserved"] = row["residual_r1_quality_diff"] == 0
            if numeric(row.get("a_r1_content_max_abs_change")):
                checks[f"{arm}/row{index}/A_global_invariant"] = row["a_r1_content_max_abs_change"] <= 1e-6
        details[arm] = {"summary": audit.get("summary"), "non_noop_pass": audit.get("non_noop_pass"),
                        "readout_uses_binding": audit.get("readout_uses_binding"),
                        "A_global_max_abs_change": max((row["a_r1_content_max_abs_change"] for row in rows if numeric(row.get("a_r1_content_max_abs_change"))), default=None)}
        if arm == "B":
            for name in ("non_noop_pass", "readout_uses_binding"):
                checks[f"B_{name}"] = audit.get(name) if isinstance(audit.get(name), bool) else None
            for name in ("packets_with_content_change", "packets_with_extra_change", "packets_with_prediction_change"):
                value = get(audit, "summary", name)
                checks[f"B_positive_{name}"] = value > 0 if numeric(value) else None
        if arm in "AB":
            control = get(payload, "arms", arm, "same_weight_inference_controls")
            expected = ("A", "A_r1_exchange") if arm == "A" else ("B", "C", "D")
            checks[f"{arm}_same_weight_full_loop_complete"] = (control.get("applicable") is True and control.get("same_weights") is True
                and isinstance(control.get("controls"), dict) and all(name in control["controls"] for name in expected)) if isinstance(control, dict) and not control.get("skipped") else None
            for mode in expected:
                result = get(control, "controls", mode)
                if not isinstance(result, dict):
                    missing.append(f"{arm}/same-weight/{mode} execution")
                    continue
                if subtree_numeric_failure(result):
                    checks[f"{arm}/{mode}/measured_full_loop_finite"] = False
                    continue
                sequences, aggregate = result.get("per_sequence"), result.get("aggregate")
                checks[f"{arm}/{mode}/two_required_sequences"] = (isinstance(sequences, list) and len(sequences) == 2
                    and all(isinstance(row, dict) for row in sequences)
                    and {row.get("seq") for row in sequences} == {"zgz_local", "zgz_global"}) if "per_sequence" in result else None
                for index, row in enumerate(sequences if isinstance(sequences, list) else []):
                    count = get(row, "n_frames")
                    checks[f"{arm}/{mode}/sequence{index}/frames_retained"] = numeric(count) and count > 0 if has(row, "n_frames") else None
                checks[f"{arm}/{mode}/aggregate_scopes"] = (isinstance(aggregate, dict)
                    and set(aggregate) - NUMERIC_ANNOTATIONS == {"all", "local", "global"}) if "aggregate" in result else None
                for scope in ("all", "local", "global"):
                    count = get(aggregate, scope, "n_frames")
                    checks[f"{arm}/{mode}/{scope}/frames_retained"] = numeric(count) and count > 0 if has(aggregate, scope, "n_frames") else None
    result = gate(checks, missing, {"audits": details,
        "eligible_threshold_scope": "fraction of sources changed among eligible records; no added threshold on fraction of all records eligible",
        "interpretation": "near strata exchange content-quality binding as well as position; r=1 preserves quality exactly; same-weight full loops are sensitivity only",
        "r1_evidence_boundary": "the frozen contract reports r1 crossing and exact quality preservation; the later B r1-only content/readout probe remains a separate nonconfirmatory supplement and cannot create a new formal victory gate"})
    if any(passed is False and ("measured" in name) for name, passed in checks.items()):
        result["status"] = "fail"
    return result


def recovery_gate(payloads, seeds):
    missing, checks, rows = [], {}, []
    for seed in seeds:
        payload = payloads.get(seed)
        for condition in ("natural", "large"):
            a = get(payload, "arms", "A", "summary", "all", condition, "recovery_auc_ra_mm")
            b = get(payload, "arms", "B", "summary", "all", condition, "recovery_auc_ra_mm")
            if a is None or b is None:
                for arm, value in (("A", a), ("B", b)):
                    if value is not None:
                        continue
                    if has(payload, "arms", arm, "summary", "all", condition, "recovery_auc_ra_mm") or subtree_numeric_failure(get(payload, "arms", arm, "summary")):
                        checks[f"seed{seed}_{condition}_{arm}_measured_finite_auc"] = False
                    else:
                        missing.append(f"seed{seed}/{condition}/{arm} recovery_auc_ra_mm not executed")
                rows.append({"seed": seed, "condition": condition, "A_auc_ra_mm": a, "B_auc_ra_mm": b,
                             "measured_numerical_failure": any(checks[key] is False for key in checks if key.startswith(f"seed{seed}_{condition}_"))})
                continue
            if not numeric(a) or not numeric(b):
                checks[f"seed{seed}_{condition}_finite_A_and_B_auc"] = False
                rows.append({"seed": seed, "condition": condition, "A_auc_ra_mm": a, "B_auc_ra_mm": b, "numerical_failure": True})
                continue
            checks[f"seed{seed}_{condition}_B_lower_than_A"] = b < a
            rows.append({"seed": seed, "condition": condition, "A_auc_ra_mm": a, "B_auc_ra_mm": b, "B_minus_A_mm": b-a})
    result = gate(checks, missing, {"per_seed": rows, "rule": "every paired seed passes separately in both natural and large; no condition cancellation"})
    if any(passed is False and ("finite" in name) for name, passed in checks.items()):
        result["status"] = "fail"
    return result


def motion_gate(payloads, seeds):
    missing, checks, comparisons, tradeoffs = [], {}, [], []
    for condition in ("clean", "small"):
        for scope in ("zgz_local", "zgz_global"):
            paired = []
            for seed in seeds:
                values = {arm: {metric: get(payloads.get(seed), "arms", arm, "summary", scope, condition, "motion", metric)
                                for metric in ("joint_velocity_gain", "joint_velocity_cosine", "velocity_lag_steps")} for arm in "AB"}
                if any(value is None for arm in values.values() for value in arm.values()):
                    for arm in "AB":
                        for metric, value in values[arm].items():
                            if value is not None:
                                continue
                            if has(payloads.get(seed), "arms", arm, "summary", scope, condition, "motion", metric) or subtree_numeric_failure(get(payloads.get(seed), "arms", arm, "summary")):
                                checks[f"seed{seed}_{scope}_{condition}_{arm}_{metric}_measured_finite"] = False
                            else:
                                missing.append(f"seed{seed}/{scope}/{condition}/{arm}/{metric} not executed")
                    continue
                if not all(numeric(value) for arm in values.values() for value in arm.values()):
                    checks[f"seed{seed}_{scope}_{condition}_finite_motion"] = False
                    continue
                paired.append({"seed": seed, "A": values["A"], "B": values["B"],
                               "gain_bias_degradation": abs(values["B"]["joint_velocity_gain"]-1)-abs(values["A"]["joint_velocity_gain"]-1),
                               "cosine_degradation": values["A"]["joint_velocity_cosine"]-values["B"]["joint_velocity_cosine"],
                               "lag_increase_steps": values["B"]["velocity_lag_steps"]-values["A"]["velocity_lag_steps"]})
            if len(paired) != len(seeds):
                continue
            averages = {metric: mean(row[metric] for row in paired) for metric in ("gain_bias_degradation", "cosine_degradation", "lag_increase_steps")}
            limits = {"gain_bias_degradation": .10, "cosine_degradation": .05, "lag_increase_steps": .5}
            for metric, tolerance in limits.items():
                checks[f"{scope}_{condition}_{metric}"] = averages[metric] <= tolerance
                if 0 < averages[metric] <= tolerance and all(row[metric] > 0 for row in paired):
                    tradeoffs.append({"scope": scope, "condition": condition, "metric": metric, "mean_degradation": averages[metric], "tolerance": tolerance})
            comparisons.append({"scope": scope, "condition": condition, "per_seed": paired, "three_seed_mean_degradation": averages})
    result = gate(checks, missing, {"comparisons": comparisons, "consistent_small_tradeoffs": tradeoffs,
        "motion_tradeoff_detected": bool(tradeoffs), "interpretation": "engineering tolerances, not significance tests; gain bias is averaged after abs(gain-1) per seed"})
    if any(passed is False and ("finite" in name) for name, passed in checks.items()):
        result["status"] = "fail"
    return result


def streams(out, artifacts):
    checks, missing, records = {}, [], []
    all_payload = None
    for phase, seeds in (("pilot", SEEDS[:2]), ("full", SEEDS)):
        payload = read(out / f"paired_streams_{phase}.json", artifacts)
        if payload is None:
            if all_payload is None:
                all_payload = read(out / "paired_streams_all.json", artifacts)
            payload = all_payload
        groups = get(payload, "groups")
        if not isinstance(groups, list):
            missing.append(f"paired_streams/{phase}")
            continue
        for seed in seeds:
            matches = [row for row in groups if row.get("phase") == phase and row.get("seed") == seed]
            if len(matches) != 1:
                missing.append(f"paired_streams/{phase}/seed{seed}")
                continue
            row = matches[0]
            prefix = "xb_p" if phase == "pilot" else "xb_"
            expected = [f"{prefix}{arm.lower()}_s{seed}" for arm in ARMS]
            if phase == "full" and seed == 3407:
                expected.append("xb_ar_s3407")
            source = row.get("streams")
            checks[f"{phase}_{seed}_reported_pass"] = row.get("initial_three_batches_identical") if isinstance(row.get("initial_three_batches_identical"), bool) else None
            if not isinstance(source, dict) or any(name not in source for name in expected):
                missing.append(f"paired_streams/{phase}/seed{seed}/expected ABCD and repeat streams")
                continue
            checks[f"{phase}_{seed}_three_batches"] = all(isinstance(source[name], list) and len(source[name]) == 3 for name in expected)
            checks[f"{phase}_{seed}_recomputed_equal"] = all(source[name] == source[expected[0]] for name in expected)
            records.append({"phase": phase, "seed": seed, "reported_pass": row.get("initial_three_batches_identical")})
    return gate(checks, missing, {"groups": records, "scope": "only initial three batches; not proof of whole training stream equality"})


def capacity_context(source):
    if not isinstance(source, dict):
        return gate({}, ["control_capacity.json"])
    checks, missing = {}, []
    for scope in ("all", "r1"):
        actual = get(source, "summary", scope, "actual_cross_fraction")
        upper = get(source, "summary", scope, "feasible_cross_upper_fraction")
        utilization = get(source, "summary", scope, "actual_over_feasible_upper")
        if not all(numeric(value) for value in (actual, upper, utilization)):
            missing.append(f"control_capacity/{scope}/actual upper utilization")
        else:
            checks[f"{scope}_actual_within_physical_upper"] = 0 <= actual <= upper <= 1 and 0 <= utilization <= 1
    return gate(checks, missing, {"summary": source.get("summary"), "first64_context": source.get("first64_context"),
        "interpretation": "diagnostic physical limits, not a new adoption threshold; weighted actual and count upper have different denominators; low C/B contrast cannot rule out all local information"})


def phase_numerics(phase, artifacts):
    """Audit every required arm/seed, even if gate_summary masked NaN in max()."""
    missing, checks, invalid, rows = [], {}, [], []
    seeds = SEEDS[:2] if phase == "pilot" else SEEDS
    prefix = "xb_p" if phase == "pilot" else "xb_"
    jobs = [(prefix + arm.lower(), seeds) for arm in ARMS]
    if phase == "full":
        jobs.append(("xb_ar", (3407,)))
    for run, expected_seeds in jobs:
        source = read(REPO / f"outputs/semkine/{run}_main_row_last.json", artifacts)
        if source is None:
            missing.append(f"{run}_main_row_last.json")
            continue
        for seed in expected_seeds:
            seed_row = get(source, "per_seed", str(seed))
            if not isinstance(seed_row, dict):
                missing.append(f"{run}/required seed{seed}")
                continue
            expected_step = 1500 if phase == "pilot" else 6000
            checks[f"{run}/{seed}/exact_step"] = seed_row.get("step") == expected_step if "step" in seed_row else None
            failed_fields = []
            for scope in ("overall", "local", "global"):
                for metric in ("mpjpe_ra_mm", "mpjpe_abs_mm", "mpvpe_ra_mm", "mpvpe_abs_mm"):
                    label = f"{run}/seed{seed}/{scope}/{metric}"
                    value = get(seed_row, scope, metric)
                    if value is None and not has(seed_row, scope, metric):
                        missing.append(label)
                    elif not numeric(value):
                        checks[label + "/finite"] = False
                        failed_fields.append(label)
                        invalid.append(label)
                    else:
                        checks[label + "/finite"] = True
            rows.append({"run": run, "seed": seed, "nonfinite_or_invalid_fields": failed_fields,
                         "retained_in_required_seed_set": True})
    result = gate(checks, missing, {"per_required_seed": rows, "nonfinite_fields": invalid,
                                  "any_required_seed_nonfinite": bool(invalid),
                                  "interpretation": "failed seeds stay required; no finite-only averaging or substitution of an earlier checkpoint"})
    if invalid:
        result["status"] = "fail"
    return result


def diagnostic_numerics(payloads, seeds):
    checks, missing, details = {}, [], {}
    for seed in seeds:
        payload = payloads.get(seed)
        if not isinstance(payload, dict):
            missing.append(f"seed{seed}/diagnostic artifact not executed")
            continue
        complete = all(isinstance(get(payload, "arms", arm, "rows"), list) and len(get(payload, "arms", arm, "rows")) == 192 for arm in ARMS)
        checks[f"seed{seed}_all_arms_measured"] = complete
        details[str(seed)] = {key: payload.get(key) for key in ("numeric_valid", "numeric_status", "first_nonfinite",
                                                              "invalid_numeric_count", "invalid_numeric_paths", "invalid_known_paths")}
        details[str(seed)]["propagated_parent_flags_descriptive_only"] = True
        for arm in ARMS:
            entry = get(payload, "arms", arm)
            if not isinstance(entry, dict):
                missing.append(f"seed{seed}/{arm}/diagnostic execution")
                continue
            scopes = {name: entry.get(name) for name in ("rows", "summary", "control_audit")}
            if arm in "AB":
                controls = get(entry, "same_weight_inference_controls", "controls")
                for name in (("A", "A_r1_exchange") if arm == "A" else ("B", "C", "D")):
                    scopes[f"same_weight_inference_controls/controls/{name}"] = get(controls, name)
            for name, subtree in scopes.items():
                if subtree is None:
                    missing.append(f"seed{seed}/{arm}/{name} not executed")
                else:
                    checks[f"seed{seed}_{arm}_{name}_numeric_valid"] = not subtree_numeric_failure(subtree)
            rows = entry.get("rows")
            for index, row in enumerate(rows if isinstance(rows, list) else []):
                for metric in ("mpjpe_ra_mm", "mpjpe_abs_mm"):
                    values = get(row, metric)
                    if values is None:
                        checks[f"seed{seed}_{arm}_row{index}_{metric}_measured_finite"] = False
                    elif not isinstance(values, list) or not all(numeric(value) for value in values):
                        checks[f"seed{seed}_{arm}_row{index}_{metric}_measured_finite"] = False
            details[str(seed)][arm] = {key: entry.get(key) for key in ("numeric_valid", "numeric_status", "first_nonfinite",
                                                                    "invalid_numeric_count", "invalid_numeric_paths", "invalid_known_paths")}
    result = gate(checks, missing, {"per_seed": details,
        "interpretation": "numeric_valid=False and null measured scores in original rows/recovery/audits/A-B controls are FAIL; retained frames/clips/seeds are not discarded or reaveraged; supplementary_r1_exchange and its propagated parent flags remain descriptive only"})
    if any(passed is False and ("numeric_valid" in name or "finite" in name) for name, passed in checks.items()):
        result["status"] = "fail"
    return result


def render(verdict, table):
    lines = [table.rstrip(), ""] if table is not None else ["完整主表尚未生成；裁决保持待验收。", ""]
    descriptions = {"pending": "待验收：仍有缺失产物或门值，尚不作研究采纳裁决。",
                    "invalid_contract": "实验合同未通过，当前结果不能用于研究采纳。",
                    "invalid_full_numerics": "完整训练必需种子存在非有限闭环误差，主效用判失败；保留失败种子，不能作机制采纳或因果解释。",
                    "stop_current_expansion": "停止扩大本轮残差网络；这不否定整个 XYZ 方向。",
                    "narrow_to_occupancy": "D 已在噪声内解释 B；研究主张收窄为空间占用／质量读出。",
                    "support_further_research": "联合门支持继续研究传感器空间证据与 MANO 可重新查询的关联机制。"}
    lines += [descriptions[verdict["decision"]], "S39 默认模型与部署权重保持原状。",
              "本轮验证固定 W50 事件包中的空间残差信息；尚未实现或验证逐事件异步 MANO 状态更新，也不构成该机制的新颖性证明。", ""]
    if verdict["pilot_numerics"]["status"] == "fail":
        lines += ["轻量信息探针出现非有限闭环误差，记为 FAIL；其失败不替代或阻塞从零完整训练的机制裁决。", ""]
    for name, value in verdict["gates"].items():
        lines.append(f"- {name}: {value['status']}")
        if value["missing"]:
            lines.append("  缺失：" + "; ".join(value["missing"][:8]) + (" …" if len(value["missing"]) > 8 else ""))
    motion = verdict["gates"]["normal_motion"]["details"]
    if motion.get("motion_tradeoff_detected"):
        lines += ["", "正常响应虽在预定容差内，仍有三 seed 一致的小幅代价，不能声称没有运动代价。"]
    lines += ["", "C 的 near 分层可能同时改变内容与质量绑定；严格 r=1 子集保持质量。控制强度应结合物理跨格可行上界解释。同权重替换只作为分布外敏感性证据。", "",
              "恢复与正常响应诊断：", "", "```json", json.dumps({"recovery": verdict["gates"]["recovery"]["details"],
                  "normal_motion": motion, "binding": verdict["gates"]["binding_removal"]["details"].get("removed_fraction")}, ensure_ascii=False, indent=2), "```", "",
              "完整合同、控制强度、逐 seed 差及输入 SHA256 见同目录 verdict.json；原始曲线和首回指标保留在 recovery JSON。"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=REPO / "outputs/xb")
    args = parser.parse_args()
    out, artifacts = args.directory, {}
    primary = primary_gates(read(out / "gates.json", artifacts))
    numerics = {phase: phase_numerics(phase, artifacts) for phase in ("pilot", "full")}
    payloads, replay, contracts = {}, {}, {}
    for phase, seeds in (("pilot", SEEDS[:2]), ("full", SEEDS)):
        payloads[phase] = {}
        for seed in seeds:
            label = f"{phase}_s{seed}"
            value = read(out / f"recovery_{label}.json", artifacts)
            payloads[phase][seed] = value
            replay[label], contracts[label] = common_replay(value), contract(value)
            if isinstance(value, dict):
                for arm in ARMS:
                    checkpoint = get(value, "arms", arm, "checkpoint")
                    expected_step = 1500 if phase == "pilot" else 6000
                    replay[label]["checks"][f"{arm}_exact_final_checkpoint"] = f"step={expected_step}.ckpt" in checkpoint if isinstance(checkpoint, str) else None
                replay[label] = gate(replay[label]["checks"], replay[label]["missing"], replay[label]["details"])
            if phase == "pilot" and numerics["pilot"]["status"] == "fail" and (replay[label]["status"] == "pending" or contracts[label]["status"] == "pending"):
                explanation = {"numerical_failure": True, "recovery_export_unavailable": True,
                    "interpretation": "pilot closed-loop numerical failure is retained as FAIL; it cannot diagnose C causality and does not block full training"}
                replay[label] = gate({"pilot_recovery_finite_and_exportable": False}, details=explanation)
                contracts[label] = gate({"pilot_control_export_finite_and_complete": False}, details=explanation)
    replay_checks = {label: value["status"] == "pass" if value["status"] != "pending" else None for label, value in replay.items() if label.startswith("full_")}
    full_replay = [value for label, value in replay.items() if label.startswith("full_")]
    identities = [value["details"].get("common_identity_sha256") for value in full_replay]
    replay_checks["three_full_seeds_same_clips_and_initial_states"] = (len(set(identities)) == 1 if all(identities)
        else False if any(value["status"] == "fail" for value in full_replay) else None)
    contract_checks = {label: value["status"] == "pass" if value["status"] != "pending" else None for label, value in contracts.items() if label.startswith("full_")}
    d_comparison_contract = gate({key: value for key, value in primary["D_explains_B"]["checks"].items()
        if key != "all_columns_D_within_noise_of_B"}, primary["D_explains_B"]["missing"],
        {"interpretation": "an invalid recorded D contrast cannot be interpreted as evidence that D does not explain B"})
    gates = {"frozen_rules": preregistration(out, artifacts), "primary_utility": primary["utility"],
             "binding_removal": primary["binding_removal"], "D_explains_B": primary["D_explains_B"],
             "D_utility": primary["D_utility"], "D_comparison_contract": d_comparison_contract,
             "full_required_seed_numerics": numerics["full"], "paired_initial_streams": streams(out, artifacts),
             "common_replay_contract": gate(replay_checks, details={"per_artifact": replay}),
             "C_and_A_control_contract": gate(contract_checks, details={"per_artifact": contracts}),
             "control_strength_context": capacity_context(read(out / "control_capacity.json", artifacts)),
             "full_diagnostic_numerics": diagnostic_numerics(payloads["full"], SEEDS),
             "recovery": recovery_gate(payloads["full"], SEEDS), "normal_motion": motion_gate(payloads["full"], SEEDS)}
    table_path = out / "TABLE_full.md"
    table = table_path.read_text() if table_path.exists() else None
    artifacts[str(table_path)] = {"present": table is not None, "sha256": digest(table_path) if table is not None else None}
    required = ("frozen_rules", "full_required_seed_numerics", "full_diagnostic_numerics", "primary_utility", "binding_removal", "paired_initial_streams", "common_replay_contract",
                "C_and_A_control_contract", "control_strength_context", "recovery", "normal_motion", "D_explains_B", "D_utility", "D_comparison_contract")
    pending = [name for name in required if gates[name]["status"] == "pending"]
    if table is None:
        pending.append("TABLE_full.md")
    integrity = ("frozen_rules", "full_required_seed_numerics", "paired_initial_streams", "common_replay_contract", "D_comparison_contract")
    if numerics["full"]["details"].get("any_required_seed_nonfinite"):
        decision = "invalid_full_numerics"
    elif gates["primary_utility"]["status"] == "fail" and gates["full_required_seed_numerics"]["status"] == "pass":
        decision = "stop_current_expansion"
    elif pending:
        decision = "pending"
    elif any(gates[name]["status"] == "fail" for name in integrity):
        decision = "invalid_contract"
    elif gates["primary_utility"]["status"] == "fail":
        decision = "stop_current_expansion"
    elif gates["D_explains_B"]["status"] == "pass":
        decision = "narrow_to_occupancy"
    elif all(gates[name]["status"] == "pass" for name in ("binding_removal", "C_and_A_control_contract", "control_strength_context", "full_diagnostic_numerics", "recovery", "normal_motion")):
        decision = "support_further_research"
    else:
        decision = "stop_current_expansion"
    pilot_recovery = recovery_gate(payloads["pilot"], SEEDS[:2])
    if pilot_recovery["status"] == "pending" and numerics["pilot"]["status"] == "fail":
        pilot_recovery = gate({"pilot_recovery_finite_and_exportable": False}, details={
            "numerical_failure": True, "recovery_export_unavailable": True,
            "missing_export_fields": pilot_recovery["missing"], "no_C_causal_inference": True})
    verdict = {"study": SECTION, "analysis_only": True, "decision": decision,
               "research_adoption": decision == "support_further_research", "pending": pending, "gates": gates,
               "pilot_numerics": numerics["pilot"], "pilot_replay_descriptive_only": {label: value for label, value in replay.items() if label.startswith("pilot_")},
               "pilot_control_contract_descriptive_only": {label: value for label, value in contracts.items() if label.startswith("pilot_")},
               "pilot_diagnostic_numerics_descriptive_only": diagnostic_numerics(payloads["pilot"], SEEDS[:2]),
               "pilot_recovery_descriptive_only": pilot_recovery,
               "inputs": artifacts, "limitations": ["zgz is both development and test under the user's fixed protocol",
                    "one repeat difference defines an engineering noise floor, not a confidence interval",
                    "near-quality C is not a pure position intervention; r=1 is quality preserving",
                    "normal-motion tolerances and finite-clip recovery direction are not significance tests",
                    "fixed W50 spatial residual validation does not validate per-event asynchronous MANO updates or establish novelty",
                    "a failure stops expansion of this residual candidate, not every XYZ research route"]}
    verdict["all_required_diagnostics_executed"] = all(isinstance(payloads[phase].get(seed), dict)
        and all(isinstance(get(payloads[phase][seed], "arms", arm, "rows"), list) and len(get(payloads[phase][seed], "arms", arm, "rows")) == 192
                for arm in ARMS) for phase, seeds in (("pilot", SEEDS[:2]), ("full", SEEDS)) for seed in seeds)
    verdict["supplementary_r1_exchange_descriptive_only"] = {str(seed): get(payloads["full"].get(seed), "arms", "B", "same_weight_inference_controls", "supplementary_r1_exchange") for seed in SEEDS}
    out.mkdir(parents=True, exist_ok=True)
    verdict = json_safe(verdict)
    (out / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (out / "VERDICT.md").write_text(render(verdict, table))
    print(json.dumps({"decision": decision, "research_adoption": verdict["research_adoption"], "pending": pending}, ensure_ascii=False))


if __name__ == "__main__":
    main()
