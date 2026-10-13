#!/usr/bin/env python3
"""Evaluate the frozen XC factorial without selecting seeds, clips, or checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tools.xc.run_study import ARMS, OUT, SEEDS, frozen, read_json, stamp, write_json


def numeric(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def get(value, *keys):
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def gate(checks, details=None):
    return {"status": "pass" if checks and all(checks.values()) else "fail", "checks": checks,
            "details": details or {}}


def mandatory_diagnostic_numeric(payload):
    """An OOD wrong-time trajectory is descriptive; true trajectories remain strict."""
    invalid = payload.get("invalid_numeric_paths", [])
    if not isinstance(invalid, list):
        return False
    descriptive = tuple(f"$.arms.{arm}.time_shift_2_control." for arm in ("g0t1", "g1t1"))
    mandatory_invalid = [path for path in invalid if not isinstance(path, str) or not path.startswith(descriptive)]
    return not mandatory_invalid and (payload.get("numeric_valid") is True or bool(invalid))


def shared_clips(payloads):
    checks, records = {}, []
    selection = read_json(OUT / "clip_selection.json")
    expected = {(seq, clip_id, condition) for seq in selection["selected_trace_indices"]
                for clip_id in range(24) for condition in ("natural", "clean", "small", "large")}
    global_reference = None
    for seed in SEEDS:
        payload = payloads.get(seed, {})
        checks[f"seed{seed}_mandatory_diagnostic_scalars_finite"] = mandatory_diagnostic_numeric(payload)
        reference = None
        for arm in ARMS:
            rows = get(payload, "arms", arm, "rows")
            if not isinstance(rows, list):
                checks[f"seed{seed}_{arm}_all_192_clips_executed"] = False
                continue
            keys = [(r.get("seq"), r.get("clip_id"), r.get("condition")) for r in rows]
            checks[f"seed{seed}_{arm}_all_192_clips_executed"] = len(rows) == 192 and len(set(keys)) == 192 and set(keys) == expected
            checks[f"seed{seed}_{arm}_all_trajectories_valid"] = all(r.get("trajectory_valid") is True for r in rows)
            signature = {key: (row.get("initial_state_sha256"), row.get("n_raw_events"),
                               row.get("previous_end_ms"), row.get("first_end_ms")) for key, row in zip(keys, rows)}
            if reference is None:
                reference = signature
            checks[f"seed{seed}_{arm}_same_initial_and_events"] = signature == reference
            if global_reference is None:
                global_reference = signature
            checks[f"seed{seed}_{arm}_same_initial_and_events_across_seeds"] = signature == global_reference
            checks[f"seed{seed}_{arm}_all11_states_preserved"] = all(len(r.get("mpjpe_ra_mm", [])) == 11 and
                                                                       len(r.get("mpjpe_abs_mm", [])) == 11 for r in rows)
            if arm.endswith("t1"):
                contract = get(payload, "arms", arm, "time_binding_contract")
                checks[f"seed{seed}_{arm}_time_control_input_contract"] = get(contract, "contract_pass") is True
                shifted = get(payload, "arms", arm, "time_shift_2_control", "rows")
                checks[f"seed{seed}_{arm}_shift192_clips_preserved"] = isinstance(shifted, list) and len(shifted) == 192
                records.append({"seed": seed, "arm": arm, "time_binding_contract": contract})
    return gate(checks, {"selection_sha256": hashlib.sha256((OUT / "clip_selection.json").read_bytes()).hexdigest(),
                         "time_controls": records, "rule": "no omitted frame, clip, start condition, or seed"})


def recovery(payloads, baseline, candidate):
    checks, rows = {}, []
    for seed in SEEDS:
        for condition in ("natural", "large"):
            base = get(payloads.get(seed), "arms", baseline, "summary", "all", condition, "recovery_auc_ra_mm")
            cand = get(payloads.get(seed), "arms", candidate, "summary", "all", condition, "recovery_auc_ra_mm")
            finite = numeric(base) and numeric(cand)
            checks[f"seed{seed}_{condition}_finite"] = finite
            checks[f"seed{seed}_{condition}_lower_auc"] = finite and cand < base
            rows.append({"seed": seed, "condition": condition, "baseline_auc_ra_mm": base, "candidate_auc_ra_mm": cand,
                         "gain_mm": base - cand if finite else None,
                         "baseline_first_gain_mm": get(payloads.get(seed), "arms", baseline, "summary", "all", condition, "first_step_correction_gain_ra_mm"),
                         "candidate_first_gain_mm": get(payloads.get(seed), "arms", candidate, "summary", "all", condition, "first_step_correction_gain_ra_mm")})
    return gate(checks, {"comparisons": rows, "rule": "both natural and large recovery AUC lower in every seed",
                         "scope": "root-aligned recovery; absolute trajectories are also retained but not a primary gate"})


def motion(payloads, baseline, candidate):
    limits = {"gain_bias_degradation": .10, "cosine_degradation": .05, "lag_increase_steps": .5}
    checks, comparisons, tradeoffs = {}, [], []
    metrics = ("joint_velocity_gain", "joint_velocity_cosine", "velocity_lag_steps")
    for scope in ("zgz_local", "zgz_global"):
        for condition in ("clean", "small"):
            paired = []
            for seed in SEEDS:
                a = {m: get(payloads.get(seed), "arms", baseline, "summary", scope, condition, "motion", m) for m in metrics}
                b = {m: get(payloads.get(seed), "arms", candidate, "summary", scope, condition, "motion", m) for m in metrics}
                finite = all(numeric(v) for item in (a, b) for v in item.values())
                checks[f"seed{seed}_{scope}_{condition}_finite_motion"] = finite
                if not finite:
                    continue
                paired.append({"seed": seed, "baseline": a, "candidate": b,
                               "gain_bias_degradation": abs(b["joint_velocity_gain"] - 1) - abs(a["joint_velocity_gain"] - 1),
                               "cosine_degradation": a["joint_velocity_cosine"] - b["joint_velocity_cosine"],
                               "lag_increase_steps": b["velocity_lag_steps"] - a["velocity_lag_steps"]})
            averages = {key: statistics.mean(row[key] for row in paired) if len(paired) == len(SEEDS) else None for key in limits}
            for key, tolerance in limits.items():
                checks[f"{scope}_{condition}_{key}_within_tolerance"] = numeric(averages[key]) and averages[key] <= tolerance
                if numeric(averages[key]) and 0 < averages[key] <= tolerance and all(row[key] > 0 for row in paired):
                    tradeoffs.append({"scope": scope, "condition": condition, "metric": key,
                                      "mean_degradation": averages[key], "tolerance": tolerance})
            comparisons.append({"scope": scope, "condition": condition, "per_seed": paired, "mean_degradation": averages})
    return gate(checks, {"comparisons": comparisons, "limits": limits, "consistent_small_tradeoffs": tradeoffs,
                         "interpretation": "engineering tolerances; no significance claim"})


def time_sensitivity(payloads):
    result = {}
    for arm in ("g0t1", "g1t1"):
        rows = []
        for seed in SEEDS:
            for condition in ("natural", "clean", "small", "large"):
                true = get(payloads.get(seed), "arms", arm, "summary", "all", condition, "recovery_auc_ra_mm")
                shift = get(payloads.get(seed), "arms", arm, "time_shift_2_control", "summary", "all", condition, "recovery_auc_ra_mm")
                rows.append({"seed": seed, "condition": condition, "true_time_auc_mm": true,
                             "shift2_auc_mm": shift, "shift_minus_true_mm": shift - true if numeric(shift) and numeric(true) else None,
                             "numeric_status": "PASS" if numeric(true) and numeric(shift) else "FAIL"})
        result[arm] = rows
    return {"descriptive_only": True, "per_arm": result,
            "interpretation": "same-weight shifted input is OOD; it can support time-use interpretation but cannot alone establish the retrained T effect"}


def record_completion(verdict, table=""):
    """Append results outside the immutable preregistration, then preserve identities."""
    marker = "XYZ-GEOMETRY-CAUSAL-TIME-20261012-RESULT"
    begin, end = (f"<!-- EXPERIMENT_RECORD:{marker}:{part} -->" for part in ("BEGIN", "END"))
    doc = REPO / "docs/EXPERIMENT_RECORDS.md"
    text = doc.read_text()
    if begin not in text:
        lines = [begin, '<a id="xyz-geometry-causal-time-result-20261012"></a>', "",
                 "### [XC] 四组几何／因果时间关联实验结果", "", table.rstrip(), "",
                 f"完成时刻：{verdict['completed_at']}；裁决：`{verdict['status']}`。", "",
                 "实际逐run身份、精确step6000、三个seed及四条件独立同seed复跑、各自对比噪声、五个条件差、交互、48共同片段的全部恢复和动作曲线均在 `outputs/xc`。失败值保留，未对有限子集平均。", "",
                 "机器判定与护栏：", "", "```json",
                 json.dumps({key: verdict.get(key) for key in
                             ("status", "supported_geometry_contexts", "supported_time_contexts", "interaction", "scope_limits", "primary_failures")},
                            ensure_ascii=False, indent=2, allow_nan=False), "```", "",
                 "原始曲线见 `outputs/xc/recovery_s*.json`；完整逐条件护栏见 `outputs/xc/verdict.json`；固定权重错配时刻只作敏感性解释。主表T为四片近似，边图仍anchor，不能声称持久逐事件MANO图、独立观测深度或端到端调度收益。负结果仅限制本实现与预算。S39默认未修改。", "", end]
        doc.write_text(text.rstrip() + "\n\n" + "\n".join(lines) + "\n")
    manifest = frozen()
    paths = [OUT / "frozen_manifest.json", OUT / "preflight.json", OUT / "clip_selection.json",
             OUT / "verdict.json", OUT / "VERDICT.md", doc]
    paths += sorted(OUT.glob("recovery_s*.json")) + sorted(OUT.glob("*failures.json"))
    paths += [OUT / "gates.json", OUT / "TABLE_full.md"]
    paths += sorted((OUT / "individual_rows").glob("*.json"))
    paths += sorted((REPO / "outputs/semkine").glob("xc_*_main_row_last.json"))
    for job in manifest["jobs"]:
        directory = REPO / f"outputs/semkine/{job['run']}"
        paths += sorted(directory.glob("*-step=6000.ckpt"))
        paths += [directory / "training_metadata.json", directory / "config_resolved.yaml",
                  directory / "pair_stream_rank0.json"]
    identities = {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in sorted(set(paths)) if path.is_file()}
    write_json(OUT / "finished_manifest.json", {"study": manifest["study"], "finished_at": stamp(),
                                                "verdict": verdict["status"], "files_sha256": identities,
                                                "source_files_sha256": manifest["files_sha256"],
                                                "prereg_sha256": manifest["prereg_sha256"],
                                                "missing_required_diagnostics": verdict.get("missing", verdict.get("missing_diagnostics", [])),
                                                "mutable_status_and_event_journal_excluded": True})


def finalize():
    manifest = frozen()
    primary_path = OUT / "gates.json"
    primary = read_json(primary_path) if primary_path.is_file() else None
    payloads, missing = {}, []
    for seed in SEEDS:
        path = OUT / f"recovery_s{seed}.json"
        if path.is_file():
            payloads[seed] = read_json(path)
        else:
            missing.append(str(path.relative_to(REPO)))
    shared = shared_clips(payloads)
    if primary is None:
        failures_path = OUT / "primary_failures.json"
        failures = read_json(failures_path) if failures_path.is_file() else {"reason": "required primary rows or gates unavailable"}
        result = {"study": manifest["study"], "completed_at": stamp(),
                  "status": "measured_primary_failure", "primary_failures": failures,
                  "missing_diagnostics": missing, "shared_replay_and_numeric_contract": shared,
                  "time_shift_sensitivity": time_sensitivity(payloads),
                  "interpretation": "required primary results failed; no finite subset averaging or earlier-checkpoint substitution",
                  "scope_limit": "this implementation and budget cannot support the proposed research claim; no universal impossibility proof"}
        write_json(OUT / "verdict.json", result)
        (OUT / "VERDICT.md").write_text("正式末点主评价失败；原始失败与完整诊断保留在 primary_failures.json 和 recovery_s*.json。未以有限种子子集或较早点替代。\n")
        record_completion(result)
        print("XC conclusion: measured_primary_failure")
        return
    contrasts = {}
    for name, value in primary["contrasts"].items():
        rec = recovery(payloads, value["baseline"], value["candidate"])
        mot = motion(payloads, value["baseline"], value["candidate"])
        passed = value["efficacy_pass"] and shared["status"] == "pass" and rec["status"] == "pass" and mot["status"] == "pass"
        contrasts[name] = {"primary": value, "recovery": rec, "normal_motion": mot, "research_support_pass": passed}
    i_values = primary["interaction_gain_mm"]["overall"]
    interaction_checks = {"all_three_seeds_positive": all(v > 0 for v in i_values),
                          "overall_mean_exceeds_interaction_repeat_noise": primary["interaction_mean_gain_mm"]["overall"] > primary["interaction_threshold_mm"]["overall"],
                          "G_at_T1_supported": contrasts["G_at_T1"]["research_support_pass"],
                          "T_at_G1_supported": contrasts["T_at_G1"]["research_support_pass"],
                          "joint_vs_G0T0_supported": contrasts["joint_vs_G0T0"]["research_support_pass"]}
    supported_g = [name for name in ("G_at_T0", "G_at_T1") if contrasts[name]["research_support_pass"]]
    supported_t = [name for name in ("T_at_G0", "T_at_G1") if contrasts[name]["research_support_pass"]]
    interaction = gate(interaction_checks, {"gain_mm": primary["interaction_gain_mm"],
                                            "threshold_mm": primary["interaction_threshold_mm"],
                                            "formula": "E10 + E01 - E00 - E11; positive means superadditive error reduction"})
    if missing:
        status = "incomplete_required_diagnostics"
    elif shared["status"] != "pass":
        status = "invalid_diagnostic_or_contract"
    elif interaction["status"] == "pass":
        status = "support_geometry_time_interaction_in_this_protocol"
    elif supported_g or supported_t:
        status = "support_specific_conditional_effects"
    else:
        status = "no_supported_effect_under_registered_gates"
    costs = {arm: {key: read_json(REPO / f"outputs/semkine/xc_{arm}_main_row_last.json").get(key)
                   for key in ("latency_ms_scaled_full1p75", "latency_ms_raw", "macs_forward_packet", "params_total", "latency_context_audit", "mac_fixture_audit")}
             for arm in ARMS}
    verdict = {"study": manifest["study"], "completed_at": stamp(), "status": status,
               "missing": missing, "prereg_sha256": manifest["prereg_sha256"],
               "shared_replay_and_numeric_contract": shared, "contrasts": contrasts, "interaction": interaction,
               "supported_geometry_contexts": supported_g, "supported_time_contexts": supported_t,
               "time_shift_sensitivity": time_sensitivity(payloads), "costs": costs,
               "scope_limits": ["G toggles original D/Z/edge bundle; no individual channel attribution",
                                "G supplies geometry derived from the prior, not measured depth",
                                "T uses four temporal centers and anchor edge graph, not exact continuous event-time graph maintenance",
                                "training leading start uses teacher noise; deployment history is own-state",
                                "zgz is both development and testing; three seeds and one repeat contrast are not a confidence interval",
                                "negative result rejects this implementation/budget, not all possible explicit3D/time methods",
                                "timing is calibrated model-query timing, not end-to-end sensor latency; no incremental-runtime claim"],
               "deployment_default": "S39 remains unchanged; this study does not authorize default replacement"}
    write_json(OUT / "verdict.json", verdict)
    table = (OUT / "TABLE_full.md").read_text()
    lines = [table.rstrip(), "", f"裁决：`{status}`。", "",
             f"G支持的条件：{', '.join(supported_g) or '未检出'}；T支持的条件：{', '.join(supported_t) or '未检出'}；交互门：{interaction['status']}。", "",
             "全部五个预注册对比、逐seed差、各自复跑噪声门、恢复与动作护栏见 verdict.json；错配时间仅作同权重敏感性诊断。", "",
             "四片近似检验的范围是事件时刻关联和matched D/Z；边图仍在anchor。任何失败只限制本实现与预算。S39默认保持原样。"]
    (OUT / "VERDICT.md").write_text("\n".join(lines) + "\n")
    record_completion(verdict, table)
    print(f"XC conclusion: {status}; missing={missing}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    finalize()


if __name__ == "__main__":
    main()
