#!/usr/bin/env python3
"""Apply fixed U1a sharing/measurement gates to existing explicit-last artifacts.

No training, evaluation, checkpoint selection, or CUDA work is launched.
2k is screening only; 6k can pass the preregistered engineering gates.
Example:
  python tools/u1a/gates.py --shared-row SHARED_main_row.json --shared-extended SHARED_extended.json \
    --untied-row UNTIED_main_row.json --untied-extended UNTIED_extended.json \
    --shared-raw SHARED_s3407_probe.json SHARED_s3408_probe.json \
    --untied-raw UNTIED_s3407_probe.json UNTIED_s3408_probe.json \
    --baseline-raw S38_s3407_probe.json S38_s3408_probe.json --output outputs/u1a/2k_gate.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SEEDS = {"3407", "3408"}
SEQUENCES = ("zgz_global", "zgz_local")
PARTS = ("overall", "global", "local")
METRICS = ("mpjpe_ra_mm", "transl_mm", "root_rot_deg")
MANIFEST_SHA = "2a4769f33ab86a02f6124ddf0096f445adaa4702e7798b2f73c01b1fb474b405"
LIMITS = {"ra_mm": 1.1, "local_mm": 1.1, "translation_ratio": 1.1, "root_deg": 1.0}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolved(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO / path).resolve()


def sha256(path):
    h = hashlib.sha256()
    with resolved(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(resolved(path).read_text(encoding="utf-8"))


def number(value, label):
    require(not isinstance(value, bool), f"{label}: boolean is not a metric")
    value = float(value)
    require(math.isfinite(value), f"{label}: non-finite metric")
    return value


def close(first, second, label, tolerance=0.01):
    require(abs(number(first, label) - number(second, label)) < tolerance, f"{label}: inconsistent values")


def source_file(source, label):
    path = resolved(source["path"])
    require(path.is_file() and sha256(path) == source["sha256"], f"{label}: file/source hash changed")
    return path


def normalize_config(cfg, *, omit_seed=False):
    cfg = copy.deepcopy(cfg)
    if omit_seed:
        cfg.pop("SEED", None)
    cfg["MODEL"].pop("U1A_MODE", None)
    cfg["MODEL"].pop("INIT_FROM", None)
    for key in ("RUN_NAME", "OUTPUT_DIR"):
        cfg["TRAIN"].pop(key, None)
    cfg.get("EVAL", {}).pop("OUTPUT_DIR", None)
    return cfg


def validate_row(row, *, mode, budget, label):
    require(row["mode"] == mode and int(row["budget"]) == budget, f"{label}: mode/budget mismatch")
    require(row["split"] == "val_core" and row["ckpt_rule"] == "last", f"{label}: only explicit last/val_core accepted")
    require(set(row["per_seed"]) == SEEDS and set(row["provenance"]) == SEEDS and int(row["n_seeds"]) == 2,
            f"{label}: exactly preregistered seeds 3407/3408 required")
    protocol = row["protocol"]
    require(protocol["manifest"] == "splits_semkine.json" and protocol["train_sequences"] == 72
            and protocol["sequences"] == list(SEQUENCES) and protocol["step_ms"] == 50
            and not protocol["single_seed_screen_only"], f"{label}: fixed data protocol mismatch")
    for seed in sorted(SEEDS):
        item, provenance = row["per_seed"][seed], row["provenance"][seed]
        require(int(item["step"]) == budget and int(provenance["step"]) == budget
                and str(provenance["seed"]) == seed and provenance["mode"] == mode, f"{label}/{seed}: source step/seed/mode mismatch")
        require(sum(item["frames_per_seq"].values()) == item["n_frames"] == 2590
                and set(item["frames_per_seq"]) == set(SEQUENCES), f"{label}/{seed}: incomplete fixed evaluation")
        require(resolved(item["run"]) == resolved(provenance["run"])
                and resolved(item["ckpt"]) == resolved(provenance["checkpoint"]["path"]),
                f"{label}/{seed}: checkpoint/run provenance mismatch")
        require(provenance["manifest"]["sha256"] == MANIFEST_SHA, f"{label}/{seed}: nonfixed manifest")
        for key in ("checkpoint", "config", "manifest", "training_metadata", "evaluation", "diagnostic_arrays"):
            source_file(provenance[key], f"{label}/{seed}/{key}")
        for part in PARTS:
            for metric in METRICS:
                number(item[part][metric], f"{label}/{seed}/{part}/{metric}")
    for part in PARTS:
        for metric in METRICS:
            expected = statistics.fmean(row["per_seed"][seed][part][metric] for seed in SEEDS)
            close(row["two_seed_mean"][part][metric], expected, f"{label}/{part}/{metric}/seed_mean", 1e-8)


def validate_diagnostics(row, diagnostics, label):
    require(set(diagnostics) == SEEDS, f"{label}: diagnostic seeds mismatch")
    for seed, js in diagnostics.items():
        per_seed, provenance = row["per_seed"][seed], row["provenance"][seed]
        require(js["run"] == resolved(per_seed["run"]).name and js["split"] == "val_core"
                and int(js["step"]) == int(row["budget"]), f"{label}/{seed}: diagnostic identity mismatch")
        require(resolved(js["ckpt"]) == resolved(provenance["checkpoint"]["path"])
                and resolved(js["manifest"]) == resolved(provenance["manifest"]["path"]),
                f"{label}/{seed}: diagnostic source mismatch")
        require(js["window"]["mode"] == "fixed" and js["window"]["ms"] == 50
                and "tf" in js and "perturb" in js and "selection_drift_mm" not in js,
                f"{label}/{seed}: missing fixed last tf/perturb diagnostic")
        # Extended JSON must be the original, hash-verified evaluation JSON.
        require(js == read_json(provenance["evaluation"]["path"]), f"{label}/{seed}: extended JSON drift")
        for part, sequence in (("global", "zgz_global"), ("local", "zgz_local"), ("overall", "overall")):
            for metric in METRICS:
                value = js["model"][sequence][metric]
                value = value[0] if isinstance(value, list) else value
                close(per_seed[part][metric], value, f"{label}/{seed}/{part}/{metric}")


def load_bundle(row_path, extended_path, mode):
    row, extended = read_json(row_path), read_json(extended_path)
    budget = int(row["budget"])
    require(budget in (2000, 6000), f"{mode}: only 2k/6k budgets accepted")
    validate_row(row, mode=mode, budget=budget, label=mode)
    require(extended["arm"] == row["arm"] and extended["checkpoint_rule"] == "last"
            and extended["provenance"] == row["provenance"], f"{mode}: row/extended provenance mismatch")
    validate_diagnostics(row, extended["per_seed"], mode)
    require("baseline" in extended and "baseline_reference" in row, f"{mode}: matched S38 reference required")
    baseline = extended["baseline"]["row"]
    validate_row(baseline, mode="baseline", budget=6000, label=f"{mode}/S38")
    validate_diagnostics(baseline, extended["baseline"]["per_seed"], f"{mode}/S38")
    reference = row["baseline_reference"]
    for key in ("arm", "per_seed", "provenance", "two_seed_mean", "budget"):
        require(reference[key] == baseline[key], f"{mode}: baseline reference drift in {key}")
    require(reference["checkpoint_rule"] == "last", f"{mode}: selected S38 source")
    configs = {seed: yaml.safe_load(resolved(row["provenance"][seed]["config"]["path"]).read_text()) for seed in SEEDS}
    for seed, cfg in configs.items():
        mc = cfg["MODEL"]
        require(int(cfg["SEED"]) == int(seed) and int(cfg["TRAIN"]["MAX_STEPS"]) == budget
                and mc["U1A_MODE"] == mode and mc["U1A_FREEZE_ENCODER"] is True,
                f"{mode}/{seed}: configuration source mismatch")
        require(mc["ROOT_FILTER_GAIN"] == 0.5 and mc["FINGER_FILTER_GAIN"] == 0.5
                and mc["ROOT_MEAS"] == "abs" and mc["FINGER_MEAS"] == "abs"
                and mc["ENCODER"] == "sparse_pyramid", f"{mode}/{seed}: measurement convention changed")
        require(mc["U1A_HIDDEN"] == 64 and mc["PREV_RENDER"] is False
                and mc["ROUTED_READOUT"] is True and mc["ACTIVE_HEAD"] is True
                and mc["PREDICT_DELTA"] is True, f"{mode}/{seed}: preregistered head interface changed")
        fixed_train = {"DEVICES": 2, "BATCH_SIZE_PER_GPU": 512, "ACCUMULATE_GRAD_BATCHES": 1,
                       "NUM_WORKERS": 14, "OPTIMIZER": "adam", "LR": 0.004, "WARMUP_STEPS": 500,
                       "PRECISION": "bf16", "LR_SCHEDULE": "cosine", "TRAINABLE_PREFIXES": ["u1a_readout."]}
        for key, expected in fixed_train.items():
            require(cfg["TRAIN"].get(key) == expected, f"{mode}/{seed}: preregistered TRAIN.{key} changed")
        source = row["provenance"][seed]
        metadata = read_json(source["training_metadata"]["path"])
        require("resumed_from" in metadata and metadata["resumed_from"] is None,
                f"{mode}/{seed}: 2k/6k must restart prepared initialization, not resume")
        expected_metadata = {"seed": int(seed), "max_steps": budget, "batch_size_per_gpu": 512,
                             "devices": 2, "accumulate_grad_batches": 1, "effective_batch": 1024,
                             "data_generator_seed": int(seed), "lr": 0.004, "lr_schedule": "cosine",
                             "input_mode": "raw_packed", "splits_manifest": "splits_semkine.json",
                             "val_core_sequences": list(SEQUENCES), "selection_policy": "explicit last only (U1a)"}
        for key, expected in expected_metadata.items():
            require(metadata.get(key) == expected, f"{mode}/{seed}: actual training metadata {key} changed")
        subjects = {"ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf"}
        train_sequences = metadata["train_sequences"]
        require(len(train_sequences) == len(set(train_sequences)) == 72
                and {sequence.split("_")[0] for sequence in train_sequences} == subjects,
                f"{mode}/{seed}: actual training split changed")
        require(resolved(cfg["MODEL"]["INIT_FROM"]) == source_file(source["initialization"], f"{mode}/{seed}/init"),
                f"{mode}/{seed}: prepared initialization source changed")
        registry = read_json(source_file(source["initialization_registry"], f"{mode}/{seed}/init_registry"))
        registered = registry["pairs"][seed]
        require(registry["split_sha256"] == MANIFEST_SHA
                and resolved(registered["arms"][mode]["init"]) == resolved(cfg["MODEL"]["INIT_FROM"])
                and resolved(registered["anchor"]) == resolved(source["frozen_s38_source"]["path"])
                and registered["anchor_sha256"] == source["frozen_s38_source"]["sha256"],
                f"{mode}/{seed}: initialization registry/source mismatch")
    return row, baseline, configs


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def validate_raw_summary(summary, records, label):
    require(summary is not None and records, f"{label}: empty diagnostic")
    frames = [record[0] for record in records]
    live = [record[1] > 0 for record in records]
    n, n_live = len(records), sum(live)
    require(n_live > 0 and summary["n_windows"] == n and summary["n_valid_measurements"] == n_live
            and summary["n_empty"] == n - n_live, f"{label}: invalid mask counts")
    close(summary["measurement_coverage"], n_live / n, f"{label}/coverage", 1e-12)
    require(summary["frame_index_sha256"] == canonical_hash(frames)
            and summary["measurement_mask_sha256"] == canonical_hash(list(zip(frames, live))),
            f"{label}: frame or live-mask hash disagrees with actual indices")
    metric = summary["measurement_metrics"]["root_geodesic_deg"]
    require(metric is not None and metric["n"] == n_live, f"{label}: raw root metric is not nonempty-only")
    value = number(metric["mean"], f"{label}/raw_root")
    require(0.0 <= value <= 180.0, f"{label}: invalid geodesic range")
    return value


def load_raw(paths, row, configs, mode):
    import numpy as np
    raw_by_seed, provenance, fingerprints = {}, {}, {}
    for path in paths:
        js = read_json(path)
        seed = str(int(js["seed"]))
        require(seed in SEEDS and seed not in raw_by_seed, f"{mode}: duplicate/unregistered raw seed")
        source = row["provenance"][seed]
        require(js["gate_eligible"] is True and js["raw"]["gate_eligible_full_window_coverage"] is True,
                f"{mode}/{seed}: partial/smoke raw is not gate eligible")
        require(js["run"] == resolved(source["run"]).name and js["split"] == "val_core"
                and int(js["step"]) == int(row["budget"]) and js["u1a_mode"] == (mode if mode != "baseline" else "off"),
                f"{mode}/{seed}: raw run/step/split/mode mismatch")
        for probe_path, probe_hash, source_key in (("ckpt", "ckpt_sha256", "checkpoint"),
                                                   ("config", "config_sha256", "config"),
                                                   ("manifest", "manifest_sha256", "manifest")):
            require(resolved(js[probe_path]) == resolved(source[source_key]["path"])
                    and js[probe_hash] == source[source_key]["sha256"], f"{mode}/{seed}: raw {source_key} differs from recursive report")
        require(js["probe_sha256"] == sha256(REPO / "tools/u1a/probe.py"), f"{mode}/{seed}: raw probe code changed")
        cfg = configs[seed]
        require(js["root_ref"] == cfg["MODEL"]["ROOT_REF"], f"{mode}/{seed}: raw reference rotation mismatch")
        sampling = js["raw"]["sampling"]
        require(sampling["step_ms"] == 50 and sampling["window_ms"] == 50 and sampling["stride"] == 1
                and sampling["max_sampled_windows_per_sequence"] is None
                and js["raw"]["protocol_windows_total"] == js["raw"]["sampled_windows_total"] == 2590,
                f"{mode}/{seed}: raw sampling differs from full 50-ms protocol")
        seqs = js["raw"]["per_sequence"]
        require(set(seqs) == set(SEQUENCES), f"{mode}/{seed}: raw sequences mismatch")
        records, values = {}, {}
        for sequence in SEQUENCES:
            item = seqs[sequence]
            n = row["per_seed"][seed]["frames_per_seq"][sequence]
            require(item["protocol_windows"] == len(item["indices"]) == n, f"{mode}/{seed}/{sequence}: incomplete indices")
            require([index["protocol_step_index"] for index in item["indices"]] == list(range(n)),
                    f"{mode}/{seed}/{sequence}: missing/reordered fixed windows")
            records[sequence] = [([sequence, int(index["run_id"]), int(index["end_ms"]), int(index["protocol_step_index"])],
                                  int(index["n_events"])) for index in item["indices"]]
            require(all(count >= 0 for _, count in records[sequence]), f"{mode}/{seed}: negative event count")
            # Same number of windows is insufficient: bind raw indices and live
            # masks to the actual hash-verified recursive evaluation arrays.
            with np.load(resolved(source["diagnostic_arrays"]["path"]), allow_pickle=False) as arrays:
                for raw_key, array_key in (("end_ms", "end"), ("run_id", "run"), ("n_events", "count")):
                    key = f"model|{sequence}|{array_key}"
                    require(key in arrays, f"{mode}/{seed}/{sequence}: recursive {array_key} array missing")
                    expected = np.asarray([index[raw_key] for index in item["indices"]], dtype=np.int64)
                    require(np.array_equal(arrays[key], expected),
                            f"{mode}/{seed}/{sequence}: raw {raw_key} differs from recursive {array_key}")
            values[sequence] = validate_raw_summary(item["summary"], records[sequence], f"{mode}/{seed}/{sequence}")
        all_records = records["zgz_global"] + records["zgz_local"]
        parts = {}
        for part, part_records in (("overall", all_records), ("global", records["zgz_global"]), ("local", records["zgz_local"])):
            parts[part] = validate_raw_summary(js["raw"][part], part_records, f"{mode}/{seed}/{part}")
        total_live = sum(js["raw"]["per_sequence"][sequence]["summary"]["n_valid_measurements"] for sequence in SEQUENCES)
        weighted = sum(values[sequence] * seqs[sequence]["summary"]["n_valid_measurements"] for sequence in SEQUENCES) / total_live
        close(parts["overall"], weighted, f"{mode}/{seed}/raw_overall_weighting", 1e-8)
        for part, sequence in (("global", "zgz_global"), ("local", "zgz_local")):
            close(parts[part], values[sequence], f"{mode}/{seed}/raw_{part}_identity", 1e-8)
        raw_by_seed[seed] = parts
        fingerprints[seed] = {"indices": all_records, "sampling": sampling, "root_ref": js["root_ref"],
                              "probe_sha256": js["probe_sha256"], "reference_composition": js["raw"]["reference_composition"],
                              "contexts": js["raw"]["contexts"]}
        provenance[seed] = {"path": str(resolved(path)), "sha256": sha256(path),
                            "source_checkpoint_sha256": js["ckpt_sha256"],
                            "valid_measurements": {part: js["raw"][part]["n_valid_measurements"] for part in PARTS}}
    require(set(raw_by_seed) == SEEDS, f"{mode}: both preregistered raw seeds required")
    return raw_by_seed, fingerprints, provenance


def validate_pair_sources(shared, untied, baseline, shared_cfg, untied_cfg):
    import torch
    cache = {}
    def checkpoint(source):
        path = str(resolved(source))
        if path not in cache:
            cache[path] = torch.load(path, map_location="cpu", weights_only=False)
        return cache[path]
    reference_cfgs = {}
    for seed in sorted(SEEDS):
        require(shared["per_seed"][seed]["frames_per_seq"] == untied["per_seed"][seed]["frames_per_seq"]
                == baseline["per_seed"][seed]["frames_per_seq"], f"{seed}: recursive frame counts differ")
        require(normalize_config(shared_cfg[seed]) == normalize_config(untied_cfg[seed]), f"{seed}: paired recipes differ beyond sharing/init paths")
        reference_cfg = yaml.safe_load(resolved(baseline["provenance"][seed]["config"]["path"]).read_text())
        reference_cfgs[seed] = reference_cfg
        for section in ("TRACK", "LOSS", "MANO"):
            require(shared_cfg[seed].get(section) == reference_cfg.get(section), f"{seed}: S38 {section} protocol differs")
        for key in ("ROOT_REF", "ROOT_FILTER_GAIN", "FINGER_FILTER_GAIN"):
            require(shared_cfg[seed]["MODEL"][key] == reference_cfg["MODEL"][key], f"{seed}: S38 {key} differs")
        reference_checkpoint = checkpoint(baseline["provenance"][seed]["checkpoint"]["path"])
        require(reference_checkpoint["global_step"] == 6000, f"{seed}: S38 source is not 6000-last")
        for label, row in (("shared", shared), ("untied", untied)):
            source = row["provenance"][seed]
            frozen = source["frozen_s38_source"]
            anchor = checkpoint(source_file(frozen, f"{label}/{seed}/anchor"))
            require(resolved(frozen["run"]) == resolved(baseline["per_seed"][seed]["run"])
                    and anchor["global_step"] == 6000, f"{label}/{seed}: wrong S38 initialization source")
            anchor_state, reference_state = anchor["state_dict"], reference_checkpoint["state_dict"]
            require(anchor_state.keys() == reference_state.keys()
                    and all(torch.equal(value, reference_state[key]) for key, value in anchor_state.items()),
                    f"{label}/{seed}: last alias differs from S38 evaluated checkpoint state")
            trained = checkpoint(source["checkpoint"]["path"])
            require(trained["global_step"] == row["budget"], f"{label}/{seed}: checkpoint step mismatch")
            encoder_keys = [key for key in anchor_state if key.startswith("event_encoder.")]
            require(encoder_keys and all(key in trained["state_dict"] and torch.equal(anchor_state[key], trained["state_dict"][key])
                                         for key in encoder_keys), f"{label}/{seed}: frozen encoder/BN changed")
    first, second = sorted(SEEDS)
    require(normalize_config(shared_cfg[first], omit_seed=True) == normalize_config(shared_cfg[second], omit_seed=True),
            "recipes differ across the two preregistered seeds")
    return reference_cfgs


def comparison(candidate, reference, raw_candidate, raw_reference, *, paired, label):
    checks = []
    def add(name, observed, maximum):
        observed, maximum = float(observed), float(maximum)
        checks.append({"name": name, "observed": observed, "maximum": maximum, "pass": observed <= maximum + 1e-9})
    if paired:
        for seed in sorted(SEEDS):
            add(f"s{seed}/recursive_RA_delta_mm",
                candidate["per_seed"][seed]["overall"]["mpjpe_ra_mm"] - reference["per_seed"][seed]["overall"]["mpjpe_ra_mm"], LIMITS["ra_mm"])
    else:
        add("mean/recursive_RA_delta_mm",
            candidate["two_seed_mean"]["overall"]["mpjpe_ra_mm"] - reference["two_seed_mean"]["overall"]["mpjpe_ra_mm"], LIMITS["ra_mm"])
    add("mean/recursive_local_RA_delta_mm",
        candidate["two_seed_mean"]["local"]["mpjpe_ra_mm"] - reference["two_seed_mean"]["local"]["mpjpe_ra_mm"], LIMITS["local_mm"])
    denominator = reference["two_seed_mean"]["overall"]["transl_mm"]
    require(denominator > 0, f"{label}: translation ratio denominator must be positive")
    add("mean/absolute_translation_ratio", candidate["two_seed_mean"]["overall"]["transl_mm"] / denominator, LIMITS["translation_ratio"])
    for seed in sorted(SEEDS):
        for part in ("overall", "global"):
            add(f"s{seed}/{part}/recursive_root_delta_deg",
                candidate["per_seed"][seed][part]["root_rot_deg"] - reference["per_seed"][seed][part]["root_rot_deg"], LIMITS["root_deg"])
            add(f"s{seed}/{part}/raw_nonempty_root_delta_deg", raw_candidate[seed][part] - raw_reference[seed][part], LIMITS["root_deg"])
    return {"pass": all(item["pass"] for item in checks), "criteria": checks}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for mode in ("shared", "untied"):
        parser.add_argument(f"--{mode}-row", required=True)
        parser.add_argument(f"--{mode}-extended", required=True)
        parser.add_argument(f"--{mode}-raw", nargs="+", required=True)
    parser.add_argument("--baseline-raw", nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="validate all sources and print; do not write a gate file")
    args = parser.parse_args(argv)
    shared, baseline, shared_cfg = load_bundle(args.shared_row, args.shared_extended, "shared")
    untied, other_baseline, untied_cfg = load_bundle(args.untied_row, args.untied_extended, "untied")
    require(shared["budget"] == untied["budget"], "paired budgets differ")
    require(shared["eval_variant"] == untied["eval_variant"], "paired evaluation variants differ")
    require(baseline == other_baseline, "paired reports have different fixed S38 reference artifacts")
    baseline_cfg = validate_pair_sources(shared, untied, baseline, shared_cfg, untied_cfg)
    shared_raw, shared_frames, shared_sources = load_raw(args.shared_raw, shared, shared_cfg, "shared")
    untied_raw, untied_frames, untied_sources = load_raw(args.untied_raw, untied, untied_cfg, "untied")
    baseline_raw, baseline_frames, baseline_sources = load_raw(args.baseline_raw, baseline, baseline_cfg, "baseline")
    for seed in sorted(SEEDS):
        require(shared_frames[seed] == untied_frames[seed] == baseline_frames[seed],
                f"{seed}: raw frames/event counts/masks/sampling/reference/source code differ")
    require(shared_frames["3407"] == shared_frames["3408"], "raw fixed protocol differs across seeds")
    gates = {"sharing": comparison(shared, untied, shared_raw, untied_raw, paired=True, label="sharing"),
             "shared_vs_S38": comparison(shared, baseline, shared_raw, baseline_raw, paired=False, label="shared/S38"),
             "untied_vs_S38": comparison(untied, baseline, untied_raw, baseline_raw, paired=False, label="untied/S38")}
    final = shared["budget"] == 6000
    eligible = final and gates["sharing"]["pass"] and gates["shared_vs_S38"]["pass"]
    input_paths = [args.shared_row, args.shared_extended, args.untied_row, args.untied_extended,
                   *args.shared_raw, *args.untied_raw, *args.baseline_raw]
    result = {"kind": "U1a_preregistered_engineering_gates", "budget": shared["budget"], "checkpoint_rule": "explicit last",
              "stage": "final_6k" if final else "screening_2k", "final_adoption_eligible": eligible,
              "gates": gates, "thresholds": LIMITS,
              "protocol": {"seeds": sorted(SEEDS), "split": "val_core", "manifest_sha256": MANIFEST_SHA,
                           "recursive_windows": 2590, "step_ms": 50, "S38_source_budget": 6000,
                           "local_metric": "recursive local root-aligned MPJPE in mm",
                           "raw_root_metric": "nonempty-only, full-window-coverage, frame-weighted geodesic mean in degrees",
                           "seed_aggregation": "equal mean of seed frame-weighted metrics"},
              "diagnostic_sources": {"shared": shared_sources, "untied": untied_sources, "S38": baseline_sources},
              "inputs": [{"path": str(resolved(path)), "sha256": sha256(path)} for path in input_paths],
              "gate_script_sha256": sha256(__file__),
              "limitations": ["2k cannot authorize final adoption", "two seeds and zgz development/test support an engineering screen only",
                              "sharing versus untied capacity differs; translation interface changed in both",
                              "passing does not establish the best architecture, graph messages, asynchronous computation, or local mesh updates"]}
    if final and gates["sharing"]["pass"] and not gates["shared_vs_S38"]["pass"] and not gates["untied_vs_S38"]["pass"]:
        result["verdict"] = "sharing added no preregistered damage, but both interfaces failed S38 usability guards"
    elif not final:
        result["verdict"] = "screening only; restart the preregistered 6k arms from prepared initialization before a final decision"
    elif eligible:
        result["verdict"] = "shared passed the current 6k sharing and S38 usability gates; eligible as the next research candidate"
    else:
        result["verdict"] = "shared not adopted under the current preregistered 6k gates; inspect the failed criteria and attribution controls"
    payload = json.dumps(result, indent=2, allow_nan=False)
    if not args.dry_run:
        output = resolved(args.output or REPO / "outputs/u1a" / f"{shared['budget'] // 1000}k" / "gate.json")
        require(output not in {resolved(path) for path in input_paths}, "gate output cannot overwrite an input")
        output.parent.mkdir(parents=True, exist_ok=True)
        temp = output.with_suffix(output.suffix + ".tmp")
        temp.write_text(payload + "\n", encoding="utf-8")
        temp.replace(output)
        print(f"wrote {output}", file=sys.stderr)
    print(payload)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(f"U1a gate refused: {error}")

