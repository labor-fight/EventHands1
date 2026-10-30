#!/usr/bin/env python3
"""Build the independent E -> frozen P -> launch A bundle; never start training.

The default action verifies inputs and writes only an explicitly requested audit
plan. --commit exclusively publishes the new formal P/A and a text source archive.
Large data, checkpoints, tapes, journals and NPZ traces remain SHA references.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.xag_paired_20261018 import verify_final as V

OUT = ROOT / "outputs/xag_paired/20261018"
C1 = ROOT / "outputs/xyz_c1_repro/20261018"
BASELINE_IDENTITY = ROOT / "outputs/xyz_c1_repro/baseline.json"
REFERENCE = C1 / "runs/xyz_c1_stable_s3407"
MUTABLE_POINTER = "outputs/xag/current_metrics.json"
TRAIN_SHA = "43a052b1bfc40b1198d88f315a2baf133fe6eaa1b21fef6a10832f89e5682583"
PAIRS = {"a0f1": [0, 1], "a1f0": [6, 7], "a1f1": [2, 3], "a2f0": [6, 7], "a2f1": [4, 5]}
FORMAL_TOOLS = ("train.py", "execute.py", "verify_final.py", "evaluate.py", "trace.py", "report.py", "freeze.py")
TEXT_SUFFIXES = {".py", ".yaml", ".yml", ".json", ".md", ".sh", ".toml"}


def encoded(value):
    return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def key(path):
    path = V.path(path).absolute()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def check_map(records):
    for name, expected in records.items():
        V.require(isinstance(expected, str) and bool(re.fullmatch("[a-f0-9]{64}", expected)), "Invalid file SHA schema: " + name)
        V.checked_file(name, expected)


def merge(records, additions):
    for name, value in additions.items():
        V.require(name not in records or records[name] == value, "Conflicting inherited/source SHA: " + name)
        records[name] = value


def add_paths(records, paths):
    for path in paths:
        merge(records, {key(path): V.sha(V.path(path))})


def verify_data_stats(records):
    for name, expected in records.items():
        actual = V.path(name).stat()
        V.require(actual.st_size == expected["bytes"] and actual.st_mtime_ns == expected["mtime_ns"], "Immutable data stat changed: " + name)


def make_acceptance(protocol, protocol_bytes):
    """Only A points to P; P never includes A's or archive-manifest's hash."""
    configs = {str(V.path(run["config"]).resolve()): run for run in protocol["runs"]}
    return {
        "status": "PASS", "preflight_evidence_sha256": protocol["preflight_acceptance_sha256"],
        "bindings": {
            "wrapper_sha256": protocol["file_sha256"]["tools/xag_paired_20261018/train.py"],
            "helper_sha256": protocol["file_sha256"]["tools/xyz_c1_repro/train.py"],
            "canonical_config_sha256": protocol["file_sha256"]["configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml"],
            "canonical_baseline_sha256": protocol["baseline_identity_sha256"],
            "source_manifest_sha256": protocol["source_manifest_sha256"],
            "protocol_sha256": digest(protocol_bytes),
            "config_sha256": {name: protocol["file_sha256"][run["config"]] for name, run in configs.items()},
            "sampler_by_config": {name: "legacy_none" for name in configs},
            "cell_by_config": {name: run["cell"] for name, run in configs.items()},
            "modes_by_config": {name: run["modes"] for name, run in configs.items()},
        },
    }


def validate_bundle(bundle):
    p, a = bundle["protocol"], bundle["acceptance"]
    V.require(p["status"] == "FROZEN_READY" and p["canonical_baseline"] == "xyz_c1_stable_s3407", "Bad proposed protocol identity")
    V.require(p["math_policy"] == V.MATH and len(p["immutable_data_stat"]) == 820, "Bad proposed numerical/data contract")
    V.require(len(p["runs"]) == 5 and {r["cell"] for r in p["runs"]} == set(V.CELLS), "Five cells required")
    for run in p["runs"]:
        V.require(run["name"] == f"xag_paired_{run['cell']}_s3407"
                  and run["prefix"] == run["name"].removesuffix("_s3407")
                  and run["modes"] == list(V.CELLS[run["cell"]])
                  and run["physical_gpus"] == PAIRS[run["cell"]]
                  and run["seed"] == 3407 and run["fresh_steps"] == 6000 and run["sampler"] == "legacy_none"
                  and V.path(run["output"]) == OUT / "runs" / run["name"], "Bad proposed arm identity")
        V.load_recipe(run)
    V.require(MUTABLE_POINTER not in p["file_sha256"], "Mutable publication pointer must not become frozen")
    forbidden = {key(OUT / "protocol.json"), key(OUT / "launch_acceptance.json"), key(OUT / "source_archive/manifest.json")}
    V.require(not forbidden.intersection(p["file_sha256"]), "E/P/A/archive hash cycle")
    V.require(a == make_acceptance(p, encoded(p)), "Acceptance is not the exact proposed E -> P -> A binding")
    V.require(bundle["protocol_bytes"] == encoded(p) and bundle["acceptance_bytes"] == encoded(a), "Proposed output bytes differ")


def build_bundle(evidence_path):
    """Read-only input verification; return proposed P/A bytes and archive plan."""
    evidence_path = V.path(evidence_path).resolve()
    V.require(evidence_path.is_relative_to(OUT / "audit"), "Independent preflight E must stay in the new audit namespace")
    evidence_bytes = evidence_path.read_bytes()
    evidence = json.loads(evidence_bytes)
    V.require(evidence.get("status") == "PASS" and isinstance(evidence.get("file_sha256"), dict)
              and evidence["file_sha256"], "Independent full-preflight evidence must be PASS with physical file hashes")
    V.require(evidence.get("kind") == "INDEPENDENT_FIVE_ARM_V3_FULL_VALIDATION_PREFLIGHT"
              and evidence.get("wrapper_sha256") == TRAIN_SHA
              and evidence.get("checks") and all(item.get("pass") is True for item in evidence["checks"]),
              "Actual independent V3 full-validation preflight checks required")
    summaries = evidence.get("runs", [])
    V.require(len(summaries) == 5 and {r["cell"] for r in summaries} == set(V.CELLS)
              and all(r["fresh_steps"] == 50 and r["real_batch_per_rank"] == 512 and r["world_size"] == 2
                      and r["OS_exit_code"] == 0 and r["validation_passes"] == 2
                      and r["validation_batches_per_pass_per_rank"] == 126 for r in summaries),
              "All five real2x512 strict50 plus two complete validation passes required")
    check_map(evidence["file_sha256"])
    baseline_protocol_path = C1 / "protocol.json"
    V.checked_file(baseline_protocol_path, V.C1_PROTOCOL_SHA)
    baseline_protocol = V.read(baseline_protocol_path)
    V.require(len(baseline_protocol["file_sha256"]) == 213 and len(baseline_protocol["immutable_data_stat"]) == 820,
              "Original complete213/820 contract required")
    check_map(baseline_protocol["file_sha256"])
    verify_data_stats(baseline_protocol["immutable_data_stat"])
    V.require(V.sha(ROOT / "tools/xag_paired_20261018/train.py") == TRAIN_SHA, "V3 train entry changed")
    baseline = V.read(BASELINE_IDENTITY)
    V.require(baseline["status"] == "ACTIVE_REPRODUCIBLE_C1_BASELINE" and baseline["run"] == "xyz_c1_stable_s3407"
              and baseline["model_seed"] == 3407 and baseline["step"] == 6000 and baseline["sampler"] == "legacy_none"
              and baseline["protocol_sha256"] == V.C1_PROTOCOL_SHA, "Canonical C1 identity changed")
    for artifact, sha_field in (("checkpoint", "checkpoint_sha256"), ("main_row", "main_row_sha256"),
                                ("full_reproduction", "full_reproduction_sha256")):
        V.checked_file(baseline[artifact], baseline[sha_field])
    proof = V.read(V.path(baseline["full_reproduction"]))
    V.require(proof["status"] == "PASS" and proof["seed"] == 3407 and proof["fresh_optimizer_steps"] == 6000,
              "Complete fresh6000 baseline replay proof required")
    files = dict(baseline_protocol["file_sha256"])
    # The current pointer is a previous-publication snapshot only. Even if E
    # records its pre-launch bytes, future publication must not trip the guard.
    merge(files, {name: value for name, value in evidence["file_sha256"].items() if key(name) != MUTABLE_POINTER})
    formal_paths = [ROOT / "tools/xag_paired_20261018" / name for name in FORMAL_TOOLS]
    V.require(all(p.is_file() for p in formal_paths), "All final tools, including report.py, must exist before sealing")
    # Include all tools/preflight producers and optional README in this namespace.
    new_sources = [p for p in (ROOT / "tools/xag_paired_20261018").iterdir() if p.is_file() and p.suffix in TEXT_SUFFIXES]
    configs = [ROOT / "configs/xag_paired_20261018" / f"xag_paired_{cell}_s3407.yaml" for cell in V.CELLS]
    add_paths(files, [*new_sources, *configs, evidence_path, baseline_protocol_path, BASELINE_IDENTITY,
                      V.path(baseline["checkpoint"]), V.path(baseline["main_row"]), V.path(baseline["full_reproduction"])])
    V.require(files[key(evidence_path)] == digest(evidence_bytes), "Independent E changed after it was read")
    for path in [ROOT / "tools/xag_paired_20261018/train.py", *configs]:
        V.require(evidence["file_sha256"].get(key(path)) == V.sha(path), "Full50 E must bind the actual V3 train/five formal configs: " + key(path))
    snapshots = [REFERENCE / f"state_rank{rank}_step{step:06}.json" for rank in (0, 1) for step in range(0, 6001, 500)]
    streams = [REFERENCE / f"{stem}_rank{rank}.{suffix}" for rank in (0, 1)
               for stem, suffix in (("indices", "i32"), ("batches", "jsonl"), ("completion", "json"), ("effective_runtime", "json"))]
    for path in snapshots + streams:
        V.require(proof["file_sha256"].get(key(path)) == V.sha(path), "Canonical replay proof does not bind required input/state artifact: " + key(path))
    trace_report = REFERENCE / "strict_trace/report.json"
    trace_states = REFERENCE / "strict_trace/states.npz"
    trace = V.read(trace_report)
    V.require(trace["status"] == "COMPLETE" and trace["n_endpoints"] == 2590
              and trace["checkpoint_sha256"] == baseline["checkpoint_sha256"]
              and trace["states_sha256"] == V.sha(trace_states), "Canonical full2590 strict trace is incomplete/different")
    numpy_helper = C1 / "audit/metric_definition_20261018/recompute.py"
    s39 = ROOT / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json"
    baseline_support = [*snapshots, *streams, REFERENCE / "training_metadata.json", REFERENCE / "config_resolved.yaml",
                        REFERENCE / "repro_final.ckpt", trace_report, trace_states, numpy_helper, s39,
                        V.path(baseline["main_row"]).with_name(V.path(baseline["main_row"]).stem + "_receipt.json"),
                        C1 / "audit/final_release_independent_readme_v2_20261018/receipt.json",
                        C1 / "audit/c1_metric_definition_final_20261018/receipt.json"]
    add_paths(files, baseline_support)
    source_manifest = ROOT / "outputs/xag/source_manifest.json"
    source = V.read(source_manifest)
    V.require(source["complete"] is True, "Incomplete original GT pseudo-source cache")
    verify_data_stats({**source["files"], **source["inputs"]})
    runs = []
    for cell in V.CELLS:
        name = f"xag_paired_{cell}_s3407"
        runs.append({"name": name, "cell": cell, "prefix": name.removesuffix("_s3407"), "modes": list(V.CELLS[cell]),
                     "physical_gpus": PAIRS[cell], "config": key(ROOT / "configs/xag_paired_20261018" / f"{name}.yaml"),
                     "output": key(OUT / "runs" / name), "seed": 3407, "fresh_steps": 6000, "sampler": "legacy_none"})
    p = {
        "status": "FROZEN_READY", "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "XAG_PAIRED_TO_STRICT_C1_20261018",
        "user_authority": "2026-10-18 explicit: retrain each of the five arms once at seed3407 for a strictly paired comparison; no multiseed selection",
        "runs": runs, "canonical_baseline": "xyz_c1_stable_s3407", "reference_baseline_output": key(REFERENCE),
        "baseline_protocol_path": key(baseline_protocol_path), "baseline_protocol_sha256": V.C1_PROTOCOL_SHA,
        "baseline_identity_path": key(BASELINE_IDENTITY), "baseline_identity_sha256": V.sha(BASELINE_IDENTITY),
        "baseline_artifacts_sha256": {key(x): files[key(x)] for x in baseline_support},
        "baseline_checkpoint_path": key(baseline["checkpoint"]), "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "baseline_main_row_path": key(baseline["main_row"]), "baseline_main_row_sha256": baseline["main_row_sha256"],
        "baseline_full_reproduction_path": key(baseline["full_reproduction"]), "baseline_full_reproduction_sha256": baseline["full_reproduction_sha256"],
        "baseline_trace_path": key(REFERENCE / "strict_trace"), "baseline_trace_report_path": key(trace_report),
        "baseline_trace_report_sha256": V.sha(trace_report), "baseline_trace_states_path": key(trace_states), "baseline_trace_states_sha256": V.sha(trace_states),
        "numpy_metric_helper_path": key(numpy_helper), "numpy_metric_helper_sha256": V.sha(numpy_helper),
        "reference_S39_row_path": key(s39), "reference_S39_row_sha256": V.sha(s39),
        "source_manifest_path": key(source_manifest), "source_manifest_sha256": V.sha(source_manifest),
        "preflight_evidence_path": str(evidence_path), "preflight_acceptance_sha256": digest(evidence_bytes),
        "file_sha256": dict(sorted(files.items())), "immutable_data_stat": baseline_protocol["immutable_data_stat"],
        "original_frozen_source_count": 186, "inherited_C1_frozen_source_count": 213,
        "math_policy": baseline_protocol["math_policy"], "environment_policy": baseline_protocol["environment_policy"],
        "resource_policy": {"world_size": 2, "batch_per_rank": 512, "workers_per_rank": 12, "prefetch": 1,
                            "Fplus_pairs": [[0, 1], [2, 3], [4, 5]], "F0_pair_sequential": [6, 7],
                            "validation": "full original at500,1000,...6000"},
        "scientific_contract": "Only existing MODEL.XAG_ASSOC/XAG_GNN switches differ from canonical; immutable source metadata enable oracle association. Same None input stream, model shared initialization, data, augmentation, loss, Adam/BF16 and6000 budget. Learned states/gradients/predictions are NOT required to match across scientific arms.",
        "acceptance": "Every rank6000 actual indices and packet small fields/selected canonical-compatible raw hashes match canonical; shared initial tensors/CPU RNG and added F+ initialization match; all actual Adam/checkpoints complete; complete2590 strict scoring/traces repeated and independently decoded.",
        "failure_policy": "Fail closed on any recipe/source/data/finite/input/init/DDP/strict-runtime/ownership/host-memory error; preserve all artifacts and stop owned jobs. No smaller batch, nondeterministic fallback, resume or score selection.",
        "comparison_limits": ["Single seed3407 descriptive paired comparison, no cross-seed reliability claim.",
                              "Oracle GT proxy routing is diagnostic and cannot be deployed.",
                              "New paired weights have new training identities; previous six-grid runs/results remain intact.",
                              "This does not uniquely explain or reproduce missing September C1 weights."],
        "previous_current_metrics_path": MUTABLE_POINTER, "previous_current_metrics_sha256": V.sha(ROOT / MUTABLE_POINTER),
        "previous_current_metrics_is_mutable_publication_pointer": True,
        "source_archive_path": key(OUT / "source_archive"),
        "source_archive_contract": "Text sources/config/JSON copied; binary/checkpoint/data/trace/tape/journal bytes referenced only. Final manifest points to P/A, never the reverse, preventing hash cycles.",
        "reporting": "Original generated nine-column table with PCA6/S39 and fixed C1; five new single3407 rows. No replacement of preselected C1 based on scores.",
    }
    pb = encoded(p); acceptance = make_acceptance(p, pb)
    bundle = {"protocol": p, "acceptance": acceptance, "protocol_bytes": pb, "acceptance_bytes": encoded(acceptance),
              "evidence_bytes": evidence_bytes}
    validate_bundle(bundle)
    check_map(p["file_sha256"])
    return bundle


def archive_plan(bundle):
    """List text copies and unchanged references; no binary duplication."""
    records = {}
    for name, expected in bundle["protocol"]["file_sha256"].items():
        path = V.path(name)
        archived = (not Path(name).is_absolute() and ".." not in Path(name).parts
                    and path.suffix in TEXT_SUFFIXES and path.stat().st_size <= 16 << 20)
        records[name] = {"sha256": expected, "bytes": path.stat().st_size,
                         "archive_path": "files/" + name if archived else None}
    return records


def commit_bundle(bundle):
    """Exclusively create formal outputs after all validation; never launch jobs."""
    validate_bundle(bundle)
    protocol = bundle["protocol"]
    forbidden = [OUT / name for name in ("protocol.json", "launch_acceptance.json", "source_archive", "status.json", "execution.lock", "freeze_receipt.json")]
    forbidden += [V.path(run["output"]) for run in protocol["runs"]]
    V.require(not any(os.path.lexists(p) for p in forbidden), "Formal namespace already occupied; preserve it")
    check_map(protocol["file_sha256"]); verify_data_stats(protocol["immutable_data_stat"])
    stage = OUT / "audit" / ("freeze_archive_stage_" + str(os.getpid()))
    stage.mkdir(parents=True, exist_ok=False)
    records = archive_plan(bundle)
    for name, record in records.items():
        if record["archive_path"]:
            content = V.path(name).read_bytes()
            V.require(digest(content) == record["sha256"], "Source changed while archiving: " + name)
            destination = stage / record["archive_path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream: stream.write(content)
    for name, content in (("protocol.json", bundle["protocol_bytes"]), ("launch_acceptance.json", bundle["acceptance_bytes"]),
                          ("preflight_evidence.json", bundle["evidence_bytes"])):
        with (stage / name).open("xb") as stream: stream.write(content)
    manifest = {"status": "COMPLETE_TEXT_SOURCE_ARCHIVE", "files": records,
                "protocol_sha256": digest(bundle["protocol_bytes"]), "launch_acceptance_sha256": digest(bundle["acceptance_bytes"]),
                "preflight_evidence_sha256": digest(bundle["evidence_bytes"]),
                "no_large_data_checkpoint_trace_tape_journal_copy": True}
    with (stage / "manifest.json").open("xb") as stream: stream.write(encoded(manifest))
    check_map(protocol["file_sha256"])
    stage.rename(OUT / "source_archive")
    for name, content in (("protocol.json", bundle["protocol_bytes"]), ("launch_acceptance.json", bundle["acceptance_bytes"])):
        with (OUT / name).open("xb") as stream: stream.write(content)
    # Actual consumer verification after publication. Failure leaves evidence intact.
    V.verify_protocol(OUT / "protocol.json")
    receipt = {"status": "FROZEN_READY_NOT_LAUNCHED", "protocol_sha256": digest(bundle["protocol_bytes"]),
               "acceptance_sha256": digest(bundle["acceptance_bytes"]), "source_archive_manifest_sha256": V.sha(OUT / "source_archive/manifest.json"),
               "frozen_file_count": len(records), "inherited_sources": 213, "immutable_data_stats": 820,
               "no_training_or_model_inference_started": True}
    with (OUT / "freeze_receipt.json").open("xb") as stream: stream.write(encoded(receipt))
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--preflight-evidence", required=True)
    parser.add_argument("--plan-output", type=Path, help="New audit-only plan destination; default mode never writes formal P/A")
    parser.add_argument("--commit", action="store_true", help="Exclusively publish new formal P/A/source archive, never start training")
    args = parser.parse_args(argv)
    V.require(args.commit or args.plan_output is not None, "Specify an audit plan destination or explicit --commit")
    bundle = build_bundle(args.preflight_evidence)
    if args.plan_output is not None:
        V.require(args.plan_output.resolve().is_relative_to(OUT / "audit"), "Plan must be saved only in the new audit namespace")
        plan = {"status": "VERIFIED_PLAN_NOT_FORMAL_PUBLICATION", "protocol": bundle["protocol"], "acceptance": bundle["acceptance"],
                "archive_records": archive_plan(bundle), "no_training_started": True}
        V.save_new(args.plan_output, plan)
    if args.commit:
        print(json.dumps(commit_bundle(bundle), indent=2))
    else:
        print(json.dumps({"status": "VERIFIED_PLAN_ONLY", "output": str(args.plan_output), "no_formal_protocol_or_acceptance_written": True}))


if __name__ == "__main__":
    main()
