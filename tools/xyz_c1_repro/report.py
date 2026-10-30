#!/usr/bin/env python3
"""Publish the predesignated baseline only after complete fresh reproduction."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/xyz_c1_repro/20261018"
XAG = ROOT / "outputs/xag/reviews/reconfirm_metrics_20261018"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_table(sources, output):
    aliases = OUT / "release" / "table_sources"
    aliases.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(ROOT / "tools/report_table.py")]
    labels = []
    for key, path, label in sources:
        alias = aliases / f"{key}_main_row.json"
        if not alias.is_symlink():
            if alias.exists():
                raise FileExistsError(alias)
            alias.symlink_to(path)
        if alias.resolve() != path.resolve():
            raise RuntimeError("Display row alias changed")
        command.append(str(aliases / key))
        labels.extend(["--label", str(aliases / key) + "=" + label])
    command.extend(labels)
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    output.write_text(result.stdout)
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    while True:
        state = read(OUT / "status.json")
        if state["status"] == "COMPLETE_REPRODUCIBLE_BASELINE_AND_ROWS" and state["baseline_ready"]:
            break
        if state["status"].startswith("FAILED"):
            raise RuntimeError("Reproduction failed; no baseline will be published")
        if not args.wait:
            raise RuntimeError("Full fresh6000 reproducibility is not complete")
        time.sleep(10)
    from reproduce import load_reference, validate_reference_products, compare_traces
    reference = load_reference()
    validated_baseline = validate_reference_products(reference)
    reproduction = read(OUT / "full_reproduction.json")
    reconfirmed = read(XAG / "final_receipt.json")
    expected_xag_receipt = "40b943d744ef65180ba181569ecf8cfab0d3ea68e88c6f4d503b143707367555"
    if sha(XAG / "final_receipt.json") != expected_xag_receipt:
        raise RuntimeError("Independent XAG completion receipt changed")
    if reproduction["status"] != "PASS" or reconfirmed["status"] != "COMPLETE":
        raise RuntimeError("Complete training and XAG verification required")
    if not reconfirmed["all_six_strict_input_and_prediction_arrays_bitwise_repeated"]:
        raise RuntimeError("Existing-arm evaluation not reproducible")
    if not reconfirmed["all_six_fresh_strict_rows_match_both_independent_strict_passes"]:
        raise RuntimeError("Existing-arm rows not independently verified")
    protocol = read(OUT / "protocol.json")
    # Do not choose among the three scores: designation is fixed before launch.
    if protocol["canonical_baseline"] != "xyz_c1_stable_s3407":
        raise RuntimeError("Canonical baseline identity changed")
    stable = OUT / "main_rows/xyz_c1_stable_main_row_step6000.json"
    replay = OUT / "main_rows/xyz_c1_stable_replay_main_row_step6000.json"
    intervention = OUT / "main_rows/xyz_c1_dedicated_main_row_step6000.json"
    rows = [read(p) for p in (stable, replay, intervention)]
    if rows[0] != validated_baseline:
        raise RuntimeError("Baseline display source differs from its verified product")
    if rows[0]["two_seed_mean"] != rows[1]["two_seed_mean"]:
        raise RuntimeError("Baseline and verification copy metrics differ")
    if any(set(row["per_seed"]) != {"3407"} for row in rows):
        raise RuntimeError("Unexpected model seed")
    from verify_final import verify_run
    failures, _, _ = verify_run(ROOT / protocol["runs"][2]["output"], "dedicated")
    if failures:
        raise RuntimeError("Sampling intervention terminal evidence changed: " + str(failures[:5]))
    for run, row, path in zip(protocol["runs"], rows, (stable, replay, intervention)):
        receipt = read(path.with_name(path.stem + "_receipt.json"))
        directory = ROOT / run["output"]
        trace = read(directory / "strict_trace/report.json")
        checkpoint = Path(row["per_seed"]["3407"]["ckpt"])
        seed = row["per_seed"]["3407"]
        if (seed["step"] != 6000 or seed["n_frames"] != 2590
                or seed["sequences"] != ["zgz_global", "zgz_local"]
                or set(seed["per_seq"]) != {"zgz_global", "zgz_local"}
                or trace["status"] != "COMPLETE" or trace["n_endpoints"] != 2590
                or receipt["row_sha256"] != sha(path)
                or receipt["wrapper_sha256"] != sha(ROOT / "tools/xyz_c1_repro/evaluate.py")
                or trace["checkpoint_sha256"] != sha(checkpoint)
                or trace["states_sha256"] != sha(directory / "strict_trace/states.npz")
                or trace["config_sha256"] != sha(ROOT / run["config"])
                or trace["source_sha256"] != sha(ROOT / "tools/xyz_c1_repro/trace.py")):
            raise RuntimeError("Final C1 main-row/trace producer or checkpoint changed")
        for block in ("overall", "local", "global"):
            if row["two_seed_mean"][block] != seed[block] or any(
                    not math.isfinite(value) for value in seed[block].values()):
                raise RuntimeError("C1 row is not one finite full-sequence seed")
        import numpy as np
        with np.load(directory / "strict_trace/states.npz") as full_trace:
            compare_traces(full_trace, full_trace)
        for sequence in trace["per_sequence"]:
            if any(sequence[metric] != row["per_seed"]["3407"]["per_seq"][sequence["seq"]][metric]
                   for metric in row["two_seed_mean"]["overall"]):
                raise RuntimeError("C1 trace differs from its generated row")
    sources = [
        ("S39", ROOT / "outputs/semkine/dt3_dz_l3_w200cfg_main_row.json",
         "S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *"),
        ("C1_baseline", stable, "XYZ C1（确定性baseline，seed3407）"),
    ]
    labels = {
        "a0f1": "XAG：原关联＋增强GNN（单seed3407）",
        "a1f0": "XAG：部位GT对应＋原GNN（oracle，单seed3407）",
        "a1f1": "XAG：部位GT对应＋增强GNN（oracle，单seed3407）",
        "a2f0": "XAG：表面GT对应＋原GNN（oracle，单seed3407）",
        "a2f1": "XAG：表面GT对应＋增强GNN（oracle，单seed3407）",
    }
    verified_xag = {}
    for cell in ("a0f0", *labels):
        path = XAG / f"main_rows/xag_{cell}_main_row_step6000.json"
        job = reconfirmed["jobs"][f"xag_{cell}_s3407"]
        receipt_path = path.with_name(path.stem + "_receipt.json")
        audits = list((XAG / "main_rows").glob("*.json"))
        if (sha(path) != job["row_sha256"] or sha(receipt_path) != job["wrapper_receipt_sha256"]
                or not any(sha(p) == job["independent_audit_sha256"] for p in audits)):
            raise RuntimeError("Reconfirmed XAG row or independent receipt changed")
        row = read(path)
        if (set(row["per_seed"]) != {"3407"} or row["per_seed"]["3407"]["n_frames"] != 2590
                or row["per_seed"]["3407"]["step"] != 6000 or row["two_seed_mean"] != job["accuracy"]):
            raise RuntimeError("Unexpected XAG model seed")
        verified_xag[cell] = path
    for cell, label in labels.items():
        sources.append((cell, verified_xag[cell], label))
    release = OUT / "release"
    release.mkdir(exist_ok=True)
    if (release / "release.json").exists():
        raise FileExistsError("Preserve existing finalized release")
    active = ROOT / "outputs/xyz_c1_repro/baseline.json"
    alias = ROOT / "outputs/semkine/xyz_c1_baseline_main_row.json"
    if active.exists() or active.is_symlink() or alias.exists() or alias.is_symlink():
        raise FileExistsError("Preserve existing active-baseline pointer or table alias")
    table_command = generate_table(sources, release / "main_table.md")
    (release / "main_table_notes.md").write_text(
        "C1为新确定性None采样基线；五格XAG为原dedicated采样/旧数值训练权重的严格重评。"
        "同seed不等于重新配对训练，差值只作描述。PCA6与S39保留原历史参考行。\n")
    control_sources = sources[:2] + [
        ("C1_sampler_control", intervention, "XYZ C1：仅换独立采样流（诊断，seed3407）"),
        ("old_XAG_C1", verified_xag["a0f0"],
         "既有XAG A0F0（旧训练，严格重评，seed3407）"),
    ]
    control_command = generate_table(control_sources, release / "sampling_control_table.md")
    delta = {block: {metric: rows[2]["two_seed_mean"][block][metric] - rows[0]["two_seed_mean"][block][metric]
                     for metric in rows[0]["two_seed_mean"][block]}
             for block in ("overall", "local", "global")}
    checkpoint = Path(rows[0]["per_seed"]["3407"]["ckpt"])
    identity = {
        "status": "ACTIVE_REPRODUCIBLE_C1_BASELINE", "model_seed": 3407, "step": 6000,
        "run": protocol["canonical_baseline"], "sampler": "legacy_none",
        "checkpoint": str(checkpoint), "checkpoint_sha256": sha(checkpoint),
        "config": "configs/xyz_c1_repro_20261018/xyz_c1_stable_s3407.yaml",
        "main_row": str(stable.relative_to(ROOT)), "main_row_sha256": sha(stable),
        "protocol": str((OUT / "protocol.json").relative_to(ROOT)), "protocol_sha256": sha(OUT / "protocol.json"),
        "full_reproduction": str((OUT / "full_reproduction.json").relative_to(ROOT)),
        "full_reproduction_sha256": sha(OUT / "full_reproduction.json"),
        "runtime_identity": str((OUT / "runtime_identity.json").relative_to(ROOT)),
        "replay_is_not_another_seed_or_average": True,
        "meaning": "C1 reference for subsequent association/GNN work; original S39 and published PCA6 reference unchanged.",
        "reproduction_entry": "tools/xyz_c1_repro/reproduce.py; native semkine/train.py alone does not enable this strict runtime",
    }
    report = {
        "status": "COMPLETE", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": identity, "main_table_command": table_command,
        "sampling_control_table_command": control_command,
        "conditional_sampling_effect_dedicated_minus_none_mm": delta,
        "historical_causal_limit": "Not a unique decomposition of the missing September weight's11.57mm gap. This identifies a fixed-seed strict-runtime sampling intervention plus a reproducible new baseline.",
        "comparison_boundary": "The five XAG weights retain their previous nondeterministic dedicated-stream training identities; strict re-evaluation is verified but does not retroactively create paired strict training with the new None baseline.",
        "source_sha256": sha(__file__),
        "row_sources": {str(path.relative_to(ROOT)): sha(path) for _, path, _ in sources + control_sources},
        "verification_sha256": {str((OUT / "full_reproduction.json").relative_to(ROOT)): sha(OUT / "full_reproduction.json"),
                                str((XAG / "final_receipt.json").relative_to(ROOT)): sha(XAG / "final_receipt.json")},
    }
    created_alias, created_active = False, False
    try:
        alias.symlink_to(stable)
        created_alias = True
        with active.open("x") as stream:
            created_active = True
            stream.write(json.dumps(identity, indent=2) + "\n")
        with (release / "release.json").open("x") as stream:
            stream.write(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    except BaseException:
        if created_active:
            active.unlink(missing_ok=True)
        if created_alias:
            alias.unlink(missing_ok=True)
        raise
    print((release / "main_table.md").read_text())
    print("Active C1 baseline:", active)


if __name__ == "__main__":
    main()
