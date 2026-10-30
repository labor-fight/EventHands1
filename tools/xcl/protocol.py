#!/usr/bin/env python3
"""Register exactly five reused and five fresh single-seed XCL identities."""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from tools.xcl.common import (BASELINE, CANONICAL, CELLS, OUT, REFERENCE, SOURCE_MANIFEST,
                             now, path, read, require, save, sha)

CONFIG_DIR = ROOT / "configs/xcl_20261019"

def make_configs():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    original = yaml.safe_load(CANONICAL.read_text())
    runs = []
    for cell, (assoc, gnn, geom, gpus) in CELLS.items():
        name = f"xcl_{cell}_s3407"
        cfg = copy.deepcopy(original)
        cfg["MODEL"].update(XAG_ASSOC=assoc, XAG_GNN=gnn, XYZ_GEOM=geom)
        cfg["DATA"]["XAG_SOURCE_ROOT"] = str(ROOT / "outputs/xag/sources")
        cfg["XCL"] = {"ENABLED": True, "CELL": cell}
        cfg["TRAIN"].update(RUN_NAME=name, OUTPUT_DIR=str(OUT / "runs" / name))
        cfg["EVAL"]["OUTPUT_DIR"] = str(OUT / "runs" / name / "eval")
        config = CONFIG_DIR / (name + ".yaml")
        content = yaml.safe_dump(cfg)
        if config.exists():
            require(config.read_text() == content, "Existing formal config differs")
        else:
            config.write_text(content)
        pre = copy.deepcopy(cfg)
        pre_name = name + "_preflight50"
        pre["TRAIN"].update(RUN_NAME=pre_name, MAX_STEPS=50, VAL_CHECK_INTERVAL=25,
                            OUTPUT_DIR=str(OUT / "preflight" / "runs" / pre_name))
        pre["EVAL"]["OUTPUT_DIR"] = str(OUT / "preflight" / "runs" / pre_name / "eval")
        preconfig = CONFIG_DIR / (pre_name + ".yaml")
        precontent = yaml.safe_dump(pre)
        if preconfig.exists():
            require(preconfig.read_text() == precontent, "Existing preflight config differs")
        else:
            preconfig.write_text(precontent)
        runs.append({"name": name, "cell": cell, "reuse": False, "seed": 3407, "step": 6000,
                     "prefix": name.removesuffix("_s3407"), "config": str(config.relative_to(ROOT)),
                     "config_sha256": sha(config), "output": str((OUT / "runs" / name).relative_to(ROOT)),
                     "checkpoint": str((OUT / "runs" / name / "repro_final.ckpt").relative_to(ROOT)),
                     "preflight_config": str(preconfig.relative_to(ROOT)),
                     "preflight_output": str((OUT / "preflight" / "runs" / pre_name).relative_to(ROOT)),
                     "physical_gpus": list(gpus), "oracle": assoc == "full_oracle" or gnn.startswith("oracle_")})
    return runs

def build():
    new_runs = make_configs()
    baseline = read(BASELINE)
    require(sha(baseline["checkpoint"]) == baseline["checkpoint_sha256"], "Canonical C1 checkpoint changed")
    reused = [{"name": baseline["run"], "cell": "a0f0", "reuse": True, "seed": 3407, "step": 6000,
               "prefix": "xyz_c1_stable", "output": str(REFERENCE.relative_to(ROOT)),
               "config": baseline["config"], "config_sha256": sha(baseline["config"]),
               "checkpoint": baseline["checkpoint"], "checkpoint_sha256": baseline["checkpoint_sha256"], "oracle": False}]
    old_status = read("outputs/xag_paired/20261018/status.json")
    for cell in ("a0f1", "a1f0", "a1f1", "a2f1"):
        name = f"xag_paired_{cell}_s3407"
        record = old_status["runs"][name]
        require(record["exit_code"] == 0, "Reused paired training did not exit successfully")
        directory = ROOT / "outputs/xag_paired/20261018/runs" / name
        for rank in range(2):
            c = read(directory / f"completion_rank{rank}.json")
            require(c["status"] == "COMPLETE" and c["step"] == c["batches"] == 6000, "Incomplete reused model")
        checkpoint = directory / "repro_final.ckpt"
        digest = sha(checkpoint)
        require(digest == read(directory / "completion_rank0.json")["final_checkpoint_sha256"], "Reused checkpoint differs")
        config = ROOT / "configs/xag_paired_20261018" / (name + ".yaml")
        reused.append({"name": name, "cell": cell, "reuse": True, "seed": 3407, "step": 6000,
                       "prefix": name.removesuffix("_s3407"), "output": str(directory.relative_to(ROOT)),
                       "config": str(config.relative_to(ROOT)), "config_sha256": sha(config),
                       "checkpoint": str(checkpoint.relative_to(ROOT)), "checkpoint_sha256": digest,
                       "oracle": cell[1] in ("1", "2")})
    names = set()
    for pattern in ("model/*.py", "semkine/*.py", "tools/xcl/**/*.py", "configs/xcl_20261019/*.yaml",
                    "tools/xyz_c1_repro/train.py", "tools/make_s36_row.py", "tools/report_table.py",
                    "tools/xag_paired_20261018/verify_final.py",
                    "config.py", "config/*.py", "tests/test_xcl*.py"):
        names.update(str(p.relative_to(ROOT)) for p in ROOT.glob(pattern) if p.is_file())
    names.update([str(BASELINE.relative_to(ROOT)), str(CANONICAL.relative_to(ROOT)), str(SOURCE_MANIFEST.relative_to(ROOT))])
    names.add("assets/mano_right.npz")
    names.update(r["config"] for r in reused)
    c1_protocol = read("outputs/xyz_c1_repro/20261018/protocol.json")
    require(sha("assets/mano_right.npz") == c1_protocol["file_sha256"]["assets/mano_right.npz"], "Canonical MANO asset changed")
    # The old 820-stat bundle also mixed in four source files. Those are now
    # migrated/revised code, covered by XCL source SHA instead of data identity.
    old_source_stats = {str(ROOT / n) for n in ("semkine/xag_sources.py", "tools/xa/step1_common.py",
                        "tools/xa/step1_source_labels.py", "tools/xag/build_sources.py")}
    data = {name: [identity["bytes"], identity["mtime_ns"]] for name, identity in c1_protocol["immutable_data_stat"].items()
            if name not in old_source_stats}
    protocol = {"study": "XCL_20261019", "status": "REGISTERED", "created_at_utc": now(), "seed": 3407,
                "runs": reused + new_runs, "canonical_trace": {"path": str((REFERENCE / "strict_trace/states.npz").relative_to(ROOT)),
                                                               "sha256": sha(REFERENCE / "strict_trace/states.npz")},
                "reference_baseline_output": str(REFERENCE.relative_to(ROOT)),
                "depth_binding": {"stage": "Before A3 preflight, bound again in five-cell formal acceptance",
                                  "binding_path": str((OUT / "depth_binding.json").relative_to(ROOT)),
                                  "manifest_path": str((OUT / "depth_manifest.json").relative_to(ROOT)),
                                  "sequences": read(SOURCE_MANIFEST)["sequences"]},
                "file_sha256": {name: sha(name) for name in sorted(names)}, "immutable_data_stat": data,
                "contract": {"steps": 6000, "sampler": "legacy_none", "world_size": 2,
                             "per_rank_batch": 512, "precision": "bf16", "workers": 12, "prefetch": 1,
                             "fresh_only": True, "selection": "fixed6000; no seed selection or repeat"},
                "training_queues": {"2,3": ["a2f0", "a3f0"], "4,5": ["c0", "a0foo"], "6,7": ["a0fod"]},
                "evaluation_gpus": [4, 5, 6, 7], "latency_rows_serial_without_training_or_trace": True,
                "thresholds_mm": {"local": 5, "global": 8, "wrist": 10},
                "inference": "Single-seed descriptive only. E1/E2/E3 and >=2 of5000/5500/6000 same direction required.",
                "fomega_contract": "Frozen rel/vin/edge/local layers; original residual features; new projection plus downstream slots/head learn; original zero-event gate retained."}
    return protocol

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs-only", action="store_true")
    args = parser.parse_args()
    if args.configs_only:
        print([r["config"] for r in make_configs()])
        return
    protocol = build()
    destination = OUT / "protocol.json"
    require(not destination.exists(), "Preserve existing protocol; explicit new revision required")
    archive = OUT / "source_archive"
    archive.mkdir(parents=True, exist_ok=False)
    for name, digest in protocol["file_sha256"].items():
        target = archive / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
        require(sha(target) == digest, "Source archive copy differs")
    save(destination, protocol, exclusive=True)
    print(destination)

if __name__ == "__main__":
    main()
