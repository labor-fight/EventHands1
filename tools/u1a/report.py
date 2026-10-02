#!/usr/bin/env python3
"""Explicit-last U1a rows, compatible with tools/report_table.py.

No selection file is read and no evaluation or training is launched. Each run
must already contain evalx_val_core_last[_VARIANT].json and its NPZ companion.
Use --measure-cost explicitly to run the existing batch-one cost protocol, or
--cost-row to reuse a cost artifact for the SAME arm. --dry-run never profiles
and never writes files; it needs --cost-row.

Example after explicit last evaluation of both seeds:
  python tools/u1a/report.py --arm u1a_shared_2k --mode shared --budget 2000 \
    --runs outputs/semkine/u1a_shared_2k_s3407 outputs/semkine/u1a_shared_2k_s3408 \
    --variant tf_pert --measure-cost \
    --baseline-runs outputs/semkine/s38_spmeas_s3407 outputs/semkine/s38_spmeas_s3408 \
    --baseline-cost-row outputs/semkine/s38_spmeas_tf_pert_main_row.json

Repeat with untied and/or --budget 6000; --arm is an explicit output name.
The reference is a fixed S38 last-checkpoint comparison, not a re-selected arm.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs" / "semkine"
METRICS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm",
           "root_rot_deg", "transl_mm")
PARTS = ("overall", "local", "global")
SEQUENCES = ("zgz_global", "zgz_local")
SUBJECTS = ("ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf")
COST_KEYS = ("latency_ms_raw", "latency_ms_scaled_full1p75", "macs_forward_packet", "params_total")
sys.path[:0] = [str(REPO), str(REPO / "model"), str(REPO / "tools")]
from report_table import BASELINE, HEADER, RULE  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def path_at_repo(value):
    p = Path(value).expanduser()
    return (p if p.is_absolute() else REPO / p).resolve()


def load_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def finite(value, label):
    value = float(value)
    require(math.isfinite(value), f"non-finite {label}")
    return value


def mean_entry(value, label):
    # evalx per-sequence entries are [mean, block-bootstrap CI low, CI high].
    return finite(value[0] if isinstance(value, list) else value, label)


def load_run(run, *, mode, budget, variant):
    run = path_at_repo(run)
    require(run.is_dir(), f"missing run: {run}")
    metadata_path = run / "training_metadata.json"
    metadata = load_json(metadata_path)
    seed = str(int(metadata["seed"]))
    require(int(metadata["max_steps"]) == budget, f"{run.name}: metadata budget is not {budget}")
    trained = metadata.get("train_sequences", [])
    subjects = sorted({s.split("_")[0] for s in trained})
    require(len(trained) == 72 and subjects == list(SUBJECTS),
            f"{run.name}: only the fixed 9-subject/72-sequence protocol is accepted")
    require(metadata.get("splits_manifest", "splits_semkine.json") == "splits_semkine.json",
            f"{run.name}: obsolete or alternate training manifest")
    config_path = run / "config_resolved.yaml"
    if not config_path.is_file():
        config_path = path_at_repo(metadata["config_path"])
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mc = cfg["MODEL"]
    require(mc.get("ROOT_MEAS") == "abs" and mc.get("FINGER_MEAS") == "abs",
            f"{run.name}: U1a/S38 requires absolute root and residual45 measurements")
    require(mc.get("ENCODER") == "sparse_pyramid", f"{run.name}: encoder is not the S38 sparse pyramid")
    if mode == "baseline":
        require(not mc.get("U1A_MODE"), f"{run.name}: S38 reference cannot contain a U1a readout")
    else:
        require(mc.get("U1A_MODE") == mode, f"{run.name}: configuration mode is not {mode}")
        require(mc.get("U1A_FREEZE_ENCODER") is True, f"{run.name}: encoder-freeze flag is missing")
    manifest = path_at_repo(cfg["DATA"]["SPLITS_MANIFEST"])
    require(manifest.name == "splits_semkine.json" and manifest.is_file(),
            f"{run.name}: fixed splits_semkine.json is unavailable")
    tag = "evalx_val_core_last" + (f"_{variant}" if variant else "")
    evaluation_path, arrays_path = run / f"{tag}.json", run / f"{tag}.npz"
    require(evaluation_path.is_file() and arrays_path.is_file(),
            f"{run.name}: explicit last evaluation JSON and NPZ are required: {tag}")
    js = load_json(evaluation_path)
    require(js.get("run") == run.name, f"{evaluation_path}: mismatched run identity")
    require(js.get("split") == "val_core", f"{run.name}: only val_core zgz is accepted")
    require("selection_drift_mm" not in js, f"{run.name}: this is a selected evaluation, not explicit last")
    require(int(js["step"]) == budget, f"{run.name}: evaluated step {js['step']} is not explicit {budget}")
    require(path_at_repo(js["manifest"]) == manifest, f"{run.name}: evaluation manifest differs from training")
    window = js.get("window", {})
    require(window.get("mode") == "fixed" and int(window.get("ms", -1)) == 50,
            f"{run.name}: expected causal fixed 50-ms evidence / recursive protocol")
    for token, diagnostic in (("tf", "tf"), ("pert", "perturb")):
        if token in variant.split("_"):
            require(diagnostic in js, f"{run.name}: requested {diagnostic} diagnostic is missing")
    checkpoints = {}
    for checkpoint in run.glob("*step=*.ckpt"):
        match = re.search(r"step=(\d+)", checkpoint.name)
        if match:
            checkpoints.setdefault(int(match.group(1)), []).append(checkpoint.resolve())
    require(checkpoints and max(checkpoints) == budget,
            f"{run.name}: newest numbered checkpoint is not the requested budget {budget}")
    checkpoint = path_at_repo(js["ckpt"])
    require(checkpoint in checkpoints[budget] and checkpoint.is_file(),
            f"{run.name}: evaluation did not use the actual latest numbered checkpoint")
    model = js["model"]
    sequences = sorted(set(model) - {"overall", "n_frames"})
    require(sequences == list(SEQUENCES), f"{run.name}: expected exactly zgz_global and zgz_local")
    counts = {s: int(model[s]["n_frames"]) for s in SEQUENCES}
    require(all(n > 0 for n in counts.values()) and sum(counts.values()) == int(model["n_frames"]),
            f"{run.name}: invalid frame accounting")

    def aggregate(ss, key):
        total = sum(counts[s] for s in ss)
        return sum(mean_entry(model[s][key], f"{run.name}/{s}/{key}") * counts[s] for s in ss) / total

    parts = {"local": {k: aggregate(["zgz_local"], k) for k in METRICS},
             "global": {k: aggregate(["zgz_global"], k) for k in METRICS},
             "overall": {k: finite(model["overall"][k], f"{run.name}/overall/{k}") for k in METRICS}}
    for key in METRICS:
        # evalx overall uses float32 per-step summation; allow its small rounding difference.
        require(abs(aggregate(SEQUENCES, key) - parts["overall"][key]) < 0.01,
                f"{run.name}: overall {key} disagrees with frame-weighted sequence means")
    per_seed = {"step": budget, "ckpt": str(checkpoint), "run": str(run),
                "sequences": list(SEQUENCES), "n_frames": sum(counts.values()),
                "frames_per_seq": counts, **parts}
    provenance = {"run": str(run), "seed": seed, "step": budget, "mode": mode,
                  "checkpoint": {"path": str(checkpoint), "sha256": file_hash(checkpoint)},
                  "config": {"path": str(config_path), "sha256": file_hash(config_path)},
                  "training_metadata": {"path": str(metadata_path), "sha256": file_hash(metadata_path)},
                  "manifest": {"path": str(manifest), "sha256": file_hash(manifest)},
                  "evaluation": {"path": str(evaluation_path), "sha256": file_hash(evaluation_path)},
                  "diagnostic_arrays": {"path": str(arrays_path), "sha256": file_hash(arrays_path)},
                  "training_provenance": metadata.get("provenance", {}),
                  "window": window, "available_diagnostics": sorted(set(js) & {"tf", "hold", "noevents", "perturb"})}
    if mode != "baseline":
        init_path = path_at_repo(mc["INIT_FROM"])
        require(init_path.is_file(), f"{run.name}: matched initialization checkpoint is unavailable")
        provenance["initialization"] = {"path": str(init_path), "sha256": file_hash(init_path)}
        registry_path = REPO / "outputs" / "u1a" / "initialization.json"
        registry = load_json(registry_path)
        pair = registry["pairs"][seed]
        require(path_at_repo(pair["arms"][mode]["init"]) == init_path,
                f"{run.name}: INIT_FROM differs from its registered matched initialization")
        anchor = path_at_repo(pair["anchor"])
        require(anchor.is_file() and file_hash(anchor) == pair["anchor_sha256"],
                f"{run.name}: frozen S38 source checkpoint changed since initialization")
        require(registry["split_sha256"] == provenance["manifest"]["sha256"],
                f"{run.name}: initialization was prepared with a different manifest")
        provenance["initialization_registry"] = {"path": str(registry_path), "sha256": file_hash(registry_path)}
        provenance["frozen_s38_source"] = {"path": str(anchor), "sha256": pair["anchor_sha256"],
                                          "run": str(anchor.parent), "registered_common_tensors": pair["loaded_common_tensors"]}
    return seed, per_seed, provenance, js, cfg


def aggregate_runs(runs, *, mode, budget, variant):
    per_seed, provenance, diagnostics, configs = {}, {}, {}, {}
    for run in runs:
        seed, result, source, js, cfg = load_run(run, mode=mode, budget=budget, variant=variant)
        require(seed not in per_seed, f"duplicate seed {seed}; it must not be averaged twice")
        per_seed[seed], provenance[seed], diagnostics[seed], configs[seed] = result, source, js, cfg
    require(per_seed, "at least one explicit run is required")
    per_seed = dict(sorted(per_seed.items(), key=lambda entry: int(entry[0])))
    first = next(iter(per_seed))
    for seed in per_seed:
        require(per_seed[seed]["frames_per_seq"] == per_seed[first]["frames_per_seq"],
                "seeds have different evaluated frame counts")
        require(provenance[seed]["manifest"]["sha256"] == provenance[first]["manifest"]["sha256"],
                "seeds have different split manifests")
        # Seed/run/init paths may differ; the measurement convention may not.
        for key in ("ROOT_REF", "ROOT_FILTER_GAIN", "FINGER_FILTER_GAIN", "U1A_MODE", "U1A_HIDDEN"):
            require(configs[seed]["MODEL"].get(key) == configs[first]["MODEL"].get(key),
                    f"seeds disagree on MODEL.{key}")
    mean = {part: {key: statistics.fmean(per_seed[s][part][key] for s in per_seed) for key in METRICS}
            for part in PARTS}
    return per_seed, mean, provenance, diagnostics, configs[first]


def imported_cost(path, *, arm, per_seed):
    path = path_at_repo(path)
    source = load_json(path)
    require(source.get("arm") == arm, f"{path}: cost source arm is not {arm}; costs cannot transfer across arms")
    require(source.get("ckpt_rule") == "last", f"{path}: only a last-checkpoint cost row is accepted")
    for seed, result in per_seed.items():
        entry = source.get("per_seed", {}).get(seed)
        require(entry and int(entry["step"]) == result["step"], f"{path}: no matching last step for seed {seed}")
        require(path_at_repo(entry["ckpt"]) == path_at_repo(result["ckpt"]),
                f"{path}: seed {seed} cost-row checkpoint differs from the evaluation")
    cost = {key: finite(source[key], f"{path}/{key}") for key in COST_KEYS}
    require(all(value > 0 for value in cost.values()), f"{path}: costs must be finite and positive")
    require(cost["params_total"].is_integer(), f"{path}: parameter count must be an integer")
    cost["params_total"] = int(cost["params_total"])
    anchor = source.get("latency_anchor_ms", source.get("anchor_full_raw_ms"))
    require(anchor is not None and float(anchor) > 0, f"{path}: full-model timing anchor missing")
    require(abs(cost["latency_ms_scaled_full1p75"] - cost["latency_ms_raw"] * 1.75 / float(anchor)) < 1e-6,
            f"{path}: latency scaling is not the full-model 1.75-ms anchor")
    cost["latency_anchor_ms"] = float(anchor)
    cost["macs_forward_packet_note"] = source.get("macs_forward_packet_note", "thop standard-layer MACs; full forward_packet")
    cost["cost_provenance"] = {"method": "reuse_same_arm_recorded_cost", "path": str(path),
                               "sha256": file_hash(path),
                               "note": "accuracy is recomputed from explicit last eval JSONs; only cost columns are reused"}
    return cost


def measure_cost(cfg, per_seed, provenance):
    # Lazy imports ensure --dry-run / --cost-row never allocate a model or touch CUDA.
    import torch
    from model import MNISTModel
    from tools.x1001 import evalx

    require(torch.cuda.is_available(), "--measure-cost requires the recorded CUDA timing environment")
    device = torch.device("cuda")
    first = next(iter(per_seed))
    checkpoint = per_seed[first]["ckpt"]
    model = MNISTModel.load_from_checkpoint(checkpoint, cfg=cfg, map_location=device).to(device).eval()
    with torch.no_grad():
        anchor = float(evalx.latency_anchor(device))
        latency = float(evalx.latency_model(model, cfg, device))
        macs = float(evalx.macs_of(cfg))
    require(anchor > 0 and latency > 0 and math.isfinite(macs) and macs > 0, "invalid measured costs")
    cost = {"latency_ms_raw": latency, "latency_anchor_ms": anchor,
            "latency_ms_scaled_full1p75": latency * 1.75 / anchor,
            "macs_forward_packet": macs, "params_total": int(sum(p.numel() for p in model.parameters())),
            "macs_forward_packet_note": "thop standard-layer MACs of whole forward_packet; nonstandard geometry/scatter excluded",
            "cost_provenance": {"method": "tools.x1001.evalx latency_anchor/latency_model/macs_of",
                                "source_seed": first, "checkpoint": provenance[first]["checkpoint"],
                                "config": provenance[first]["config"], "device": torch.cuda.get_device_name(device),
                                "latency_sequence": "lyq_local", "step_ms": 50, "max_packets": 600,
                                "passes": 3, "anchor_full_ms": 1.75}}
    del model
    torch.cuda.empty_cache()
    return cost


def make_row(arm, mode, budget, variant, aggregated, cost):
    per_seed, mean, provenance, _, _ = aggregated
    return {"arm": arm, "split": "val_core", "ckpt_rule": "last", "eval_variant": variant,
            "mode": mode, "budget": budget, "n_seeds": len(per_seed),
            "two_seed_mean": mean, "per_seed": per_seed, **cost,
            "provenance": provenance,
            "protocol": {"manifest": "splits_semkine.json", "train_subjects": list(SUBJECTS),
                         "train_sequences": 72, "sequences": list(SEQUENCES), "step_ms": 50,
                         "checkpoint_rule": "explicit last numbered checkpoint at the requested training budget",
                         "seed_aggregation": "equal mean of each seed's frame-weighted metric",
                         "single_seed_screen_only": len(per_seed) == 1},
            "note": "two_seed_mean retains the legacy key for the N-seed mean; diagnostics are in the companion extended JSON"}


def table_row(row, label):
    mean = row["two_seed_mean"]
    ra = f"{mean['overall']['mpjpe_ra_mm']:.2f}"
    seeds = row["per_seed"]
    if len(seeds) > 1:
        ra += "（" + " / ".join(f"{seeds[s]['overall']['mpjpe_ra_mm']:.2f}" for s in seeds) + "）"
    else:
        label += "（单种子 screening）"
    return (f"| {label} | {mean['local']['mpjpe_ra_mm']:.2f} | {mean['global']['mpjpe_ra_mm']:.2f} "
            f"| {mean['local']['mpvpe_ra_mm']:.2f} | {mean['global']['mpvpe_ra_mm']:.2f} | {ra} "
            f"| {row['latency_ms_scaled_full1p75']:.2f} ms | {row['macs_forward_packet'] / 1e9:.3f} G "
            f"| {row['params_total'] / 1e6:.2f} M |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", required=True, help="explicit output basename, e.g. u1a_shared_2k")
    parser.add_argument("--mode", required=True, choices=("shared", "untied", "baseline"))
    parser.add_argument("--budget", required=True, type=int, choices=(2000, 6000))
    parser.add_argument("--runs", nargs="+", required=True, help="explicit run directories; no glob or selection fallback")
    parser.add_argument("--variant", default="tf_pert", help="suffix after evalx_val_core_last_; use '' for plain eval")
    costs = parser.add_mutually_exclusive_group(required=True)
    costs.add_argument("--cost-row", help="existing last main row for this exact arm; reuse cost columns only")
    costs.add_argument("--measure-cost", action="store_true", help="explicitly run existing CUDA cost protocol")
    parser.add_argument("--baseline-runs", nargs="+", help="matched-seed frozen S38 6000-last reference directories")
    parser.add_argument("--baseline-cost-row", help="existing S38 last row; use its cost columns only")
    parser.add_argument("--baseline-arm", default="s38_spmeas", help="arm identity in the reference cost row")
    parser.add_argument("--baseline-variant", default="tf_pert")
    parser.add_argument("--label", help="table label; default derives mode and budget")
    parser.add_argument("--dry-run", action="store_true", help="validate/hash/print only; no writing or GPU profiling")
    args = parser.parse_args(argv)
    require(re.fullmatch(r"[A-Za-z0-9_-]+", args.arm) is not None, "--arm must be a simple artifact basename")
    require(re.fullmatch(r"[A-Za-z0-9_-]*", args.variant) is not None, "invalid --variant")
    require(not (args.dry_run and args.measure_cost), "--dry-run requires --cost-row and cannot measure cost")
    require(bool(args.baseline_runs) == bool(args.baseline_cost_row),
            "--baseline-runs and --baseline-cost-row must be supplied together")
    require(args.mode == "baseline" or args.baseline_runs is not None,
            "a U1a result requires explicit matched S38 baseline runs and cost row")
    aggregate = aggregate_runs(args.runs, mode=args.mode, budget=args.budget, variant=args.variant)
    cost = imported_cost(args.cost_row, arm=args.arm, per_seed=aggregate[0]) if args.cost_row else measure_cost(aggregate[4], aggregate[0], aggregate[2])
    row = make_row(args.arm, args.mode, args.budget, args.variant, aggregate, cost)
    rows = []
    extended = {"arm": args.arm, "checkpoint_rule": "last", "per_seed": aggregate[3],
                "provenance": aggregate[2], "cost_provenance": cost["cost_provenance"]}
    if args.baseline_runs:
        reference = aggregate_runs(args.baseline_runs, mode="baseline", budget=6000, variant=args.baseline_variant)
        require(set(reference[0]) == set(aggregate[0]), "U1a and frozen S38 baseline must have exactly the same seeds")
        for seed in aggregate[0]:
            require(reference[0][seed]["frames_per_seq"] == aggregate[0][seed]["frames_per_seq"], "reference evaluates different frames")
            require(reference[2][seed]["manifest"]["sha256"] == aggregate[2][seed]["manifest"]["sha256"], "reference uses a different manifest")
            if args.mode != "baseline":
                require(path_at_repo(aggregate[2][seed]["frozen_s38_source"]["run"]) == path_at_repo(reference[0][seed]["run"]),
                        f"seed {seed}: S38 baseline run differs from the frozen encoder source")
        require(aggregate[4].get("TRACK") == reference[4].get("TRACK"), "reference uses a different tracking / initialization-noise protocol")
        for key in ("ROOT_REF", "ROOT_FILTER_GAIN", "FINGER_FILTER_GAIN"):
            require(aggregate[4]["MODEL"].get(key) == reference[4]["MODEL"].get(key),
                    f"U1a and frozen S38 reference differ on MODEL.{key}")
        reference_cost = imported_cost(args.baseline_cost_row, arm=args.baseline_arm, per_seed=reference[0])
        reference_row = make_row(args.baseline_arm, "baseline", 6000, args.baseline_variant, reference, reference_cost)
        rows.append(table_row(reference_row, "S38 spmeas（固定 last 参照）"))
        row["baseline_reference"] = {"arm": args.baseline_arm, "checkpoint_rule": "last", "budget": 6000,
                                     "two_seed_mean": reference[1], "per_seed": reference[0], "provenance": reference[2]}
        extended["baseline"] = {"row": reference_row, "per_seed": reference[3]}
    label = args.label or ("S38 spmeas（固定 last 参照）" if args.mode == "baseline" else f"U1a {args.mode} {args.budget // 1000}k（冻结 S38 编码器）")
    rows.append(table_row(row, label))
    table = "\n".join([HEADER, RULE, BASELINE, *rows])
    if not args.dry_run:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{args.arm}_main_row.json").write_text(json.dumps(row, indent=2, allow_nan=False), encoding="utf-8")
        (OUT / f"{args.arm}_extended.json").write_text(json.dumps(extended, indent=2, allow_nan=False), encoding="utf-8")
        report_path = REPO / "outputs" / "u1a" / f"{args.arm}_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(table + "\n", encoding="utf-8")
    print(table)
    # Main-row evidence is inspectable in dry-run without changing historical artifacts.
    if args.dry_run:
        print(f"\ndry-run: {args.arm}; {len(row['per_seed'])} seeds; explicit last step {args.budget}; no files written or GPU work")
        for seed, source in row["provenance"].items():
            print(f"  s{seed}: eval SHA256 {source['evaluation']['sha256']}; checkpoint SHA256 {source['checkpoint']['sha256']}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as error:
        raise SystemExit(f"U1a report refused: {error}")
