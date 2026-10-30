#!/usr/bin/env python3
"""Strict XCL E1--E5 evaluation, with the original packet, FK and cost functions.

One process runs one registered arm/readout. ``--mode row`` measures latency and
must be scheduled serially. E1 repeats use independent processes and fresh loads;
``--compare`` compares every array exactly. E5 uses --mode e1 --step 5000/5500.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "model")]
COUNTS = {"zgz_global": 1386, "zgz_local": 1204}
MODES = ("row", "e1", "e2", "e3", "gt_pose", "gt_transl")
POLICY_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "3407",
              "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1",
              "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONDONTWRITEBYTECODE": "1"}


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def sha(value):
    h = hashlib.sha256()
    with Path(value).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_new(filename, value):
    filename = Path(filename)
    with filename.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def configure_environment():
    previous = os.environ.get("PYTHONHASHSEED")
    os.environ.update(POLICY_ENV)
    if previous != "3407":
        os.execve(sys.executable, [sys.executable, *sys.argv], dict(os.environ))


def set_policy():
    # Same flags as tools/xag_paired_20261018/evaluate.py, without legacy imports.
    import torch
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)


def resolve_run(protocol, identifier):
    rows = [r for r in protocol["runs"] if identifier in (r["name"], r.get("cell"), r.get("prefix"))]
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one registered run for {identifier!r}")
    run = rows[0]
    if run.get("seed", 3407) != 3407 or run.get("step", run.get("fresh_steps", 6000)) != 6000:
        raise ValueError("XCL requires registered seed3407 fresh6000 identities")
    return run


def resolve_checkpoint(run, step):
    if step not in (5000, 5500, 6000):
        raise ValueError("Only preregistered steps 5000/5500/6000 are evaluated")
    if step == 6000 and run.get("checkpoint"):
        checkpoint = path(run["checkpoint"])
    else:
        candidates = [p for p in path(run["output"]).glob("*step=*.ckpt")
                      if (m := re.search(r"step=(\d+)", p.name)) and int(m[1]) == step]
        if len(candidates) != 1:
            raise RuntimeError(f"Ambiguous or missing step{step} checkpoint: {candidates}")
        checkpoint = candidates[0]
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = sha(checkpoint)
    expected = run.get("checkpoint_sha256") if step == 6000 else run.get("checkpoint_sha256_by_step", {}).get(str(step))
    if run.get("reuse") is False:
        # Newly trained weights acquire identity from the complete trajectory
        # receipt; the pre-training protocol itself remains immutable.
        receipt = json.loads((ROOT / "outputs/xcl/20261019/receipts" / ("formal_" + run["cell"] + ".json")).read_text())
        if receipt.get("status") != "PASS":
            raise RuntimeError("New model has no verified training trajectory")
        expected = receipt["checkpoint_sha256"] if step == 6000 else receipt["checkpoint_sha256_by_step"][str(step)]
    if expected and digest != expected:
        raise RuntimeError("Registered checkpoint SHA256 changed")
    config = path(run["config"])
    if run.get("config_sha256") and sha(config) != run["config_sha256"]:
        raise RuntimeError("Registered configuration SHA256 changed")
    return checkpoint, config


def endpoint_grid(aux):
    import numpy as np
    rows = []
    for rid, (a, b) in enumerate(np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2)):
        previous = int(a)
        for end in np.arange(int(a) + 49, int(b), 50, dtype=np.int64):
            rows.append((rid, int(end), previous, previous == int(a)))
            previous = int(end)
    return rows


def select_previous(mode, raw, truth, common=None):
    """Pure state-contract helper, independently tested without GPU work."""
    import numpy as np
    result = np.asarray(raw).copy()
    if mode == "e2":
        if common is None:
            raise ValueError("E2 requires canonical C1 incoming")
        result[:] = common
    elif mode == "e3":
        result[:] = truth
    elif mode == "gt_pose":
        result[3:] = truth[3:]
    elif mode == "gt_transl":
        result[:3] = truth[:3]
    elif mode not in ("e1", "row"):
        raise ValueError(mode)
    if result.shape != (51,) or result.dtype != np.float32 or not np.isfinite(result).all():
        raise ValueError("Incoming state must be finite float32[51]")
    return result


def score(mano, prediction, target, beta, device):
    import numpy as np
    from semkine import eval_track as ET
    from tools.xdl.geometry import rotations, geodesic_deg
    pj, pv = ET.mano_fk(mano, prediction, beta, device)
    gj, gv = ET.mano_fk(mano, target, beta, device)
    mean = mano.hands_mean.detach().cpu().numpy()
    pr, pf = rotations(prediction, mean)
    gr, gf = rotations(target, mean)
    return {"mpjpe_ra_mm": np.linalg.norm(ET.root_align(pj) - ET.root_align(gj), axis=-1).mean(-1) * 1000,
            "mpvpe_ra_mm": np.linalg.norm(ET.root_align(pv) - ET.root_align(gv), axis=-1).mean(-1) * 1000,
            "mpjpe_abs_mm": np.linalg.norm(pj - gj, axis=-1).mean(-1) * 1000,
            "mpvpe_abs_mm": np.linalg.norm(pv - gv, axis=-1).mean(-1) * 1000,
            "wrist_mm": np.linalg.norm(pj[:, 0] - gj[:, 0], axis=-1) * 1000,
            "root_deg": geodesic_deg(pr, gr), "finger_deg": geodesic_deg(pf, gf).mean(-1)}


def array_digest(array):
    h = hashlib.sha256()
    h.update(str(array.dtype).encode()); h.update(str(array.shape).encode())
    h.update(array.tobytes(order="C"))
    return h.hexdigest()


def compare_traces(first, second):
    import numpy as np
    with np.load(first, allow_pickle=False) as a, np.load(second, allow_pickle=False) as b:
        if set(a.files) != set(b.files):
            raise RuntimeError("Trace field sets differ")
        for name in a.files:
            if a[name].dtype != b[name].dtype or not np.array_equal(a[name], b[name]):
                raise RuntimeError(f"Fresh trace array differs: {name}")
    return {"status": "PASS", "first_sha256": sha(first), "second_sha256": sha(second),
            "comparison": "All field names, dtypes, shapes and values exactly equal"}


def trace(run, checkpoint, cfg, output, mode, canonical_trace=None, intervention="none"):
    import numpy as np
    import torch
    from mano_layer import ManoLayer
    from model import MNISTModel
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split, splits_manifest
    from tools.xcl.interventions import apply, adapt_config
    device = torch.device("cuda")
    model = MNISTModel.load_from_checkpoint(str(checkpoint), cfg=cfg, map_location=device, strict=True).to(device).eval()
    if intervention.startswith("route_"):
        cfg = adapt_config(cfg, intervention)
    mano = ManoLayer(cfg["MANO"]["NPZ"], add_mean=False).to(device).eval()
    sequences = sequences_for_split(Path(cfg["DATA"]["ROOT"]), "val_core", splits_manifest(cfg))
    if [s for s, _ in sequences] != list(COUNTS):
        raise RuntimeError("Expected complete original zgz_global/zgz_local")
    common = np.load(canonical_trace, allow_pickle=False) if mode == "e2" else None
    arrays, results, fingerprints = {}, [], []
    native = model.forward_packet
    rng = np.random.default_rng(0)
    try:
        with torch.no_grad(), apply(model, intervention):
            for sequence, directory in sequences:
                _, _, aux, gt = ET.load_sequence(Path(cfg["DATA"]["ROOT"]), directory, sequence)
                rows = endpoint_grid(aux)
                if len(rows) != COUNTS[sequence]:
                    raise RuntimeError("Original endpoint denominator changed")
                incoming, used, predictions, targets, priors = [], [], [], [], []
                if common is not None:
                    common_values = common[sequence + "/incoming"]
                    if common_values.shape != (COUNTS[sequence], 51):
                        raise RuntimeError("Incomplete canonical C1 trace")
                    if sequence + "/end_ms" in common and not np.array_equal(common[sequence + "/end_ms"], [r[1] for r in rows]):
                        raise RuntimeError("Canonical C1 endpoint identity differs")
                def capture(packet):
                    i = len(predictions)
                    if i >= len(rows):
                        raise RuntimeError("Too many evaluation packets")
                    rid, end, previous_ms, first = rows[i]
                    raw = packet.prev_state.detach().cpu().numpy()[0].copy()
                    truth = np.asarray(gt[previous_ms], np.float32).copy()
                    current = np.asarray(gt[end], np.float32).copy()
                    previous = select_previous(mode, raw, truth, common_values[i] if common is not None else None)
                    # Supply FΩ end GT without changing XYZ_ASSOC_PREV or routing.
                    packet = replace(packet, prev_state=torch.from_numpy(previous).view(1, -1).to(device),
                                     target=torch.from_numpy(current).view(1, -1).to(device))
                    pred = native(packet)
                    value = pred.detach().cpu().numpy()[0].copy()
                    if value.shape != (51,) or value.dtype != np.float32 or not np.isfinite(value).all():
                        raise FloatingPointError(f"Nonfinite/invalid endpoint retained as failure: {sequence}:{end}")
                    if not first and not np.array_equal(raw, predictions[-1]):
                        raise RuntimeError("Native closed loop previous output contract changed")
                    incoming.append(raw); used.append(previous); predictions.append(value); targets.append(current); priors.append(truth)
                    fingerprints.append({"seq": sequence, "end_ms": end, "run_id": rid,
                                         "events_sha256": array_digest(packet.events.detach().cpu().contiguous().numpy())})
                    if len(fingerprints) % 250 == 0:
                        temporary = output / "progress.tmp"
                        temporary.write_text(json.dumps({"status": "TRACING", "n_endpoints": len(fingerprints), "expected": 2590}) + "\n")
                        temporary.replace(output / "progress.json")
                    return pred
                model.forward_packet = capture
                result = ET.track_sequence(model, mano, cfg, Path(cfg["DATA"]["ROOT"]), directory,
                                           sequence, 50, device, rng, 1.0, 1.0, None)
                if result is None or result["n_frames"] != len(predictions) or len(predictions) != COUNTS[sequence]:
                    raise RuntimeError("Incomplete evaluation; no endpoints may be dropped")
                beta = torch.tensor(np.array(aux["betas"]), device=device, dtype=torch.float32).view(1, -1)
                predictions, targets = np.stack(predictions), np.stack(targets)
                metrics = score(mano, predictions, targets, beta, device)
                for key in ET_METRICS:
                    if not np.isclose(metrics[key].mean(), result[key], rtol=0, atol=1e-9):
                        raise RuntimeError(f"Canonical FK/scoring mismatch: {sequence}/{key}")
                for key, value in {"incoming": np.stack(incoming), "incoming_used": np.stack(used),
                                   "prediction": predictions, "target": targets, "gt_previous": np.stack(priors),
                                   "end_ms": np.array([r[1] for r in rows], np.int64),
                                   "run_id": np.array([r[0] for r in rows], np.int64), **metrics}.items():
                    if not np.isfinite(value).all():
                        raise FloatingPointError("Nonfinite metrics cannot be omitted")
                    arrays[sequence + "/" + key] = value
                results.append({"seq": sequence, "n_frames": len(predictions),
                                **{key: float(value.mean()) for key, value in metrics.items()}})
    finally:
        model.forward_packet = native
        if common is not None:
            common.close()
    if len(fingerprints) != 2590:
        raise RuntimeError("Incomplete original zgz denominator")
    np.savez_compressed(output / "states.npz", **arrays)
    save_new(output / "packets.json", fingerprints)
    blocks = {"global": results[0], "local": results[1],
              "overall": {key: sum(r[key] * r["n_frames"] for r in results) / 2590
                          for key in results[0] if key not in ("seq", "n_frames")}}
    return {"status": "COMPLETE", "n_endpoints": 2590, "n_failed": 0, "per_sequence": results,
            "metrics": blocks, "states_sha256": sha(output / "states.npz"),
            "packets_sha256": sha(output / "packets.json"),
            "canonical_trace_sha256": sha(canonical_trace) if mode == "e2" else None}


ET_METRICS = ("mpjpe_ra_mm", "mpvpe_ra_mm", "mpjpe_abs_mm", "mpvpe_abs_mm")


@contextmanager
def oracle_cost_targets(original, cfg):
    """Supply true end targets for native latency packets, synthetic depth for MACs.

    The original event windows, timing loop and synthetic MAC fixture stay in use.
    Offline oracle target/cache preparation is outside latency, as in old A1/A2.
    """
    import torch
    from model import MNISTModel
    ET = original.ET
    from semkine.dataset import sequences_for_split, splits_manifest
    sequence, directory = next((s, d) for split in ("val_core", "train")
                               for s, d in sequences_for_split(Path(cfg["DATA"]["ROOT"]), split, splits_manifest(cfg)) if s == "lyq_local")
    _, _, _, gt = ET.load_sequence(Path(cfg["DATA"]["ROOT"]), directory, sequence)
    window, packet, forward = ET._window_events, ET.make_eval_packet, MNISTModel.forward_packet
    last_end = [None]
    def capture_window(*args, **kwargs):
        last_end[0] = int(args[3] if len(args) > 3 else kwargs["end"])
        return window(*args, **kwargs)
    def target_packet(*args, **kwargs):
        batch = packet(*args, **kwargs)
        if last_end[0] is not None:
            batch.target = torch.from_numpy(gt[last_end[0]].copy()).view(1, -1).to(batch.prev_state.device)
        return batch
    def synthetic_depth(self, batch, *args, **kwargs):
        # Real evaluation data MUST load their sidecar; only CPU synthetic MAC fixture gets this.
        if batch.events.device.type == "cpu" and getattr(self, "xag_assoc_mode", "base") == "full_oracle" and getattr(batch, "xag_depth", None) is None:
            batch.xag_depth = torch.full((batch.events.shape[0],), float(batch.prev_state[0, 2]), dtype=torch.float32)
        return forward(self, batch, *args, **kwargs)
    ET._window_events, ET.make_eval_packet, MNISTModel.forward_packet = capture_window, target_packet, synthetic_depth
    try:
        yield
    finally:
        ET._window_events, ET.make_eval_packet, MNISTModel.forward_packet = window, packet, forward


def make_main_row(run, checkpoint, cfg, output, trace_report, step):
    """Run the original row generator, adapting only registered eval identity/input."""
    from tools import make_s36_row as original
    from semkine.dataset import sequences_for_split, splits_manifest
    row_path = output / f"{run.get('prefix', run['name'].removesuffix('_s3407'))}_main_row_step{step}.json"
    per_seq = {r["seq"]: {k: r[k] for k in ET_METRICS} for r in trace_report["per_sequence"]}
    aggregate = {"step": step, "ckpt": str(checkpoint), "sequences": list(COUNTS), "n_frames": 2590,
                 "per_seq": per_seq, **{block: {k: trace_report["metrics"][block][k] for k in ET_METRICS}
                                       for block in ("overall", "local", "global")}}
    class Redirect:
        def __truediv__(self, name):
            return row_path if str(name).endswith(f"_main_row_step{step}.json") else ROOT / name
    saved_root, saved_eval, saved_argv = original.REPO, original.eval_seed, sys.argv
    def registered_eval(name, seed, device, split, subject, policy):
        if seed != 3407 or policy != step or split != "val_core" or subject is not None:
            raise ValueError("Original row requested an unregistered scoring identity")
        return aggregate, cfg, sequences_for_split(Path(cfg["DATA"]["ROOT"]), "val_core", splits_manifest(cfg)), Path(cfg["DATA"]["ROOT"])
    try:
        original.REPO, original.eval_seed = Redirect(), registered_eval
        sys.argv = [str(ROOT / "tools/make_s36_row.py"), "--run", run.get("prefix", run["name"].removesuffix("_s3407")),
                    "--step", str(step), "--seeds", "3407", "--split", "val_core"]
        with oracle_cost_targets(original, cfg):
            original.main()
    finally:
        original.REPO, original.eval_seed, sys.argv = saved_root, saved_eval, saved_argv
    row = json.loads(row_path.read_text())
    import math
    for key in ("latency_ms_scaled_full1p75", "macs_forward_packet", "params_total"):
        if not math.isfinite(row[key]) or row[key] <= 0:
            raise ValueError(f"Invalid original full-model cost: {key}")
    if row["latency_packets"] != 600:
        raise ValueError("Latency must use all 600 registered packets")
    return {"path": str(row_path), "sha256": sha(row_path), "original_generator_sha256": sha(ROOT / "tools/make_s36_row.py")}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--protocol"); p.add_argument("--run"); p.add_argument("--mode", choices=MODES, default="e1")
    p.add_argument("--step", type=int, choices=(5000, 5500, 6000), default=6000)
    p.add_argument("--output-dir", type=Path); p.add_argument("--canonical-trace")
    p.add_argument("--intervention", choices=("none", "geometry_zero", "h3_zero", "residual_zero", "route_a2", "route_a3"), default="none")
    p.add_argument("--compare", nargs=2, metavar=("FIRST_NPZ", "SECOND_NPZ"))
    args = p.parse_args(argv)
    if args.compare:
        print(json.dumps(compare_traces(*map(path, args.compare)), indent=2)); return
    if not all((args.protocol, args.run, args.output_dir)):
        p.error("--protocol, --run and --output-dir are required")
    if args.step != 6000 and (args.mode != "e1" or args.intervention != "none"):
        p.error("E5 late checkpoints are unmodified E1 only")
    if args.mode == "row" and args.intervention != "none":
        p.error("Main rows cannot be intervention results")
    if args.intervention.startswith("route_") and args.mode not in ("e2", "e3"):
        p.error("OOD routing crosses are registered only for E2/E3")
    protocol_path = path(args.protocol)
    protocol_sha = sha(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    run = resolve_run(protocol, args.run)
    checkpoint, config = resolve_checkpoint(run, args.step)
    canonical = args.canonical_trace or protocol.get("canonical_trace")
    if isinstance(canonical, dict):
        expected = canonical.get("sha256"); canonical = canonical["path"]
        if expected and sha(path(canonical)) != expected:
            raise ValueError("Canonical C1 trace SHA256 changed")
    if args.mode == "e2" and not canonical:
        p.error("E2 requires --canonical-trace or protocol canonical_trace")
    output = path(args.output_dir)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Preserve existing evaluation; choose a fresh output directory")
    if not output.resolve().is_relative_to(ROOT / "outputs/xcl"):
        raise ValueError("Write evaluation artifacts only under outputs/xcl")
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if physical not in ("4", "5", "6", "7"):
        raise ValueError("XCL evaluation uses exactly one physical GPU4/5/6/7 per process")
    configure_environment(); set_policy()
    from config import load_config
    from tools.xcl.interventions import adapt_config
    cfg = load_config(config)
    # Load weights with their native architecture, then override routing at runtime.
    if args.intervention.startswith("route_"):
        cfg = adapt_config(cfg, args.intervention)
        cfg["MODEL"]["XAG_ASSOC"] = load_config(config).get("MODEL", {}).get("XAG_ASSOC", "base")
    output.mkdir(parents=True, exist_ok=False)
    identity = {"run": run["name"], "cell": run.get("cell"), "seed": 3407, "step": args.step,
                "mode": args.mode, "intervention": args.intervention, "expected_denominator": 2590,
                "protocol_sha256": protocol_sha, "checkpoint_sha256": sha(checkpoint), "config_sha256": sha(config),
                "script_sha256": sha(__file__), "environment": {k: os.environ[k] for k in POLICY_ENV},
                "physical_gpus": os.environ.get("CUDA_VISIBLE_DEVICES"), "runtime_policy": "XCL_STRICT_C1_20261019"}
    save_new(output / "launch.json", identity)
    try:
        report = trace(run, checkpoint, cfg, output, args.mode, path(canonical) if canonical else None, args.intervention)
        if args.mode == "row":
            report["main_row"] = make_main_row(run, checkpoint, cfg, output, report, args.step)
        if sha(protocol_path) != protocol_sha or sha(checkpoint) != identity["checkpoint_sha256"] or sha(config) != identity["config_sha256"]:
            raise RuntimeError("Registered input identity changed during evaluation")
        save_new(output / "report.json", {**identity, **report, "no_training_or_selection": True})
    except BaseException as exc:
        save_new(output / "failure.json", {**identity, "status": "FAILED_PRESERVED", "error": f"{type(exc).__name__}: {exc}",
                                          "no_failed_endpoints_excluded_from_denominator": True})
        raise
    print("COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
