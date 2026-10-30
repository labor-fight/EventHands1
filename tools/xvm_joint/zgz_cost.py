"""Frozen XJ zgz-report cost measurement on the original lyq_local fixture.

This command never loads zgz. It loads lyq_local only for the declared
main-table latency fixture. The original GT run-start pose initializes this
cost-only replay without noise; packet beta is zero and later previous states
are the model's own outputs. No latency target or accuracy metric is scored.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from .train import configure_device, dump, sha
from .zgz_resources import PacketCase, measure_resources


def prepare_latency_cases(protocol, mode, device):
    """Stage the original 600 lyq_local windows outside the timed region.

    The GT metadata access is resource initialization only. It is separate
    from the inference firewall, which still surrounds actual packet forwards.
    There are no zgz sample/label reads in this function.
    """
    from semkine import eval_track as ET
    root = Path(protocol["data_root"])
    resource = protocol["resources"]
    if resource["timing_recording"] != "lyq_local":
        raise RuntimeError("Only the frozen original training timing fixture is allowed")
    split = json.loads(Path(protocol["split"]["path"]).read_text())
    if "lyq_local" not in split["train"]["trials"]:
        raise RuntimeError("The original timing fixture is not in the training split")
    events, offsets, aux, pos51 = ET.load_sequence(root, "train", "lyq_local")
    tsub_path = root/"train/lyq_local_tsub.npy"
    tsub = np.load(tsub_path, mmap_mode="r") if tsub_path.exists() else None
    # In particular, aux['betas'] is not used to construct model inputs.
    camera = np.asarray(aux["camera_K"], np.float32).copy()
    runs = np.asarray(aux["valid_runs_ms"], np.int64).reshape(-1, 2).copy()
    K = torch.from_numpy(camera).to(device).reshape(1, 3, 3)
    beta0 = torch.zeros((1, 10), device=device)
    maximum = int(resource["max_packets"])
    cases = []
    run_summary = []
    settings = protocol["modes"][mode]
    win = int(settings["window_ms"])
    for run_id, (a, b) in enumerate(runs):
        initial = torch.from_numpy(pos51[int(a)].copy()).float().reshape(1, 51).to(device)
        used = 0
        for end in np.arange(int(a)+49, int(b), 50, dtype=np.int64):
            current_window = min(win, int(end)-int(a)+1) if settings.get("clip_run_start") else win
            event = ET._window_events(events, offsets, tsub, int(end), int(current_window))
            packet = ET.make_eval_packet(event, initial, beta0, K, int(current_window), device)
            cases.append(PacketCase(packet, reset=used == 0, run_start_ms=int(a), sequence_id=run_id))
            used += 1
            if len(cases) >= maximum:
                break
        if used:
            run_summary.append({"run_id": int(run_id), "start_ms": int(a), "stop_ms": int(b), "packets": used})
        if len(cases) >= maximum:
            break
    if not cases or len(cases) != maximum:
        raise RuntimeError("The frozen timing fixture lacks the required packet coverage")
    observed = {"recording": "lyq_local", "legacy_directory": "train", "packets": len(cases),
                "runs": run_summary, "initialization": "original GT run-start pose, no noise, timing only",
                "beta_input": "zeros10", "window_mode": mode,
                "source_data_scope": "lyq_local timing-only inputs and GT initial pose; no zgz access",
                "packet_staging_excluded_from_latency": True}
    return cases, observed


def run(protocol_path, sourcefreeze_path, *, mode, device, output):
    from .zgz_eval import verify_identity, make_model
    protocol, _, config = verify_identity(protocol_path, sourcefreeze_path)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Cost evidence already exists; no silent overwrite")
    if mode not in protocol["modes"]:
        raise ValueError("Unknown frozen evaluation mode")
    random.seed(int(protocol["seed"]))
    np.random.seed(int(protocol["seed"]))
    torch.manual_seed(int(protocol["seed"]))
    torch.set_num_threads(int(protocol["cpu_threads"]))
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    os.environ["XJ_GPU_MEMORY_LIMIT_GIB"] = str(protocol["gpu_memory_cap_gib_per_process"])
    device, device_receipt = configure_device(device, config, engineering=True)
    if device.type != "cuda" or device_receipt["physical_gpu"] not in protocol["devices"]:
        raise RuntimeError("The frozen actual GPU2/3 timing device is required")
    started = time.perf_counter()
    try:
        # This is the unchanged official full-model calibration, before any XJ
        # instance is allocated. The consuming entry point is not monkeypatched.
        from tools import make_s36_row as main_table
        anchor = float(main_table.latency_anchor(device))
        cases, fixture = prepare_latency_cases(protocol, mode, device)
        arms = {}
        for arm in protocol["arms"]:
            def factory(target_device, current_arm=arm):
                model, _, _ = make_model(protocol, current_arm, mode, target_device)
                return model
            arms[arm] = measure_resources(factory, cases, device=device, anchor_raw_ms=anchor,
                window_ms=int(protocol["modes"][mode]["window_ms"]),
                max_packets=int(protocol["resources"]["max_packets"]),
                passes=int(protocol["resources"]["passes"]),
                warmup_packets=int(protocol["resources"]["warmup_packets"]))
            dump(output.with_suffix(output.suffix+".progress.json"),
                 {"status": "RUNNING", "mode": mode, "completed_arms": list(arms),
                  "elapsed_s": time.perf_counter()-started})
        # Recheck executable/config/weight identities after actual measurement.
        verify_identity(protocol_path, sourcefreeze_path)
        result = {"schema": "xj_zgz_cost_v1", "status": "COMPLETE", "mode": mode,
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "protocol_sha256": sha(protocol_path), "sourcefreeze_sha256": sha(sourcefreeze_path),
                  "checkpoint_sha256": protocol["checkpoint"]["sha256"],
                  "device": device_receipt, "fixture": fixture, "arms": arms,
                  "elapsed_s": time.perf_counter()-started,
                  "scope": "serial complete packet latency on lyq_local, original full1.75ms anchor; CPU synthetic all-standard-layer MACs; no zgz reads or accuracy scoring"}
        dump(output, result)
        return result
    except BaseException as exc:
        dump(output.with_suffix(output.suffix+".failure.json"),
             {"status": "FAILED_PRESERVED", "mode": mode, "error": repr(exc),
              "traceback": traceback.format_exc(), "elapsed_s": time.perf_counter()-started,
              "protocol_sha256": sha(protocol_path), "sourcefreeze_sha256": sha(sourcefreeze_path)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--source-freeze", required=True)
    parser.add_argument("--mode", choices=("w50", "s39"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run(args.protocol, args.source_freeze, mode=args.mode, device=args.device, output=args.out)
    print(json.dumps({"status": result["status"], "mode": result["mode"], "arms": result["arms"]},
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
