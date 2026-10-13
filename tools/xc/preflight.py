#!/usr/bin/env python3
"""Actual-batch CUDA and trainer-entry checks before freezing the isolated XC study.

This writes a draft while analysis helpers are still being built. ``--finalize``
rehashes the complete current source_files() set; it never launches formal runs.
Only physical GPU4 is accepted for the memory/trainer smoke. Failures are retained
and the script never silently lowers the batch size or selects another checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "model"))
OUT = REPO / "outputs/xc"
PYTHON = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
CRITICAL = ("semkine/xyz_time_study.py", "tools/xc/train.py", "tools/xc/evaluation.py",
            "semkine/dataset.py", "semkine/events.py", "semkine/encoder.py",
            "semkine/lie.py", "semkine/xyz_mesh.py", "model/model.py",
            "model/pose_repr.py", "model/mano_layer.py", "semkine/train.py",
            "semkine/eval_track.py", "assets/mano_right.npz",
            "data/hand_data51/splits_semkine.json")


def stamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def say(value):
    print(json.dumps({"time": stamp(), **value}, ensure_ascii=False, allow_nan=False), flush=True)


def source_hashes():
    from tools.xc.run_study import source_files
    files = sorted(set(source_files() + [REPO / name for name in CRITICAL]))
    return ({str(p.relative_to(REPO)): sha(p) for p in files if p.is_file()},
            [str(p.relative_to(REPO)) for p in files if not p.is_file()])


def tensor_hash(values):
    h = hashlib.sha256()
    for name, value in values:
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(str(value.dtype).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def pair_hash(pair):
    names = ("sequence_id", "t_start_us", "t_end_us", "target", "prev_state", "events",
             "ptr", "delta_t_s", "betas", "camera_K", "is_sequence_start", "is_sequence_end")
    return tensor_hash([(f"{index}.{name}", getattr(packet, name))
                        for index, packet in enumerate(pair) for name in names])


def finite_tensors(named):
    checked, bad = 0, []
    for name, value in named:
        if value is not None and value.is_floating_point():
            checked += 1
            if not bool(value.detach().isfinite().all()):
                bad.append(name)
    return {"checked": checked, "nonfinite": bad}


def config(arm, seed=3407):
    import yaml
    return yaml.safe_load((REPO / f"configs/xc/xc_{arm}_s{seed}.yaml").read_text())


def immutable_default_sources():
    baseline = json.loads((REPO / "outputs/xb/frozen_manifest.json").read_text())
    files = ("model/model.py", "semkine/xyz_mesh.py",
             "semkine/train.py", "semkine/dataset.py", "semkine/eval_track.py")
    checks = {name: {"current": sha(REPO / name),
                     "xb_frozen": baseline["files_sha256"][name]}
              for name in files}
    if any(row["current"] != row["xb_frozen"] for row in checks.values()):
        raise AssertionError("an unchanged core source differs from the prior XB freeze")
    review = {"model/mano_layer.py": "59530955b530553b55933cf7dcd208018610e1751800088418b6063a18af08b5",
              "model/pose_repr.py": "88f597863b97b734cce6eed64501f05721aee33c7da94739883a1a7bf703b50a"}
    if any(sha(REPO / name) != digest for name, digest in review.items()):
        raise AssertionError("MANO/pose source differs from the independent read-only review snapshot")
    return {"prior_frozen_manifest": "outputs/xb/frozen_manifest.json", "files": checks,
            "independent_review_snapshot_sha256": review,
            "review_snapshot_note": "captured by the independent reviewer before writing/running preflight; MANO was not in XB frozen_manifest"}


def cpu_tests(smoke):
    tests = sorted(set([REPO / "tests/test_xyz_time_study.py"]
                       + list((REPO / "tests").glob("test_xc*.py"))))
    cmd = [PYTHON, "-m", "pytest", "-q"] + [str(p.relative_to(REPO)) for p in tests]
    test_stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    log = smoke / f"cpu_tests_{test_stamp}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with log.open("w") as handle:
        result = subprocess.run(cmd, cwd=REPO, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise AssertionError(f"CPU tests failed; see {log}")
    return {"command": cmd, "log": str(log.relative_to(REPO)), "log_sha256": sha(log),
            "test_source_sha256": {str(p.relative_to(REPO)): sha(p) for p in tests},
            "summary": log.read_text().splitlines()[-1]}


def initializations():
    import torch
    from semkine.xyz_time_study import TimeXYZModel
    rows = {}
    for seed in (3407, 3408, 3409):
        hashes, counts = {}, {}
        reference = None
        for arm in ("g0t0", "g1t0", "g0t1", "g1t1"):
            torch.manual_seed(seed)
            model = TimeXYZModel(config(arm, seed)).cpu()
            state = model.state_dict()
            if reference is None:
                reference = {name: value.clone() for name, value in state.items()}
            elif state.keys() != reference.keys() or not all(torch.equal(value, reference[name])
                                                             for name, value in state.items()):
                raise AssertionError(f"same-seed initial state mismatch in {arm}/{seed}")
            counts[arm] = sum(p.numel() for p in model.parameters())
            hashes[arm] = tensor_hash(sorted(state.items()))
            if bool(model.xyz.fc1.weight[:, -51:].any()):
                raise AssertionError("added history columns must initialize to exact zero")
            del model
        if len(set(counts.values())) != 1:
            raise AssertionError("four arms differ in parameter count")
        rows[str(seed)] = {"parameter_counts": counts, "state_dict_sha256": hashes,
                           "all_tensors_bitwise_equal": True}
    gc.collect()
    return rows


def actual_batch():
    import numpy as np
    import torch
    from semkine.dataset import build_dataset
    from semkine.events import collate_packets
    cfg = config("g0t0")
    components = np.load(REPO / cfg["MANO"]["NPZ"])["hands_components"]
    dataset = build_dataset(cfg, "train", components, train=True)
    if len(dataset.sequences) != 72:
        raise AssertionError("preflight must use all 72 training sequences")
    def collate(items):
        return tuple(collate_packets(list(col)) for col in zip(*items))
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True, num_workers=12,
                                        persistent_workers=False, prefetch_factor=1, pin_memory=True,
                                        drop_last=True, collate_fn=collate,
                                        generator=torch.Generator().manual_seed(3407))
    iterator = iter(loader)
    pair = next(iterator)
    del iterator, loader
    for packet in pair:
        packet.validate()
    result = {"source": "immutable first actual pair batch reused across the four arms",
              "seed": 3407, "batch_size": 512, "train_sequences": len(dataset.sequences),
              "train_subjects": sorted({name.split("_")[0] for name, _ in dataset.sequences}),
              "leading_events": pair[0].events.shape[0], "main_events": pair[1].events.shape[0],
              "paired_sha256": pair_hash(pair),
              "window_ms_range": [float(pair[1].delta_t_s.min() * 1000),
                                   float(pair[1].delta_t_s.max() * 1000)],
              "past_event_intervals_end_at_main_start": bool(torch.equal(pair[0].t_end_us,
                                                                           pair[1].t_start_us))}
    if not result["past_event_intervals_end_at_main_start"]:
        raise AssertionError("leading/main intervals are not contiguous")
    del dataset
    return pair, result


def cuda_arm(arm, pair):
    import torch
    from semkine.xyz_time_study import TimeXYZModel
    torch.manual_seed(3407)
    model = TimeXYZModel(config(arm)).cuda().train()
    model.xyz_keep_aux = True
    generator = torch.Generator(device="cuda").manual_seed(19073)
    with torch.no_grad():
        model.xyz.fc2.weight.copy_(torch.randn(model.xyz.fc2.weight.shape, generator=generator,
                                               device="cuda") * .002)
        model.xyz.fc2.bias.copy_(torch.randn(model.xyz.fc2.bias.shape, generator=generator,
                                             device="cuda") * .0002)
    batches = tuple(p.to("cuda") for p in pair)
    optimizer = torch.optim.Adam(model.parameters(), lr=.004)
    torch.cuda.reset_peak_memory_stats()
    began = time.monotonic()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        predicted, target, betas, _ = model._predict_batch(batches)
        loss, parts = model._compute_loss(predicted, target, betas)
    objective = loss.float().log10() if model.log10_loss else loss.float()
    numeric = finite_tensors([("prediction", predicted), ("loss", loss), ("objective", objective)]
                            + list(parts.items()))
    if numeric["nonfinite"]:
        raise AssertionError(f"{arm} nonfinite forward/loss: {numeric}")
    aux = model.xyz_last
    geometry_check = finite_tensors([(name, aux[name]) for name in
                                     ("physical_velocity", "motion_delta", "verts", "tok")]
                                    + [(f"assoc.{name}", value) for name, value in aux["assoc"].items()])
    selected_bins = aux["time_bin"][aux["mask"]]
    bins = sorted(int(value) for value in selected_bins.unique().cpu())
    dynamic = "verts_by_time" in aux
    nonzero = bool((aux["physical_velocity"] != 0).any())
    if not nonzero or bins != [0, 1, 2, 3] or (arm.endswith("t1") and not dynamic):
        raise AssertionError(f"dynamic coverage missing: {arm}, velocity={nonzero}, bins={bins}, dynamic={dynamic}")
    if dynamic:
        geometry_check["nonfinite"].extend(finite_tensors([
            ("verts_by_time", aux["verts_by_time"]),
            ("record_geometry", aux["record_geometry"])])["nonfinite"])
    if geometry_check["nonfinite"]:
        raise AssertionError(f"{arm} nonfinite geometry: {geometry_check}")
    objective.backward()
    grads = finite_tensors((name, p.grad) for name, p in model.named_parameters())
    if grads["nonfinite"] or grads["checked"] == 0:
        raise AssertionError(f"{arm} invalid gradients: {grads}")
    if not bool((model.xyz.fc1.weight.grad[:, -51:] != 0).any()):
        raise AssertionError(f"{arm} common history columns have no gradient")
    optimizer.step()
    weights = finite_tensors(model.state_dict().items())
    opt = finite_tensors((f"state{i}.{name}", value) for i, state in enumerate(optimizer.state.values())
                         for name, value in state.items() if torch.is_tensor(value))
    if weights["nonfinite"] or opt["nonfinite"]:
        raise AssertionError(f"{arm} invalid post-step state: weights={weights}, optimizer={opt}")
    torch.cuda.synchronize()
    result = {"config": f"configs/xc/xc_{arm}_s3407.yaml", "config_sha256": sha(REPO / f"configs/xc/xc_{arm}_s3407.yaml"),
              "actual_microbatch": 512, "leading_plus_main_forward": True,
              "precision": "bf16 autocast; declared geometry/SO3 fp32", "physical_gpu": 4,
              "forced_nonzero_fc2": True, "nonzero_velocity": nonzero,
              "four_time_bins": bins, "dynamic_geometry_executed": dynamic,
              "paired_sha256": pair_hash(pair), "objective": float(objective),
              "gradient_audit": grads, "weight_audit": weights, "optimizer_audit": opt,
              "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
              "elapsed_s": time.monotonic() - began, "actual_optimizer_steps": 1,
              "history_cleared": model._time_history is None,
              "parameter_count": sum(p.numel() for p in model.parameters())}
    del predicted, target, betas, objective, loss, parts, aux, batches, optimizer, model
    gc.collect()
    torch.cuda.empty_cache()
    return result


def trainer_smoke(smoke):
    import torch
    import yaml
    cfg = config("g1t1")
    directory = smoke / "entry_g1t1_s3407"
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError("smoke output already exists; refuse implicit resume or overwrite")
    cfg["TRAIN"].update(MAX_STEPS=2, SAVE_EVERY_N_STEPS=2, VAL_CHECK_INTERVAL=500,
                         OUTPUT_DIR=str(directory), RUN_NAME="xc_entry_smoke_g1t1_s3407",
                         NUM_WORKERS=12, PREFETCH_FACTOR=1)
    cfg["EVAL"]["OUTPUT_DIR"] = str(directory / "eval")
    smoke_cfg = smoke / "entry_config.yaml"
    smoke_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False))
    command = [PYTHON, "tools/xc/train.py", "--config", str(smoke_cfg), "--no-logger"]
    log = smoke / "trainer_entry.log"
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=REPO, env=dict(os.environ), stdout=handle,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"2-step trainer-entry smoke failed; see {log}")
    checkpoints = list(directory.glob("*-step=2.ckpt"))
    if len(checkpoints) != 1:
        raise AssertionError(f"expected exactly one named step2 smoke checkpoint, got {checkpoints}")
    checkpoint = torch.load(checkpoints[0], map_location="cpu")
    if checkpoint.get("global_step") != 2:
        raise AssertionError("trainer smoke did not reach exact optimizer step2")
    numeric = finite_tensors(checkpoint["state_dict"].items())
    if numeric["nonfinite"]:
        raise AssertionError(f"nonfinite smoke checkpoint weights: {numeric}")
    metadata_path = directory / "training_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    expected = {"seed": 3407, "max_steps": 2, "devices": 1,
                "batch_size_per_gpu": 512, "accumulate_grad_batches": 2,
                "effective_batch": 1024, "motion_delta_dim": 51,
                "xyz_geometry": True, "xyz_time_mode": "extrapolate4",
                "precision": "bf16", "data_generator_seed": 3407,
                "cuda_visible_devices": "4", "model_class": "TimeXYZModel",
                "resumed_from": None, "frozen_manifest_sha256": None,
                "requested_data_unroll_pair": True, "effective_track_unroll_pair": True}
    bad = {key: {"actual": metadata.get(key), "expected": value}
           for key, value in expected.items() if key not in metadata or metadata[key] != value}
    if bad:
        raise AssertionError(f"trainer smoke metadata mismatch: {bad}")
    for name, digest in metadata["source_sha256"].items():
        if sha(REPO / name) != digest:
            raise AssertionError(f"trainer metadata source drift: {name}")
    return {"actual_optimizer_steps": 2, "formal_study_run": False,
            "command": command, "config": str(smoke_cfg.relative_to(REPO)),
            "config_sha256": sha(smoke_cfg), "checkpoint": str(checkpoints[0].relative_to(REPO)),
            "checkpoint_sha256": sha(checkpoints[0]), "global_step": checkpoint["global_step"],
            "metadata": str(metadata_path.relative_to(REPO)), "metadata_sha256": sha(metadata_path),
            "metadata_contract": expected, "state_dict_audit": numeric,
            "log": str(log.relative_to(REPO)), "log_sha256": sha(log)}


def default_s39(smoke):
    import numpy as np
    import torch
    import yaml
    from model import MNISTModel
    from semkine import eval_track as ET
    from semkine.xyz_time_study import TimeXYZModel
    folder = REPO / "outputs/semkine/dt_dz_l3_s3407"
    cfg = yaml.safe_load((folder / "config_resolved.yaml").read_text())
    checkpoint_path = folder / "dt_dz_l3_s3407-step=6000.ckpt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    baseline = MNISTModel(cfg).cuda().eval()
    baseline.load_state_dict(checkpoint["state_dict"], strict=True)
    from semkine.dataset import sequences_for_split
    legacy_dirs = dict(sequences_for_split(REPO / "data/hand_data51", "val_core"))
    events, offsets, aux, pos51 = ET.load_sequence(REPO / "data/hand_data51", legacy_dirs["zgz_global"], "zgz_global")
    run_start = int(np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)[0, 0])
    x = torch.from_numpy(ET.build_lnes(events, offsets, run_start + 199, 200)).unsqueeze(0).cuda()
    previous = torch.tensor(pos51[run_start], device="cuda").reshape(1, 51)
    betas = torch.tensor(aux["betas"], device="cuda").reshape(1, -1)
    camera = torch.tensor(aux["camera_K"], device="cuda").reshape(1, 3, 3)
    if hasattr(baseline, "set_hand_context"):
        baseline.set_hand_context(betas, camera)
    before_state = tensor_hash(sorted(baseline.state_dict().items()))
    with torch.no_grad():
        before = baseline(x, previous, betas=betas, camera_K=camera).clone()
        temporary = TimeXYZModel(config("g1t1"))
        del temporary
        after = baseline(x, previous, betas=betas, camera_K=camera).clone()
    after_state = tensor_hash(sorted(baseline.state_dict().items()))
    if not torch.equal(before, after) or before_state != after_state:
        raise AssertionError("constructing the study subclass changed actual S39 baseline")
    result = {"checkpoint": str(checkpoint_path.relative_to(REPO)), "checkpoint_sha256": sha(checkpoint_path),
              "actual_w200_input": True, "prediction_bitwise_equal": True,
              "state_dict_bitwise_equal": True, "baseline_state_dict_sha256": before_state}
    del baseline, checkpoint, x, previous, betas, camera, before, after
    gc.collect()
    torch.cuda.empty_cache()
    return result


def supplementary_checks():
    """Small operator/interface checks, never a main-row or latency measurement."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("supplementary CUDA check exclusively uses physical GPU4")
    import dataclasses
    import numpy as np
    import torch
    from semkine import eval_track as ET
    from semkine.dataset import sequences_for_split
    from semkine.xyz_time_study import TimeXYZModel
    from tools.xc import diagnostics as diagnostic
    from tools.xc import make_row as row_tools
    report = json.loads((OUT / "preflight_draft.json").read_text())
    if report.get("status") != "DRAFT_PASS" or report.get("failures") != []:
        raise RuntimeError("passing actual512 draft required before supplementary checks")
    for name, digest in report["critical_source_sha256"].items():
        if sha(REPO / name) != digest:
            raise RuntimeError(f"critical source drifted before supplement: {name}")
    evidence = {"started_at": stamp(), "formal_result": False,
                "source_sha256": {name: sha(REPO / name) for name in
                                  ("tools/xc/preflight.py", "tools/xc/diagnostics.py", "tools/xc/make_row.py")}}
    root = REPO / "data/hand_data51"
    directory = dict(sequences_for_split(root, "val_core"))["zgz_global"]
    events, offsets, aux, truth = ET.load_sequence(root, directory, "zgz_global")
    first_ms = int(np.asarray(aux["valid_runs_ms"]).reshape(-1, 2)[0, 0])
    tsub = np.load(root / directory / "zgz_global_tsub.npy", mmap_mode="r")
    ev = ET._window_events(events, offsets, tsub, first_ms + 49, 50)
    for geometry in (False, True):
        torch.manual_seed(3407)
        model = TimeXYZModel(config("g1t1" if geometry else "g0t1")).cuda().eval()
        model.xyz_keep_aux = True
        with torch.no_grad():
            generator = torch.Generator(device="cuda").manual_seed(11973)
            model.xyz.fc2.weight.copy_(torch.randn(model.xyz.fc2.weight.shape, generator=generator,
                                                   device="cuda") * .002)
            beta = torch.tensor(aux["betas"], device="cuda").reshape(1, -1)
            camera = torch.tensor(aux["camera_K"], device="cuda").reshape(1, 3, 3)
            prior = torch.tensor(truth[first_ms], device="cuda").reshape(1, 51)
            packet = ET.make_eval_packet(ev, prior, beta, camera, 50, "cuda")
            for enabled in (False, True):
                model.reset_time_history()
                model.time_mode = "static"
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                    reference = model.forward_packet(packet).clone()
                model.reset_time_history()
                model.time_mode = "extrapolate4"
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                    actual = model.forward_packet(packet).clone()
                if not torch.equal(reference, actual):
                    raise AssertionError(f"CUDA no-history parity failed G={geometry}, bf16={enabled}")
                model.set_time_history(prior, .049, query_start_offset_s=.001)
                model.time_mode = "static"
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                    stationary_reference = model.forward_packet(packet).clone()
                model.time_mode = "extrapolate4"
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled):
                    stationary_actual = model.forward_packet(packet).clone()
                if not torch.equal(stationary_reference, stationary_actual):
                    raise AssertionError(f"CUDA zero-speed parity failed G={geometry}, bf16={enabled}")
                evidence[f"cuda_fallback_G{int(geometry)}_bf16{int(enabled)}"] = {
                    "no_history_bitwise_equal": True, "zero_velocity_bitwise_equal": True,
                    "n_actual_raw_events": len(ev), "nonzero_head": True}
        del model, packet, reference, actual, stationary_reference, stationary_actual
        gc.collect()
        torch.cuda.empty_cache()
    # deterministic tiny common-input audit stays on CPU because this parent
    # process did not configure CUBLAS_WORKSPACE_CONFIG before CUDA initialization.
    cfg, sequences, traces, selected, _ = diagnostic.fixed_inputs()
    first_start = selected["zgz_global"][0]
    model = TimeXYZModel(config("g1t1")).cpu().eval()
    with torch.no_grad():
        torch.manual_seed(11973)
        model.xyz.fc2.weight.normal_(std=.002)
    original_window = diagnostic.D.ET._window_events
    def tiny_window(*args, **kwargs):
        full = original_window(*args, **kwargs)
        indices = np.linspace(0, len(full) - 1, min(8, len(full))).astype(int)
        return full[indices].copy()
    try:
        diagnostic.D.ET._window_events = tiny_window
        audit = diagnostic.common_time_audit(model, sequences, traces,
                                             {"zgz_global": [first_start]}, torch.device("cpu"))
    finally:
        diagnostic.D.ET._window_events = original_window
    if audit["contract_pass"] is not True or audit["records"][0]["n_raw_events"] != 8:
        raise AssertionError(f"actual tiny common-time audit failed: {audit}")
    evidence["actual8_event_common_time_audit"] = {"device": "cpu", "scope": "one fixed natural frame, uniformly eight events",
                                                    "audit": audit, "diagnostic_strictness_unchanged": True}
    del model
    # Exercise own-loop staging/reset/actual clocks using eight real lyq_local
    # packets. This is an interface check, not a published timing sample.
    torch.manual_seed(3407)
    model = TimeXYZModel(config("g1t1")).cuda().eval()
    with torch.no_grad():
        generator = torch.Generator(device="cuda").manual_seed(11973)
        model.xyz.fc2.weight.copy_(torch.randn(model.xyz.fc2.weight.shape, generator=generator,
                                               device="cuda") * .002)
    packets, contexts, counts, audit = row_tools.stage_latency_packets(model, config("g1t1"), root,
                                                                       torch.device("cuda"), max_packets=8)
    if (audit["n_packets"] != 8 or audit["n_nonzero_history_packets"] != 7 or
            audit["n_dynamic_geometry_packets"] != 7 or contexts[0]["previous_incoming_state"] is not None or
            contexts[0]["query_start_offset_s"] != 0 or contexts[1]["past_interval_s"] != .049):
        raise AssertionError(f"latency staging tiny interface failed: {audit}")
    for packet, context in zip(packets, contexts):
        row_tools.restore_latency_context(model, context)
        with torch.no_grad():
            prediction = model.forward_packet(packet)
        if not bool(torch.isfinite(prediction).all()):
            raise AssertionError("restored staged packet had a nonfinite prediction")
    evidence["latency_staging_interface"] = {"formal_latency": False, "audit": audit}
    del model, packets, contexts
    gc.collect()
    torch.cuda.empty_cache()
    macs, note = row_tools.macs_forward_packet(config("g1t1"))
    if not np.isfinite(macs) or macs <= 0 or row_tools._MAC_AUDIT.get("dynamic_geometry_executed") is not True:
        raise AssertionError("standard-layer MAC fixture failed to execute dynamic T1 geometry")
    evidence["mac_fixture_interface"] = {"formal_main_row": False, "standard_layer_lower_bound_macs": macs,
                                          "note": note, "audit": row_tools._MAC_AUDIT}
    evidence.update(status="PASS", finished_at=stamp())
    report["checks"]["supplementary_interfaces"] = {"status": "PASS", "result": evidence}
    write(OUT / "preflight_draft.json", report)
    write(Path(report["smoke_directory"]) / "supplementary_interfaces.json", evidence)
    say({"preflight_supplement": "PASS", "scope": "fallback and reporting interfaces only; no new research score"})


def run_draft():
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "4":
        raise RuntimeError("preflight exclusively uses physical GPU4: CUDA_VISIBLE_DEVICES=4")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected one visible authorized CUDA device")
    when = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    smoke = OUT / "smoke" / when
    smoke.mkdir(parents=True)
    sources, missing = source_hashes()
    previous_failures = [{"artifact": str(path.relative_to(REPO)), "sha256": sha(path)}
                         for path in sorted((OUT / "smoke").glob("*/failed_draft.json"))]
    report = {"study": "XYZ-GEOMETRY-CAUSAL-TIME-20261012", "started_at": stamp(),
              "status": "DRAFT_RUNNING", "failures": [], "checks": {},
              "smoke_directory": str(smoke.relative_to(REPO)),
              "physical_gpu": 4, "gpu_name": torch.cuda.get_device_name(),
              "source_sha256": sources, "missing_source_files": missing,
              "critical_source_sha256": {name: sha(REPO / name) for name in CRITICAL},
              "source_snapshot_note": "draft; helpers may still be under construction; no freeze permission",
              "historical_failed_preflight_artifacts": previous_failures}
    draft = OUT / "preflight_draft.json"
    write(draft, report)
    def check(name, function):
        say({"preflight_check": name, "state": "start"})
        began = time.monotonic()
        try:
            value = function()
            report["checks"][name] = {"status": "PASS", "elapsed_s": time.monotonic() - began,
                                       "result": value}
            say({"preflight_check": name, "state": "PASS", "elapsed_s": time.monotonic() - began})
            write(draft, report)
            return value
        except Exception as error:
            details = {"check": name, "type": type(error).__name__, "error": str(error),
                       "traceback": traceback.format_exc()}
            report["failures"].append(details)
            report["checks"][name] = {"status": "FAIL", "elapsed_s": time.monotonic() - began}
            report.update(status="DRAFT_FAIL", finished_at=stamp())
            write(draft, report)
            say({"preflight_check": name, "state": "FAIL", "error": str(error)})
            raise
    check("unchanged_default_sources", immutable_default_sources)
    check("same_seed_initializations", initializations)
    pair, information = actual_batch()
    report["actual_raw_batch"] = information
    write(draft, report)
    for arm in ("g0t0", "g1t0", "g0t1", "g1t1"):
        check(f"cuda_actual512_{arm}", lambda arm=arm: cuda_arm(arm, pair))
    del pair
    gc.collect()
    check("actual_s39_default_invariance", lambda: default_s39(smoke))
    check("trainer_entry_step2", lambda: trainer_smoke(smoke))
    check("cpu_operator_tests", lambda: cpu_tests(smoke))
    check("unchanged_default_sources_after", immutable_default_sources)
    report.update(status="DRAFT_PASS", finished_at=stamp())
    report["source_sha256_at_draft_end"], report["missing_source_files_at_draft_end"] = source_hashes()
    write(draft, report)
    say({"preflight": "DRAFT_PASS", "artifact": str(draft.relative_to(REPO)),
         "final_freeze_permission": False})


def finalize():
    report = json.loads((OUT / "preflight_draft.json").read_text())
    if report.get("status") != "DRAFT_PASS" or report.get("failures") != []:
        raise RuntimeError("complete passing draft required")
    if report["checks"].get("supplementary_interfaces", {}).get("status") != "PASS":
        raise RuntimeError("supplementary fallback/reporting interface check required")
    for name, digest in report["critical_source_sha256"].items():
        if sha(REPO / name) != digest:
            raise RuntimeError(f"critical exercised source drifted since CUDA smoke: {name}; rerun/revalidate before final")
    hashes, missing = source_hashes()
    if missing:
        raise RuntimeError(f"final helper inputs missing: {missing}")
    tested = report["checks"]["cpu_operator_tests"]["result"]["test_source_sha256"]
    current_tests = set(str(p.relative_to(REPO)) for p in (REPO / "tests").glob("test_xc*.py"))
    current_tests.add("tests/test_xyz_time_study.py")
    if set(tested) != current_tests or any(sha(REPO / name) != digest for name, digest in tested.items()):
        old_tests = report["checks"]["cpu_operator_tests"]
        report["checks"]["cpu_operator_tests_at_draft"] = old_tests
        report["checks"]["cpu_operator_tests"] = {"status": "PASS",
                                                   "result": cpu_tests(REPO / report["smoke_directory"])}
    prior = report.pop("source_sha256")
    report["source_sha256_at_draft_start"] = prior
    report.update(status="PASS", finalized_at=stamp(), source_sha256=hashes,
                  missing_source_files=[], source_snapshot_note="final complete source_files() identity snapshot")
    write(OUT / "preflight.json", report)
    say({"preflight": "PASS", "artifact": "outputs/xc/preflight.json", "source_files": len(hashes)})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalize", action="store_true", help="finalize an exercised passing draft after all helper files exist")
    parser.add_argument("--supplement", action="store_true", help="tiny fallback/diagnostic/timing/MAC interfaces on an existing passing draft")
    args = parser.parse_args()
    if args.finalize and args.supplement:
        parser.error("choose --finalize or --supplement")
    if args.supplement:
        supplementary_checks()
    else:
        finalize() if args.finalize else run_draft()
