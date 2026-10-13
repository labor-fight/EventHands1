#!/usr/bin/env python3
"""Prepare and execute the preregistered S37 residual ABCD study on GPUs 4–7.

Preparation never launches a GPU process.  `freeze` captures configs, split, code,
and the preregistration section.  `run` checks that frozen identity first and
executes every requested arm, independent of pilot accuracy.  No checkpoint
selection runs.  Existing exact final checkpoints are audited before reuse.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading

import yaml

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/xb"
CONFIG_DIR = REPO / "configs/xb"
PYTHON = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
GPUS = (4, 5, 6, 7)
SEEDS = (3407, 3408, 3409)
ARMS = {"a": "A", "b": "B", "c": "C", "d": "D"}
SECTION = "XYZ-RESIDUAL-ABCD-20261011"
EVENT_LOCK = threading.Lock()


def stamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def event(**kw):
    with EVENT_LOCK:
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "events.jsonl").open("a") as f:
            f.write(json.dumps({"time": stamp(), **kw}, ensure_ascii=False) + "\n")
        print(json.dumps(kw, ensure_ascii=False), flush=True)


def prereg_text():
    text = (REPO / "docs/EXPERIMENT_RECORDS.md").read_text()
    begin, end = (f"<!-- EXPERIMENT_RECORD:{SECTION}:{kind} -->" for kind in ("BEGIN", "END"))
    if text.count(begin) != 1 or text.count(end) != 1:
        raise RuntimeError("missing/ambiguous ABCD preregistration section")
    return text.split(begin, 1)[1].split(end, 1)[0]


def config_for(arm, seed, phase, repeat=False):
    cfg = yaml.safe_load((REPO / "configs/xa/xa_c0_s3407.yaml").read_text())
    run = f"xb_p{arm}_s{seed}" if phase == "pilot" else f"xb_{arm}{'r' if repeat else ''}_s{seed}"
    cfg["SEED"] = seed
    cfg["MODEL"].update(XYZ_RESIDUAL_MODE=ARMS[arm], XYZ_RESIDUAL_SEED=19073)
    cfg["TRAIN"].update(DEVICES=1, STRATEGY="auto", BATCH_SIZE_PER_GPU=512,
                         ACCUMULATE_GRAD_BATCHES=2, MAX_STEPS=1500 if phase == "pilot" else 6000,
                         RUN_NAME=run, OUTPUT_DIR=f"outputs/semkine/{run}")
    cfg["EVAL"]["OUTPUT_DIR"] = f"outputs/semkine/{run}/eval"
    if phase == "pilot":
        cfg["TRAIN"].update(PRECISION="32", ACCUMULATE_GRAD_BATCHES=1)
        cfg["MODEL"]["INIT_FROM"] = "outputs/semkine/xa_c0_s3407/xa_c0_s3407-step=6000.ckpt"
        cfg["TRAIN"]["TRAINABLE_PREFIXES"] = ["xyz.residual.", "xyz.fc1.", "xyz.fc2."]
    return run, cfg


def jobs():
    out = []
    for phase, seeds in (("pilot", SEEDS[:2]), ("full", SEEDS)):
        for seed in seeds:
            for arm in ARMS:
                run, cfg = config_for(arm, seed, phase)
                out.append({"run": run, "phase": phase, "arm": arm, "seed": seed,
                            "config": f"configs/xb/{run}.yaml", "max_steps": cfg["TRAIN"]["MAX_STEPS"],
                            "effective_batch": 512 if phase == "pilot" else 1024, "repeat": False})
    run, cfg = config_for("a", 3407, "full", repeat=True)
    out.append({"run": run, "phase": "full", "arm": "a", "seed": 3407,
                "config": f"configs/xb/{run}.yaml", "max_steps": 6000,
                "effective_batch": 1024, "repeat": True})
    return out


def prepare():
    if (OUT / "frozen_manifest.json").exists():
        raise RuntimeError("study is already frozen; refuse to regenerate configs")
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    js = jobs()
    for job in js:
        run, cfg = config_for(job["arm"], job["seed"], job["phase"], job["repeat"])
        path = REPO / job["config"]
        path.write_text(f"# [XYZ-RESIDUAL-ABCD] {run}; prereg docs/EXPERIMENT_RECORDS.md#xyz-residual-abcd-20261011\n"
                        + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
        job["config_sha256"] = sha(path)
    manifest = {"study": SECTION, "created_at": stamp(), "status": "draft_no_gpu_launch",
                "allowed_physical_gpus": list(GPUS), "python": PYTHON, "jobs": js,
                "primary_checkpoint": "exact MAX_STEPS, no selection",
                "evaluation": {"window_ms": 50, "split": "val_core", "subject": "zgz",
                               "manifest": "data/hand_data51/splits_semkine.json", "rng_seed": 0},
                "total_optimizer_steps": sum(j["max_steps"] for j in js),
                "total_samples": sum(j["max_steps"] * j["effective_batch"] for j in js)}
    write_json(OUT / "study_manifest.json", manifest)
    lines = ["# ABCD execution queue (draft; no GPU tasks launched)", "",
             "Allowed physical GPUs: 4, 5, 6, 7. Pilot effective batch 512; full effective batch 1 × 512 × 2 = 1024.", "",
             "| Phase | Run | Mode | Seed | Steps | Config SHA256 |", "|---|---|---|---|---|---|"]
    lines += [f"| {j['phase']} | {j['run']} | {ARMS[j['arm']]} | {j['seed']} | {j['max_steps']} | {j['config_sha256']} |"
              for j in js]
    (OUT / "QUEUE.md").write_text("\n".join(lines) + "\n")
    print(f"Prepared {len(js)} configurations; {manifest['total_optimizer_steps']} optimizer steps. No launch.")


def freeze():
    path = OUT / "frozen_manifest.json"
    if path.exists():
        raise RuntimeError("frozen manifest already exists")
    manifest = json.loads((OUT / "study_manifest.json").read_text())
    files = [REPO / j["config"] for j in manifest["jobs"]]
    files += [REPO / p for p in ("AGENTS.md", "semkine/xyz_mesh.py", "model/model.py",
                                 "semkine/train.py", "semkine/dataset.py", "semkine/eval_track.py",
                                 "tools/make_s36_row.py", "tools/report_table.py", "assets/mano_right.npz",
                                 "data/hand_data51/splits_semkine.json", "semkine/xyz_residual.py",
                                 "outputs/xb/c0_train_bank.pt", "outputs/xb/c0_train_bank.json",
                                 "tests/test_xyz_residual.py")]
    files += sorted((REPO / "tools/xb").glob("*.py"))
    files += sorted((REPO / "tests").glob("test_xb*.py"))
    files += [REPO / "outputs/semkine/xa_c0_s3407/xa_c0_s3407-step=6000.ckpt"]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise RuntimeError(f"missing frozen inputs: {missing}")
    manifest.update(status="frozen_before_gpu_training", frozen_at=stamp(),
                    files_sha256={str(p.relative_to(REPO)): sha(p) for p in files},
                    prereg_sha256=hashlib.sha256(prereg_text().encode()).hexdigest(),
                    prereg_text=prereg_text())
    write_json(path, manifest)
    print(f"Frozen {len(files)} inputs in {path}")


def frozen():
    manifest = json.loads((OUT / "frozen_manifest.json").read_text())
    bad = [p for p, digest in manifest["files_sha256"].items() if not (REPO / p).is_file() or sha(REPO / p) != digest]
    if bad:
        raise RuntimeError(f"frozen inputs changed: {bad}")
    return manifest


def final_checkpoint(job):
    run = REPO / f"outputs/semkine/{job['run']}"
    matches = list(run.glob(f"*-step={job['max_steps']}.ckpt"))
    if len(matches) > 1:
        raise RuntimeError(f"ambiguous final checkpoint for {job['run']}: {matches}")
    return matches[0] if matches else None


def check_run_metadata(job):
    run = REPO / f"outputs/semkine/{job['run']}"
    meta = json.loads((run / "training_metadata.json").read_text())
    assertions = {"seed": job["seed"], "max_steps": job["max_steps"], "devices": 1,
                  "batch_size_per_gpu": 512, "accumulate_grad_batches": 1 if job["phase"] == "pilot" else 2,
                  "effective_batch": job["effective_batch"],
                  "cuda_visible_devices": str(int(meta["cuda_visible_devices"]))}
    for key, value in assertions.items():
        if meta.get(key) != value:
            raise RuntimeError(f"{job['run']} invalid metadata {key}: {meta.get(key)} != {value}")
    if int(meta["cuda_visible_devices"]) not in GPUS:
        raise RuntimeError(f"unauthorized GPU: {meta['cuda_visible_devices']}")
    expected_subjects = ["ch", "lfz", "lpc", "lr", "ly", "lyh", "lyq", "ycy", "ylf"]
    if sorted(meta["train_subjects"]) != sorted(expected_subjects) or len(meta["train_sequences"]) != 72:
        raise RuntimeError(f"{job['run']} wrong train split")
    if sorted(meta["val_core_sequences"]) != ["zgz_global", "zgz_local"]:
        raise RuntimeError(f"{job['run']} wrong eval split")
    if meta["provenance"]["config_sha256"] != sha(REPO / job["config"]):
        raise RuntimeError(f"{job['run']} config identity drift")


def job_env(gpu):
    if gpu not in GPUS:
        raise RuntimeError(f"GPU {gpu} is outside authorized 4–7")
    return {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "NCCL_P2P_DISABLE": "1"}


def train_one(job, gpu):
    name = job["run"]
    if final_checkpoint(job):
        check_run_metadata(job)
        event(action="reuse_completed", run=name, gpu=gpu)
        return True
    run_dir = REPO / f"outputs/semkine/{name}"
    if run_dir.exists() and any(run_dir.glob("*.ckpt")):
        raise RuntimeError(f"partial existing {name}; decide explicit resume/archive before rerun")
    log = OUT / f"logs/{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if job["phase"] == "pilot":
        # CLI is kept concrete in the manifest before freeze; no full-trunk pilot training.
        command = [PYTHON, "tools/xb/probe.py", "--config", job["config"],
                   "--init-ckpt", "outputs/semkine/xa_c0_s3407/xa_c0_s3407-step=6000.ckpt",
                   "--bank", "outputs/xb/c0_train_bank.pt", "--freeze", "probe", "--device", "cuda"]
    else:
        command = [PYTHON, "semkine/train.py", "--config", job["config"], "--seed", str(job["seed"]),
                   "--run-name", name, "--output-dir", f"outputs/semkine/{name}"]
    event(action="train_start", run=name, gpu=gpu, command=command, log=str(log))
    with log.open("w") as f:
        result = subprocess.run(command, cwd=REPO, env=job_env(gpu), stdout=f, stderr=subprocess.STDOUT)
    ok = result.returncode == 0 and final_checkpoint(job) is not None
    if ok:
        check_run_metadata(job)
    event(action="train_finish", run=name, gpu=gpu, returncode=result.returncode, exact_last=ok)
    return ok


def run(phase):
    manifest = frozen()
    js = [j for j in manifest["jobs"] if phase == "all" or j["phase"] == phase]
    # Keep the stage boundary explicit; pilot results never cancel the full phase.
    for current in ("pilot", "full"):
        todo = [j for j in js if j["phase"] == current]
        if not todo:
            continue
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            pending = {}
            remaining = iter(todo)
            for gpu in GPUS:
                job = next(remaining, None)
                if job:
                    pending[pool.submit(train_one, job, gpu)] = (job, gpu)
            failures = []
            while pending:
                done, _ = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    job, gpu = pending.pop(future)
                    try:
                        if not future.result():
                            failures.append(job["run"])
                    except Exception as exc:
                        failures.append(job["run"])
                        event(action="job_exception", run=job["run"], error=str(exc))
                    nxt = next(remaining, None)
                    if nxt:
                        pending[pool.submit(train_one, nxt, gpu)] = (nxt, gpu)
            if failures:
                raise RuntimeError(f"{current} incomplete jobs: {failures}; no scientific conclusions")
        event(action="phase_train_done", phase=current)
        check_streams(current)


def check_streams(phase):
    summaries = []
    for current, seeds in (("pilot", SEEDS[:2]), ("full", SEEDS)):
        if phase not in (current, "all"):
            continue
        for seed in seeds:
            prefix = "xb_p" if current == "pilot" else "xb_"
            names = [f"{prefix}{arm}_s{seed}" for arm in ARMS]
            if current == "full" and seed == 3407:
                names.append("xb_ar_s3407")
            streams = {}
            for name in names:
                base = REPO / f"outputs/semkine/{name}"
                path = base / ("pair_stream.json" if current == "pilot" else "pair_stream_rank0.json")
                records = json.loads(path.read_text())
                if len(records) != 3:
                    raise RuntimeError(f"{name} stream audit has {len(records)} rows; require initial three")
                streams[name] = [row["sha256"] for row in records]
            paired = all(value == streams[names[0]] for value in streams.values())
            summaries.append({"phase": current, "seed": seed, "streams": streams,
                              "initial_three_batches_identical": paired})
    path = OUT / f"paired_streams_{phase}.json"
    write_json(path, {"scope": "first three batches only; not a full-trajectory equality proof", "groups": summaries})
    if not all(row["initial_three_batches_identical"] for row in summaries):
        raise RuntimeError(f"ABCD paired stream mismatch; see {path}")
    print(f"Paired stream audit passed for {len(summaries)} seed groups.")


def row_tasks(phase):
    tasks = []
    if phase in ("pilot", "all"):
        tasks += [(f"xb_p{arm}", SEEDS[:2], 1500) for arm in ARMS]
    if phase in ("full", "all"):
        tasks += [(f"xb_{arm}", SEEDS, 6000) for arm in ARMS]
        tasks.append(("xb_ar", (3407,), 6000))
    return tasks


def row_one(task, gpu):
    run_name, seeds, step = task
    log = OUT / f"logs/{run_name}_row.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    # --step last is validated against exact MAX_STEPS before invocation.
    all_jobs = jobs()
    for seed in seeds:
        job = next(j for j in all_jobs if j["run"] == f"{run_name}_s{seed}")
        if final_checkpoint(job) is None:
            raise RuntimeError(f"missing exact final checkpoint: {job['run']}")
        check_run_metadata(job)
    command = [PYTHON, "tools/make_s36_row.py", "--run", run_name, "--seeds", ",".join(map(str, seeds)), "--step", "last"]
    event(action="row_start", run=run_name, gpu=gpu, command=command)
    with log.open("w") as f:
        result = subprocess.run(command, cwd=REPO, env=job_env(gpu), stdout=f, stderr=subprocess.STDOUT)
    path = REPO / f"outputs/semkine/{run_name}_main_row_last.json"
    if result.returncode or not path.exists():
        raise RuntimeError(f"row failed {run_name}; see {log}")
    row = json.loads(path.read_text())
    if any(v["step"] != step for v in row["per_seed"].values()):
        raise RuntimeError(f"row {run_name} did not score exact last={step}")
    row["protocol"].update(n_seeds=len(seeds), effective_batch=512 if step == 1500 else 1024, raw_window_ms=50,
                           study=SECTION,
                           note=f"accuracy = {len(seeds)}-seed mean, exact last step {step}, no checkpoint selection")
    write_json(path, row)
    event(action="row_finish", run=run_name, gpu=gpu, path=str(path))


def rows(phase):
    frozen()
    tasks = row_tasks(phase)
    # One full evaluation group per physical GPU. Avoid concurrent timing on the same device.
    def worker(gpu, work):
        for task in work:
            row_one(task, gpu)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(worker, gpu, tasks[i::4]) for i, gpu in enumerate(GPUS)]
        for future in futures:
            future.result()


def diagnostic_one(phase, seed, gpu):
    prefix = "xb_p" if phase == "pilot" else "xb_"
    output = OUT / f"recovery_{phase}_s{seed}.json"
    log = OUT / f"logs/recovery_{phase}_s{seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [PYTHON, "tools/xb/diagnostics.py", "--reference", "outputs/semkine/xa_c0_s3407",
               "--output", str(output), "--device", "cuda", "--steps", "10", "--clips-per-seq", "24",
               "--trace-cache", "outputs/xb/c0_val_reference.pt"]
    for arm in ARMS:
        command += ["--arm", f"{ARMS[arm]}=outputs/semkine/{prefix}{arm}_s{seed}"]
    event(action="diagnostics_start", run=f"{phase}_s{seed}", gpu=gpu, command=command)
    with log.open("w") as f:
        result = subprocess.run(command, cwd=REPO, env=job_env(gpu), stdout=f, stderr=subprocess.STDOUT)
    if result.returncode or not output.exists():
        raise RuntimeError(f"diagnostics failed for {phase}_s{seed}; see {log}")
    event(action="diagnostics_finish", run=f"{phase}_s{seed}", gpu=gpu, path=str(output))


def diagnostics(phase):
    frozen()
    for current, seeds in (("pilot", SEEDS[:2]), ("full", SEEDS)):
        if phase not in (current, "all"):
            continue
        # The first seed creates the shared fixed reference trace before concurrent readers.
        diagnostic_one(current, seeds[0], GPUS[0])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(diagnostic_one, current, seed, GPUS[i + 1])
                       for i, seed in enumerate(seeds[1:])]
            for future in futures:
                future.result()


def report(phase):
    run_names = [task[0] for task in row_tasks(phase)]
    labels = {"a": "A：全包容量对照", "b": "B：4×4局部残差", "c": "C：局部内容置乱", "d": "D：仅空间占用/质量"}
    command = [PYTHON, "tools/report_table.py", "--tag", "last", "xa_c0", *run_names,
               "--label", "xa_c0=旧匹配 C0（两 seed，末步参考）"]
    for name in run_names:
        key = name.removeprefix("xb_p").removeprefix("xb_")
        label = "A：同种子独立复跑" if key == "ar" else labels[key]
        command += ["--label", f"{name}={('探针 ' if name.startswith('xb_p') else '') + label}"]
    result = subprocess.run(command, cwd=REPO, check=True, capture_output=True, text=True)
    reference = subprocess.run([PYTHON, "tools/report_table.py", "--no-baseline", "dt3_dz_l3_w200cfg",
                                "--label", "dt3_dz_l3_w200cfg=S39：W200 + 自适应滤波 + 证据门（完整栈参考）"],
                               cwd=REPO, check=True, capture_output=True, text=True)
    table = result.stdout.rstrip() + "\n" + "\n".join(reference.stdout.splitlines()[2:]) + "\n"
    (OUT / f"TABLE_{phase}.md").write_text(table)
    print(table, end="")


def gate_summary():
    def get(name):
        return json.loads((REPO / f"outputs/semkine/{name}_main_row_last.json").read_text())
    all_rows = {arm: get(f"xb_{arm}") for arm in ARMS}
    repeated = get("xb_ar")["per_seed"]["3407"]
    blocks = ("overall", "local", "global")
    noise = {q: abs(all_rows["a"]["per_seed"]["3407"][q]["mpjpe_ra_mm"] - repeated[q]["mpjpe_ra_mm"]) for q in blocks}
    tau = {q: max(1.1, 2 * noise[q]) for q in blocks}
    delta = {}
    for arm in ("b", "c", "d"):
        delta[arm] = {q: [all_rows["a"]["per_seed"][str(s)][q]["mpjpe_ra_mm"]
                          - all_rows[arm]["per_seed"][str(s)][q]["mpjpe_ra_mm"] for s in SEEDS] for q in blocks}
    means = {arm: {q: sum(v) / len(v) for q, v in groups.items()} for arm, groups in delta.items()}
    b_gain = means["b"]["overall"]
    removal = (b_gain - means["c"]["overall"]) / b_gain if b_gain > 0 else None
    efficacy = (all(v > 0 for v in delta["b"]["overall"]) and b_gain > tau["overall"]
                and all(means["b"][q] >= -tau[q] for q in ("local", "global")))
    occupancy_equivalent = all(abs(means["b"][q] - means["d"][q]) <= tau[q] for q in blocks)
    out = {"preregistration": SECTION, "primary": "exact step6000/raw50 self closed loop",
           "delta_convention": "A error minus candidate error; positive means improvement",
           "seeds": list(SEEDS), "same_seed_repeat_abs_difference_mm": noise, "threshold_mm": tau,
           "paired_gain_mm": delta, "mean_gain_mm": means,
           "B_efficacy_gate": efficacy, "binding_gain_removed_fraction": removal,
           "binding_at_least_half_gate": removal is not None and removal >= 0.5,
           "D_explains_B_within_noise": occupancy_equivalent,
           "mechanism_contract_and_recovery_gate": "separate fixed probes; not inferred from these scalars",
           "research_adoption": "pending mechanism acceptance and recovery/motion probes"}
    write_json(OUT / "gates.json", out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


def complete():
    """Run both mandatory stages and their exact-last rows without an accuracy-dependent skip."""
    frozen()
    pipeline = [("train_pilot", lambda: run("pilot")), ("rows_pilot", lambda: rows("pilot")),
                ("report_pilot", lambda: report("pilot")), ("train_full", lambda: run("full")),
                ("rows_full", lambda: rows("full")), ("report_full", lambda: report("full")),
                ("primary_gates", gate_summary)]
    record = {"started_at": stamp(), "status": "running", "finished": [],
              "mechanism_and_recovery": "pending separate fixed diagnostics; primary gates do not imply research adoption"}
    path = OUT / "complete_status.json"
    write_json(path, record)
    for label, action in pipeline:
        record.update(current=label, updated_at=stamp())
        write_json(path, record)
        event(action="pipeline_stage_start", stage=label)
        try:
            action()
        except Exception as exc:
            record.update(status="failed", error=str(exc), failed_at=stamp())
            write_json(path, record)
            status()
            event(action="pipeline_error", stage=label, error=str(exc))
            raise
        record["finished"].append(label)
        record.update(updated_at=stamp())
        write_json(path, record)
        event(action="pipeline_stage_finish", stage=label)
    record.update(status="training_and_primary_evaluation_complete", completed_at=stamp())
    write_json(path, record)
    status()


def status():
    manifest = json.loads((OUT / "study_manifest.json").read_text())
    observations = {}
    event_path = OUT / "events.jsonl"
    if event_path.exists():
        for line in event_path.read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "run" in item:
                observations[item["run"]] = item
    def progress(job):
        base = REPO / f"outputs/semkine/{job['run']}"
        steps = [int(m.group(1)) for p in base.glob("*step=*.ckpt")
                 if (m := re.search(r"step=(\d+)", p.name))]
        curve_step = 0
        curve_path = base / "training_curve.json"
        if curve_path.exists():
            try:
                curve = json.loads(curve_path.read_text())
                curve_step = max((int(row["step"]) for row in curve), default=0)
            except json.JSONDecodeError:
                pass
        log = OUT / f"logs/{job['run']}.log"
        recent = observations.get(job["run"], {})
        return {**job, "exact_last_exists": final_checkpoint(job) is not None,
                "latest_saved_step": max(steps, default=0), "latest_probe_log_step": curve_step,
                "last_action": recent.get("action", "not_launched"), "gpu": recent.get("gpu"),
                "last_event_time": recent.get("time"), "log_bytes": log.stat().st_size if log.exists() else 0}
    state = {"time": stamp(), "frozen": (OUT / "frozen_manifest.json").exists(),
             "jobs": [progress(j) for j in manifest["jobs"]]}
    write_json(OUT / "status.json", state)
    for j in state["jobs"]:
        step = max(j["latest_saved_step"], j["latest_probe_log_step"])
        status_name = "DONE" if j["exact_last_exists"] else j["last_action"]
        print(f"{status_name:16s} {j['phase']:5s} {j['run']} step>={step}/{j['max_steps']} gpu={j['gpu']} log={j['log_bytes']}B")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("prepare", "freeze", "run", "rows", "report", "gates", "status", "streams", "diagnostics", "complete"))
    ap.add_argument("--phase", choices=("pilot", "full", "all"), default="all")
    args = ap.parse_args()
    os.chdir(REPO)
    if args.command in ("run", "rows", "report", "streams", "diagnostics"):
        {"run": run, "rows": rows, "report": report, "streams": check_streams, "diagnostics": diagnostics}[args.command](args.phase)
    else:
        {"prepare": prepare, "freeze": freeze, "gates": gate_summary, "status": status, "complete": complete}[args.command]()


if __name__ == "__main__":
    main()
