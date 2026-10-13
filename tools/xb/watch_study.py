#!/usr/bin/env python3
"""Observe the live ABCD queue without changing candidates, scores, or budgets."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import subprocess
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/xb"


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def snapshot():
    manifest = read_json(OUT / "frozen_manifest.json")
    events = {}
    for line in (OUT / "events.jsonl").read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if row.get("run"):
            events[row["run"]] = row
    rows, alerts = [], []
    for job in manifest["jobs"]:
        if job["phase"] != "full":
            continue
        name = job["run"]
        directory = REPO / "outputs/semkine" / name
        saved = [int(match.group(1)) for p in directory.glob("*step=*.ckpt")
                 if (match := re.search(r"step=(\d+)", p.name))]
        row = {"run": name, "arm": job["arm"], "seed": job["seed"], "repeat": job["repeat"],
               "max_steps": job["max_steps"], "saved_step": max(saved, default=0),
               "complete": job["max_steps"] in saved,
               "last_action": events.get(name, {}).get("action", "not_launched"),
               "physical_gpu": events.get(name, {}).get("gpu")}
        log_directory = directory / "logs" / name
        if log_directory.exists():
            try:
                acc = EventAccumulator(str(log_directory), size_guidance={"scalars": 1}).Reload()
                for tag in ("train_loss", "val_loss", "train_samples_per_sec"):
                    if tag in acc.Tags()["scalars"] and acc.Scalars(tag):
                        latest = acc.Scalars(tag)[-1]
                        finite = math.isfinite(latest.value)
                        row[tag] = {"step": latest.step, "value": latest.value if finite else None,
                                    "finite": finite, "age_seconds": max(0, time.time()-latest.wall_time)}
                        if not finite:
                            alerts.append(f"{name}/{tag}: measured nonfinite scalar at step {latest.step}")
            except (OSError, ValueError) as error:
                row["event_read_error"] = str(error)
        if not row["complete"] and row["last_action"] == "train_start":
            log = OUT / "logs" / f"{name}.log"
            if log.exists():
                tail = log.read_bytes()[-12000:].decode(errors="replace")
                if "Traceback (most recent call last)" in tail or "CUDA out of memory" in tail:
                    alerts.append(f"{name}: exception in training log {log}")
                age = row.get("train_loss", {}).get("age_seconds")
                if age is not None and age > 900:
                    alerts.append(f"{name}: no new training scalar for {int(age)} seconds; inspect process/validation")
        rows.append(row)
    gpu = subprocess.run(["nvidia-smi", "--id=4,5,6,7",
                          "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=10)
    queue, finish = read_json(OUT / "complete_status.json"), read_json(OUT / "finish_status.json")
    for name, state in (("primary_queue", queue), ("finish_queue", finish)):
        if state and state.get("status") == "failed":
            alerts.append(f"{name}: {state.get('error')}")
    return {"time": dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(),
            "primary_queue": queue, "finish_queue": finish, "full_jobs": rows,
            "completed_full_jobs": sum(row["complete"] for row in rows), "total_full_jobs": len(rows),
            "authorized_gpu_snapshot": gpu.stdout.splitlines(), "gpu_query_error": gpu.stderr if gpu.returncode else None,
            "alerts": alerts, "engineering_only": True,
            "note": "losses/early saved steps are health observations; never checkpoint selection or efficacy results"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("interval must be positive")
    while True:
        value = snapshot()
        target = OUT / "live_status.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
        temporary.replace(target)
        with (OUT / "live_history.jsonl").open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+"\n")
        active = {row["run"]: row.get("train_loss", {}).get("step", row["saved_step"])
                  for row in value["full_jobs"] if not row["complete"] and row["last_action"] == "train_start"}
        print(json.dumps({"time":value["time"], "phase": (value["primary_queue"] or {}).get("current"),
                          "full_done":value["completed_full_jobs"], "active_steps":active,
                          "finish_status":(value["finish_queue"] or {}).get("status"), "alerts":value["alerts"]},
                         ensure_ascii=False), flush=True)
        if args.once or (value["finish_queue"] or {}).get("status") in ("complete", "failed"):
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
