#!/usr/bin/env python3
"""Fixed XCL evaluation queue: serial latency rows, then GPU4--7 FIFO readers."""
from __future__ import annotations
import argparse
from collections import deque
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.xcl.evaluate import compare_traces, path, resolve_checkpoint, sha


def jobs(protocol, output):
    rows, traces = [], []
    def job(run, mode, step=6000, repeat=1, intervention="none"):
        name = f"{mode}_step{step}_repeat{repeat}_{intervention}"
        return {"run": run["name"], "cell": run["cell"], "mode": mode, "step": step,
                "intervention": intervention, "output": str(output / run["name"] / name)}
    for run in protocol["runs"]:
        rows.append(job(run, "row"))
        traces.extend(job(run, mode, step, repeat) for mode, step, repeat in (
            ("e1", 6000, 1), ("e1", 6000, 2), ("e2", 6000, 1), ("e3", 6000, 1),
            ("gt_pose", 6000, 1), ("gt_transl", 6000, 1), ("e1", 5000, 1), ("e1", 5500, 1)))
        if run["cell"] in ("a0f0", "a0f1", "a2f1", "a3f0"):
            traces.extend(job(run, mode, intervention=intervention) for intervention in (
                "geometry_zero", "h3_zero", "residual_zero") for mode in ("e1", "e2", "e3"))
        if run["cell"] in ("a0f0", "a0f1"):
            traces.extend(job(run, mode, intervention=intervention) for intervention in (
                "route_a2", "route_a3") for mode in ("e2", "e3"))
    return rows, traces


class Queue:
    def __init__(self, protocol_file, protocol, output):
        self.protocol_file, self.protocol, self.output = protocol_file, protocol, output
        self.owned, self.handles = {}, []
        self.state = {"status": "STARTING", "protocol_sha256": sha(protocol_file), "jobs": []}

    def publish(self):
        tmp = self.output / "queue.tmp"
        tmp.write_text(json.dumps(self.state, indent=2, allow_nan=False) + "\n")
        tmp.replace(self.output / "queue.json")

    def launch(self, job, gpu):
        command = [sys.executable, "tools/xcl/evaluate.py", "--protocol", str(self.protocol_file),
                   "--run", job["run"], "--mode", job["mode"], "--step", str(job["step"]),
                   "--intervention", job["intervention"], "--output-dir", job["output"]]
        log = Path(job["output"]).with_suffix(".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        handle = log.open("x"); self.handles.append(handle)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        record = {**job, "gpu": gpu, "pid": child.pid, "status": "RUNNING"}
        self.state["jobs"].append(record); self.owned[child.pid] = (child, record)
        self.publish()
        return child

    def complete(self, child):
        child, record = self.owned[child.pid]
        code = child.wait()
        record.update(status="EXITED", exit_code=code)
        if code != 0:
            raise RuntimeError(f"Evaluation failed: {record['run']}/{record['mode']} exit={code}")
        report_file = Path(record["output"]) / "report.json"
        report = json.loads(report_file.read_text())
        if report["status"] != "COMPLETE" or report["n_endpoints"] != 2590 or report["n_failed"] != 0:
            raise RuntimeError("Incomplete evaluation denominator")
        record.update(status="COMPLETE", report_sha256=sha(report_file))
        del self.owned[child.pid]
        self.publish()

    def cleanup(self):
        for child, _ in self.owned.values():
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 10
        while any(child.poll() is None for child, _ in self.owned.values()) and time.monotonic() < deadline:
            time.sleep(.2)
        for child, _ in self.owned.values():
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            child.wait()

    def run(self, rows, traces):
        try:
            self.state["status"] = "SERIAL_LATENCY_ROWS"; self.publish()
            for index, job in enumerate(rows):
                self.complete(self.launch(job, (4, 5, 6, 7)[index % 4]))
            self.state["status"] = "FOUR_GPU_READOUTS"; self.publish()
            pending, active = deque(traces), {}
            while pending or active:
                for gpu in (4, 5, 6, 7):
                    if gpu not in active and pending:
                        active[gpu] = self.launch(pending.popleft(), gpu)
                for gpu, child in list(active.items()):
                    if child.poll() is not None:
                        self.complete(child); del active[gpu]
                if pending or active:
                    time.sleep(1)
            comparisons = {}
            for run in self.protocol["runs"]:
                directory = self.output / run["name"]
                first = directory / "e1_step6000_repeat1_none/states.npz"
                comparisons[run["name"]] = compare_traces(first, directory / "e1_step6000_repeat2_none/states.npz")
                compare_traces(first, directory / "row_step6000_repeat1_none/states.npz")
            self.state["fresh_repeat_comparisons"] = comparisons
            command = [sys.executable, "tools/xcl/report.py", "--protocol", str(self.protocol_file),
                       "--evaluations-root", str(self.output), "--output-dir", str(self.output.parent / "report")]
            # Reporter is CPU only and remains in this owned process group.
            subprocess.run(command, cwd=ROOT, check=True)
            self.state["status"] = "COMPLETE"; self.publish()
        except BaseException as exc:
            self.cleanup()
            self.state.update(status="FAILED_PRESERVED", error=f"{type(exc).__name__}: {exc}")
            self.publish()
            raise
        finally:
            for handle in self.handles:
                handle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--protocol", required=True); parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    protocol_file = path(args.protocol).resolve(); protocol = json.loads(protocol_file.read_text())
    output = protocol_file.parent / "evaluations"
    rows, traces = jobs(protocol, output)
    if args.dry_run:
        print(json.dumps({"serial_main_rows": rows, "parallel_readouts": traces, "physical_gpus": [4, 5, 6, 7]}, indent=2)); return
    from tools.xcl.common import verify_protocol
    verify_protocol(protocol_file)
    for run in protocol["runs"]:
        if not run["reuse"]:
            receipt = json.loads((protocol_file.parent / "receipts" / ("formal_" + run["cell"] + ".json")).read_text())
            if receipt.get("status") != "PASS" or receipt.get("phase") != "formal":
                raise RuntimeError("All five training acceptance receipts are required before evaluation")
        for step in (5000, 5500, 6000):
            resolve_checkpoint(run, step)
    output.mkdir(parents=True, exist_ok=False)
    def interrupted(number, frame):
        raise InterruptedError("Evaluation queue interrupted: " + str(number))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    Queue(protocol_file, protocol, output).run(rows, traces)


if __name__ == "__main__":
    main()
