#!/usr/bin/env python3
"""C1SF detached driver, amendment 1 (docs/EXPERIMENT_RECORDS.md#c1sf-amend1-20261029); physical GPUs 2-7.

Step 1 and the evaluation pairing check already ran under the v1 protocol (step1.json, pairing_eval.json);
this driver requires them and starts at the preflights:
  1. engineering preflights (50 steps, one GPU pair each) of c1replay, dz, zr, so3 on pairs 2-3 / 4-5 / 6-7.
     c1replay (canonical recipe + amendment-1 loader resources) must reproduce C1's recorded steps 1-3 and every
     training packet bit for bit, otherwise no formal training starts.
  2. formal 6000-step training of dz, zr, so3 at the same time, one pair each.
  3. each finished arm's E1/E2/E3/E5/intervention readouts on GPUs no training uses, two trace processes per
     GPU (batch-1 traces leave most of an L20 idle; results are deterministic and independent of sharing).
  4. main rows (latency) serially on GPU 4 with nothing else on 2-7, then decide.py.
A failed training or evaluation is preserved and its arm is not retried (pre-registered default).
"""
from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.c1sf.common import (ARMS, FORMAL_ARMS, OUT, PHYSICAL_GPUS, PROTOCOL, config_path, now, read, require,  # noqa: E402
                               save, sha, verify_protocol, verify_run)

PY = sys.executable
STATUS = OUT / "status_r2.json"
LOGS = OUT / "logs_r2"
PAIRS = ((2, 3), (4, 5), (6, 7))
EVAL_SLOTS_PER_GPU = 2
EVAL_PLAN = (("e1", 6000, "none", 1), ("e1", 6000, "none", 2), ("e1", 5000, "none", 1), ("e1", 5500, "none", 1),
             ("e2", 6000, "none", 1), ("e3", 6000, "none", 1), ("e3", 6000, "geometry_zero", 1),
             ("e3", 6000, "h3_zero", 1))


def gpu_busy(gpu):
    q = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader"],
                       capture_output=True, text=True, check=True)
    return bool(q.stdout.strip())


def live_group(pid):
    try:
        os.killpg(pid, 0)
        return True
    except ProcessLookupError:
        return False


class Chain:
    def __init__(self):
        self.state = {"status": "STARTING", "started_at_utc": now(), "pid": os.getpid(), "jobs": {}, "failures": [],
                      "protocol_sha256": sha(PROTOCOL)}
        self.children = {}

    def publish(self):
        save(STATUS, self.state)

    def spawn(self, key, command, env_extra):
        env = dict(os.environ, PYTHONHASHSEED="3407", CUDA_DEVICE_ORDER="PCI_BUS_ID", **env_extra)
        log = (LOGS / (key.replace(":", "_") + ".log")).open("x")
        p = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        self.children[key] = p
        self.state["jobs"][key] = {"status": "RUNNING", "pid": p.pid, "command": command, "started_at_utc": now(),
                                   "gpus": env_extra.get("CUDA_VISIBLE_DEVICES")}
        self.publish()
        return p

    def finished(self, key):
        """None while running; else exit code, after the whole process group is gone."""
        p = self.children[key]
        code = p.poll()
        if code is None or live_group(p.pid):
            return None
        job = self.state["jobs"][key]
        job.update(status="EXIT_0" if code == 0 else "FAILED_PRESERVED", exit_code=code, ended_at_utc=now())
        if code != 0:
            self.state["failures"].append(key)
        self.publish()
        return code

    def start_eval(self, job, gpu):
        run, mode, step, intervention, repeat = job
        key = "eval:" + ":".join(map(str, job))
        command = [PY, "tools/c1sf/evaluate.py", "--run", run, "--mode", mode, "--step", str(step),
                   "--intervention", intervention, "--repeat", str(repeat)]
        self.spawn(key, command, {"CUDA_VISIBLE_DEVICES": str(gpu)})
        return key

    def start_training(self, arm, phase, pair):
        env = {"CUDA_VISIBLE_DEVICES": ",".join(map(str, pair)), "MASTER_PORT": str(29600 + pair[0]),
               "PL_INTER_BATCH_PARALLELISM": "0"}
        command = [PY, "tools/c1sf/train.py", "--arm", arm, "--phase", phase, "--config", str(config_path(arm, phase))]
        self.spawn(f"{phase}:{arm}", command, env)

    def accept(self, arm, phase):
        receipt = verify_run(arm, phase)
        receipt.update(protocol_sha256=sha(PROTOCOL), wrapper_sha256=sha(ROOT / "tools/c1sf/train.py"),
                       verified_at_utc=now(), engineering_only_no_accuracy_result=True)
        save(OUT / "receipts" / f"{phase}_{arm}.json", receipt, exclusive=True)

    def schedule(self, phase, arms, evaluate):
        """Train `arms` one pair each (in order, as pairs free up); evaluate finished formal arms on spare GPUs."""
        pending, training, evals, queue = list(arms), {}, {}, []
        while pending or training or queue or evals:
            used = {g for pair in training.values() for g in pair}
            for pair in PAIRS:
                eval_on_pair = any(g in pair for g in evals.values())
                if pending and not used.intersection(pair) and not eval_on_pair:
                    arm = pending.pop(0)
                    self.start_training(arm, phase, pair)
                    training[arm] = pair
                    used |= set(pair)
            for arm in list(training):
                code = self.finished(f"{phase}:{arm}")
                if code is None:
                    continue
                del training[arm]
                if code == 0:
                    try:
                        self.accept(arm, phase)
                        if evaluate:
                            queue.extend((arm, *job) for job in EVAL_PLAN)
                    except Exception as exc:  # preserved; the arm stops here
                        self.state["failures"].append(f"{phase}_verify:{arm}: {exc}")
                        self.publish()
            used = {g for pair in training.values() for g in pair}
            # a whole free pair stays reserved for a pending training
            reserved = set()
            if pending:
                for pair in PAIRS:
                    if not used.intersection(pair):
                        reserved = set(pair)
                        break
            for gpu in PHYSICAL_GPUS:
                while queue and gpu not in used and gpu not in reserved and \
                        sum(g == gpu for g in evals.values()) < EVAL_SLOTS_PER_GPU:
                    key = self.start_eval(queue.pop(0), gpu)
                    evals[key] = gpu
            for key in list(evals):
                if self.finished(key) is not None:
                    del evals[key]
            time.sleep(10)

    def run(self):
        require(not STATUS.exists(), "Fresh-only driver; preserve the previous status")
        LOGS.mkdir(parents=True, exist_ok=False)
        verify_protocol()
        require(read(OUT / "pairing_eval.json")["status"] == "PASS" and read(OUT / "step1.json")["status"] == "COMPLETE",
                "Step 1 / evaluation pairing must be complete before the amended chain")
        busy = [g for g in PHYSICAL_GPUS if gpu_busy(g)]
        require(not busy, f"GPUs {busy} are not idle; C1SF does not share 2-7")
        try:
            self.state["status"] = "ENGINEERING_PREFLIGHT"
            self.publish()
            self.schedule("preflight", tuple(ARMS), evaluate=False)
            replay = OUT / "receipts/preflight_c1replay.json"
            require(replay.is_file() and read(replay)["c1_code_replay_exact"] is True,
                    "C1 code replay is not exact; formal pairing premise fails (pre-registered stop)")
            ready = [a for a in FORMAL_ARMS if (OUT / "receipts" / f"preflight_{a}.json").is_file()]
            self.state["status"] = "FORMAL_TRAINING_AND_EVALUATION"
            self.publish()
            self.schedule("formal", ready, evaluate=True)
            self.state["status"] = "MAIN_ROWS_SERIAL"
            self.publish()
            for run in ("c1", *FORMAL_ARMS):
                if run != "c1" and not (OUT / "receipts" / f"formal_{run}.json").is_file():
                    continue
                key = self.start_eval((run, "row", 6000, "none", 1), 4)
                while self.finished(key) is None:
                    time.sleep(10)
            subprocess.run([PY, "tools/c1sf/decide.py", "all"], cwd=ROOT, check=True)
            self.state.update(status="COMPLETE" if not self.state["failures"] else "COMPLETE_WITH_PRESERVED_FAILURES",
                              ended_at_utc=now())
        except BaseException as exc:
            self.cleanup()
            self.state.update(status="STOPPED_PRESERVED", error=f"{type(exc).__name__}: {exc}", ended_at_utc=now())
            raise
        finally:
            self.publish()

    def cleanup(self):
        for p in self.children.values():
            if live_group(p.pid):
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 60
        while any(live_group(p.pid) for p in self.children.values()) and time.monotonic() < deadline:
            time.sleep(1)
        for p in self.children.values():
            if live_group(p.pid):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def main():
    def interrupted(number, frame):
        raise InterruptedError("C1SF chain interrupted by signal " + str(number))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    Chain().run()


if __name__ == "__main__":
    main()
