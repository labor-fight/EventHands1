"""Fixed additional deployment validation, with no GT shape input or tuning."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from .protocol import ROOT, OUT, write_json
from .orchestrate import PYTHON
from .gpu_memory_observer import MemoryObserver

ARMS = ("canonical", "s43", "enhanced", "no_order", "full")


def run(command, name, gpu):
    folder = OUT / "mean_shape"
    log = folder / "logs" / (name + ".log")
    log.parent.mkdir(parents=True, exist_ok=True)
    if log.exists():
        raise FileExistsError(log)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMP_NUM_THREADS="2")
    started = time.time()
    observer = MemoryObserver(gpu) if name.startswith("resource_") else None
    with log.open("x") as output:
        process = subprocess.Popen(command, cwd=ROOT, env=env,
                                   stdout=output, stderr=subprocess.STDOUT)
        while process.poll() is None:
            if observer:
                observer.sample()
            time.sleep(.05)
        code = process.returncode
    record = dict(name=name, command=command, code=code, seconds=time.time()-started,
                  physical_gpu=gpu, log=str(log))
    if observer:
        observer.sample()
        record["NVML_device_memory_observation"] = observer.close()
    write_json(folder / "executions" / (name + ".json"), record)
    print(json.dumps(record), flush=True)
    if code:
        raise RuntimeError("Deployment validation failed; retain receipt and diagnose: " + name)
    return record


def main():
    required = (OUT / "postprocess_complete.json", OUT / "finalization_complete.json")
    while not all(path.exists() for path in required):
        time.sleep(5)
    extension = OUT / "mean_shape_deployment_extension.json"
    if not extension.exists():
        raise RuntimeError("Fixed deployment extension must predate formal scores")
    paths = sorted((ROOT / "tools/s43_rel").glob("mean_shape*.py"))
    paths.append(ROOT / "tools/s43_rel/gpu_memory_observer.py")
    write_json(OUT / "mean_shape/formal_sourcefreeze.json", {
        "time":time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "extension_sha256":hashlib.sha256(extension.read_bytes()).hexdigest(),
        "source_sha256":{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "policy":"No further training/model selection; fixed final6000, beta0, source/predictions precede GT scoring."})
    records = []
    # All earlier training/replays/audits have ended. Resources run alone on4.
    for arm in ARMS:
        command = [PYTHON, "-m", "tools.s43_rel.mean_shape_benchmark", "--arm", arm,
                   "--out", str(OUT / "mean_shape/resources" / (arm + "_h50.json")),
                   "--device", "cuda:0"]
        records.append(run(command, "resource_" + arm, 4))
    cases = [(arm, recording) for arm in ARMS for recording in ("zgz_local", "zgz_global")]

    def lane(index):
        result = []
        for arm, recording in cases[index::4]:
            config = ROOT / "configs/s43_rel/r1" / (("s43" if arm == "canonical" else arm) + ".yaml")
            command = [PYTHON, "-m", "tools.s43_rel.mean_shape_replay", "--config", str(config),
                       "--recording", recording, "--out", str(OUT / "mean_shape/replays" / (arm + "_" + recording + "_h50")),
                       "--device", "cuda:0", "--threads", "2"]
            if arm != "canonical":
                command += ["--checkpoint", str(OUT / "train/full_r1" / arm / "last.ckpt")]
            result.append(run(command, "replay_" + arm + "_" + recording, 4 + index))
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lane, range(4)):
            records.extend(result)
    records.append(run([PYTHON, "-m", "tools.s43_rel.mean_shape_report"], "mean_shape_report", ""))
    write_json(OUT / "mean_shape_deployment_complete.json", {
        "execution_complete": True, "resource_cases":5, "replay_cases":10, "tasks":records,
        "claim":"Separate mean-shape deployment test complete. Accuracy/unique value remain empirical outcomes."})


if __name__ == "__main__":
    main()
