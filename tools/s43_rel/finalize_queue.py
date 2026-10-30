"""Wait for actual final workloads, then run fixed audits/tests/report only."""
import json
import os
from pathlib import Path
import subprocess
import time

from .protocol import ROOT, OUT, write_json
from .orchestrate import PYTHON


def main():
    while not (OUT / "postprocess_complete.json").exists():
        time.sleep(5)
    directory = OUT / "finalization"
    directory.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OMP_NUM_THREADS="2")
    test_files = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "tests").glob("test_s43_rel_*.py"))
    tasks = [
        ("control_prefix_final", [PYTHON, "-m", "tools.s43_rel.control_prefix_audit",
          "--checkpoint-dir", str(OUT / "train/full_r1"), "--require-final",
          "--out", str(OUT / "audits/control_prefix_final.json")]),
        ("unit_tests_final", [PYTHON, "-m", "pytest", "-q", *test_files]),
        ("artifact_report", [PYTHON, "-m", "tools.s43_rel.report"]),
    ]
    records = []
    for name, command in tasks:
        log = directory / (name + ".log")
        if log.exists():
            raise FileExistsError(log)
        started = time.time()
        with log.open("x") as output:
            code = subprocess.run(command, cwd=ROOT, env=env,
                                  stdout=output, stderr=subprocess.STDOUT).returncode
        record = dict(name=name, command=command, code=code, seconds=time.time() - started,
                      log=str(log), CUDA_VISIBLE_DEVICES="")
        write_json(directory / (name + ".execution.json"), record)
        print(json.dumps(record), flush=True)
        records.append(record)
        if code:
            raise RuntimeError("Final audit failed; preserve log and diagnose: " + name)
    write_json(OUT / "finalization_complete.json", dict(execution_complete=True, tasks=records,
        scope="Required workloads/audits/tests/report completed. Scientific outcome is in report/results.json; no success implied."))


if __name__ == "__main__":
    main()
