#!/usr/bin/env python3
"""Finish mandatory diagnostics and the evidence record after the frozen study queue.

This independent supervisor does not alter frozen candidate implementations or budgets.
It waits for exact-last evaluation, runs the finite-value-reporting diagnostic adapter,
then applies the already registered decision rules. A failed queue remains a failure.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/xb"
PYTHON = "/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"
LOCK = threading.Lock()
RECORD = "XYZ-RESIDUAL-ABCD-EXECUTION-20261011"
NUMERIC_ANNOTATIONS = {"numeric_valid", "numeric_status", "first_nonfinite", "invalid_numeric_count",
                       "invalid_numeric_paths", "invalid_known_count", "invalid_known_paths"}


def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def event(**values):
    with LOCK:
        row = {"time": now(), **values}
        with (OUT / "finish_events.jsonl").open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)


def diagnostic(phase, seed, gpu):
    prefix = "xb_p" if phase == "pilot" else "xb_"
    output = OUT / f"recovery_{phase}_s{seed}.json"
    command = [PYTHON, "tools/xb/safe_diagnostics.py", "--reference", "outputs/semkine/xa_c0_s3407",
               "--output", str(output), "--device", "cuda", "--steps", "10", "--clips-per-seq", "24",
               "--trace-cache", "outputs/xb/c0_val_reference.pt"]
    for arm in "abcd":
        command += ["--arm", f"{arm.upper()}=outputs/semkine/{prefix}{arm}_s{seed}"]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4",
           "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"}
    if gpu not in (4, 5, 6, 7):
        raise RuntimeError(f"unauthorized GPU {gpu}")
    event(action="diagnostics_start", phase=phase, seed=seed, gpu=gpu, command=command)
    log = OUT / f"logs/recovery_{phase}_s{seed}.log"
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=REPO, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode or not output.is_file():
        raise RuntimeError(f"{phase}/{seed} diagnostics failed: see {log}")
    payload = json.loads(output.read_text())
    if set(payload.get("arms", {})) - NUMERIC_ANNOTATIONS != {"C0", "A", "B", "C", "D"}:
        raise RuntimeError(f"incomplete diagnostic arm set in {output}")
    for arm in ("C0", "A", "B", "C", "D"):
        artifact = payload["arms"][arm]
        if len(artifact["rows"]) != 192:
            raise RuntimeError(f"{output}/{arm}: expected all 48 clips x four conditions")
    event(action="diagnostics_finish", phase=phase, seed=seed, gpu=gpu, sha256=sha(output))


def append_record():
    path = REPO / "docs/EXPERIMENT_RECORDS.md"
    existing = path.read_text()
    marker = f"<!-- EXPERIMENT_RECORD:{RECORD}:BEGIN -->"
    if marker in existing:
        raise RuntimeError("execution record already exists; refuse to duplicate or overwrite")
    verdict = (OUT / "VERDICT.md").read_text()
    pilot = (OUT / "TABLE_pilot.md").read_text()
    body = (f"\n\n---\n\n{marker}\n<a id=\"xyz-residual-abcd-execution-20261011\"></a>\n\n"
            f"### XYZ-RESIDUAL-ABCD 执行记录（完成时间 {now()}）\n\n"
            "预注册见 [固定协议](#xyz-residual-abcd-20261011)；训练、末步评价及诊断全部完成后生成本节。"
            "完整训练使用物理 GPU4–7，未选择中间检查点。\n\n"
            "轻量探针末步结果：\n\n" + pilot + "\n"
            "八个轻量探针的训练损失有限，但完整自身递推均出现非有限数值，"
            "不能排序或作为闭环增益。CPU 审计确认先前态指数放大再 FK 溢出，"
            "原 C0 的完整2590步有限，公共骨干保持原值；详见 `outputs/xb/pilot_nonfinite_audit.json`。\n\n"
            "数值故障处理仅由独立诊断适配器将非有限值显式存为 null/FAIL，保留全部帧、片段和种子，"
            "未修改冻结候选、训练预算或判定门。仅r=1交换属于补充的分布外灵敏度诊断，"
            "不追加正式采纳门；实现和时间戳见 `outputs/xb/diagnostic_addendum.json`。\n\n"
            "主评分完成后，首轮诊断因 CUDA 浮点归约的非确定性触发公共输入严格相等断言而中断。"
            "同一 B 重复运行也复现该差异；仅公共输入审计启用确定性算法后，原严格检查保留，"
            "补充 r=1 公共帧审计使用相同作用域，退出审计即恢复原标志；cuBLAS 工作区配置在进程启动时设置。"
            "原失败日志、适配器历史版本及运行时偏离时间戳保存在 "
            "`outputs/xb/runtime_audit_fix.json` 指向的归档；训练、主评分、恢复评分和所有判定阈值未修改。\n\n"
            + verdict + "\n"
            "机器可读证据位于 `outputs/xb/verdict.json`、`gates.json`、全部 `recovery_*.json`；"
            "来源与 SHA 见 `frozen_manifest.json`、`finished_manifest.json`。\n\n"
            f"<!-- EXPERIMENT_RECORD:{RECORD}:END -->\n")
    path.write_text(existing + body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    state = {"started_at": now(), "status": "waiting_for_frozen_training_and_rows", "finished": []}
    state_path = OUT / "finish_status.json"
    write_json(state_path, state)
    try:
        while True:
            queue = json.loads((OUT / "complete_status.json").read_text())
            if queue["status"] == "training_and_primary_evaluation_complete":
                break
            if queue["status"] == "failed":
                raise RuntimeError(f"frozen training/evaluation queue failed: {queue.get('error')}")
            if not args.wait:
                raise RuntimeError("primary study is still running; use --wait")
            time.sleep(30)
        state.update(status="running_diagnostics", updated_at=now())
        write_json(state_path, state)
        sys.path.insert(0, str(REPO / "tools/xb"))
        import run_study
        run_study.frozen()
        diagnostic("pilot", 3407, 4)  # Creates the one shared fixed reference trace.
        state["finished"].append("pilot_s3407")
        write_json(state_path, state)
        remaining = [("pilot", 3408), ("full", 3407), ("full", 3408), ("full", 3409)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(diagnostic, phase, seed, gpu)
                       for (phase, seed), gpu in zip(remaining, (4, 5, 6, 7))]
            for future in futures:
                future.result()
        state["finished"] += [f"{phase}_s{seed}" for phase, seed in remaining]
        state.update(status="applying_registered_gates", updated_at=now())
        write_json(state_path, state)
        subprocess.run([PYTHON, "tools/xb/finalize.py"], cwd=REPO, check=True)
        verdict = json.loads((OUT / "verdict.json").read_text())
        if verdict.get("pending"):
            raise RuntimeError(f"decision has unresolved mandatory evidence: {verdict['pending']}")
        artifacts = [OUT / "TABLE_pilot.md", OUT / "TABLE_full.md", OUT / "gates.json", OUT / "verdict.json",
                     OUT / "VERDICT.md", OUT / "diagnostic_addendum.json", OUT / "pilot_nonfinite_audit.json",
                     OUT / "c0_val_reference.pt"]
        artifacts += sorted(OUT.glob("recovery_*.json"))
        artifacts += [OUT / "frozen_manifest.json", OUT / "control_capacity.json", OUT / "preflight.json"]
        artifacts += [OUT / "runtime_audit_fix.json", OUT / "audit_nondeterminism_probe.json",
                      OUT / "deterministic_audit_full_packet_check.json"]
        artifacts += sorted(path for archive in OUT.glob("runtime_failure_*")
                            for path in archive.rglob("*") if path.is_file())
        for job in run_study.jobs():
            checkpoint = run_study.final_checkpoint(job)
            if checkpoint is None:
                raise RuntimeError(f"missing registered endpoint for final provenance: {job['run']}")
            artifacts.append(checkpoint)
            directory = checkpoint.parent
            artifacts += [directory / "training_metadata.json", directory / "config_resolved.yaml",
                          directory / ("pair_stream.json" if job["phase"] == "pilot" else "pair_stream_rank0.json")]
        artifacts += [REPO / "outputs/semkine/xa_c0_s3407" / name
                      for name in ("xa_c0_s3407-step=6000.ckpt", "config_resolved.yaml")]
        scripts = [REPO / f"tools/xb/{name}.py" for name in ("safe_diagnostics", "finalize", "finish_study")]
        # Prove every required result and weight exists before recording completion.
        files_sha256 = {str(p.relative_to(REPO)): sha(p) for p in artifacts + scripts}
        append_record()
        record = REPO / "docs/EXPERIMENT_RECORDS.md"
        files_sha256[str(record.relative_to(REPO))] = sha(record)
        write_json(OUT / "finished_manifest.json", {"completed_at": now(), "decision": verdict["decision"],
                   "files_sha256": files_sha256})
        state.update(status="complete", decision=verdict["decision"], completed_at=now())
        write_json(state_path, state)
        event(action="study_complete", decision=verdict["decision"])
    except Exception as error:
        state.update(status="failed", error=str(error), failed_at=now())
        write_json(state_path, state)
        event(action="finish_error", error=str(error))
        raise


if __name__ == "__main__":
    main()
