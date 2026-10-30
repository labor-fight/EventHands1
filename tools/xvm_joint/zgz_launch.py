"""Run all fixed XJ zgz workers, then serial packet costs on each GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .train import dump, sha

ROOT = Path(__file__).resolve().parents[2]
GPU_BY_MODE = {'w50': 'GPU-02fdb384-799a-0b64-edec-c8c5bb79879d',
               's39': 'GPU-d1abbac0-1c62-62cd-4803-15a5a172398e'}


def run(protocol_path, freeze_path, output):
    from .zgz_eval import verify_identity
    protocol = verify_identity(protocol_path, freeze_path)[0]
    output = Path(output).resolve()
    if not (output/'engineering_gpu.json').exists():
        raise RuntimeError('Final entry point requires the real zero-GT engineering receipt')
    engineering = json.loads((output/'engineering_gpu.json').read_text())
    if engineering['status'] != 'PASS' or engineering['eval_sourcefreeze_sha256'] != sha(freeze_path):
        raise RuntimeError('Real engineering must pass this execution freeze')
    if (output/'launch.json').exists():
        raise FileExistsError('No silent queue rerun')
    logs = output/'logs'
    logs.mkdir(exist_ok=True)
    dump(output/'launch.json', dict(pid=os.getpid(), created_utc=datetime.now(timezone.utc).isoformat(),
         protocol_sha256=sha(protocol_path), sourcefreeze_sha256=sha(freeze_path),
         GPU_by_mode=GPU_BY_MODE, stage='accuracy then per-GPU serial costs then original table tools'))
    children, handles = [], []
    started = time.monotonic()

    def start(command, mode, log_name):
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=GPU_BY_MODE[mode], OMP_NUM_THREADS=str(protocol['cpu_threads']),
                   MKL_NUM_THREADS=str(protocol['cpu_threads']),
                   XJ_GPU_MEMORY_LIMIT_GIB=str(protocol['gpu_memory_cap_gib_per_process']))
        handle = (logs/log_name).open('x')
        handles.append(handle)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
        children.append(process)
        return process

    def wait_stage(processes, name):
        while True:
            codes = [process.poll() for process in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError(f'{name} worker failure, return codes {codes}; evidence retained')
            dump(output/'progress.json', dict(status='RUNNING', stage=name,
                 finished_workers=sum(code == 0 for code in codes), total_workers=len(codes),
                 elapsed_s=time.monotonic()-started))
            if all(code == 0 for code in codes):
                return
            if (time.monotonic()-started)*len(protocol['devices']) > protocol['budget_gpu_hours']*3600:
                raise TimeoutError('Frozen GPU wall-time budget exhausted')
            time.sleep(1)

    try:
        workers = []
        for mode in protocol['modes']:
            for arm in protocol['arms']:
                for sequence in protocol['sequences']:
                    command = [sys.executable, '-m', 'tools.xvm_joint.zgz_eval',
                               '--protocol', str(protocol_path), '--source-freeze', str(freeze_path),
                               '--mode', mode, '--arm', arm, '--sequence', sequence,
                               '--device', 'cuda:0', '--out', str(output/'accuracy'/mode/arm/sequence)]
                    workers.append(start(command, mode, mode+'_'+arm+'_'+sequence+'.log'))
        wait_stage(workers, 'accuracy')
        # All prediction workers are finished. Each card now has exactly one
        # resource process, which benchmarks primary/control serially.
        workers = []
        for mode in protocol['modes']:
            command = [sys.executable, '-m', 'tools.xvm_joint.zgz_cost',
                       '--protocol', str(protocol_path), '--source-freeze', str(freeze_path),
                       '--mode', mode, '--device', 'cuda:0', '--out', str(output/'costs'/(mode+'.json'))]
            workers.append(start(command, mode, mode+'_costs.log'))
        wait_stage(workers, 'resources')
        command = [sys.executable, '-m', 'tools.xvm_joint.zgz_report',
                   '--protocol', str(protocol_path), '--source-freeze', str(freeze_path), '--out', str(output)]
        with (logs/'report.log').open('x') as handle:
            subprocess.run(command, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
        dump(output/'complete.json', dict(status='COMPLETE', elapsed_s=time.monotonic()-started,
             report_sha256=sha(output/'report.json'), table_sha256=sha(output/'main_table.md')))
    except BaseException as exc:
        for process in children:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic()+10
        for process in children:
            try:
                process.wait(timeout=max(0., deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        dump(output/'failure.json', dict(status='FAILED_PRESERVED', error=repr(exc),
             elapsed_s=time.monotonic()-started))
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True)
    parser.add_argument('--source-freeze', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    run(Path(args.protocol).resolve(), Path(args.source_freeze).resolve(), args.out)


if __name__ == '__main__':
    main()
