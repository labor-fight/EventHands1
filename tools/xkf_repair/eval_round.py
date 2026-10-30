#!/usr/bin/env python3
"""Orchestrate the frozen R2 evaluations, retaining every subprocess outcome."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/xkf/repair_20261015_r2'
PYTHON = '/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arms', nargs='+', choices=['control', 'repair'], required=True)
    parser.add_argument('--kinds', nargs='+', choices=['debug', 'rollout', 'direct', 'row'], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--status-name', required=True)
    args = parser.parse_args()
    jobs = [(arm, seed) for arm in args.arms for seed in (3407, 3408, 3409)]
    if len(set(jobs)) != len(jobs) or len(set(args.kinds)) != len(args.kinds):
        parser.error('duplicate arms or kinds')
    if args.device == 'cpu' and ('row' in args.kinds or args.gpus):
        parser.error('row timing requires a GPU; CPU mode must not allocate GPUs')
    if args.device == 'cuda' and (not args.gpus or len(args.gpus) != len(jobs)
                                  or len(set(args.gpus)) != len(jobs)
                                  or any(g not in range(2, 8) for g in args.gpus)):
        parser.error('one distinct authorized GPU2-7 is required per job')
    manifest = json.loads((OUT / 'frozen_manifest.json').read_text())
    for name, expected in manifest['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'frozen source mismatch: {name}')
    if hashlib.sha256((OUT / 'protocol.json').read_bytes()).hexdigest() != manifest['protocol_sha256']:
        raise RuntimeError('frozen protocol mismatch')
    status_path = OUT / args.status_name
    if status_path.exists():
        raise FileExistsError(status_path)
    evaluations = OUT / 'evaluations'
    evaluations.mkdir(exist_ok=True)
    declared = {}
    for index, (arm, seed) in enumerate(jobs):
        checkpoint = OUT / f'{arm}_s{seed}/xkf_{arm}_s{seed}-step=2200.ckpt'
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        stages = []
        for kind in args.kinds:
            prefix = evaluations / f'{arm}_s{seed}_step2200'
            if kind == 'debug':
                output = Path(str(prefix) + '.json')
            elif kind == 'row':
                output = ROOT / f'outputs/semkine/xkf_r2_{arm}_s{seed}_step2200_main_row.json'
            else:
                output = Path(str(prefix) + f'_{kind}')
            if output.exists():
                raise FileExistsError(output)
            command = [PYTHON, '-u', 'tools/xkf_repair/evaluate.py', '--kind', kind,
                       '--config', f'configs/xkf_repair/{arm}_s{seed}.yaml',
                       '--checkpoint', str(checkpoint), '--device',
                       'cuda:0' if args.device == 'cuda' else 'cpu',
                       '--output-dir' if kind == 'direct' else '--output', str(output)]
            if kind == 'debug':
                command += ['--gradients', '--compare', str(evaluations / f'{arm}_s{seed}_initial.json')]
            stages.append({'kind': kind, 'command': command, 'output': str(output),
                           'log': str(prefix) + f'_{kind}.log', 'status': 'PENDING'})
        declared[f'{arm}_s{seed}'] = {'gpu': args.gpus[index] if args.device == 'cuda' else None,
                                     'stages': stages}
    record = {'kind': 'R2_EVALUATION_ORCHESTRATION', 'formal': False,
              'orchestrator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'protocol_sha256': manifest['protocol_sha256'], 'device': args.device,
              'started_unix': time.time(), 'jobs': declared}
    lock = threading.Lock()

    def save():
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        temporary.replace(status_path)

    save()

    def run(name, job):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job['gpu']) if args.device == 'cuda' else '',
                   OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
        for stage in job['stages']:
            start = time.time()
            with open(stage['log'], 'w') as log:
                with subprocess.Popen(stage['command'], cwd=ROOT, env=env,
                                      stdout=log, stderr=subprocess.STDOUT) as process:
                    with lock:
                        stage.update(status='RUNNING', pid=process.pid, started_unix=start)
                        save()
                    code = process.wait()
            with lock:
                stage.update(status='COMPLETE' if code == 0 else 'FAILED', returncode=code,
                             elapsed_seconds=time.time() - start)
                save()
            print(json.dumps({'job': name, 'kind': stage['kind'], 'returncode': code}), flush=True)
            if code != 0:
                return False
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(run, name, job) for name, job in declared.items()]
        outcomes = [future.result() for future in futures]
    record.update(status='COMPLETE' if all(outcomes) else 'FAILED', finished_unix=time.time())
    save()
    if not all(outcomes):
        raise RuntimeError(f'evaluation failed; retained logs and status: {status_path}')


if __name__ == '__main__':
    main()
