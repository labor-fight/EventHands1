#!/usr/bin/env python3
"""Run a paired U1a phase on four idle two-GPU/NUMA-local partitions."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

R = Path(__file__).resolve().parents[2]
P = '/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python'
O = R/'outputs/u1a'
O.mkdir(parents=True, exist_ok=True)
ap = argparse.ArgumentParser()
ap.add_argument('phase', choices=['debug', '2k', '6k'])
args = ap.parse_args()
lock = open(O/'phase.lock', 'w')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
seeds = [3407] if args.phase == 'debug' else [3407, 3408]
specs = [(seed, mode) for seed in seeds for mode in ['shared', 'untied']]
q = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'], text=True)
mem = {int(l.split(',')[0]): int(l.split(',')[1]) for l in q.strip().splitlines()}
assert all(mem[g] < 1024 for g in range(2*len(specs))), 'GPU is occupied; do not share another job'
rows = []
processes = []
for i, (seed, mode) in enumerate(specs):
    run = f'u1a_{mode}_{args.phase}_s{seed}'
    cfg = R/f'configs/u1a/{run}.yaml'
    assert cfg.is_file()
    run_dir = R/'outputs/semkine'/run
    assert not (run_dir/'last.ckpt').exists(), f'completed run exists: {run}'
    log = O/f'{run}.log'
    assert not log.exists(), f'log exists; inspect before retry: {run}'
    gpu = [2*i, 2*i+1]
    physical = list(range(18*i, 18*i+18))
    cpus = physical + [c+72 for c in physical]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpu)),
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               PYTHONUNBUFFERED='1', NCCL_P2P_DISABLE='1')
    argv = ['/usr/bin/time', '-v', '-o', str(log.with_suffix('.time')),
            'taskset', '-c', ','.join(map(str,cpus)), P, '-u', 'semkine/train.py', '--config', str(cfg)]
    with log.open('w') as fo:
        p = subprocess.Popen(argv, cwd=R, env=env, stdout=fo, stderr=subprocess.STDOUT)
    rows.append({'run':run, 'pid':p.pid, 'gpu':gpu, 'cpus':cpus, 'log':str(log),
                 'start_unix':time.time(), 'status':'running', 'argv':argv,
                 'communication_env': {'NCCL_P2P_DISABLE':'1'}})
    processes.append(p)
    print(f'launched {run} pid={p.pid} gpu={gpu}', flush=True)
state = O/f'{args.phase}_state.json'
while True:
    for row, p in zip(rows, processes):
        rc = p.poll()
        if rc is not None and row['status'] == 'running':
            row.update(status='complete' if rc == 0 else 'failed', exit_code=rc, end_unix=time.time())
            print(f"{row['run']} {row['status']} rc={rc}", flush=True)
    temp = state.with_suffix('.tmp')
    temp.write_text(json.dumps({'phase':args.phase, 'runner_pid':os.getpid(),
                                'updated_unix':time.time(), 'jobs':rows}, indent=2))
    temp.replace(state)
    if all(p.poll() is not None for p in processes):
        break
    time.sleep(10)
raise SystemExit(0 if all(r['exit_code'] == 0 for r in rows) else 1)
