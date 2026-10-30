#!/usr/bin/env python3
"""Run one frozen three-seed debug arm and retain child identities/failures."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import hashlib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/xkf/repair_20261015_r2'
PYTHON = '/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['control', 'repair'])
    args = ap.parse_args()
    lock = (OUT / 'training.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = OUT / 'frozen_manifest.json'
    frozen = json.loads(manifest.read_text())
    protocol_path = OUT / 'protocol.json'
    if hashlib.sha256(protocol_path.read_bytes()).hexdigest() != frozen['protocol_sha256']:
        raise RuntimeError('protocol identity changed after freeze')
    protocol = json.loads(protocol_path.read_text())
    for identities in protocol['lineage'].values():
        for identity in identities.values():
            if hashlib.sha256(Path(identity['path']).read_bytes()).hexdigest() != identity['sha256']:
                raise RuntimeError(f'checkpoint lineage changed: {identity["path"]}')
    for name, expected in frozen['source_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'frozen mismatch {name}')
    for seed in (3407, 3408, 3409):
        if (OUT / f'{args.arm}_s{seed}').exists():
            raise FileExistsError(OUT / f'{args.arm}_s{seed}')
    processes, logs = {}, []
    status_path = OUT / f'{args.arm}_round_status.json'
    started = time.time()
    try:
        for index, seed in enumerate((3407, 3408, 3409)):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=f'{2 + index * 2},{3 + index * 2}',
                       MASTER_ADDR='127.0.0.1', MASTER_PORT=str(29841 + index),
                       OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       NCCL_P2P_DISABLE='1')
            checkpoint = ROOT / f'outputs/xkf/debug_20261015_r1/xkf_s{seed}/xkf_s{seed}-step=2000.ckpt'
            command = [PYTHON, '-u', 'tools/xkf_repair/train.py', '--config',
                       f'configs/xkf_repair/{args.arm}_s{seed}.yaml', '--preflight',
                       '--resume', str(checkpoint), '--frozen-manifest', str(manifest)]
            log = (OUT / f'{args.arm}_s{seed}.log').open('w')
            logs.append(log)
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            processes[seed] = proc
        while True:
            codes = {str(seed): process.poll() for seed, process in processes.items()}
            value = {'arm': args.arm, 'start_step': 2000, 'endpoint': 2200,
                     'elapsed_seconds': time.time() - started,
                     'pids': {str(seed): p.pid for seed, p in processes.items()}, 'returncodes': codes}
            value['status'] = ('FAILED' if any(c is not None and c != 0 for c in codes.values()) else
                               'COMPLETE' if all(c == 0 for c in codes.values()) else 'RUNNING')
            tmp = status_path.with_suffix('.tmp')
            tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(status_path)
            if value['status'] == 'FAILED':
                raise RuntimeError(f'child training failed: {codes}')
            if value['status'] == 'COMPLETE':
                print(json.dumps(value), flush=True)
                break
            time.sleep(10)
    finally:
        for process in processes.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
