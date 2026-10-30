#!/usr/bin/env python3
"""Run fixed-cohort readiness once each declared I-only endpoint is complete.

This read-only evaluator never starts training. It waits for the six declared
train_finish files and retains any failed evaluator output for inspection.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/xkf/fusion_20261015_r3/direct_finetune/incremental_finetune'


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-wait-seconds', type=int, default=7200)
    args = parser.parse_args()
    output = BASE / 'readiness_round'
    output.mkdir(parents=True, exist_ok=False)
    tasks = [(arm, seed) for arm in ('raw', 'aug') for seed in (3407, 3408, 3409)]
    expected = {f'{arm}_s{seed}' for arm, seed in tasks}
    started, launched, running, finished = time.monotonic(), set(), {}, {}
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def evaluate(arm, seed):
        key = f'{arm}_s{seed}'
        directory = BASE / key
        checkpoint = directory / f'xkf_ionly_{key}-step=2600.ckpt'
        command = [sys.executable, str(ROOT / 'tools/xkf_fusion/readiness_incremental.py'),
                   '--config', str(directory / 'config_resolved.yaml'),
                   '--checkpoint', str(checkpoint), '--output-dir', str(output / key)]
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
        before = time.monotonic()
        with (output / (key + '.log')).open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record = dict(returncode=result.returncode, elapsed_seconds=time.monotonic() - before,
                      command=command, log=str(output / (key + '.log')))
        if result.returncode == 0:
            evidence = json.loads((output / key / 'result.json').read_text())
            record.update(identity=evidence['identity'], readiness=evidence['readiness'])
        return record

    with ThreadPoolExecutor(max_workers=3) as pool:
        while len(finished) < len(tasks):
            for arm, seed in tasks:
                key = f'{arm}_s{seed}'
                marker = BASE / key / 'train_finish.json'
                if key in launched or not marker.exists():
                    continue
                finish = json.loads(marker.read_text())
                if finish.get('status') != 'DEBUG200_COMPLETE' or finish.get('global_step') != 2600:
                    raise ValueError('unexpected endpoint marker: ' + str(marker))
                launched.add(key)
                running[key] = pool.submit(evaluate, arm, seed)
                print(json.dumps(dict(event='started_readiness', run=key)), flush=True)
            for key in list(running):
                future = running[key]
                if future.done():
                    finished[key] = future.result()
                    del running[key]
                    print(json.dumps(dict(event='completed_readiness', run=key,
                                          returncode=finished[key]['returncode'])), flush=True)
            write(output / 'progress.json', dict(launched=sorted(launched), running=sorted(running),
                  completed=finished, pending=sorted(expected - launched), source_sha256=source,
                  elapsed_seconds=time.monotonic() - started, automatic_full_training=False))
            if len(finished) == len(tasks):
                break
            if time.monotonic() - started > args.max_wait_seconds:
                raise TimeoutError('remaining endpoints/evaluations not complete; existing evidence retained')
            time.sleep(5)
    write(output / 'result.json', dict(status='COMPLETE' if all(v['returncode'] == 0 for v in finished.values()) else 'EVALUATION_ERROR',
          results=finished, source_sha256=source, automatic_full_training=False,
          all_original_readiness_pass=all(v.get('readiness', {}).get('status') == 'PASS' for v in finished.values())))


if __name__ == '__main__':
    main()
