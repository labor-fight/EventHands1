"""Bounded recording-parallel launcher for the fixed XJ SELECT evaluation."""
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


def run(protocol_path, freeze_path, output, *, dry_run=False):
    from .evaluate import verify_evaluation_identity
    protocol, frozen, _, _, _ = verify_evaluation_identity(protocol_path, freeze_path)
    root = Path(__file__).resolve().parents[2]
    output = Path(output).resolve()
    execution = frozen['execution']
    concurrency = int(execution['recording_workers_per_gpu'])
    if not 1 <= concurrency <= 4:
        raise ValueError('One to four independent recording workers per GPU')
    inventory = json.loads((Path(protocol['data_root']) / 'splits_semkine.json').read_text())
    queues = {person: sorted(name for name in inventory['train']['trials']
                            if name.split('_')[0] == person) for person in protocol['subjects']}
    if sum(map(len, queues.values())) != protocol['expected_recordings']:
        raise RuntimeError('Complete SELECT inventory required')
    plan = dict(schema='xj_select_recording_queue_v1',
                eval_protocol_sha256=sha(protocol_path), eval_sourcefreeze_sha256=sha(freeze_path),
                queues=queues, workers_per_gpu=concurrency,
                device_by_person=execution['device_by_person'],
                scope='Independent per-recording four-arm states; same fixed checkpoint, seed and original recording inventory')
    if dry_run:
        return dict(status='DRY_RUN', **plan)
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'queue_manifest.json').exists():
        raise FileExistsError('An evaluation queue already exists')
    dump(output / 'queue_manifest.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), **plan))
    logs = output / 'logs'
    logs.mkdir(exist_ok=True)
    active, completed, handles = {}, [], []
    started = time.monotonic()
    # Two physical GPUs in parallel: total GPU-hours is bounded by GPU count
    # times wall clock. Child timeouts are an additional local bound.
    wall_budget = float(protocol['max_gpu_hours']) * 3600 / len(protocol['devices'])
    try:
        while active or any(queues.values()):
            if time.monotonic() - started > wall_budget:
                raise TimeoutError('Frozen total GPU wall-time budget exhausted')
            for person, names in queues.items():
                current = sum(row['person'] == person for row in active.values())
                while names and current < concurrency:
                    name = names.pop(0)
                    destination = output / 'parts' / name
                    command = [sys.executable, '-m', 'tools.xvm_joint.evaluate', 'person',
                               '--eval-protocol', str(Path(protocol_path).resolve()),
                               '--eval-source-freeze', str(Path(freeze_path).resolve()),
                               '--person', person, '--recording', name,
                               '--device', 'cuda:0', '--out', str(destination)]
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=execution['device_by_person'][person]['uuid'],
                               OMP_NUM_THREADS=str(execution['cpu_threads']),
                               MKL_NUM_THREADS=str(execution['cpu_threads']),
                               XJ_GPU_MEMORY_LIMIT_GIB=str(execution['gpu_memory_cap_gib']))
                    handle = (logs / (name + '.log')).open('x')
                    handles.append(handle)
                    process = subprocess.Popen(command, cwd=root, env=env, stdout=handle,
                                               stderr=subprocess.STDOUT)
                    active[process.pid] = dict(process=process, person=person, recording=name,
                                               command=command, output=str(destination))
                    dump(logs / (name + '.launch.json'),
                         dict(pid=process.pid, person=person, recording=name, command=command,
                              device=execution['device_by_person'][person],
                              eval_sourcefreeze_sha256=sha(freeze_path)))
                    current += 1
            for pid, row in list(active.items()):
                code = row['process'].poll()
                if code is None:
                    continue
                if code:
                    raise RuntimeError(f"Evaluation worker {row['recording']} exited {code}; evidence preserved")
                summary = Path(row['output']) / 'summary.json'
                if not summary.exists():
                    raise RuntimeError('Successful worker lacks complete summary')
                completed.append(str(summary))
                del active[pid]
            dump(output / 'queue_progress.json', dict(status='RUNNING', completed_recordings=len(completed),
                 total_recordings=protocol['expected_recordings'], elapsed_s=time.monotonic()-started,
                 active=[{k: v for k, v in row.items() if k != 'process'} for row in active.values()]))
            time.sleep(1)
        command = [sys.executable, '-m', 'tools.xvm_joint.evaluate', 'combine',
                   '--eval-protocol', str(Path(protocol_path).resolve()),
                   '--eval-source-freeze', str(Path(freeze_path).resolve()),
                   '--out', str(output / 'results.json'), '--person-summaries', *sorted(completed)]
        with (logs / 'combine.log').open('x') as handle:
            subprocess.run(command, cwd=root, check=True, stdout=handle, stderr=subprocess.STDOUT)
        result = dict(status='COMPLETE', completed_recordings=len(completed),
                      elapsed_s=time.monotonic()-started, results_sha256=sha(output/'results.json'))
        dump(output / 'queue_complete.json', result)
        return result
    except BaseException as exc:
        for row in active.values():
            if row['process'].poll() is None:
                row['process'].terminate()
        reap_deadline = time.monotonic() + 10
        for row in active.values():
            try:
                row['process'].wait(timeout=max(0., reap_deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                row['process'].kill()
                row['process'].wait()
        dump(output / 'queue_failure.json', dict(status='FAILED_PRESERVED', error=repr(exc),
             completed_summaries=completed, elapsed_s=time.monotonic()-started))
        raise
    finally:
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-protocol', required=True)
    parser.add_argument('--eval-source-freeze', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args.eval_protocol, args.eval_source_freeze, args.out, dry_run=args.dry_run),
                     ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
