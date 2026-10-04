#!/usr/bin/env python3
"""Finish the already-running fixed 6k experiment; stop on any failed check."""
import fcntl
import json
import os
import signal
import subprocess
import time
from pathlib import Path

R = Path(__file__).resolve().parents[2]
O = R / 'outputs/u1a'
P = '/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python'
STATUS = O / '6k_completion_state.json'
EXPECTED = {f'u1a_{mode}_6k_s{seed}' for seed in (3407, 3408) for mode in ('shared', 'untied')}


def record(stage, **extra):
    obj = {'stage': stage, 'runner_pid': os.getpid(), 'updated_unix': time.time(), **extra}
    temp = STATUS.with_name(STATUS.name + f'.tmp.{os.getpid()}')
    temp.write_text(json.dumps(obj, indent=2))
    temp.replace(STATUS)
    print(stage, flush=True)


def command(script, *arguments):
    return [P, '-u', f'tools/u1a/{script}', *arguments]


def wait_for_training():
    while True:
        source = O / '6k_state.json'
        if source.exists():
            state = json.loads(source.read_text())
            jobs = state['jobs']
            if state.get('phase') != '6k' or len(jobs) != 4 or {j['run'] for j in jobs} != EXPECTED:
                raise RuntimeError('6k training state does not match the four preregistered runs')
            if any(j['status'] == 'failed' for j in jobs):
                raise RuntimeError('a fixed 6k training job failed; inspect logs')
            if all(j['status'] == 'complete' and j.get('exit_code') == 0 for j in jobs):
                return
        time.sleep(10)


def wait_for_phase_release():
    # Obtain and immediately release the actual runner mutex. A fixed sleep
    # cannot establish that the training runner has released ownership.
    with (O / 'phase.lock').open('a') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(5)
                continue
            fcntl.flock(lock, fcntl.LOCK_UN)
            return


def children_snapshot(children):
    return [{'script': name, 'pid': process.pid, 'log': str(log),
             'status': 'running' if process.poll() is None else ('complete' if process.returncode == 0 else 'failed'),
             'exit_code': process.returncode}
            for name, process, log in children]


def terminate_own_children(children):
    # Every runner below was started in its own session. Its SIGTERM handler
    # performs cleanup of its own GPU workers before exiting.
    for _, process, _ in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    for _, process, _ in children:
        if process.poll() is None:
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)


def interrupt(signum, _frame):
    raise SystemExit(f'completion runner received signal {signum}')


def main():
    O.mkdir(parents=True, exist_ok=True)
    with (O / '6k_completion.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('another 6k completion runner owns the lock; no state or child was changed')
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        children = []
        try:
            record('waiting_for_training')
            wait_for_training()
            record('waiting_for_phase_lock_release')
            wait_for_phase_release()
            record('verifying_last_checkpoints')
            subprocess.run(command('verify.py', '6k'), cwd=R, check=True)
            plans = [('evaluate_phase.py', O / '6k_evaluation_runner.log', ['6k', '--wait-for-raw']),
                     ('run_raw.py', O / '6k_raw_runner.log', ['6k'])]
            # Batch preflight precedes any long-running evaluation/probe child.
            for _, log, _ in plans:
                if log.exists():
                    raise RuntimeError(f'output log already exists; inspect/archive before retry: {log}')
            if (O / '6k_gate.json').exists():
                raise RuntimeError('6k_gate.json already exists; inspect/archive before retry')
            record('preflighting_evaluation_and_raw')
            for script, _, arguments in plans:
                subprocess.run(command(script, *arguments, '--preflight'), cwd=R, check=True)
            record('recursive_and_raw_evaluation')
            for script, log, arguments in plans:
                with log.open('x') as stream:
                    process = subprocess.Popen(command(script, *arguments), cwd=R,
                                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                children.append((script, process, log))
                record('recursive_and_raw_evaluation', children=children_snapshot(children))
            while True:
                snapshot = children_snapshot(children)
                if any(child['status'] == 'failed' for child in snapshot):
                    raise RuntimeError(f'6k evaluation/probe child failed: {snapshot}')
                if all(child['status'] == 'complete' for child in snapshot):
                    break
                record('recursive_and_raw_evaluation', children=snapshot)
                time.sleep(5)
            record('applying_preregistered_gates', children=children_snapshot(children))
            argv = command('gates.py')
            for mode in ('shared', 'untied'):
                argv += [f'--{mode}-row', f'outputs/semkine/u1a_{mode}_6k_main_row.json',
                         f'--{mode}-extended', f'outputs/semkine/u1a_{mode}_6k_extended.json', f'--{mode}-raw']
                argv += [f'outputs/semkine/u1a_{mode}_6k_s{seed}/probe_raw_last.json' for seed in (3407, 3408)]
            argv += ['--baseline-raw',
                     *[f'outputs/semkine/s38_spmeas_s{seed}/probe_raw_last.json' for seed in (3407, 3408)],
                     '--output', 'outputs/u1a/6k_gate.json']
            subprocess.run(argv, cwd=R, check=True)
            result = json.loads((O / '6k_gate.json').read_text())
            record('complete', sharing_gate_pass=result['gates']['sharing']['pass'],
                   shared_S38_guard_pass=result['gates']['shared_vs_S38']['pass'],
                   untied_S38_guard_pass=result['gates']['untied_vs_S38']['pass'],
                   final_adoption_eligible=result['final_adoption_eligible'],
                   children=children_snapshot(children))
        except BaseException as error:
            terminate_own_children(children)
            record('failed', error=str(error), children=children_snapshot(children))
            raise


if __name__ == '__main__':
    main()
