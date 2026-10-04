#!/usr/bin/env python3
"""Full raw probes on odd GPUs while recursive evaluations use even GPUs.

All outputs/resources are checked before any probe starts. This runner never
trains. --preflight performs read-only checks; no locks, output or GPU work.
"""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

R = Path(__file__).resolve().parents[2]
P = '/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python'
O = R / 'outputs/u1a'
sys.path.insert(0, str(R))
from tools.u1a import evaluate_phase as EP


def read_json(path):
    return json.loads(path.read_text())


def cached_baseline(path, source, seed):
    js = read_json(path)
    require = EP.require
    require(js.get('gate_eligible') is True and js['raw'].get('gate_eligible_full_window_coverage') is True,
            f'cached S38 raw is partial/not gate eligible: {path}')
    require(js.get('seed') == seed and js.get('step') == 6000 and js.get('split') == 'val_core'
            and js.get('u1a_mode') == 'off', f'cached S38 raw identity mismatch: {path}')
    require(js.get('probe_sha256') == EP.sha256(R / 'tools/u1a/probe.py'),
            f'cached S38 raw probe source changed: {path}')
    for path_key, hash_key, source_key in [('ckpt', 'ckpt_sha256', 'checkpoint'),
                                          ('config', 'config_sha256', 'config'),
                                          ('manifest', 'manifest_sha256', 'manifest')]:
        require(Path(js[path_key]).resolve() == Path(source[source_key]['path']).resolve()
                and js[hash_key] == source[source_key]['sha256'], f'cached S38 raw {source_key} mismatch: {path}')
    require(js['raw']['sampled_windows_total'] == js['raw']['protocol_windows_total'] == 2590,
            f'cached S38 raw coverage mismatch: {path}')
    return {'path': str(path), 'sha256': EP.sha256(path), 'seed': seed, 'step': 6000}


def preflight(phase, idle_memory_mb=1024):
    state_path = O / f'{phase}_raw_state.json'
    EP.require(not state_path.exists(), f'old raw state exists; inspect/archive before retry: {state_path}')
    budget, phase_jobs = EP.build_plan(phase, P)
    EP.require((O / f'{phase}_state.json').is_file(), 'training state is missing')
    verification = EP.verify_phase(phase, budget, phase_jobs)
    EP.validate_cpu_partitions(phase_jobs)
    baseline, _ = EP.baseline_preflight()
    jobs = []
    for seed in (3407, 3408):
        for mode in ('shared', 'untied'):
            jobs.append({'run': f'u1a_{mode}_{phase}_s{seed}', 'status': 'pending'})
    reused = []
    for seed in (3407, 3408):
        run = f's38_spmeas_s{seed}'
        target = R / 'outputs/semkine' / run / 'probe_raw_last.json'
        if target.exists():
            reused.append(cached_baseline(target, baseline['provenance'][str(seed)], seed))
        else:
            jobs.append({'run': run, 'status': 'pending'})
    # Check every scheduled output before the first process is created.
    for job in jobs:
        directory = R / 'outputs/semkine' / job['run']
        job.update(directory=str(directory), target=str(directory / 'probe_raw_last.json'),
                   log=str(O / f"{job['run']}.raw.log"))
        EP.require(not Path(job['target']).exists() and not Path(job['log']).exists(),
                   f"old raw artifact exists; inspect/archive before retry: {job['run']}")
    idle = EP.require_idle([1, 3, 5, 7], idle_memory_mb)
    return jobs, verification, reused, idle


def interrupt(signum, _frame):
    raise SystemExit(f'raw runner received signal {signum}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['2k', '6k'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--idle-memory-mb', type=int, default=1024)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args(argv)
    EP.require(1 <= args.threads <= 9 and args.idle_memory_mb > 0, 'invalid raw threads/idle threshold')
    if args.preflight:
        jobs, verification, reused, idle = preflight(args.phase, args.idle_memory_mb)
        print(json.dumps({'preflight': 'passed', 'phase': args.phase, 'jobs': jobs,
                          'verification': verification, 'reused_S38_raw': reused, 'gpu_idle': idle}, indent=2))
        return
    O.mkdir(parents=True, exist_ok=True)
    state_path = O / f'{args.phase}_raw_state.json'
    with (O / 'raw.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('another raw runner owns the lock; no state or child was changed')
        # Refuse old state before entering the failure recorder: do not overwrite it.
        EP.require(not state_path.exists(), f'old raw state exists; inspect/archive before retry: {state_path}')
        signal.signal(signal.SIGTERM, interrupt)
        signal.signal(signal.SIGINT, interrupt)
        children, active = [], {}
        state = {'phase': args.phase, 'runner_pid': os.getpid(), 'start_unix': time.time(),
                 'stage': 'preflight', 'jobs': []}
        try:
            jobs, verification, reused, idle = preflight(args.phase, args.idle_memory_mb)
            state.update(stage='validated', jobs=jobs, verification=verification,
                         reused_S38_raw=reused, gpu_idle_before_raw=idle)
            EP.atomic_state(state_path, state)
            queue = list(jobs)
            while queue or active:
                for slot in range(4):
                    if slot in active or not queue:
                        continue
                    gpu = 2 * slot + 1
                    EP.require_idle([gpu], args.idle_memory_mb)
                    job = queue.pop(0)
                    physical = list(range(slot * 18 + 9, slot * 18 + 18))
                    cpus = physical + [cpu + 72 for cpu in physical]
                    env = EP.environment(gpu, args.threads)
                    command = ['taskset', '-c', ','.join(map(str, cpus)), P, '-u', 'tools/u1a/probe.py',
                               '--run-dir', job['directory'], '--ckpt', 'last', '--output', job['target'],
                               '--threads', str(args.threads)]
                    # x-mode makes the final log creation race a refusal, never an overwrite.
                    with Path(job['log']).open('x') as log:
                        process = subprocess.Popen(command, cwd=R, env=env, stdout=log,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                    children.append((job, process))
                    job.update(gpu=gpu, cpus=cpus, pid=process.pid, argv=command,
                               status='running', start_unix=time.time())
                    active[slot] = (job, process)
                    state['stage'] = 'running'
                    EP.atomic_state(state_path, state)
                    print(f"raw started {job['run']} gpu={gpu}", flush=True)
                for slot, (job, process) in list(active.items()):
                    rc = process.poll()
                    if rc is not None:
                        job.update(status='complete' if rc == 0 else 'failed', exit_code=rc, end_unix=time.time())
                        del active[slot]
                        print(f"raw {job['run']} {job['status']} rc={rc}", flush=True)
                        if rc != 0:
                            raise RuntimeError(f"raw probe failed: {job['run']} rc={rc}")
                EP.atomic_state(state_path, state)
                if queue or active:
                    time.sleep(5)
            state.update(stage='complete', end_unix=time.time())
            EP.atomic_state(state_path, state)
        except BaseException as error:
            EP.terminate_own_children(children)
            state.update(stage='failed', error=str(error), end_unix=time.time())
            EP.atomic_state(state_path, state)
            raise


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        raise SystemExit(f'U1a raw evaluation refused/failed: {error}')
