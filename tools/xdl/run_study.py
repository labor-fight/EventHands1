#!/usr/bin/env python3
"""Durable scheduler for the authorized five exact-step6000 XDL runs.

Training has priority over single-card evaluation. No outcome-based arm or
checkpoint selection is performed. Time budgets charge allocated cards, even
when a process waits for input. A failed task remains a failure in the journal.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / 'outputs/xdl'
ARMS = ['f1', 'q1', 'n1', 'f0', 'q0']
CAPS = {'training':60.0, 'evaluation':12.0}
PREPHASE_RESERVED_HOURS = 10.0


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    p = Path(path)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def live_cost(completed, running, now):
    spent = {k:float(completed.get(k, 0)) for k in CAPS}
    for job in running:
        spent[job['phase']] += (now-job['started_mono'])*len(job['cards'])/3600
    return spent


def exhausted(spent):
    return next((k for k, cap in CAPS.items() if spent.get(k, 0) >= cap), None)


def load_prior_cost(path):
    if not path:
        return {k:0.0 for k in CAPS}
    record = json.loads(Path(path).read_text())
    journal = Path(record['source_journal'])
    if not journal.is_absolute():
        journal = REPO/journal
    if sha(journal) != record['source_journal_sha256']:
        raise RuntimeError('budget carryover source journal changed')
    prior = json.loads(journal.read_text())
    cost = record['gpu_hours_allocated']
    if prior.get('status') != 'stopped_incomplete' or cost != prior['gpu_hours_allocated']:
        raise RuntimeError('budget carryover differs from stopped source journal')
    if any(k not in cost or not 0 <= float(cost[k]) < CAPS[k] for k in CAPS):
        raise RuntimeError('budget carryover is invalid or already exhausted')
    return {k:float(cost[k]) for k in CAPS}


def next_job(free, waiting, ready):
    if waiting and len(free) >= 2:
        return 'training', waiting[0], sorted(free)[:2]
    if ready and free:
        return 'evaluation', ready[0], [min(free)]
    return None


def verify_endpoint(directory):
    p = Path(directory)
    value = json.loads((p/'train_finish.json').read_text())
    if value.get('status') != 'completed_exact_step6000' or value.get('global_step') != 6000 or value.get('preflight_not_formal', True):
        raise RuntimeError('formal exact step6000 completion required')
    checkpoint = Path(value['checkpoint'])
    if not checkpoint.is_absolute():
        checkpoint = REPO/checkpoint
    matches = [f for f in p.glob('*step=*.ckpt')
               if re.search(r'step=(\d+)', f.name)
               and int(re.search(r'step=(\d+)', f.name).group(1)) == 6000]
    if len(matches) != 1 or checkpoint.resolve() != matches[0].resolve() or sha(checkpoint) != value['checkpoint_sha256']:
        raise RuntimeError('unique exact step6000 checkpoint identity mismatch')
    import torch
    if int(torch.load(checkpoint, map_location='cpu').get('global_step', -1)) != 6000:
        raise RuntimeError('saved optimizer step must be 6000')
    return value


def check_freeze(path):
    value = json.loads(Path(path).read_text())
    entries = value['source_sha256']
    if len(entries) < 20:
        raise RuntimeError('incomplete freeze')
    for name, expected in entries.items():
        p = Path(name) if Path(name).is_absolute() else REPO/name
        if sha(p) != expected:
            raise RuntimeError(f'frozen identity changed: {name}')
    data = json.loads((REPO/value['data_identity']).read_text())
    for name, expected in data['files'].items():
        stat = Path(name).stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != (expected['bytes'], expected['mtime_ns'], expected['inode']):
            raise RuntimeError(f'hashed data file metadata changed: {name}')
    for review in value['launch_reviews']:
        r = json.loads((REPO/review).read_text())
        if r.get('status') not in ('PASS', 'PASS_before_root_freeze'):
            raise RuntimeError(f'launch review is not PASS: {review}')
    return sha(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--evaluate-run', choices=ARMS)
    parser.add_argument('--prior-cost-json')
    args = parser.parse_args()
    manifest = Path(args.manifest).resolve()
    freeze_sha = check_freeze(manifest)
    for key, value in json.loads(manifest.read_text()).get('required_environment', {}).items():
        if os.environ.get(key) != value:
            raise RuntimeError(f'frozen runtime setting differs: {key}')
    expected_prior = json.loads(manifest.read_text()).get('prior_cost_json')
    if not args.evaluate_run and expected_prior:
        if not args.prior_cost_json or Path(args.prior_cost_json).resolve() != (REPO/expected_prior).resolve():
            raise RuntimeError('frozen prior GPU budget must be carried over')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1')
    python = sys.executable
    if args.evaluate_run:
        arm = args.evaluate_run
        run = f'xdl_{arm}_s3407'
        verify_endpoint(REPO/f'outputs/semkine/{run}')
        cfg = f'configs/xdl/{run}.yaml'
        for command in (
            [python,'-u','tools/xdl/make_row.py','--run',run,'--config',cfg,'--device','cuda:0'],
            [python,'-u','tools/xdl/diagnostics.py','diagnose','--run',run,'--config',cfg,'--device','cuda:0','--sections','bank,recovery,motion','--batch-size','1']):
            subprocess.run(command, cwd=REPO, env=env, check=True)
        d = json.loads((OUT/f'diagnostics/{run}/diagnostic.json').read_text())
        if d.get('partial_smoke') or d.get('numeric_status') != 'PASS':
            raise RuntimeError('formal diagnostics failed; retained raw results')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    journal = OUT/'runtime_journal.json'
    if journal.exists():
        raise RuntimeError('journal already exists; inspect prior execution before restarting')
    for arm in ARMS:
        directory = REPO/f'outputs/semkine/xdl_{arm}_s3407'
        if directory.exists() and any(directory.iterdir()):
            raise RuntimeError(f'fresh initialization requires empty directory: {directory}')
    waiting, ready, free = list(ARMS), [], list(range(2,8))
    running, finished, errors = [], [], []
    prior_cost = load_prior_cost(args.prior_cost_json)
    completed_cost = dict(prior_cost)
    started = time.monotonic()
    interrupted = [None]
    def handle(signum, _):
        interrupted[0] = f'signal {signum}'
    signal.signal(signal.SIGTERM, handle)
    signal.signal(signal.SIGINT, handle)
    def snapshot(status):
        now = time.monotonic()
        cost = live_cost(completed_cost, running, now)
        public = [{k:v for k,v in j.items() if k not in ('process','handle')} for j in running]
        atomic(journal, {'study':'XD-V2-RAW-RECOVERY-20261013','status':status,
            'time_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
            'freeze_sha256':freeze_sha,'elapsed_s':now-started,
            'waiting_training':waiting,'ready_evaluation':ready,'running':public,
            'finished':finished,'errors':errors,'gpu_hours_allocated':cost,
            'prior_attempt_gpu_hours':prior_cost,
            'prior_cost_source':args.prior_cost_json,
            'runtime_transport':{k:os.environ.get(k) for k in ('NCCL_P2P_DISABLE','NCCL_IB_DISABLE')},
            'phase_caps_gpu_hours':CAPS,'prephase_reserved_gpu_hours':PREPHASE_RESERVED_HOURS,
            'conservative_total_gpu_hours':PREPHASE_RESERVED_HOURS+sum(cost.values()),
            'second_seed_authorized':False})
        return cost
    try:
        while waiting or ready or running:
            for job in list(running):
                code = job['process'].poll()
                if code is None:
                    continue
                elapsed = time.monotonic()-job['started_mono']
                completed_cost[job['phase']] += elapsed*len(job['cards'])/3600
                job['handle'].close()
                running.remove(job)
                free.extend(job['cards'])
                result = {k:v for k,v in job.items() if k not in ('process','handle')}
                result.update(exit_code=code, elapsed_s=elapsed, gpu_hours=elapsed*len(job['cards'])/3600)
                if code == 0 and job['phase'] == 'training':
                    try:
                        result['endpoint'] = verify_endpoint(REPO/f'outputs/semkine/xdl_{job["arm"]}_s3407')
                        ready.append(job['arm'])
                    except Exception as exc:
                        result['validation_error'] = repr(exc)
                        errors.append(result)
                elif code != 0:
                    errors.append(result)
                finished.append(result)
            cost = snapshot('running')
            budget = exhausted(cost)
            if budget or interrupted[0]:
                raise RuntimeError(interrupted[0] or f'{budget} GPU time cap reached; incomplete runs retained')
            while (choice := next_job(free, waiting, ready)) is not None:
                check_freeze(manifest)
                phase, arm, cards = choice
                (waiting if phase == 'training' else ready).remove(arm)
                for card in cards:
                    free.remove(card)
                run = f'xdl_{arm}_s3407'
                task_env = env.copy()
                task_env.update(CUDA_VISIBLE_DEVICES=','.join(map(str,cards)),
                                MASTER_PORT=str(29620+ARMS.index(arm)))
                if phase == 'training':
                    command = [python,'-u','tools/xdl/train.py','--config',f'configs/xdl/{run}.yaml','--frozen-manifest',str(manifest)]
                else:
                    command = [python,'-u',str(Path(__file__).resolve()),'--manifest',str(manifest),'--evaluate-run',arm]
                log = OUT/f'formal_{phase}_{run}.log'
                handle = log.open('w')
                process = subprocess.Popen(command, cwd=REPO, env=task_env, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                running.append({'phase':phase,'arm':arm,'cards':cards,'pid':process.pid,
                    'started_mono':time.monotonic(),'started_utc':dt.datetime.now(dt.timezone.utc).isoformat(),
                    'log':str(log),'argv':command,'process':process,'handle':handle})
                snapshot('running')
            time.sleep(2)
    except BaseException as exc:
        errors.append({'scheduler_error':repr(exc)})
        for job in running:
            try:
                os.killpg(job['process'].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        snapshot('stopped_incomplete')
        raise
    snapshot('completed_all_five' if not errors else 'finished_with_failures')
    if errors:
        raise RuntimeError('failed tasks retained; inspect journal and logs')
    print('All five exact-step6000 runs and complete evaluations finished.', flush=True)


if __name__ == '__main__':
    main()
