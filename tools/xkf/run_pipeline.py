#!/usr/bin/env python3
"""Durable XKF debug -> gated fresh formal-training controller.

Start only after the existing three step200 jobs have completed. The frozen
debug criteria are never relaxed: continue 800/1600/2000 if finite, stop if they
still fail, and launch fresh step6000 jobs only after all three seeds pass.
All debug evaluations use CPU. No subprocess from another owner is terminated.

    python tools/xkf/run_pipeline.py --debug-root outputs/xkf/debug_20261015_r1

State, process identities, logs, snapshots and gate evidence live in the debug
root's pipeline/ directory. Restart is safe after a completed common stage;
an interrupted incomplete training stage requires inspection instead of blind
relaunch. A live orphaned child also prevents another controller from starting.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[2]
SEEDS = (3407, 3408, 3409)
CARDS = {3407: '2,3', 3408: '4,5', 3409: '6,7'}
PORTS = {3407: '29731', 3408: '29732', 3409: '29733'}
STAGES = (200, 800, 1600, 2000)
MODES = ('fused', 'i', 'd', 'd_zero')
DEFAULT_PYTHON = '/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python'
INTERRUPTED_SIGNAL = None


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temporary.open('w') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def proc_identity(pid):
    """Linux PID start ticks prevent an unrelated recycled PID matching a job."""
    try:
        value = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if value[0] == 'Z':
            return None
        return {'pid': int(pid), 'start_ticks': value[19]}
    except (OSError, IndexError):
        return None


def verify_sources(manifest, manifest_path=None, manifest_sha=None):
    check_interrupted()
    if manifest_path and digest(manifest_path) != manifest_sha:
        raise RuntimeError('Frozen manifest changed while the pipeline was running')
    entries = manifest.get('source_sha256', {})
    if not entries:
        raise RuntimeError('Missing frozen source map')
    for name, expected in entries.items():
        path = Path(name)
        path = path if path.is_absolute() else REPO / path
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f'Frozen source/config/split identity mismatch: {name}')


def finite_debug(value):
    """Numerical health allows further debug; it can never authorize full training."""
    cohorts, snippets = value.get('cohorts', {}), value.get('snippets', {})
    return (set(cohorts) == {'train', 'check', 'zgz'} and bool(snippets)
            and all(v.get('finite') is True for v in cohorts.values())
            and all(v.get('finite') is True for v in snippets.values())
            and value.get('gradients', {}).get('pass') is True)


def all_readiness_pass(evaluations, frozen_criteria):
    if set(evaluations) != set(SEEDS):
        return False
    for seed, value in evaluations.items():
        gate = value.get('readiness', {})
        checks = gate.get('checks', {})
        if not (int(value.get('identity', {}).get('seed', -1)) == seed
                and finite_debug(value) and gate.get('status') == 'PASS'
                and gate.get('ready_for_full_training') is True
                and checks and all(v is True for v in checks.values())
                and gate.get('criteria') == frozen_criteria):
            return False
    return True


class PipelineInterrupted(Exception):
    pass


def check_interrupted():
    if INTERRUPTED_SIGNAL is not None:
        raise PipelineInterrupted(f'Controller received signal {INTERRUPTED_SIGNAL}')


class Supervisor:
    """Logged process groups owned by one controller; no global pkill or GPU kills."""

    def __init__(self, directory, verify, status):
        self.directory = Path(directory)
        self.jobs_dir = self.directory / 'jobs'
        self.logs_dir = self.directory / 'logs'
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.verify, self.status = verify, status
        self.owned = {}

    def reject_live_previous_jobs(self):
        for path in self.jobs_dir.glob('*.json'):
            value = read_json(path)
            identity = value.get('process')
            if value.get('status') == 'RUNNING' and identity and proc_identity(identity['pid']) == identity:
                raise RuntimeError(f'Previous controller child is still running: {path.name}, PID {identity["pid"]}')

    def stop_owned(self):
        owned = list(self.owned.items())
        for _, item in owned:
            # A failed DDP leader can exit before its rank/worker children. Its
            # process group is still ours and must also be reaped on failure.
            try:
                os.killpg(item['process'].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 15
        while any(item['process'].poll() is None for _, item in owned) and time.monotonic() < deadline:
            time.sleep(.1)
        for job, item in owned:
            process = item['process']
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            item['log'].close()
            old_status = item['record']['status']
            item['record'].update(status=old_status if old_status == 'FAILED' else 'INTERRUPTED',
                                  finished_utc=now(), returncode=process.returncode)
            atomic_json(self.jobs_dir / f'{job}.json', item['record'])
        self.owned.clear()

    def run(self, jobs, phase):
        """jobs: [(name, argv, environment-overrides)]. Fail fast and reap all own peers."""
        self.verify()
        self.status(phase=phase, active_jobs=[v[0] for v in jobs])
        try:
            for name, argv, overrides in jobs:
                check_interrupted()
                path = self.jobs_dir / f'{name}.json'
                if path.exists():
                    # Each retry retains the complete preceding process/log evidence.
                    stamp = uuid.uuid4().hex[:10]
                    shutil.copy2(path, path.with_name(path.stem + f'.attempt-{stamp}.json'))
                logpath = self.logs_dir / f'{name}.{uuid.uuid4().hex[:10]}.log'
                log = logpath.open('wb')
                env = os.environ.copy()
                env.update(OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
                           NCCL_P2P_DISABLE='1', NCCL_ASYNC_ERROR_HANDLING='1', NCCL_DEBUG='WARN')
                env.update(overrides)
                try:
                    process = subprocess.Popen(argv, cwd=REPO, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                record = {'status': 'RUNNING', 'argv': argv, 'environment_overrides': overrides,
                          'process': proc_identity(process.pid), 'started_utc': now(), 'log': str(logpath),
                          'controller_pid': os.getpid()}
                self.owned[name] = {'process': process, 'record': record, 'log': log}
                atomic_json(path, record)
                check_interrupted()
            last_check = 0.
            while self.owned:
                check_interrupted()
                for name, item in list(self.owned.items()):
                    code = item['process'].poll()
                    if code is None:
                        continue
                    item['log'].close()
                    item['record'].update(status='SUCCESS' if code == 0 else 'FAILED',
                                          returncode=code, finished_utc=now())
                    atomic_json(self.jobs_dir / f'{name}.json', item['record'])
                    if code:
                        raise RuntimeError(f'Job {name} exited {code}; see {item["record"]["log"]}')
                    del self.owned[name]
                if time.monotonic() - last_check >= 30:
                    self.verify()
                    self.status(phase=phase, active_jobs=list(self.owned), heartbeat_utc=now())
                    last_check = time.monotonic()
                if self.owned:
                    time.sleep(2)
        except BaseException:
            self.stop_owned()
            raise
        self.verify()
        self.status(phase=phase, active_jobs=[])


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.debug_root).resolve()
        self.directory = self.root / 'pipeline'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / 'frozen_manifest.json'
        self.manifest = read_json(self.manifest_path)
        self.manifest_sha = digest(self.manifest_path)
        self.protocol_path = self.root / 'protocol.json'
        self.protocol = read_json(self.protocol_path)
        if digest(self.protocol_path) != self.manifest['protocol_sha256']:
            raise RuntimeError('Frozen protocol identity mismatch')
        self.criteria = self.protocol['debug_readiness_criteria']
        self.controller_sha = digest(__file__)
        self.state_path = self.directory / 'status.json'
        self.state = read_json(self.state_path) if self.state_path.exists() else {}
        if self.state and (self.state.get('manifest_sha256') != self.manifest_sha
                           or self.state.get('controller_sha256') != self.controller_sha):
            raise RuntimeError('Controller/manifest identity changed since previous execution; inspect before restarting')
        self.state.update(controller_sha256=self.controller_sha, manifest_sha256=self.manifest_sha,
                          controller_pid=os.getpid(), started_utc=now())
        self.supervisor = Supervisor(self.directory, self.verify, self.status)
        self.supervisor.reject_live_previous_jobs()
        self.verify()

    def status(self, **updates):
        self.state.update(updates, updated_utc=now())
        atomic_json(self.state_path, self.state)
        with (self.directory / 'events.jsonl').open('a') as f:
            f.write(json.dumps({'utc': now(), **updates}, ensure_ascii=False, allow_nan=False) + '\n')

    def verify(self):
        verify_sources(self.manifest, self.manifest_path, self.manifest_sha)
        if digest(self.protocol_path) != self.manifest['protocol_sha256'] or digest(__file__) != self.controller_sha:
            raise RuntimeError('Frozen protocol/controller changed during execution')

    def config(self, seed):
        return REPO / f'configs/xkf/xkf_s{seed}.yaml'

    def arm(self, seed, formal=False):
        return (REPO / 'outputs/semkine' if formal else self.root) / f'xkf_s{seed}'

    def checkpoint(self, seed, step, formal=False):
        return self.arm(seed, formal) / f'xkf_s{seed}-step={step}.ckpt'

    def validate_training(self, seed, step, formal=False):
        directory = self.arm(seed, formal)
        finish = read_json(directory / 'train_finish.json')
        metadata = read_json(directory / 'training_metadata.json')
        checkpoint = self.checkpoint(seed, step, formal)
        if (int(finish.get('global_step', -1)) != step or not checkpoint.is_file()
                or digest(checkpoint) != finish.get('checkpoint_sha256')):
            raise RuntimeError(f'Missing/mismatched exact checkpoint completion for seed{seed} step{step}')
        if (finish.get('preflight_not_formal') is not (not formal)
                or metadata.get('seed') != seed or metadata.get('config_sha256') != digest(self.config(seed))
                or metadata.get('effective_batch') != 1024 or metadata.get('devices') != 2
                or metadata.get('batch_size_per_gpu') != 512):
            raise RuntimeError(f'Training contract mismatch for seed{seed} step{step}')
        for rank in range(2):
            exposure = read_json(directory / f'actual_exposure_rank{rank}.json')
            if int(exposure.get('final_global_step', -1)) != step:
                raise RuntimeError(f'Incomplete rank exposure for seed{seed} step{step} rank{rank}')
        initial = directory / 'initial_state.ckpt'
        if not initial.is_file() or digest(initial) != metadata.get('initial_checkpoint_sha256'):
            raise RuntimeError(f'Saved initial checkpoint identity mismatch for seed{seed}')
        return checkpoint

    def current_debug_step(self):
        steps = [int(read_json(self.arm(seed) / 'train_finish.json').get('global_step', -1)) for seed in SEEDS]
        if len(set(steps)) != 1 or steps[0] not in STAGES:
            raise RuntimeError(f'Debug jobs are incomplete or at different stages: {dict(zip(SEEDS, steps))}; inspect before restarting')
        step = steps[0]
        if step < self.args.start_after_existing_step:
            raise RuntimeError(f'Existing debug step{step} has not reached requested start endpoint')
        for seed in SEEDS:
            self.validate_training(seed, step)
        if step == 200:
            marker = self.root / 'status.txt'
            if not marker.is_file() or 'debug training complete' not in marker.read_text():
                raise RuntimeError('Existing step200 launcher has not marked all three processes completed')
        return step

    def eval_path(self, seed, step):
        return self.directory / 'evaluations' / f'xkf_s{seed}_step{step}.json'

    def valid_eval(self, seed, step):
        path = self.eval_path(seed, step)
        checkpoint = self.arm(seed) / 'initial_state.ckpt' if step == 0 else self.checkpoint(seed, step)
        if not path.is_file():
            return False
        value = read_json(path)
        identity = value.get('identity', {})
        if (identity.get('global_step') != step or identity.get('seed') != seed
                or identity.get('checkpoint_sha256') != digest(checkpoint)
                or identity.get('config_sha256') != digest(self.config(seed))
                or value.get('producer_sha256') != self.manifest['source_sha256']['tools/xkf/debug_eval.py']
                or value.get('readiness', {}).get('criteria') != self.criteria
                or value.get('device') != 'cpu'):
            raise RuntimeError(f'Existing evaluation identity mismatch: {path}')
        return True

    def evaluate(self, step):
        for target in (0, step):
            jobs = []
            for seed in SEEDS:
                if self.valid_eval(seed, target):
                    continue
                checkpoint = self.arm(seed) / 'initial_state.ckpt' if target == 0 else self.checkpoint(seed, target)
                argv = [self.args.python, '-u', 'tools/xkf/debug_eval.py', '--config', str(self.config(seed)),
                        '--checkpoint', str(checkpoint), '--output', str(self.eval_path(seed, target)),
                        '--device', 'cpu', '--cpu-threads', str(self.args.cpu_threads), '--gradients']
                if target:
                    argv += ['--compare', str(self.eval_path(seed, 0))]
                jobs.append((f'eval_s{seed}_step{target}', argv, {'CUDA_VISIBLE_DEVICES': ''}))
            if jobs:
                self.supervisor.run(jobs, f'debug_evaluation_step{target}')
            for seed in SEEDS:
                if not self.valid_eval(seed, target):
                    raise RuntimeError('Evaluation subprocess succeeded without its identity-bound result')
        evaluations = {seed: read_json(self.eval_path(seed, step)) for seed in SEEDS}
        if not all(finite_debug(v) for v in evaluations.values()):
            raise FloatingPointError(f'Nonfinite/inactive branch or failed snippet at step{step}; further training rejected')
        return evaluations

    def snapshot(self, step):
        destination = self.directory / 'snapshots' / f'step{step}'
        if destination.exists():
            snapshot = read_json(destination / 'manifest.json')
            for name, expected in snapshot['files'].items():
                if digest(destination / name) != expected:
                    raise RuntimeError(f'Previous stage snapshot changed: {name}')
            return
        temporary = destination.with_name(destination.name + '.tmp.' + uuid.uuid4().hex[:8])
        temporary.mkdir(parents=True)
        files = {}
        checkpoints = {}
        for seed in SEEDS:
            arm = self.arm(seed)
            selected = [arm / name for name in ('config_resolved.yaml', 'training_metadata.json', 'progress.json',
                        'train_finish.json', 'learning_latest.json', 'learning.jsonl',
                        'actual_exposure_rank0.json', 'actual_exposure_rank1.json')]
            selected += list(arm.glob('nonfinite*.json'))
            selected += [self.eval_path(seed, step)]
            initial_log = self.root / 'logs' / f'xkf_s{seed}.log'
            if step == 200 and initial_log.exists():
                selected.append(initial_log)
            for source in selected:
                if not source.exists():
                    continue
                name = f'xkf_s{seed}/{source.name}'
                target = temporary / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
                files[name] = digest(target)
            checkpoints[str(seed)] = {'checkpoint': str(self.checkpoint(seed, step)),
                                      'sha256': digest(self.checkpoint(seed, step)),
                                      'initial_sha256': digest(arm / 'initial_state.ckpt')}
        atomic_json(temporary / 'manifest.json', {'step': step, 'created_utc': now(), 'files': files,
                    'checkpoints': checkpoints, 'frozen_manifest_sha256': self.manifest_sha})
        temporary.rename(destination)

    def resume(self, previous, target):
        self.snapshot(previous)
        jobs = []
        for seed in SEEDS:
            self.validate_training(seed, previous)
            if self.checkpoint(seed, target).exists():
                raise RuntimeError(f'Target checkpoint already exists without matched completion: {self.checkpoint(seed, target)}')
            argv = [self.args.python, '-u', 'tools/xkf/train.py', '--config', str(self.config(seed)),
                    '--output-dir', str(self.arm(seed)), '--run-name', f'xkf_s{seed}', '--preflight',
                    '--resume', str(self.checkpoint(seed, previous)), '--max-steps', str(target),
                    '--frozen-manifest', str(self.manifest_path)]
            jobs.append((f'train_s{seed}_step{target}', argv,
                         {'CUDA_VISIBLE_DEVICES': CARDS[seed], 'MASTER_PORT': PORTS[seed]}))
        self.status(debug_step=target, previous_debug_step=previous)
        self.supervisor.run(jobs, f'debug_training_to_step{target}')
        for seed in SEEDS:
            self.validate_training(seed, target)
            initial_before = read_json(self.directory / 'snapshots' / f'step{previous}' / 'manifest.json')['checkpoints'][str(seed)]['initial_sha256']
            if digest(self.arm(seed) / 'initial_state.ckpt') != initial_before:
                raise RuntimeError(f'Resume changed saved initialization for seed{seed}')

    def write_gate(self, step, evaluations):
        if not all_readiness_pass(evaluations, self.criteria):
            raise RuntimeError('Full training rejected: not all three unchanged readiness gates passed')
        self.verify()
        path = self.directory / 'readiness_gate.json'
        atomic_json(path, {'status': 'PASS', 'all_three_seeds_passed': True, 'debug_step': step,
                          'created_utc': now(), 'criteria': self.criteria,
                          'source_sha256': self.manifest['source_sha256'],
                          'frozen_manifest_sha256': self.manifest_sha,
                          'evaluations': {str(seed): {'path': str(self.eval_path(seed, step)),
                          'sha256': digest(self.eval_path(seed, step)), 'checks': value['readiness']['checks']}
                          for seed, value in evaluations.items()}, 'formal_mechanism_gates_assessed': False})
        return path

    def formal_training(self, gate):
        for seed in SEEDS:
            if self.arm(seed, True).exists():
                raise RuntimeError(f'Fresh formal destination already exists; never overwrite: {self.arm(seed, True)}')
        self.status(formal_started=True, gate=str(gate))
        self.supervisor.run([('formal_training_all_seeds', ['bash', 'tools/xkf/launch.sh', 'full', '6000',
                            str(REPO / 'outputs/semkine')], {'XKF_DEBUG_GATE': str(gate),
                            'XKF_FROZEN_MANIFEST': str(self.manifest_path), 'XKF_PYTHON': self.args.python,
                            'CUDA_VISIBLE_DEVICES': '2,3,4,5,6,7'})], 'formal_training_step6000')
        for seed in SEEDS:
            self.validate_training(seed, 6000, True)

    def formal_evaluation(self):
        rows = []
        for mode in MODES:
            jobs = []
            for seed in SEEDS:
                self.validate_training(seed, 6000, True)
                suffix = '' if mode == 'fused' else f'_k{mode}'
                output = REPO / f'outputs/semkine/xkf_s{seed}{suffix}_main_row.json'
                if output.exists():
                    value = read_json(output)
                    if (value.get('numeric_status') != 'PASS'
                            or value.get('protocol', {}).get('exact_optimizer_step') != 6000
                            or value.get('provenance', {}).get('checkpoint_sha256') != digest(self.checkpoint(seed, 6000, True))
                            or value.get('xkf', {}).get('kmode') != mode):
                        raise RuntimeError(f'Existing formal row identity mismatch: {output}')
                else:
                    argv = [self.args.python, '-u', 'tools/xkf/make_row.py',
                            '--run', f'xkf_s{seed}', '--device', 'cuda:0', '--kmode', mode]
                    if mode != 'fused':
                        argv.append('--skip-latency')
                    jobs.append((f'formal_row_s{seed}_{mode}', argv,
                                 {'CUDA_VISIBLE_DEVICES': CARDS[seed].split(',')[0]}))
                rows.append(str(output))
            if jobs:
                self.supervisor.run(jobs, f'formal_rows_{mode}')
            for seed in SEEDS:
                suffix = '' if mode == 'fused' else f'_k{mode}'
                output = REPO / f'outputs/semkine/xkf_s{seed}{suffix}_main_row.json'
                value = read_json(output)
                if (value.get('numeric_status') != 'PASS' or value.get('finite_subset_reduction') is not False
                        or value.get('protocol', {}).get('exact_optimizer_step') != 6000
                        or value.get('provenance', {}).get('checkpoint_sha256') != digest(self.checkpoint(seed, 6000, True))
                        or value.get('xkf', {}).get('kmode') != mode):
                    raise RuntimeError(f'Formal evaluator did not produce a valid exact-step row: {output}')
        report = [self.args.python, 'tools/report_table.py', 'dt3_dz_l3_w200cfg',
                  *[f'xkf_s{seed}' for seed in SEEDS], '--label',
                  'dt3_dz_l3_w200cfg=S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *']
        self.supervisor.run([('formal_report_table', report, {'CUDA_VISIBLE_DEVICES': ''})], 'formal_table_generation')
        report_job = read_json(self.directory / 'jobs/formal_report_table.json')
        table_tmp = self.directory / 'main_table.md.tmp'
        shutil.copyfile(report_job['log'], table_tmp)
        table_tmp.replace(self.directory / 'main_table.md')
        diagnostics = REPO / 'tools/xkf/diagnostics.py'
        if not diagnostics.is_file():
            self.status(status='ROWS_COMPLETE_MECHANISM_PENDING', phase='finished', formal_rows=rows,
                        reason='tools/xkf/diagnostics.py is missing; formal G1-G5 remain unassessed')
            return
        help_job = ('diagnostics_cli', [self.args.python, str(diagnostics), '--help'], {'CUDA_VISIBLE_DEVICES': ''})
        self.supervisor.run([help_job], 'diagnostics_cli_validation')
        help_record = read_json(self.directory / 'jobs/diagnostics_cli.json')
        help_text = Path(help_record['log']).read_text()
        if not all(flag in help_text for flag in ('--run', '--output', '--device')):
            self.status(status='ROWS_COMPLETE_MECHANISM_PENDING', phase='finished', formal_rows=rows,
                        reason='Existing diagnostics CLI lacks --run/--output/--device; manual integration required')
            return
        diagnostics_sha = digest(diagnostics)
        jobs = [(f'formal_diagnostics_s{seed}', [self.args.python, '-u', str(diagnostics), '--run', f'xkf_s{seed}',
                 '--device', 'cpu', '--output', str(self.directory / 'diagnostics' / f'xkf_s{seed}.json')],
                 {'CUDA_VISIBLE_DEVICES': ''}) for seed in SEEDS]
        (self.directory / 'diagnostics').mkdir(exist_ok=True)
        self.supervisor.run(jobs, 'formal_mechanism_diagnostics')
        if digest(diagnostics) != diagnostics_sha:
            raise RuntimeError('Diagnostics implementation changed during evaluation')
        self.status(status='FORMAL_ARTIFACTS_COMPLETE_REVIEW_REQUIRED', phase='finished', formal_rows=rows,
                    diagnostics_producer_sha256=diagnostics_sha,
                    reason='Formal rows and diagnostic artifacts exist; individual mechanism gate verdicts require review')

    def run(self):
        if self.state.get('status') in ('ROWS_COMPLETE_MECHANISM_PENDING', 'FORMAL_ARTIFACTS_COMPLETE_REVIEW_REQUIRED'):
            return
        self.status(status='RUNNING', phase='validating')
        if self.state.get('formal_started'):
            # An interrupted fresh formal launch is never silently relaunched.
            for seed in SEEDS:
                self.validate_training(seed, 6000, True)
            self.formal_evaluation()
            return
        step = self.current_debug_step()
        while True:
            self.status(debug_step=step)
            evaluations = self.evaluate(step)
            self.snapshot(step)
            if all_readiness_pass(evaluations, self.criteria):
                gate = self.write_gate(step, evaluations)
                self.formal_training(gate)
                self.formal_evaluation()
                return
            failed = {str(seed): {k: v for k, v in value['readiness']['checks'].items() if v is not True}
                      for seed, value in evaluations.items()}
            if step == STAGES[-1]:
                self.status(status='STOPPED_DEBUG_GATE_FAIL', phase='finished', debug_step=step,
                            failed_checks=failed, reason='Debug still fails at step2000; full training was not launched')
                return
            target = next(v for v in STAGES if v > step)
            self.status(phase='debug_gate_failed_continue', failed_checks=failed, next_debug_step=target)
            self.resume(step, target)
            step = target


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--debug-root', required=True)
    ap.add_argument('--start-after-existing-step', type=int, choices=[200], default=200)
    ap.add_argument('--python', default=DEFAULT_PYTHON)
    ap.add_argument('--cpu-threads', type=int, default=4)
    args = ap.parse_args()
    root = Path(args.debug_root).resolve()
    if not root.is_dir():
        ap.error('debug root must already exist')
    lock = (root / 'pipeline.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit('Another XKF pipeline controller holds the lock')
    lock.seek(0)
    lock.truncate()
    lock.write(json.dumps({'pid': os.getpid(), 'started_utc': now()}) + '\n')
    lock.flush()
    pipeline = None

    def interrupted(signum, _frame):
        # Defer the exception until after process registration, so Ctrl-C in
        # Popen cannot leave a child outside the owned-process cleanup map.
        global INTERRUPTED_SIGNAL
        INTERRUPTED_SIGNAL = signum

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    try:
        pipeline = Pipeline(args)
        pipeline.run()
    except BaseException as error:
        if pipeline is not None:
            pipeline.supervisor.stop_owned()
            pipeline.status(status='STOPPED_INTERRUPTED' if isinstance(error, (KeyboardInterrupt, PipelineInterrupted))
                            else 'STOPPED_FAILURE', phase='stopped', error=f'{type(error).__name__}: {error}')
        else:
            atomic_json(root / 'pipeline_start_failure.json', {'utc': now(), 'error': f'{type(error).__name__}: {error}'})
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
