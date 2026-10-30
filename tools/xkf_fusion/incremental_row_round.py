"""Launch the six complete rows as each GPU finishes its fixed probes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/incremental_finetune'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    planpath = BASE/'fullrow_execution_plan.json'
    plan = json.loads(planpath.read_text())
    if plan['status'] != 'PREPARED_NOT_STARTED' or len(plan['jobs']) != 6:
        raise ValueError('unexpected complete-row plan')
    out = BASE/'fullrow_round'
    out.mkdir(exist_ok=False)
    jobs = {f'{v["arm"]}_s{v["seed"]}': v for v in plan['jobs']}
    if len(jobs) != 6 or {v['gpu'] for v in jobs.values()} != set(range(2, 8)):
        raise ValueError('one distinct row job per authorized GPU is required')
    sources = {str(planpath): digest(planpath), __file__: digest(__file__)}
    started, running, done, launched = time.monotonic(), {}, {}, set()

    def evaluate(job):
        if digest(job['script']) != job['script_sha256']:
            raise ValueError('row command changed')
        if digest(ROOT/'tools/xkf_fusion/make_row_incremental.py') != plan['producer_sha256']:
            raise ValueError('row producer changed')
        before = time.monotonic()
        result = subprocess.run(['bash', job['script']], cwd=ROOT)
        record = dict(returncode=result.returncode, elapsed_seconds=time.monotonic()-before, **job)
        if result.returncode == 0:
            row = json.loads(Path(job['output']).read_text())
            if row['numeric_status'] != 'PASS' or row['protocol']['formal'] is not False:
                raise ValueError('unexpected full row status')
            record['output_sha256'] = digest(job['output'])
        return record

    with ThreadPoolExecutor(max_workers=6) as pool:
        while len(done) < 6:
            for key, job in jobs.items():
                marker = BASE/'execution_logs'/f'{key}_exit.json'
                if key in launched or not marker.exists():
                    continue
                release = json.loads(marker.read_text())
                if any(release[field] != job[field] for field in ('arm', 'seed', 'gpu')):
                    raise ValueError('GPU release identity differs')
                if release['actual112_exit'] != 0 or release['canonical_motion_exit'] != 0:
                    raise RuntimeError('probe execution failed: '+key)
                launched.add(key)
                running[key] = pool.submit(evaluate, job)
                print(json.dumps(dict(event='start_complete_row', run=key, gpu=job['gpu'])), flush=True)
            for key in list(running):
                if running[key].done():
                    done[key] = running.pop(key).result()
                    print(json.dumps(dict(event='completed_row', run=key, returncode=done[key]['returncode'])), flush=True)
            write(out/'progress.json', dict(running=sorted(running), pending=sorted(set(jobs)-launched),
                  completed=done, elapsed_seconds=time.monotonic()-started, automatic_full_training=False))
            if len(done) == 6:
                break
            if time.monotonic()-started > 7200:
                raise TimeoutError('pending evaluation/GPU releases; artifacts retained')
            time.sleep(5)
    if any(digest(path) != value for path, value in sources.items()):
        raise ValueError('row orchestration source/plan changed')
    result = dict(status='COMPLETE' if all(v['returncode'] == 0 for v in done.values()) else 'EVALUATION_ERROR',
                  jobs=done, source_sha256=sources, automatic_full_training=False)
    if result['status'] == 'COMPLETE':
        runs = []
        for arm in ('raw', 'aug'):
            run = f'xkf_r3_ionly_{arm}_step2600'
            subprocess.run([sys.executable, 'tools/xkf_fusion/finalize_rows.py', '--input-dir',
                            str(BASE/f'fullrow_{arm}'), '--run', run], cwd=ROOT, check=True)
            runs.append(run)
        command = [sys.executable, 'tools/report_table.py', 'dt3_dz_l3_w200cfg',
                   'xkf_r3_raw_motion_ramp_step2400', *runs,
                   '--label', 'dt3_dz_l3_w200cfg=S39：dt_dz_l3，W=200 + 自适应滤波 + 证据门 ramp1000 *',
                   '--label', 'xkf_r3_raw_motion_ramp_step2400=XKF 修复前（I2200/D2400）',
                   '--label', 'xkf_r3_ionly_raw_step2600=XKF 原始事件I修复（debug，I2400/D2400）',
                   '--label', 'xkf_r3_ionly_aug_step2600=XKF 增强事件I对照（debug，I2400/D2400）']
        with (out/'main_table.md').open('w') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, check=True)
        result['merged_runs'] = runs
        result['generated_table_sha256'] = digest(out/'main_table.md')
    write(out/'result.json', result)


if __name__ == '__main__':
    main()
