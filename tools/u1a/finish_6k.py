#!/usr/bin/env python3
"""Finish the already-running fixed 6k experiment; stop on any failed check."""
import json
import subprocess
import time
from pathlib import Path
R = Path(__file__).resolve().parents[2]
O = R/'outputs/u1a'
P = '/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python'
status = O/'6k_completion_state.json'
def record(stage, **extra):
    obj = {'stage':stage,'updated_unix':time.time(),**extra}
    temp = status.with_suffix('.tmp')
    temp.write_text(json.dumps(obj,indent=2))
    temp.replace(status)
    print(stage,flush=True)
def command(script, *arguments):
    return [P,'-u',f'tools/u1a/{script}',*arguments]
try:
    record('waiting_for_training')
    while True:
        source = O/'6k_state.json'
        if source.exists():
            state = json.loads(source.read_text())
            jobs = state['jobs']
            if any(j['status'] == 'failed' for j in jobs):
                raise RuntimeError('a fixed 6k training job failed; inspect logs')
            if len(jobs) == 4 and all(j['status'] == 'complete' and j['exit_code'] == 0 for j in jobs):
                break
        time.sleep(10)
    # The trainer runner releases its exclusive lock after persisting completion.
    time.sleep(2)
    record('verifying_last_checkpoints')
    subprocess.run(command('verify.py','6k'),cwd=R,check=True)
    record('recursive_and_raw_evaluation')
    children = []
    for script, name in [('evaluate_phase.py','6k_evaluation_runner.log'),('run_raw.py','6k_raw_runner.log')]:
        log = O/name
        assert not log.exists(), f'output log already exists: {log}'
        with log.open('x') as stream:
            children.append(subprocess.Popen(command(script,'6k'),cwd=R,stdout=stream,stderr=subprocess.STDOUT))
    codes = [child.wait() for child in children]
    if any(code != 0 for code in codes):
        raise RuntimeError(f'6k evaluation failed: {codes}')
    record('applying_preregistered_gates')
    argv = command('gates.py')
    for mode in ['shared','untied']:
        argv += [f'--{mode}-row',f'outputs/semkine/u1a_{mode}_6k_main_row.json',
                 f'--{mode}-extended',f'outputs/semkine/u1a_{mode}_6k_extended.json',f'--{mode}-raw']
        argv += [f'outputs/semkine/u1a_{mode}_6k_s{seed}/probe_raw_last.json' for seed in [3407,3408]]
    argv += ['--baseline-raw',*[f'outputs/semkine/s38_spmeas_s{seed}/probe_raw_last.json' for seed in [3407,3408]],
             '--output','outputs/u1a/6k_gate.json']
    subprocess.run(argv,cwd=R,check=True)
    result = json.loads((O/'6k_gate.json').read_text())
    record('complete',sharing_gate_pass=result['gates']['sharing']['pass'],
           shared_S38_guard_pass=result['gates']['shared_vs_S38']['pass'],
           untied_S38_guard_pass=result['gates']['untied_vs_S38']['pass'],
           final_adoption_eligible=result['final_adoption_eligible'])
except BaseException as error:
    record('failed',error=str(error))
    raise
