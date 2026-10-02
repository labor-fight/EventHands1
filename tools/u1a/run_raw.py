#!/usr/bin/env python3
"""Full raw probes on odd GPUs while recursive evaluations use even GPUs."""
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
ap = argparse.ArgumentParser()
ap.add_argument('phase', choices=['2k','6k'])
args = ap.parse_args()
lock = open(O/'raw.lock','a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
verification = json.loads((O/f'{args.phase}_verification.json').read_text())
assert len(verification) == 4 and all(v['frozen_encoder_bitwise'] and v['paired_rank_streams_equal']
    and v['all_state_finite'] for v in verification.values())
train = json.loads((O/f'{args.phase}_state.json').read_text())
assert all(j['status'] == 'complete' and j['exit_code'] == 0 for j in train['jobs'])
queue = [f'u1a_{mode}_{args.phase}_s{seed}' for seed in [3407,3408] for mode in ['shared','untied']]
for seed in [3407,3408]:
    run = f's38_spmeas_s{seed}'
    target = R/'outputs/semkine'/run/'probe_raw_last.json'
    if not target.exists():
        queue.append(run)
    else:
        assert json.loads(target.read_text())['raw']['gate_eligible_full_window_coverage']
rows = []
active = {}
state = O/f'{args.phase}_raw_state.json'
while queue or active:
    for slot in range(4):
        if slot in active or not queue:
            continue
        gpu = 2*slot+1
        mem = int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.used',
                   '--format=csv,noheader,nounits'],text=True).strip())
        assert mem < 1024, f'raw GPU {gpu} occupied'
        uuid = subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=uuid',
                    '--format=csv,noheader'],text=True).strip()
        apps = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid',
                    '--format=csv,noheader'],text=True).splitlines()
        assert uuid not in apps, f'raw GPU {gpu} has an active compute process'
        run = queue.pop(0)
        directory = R/'outputs/semkine'/run
        target = directory/'probe_raw_last.json'
        log = O/f'{run}.raw.log'
        assert not target.exists() and not log.exists(), f'raw artifact exists: {run}'
        phys = list(range(slot*18+9,slot*18+18))
        cpus = phys+[c+72 for c in phys]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',
                   MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
        argv = ['taskset','-c',','.join(map(str,cpus)),P,'-u','tools/u1a/probe.py',
                '--run-dir',str(directory),'--ckpt','last','--output',str(target)]
        with log.open('w') as out:
            process = subprocess.Popen(argv,cwd=R,env=env,stdout=out,stderr=subprocess.STDOUT)
        row = {'run':run,'gpu':gpu,'pid':process.pid,'argv':argv,'log':str(log),
               'status':'running','start_unix':time.time()}
        rows.append(row)
        active[slot] = (process,row)
        print(f'raw started {run} gpu={gpu}',flush=True)
    for slot,(process,row) in list(active.items()):
        rc = process.poll()
        if rc is not None:
            row.update(status='complete' if rc == 0 else 'failed',exit_code=rc,end_unix=time.time())
            del active[slot]
            print(f"raw {row['run']} {row['status']} rc={rc}",flush=True)
    temp = state.with_suffix('.tmp')
    temp.write_text(json.dumps({'phase':args.phase,'updated_unix':time.time(),'jobs':rows},indent=2))
    temp.replace(state)
    if queue or active:
        time.sleep(5)
raise SystemExit(0 if all(r['exit_code'] == 0 for r in rows) else 1)
