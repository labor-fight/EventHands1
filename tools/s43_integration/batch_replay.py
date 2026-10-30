"""Bounded CPU orchestration of complete recordings; no truncated runs."""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_integration_20261027'
ARMS=['a0_frozen','a1_legacy','a2_enhanced','a3_br','a4_br_adapter']


def run_job(job):
    arm,seq=job;prefix=OUT/'replay'/f'{arm}_{seq}_public'
    result=Path(str(prefix)+'.json')
    if result.exists():
        obj=json.loads(result.read_text())
        if not (obj.get('full_recording',False) and obj.get('all_valid_native_events_consumed',False)
                and obj['runs_processed']==obj['valid_runs_total']):
            raise ValueError(f'existing report lacks completion proof: {result}')
        return {'arm':arm,'seq':seq,'existing':True,'report':str(result),'exit_code':0}
    command=[sys.executable,'-m','tools.s43_integration.replay','--config',
             str(ROOT/'configs/s43_integration'/f'{arm}.yaml'),'--seq',seq,'--out',str(prefix),
             '--threads','1','--init','public','--duplicate-query']
    if arm!='a0_frozen':command+=['--checkpoint',str(OUT/'train'/arm/'last.ckpt')]
    log=OUT/f'replay_{arm}_{seq}.log';started=time.perf_counter()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.open('x') as handle:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    return {'arm':arm,'seq':seq,'report':str(result),'exit_code':process.returncode,
            'seconds':time.perf_counter()-started,'command':command,'log':str(log)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',type=int,default=8)
    p.add_argument('--seqs',nargs='+',default=['ycy_local','ycy_global','zgz_local','zgz_global'])
    p.add_argument('--arms',nargs='+',default=ARMS);p.add_argument('--receipt',type=Path,default=OUT/'replay_batch.json')
    a=p.parse_args();started=time.perf_counter();results=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pending={pool.submit(run_job,(arm,seq)):(arm,seq) for arm in a.arms for seq in a.seqs}
        for future in as_completed(pending):
            try:row=future.result()
            except Exception as error:row={'job':pending[future],'exit_code':-1,'error':str(error)}
            results.append(row);print(json.dumps(row,ensure_ascii=False),flush=True)
    if a.receipt.exists():raise FileExistsError(a.receipt)
    a.receipt.write_text(json.dumps({'workers':a.workers,'seconds':time.perf_counter()-started,
                 'all_pass':all(r['exit_code']==0 for r in results),'jobs':results},ensure_ascii=False,indent=2))
    if not all(r['exit_code']==0 for r in results):raise SystemExit(1)


if __name__=='__main__':main()
