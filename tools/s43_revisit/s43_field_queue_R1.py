"""Stage5 five fixed arms: all pilot engineering gates, then all full budgets.

One subprocess per physical GPU4–7. The fifth arm waits for the first free
worker; no accuracy selection or early full-result scoring occurs here.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,json,os,queue,subprocess,sys,threading,time
from pathlib import Path

from . import s43_field_training_R1 as T


def run_stage(stage,freeze,*,resume_existing=False):
    jobs=queue.Queue()
    for arm in T.ARMS:jobs.put(arm)
    stop=threading.Event();lock=threading.Lock();processes={};records=[]
    logdir=T.OUT/'logs'/stage;logdir.mkdir(parents=True,exist_ok=True)
    def worker(gpu):
        while not stop.is_set():
            try:arm=jobs.get_nowait()
            except queue.Empty:return
            cfg=T.OUT/'configs'/stage/(arm+'.json');dest=T.OUT/'train'/stage/arm
            if resume_existing and (dest/'complete.json').exists():
                c=T.read(dest/'complete.json')
                if c['actual_supervised_outputs']!=T.STAGES[stage]['target'] or c['sourcefreeze_sha256']!=T.sha(freeze):raise RuntimeError('completed-run resume identity')
                with lock:records.append(dict(stage=stage,arm=arm,returncode=0,reused_complete=True))
                continue
            command=[sys.executable,'-m','tools.s43_revisit.s43_field_training_R1','train','--config',str(cfg),
              '--source-freeze',str(freeze),'--out',str(dest),'--device','cuda:0']
            if resume_existing and dest.exists() and any(dest.iterdir()):
                if not (dest/'last.ckpt').exists():raise RuntimeError('no full resumable checkpoint; preserve failure and request engineering disposition')
                command+=['--resume',str(dest/'last.ckpt')]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',PYTHONUTF8='1',
              OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='8',OMP_NUM_THREADS='8')
            begin=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
            suffix=('.resume_'+str(time.time_ns())) if '--resume' in command else ''
            with (logdir/(arm+suffix+'.log')).open('wb') as log:
                p=subprocess.Popen(command,cwd=T.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                with lock:processes[arm]=p
                code=p.wait()
            record=dict(stage=stage,arm=arm,physical_gpu=gpu,pid=p.pid,returncode=code,command=command,
              started_utc=started,finished_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-begin)
            T.dump(logdir/(arm+suffix+'.job.json'),record)
            with lock:
                records.append(record);processes.pop(arm,None)
                if code:
                    stop.set()
                    for other in processes.values():other.terminate()
            if code:raise RuntimeError(f'{stage}/{arm} failed; preserve logs/checkpoints, no automatic scientific rerun')
    with ThreadPoolExecutor(4) as pool:
        for future in as_completed([pool.submit(worker,gpu) for gpu in (4,5,6,7)]):future.result()
    if len(records)!=5 or any(r['returncode'] for r in records):raise RuntimeError('incomplete five-arm stage')
    T.dump(T.OUT/(stage+'_training_jobs_complete.json'),dict(stage=stage,records=records,all5_fixed_budgets_complete=True))


def audit(stage,freeze):
    command=[sys.executable,'-m','tools.s43_revisit.s43_field_training_R1','audit','--stage',stage,'--source-freeze',str(freeze)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2')
    with (T.OUT/'logs'/(stage+'_audit.log')).open('wb') as log:
        subprocess.run(command,cwd=T.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source-freeze',type=Path,default=T.OUT/'engineering/actual_sourcefreeze.json')
    p.add_argument('--resume-stage',choices=('small','full'),default='small')
    p.add_argument('--resume-existing',action='store_true');a=p.parse_args()
    T.verify_frozen(a.source_freeze,full_input_check=True)
    T.OUT.mkdir(parents=True,exist_ok=True);(T.OUT/'logs').mkdir(exist_ok=True)
    lock=T.OUT/'orchestrator.lock'
    if lock.exists() and a.resume_existing:
        old=T.read(T.OUT/'orchestrator_launch.json')
        try:os.kill(int(old['pid']),0)
        except ProcessLookupError:pass
        else:raise RuntimeError('original orchestrator is still alive')
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():continue
            try:argv=(process/'cmdline').read_bytes().split(b'\0')
            except (FileNotFoundError,PermissionError,ProcessLookupError):continue
            if any(argv[i:i+3]==[b'-m',b'tools.s43_revisit.s43_field_training_R1',b'train'] for i in range(len(argv)-2)):
                raise RuntimeError('a prior training worker is still alive')
        T.dump(T.OUT/('prior_launch_'+str(time.time_ns())+'.json'),old);lock.unlink()
    descriptor=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);os.close(descriptor)
    T.dump(T.OUT/'orchestrator_launch.json',dict(pid=os.getpid(),command=sys.argv,created_utc=datetime.now(timezone.utc).isoformat(),
      sourcefreeze_sha256=T.sha(a.source_freeze),stages=['small','full'],arms=list(T.ARMS),accuracy_gating=False))
    try:
        if a.resume_stage=='small':run_stage('small',a.source_freeze,resume_existing=a.resume_existing);audit('small',a.source_freeze)
        elif T.read(T.OUT/'audits/pilot.json')['status']!='PASS':raise RuntimeError('pilot gate not satisfied')
        # Every full factory starts fresh from canonical, never pilot weights.
        run_stage('full',a.source_freeze,resume_existing=a.resume_existing);audit('full',a.source_freeze)
        T.dump(T.OUT/'all_final_weights_complete.json',dict(status='COMPLETE',sourcefreeze_sha256=T.sha(a.source_freeze),
          arms=list(T.ARMS),supervised_outputs_per_arm=6144000,total_supervised_outputs=30720000,
          fixed_last_weights=True,no_new_zgz_scores_yet=True,final_weights={arm:str(T.OUT/'train/full'/arm/'final_weights.ckpt') for arm in T.ARMS}))
    except BaseException as exc:
        T.dump(T.OUT/'orchestrator_failure.json',dict(status='FAILED',error=repr(exc),created_utc=datetime.now(timezone.utc).isoformat(),
          no_full_accuracyscoring=True,old_negative_results_unchanged=True))
        raise


if __name__=='__main__':main()
