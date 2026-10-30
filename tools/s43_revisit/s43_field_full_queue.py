"""Stage5 five fixed arms: all pilot engineering gates, then all full budgets.

One subprocess per physical GPU4–7. The fifth arm waits for the first free
worker; no accuracy selection or early full-result scoring occurs here.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,json,os,queue,subprocess,sys,threading,time
from pathlib import Path

from . import s43_field_full_training as T
from . import s43_field_training_R1 as P
import signal


def run_stage(stage,freeze,*,resume_existing=False):
    jobs=queue.Queue()
    for arm in T.ARMS:jobs.put(arm)
    stop=threading.Event();lock=threading.Lock();processes={};records=[]
    logdir=T.OUT/'logs'/stage;logdir.mkdir(parents=True,exist_ok=True)
    def worker(gpu):
        while not stop.is_set():
            try:arm=jobs.get_nowait()
            except queue.Empty:return
            cfg=T.OUT/'configs/full_fastreadout'/(arm+'.json');dest=T.OUT/'train'/stage/arm
            if resume_existing and (dest/'complete.json').exists():
                c=T.read(dest/'complete.json')
                if c['actual_supervised_outputs']!=T.STAGES[stage]['target'] or c['sourcefreeze_sha256']!=T.sha(freeze):raise RuntimeError('completed-run resume identity')
                with lock:records.append(dict(stage=stage,arm=arm,returncode=0,reused_complete=True))
                continue
            command=[sys.executable,'-m','tools.s43_revisit.s43_field_full_training','train','--config',str(cfg),
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
    command=[sys.executable,'-m','tools.s43_revisit.s43_field_full_training','audit','--stage',stage,'--source-freeze',str(freeze)]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2')
    with (T.OUT/'logs'/(stage+'_audit.log')).open('wb') as log:
        subprocess.run(command,cwd=T.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)


def adopt_pilot(controller_pid,freeze):
    """Pause only the old dispatcher after all five jobs have been launched.

    Pilot children continue untouched. Their existing fixed predictions/Adam
    state are not reused by full; only the original engineering gate is adopted.
    """
    launch=P.read(P.OUT/'orchestrator_launch.json')
    if int(launch['pid'])!=controller_pid or launch['sourcefreeze_sha256']!=T.PILOT_FREEZE_SHA:raise RuntimeError('pilot dispatcher identity')
    process=Path('/proc')/str(controller_pid)
    args=(process/'cmdline').read_bytes().split(b'\0')
    if b'tools.s43_revisit.s43_field_queue_R1' not in args:raise RuntimeError('refuse to signal a different process')
    while not all((P.OUT/'train/small'/arm/'manifest.json').exists() for arm in P.ARMS):
        if (P.OUT/'orchestrator_failure.json').exists():raise RuntimeError('pilot already failed')
        time.sleep(5)
    if any((T.OUT/'train/full'/arm).exists() for arm in T.ARMS):raise RuntimeError('old full already started; no automatic execution migration')
    workers={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():continue
        try:argv=(entry/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if b'tools.s43_revisit.s43_field_training_R1' not in argv or b'--out' not in argv:continue
        out=Path(argv[argv.index(b'--out')+1].decode())
        if out.parent==P.OUT/'train/small':workers[out.name]=int(entry.name)
    os.kill(controller_pid,signal.SIGSTOP)
    for _ in range(40):
        if (process/'stat').read_text().split(') ',1)[1].split()[0]=='T':break
        time.sleep(.05)
    else:raise RuntimeError('dispatcher stop was not confirmed')
    if any((T.OUT/'train/full'/arm).exists() for arm in T.ARMS):
        os.kill(controller_pid,signal.SIGCONT)
        raise RuntimeError('old full appeared before confirmed stop; migration rejected')
    T.dump(T.OUT/'pilot_dispatcher_paused_for_full_execution.json',dict(pid=controller_pid,created_utc=datetime.now(timezone.utc).isoformat(),
      pilot_workers=workers,reason='All5 pilot jobs already launched. Children continue unchanged; prevent old dispatcher automatically starting old full execution.',
      pilot_sourcefreeze_sha256=T.PILOT_FREEZE_SHA,full_sourcefreeze_sha256=T.sha(freeze)))
    try:
        while not all((P.OUT/'train/small'/arm/'complete.json').exists() for arm in P.ARMS):
            if any((P.OUT/'train/small'/arm/'failure.json').exists() for arm in P.ARMS):raise RuntimeError('pilot worker failed; original guard resumes')
            for arm in P.ARMS:
                if (P.OUT/'train/small'/arm/'complete.json').exists():continue
                pid=workers.get(arm);stat=Path('/proc')/str(pid)/'stat'
                if pid is None or not stat.exists():raise RuntimeError('pilot worker exited without sealed complete: '+arm)
                try:state=stat.read_text().split(') ',1)[1].split()[0]
                except FileNotFoundError:state='exited'
                if state in ('Z','exited'):raise RuntimeError('pilot worker exited without sealed complete: '+arm)
            T.verify_frozen(freeze)
            time.sleep(10)
        # Recompute the unchanged original five-arm gate, not merely a status string.
        P.audit('small',T.PILOT_FREEZE)
        T.verify_frozen(freeze,full_input_check=True)
        receipt=dict(status='PASS',pilot_sourcefreeze_sha256=T.PILOT_FREEZE_SHA,full_sourcefreeze_sha256=T.sha(freeze),
          original_pilot_audit_sha256=T.sha(P.OUT/'audits/pilot.json'),all5_fresh_pilots_completed=True,
          pilot_completes={arm:T.sha(P.OUT/'train/small'/arm/'complete.json') for arm in P.ARMS},
          original_dispatcher_pid=controller_pid,transfer='dispatcher only; all five actual32768 budgets unchanged; all full runs freshcanonical',
          created_utc=datetime.now(timezone.utc).isoformat())
        T.dump(T.OUT/'pilot_full_execution_transfer.json',receipt)
        # The stopped old dispatcher has no remaining job to launch. Its children
        # have sealed complete artifacts. Preserve its launch/log identity.
        os.kill(controller_pid,signal.SIGKILL)
    except BaseException:
        try:os.kill(controller_pid,signal.SIGCONT)
        except ProcessLookupError:pass
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-freeze',type=Path,default=T.OUT/'engineering/full_execution_sourcefreeze.json')
    p.add_argument('--pilot-controller-pid',type=int)
    p.add_argument('--resume-existing',action='store_true');a=p.parse_args()
    T.verify_frozen(a.source_freeze,full_input_check=True)
    lock=T.OUT/'full_execution_orchestrator.lock'
    if lock.exists():raise RuntimeError('full execution dispatcher lock already exists; inspect before resuming')
    fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);os.close(fd)
    T.dump(T.OUT/'full_execution_orchestrator_launch.json',dict(pid=os.getpid(),command=sys.argv,created_utc=datetime.now(timezone.utc).isoformat(),
      sourcefreeze_sha256=T.sha(a.source_freeze),pilot_sourcefreeze_sha256=T.PILOT_FREEZE_SHA,
      execution_variant=T.EXECUTION_VARIANT,stages=['adopt_unchanged_pilot','full'],arms=list(T.ARMS),accuracy_gating=False))
    try:
        if a.pilot_controller_pid is not None:adopt_pilot(a.pilot_controller_pid,a.source_freeze)
        else:
            gate=T.read(T.OUT/'pilot_full_execution_transfer.json')
            if gate['status']!='PASS' or gate['full_sourcefreeze_sha256']!=T.sha(a.source_freeze):raise RuntimeError('explicit pilot transfer required')
            P.audit('small',T.PILOT_FREEZE)
        run_stage('full',a.source_freeze,resume_existing=a.resume_existing)
        audit('full',a.source_freeze)
        T.dump(T.OUT/'all_final_weights_complete.json',dict(status='COMPLETE',sourcefreeze_sha256=T.sha(a.source_freeze),
          sourcefreeze_path=str(a.source_freeze.resolve()),pilot_sourcefreeze_sha256=T.PILOT_FREEZE_SHA,
          pilot_audit_sha256=T.sha(T.OUT/'audits/pilot.json'),execution_variant=T.EXECUTION_VARIANT,
          arms=list(T.ARMS),supervised_outputs_per_arm=6144000,total_supervised_outputs=30720000,
          fixed_last_weights=True,no_new_zgz_scores_yet=True,final_weights={arm:str(T.OUT/'train/full'/arm/'final_weights.ckpt') for arm in T.ARMS}))
    except BaseException as exc:
        T.dump(T.OUT/'full_execution_orchestrator_failure.json',dict(status='FAILED',error=repr(exc),created_utc=datetime.now(timezone.utc).isoformat(),
          no_full_accuracyscoring=True,old_negative_results_unchanged=True))
        raise

if __name__=='__main__':main()
