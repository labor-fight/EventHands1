"""Automatic fixed-training -> sealed collection -> saved-artifact report chain.

No new scientific choices/retries are made here. This outer orchestration
identity is separate from the already frozen training and evaluation code.
"""
import argparse,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from . import s43_field_full_training as T


def execute(command,name,logdir):
    begin=time.perf_counter();started=datetime.now(timezone.utc).isoformat()
    with (logdir/(name+'.log')).open('wb') as log:
        process=subprocess.Popen(command,cwd=T.ROOT,env=dict(os.environ,PYTHONUTF8='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2',OMP_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT)
        code=process.wait()
    receipt=dict(command=command,pid=process.pid,returncode=code,started_utc=started,
      completed_utc=datetime.now(timezone.utc).isoformat(),wall_seconds=time.perf_counter()-begin)
    T.dump(logdir/(name+'.execution.json'),receipt)
    if code:raise RuntimeError(f'{name} failed with code{code}; preserve artifacts, no automatic scientific rerun')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pilot-controller-pid',type=int,required=True)
    p.add_argument('--source-freeze',type=Path,default=T.OUT/'engineering/full_execution_sourcefreeze.json')
    p.add_argument('--eval-root',type=Path,default=T.BASE_OUT/'s43_full_field_R1_evaluation_v1')
    a=p.parse_args();T.verify_frozen(a.source_freeze)
    logs=T.OUT/'logs/postmaster';logs.mkdir(parents=True,exist_ok=True)
    lock=T.OUT/'postmaster.lock';fd=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644);os.close(fd)
    source_sha=T.sha(__file__)
    T.dump(T.OUT/'postmaster_launch.json',dict(status='RUNNING',pid=os.getpid(),command=sys.argv,
      created_utc=datetime.now(timezone.utc).isoformat(),postmaster_source_sha256=source_sha,
      training_sourcefreeze_sha256=T.sha(a.source_freeze),training_sourcefreeze_path=str(a.source_freeze.resolve()),
      eval_root=str(a.eval_root.resolve()),phases=['fixed_training_and_audit','collect_wait','saved_artifact_report_write_docs'],
      no_scientific_retry=True,collect_source_identity='Final evaluation freeze owns collector identity; collector is imported freshly only after all weights.',
      report_source_identity='Root-owned saved-artifact-only report module source recorded at invocation.'))
    try:
        records=[]
        command=[sys.executable,'-m','tools.s43_revisit.s43_field_full_queue','--pilot-controller-pid',str(a.pilot_controller_pid),'--source-freeze',str(a.source_freeze)]
        records.append(execute(command,'training_and_audit',logs))
        if T.sha(__file__)!=source_sha:raise RuntimeError('postmaster changed during training')
        T.verify_frozen(a.source_freeze)
        gate=T.read(T.OUT/'all_final_weights_complete.json');audit=T.read(T.OUT/'audits/full.json')
        if gate['status']!='COMPLETE' or audit['status']!='PASS' or gate['sourcefreeze_sha256']!=T.sha(a.source_freeze) or audit['sourcefreeze_sha256']!=T.sha(a.source_freeze):raise RuntimeError('fixed training/audit gate')
        if gate['supervised_outputs_per_arm']!=6144000 or tuple(gate['arms'])!=T.ARMS:raise RuntimeError('all5 actual6144000 required')
        command=[sys.executable,'-m','tools.s43_revisit.s43_field_collect','--out',str(a.eval_root),'--train-root',str(T.OUT),'--wait']
        records.append(execute(command,'sealed_collection',logs))
        if T.read(a.eval_root/'evaluation_complete.json')['status']!='COMPLETE':raise RuntimeError('strict evaluation cohort incomplete')
        report=T.ROOT/'tools/s43_revisit/report_full_field.py'
        if not report.exists():raise RuntimeError('Root saved-artifact report module is missing; retain completed training/collection')
        T.dump(logs/'report_source_identity.json',dict(path=str(report),sha256=T.sha(report),GT_reading=False,source='Root report module; no training/scoring rerun'))
        command=[sys.executable,'-m','tools.s43_revisit.report_full_field','--eval-root',str(a.eval_root),'--train-root',str(T.OUT),'--write-docs']
        records.append(execute(command,'saved_artifact_report',logs))
        T.dump(T.OUT/'training_collection_reporting_complete.json',dict(status='COMPLETE',completed_utc=datetime.now(timezone.utc).isoformat(),
          training_sourcefreeze_sha256=T.sha(a.source_freeze),postmaster_source_sha256=source_sha,eval_root=str(a.eval_root),
          all_final_weights_sha256=T.sha(T.OUT/'all_final_weights_complete.json'),evaluation_complete_sha256=T.sha(a.eval_root/'evaluation_complete.json'),executions=records,
          no_scientific_retry=True,old_negative_results_preserved=True))
    except BaseException as exc:
        T.dump(T.OUT/'postmaster_failure.json',dict(status='FAILED',error=repr(exc),created_utc=datetime.now(timezone.utc).isoformat(),
          training_sourcefreeze_sha256=T.sha(a.source_freeze),postmaster_source_sha256=source_sha,no_automatic_scientific_retry=True,
          completed_artifacts_preserved=True))
        raise

if __name__=='__main__':main()
