"""Fixed Stage5 evaluation collector; all predictions precede target scoring.

Wait for five audited full weights, persist the fixed120 traces, then score.
Resources execute sequentially on an otherwise idle physical GPU, never in
parallel with training or this collector's accuracy workers.
"""
from __future__ import annotations
import argparse,concurrent.futures,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from . import s43_field_replay as R
from tools.s43_rel.gpu_memory_observer import MemoryObserver


def verify_sources(freeze):
    for path,expected in freeze['dependencies'].items():
        if R.sha(path)!=expected:raise RuntimeError('frozen evaluation source changed: '+path)


def query_gpu_processes(gpu):
    if gpu not in (4,5,6,7):raise ValueError('physicalGPU4–7 only')
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader','-i',str(gpu)],text=True,timeout=15).strip()
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    pids=sorted(int(row.split(',')[1].strip()) for row in value.splitlines() if row.split(',')[0].strip()==uuid)
    return dict(physical_gpu=gpu,gpu_uuid=uuid,pids=pids)


def wait_resource_idle(gpu,out,name,*,query=None,sleep=None,interval_seconds=5):
    """Read-only queue on the fixed GPU; no process killing or device picking."""
    if not 0<interval_seconds<=60:raise ValueError('resource wait interval must be<=60s')
    query=query or query_gpu_processes;sleep=sleep or time.sleep
    folder=Path(out)/'resource_wait';folder.mkdir(parents=True,exist_ok=True);started=time.time();checks=0;last=None
    while True:
        state=query(gpu);checks+=1;foreign=[p for p in state['pids'] if p!=os.getpid()]
        record=dict(state,name=name,status='WAITING' if foreign else 'IDLE',foreign_pids=foreign,
            checks=checks,wait_seconds=time.time()-started,interval_seconds=interval_seconds,read_only=True)
        R.dump(folder/(name+'.json'),record)
        with (folder/(name+'.jsonl')).open('a') as f:f.write(json.dumps(record)+'\n')
        if foreign!=last:print(json.dumps(dict(resource_wait=name,status=record['status'],physical_gpu=gpu,foreign_pids=foreign)),flush=True);last=foreign
        if not foreign:return record
        sleep(interval_seconds)


def collector_freeze(out):
    out=Path(out);path=out/'actual_evaluation_sourcefreeze.json'
    if path.exists():
        d=R.read(path);verify_sources(d);return d
    names=['s43_field_replay.py','s43_field_native.py','s43_field_perturb.py','s43_field_benchmark.py','s43_field_operations.py','s43_field_collect.py']
    files=[Path(__file__).with_name(n) for n in names]+[R.PROTOCOL,
        R.BASE/'stage5_s43_full_protocol.json',R.BASE/'stage5_box_time_addendum.json',
        R.BASE/'stage5_numeric_policy_clarification.json',R.BASE/'stage5_reacquisition_R1_protocol.json',
        R.TRAIN/'engineering/actual_sourcefreeze.json',R.TRAIN/'engineering/full_execution_sourcefreeze.json',
        R.ENGINEERING/'implementation_extension.json',R.ENGINEERING/'receipt.json',R.ROOT/'tools/s43_rel/gpu_memory_observer.py']
    files.append(out/'evaluation_plan.json')
    dependencies=dict(R.code_dependencies());dependencies.update({str(p.resolve()):R.sha(p) for p in files})
    d=dict(schema='s43_stage5_actual_evaluation_sourcefreeze_v1',created_ns=time.time_ns(),dependencies=dependencies,
        protocol_sha256=R.PROTOCOL_SHA,command=sys.argv,scientific_scores_seen=False,
        no_GT_pose_access_before_all120_predictions=True,formal_resources_after_training_and_accuracy_workers_finished=True)
    R.dump(path,d);return d


def run_process(command,name,gpu,out,*,observe_memory=False):
    out=Path(out);log=out/'logs'/(name+'.log');receipt=out/'executions'/(name+'.json')
    if receipt.exists():
        previous=R.read(receipt)
        if previous['code']==0 and previous['command']==command and previous.get('resource_exclusivity_valid',True):return previous
        raise RuntimeError('failed prior execution retained; inspect before explicit fresh attempt: '+str(receipt))
    log.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),CUDA_DEVICE_ORDER='PCI_BUS_ID',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2')
    started=time.time();observer=MemoryObserver(gpu) if observe_memory else None
    foreign_observations=[];last_process_check=float('-inf')
    with log.open('x') as f:
        process=subprocess.Popen(command,cwd=R.ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
        while process.poll() is None:
            if observer:
                observer.sample()
                if time.monotonic()-last_process_check>=.5:
                    last_process_check=time.monotonic();state=query_gpu_processes(gpu)
                    foreign=[pid for pid in state['pids'] if pid not in (process.pid,os.getpid())]
                    if foreign:foreign_observations.append(dict(observed_ns=time.time_ns(),foreign_pids=foreign,gpu_uuid=state['gpu_uuid']))
            time.sleep(.05)
        code=process.returncode
    record=dict(command=command,name=name,code=code,physical_gpu=gpu,log=str(log),elapsed_seconds=time.time()-started)
    if observer:
        observer.sample();record['NVML_device_memory_observation']=observer.close()
        record.update(resource_exclusivity_valid=not foreign_observations,foreign_process_observations=foreign_observations,
            process_occupancy_sampling_period_seconds=.5,foreign_process_policy='observedinterference invalidatesresource; do notkillothers or silentlyretry/change workload')
    R.dump(receipt,record);print(json.dumps(dict(name=name,code=code,seconds=record['elapsed_seconds'])),flush=True)
    if code:raise RuntimeError('execution failed; original log/receipt preserved: '+str(log))
    if foreign_observations:raise RuntimeError('resourceGPU became shared; outputs/logs preserved but performance invalid: '+str(receipt))
    return record


def export_explicit_known_shape_inputs(plan,out,root):
    """An explicitly GT-assisted input provider, never called by beta0 replay."""
    R.validate_plan(plan)
    if len(R.validate_cohort(plan,profiles={'primary'},root=root))!=36:raise RuntimeError('complete unique36 beta0 first')
    from tools.s43_rel.mean_shape_replay import MeanShapeInputReader
    folder=Path(out)/'provided_known_beta';folder.mkdir(parents=True,exist_ok=True)
    for rec in ('zgz_local','zgz_global'):
        dest=folder/(rec+'.json')
        if dest.exists():continue
        reader=MeanShapeInputReader(rec,root);aux=Path(str(reader.base)+'_aux.npz')
        with np.load(aux,allow_pickle=False) as z:beta=np.asarray(z['betas'],np.float32).reshape(10)
        R.dump(dest,dict(condition='explicit_known_beta_GT_assisted_diagnostic',external_beta10=beta.tolist(),
            recording=rec,source_file=str(aux),source_sha256=R.sha(aux),source_member='betas',
            extracted_after_all36_beta0_predictions_ns=time.time_ns(),GT_pose_opened=False,
            interpretation='externallyprovidedactualshape; conditionaldiagnostic, not beta0 deployment or automaticupperbound'))
    return folder


def predict_jobs(jobs,out,freeze,*,root=R.DATA_ROOT,known_folder=None):
    pending=iter(jobs)
    def worker(gpu):
        records=[]
        while True:
            # next(iterator) is atomic under CPython's GIL; independent jobs
            # are dynamically assigned, with no per-GPU imbalance assumption.
            try:job=next(pending)
            except StopIteration:return records
            verify_sources(freeze)
            path=Path(job['out']);summary=path.with_suffix('.json')
            if summary.exists():
                d=R.read(summary)
                if d.get('status')!='COMPLETE' or R.sha(path.with_suffix('.npz'))!=d.get('prediction_sha256'):raise RuntimeError('prior trace incomplete; preserve and inspect '+str(path))
                continue
            command=[sys.executable,'-m','tools.s43_revisit.s43_field_replay','predict','--arm',job['arm'],'--recording',job['recording'],
                '--out',job['out'],'--root',str(root),'--device','cuda:0','--threads','2','--hop',str(job['hop']),
                '--commit-ms',str(job['commit_ms']),'--perturbation',job['perturbation']]
            if job['checkpoint'] is not None:command+=['--checkpoint',job['checkpoint']]
            if job['known_beta']:
                if known_folder is None:raise RuntimeError('explicitknownshape provider required')
                command+=['--provided-beta',str(known_folder/(job['recording']+'.json'))]
            name='_'.join((job['profile'],job['arm'],job['recording']))
            records.append(run_process(command,name,gpu,out));verify_sources(freeze)
        return records
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        futures=[pool.submit(worker,gpu) for gpu in (4,5,6,7)]
        return [row for f in futures for row in f.result()]


def collect(out,*,train_root=R.TRAIN,root=R.DATA_ROOT,wait=False,phase='all',resource_gpu=4):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if R.sha(R.PROTOCOL)!=R.PROTOCOL_SHA:raise RuntimeError('frozen evaluation protocol identity')
    if wait:
        while not (Path(train_root)/'all_final_weights_complete.json').exists():time.sleep(5)
        # The wait may span a later engineering/source stage. Start a fresh
        # interpreter so the source freeze cannot hash newer disk files while
        # executing a stale module object imported before the wait.
        command=[sys.executable,'-m','tools.s43_revisit.s43_field_collect','--out',str(out),
            '--train-root',str(train_root),'--root',str(root),'--phase',phase,'--resource-gpu',str(resource_gpu)]
        return subprocess.run(command,cwd=R.ROOT,check=True).returncode
    gate=R.all_weights_gate(train_root)
    plan_path=out/'evaluation_plan.json'
    plan=R.read(plan_path) if plan_path.exists() else R.evaluation_plan(out,train_root)
    R.validate_plan(plan,gate)
    freeze=collector_freeze(out)
    if phase in ('all','accuracy'):
        predict_jobs([j for j in plan['jobs'] if j['profile']=='primary'],out,freeze,root=root)
        known=export_explicit_known_shape_inputs(plan,out,root)
        predict_jobs([j for j in plan['jobs'] if j['profile']!='primary'],out,freeze,root=root,known_folder=known)
        loaded=R.validate_cohort(plan,root=root)
        R.dump(out/'all_predictions_complete.json',dict(status='COMPLETE',total_traces=len(loaded),primary36_complete=True,
            completed_ns=time.time_ns(),sourcefreeze_sha256=R.sha(out/'actual_evaluation_sourcefreeze.json'),
            predictions={j['out']:R.sha(Path(j['out']).with_suffix('.npz')) for j in plan['jobs']},GT_pose_accessed=False))
        # Only now may the isolated score process open target poses/shape.
        command=[sys.executable,'-m','tools.s43_revisit.s43_field_replay','score','--plan',str(plan_path),'--out',str(out/'report'),'--root',str(root),'--train-root',str(train_root)]
        run_process(command,'saved_predictions_score',4,out)
    if phase in ('all','resources'):
        if resource_gpu not in (4,5,6,7):raise ValueError('physical GPU4–7 only')
        from .s43_field_benchmark import prepare_resource_inputs
        source_folder=out/'resource_inputs'
        if not (source_folder/'native.json').exists():prepare_resource_inputs(source_folder,root)
        jobs=[]
        for arm in R.MODELS:
            for source in ('native','noise30'):
                for hop,commit in ((25,25),(50,50),(100,100),(25,50)):
                    jobs.append((arm,source,hop,commit,arm in ('canonical','s43')))
                if arm in ('canonical','s43'):jobs.append((arm,source,50,50,False))
        results=[]
        for arm,source,hop,commit,native in jobs:
            verify_sources(freeze);kind='native_s43' if native else 'shared_audit' if arm in ('s43','canonical') else 'field'
            name=f'resource_{arm}_{source}_q{hop}_c{commit}_{kind}';dest=out/'resources'/(name+'.json')
            wait_resource_idle(resource_gpu,out,name)
            command=[sys.executable,'-m','tools.s43_revisit.s43_field_benchmark','run','--arm',arm,'--source',str(source_folder/source),
                '--out',str(dest),'--device','cuda:0','--threads','2','--hop',str(hop),'--commit-ms',str(commit),'--train-root',str(train_root)]
            if arm!='canonical':command+=['--checkpoint',gate['checkpoints'][arm]]
            if native:command+=['--native-s43']
            execution=run_process(command,name,resource_gpu,out,observe_memory=True)
            result=R.read(dest);results.append(dict(name=name,status=result['status'],resource=str(dest),sha256=R.sha(dest),
                execution=execution,NVML_device_memory_observation=execution.get('NVML_device_memory_observation')))
        R.dump(out/'resources_complete.json',dict(status='COMPLETE',fixed_workloads=len(results),failed_workloads=sum(r['status']!='COMPLETE' for r in results),results=results,
            negative_workload_policy='failedoutputs preserved; no wholeflowperformance claim frompartial traces',sourcefreeze_sha256=R.sha(out/'actual_evaluation_sourcefreeze.json')))
    verify_sources(freeze)
    if (out/'all_predictions_complete.json').exists() and (out/'report/results.json').exists() and (out/'resources_complete.json').exists():
        R.dump(out/'evaluation_complete.json',dict(status='COMPLETE',completed_ns=time.time_ns(),allfive_full_weights_gate=gate,
            report_sha256=R.sha(out/'report/results.json'),resources_sha256=R.sha(out/'resources_complete.json'),
            sourcefreeze_sha256=R.sha(out/'actual_evaluation_sourcefreeze.json'),source_files_unchanged=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--train-root',default=str(R.TRAIN));p.add_argument('--root',default=str(R.DATA_ROOT))
    p.add_argument('--wait',action='store_true');p.add_argument('--phase',choices=('all','accuracy','resources'),default='all');p.add_argument('--resource-gpu',type=int,default=4)
    collect(**vars(p.parse_args()))


if __name__=='__main__':main()
