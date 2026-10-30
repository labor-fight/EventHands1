"""Run all six unchanged debug gate checks only after each full-row process exits."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/xkf/fusion_20261015_r3/direct_finetune/incremental_finetune'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def main():
 sys.path[:0]=[str(ROOT),str(ROOT/'model')]
 from tools.xkf_fusion.make_row_incremental import POLICY
 planpath=BASE/'fullrow_execution_plan.json';rowplan=json.loads(planpath.read_text())
 jobs={f'{j["arm"]}_s{j["seed"]}':j for j in rowplan['jobs']}
 if set(jobs)!={f'{a}_s{s}' for a in ('raw','aug') for s in (3407,3408,3409)} or {j['gpu'] for j in jobs.values()}!=set(range(2,8)):raise ValueError('exact six fixed endpoints/GPU mapping required')
 out=BASE/'original_gates';out.mkdir(exist_ok=False)
 policy=out/'policy.json';write(policy,POLICY)
 script=ROOT/'tools/xkf_fusion/debug_gates_incremental.py'
 source={str(p.resolve()):sha(p) for p in (Path(__file__),script,planpath,policy)}
 write(out/'execution_plan.json',{'formal':False,'automatic_full_training':False,'policy':POLICY,'source_sha256':source,'jobs':jobs,'release_rule':'fullrow_round/progress completed matching arm/seed/GPU returncode0 and output SHA; actual112+Iprobe+fullrow+latency already exited','scope':'unchanged G1-G4 formulas on debug2600,2590 one-step packets,65 starts; original G3 short denominator remains unassessable, not formal acceptance'})
 def execute(key,job,release):
  if any(sha(p)!=v for p,v in source.items()):raise ValueError('gate orchestration source/input changed')
  if any(release[k]!=job[k] for k in ('arm','seed','gpu','output')) or release['returncode']!=0 or release['output_sha256']!=sha(job['output']):raise ValueError('invalid full-row release identity')
  run=BASE/key;output=out/f'{key}.json';log=out/f'{key}.log'
  if output.exists() or output.with_suffix('.npz').exists() or log.exists():raise FileExistsError(key)
  command=[sys.executable,'-u',str(script),'--config',str(run/'config_resolved.yaml'),'--checkpoint',str(run/f'xkf_ionly_{key}-step=2600.ckpt'),'--policy',str(policy),'--output',str(output),'--main-row',job['output'],'--device','cuda:0','--batch-size','1']
  env={**os.environ,'CUDA_VISIBLE_DEVICES':str(job['gpu']),'OMP_NUM_THREADS':'4'};before=time.monotonic()
  with log.open('w') as stream:result=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
  record={'arm':job['arm'],'seed':job['seed'],'gpu':job['gpu'],'returncode':result.returncode,'elapsed_seconds':time.monotonic()-before,'output':str(output),'log':str(log),'fullrow_release':release}
  if result.returncode==0:
   value=json.loads(output.read_text())
   if value['formal'] is not False or value['identity']['global_step']!=2600 or value['policy']!=POLICY:raise ValueError('wrong gate result identity')
   record.update(output_sha256=sha(output),gates={k:v['status'] for k,v in value['gates'].items()},numeric_status=value['numeric_status'])
  return record
 started=time.monotonic();running={};done={};launched=set()
 with ThreadPoolExecutor(max_workers=6) as pool:
  while len(done)<6:
   releasepath=BASE/'fullrow_round/progress.json';releases=json.loads(releasepath.read_text()).get('completed',{}) if releasepath.exists() else {}
   for key,job in jobs.items():
    if key in launched or key not in releases:continue
    if releases[key]['returncode']!=0:raise RuntimeError('full row failed; gate not launched: '+key)
    launched.add(key);running[key]=pool.submit(execute,key,job,releases[key]);print(json.dumps({'event':'start_original_gates','run':key,'gpu':job['gpu']}),flush=True)
   for key in list(running):
    if running[key].done():done[key]=running.pop(key).result();print(json.dumps({'event':'finish_original_gates','run':key,**done[key]}),flush=True)
   write(out/'progress.json',{'running':sorted(running),'pending':sorted(set(jobs)-launched),'completed':done,'elapsed_seconds':time.monotonic()-started,'automatic_full_training':False})
   if len(done)==6:break
   if time.monotonic()-started>10800:raise TimeoutError('waiting for fixed six endpoints/full rows')
   time.sleep(5)
 if any(sha(p)!=v for p,v in source.items()):raise ValueError('gate source/input changed')
 write(out/'execution_result.json',{'status':'COMPLETE' if all(v['returncode']==0 for v in done.values()) else 'EVALUATION_ERROR','formal':False,'automatic_full_training':False,'jobs':done,'source_sha256':source})
if __name__=='__main__':main()
