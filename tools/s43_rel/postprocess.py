"""Fixed post-training service measurements and audits, without tuning."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .protocol import ROOT,OUT,write_json
from .orchestrate import ARMS,PYTHON

def run(command,name,gpu="4",allow_failure=False):
    folder=OUT/"postprocess";folder.mkdir(parents=True,exist_ok=True)
    log=folder/f"{name}.log"
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OPENBLAS_NUM_THREADS="1",
             MKL_NUM_THREADS="1",OMP_NUM_THREADS="2")
    t=time.time()
    with log.open("w") as f:
        code=subprocess.run(command,cwd=str(ROOT),env=env,stdout=f,stderr=subprocess.STDOUT).returncode
    record=dict(name=name,command=command,code=code,seconds=time.time()-t,
                CUDA_VISIBLE_DEVICES=gpu,log=str(log))
    write_json(folder/f"{name}.execution.json",record)
    print(json.dumps(record),flush=True)
    if code and not allow_failure:raise RuntimeError(f"{name} failed; inspect {log}")
    return record

def main():
    p=argparse.ArgumentParser();p.add_argument("--wait",action="store_true");a=p.parse_args()
    required=[OUT/f"final_pipeline_{arm}_complete.json" for arm in ARMS]
    required+=[OUT/"eval/canonical_final_h50/s43_queue_complete.json"]
    if a.wait:
        while not all(path.exists() for path in required):
            time.sleep(5)
    if not all(path.exists() for path in required):raise RuntimeError("Final streams incomplete")
    tasks=[]
    # Same 3.2s source prefix and same common warmup, cadence varies separately.
    for arm in ARMS:
        for hop in (25,50,100):
            destination=OUT/"benchmark/final/actual"/f"{arm}_h{hop}.json"
            command=[PYTHON,"-m","tools.s43_rel.benchmark","--config",
              str(ROOT/"configs/s43_rel/r1"/f"{arm}.yaml"),"--checkpoint",
              str(OUT/"train/full_r1"/arm/"step=6000.ckpt"),"--recording","ch_local",
              "--hop",str(hop),"--queries",str(3200//hop),"--out",str(destination),
              "--device","cuda:0","--threads","2"]
            tasks.append(run(command,f"benchmark_actual_{arm}_h{hop}"))
    command=[PYTHON,"-m","tools.s43_rel.benchmark","--config",
      str(ROOT/"configs/s43_rel/r1/s43.yaml"),"--recording","ch_local","--hop","50",
      "--queries","64","--out",str(OUT/"benchmark/final/actual/canonical_h50.json"),
      "--device","cuda:0","--threads","2"]
    tasks.append(run(command,"benchmark_actual_canonical_h50"))
    # Synthetic rate stress is explicitly separate; a failed workload is a result.
    for arm in ARMS:
        for rate in (10000,50000,150000):
            for hop in (25,50,100):
                destination=OUT/"benchmark/final/synthetic"/f"{arm}_n{rate}_h{hop}.json"
                command=[PYTHON,"-m","tools.s43_rel.benchmark","--config",
                  str(ROOT/"configs/s43_rel/r1"/f"{arm}.yaml"),"--checkpoint",
                  str(OUT/"train/full_r1"/arm/"step=6000.ckpt"),"--hop",str(hop),
                  "--queries",str(400//hop),"--synthetic-events-per50",str(rate),
                  "--out",str(destination),"--device","cuda:0","--threads","2"]
                rec=run(command,f"benchmark_synthetic_{arm}_n{rate}_h{hop}",allow_failure=True)
                tasks.append(rec)
                if rec["code"]:
                    write_json(destination.with_name(destination.stem+"_failed.json"),
                               dict(rec,scope="Failed synthetic full-service stress; not a hand accuracy test."))
    command=[PYTHON,"-m","tools.s43_rel.robustness","--configs-dir",
      str(ROOT/"configs/s43_rel/r1"),"--checkpoint-dir",str(OUT/"train/full_r1"),
      "--nominal-eval-dir",str(OUT/"eval/full_final_h50"),"--recordings","zgz_local","zgz_global",
      "--out",str(OUT/"robustness/full_final"),"--device","cpu"]
    tasks.append(run(command,"robustness_final",gpu=""))
    command=[PYTHON,str(OUT/"audits/r1_training_online.py"),"--checkpoint",
             str(OUT/"train/full_r1/full/last.ckpt"),"--out",
             str(OUT/"audits/full_training_online.json")]
    tasks.append(run(command,"trained_prefix_final",gpu=""))
    tasks.append(run([PYTHON,"-m","tools.s43_rel.full_training_audit"],"full_training_audit",gpu=""))
    write_json(OUT/"postprocess_complete.json",dict(complete=True,tasks=tasks,
        checkpoint_policy="Fixed final6000 only; weights-only file for deployment resource benchmark."))

if __name__=="__main__":main()
