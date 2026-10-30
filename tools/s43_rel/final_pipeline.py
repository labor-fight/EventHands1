"""Run only fixed final checkpoints after their 6000-step jobs complete."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from .protocol import ROOT,OUT,write_json
from .orchestrate import ARMS,PYTHON

def worker(arm,pid):
    directory=OUT/"train/full_r1"/arm
    while not (directory/"complete.json").exists():
        try:os.kill(pid,0)
        except ProcessLookupError:
            raise RuntimeError(f"Training {arm} exited without its completion receipt")
        time.sleep(5)
    receipt=json.loads((directory/"complete.json").read_text())
    if receipt["completed_steps"]!=6000 or receipt["actual_outputs"]!=192000:
        raise RuntimeError("Refuse incomplete final checkpoint")
    jobs=[]
    cases=[("full_final_h50",r,50,None) for r in ("ycy_local","ycy_global","zgz_local","zgz_global")]
    for hop in (25,100):
        cases += [(f"full_final_h{hop}",r,hop,None) for r in ("zgz_local","zgz_global")]
    cases += [("full_final_h25_fixed50",r,25,50) for r in ("zgz_local","zgz_global")]
    for tag,recording,hop,commit in cases:
        dest=OUT/"eval"/tag/f"{arm}_{recording}_h{hop}"
        log=OUT/"logs"/f"{tag}_{arm}_{recording}.log";log.parent.mkdir(parents=True,exist_ok=True)
        cmd=[PYTHON,"-m","tools.s43_rel.replay","--config",
             str(ROOT/"configs/s43_rel/r1"/f"{arm}.yaml"),
             "--checkpoint",str(directory/"last.ckpt"),"--recording",recording,
             "--hop",str(hop),"--out",str(dest),"--device","cuda:0","--threads","2",
             "--duplicate-query"]
        if commit:cmd+=["--fixed-commit-ms",str(commit)]
        before=time.time()
        with log.open("w") as f:
            code=subprocess.run(cmd,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT).returncode
        jobs.append(dict(tag=tag,recording=recording,hop=hop,commit=commit,
                         code=code,seconds=time.time()-before,out=str(dest),command=cmd))
        if code:
            write_json(OUT/f"final_pipeline_{arm}_failed.json",jobs)
            raise SystemExit(code)
    write_json(OUT/f"final_pipeline_{arm}_complete.json",jobs)

def launch():
    path=OUT/"final_pipeline_jobs.json"
    if path.exists():raise FileExistsError(path)
    train=json.loads((OUT/"jobs_full_r1.json").read_text())
    jobs=[]
    for job in train["jobs"]:
        arm=job["arm"];gpu=job["physical_gpu"]
        cmd=[PYTHON,"-m","tools.s43_rel.final_pipeline","worker","--arm",arm,
             "--train-pid",str(job["pid"])]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS="1",
                 OMP_NUM_THREADS="2",MKL_NUM_THREADS="1")
        log=OUT/"logs"/f"final_pipeline_{arm}.log"
        with log.open("w") as f:
            process=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=f,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append(dict(arm=arm,gpu=gpu,pid=process.pid,command=cmd,log=str(log)))
    write_json(path,jobs)
    print(json.dumps(jobs,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["launch","worker"])
    p.add_argument("--arm",choices=ARMS);p.add_argument("--train-pid",type=int)
    a=p.parse_args()
    if a.command=="launch":launch()
    else:worker(a.arm,a.train_pid)
