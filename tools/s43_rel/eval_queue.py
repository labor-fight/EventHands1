"""Sequential recordings on one assigned GPU, parallel arm queues across GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .protocol import ROOT,OUT,write_json
from .orchestrate import ARMS,PYTHON

def worker(tag,arm,recordings,hop,stage,config_dir):
    jobs=[]
    for recording in recordings:
        prefix=OUT/"eval"/tag/f"{arm}_{recording}_h{hop}"
        log=OUT/"logs"/f"eval_{tag}_{arm}_{recording}_h{hop}.log"
        command=[PYTHON,"-m","tools.s43_rel.replay","--config",
                 str(Path(config_dir)/f"{arm}.yaml"),"--recording",recording,
                 "--hop",str(hop),"--out",str(prefix),"--device","cuda:0","--threads","2",
                 "--duplicate-query"]
        if stage!="canonical":
            command+=["--checkpoint",str(OUT/"train"/stage/arm/"last.ckpt")]
        log.parent.mkdir(parents=True,exist_ok=True)
        t=time.time()
        with log.open("w") as f:
            result=subprocess.run(command,cwd=str(ROOT),stdout=f,stderr=subprocess.STDOUT)
        jobs.append(dict(recording=recording,returncode=result.returncode,
                         seconds=time.time()-t,out=str(prefix),command=command))
        if result.returncode:
            write_json(OUT/"eval"/tag/f"{arm}_queue_failed.json",jobs)
            raise SystemExit(result.returncode)
    write_json(OUT/"eval"/tag/f"{arm}_queue_complete.json",jobs)

def launch(tag,stage,recordings,hop,config_dir):
    path=OUT/f"eval_jobs_{tag}.json"
    if path.exists():raise FileExistsError(path)
    jobs=[]
    for arm,gpu in zip(ARMS,(4,5,6,7)):
        cmd=[PYTHON,"-m","tools.s43_rel.eval_queue","worker","--tag",tag,"--stage",stage,
             "--arm",arm,"--recordings",*recordings,"--hop",str(hop),"--config-dir",str(Path(config_dir).resolve())]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS="1",
                 MKL_NUM_THREADS="1",OMP_NUM_THREADS="2")
        log=OUT/"logs"/f"evalqueue_{tag}_{arm}.log";log.parent.mkdir(parents=True,exist_ok=True)
        with log.open("w") as f:
            p=subprocess.Popen(cmd,cwd=str(ROOT),env=env,stdout=f,stderr=subprocess.STDOUT,
                               start_new_session=True)
        jobs.append(dict(arm=arm,gpu=gpu,pid=p.pid,command=cmd,log=str(log)))
    write_json(path,jobs);print(json.dumps(jobs,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["launch","worker"])
    p.add_argument("--tag",required=True);p.add_argument("--stage",required=True)
    p.add_argument("--arm",choices=ARMS);p.add_argument("--config-dir",default="configs/s43_rel");p.add_argument("--recordings",nargs="+",required=True)
    p.add_argument("--hop",type=int,default=50);a=p.parse_args()
    if a.command=="launch":launch(a.tag,a.stage,a.recordings,a.hop,a.config_dir)
    else:worker(a.tag,a.arm,a.recordings,a.hop,a.stage,a.config_dir)
