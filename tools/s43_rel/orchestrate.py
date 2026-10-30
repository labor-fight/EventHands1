"""Local four-GPU stage launcher; it never terminates unrelated jobs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from .protocol import ROOT, OUT, sha, write_json

ARMS=("s43","enhanced","no_order","full")
PYTHON="/data1/lyq/miniconda3/envs/EventHandsTrain/bin/python"


def authorize(stage,config_dir,tag):
    paths=[OUT/"mechanisms/e12/summary.json", OUT/"mechanisms/e3/summary.json"]
    for path in paths:
        if not path.exists():raise RuntimeError(f"Mechanism experiment incomplete: {path}")
        json.loads(path.read_text())
    configs={str((config_dir/f"{arm}.yaml").resolve()):
             sha(config_dir/f"{arm}.yaml") for arm in ARMS}
    auth={"schema":"s43_rel_stage_authorization_v1",
          "protocol_sha256":sha(OUT/"protocol.json"),
          "mechanism_experiments_completed":True,
          "mechanism_results":{str(p):sha(p) for p in paths},
          "small_training_authorized":True,"full_training_authorized":stage=="full",
          "user_authorization":"Current explicit request: run mechanisms, small correction and full paired training.",
          "scientific_success_required_for_execution":False,
          "config_sha256":configs,"at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    path=OUT/f"authorization_{tag}.json"
    write_json(path,auth)
    return path


def launch(stage,tag=None,config_dir="configs/s43_rel"):
    tag=tag or stage
    config_dir=Path(config_dir).resolve()
    authorization=authorize(stage,config_dir,tag)
    statepath=OUT/f"jobs_{tag}.json"
    if statepath.exists():raise FileExistsError(statepath)
    jobs=[]
    for gpu,arm in zip((4,5,6,7),ARMS):
        dest=OUT/"train"/tag/arm
        if dest.exists() and any(dest.iterdir()):raise FileExistsError(dest)
        logpath=OUT/"logs"/f"{tag}_{arm}.log"
        logpath.parent.mkdir(parents=True,exist_ok=True)
        command=[PYTHON,"-m","tools.s43_rel.training","train","--config",
                 str(config_dir/f"{arm}.yaml"),"--stage",stage,
                 "--out-dir",str(dest),"--device","cuda:0","--protocol",
                 str(OUT/"protocol.json"),"--authorization",str(authorization)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OPENBLAS_NUM_THREADS="1",
                 OMP_NUM_THREADS="4",MKL_NUM_THREADS="1")
        with logpath.open("w") as log:
            process=subprocess.Popen(command,cwd=str(ROOT),env=env,stdout=log,
                                     stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append(dict(arm=arm,physical_gpu=gpu,pid=process.pid,command=command,
                         out=str(dest),log=str(logpath),started=time.time()))
    write_json(statepath,dict(stage=stage,tag=tag,jobs=jobs,authorization=str(authorization)))
    print(json.dumps(dict(stage=stage,tag=tag,jobs=jobs),ensure_ascii=False,indent=2))


def status(tag):
    state=json.loads((OUT/f"jobs_{tag}.json").read_text())
    rows=[]
    for job in state["jobs"]:
        complete=Path(job["out"])/"complete.json"
        curve=Path(job["out"])/"curve.jsonl"
        last=json.loads(curve.read_text().splitlines()[-1]) if curve.exists() and curve.stat().st_size else {}
        try:os.kill(job["pid"],0);alive=True
        except ProcessLookupError:alive=False
        rows.append(dict(arm=job["arm"],gpu=job["physical_gpu"],pid=job["pid"],
                         alive=alive,complete=complete.exists(),step=last.get("step"),
                         seconds=last.get("seconds"),loss=last.get("loss"),
                         outputs=last.get("outputs"),sample_digest=last.get("sample_digest")))
    print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("command",choices=["launch","status"])
    p.add_argument("--stage",choices=["small","full"],default="small");p.add_argument("--tag");p.add_argument("--config-dir",default="configs/s43_rel")
    a=p.parse_args()
    if a.command=="launch":launch(a.stage,a.tag,a.config_dir)
    else:status(a.tag or a.stage)
