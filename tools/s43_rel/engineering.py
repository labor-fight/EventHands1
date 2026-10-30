"""Actual CUDA engineering profile; no optimizer and no SELECT/test metrics."""
import argparse
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
import torch
from .core import RelConfig
from .geometry import GeometryConfig
from .data import ContinuousRunLanes
from .model import load_relation_model
from .training import RelationRuntime, set_training_mode
from .protocol import write_json


def run(config, output, batch=8, rounds=5):
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    model,cfg=load_relation_model(config,device="cuda:0")
    set_training_mode(model,True)
    lanes=ContinuousRunLanes(cfg["REL"]["DATA_ROOT"],subjects=("ch",),batch_size=batch,seed=3407)
    runtime=RelationRuntime(model,RelConfig(**cfg["MEMORY"]),GeometryConfig(**cfg["GEOMETRY"]),batch)
    torch.cuda.reset_peak_memory_stats()
    rows=[]
    for step in range(rounds):
        torch.cuda.synchronize();started=time.perf_counter()
        items=lanes.next_batch()
        source=time.perf_counter()
        reps,previous,beta,camera,snaps=runtime.inputs(items)
        torch.cuda.synchronize();prepared=time.perf_counter()
        with torch.autocast("cuda",dtype=torch.bfloat16):
            pred=model(reps,previous,beta,camera)
            target=torch.as_tensor(lanes.targets(items),device=previous.device)
            loss,_=model.loss(pred,target,beta)
            objective=loss.clamp_min(1e-12).log10() if model.base.log10_loss else loss
        objective.backward()
        torch.cuda.synchronize();network=time.perf_counter()
        runtime.commit(items,pred,snaps)
        torch.cuda.synchronize();stop=time.perf_counter()
        if not bool(torch.isfinite(pred).all() and torch.isfinite(objective)):
            raise ArithmeticError("nonfinite actual CUDA smoke")
        rows.append(dict(round=step,outputs=batch,events=sum(len(i["events"]) for i in items),
                         loader_seconds=source-started,prep_seconds=prepared-source,
                         forward_backward_seconds=network-prepared,commit_seconds=stop-network,
                         service_seconds=stop-started))
        model.zero_grad(set_to_none=True)
        print(json.dumps(rows[-1]),flush=True)
    hot=rows[1:]
    report=dict(scope="Actual CUDA engineering, zero optimizer updates, ch only, not final inference latency or task evaluation",
                physical_cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"),
                batch=batch,rounds=rounds,rows=rows,params=model.counts(),
                hot_service_seconds=float(np.mean([r["service_seconds"] for r in hot])),
                hot_outputs_per_second=sum(r["outputs"] for r in hot)/sum(r["service_seconds"] for r in hot),
                cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(),
                process_maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                runtime_timing=runtime.timing,optimizer_updates=0)
    write_json(output,report)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--config",required=True)
    p.add_argument("--out",required=True);p.add_argument("--batch",type=int,default=8)
    p.add_argument("--rounds",type=int,default=5)
    a=p.parse_args();run(a.config,a.out,a.batch,a.rounds)
