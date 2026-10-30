"""No-label parameter-gradient parity companion to the readout audit."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from .field_core import TimeFieldMemory
from .field_tensor import tensorize as old
from .field_tensor_fast import tensorize_memories
from .field_tensor_fast_engineering import memory_hash,write,sha,ROOT
from .s43_field_training_R1 import build_model
from .s43_field_thread_audit import conditions,tensor_hash
from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cpu');a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    paths=[Path(__file__),ROOT/'tools/s43_revisit/field_tensor_fast.py',ROOT/'tools/s43_revisit/field_tensor.py',
           ROOT/'tools/s43_revisit/s43_field_model_reacquire.py',ROOT/'tools/s43_revisit/s43_field_training_R1.py']
    sources={str(x):sha(x) for x in paths};write(a.out/'sourcefreeze.json',dict(sources=sources,task_scores=False,optimizer_steps=0))
    with InferenceAccessGuard() as guard:
        reader=MeanShapeInputReader('ch_local');packet=reader.packet(0,50)
        ms=[TimeFieldMemory(seq_id=str(i)) for i in range(8)]
        for m in ms:m.ingest(packet['events'],50,0,source_range=packet['source_range'])
        before=[memory_hash(m) for m in ms]
        model,_=build_model('field_static',a.device);model.eval()
        states={k:tensor_hash(v) for k,v in model.state_dict().items()}
        prev,beta,_,ref,cut,hist=conditions(8)
        K=torch.from_numpy(reader.K_cam)[None].repeat(8,1,1)
        prev,beta,K=[x.to(a.device) for x in (prev,beta,K)]
        params=list(model.named_parameters());reference=None;rows=[]
        for label,fast in (('old_a',False),('direct_a',True),('old_b',False),('direct_b',True)):
            rng=torch.get_rng_state().clone();cuda_rng=torch.cuda.get_rng_state().clone() if torch.device(a.device).type=='cuda' else None
            f=tensorize_memories(ms,device=a.device) if fast else old([m.readout() for m in ms],a.device)
            pred=model(f.global40,prev,beta_input=beta,K_source=K,ref_ms=ref,cutoff_ms=cut,histories=hist)
            # A fixed public scalar functional tests backward equivalence;
            # it is not a supervised loss or a task score.
            objective=pred.square().mean()
            gradients=torch.autograd.grad(objective,[p for _,p in params],allow_unused=True)
            gradients=[None if g is None else g.detach().cpu() for g in gradients]
            prediction=pred.detach().cpu()
            if reference is None:reference=(prediction,gradients)
            pmax=float((prediction-reference[0]).abs().max());gmax=0.;nonzero=0;used=0;worst=[]
            for (name,_),g,r in zip(params,gradients,reference[1]):
                assert (g is None)==(r is None),name
                if g is not None:
                    assert torch.isfinite(g).all(),name
                    used+=1;nonzero+=int(torch.count_nonzero(g).item()>0)
                    diff=float((g-r).abs().max());gmax=max(gmax,diff)
                    if diff:worst.append(dict(name=name,maxabs=diff,reference_max=float(r.abs().max())))
            assert pmax==0 and torch.equal(rng,torch.get_rng_state())
            if cuda_rng is not None:assert torch.equal(cuda_rng,torch.cuda.get_rng_state())
            rows.append(dict(label=label,maxprediction=pmax,maxparametergradient=gmax,
                used_parameter_tensors=used,nonzero_parameter_tensors=nonzero,RNG_unchanged=True,
                nonbitwise_parameters=worst))
            del pred,objective,gradients,f
        assert before==[memory_hash(m) for m in ms]
        assert states=={k:tensor_hash(v) for k,v in model.state_dict().items()}
        model.close_geometry_workers()
    assert sources=={str(x):sha(x) for x in paths}
    result=dict(status='MEASURED_NUMERICAL_VARIATION' if any(r['maxparametergradient'] for r in rows) else 'PASS_BITWISE',scope='B8 actualch0:50ms publicbeta0/publiccontext R1 evalmode parametergradient; noGT/objectivesupervision/optimizer',
        rows=rows,sourcefreeze_sha256=sha(a.out/'sourcefreeze.json'),state_source_buffers_weights_unchanged=True,
        inference_access_guard=guard.receipt(),task_scores_read=False,optimizer_steps=0,shared_host=True,
        CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    write(a.out/'result.json',result);print(json.dumps(result))


if __name__=='__main__':main()
