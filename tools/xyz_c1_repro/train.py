#!/usr/bin/env python3
"""Explicit deterministic wrapper around the unchanged semkine.train.main.

Lightning 1.9 re-executes this file for other DDP ranks. Wrapper choices are
exported before calling the original entry so the same policy applies there.
No resume is offered: the original checkpoint does not restore sampler state.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
import sys,time

ENV={"CUBLAS_WORKSPACE_CONFIG":":4096:8","PYTHONHASHSEED":"3407",
     "OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"1",
     "CUDA_DEVICE_ORDER":"PCI_BUS_ID","NCCL_ALGO":"Ring","NCCL_PROTO":"Simple",
     "NCCL_MIN_NCHANNELS":"1","NCCL_MAX_NCHANNELS":"1","NCCL_P2P_DISABLE":"1",
     "NCCL_COLLNET_ENABLE":"0","NCCL_NVLS_ENABLE":"0","NCCL_ASYNC_ERROR_HANDLING":"1",
     "PYTHONDONTWRITEBYTECODE":"1","PL_INTER_BATCH_PARALLELISM":"0"}
old_hash_seed=os.environ.get("PYTHONHASHSEED")
os.environ.update(ENV)
if old_hash_seed!="3407":
    os.execv(sys.executable,[sys.executable,*sys.argv])
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"model"),str(ROOT)]
os.chdir(ROOT)
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader as NativeDataLoader,RandomSampler
from config import load_config
import semkine.train as ORIGINAL

FIELDS=("sequence_id","t_start_us","t_end_us","target","prev_state","betas","camera_K",
        "ptr","delta_t_s","is_sequence_start","is_sequence_end")
RAW_FIELDS=("events",*FIELDS,"lnes","lnes_window_ms","xag_source_index","xag_source_ms",
            "xag_part","xag_face","xag_bary","xag_valid")

def file_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(value):
    a=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(a).cast("B")).hexdigest()
def packet_sha(batch,names):
    h=hashlib.sha256()
    for name in names:
        value=getattr(batch,name,None);h.update(name.encode())
        if value is None:h.update(b"None");continue
        a=value.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode());h.update(str(a.shape).encode())
        h.update(memoryview(a).cast("B"))
    return h.hexdigest()
def normalized(value):
    if torch.is_tensor(value):return {"tensor_sha256":tensor_sha(value),"shape":list(value.shape),"dtype":str(value.dtype)}
    if isinstance(value,dict):return {str(k):normalized(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [normalized(v) for v in value]
    if isinstance(value,(str,int,float,bool))or value is None:return value
    return str(value)
def save(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");temp.replace(path)
def selected_step(step,phase,maximum,epoch_batches):
    return (phase=='preflight' or step<=3 or step%500 in(0,1,499) or step==maximum
            or (epoch_batches>0 and step%epoch_batches in(0,1,epoch_batches-1)))
def check_formal_binding(options,config_path):
    if not options.acceptance:raise ValueError('formal launch acceptance required')
    acceptance=json.loads(Path(options.acceptance).read_text())
    bindings=acceptance.get('bindings',{});key=str(Path(config_path).resolve())
    if acceptance.get('status')!='PASS':raise ValueError('preflight acceptance not PASS')
    if bindings.get('wrapper_sha256')!=file_sha(__file__):raise ValueError('wrapper differs from accepted source')
    if bindings.get('protocol_sha256')!=file_sha(options.protocol):raise ValueError('protocol differs from accepted protocol')
    if bindings.get('config_sha256',{}).get(key)!=file_sha(config_path):raise ValueError('config differs from accepted config')
    if bindings.get('sampler_by_config',{}).get(key)!=options.sampler:raise ValueError('sampler differs from accepted config role')
    protocol=json.loads(Path(options.protocol).read_text())
    if acceptance.get('preflight_evidence_sha256')!=protocol.get('preflight_acceptance_sha256'):
        raise ValueError('preflight evidence binding differs from frozen protocol')
def generator_info(g):
    return None if g is None else {"type":type(g).__module__+"."+type(g).__qualname__,
        "initial_seed":int(g.initial_seed()),"state_sha256":tensor_sha(g.get_state())}
def loader_info(loader):
    s=loader.sampler;under=getattr(getattr(s,"dataset",None),"_sampler",s)
    return {"loader_class":type(loader).__module__+"."+type(loader).__qualname__,
        "sampler_class":type(s).__module__+"."+type(s).__qualname__,
        "underlying_class":type(under).__module__+"."+type(under).__qualname__,
        "loader_generator":generator_info(loader.generator),
        "underlying_generator":generator_info(getattr(under,"generator",None)),
        "sampler_seed":getattr(s,"seed",None),"sampler_shuffle":getattr(s,"shuffle",None),
        "sampler_rank":getattr(s,"rank",None),"sampler_epoch":getattr(s,"epoch",None),
        "batch_size":loader.batch_size,"num_workers":loader.num_workers,
        "persistent_workers":loader.persistent_workers,"prefetch_factor":loader.prefetch_factor,
        "dataset_samples":len(loader.dataset),"drop_last":loader.drop_last}

def flags():
    return {"deterministic_algorithms":torch.are_deterministic_algorithms_enabled(),
        "warn_only":torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn_deterministic":torch.backends.cudnn.deterministic,"cudnn_benchmark":torch.backends.cudnn.benchmark,
        "matmul_allow_tf32":torch.backends.cuda.matmul.allow_tf32,"cudnn_allow_tf32":torch.backends.cudnn.allow_tf32,
        "matmul_precision":torch.get_float32_matmul_precision(),
        "bf16_reduced_precision_reduction":torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "fp16_reduced_precision_reduction":torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}

class ReproAudit(pl.Callback):
    def __init__(self,options):
        self.options=options;self.steps=0;self.rolling=hashlib.sha256();self.start=time.monotonic()
    def _selected(self,step):
        return selected_step(step,self.options.phase,self.max_steps,self.epoch_batches)
    def _snapshot(self,trainer,model,step):
        value={"step":step,"model":normalized(model.state_dict()),
            "optimizers":[normalized(o.state_dict())for o in trainer.optimizers],
            "schedulers":[normalized(s.scheduler.state_dict())for s in trainer.lr_scheduler_configs],
            "torch_cpu_rng":tensor_sha(torch.get_rng_state()),
            "torch_cuda_rng":tensor_sha(torch.cuda.get_rng_state()),
            "input_stream_sha256":self.rolling.hexdigest(),"batches":self.steps}
        save(self.out/f"state_rank{trainer.global_rank}_step{step:06}.json",value)
        return value
    def on_fit_start(self,trainer,model):
        self.out=Path(model.cfg['TRAIN']['OUTPUT_DIR']);self.max_steps=trainer.max_steps
        cfg=model.cfg
        f=flags()
        if not f['deterministic_algorithms']or f['warn_only']or f['cudnn_benchmark']:raise RuntimeError('determinism disabled')
        if trainer.world_size!=2 or self.max_steps not in (20,50,6000):raise RuntimeError('unsupported run budget/world size')
        if cfg['SEED']!=3407 or cfg['TRAIN']['BATCH_SIZE_PER_GPU']!=512:raise RuntimeError('effective seed/batch changed')
        if self.max_steps!=cfg['TRAIN']['MAX_STEPS']or trainer.accumulate_grad_batches!=1 or str(trainer.precision)!='bf16':
            raise RuntimeError('effective optimizer step/precision contract changed')
        if self.options.phase=='formal':
            if self.max_steps!=6000:raise RuntimeError('formal budget changed')
            check_formal_binding(self.options,cfg['_config_path'])
        elif self.max_steps not in(20,50):raise RuntimeError('preflight budget changed')
        self._snapshot(trainer,model,0)
    def on_train_start(self,trainer,model):
        loader=trainer.train_dataloader
        while hasattr(loader,'loaders'):
            loader=loader.loaders
        if isinstance(loader,(list,tuple)):raise RuntimeError('unexpected combined loader')
        self.dataset=loader.dataset;index=self.dataset.index
        self.epoch_batches=len(loader)
        if int(trainer.num_training_batches)!=self.epoch_batches:raise RuntimeError('training epoch denominator changed')
        starts=np.r_[0,np.flatnonzero(np.diff(index[:,0]))+1,len(index)]
        self.slices=[(int(a),index[a:b,1])for a,b in zip(starts[:-1],starts[1:])]
        info=loader_info(loader)
        if not info['sampler_class'].endswith('.DistributedSamplerWrapper')or not info['underlying_class'].endswith('.RandomSampler'):
            raise RuntimeError('actual Lightning sampler identity changed')
        actual=info['underlying_generator']
        if(self.options.sampler=='legacy_none')!=(actual is None):raise RuntimeError('actual sampler generator differs from requested policy')
        if actual is not None and actual['initial_seed']!=3407:raise RuntimeError('actual dedicated seed changed')
        if info['loader_generator']!=actual:raise RuntimeError('loader and underlying sampler generators differ')
        self.tape=(self.out/f'indices_rank{trainer.global_rank}.i32').open('xb')
        self.journal=(self.out/f'batches_rank{trainer.global_rank}.jsonl').open('x')
        save(self.out/f'effective_runtime_rank{trainer.global_rank}.json',{
            'phase':self.options.phase,'sampler_policy':self.options.sampler,'actual_train_loader':info,
            'actual_math_flags':flags(),'environment':{k:os.environ.get(k)for k in ENV},
            'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'local_rank':trainer.local_rank,
            'global_rank':trainer.global_rank,'world_size':trainer.world_size,'precision':str(trainer.precision),
            'effective_seed':int(model.cfg['SEED']),'effective_max_steps':int(trainer.max_steps),
            'effective_accumulate_grad_batches':trainer.accumulate_grad_batches,'epoch_batches':self.epoch_batches,
            'torch':torch.__version__,'lightning':pl.__version__,'cuda':torch.version.cuda,
            'cudnn':torch.backends.cudnn.version(),'nccl':torch.cuda.nccl.version(),'gpu':torch.cuda.get_device_name(),
            'config_sha256':file_sha(model.cfg['_config_path']),'wrapper_sha256':file_sha(__file__),
            'protocol':self.options.protocol,'protocol_sha256':file_sha(self.options.protocol),
            'original_entry_sha256':file_sha(ROOT/'semkine/train.py'),
            'original_metadata_boundary':'Original data_generator_seed describes its own branch; actual attributes above are authoritative for this explicit wrapper override.',
            'fresh_only':True,'resume_supported':False})
    def on_train_batch_start(self,trainer,model,batch,batch_idx):
        self.pending=dict(model._repro_cpu_fingerprint)
        step=int(trainer.global_step)+1
        if self.pending['step']!=step:raise RuntimeError('CPU fingerprint step mismatch')
        seq=batch.sequence_id.detach().cpu().numpy();ends=batch.t_end_us.detach().cpu().numpy()//1000-1
        indices=np.empty(len(seq),dtype=np.int32)
        for si in np.unique(seq):
            rows=np.flatnonzero(seq==si);start,available=self.slices[int(si)]
            found=np.searchsorted(available,ends[rows])
            if not np.array_equal(available[found],ends[rows]):raise RuntimeError('cannot reconstruct actual dataset index')
            indices[rows]=start+found
        self.pending.update(step=step,epoch=int(trainer.current_epoch),batch_size=batch.batch_size,
            indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
        if batch.batch_size!=512:raise RuntimeError('per-rank batch changed')
        self.tape.write(indices.tobytes());self.steps+=1
    def on_before_optimizer_step(self,trainer,model,optimizer,opt_idx):
        step=int(trainer.global_step)+1
        if self._selected(step):
            self.pending['gradients']=normalized({n:p.grad for n,p in model.named_parameters()})
            self.pending['prediction_sha256']=tensor_sha(model._repro_last_prediction)
            if not all(p.grad is None or bool(torch.isfinite(p.grad).all())for p in model.parameters()):
                raise FloatingPointError('nonfinite gradient')
    def on_train_batch_end(self,trainer,model,outputs,batch,batch_idx):
        step=int(trainer.global_step)
        if step!=self.pending['step']:raise RuntimeError('unexpected optimizer-step contract')
        loss=outputs['loss']if isinstance(outputs,dict)else outputs
        if not bool(torch.isfinite(loss).all()):raise FloatingPointError('nonfinite objective')
        self.pending['objective_sha256']=tensor_sha(loss)
        identity={k:self.pending[k]for k in('step','indices_sha256','small_fields_sha256','full_packet_sha256')}
        self.rolling.update(json.dumps(identity,sort_keys=True).encode())
        self.journal.write(json.dumps(self.pending,allow_nan=False)+'\n');self.journal.flush();self.tape.flush()
        if self.options.phase=='preflight'or step%500==0 or step==self.max_steps:
            self._snapshot(trainer,model,step)
        if step%50==0 or step==1 or step==self.max_steps:
            save(self.out/f'progress_rank{trainer.global_rank}.json',{'status':'TRAINING','step':step,
                'max_steps':self.max_steps,'elapsed_seconds':time.monotonic()-self.start,
                'input_stream_sha256':self.rolling.hexdigest(),'batches':self.steps})
    def on_train_end(self,trainer,model):
        self.tape.close();self.journal.close()
        trainer.save_checkpoint(str(self.out/'repro_final.ckpt'))
        trainer.strategy.barrier()
        save(self.out/f'completion_rank{trainer.global_rank}.json',{'status':'COMPLETE','step':int(trainer.global_step),
            'batches':self.steps,'elapsed_seconds':time.monotonic()-self.start,'input_stream_sha256':self.rolling.hexdigest(),
            'indices_sha256':file_sha(self.out/f'indices_rank{trainer.global_rank}.i32'),
            'final_checkpoint_sha256':file_sha(self.out/'repro_final.ckpt')})

def main():
    ap=argparse.ArgumentParser(add_help=False)
    ap.add_argument('--sampler',choices=('legacy_none','dedicated'),default=os.environ.get('XYZ_C1_REPRO_SAMPLER','legacy_none'))
    ap.add_argument('--phase',choices=('preflight','formal'),default=os.environ.get('XYZ_C1_REPRO_PHASE','preflight'))
    ap.add_argument('--protocol',default=os.environ.get('XYZ_C1_REPRO_PROTOCOL'))
    ap.add_argument('--acceptance',default=os.environ.get('XYZ_C1_REPRO_ACCEPTANCE'))
    options,rest=ap.parse_known_args()
    if not options.protocol or not Path(options.protocol).is_file():raise ValueError('explicit existing protocol required')
    for k,v in [('SAMPLER',options.sampler),('PHASE',options.phase),('PROTOCOL',options.protocol),('ACCEPTANCE',options.acceptance or'')]:
        os.environ['XYZ_C1_REPRO_'+k]=v
    # This wrapper accepts only a registered config from the original CLI.
    # Original overrides would otherwise change the actual recipe after its gate.
    cp=argparse.ArgumentParser(add_help=False,allow_abbrev=False);cp.add_argument('--config',required=True);known=cp.parse_args(rest)
    cfg=load_config(known.config)
    if '--resume'in rest or cfg['MODEL'].get('INIT_FROM')or cfg['MODEL'].get('DISTILL_CKPT'):raise ValueError('fresh-only wrapper')
    if any(k.startswith('XAG_')for k in cfg['MODEL'])or cfg['DATA'].get('XAG_SOURCE_ROOT'):raise ValueError('this C1 baseline wrapper requires no XAG keys/source cache')
    if cfg['SEED']!=3407 or cfg['TRAIN']['BATCH_SIZE_PER_GPU']!=512 or cfg['TRAIN']['DEVICES']!=2:raise ValueError('wrong seed/batch/world size')
    if options.phase=='preflight'and int(cfg['TRAIN']['MAX_STEPS'])not in(20,50):raise ValueError('preflight limited to20/50steps')
    if options.phase=='formal':
        if int(cfg['TRAIN']['MAX_STEPS'])!=6000:
            raise ValueError('formal requires6000step config and explicit passed preflight receipt')
        check_formal_binding(options,known.config)
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False
    torch.set_float32_matmul_precision('highest');torch.set_num_threads(4);torch.set_num_interop_threads(1)
    original_loader=ORIGINAL.DataLoader
    def controlled_loader(*args,**kwargs):
        if kwargs.get('shuffle') is True:
            kwargs['generator']=None if options.sampler=='legacy_none'else torch.Generator().manual_seed(3407)
        return original_loader(*args,**kwargs)
    ORIGINAL.DataLoader=controlled_loader
    original_trainer=pl.Trainer
    class ControlledTrainer(original_trainer):
        def __init__(self,*args,**kwargs):
            kwargs['deterministic']=True;kwargs['benchmark']=False
            kwargs['strategy']=DDPStrategy(find_unused_parameters=False,process_group_backend='nccl',
                bucket_cap_mb=25,gradient_as_bucket_view=False,broadcast_buffers=True)
            kwargs['callbacks']=[*kwargs.get('callbacks',[]),ReproAudit(options)]
            kwargs['enable_progress_bar']=False
            super().__init__(*args,**kwargs)
    ORIGINAL.pl.Trainer=ControlledTrainer
    # This hook hashes the native CPU packet before Lightning moves it. It
    # neither changes packet/dataclass type nor DataLoader/sampler subclasses.
    def before_transfer(self,batch,dataloader_idx):
        if self.training:
            step=int(self.global_step)+1;maximum=int(self.cfg['TRAIN']['MAX_STEPS'])
            raw=selected_step(step,options.phase,maximum,int(self.trainer.num_training_batches))
            self._repro_cpu_fingerprint={'step':step,'small_fields_sha256':packet_sha(batch,FIELDS),
                'full_packet_sha256':packet_sha(batch,RAW_FIELDS)if raw else None,
                'events':int(batch.events.shape[0])}
        return batch
    ORIGINAL.MNISTModel.on_before_batch_transfer=before_transfer
    forward=ORIGINAL.MNISTModel.forward_packet
    def capture_prediction(self,batch,*args,**kwargs):
        result=forward(self,batch,*args,**kwargs)
        if self.training:self._repro_last_prediction=result.detach()
        return result
    ORIGINAL.MNISTModel.forward_packet=capture_prediction
    sys.argv=[sys.argv[0],*rest]
    ORIGINAL.main()

if __name__=='__main__':main()
