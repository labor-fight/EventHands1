"""P3 CPU-only functional path/Jacobian audit, no target or task-risk reads."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from tools.s43_rel.core import RelationMemory,RelConfig
from tools.s43_rel.geometry import ManoGeometry,event_intrinsics
from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,InferenceAccessGuard
from tools.s43_rel.data import last_lnes,representation_from_snapshot
from tools.s43_rel.model import load_relation_model

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_revisit_20261028'
OLD=ROOT/'outputs/s43_rel_20261027'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def guards(protocol):
    p=json.loads(Path(protocol).read_text());bad={k:v for k,v in p['dependencies'].items() if sha(ROOT/k)!=v}
    if bad:raise RuntimeError(('frozen dependency mismatch',list(bad)))
    return p
def norm(x):return {'norm':float(torch.linalg.vector_norm(x.detach().float())),'max_abs':float(x.detach().abs().max())} if x is not None else {'norm':0.,'max_abs':0.,'disconnected':True}
def weights_hash(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()

def payloads():
    with InferenceAccessGuard() as guard:
        r=MeanShapeInputReader('ch_local');m=RelationMemory(RelConfig(),seq_id='P3')
        m.reset('P3',boundary_ms=0)
        for ms in range(200):
            packet=r.packet(ms,ms+1);m.ingest(packet['events'],watermark_ms=ms+1,packet_ordinal=ms,source_range=packet['source_range'])
        prev=np.array([0.,0.,.6]+[0.]*48,np.float32);provider=ManoGeometry()
        binding=provider.bind(prev,r.betas,event_intrinsics(r.K_cam),m.topology)
        snap=m.readout(200,edge_ids=binding.edge_ids)
        item={'events':r.packet(150,200)['events'],'lnes':last_lnes(r.packet(150,200)['events'],150,200),
              'start_ms':150,'cutoff_ms':200,'seq_id':'P3','source_range':(int(r.offsets[150]),int(r.offsets[200])),'packet_ordinal':199}
        real=representation_from_snapshot(item,snap,'full',binding.read(snap))
        # A declared synthetic quiet100ms suffix supplies a consistent empty-LN
        # but nonzero decaying-memory control, rather than fabricated raw order.
        end=int(r.offsets[200]);m.ingest(np.empty((0,4),np.int64),watermark_ms=300,packet_ordinal=200,source_range=(end,end))
        tail=m.readout(300,edge_ids=binding.edge_ids)
        item2=dict(item,events=np.empty((0,4),np.int64),lnes=np.zeros_like(item['lnes']),start_ms=250,cutoff_ms=300,packet_ordinal=200)
        quiet=representation_from_snapshot(item2,tail,'full',binding.read(tail))
    return real,quiet,prev,r.betas,r.K_cam,{'recording':'ch_local','native_ms':[0,200],
        'events':end,'inference_access_guard':guard.receipt(),'quiet_control':'synthetic complete no-event suffix[200,300), not an actual-data task claim'}

def local_forward(model,rep,prev,beta,K,mode='full',need_grad=False,gate=None):
    leaf=lambda x:torch.tensor(np.asarray(x,np.float32)[None],requires_grad=need_grad)
    values={k:leaf(rep[k]) for k in ('lnes','sensor','f','a','lam')}
    p=torch.tensor(prev[None]);b=torch.tensor(beta[None]);camera=torch.tensor(K[None])
    condition=torch.tensor([[rep.get('delta_commit_ms',50.)/50.,rep.get('output_lag_ms',1.),float(rep.get('missing',False))]])
    capture={}
    def prehook(module,args):capture['old_input']=args[0].detach()
    handle=model.base.conv1.register_forward_pre_hook(prehook)
    old_gate=model.base.zero_event_gate
    if gate is not None:model.base.zero_event_gate=gate
    try:
        model._active_sensor=None if mode in ('offSensor','offBoth') else values['sensor'].permute(0,3,1,2).contiguous()
        typed=model.typed_adapter(values['f'],values['a'],values['lam'],condition)
        model._active_typed=None if mode in ('offTyped','offBoth') else typed
        out=model.base(values['lnes'],p,betas=b,camera_K=camera)
    finally:
        model._active_sensor=model._active_typed=None;model.base.zero_event_gate=old_gate;handle.remove()
    return out,values,capture,typed

def derivative_case(model,rep,prev,beta,K,gate=None):
    out,values,capture,typed=local_forward(model,rep,prev,beta,K,need_grad=True,gate=gate)
    gen=torch.Generator().manual_seed(3407);v=torch.randn(51,generator=gen);v/=v.norm()
    scales=torch.tensor([.005]*3+[.1]*48);scalar=(out[0]*(v/scales)).sum()
    parameters={'sensor_weight':model.sensor_adapter.weight,'typed_direction_weight':model.typed_adapter.direction[0].weight,
                'typed_support_weight':model.typed_adapter.support[0].weight,'typed_output_weight':model.typed_adapter.output.weight,
                'typed_output_bias':model.typed_adapter.output.bias,'base_conv1_weight':model.base.conv1.weight,
                'base_fc_weight':model.base.rn.fc.weight,'previous_MLP_first_weight':next(model.base.prev_mlp.parameters())}
    keys=list(values)+list(parameters);objects=list(values.values())+list(parameters.values())
    grads=torch.autograd.grad(scalar,objects,allow_unused=True)
    g=dict(zip(keys,grads));m0=(capture['old_input'].abs().amax(1,keepdim=True)>0)
    halo=F.max_pool2d(m0.float(),3,1,1)>0;region={'old_mask':m0[0,0],'halo_only':(halo&~m0)[0,0],'far_outside_halo':(~halo)[0,0]}
    sensor_grad=g['sensor'][0,:,:,:6] if g['sensor'] is not None else torch.zeros((180,240,6))
    regions={k:{'pixels':int(mask.sum()),'gradient':norm(sensor_grad[mask]),
                'nonzero_A_input_entries':int(torch.count_nonzero(values['sensor'][0,:,:,:6][mask]))} for k,mask in region.items()}
    return {'projection':'fixed random seed3407, translation/.005 and rotations/.1; not a task loss',
        'output_minus_previous':norm(out-torch.tensor(prev[None])),'typed_pre_old_gate':norm(typed),
        'input_and_parameter_gradients':{k:norm(t) for k,t in g.items()},'sensor_input_regions':regions,
        'zero_event_gate':model.base.zero_event_gate if gate is None else gate,
        'old_LNES_empty':not bool(np.any(rep['lnes']))},out.detach(),region

def output_difference(model,a,b,beta):
    delta=(a-b)[0].detach().numpy()
    with torch.no_grad():
        _,ja=model.base._fk(a,torch.tensor(beta[None]));_,jb=model.base._fk(b,torch.tensor(beta[None]))
    return {'translation_parameter_difference_m':float(np.linalg.norm(delta[:3])),
            'root_AA_coordinate_difference_rad':float(np.linalg.norm(delta[3:6])),
            'local_AA_coordinate_difference_rad':float(np.linalg.norm(delta[6:])),
            'predicted_FK_joint_displacement_mm':float(torch.linalg.vector_norm(ja-jb,dim=-1).mean()*1000),
            'meaning':'prediction sensitivity, not error, improvement or task evidence'}

def run(protocol,out):
    torch.set_num_threads(1);guards(protocol)
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    real,quiet,prev,beta,K,source=payloads();config=ROOT/'configs/s43_rel/r1/full.yaml'
    checkpoint=OLD/'train/full_r1/full/step=6000.ckpt'
    report={'schema':'s43_revisit_P3_interface_engineering_v1','protocol_sha256':sha(protocol),'source_sha256':sha(__file__),
        'prior_source_findings_disclosed_before_prereg':True,'input_source':source,'GT_target_read':False,
        'new_task_scores':False,'GPU_used':False,'optimizer_used':False,'cases':{},
        'gradient_contract':'The production builder detaches fixed observations. Local feature tensors are made independent differentiable variables only for sensitivity; no history/geometry-training gradient is claimed.'}
    for label,path in [('canonical_initialized_zero_new_paths',None),('trained_full6000',checkpoint)]:
        model,cfg=load_relation_model(config,path,device='cpu');model.eval();before=weights_hash(model)
        with torch.no_grad():production=model([real],torch.tensor(prev[None]),torch.tensor(beta[None]),torch.tensor(K[None]))
        normal,y,regions=derivative_case(model,real,prev,beta,K)
        empty,_,_=derivative_case(model,quiet,prev,beta,K)
        ungated,_,_=derivative_case(model,quiet,prev,beta,K,gate=False)
        equivalence=float((production-y).abs().max())
        modes={}
        with torch.no_grad():
            for mode in ('full','offSensor','offTyped','offBoth'):modes[mode]=local_forward(model,real,prev,beta,K,mode=mode)[0]
            far=np.argwhere(regions['far_outside_halo'].numpy())[0]
            altered=dict(real,sensor=real['sensor'].copy());altered['sensor'][far[0],far[1],0]+=1.
            far_out=local_forward(model,altered,prev,beta,K)[0]
        entry={'checkpoint':None if path is None else str(path),'checkpoint_sha256':None if path is None else sha(path),
            'production_vs_differentiable_feature_forward_max_abs':equivalence,'normal':normal,'quiet_nonzero_tail_old_gate':empty,
            'quiet_nonzero_tail_gate_disabled_control':ungated,
            'I1_same_prefix_path_deletions':{mode:output_difference(model,modes['full'],value,beta) for mode,value in modes.items()},
            'far_sensor_finite_change':{'pixel':far.tolist(),'change_channel0':1.,'output_max_abs':float((modes['full']-far_out).abs().max())},
            'weights_BN_and_buffers_unchanged':weights_hash(model)==before}
        if equivalence!=0 or entry['far_sensor_finite_change']['output_max_abs']!=0:raise AssertionError('engineering forward/mask parity failed')
        if empty['output_minus_previous']['max_abs']!=0 or any(empty['input_and_parameter_gradients'][k]['max_abs']!=0 for k in ('sensor','f','a','lam')):
            raise AssertionError('expected old empty-LNES hard-zero path did not hold')
        if normal['sensor_input_regions']['far_outside_halo']['gradient']['max_abs']!=0:raise AssertionError('far-mask Jacobian not hard zero')
        report['cases'][label]=entry
    guards(protocol)
    report['old_dependencies_unchanged']=True
    report['interpretation']=['Hard-zero old-gate/mask derivatives obstruct the declared alternate evidence paths.',
        'Zero new input derivatives at initialization are expected from zero output weights; nonzero parameter gradients distinguish this from a disconnected branch.',
        'Nonzero trained derivatives or large frozen-path deletions do not prove useful information, adequate optimization, or causal sufficiency.',
        'The 3x3 sensor adapter can read an input halo around old active sites; only far-outside-halo is provably invisible.']
    write(out/'results.json',report);print(json.dumps({'complete':True,'out':str(out/'results.json')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,default=OUT/'stage1_protocol.json')
    p.add_argument('--out',type=Path,default=OUT/'interface_audit');a=p.parse_args();run(a.protocol,a.out)
