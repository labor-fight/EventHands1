"""CPU algebra audit of frozen static sign-slot duplication; no task scores.

Uses only public MANO/calibration and random native-cell counts. No optimizer,
target, empirical evaluation recording or GPU is accessed. Frozen files stay
unchanged; the structural identity holds independently of learned weights.
"""
from __future__ import annotations
import argparse,hashlib,json,os
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import torch
from torch.nn import functional as F
from threadpoolctl import threadpool_info
from .field_core import TimeFieldConfig,TimeFieldSnapshot,ro
from .field_tensor import tensorize
from .s43_field_full_training import build_model

ROOT=Path(__file__).resolve().parents[2]


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def public_random_field():
    rng=np.random.default_rng(3407);c=TimeFieldConfig()
    counts=rng.integers(0,4,(50,90,120,2),dtype=np.uint32)
    row_n=counts.sum((1,2,3),dtype=np.uint64);offset=np.r_[0,np.cumsum(row_n,dtype=np.uint64)].astype(np.int64)
    snapshot=TimeFieldSnapshot(c,50,0,'public_random_native_counts',1,ro(counts),ro(np.arange(50)),
        ro(np.ones(50,bool)),ro(np.column_stack((offset[:-1],offset[1:]))),ro(row_n),dict(SIMULATED=True))
    before=snapshot.content_sha256();field=tensorize([snapshot]).global40
    assert snapshot.content_sha256()==before
    return field,dict(seed=3407,native_count_shape=list(counts.shape),count_range=[0,3],
        field_sha256=thash(field),native_count_sha256=hashlib.sha256(counts.tobytes()).hexdigest(),
        note='Random legal nonnegative r2countfield for algebra, not a physical renderer or task-data information bound.')


def pair_summary(t,axis):
    x=t.movedim(axis,0);a,b=x[::2],x[1::2]
    equal=(a==b).reshape(45,-1).all(1)
    return dict(equal_pairs=int(equal.sum()),total_pairs=45,all_pairs_bitwise_equal=bool(equal.all()),
        max_pair_abs=float((a.float()-b.float()).abs().max()))


def inspect(model,field,pose,beta,K):
    refs=np.array([0],np.int64);cutoffs=np.array([50],np.int64);hist=[()]
    before={k:thash(v) for k,v in model.state_dict().items()};rng=torch.get_rng_state().clone()
    with torch.inference_mode():
        plan=model.prepare_queries_cpu(pose,beta,K,refs,cutoffs,hist)[0]
        local,metadata=model._local_patches(field,pose,beta,K,refs,cutoffs,hist)
        encoded=model.local_encoder(local.reshape(-1,5,7,7)).reshape(1,8,90,8)
        first=model.local_head[0];x=encoded.double();w=first.weight.detach().double().reshape(128,8,45,2,8);bias=first.bias.detach().double()
        full=F.linear(x.reshape(1,5760),w.reshape(128,5760),bias)
        merged=F.linear(x[:,:,::2].reshape(1,2880),w.sum(3).reshape(128,2880),bias)
        # An arbitrary antisymmetric sign-column perturbation is a guaranteed
        # null direction only when the two supplied feature slots coincide.
        d=torch.linspace(-.01,.01,128*8*45*8,dtype=torch.float64).reshape(128,8,45,8)
        changed=w.clone();changed[:,:,:,0]+=d;changed[:,:,:,1]-=d
        null=F.linear(x.reshape(1,5760),changed.reshape(128,5760),bias)
    assert before=={k:thash(v) for k,v in model.state_dict().items()}
    assert torch.equal(rng,torch.get_rng_state())
    q=plan.query
    unique_centers=[int(torch.unique(q.centers_px[t],dim=0).shape[0]) for t in range(8)]
    unique_encoded=[int(torch.unique(encoded[0,t],dim=0).shape[0]) for t in range(8)]
    result=dict(requested_mode=model.arm,used_mode=plan.metadata['used_mode'],
        fallback=plan.metadata['used_uniform_reacquisition'],fallback_reason=plan.metadata['invalid_reason'],
        grid=pair_summary(q.grid,1),centers=pair_summary(q.centers_px,1),valid=pair_summary(q.valid,1),
        in_view=pair_summary(q.in_view,1),interp=pair_summary(q.full_interp_support,1),
        association=pair_summary(plan.association_flag,1),local_patches=pair_summary(local,2),local_encoded=pair_summary(encoded,2),
        unique_centers_per_time=unique_centers,unique_encoded_per_time=unique_encoded,
        encoder_output_shape=list(encoded.shape),association_flag3_nonzero=int(torch.count_nonzero(local[:,:, :,3])),
        actual_interpolation_flag4_nonzero=int(torch.count_nonzero(local[:,:,:,4])),
        field_channels0_2_nonzero=int(torch.count_nonzero(local[:,:,:,:3])),
        first_linear_declared_input_columns=5760,first_linear_output_units=128,
        merge_sign_first_linear_max_abs_FP64=float((full-merged).abs().max()),
        arbitrary_antisymmetric_weight_change_max_abs_FP64=float((full-null).abs().max()),
        NN_weights_buffers_and_RNG_unchanged=True,grid_sha256=thash(q.grid),local_encoded_sha256=thash(encoded),
        metadata_semantics={k:metadata[0][k] for k in ('requested_mode','used_mode','invalid_reason','used_uniform_reacquisition')})
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if torch.cuda.is_available():raise RuntimeError('CPU-only audit requires CUDA_VISIBLE_DEVICES empty')
    a.out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    files=[Path(__file__),*[ROOT/'tools/s43_revisit'/n for n in ('field_query.py','s43_field_model.py','s43_field_model_reacquire.py','field_core.py','field_tensor.py','s43_field_full_training.py')],ROOT/'assets/mano_right.npz']
    sources={str(p):sha(p) for p in files};write(a.out/'sourcefreeze.json',dict(sources=sources,scope='public CPU algebra only',GT=False,GPU=False))
    field,field_meta=public_random_field();pose=torch.zeros(1,51);pose[:,2]=.6;beta=torch.zeros(1,10)
    K=torch.tensor([[[600.,0,320.],[0,600.,240.],[0,0,1.]]])
    static,_=build_model('field_static','cpu');box,_=build_model('field_box','cpu');static.eval();box.eval()
    # Same actual shared encoder weights, so comparing modes does not introduce
    # a random parameter change. Both factories use the prescribed public seed.
    assert all(torch.equal(v,box.local_encoder.state_dict()[k]) for k,v in static.local_encoder.state_dict().items())
    bad=pose.clone();bad[:,2]=-.2
    rows=dict(static=inspect(static,field,pose,beta,K),hand_box=inspect(box,field,pose,beta,K),
        static_R1_fallback=inspect(static,field,bad,beta,K),box_R1_fallback=inspect(box,field,bad,beta,K))
    s=rows['static']
    for name in ('grid','centers','valid','in_view','interp','association','local_patches','local_encoded'):
        assert s[name]['all_pairs_bitwise_equal'],name
    assert s['unique_centers_per_time']==[45]*8 and s['unique_encoded_per_time']==[45]*8
    assert s['merge_sign_first_linear_max_abs_FP64']<1e-12 and s['arbitrary_antisymmetric_weight_change_max_abs_FP64']<1e-12
    for key in ('hand_box','static_R1_fallback','box_R1_fallback'):
        assert rows[key]['unique_centers_per_time']==[90]*8 and rows[key]['unique_encoded_per_time']==[90]*8
        assert not rows[key]['grid']['all_pairs_bitwise_equal']
    for key in ('static_R1_fallback','box_R1_fallback'):
        assert rows[key]['fallback'] and rows[key]['association_flag3_nonzero']==0
        assert rows[key]['actual_interpolation_flag4_nonzero']>0 and rows[key]['field_channels0_2_nonzero']>0
    assert rows['static_R1_fallback']['grid_sha256']==rows['box_R1_fallback']['grid_sha256']
    assert rows['static_R1_fallback']['local_encoded_sha256']==rows['box_R1_fallback']['local_encoded_sha256']
    assert all(sha(p)==h for p,h in sources.items())
    result=dict(schema='s43_field_static_slot_algebra_audit_v1',status='PASS_STRUCTURAL_DUPLICATION_CONFIRMED',
        created_utc=datetime.now(timezone.utc).isoformat(),sources=sources,conditions=dict(public_pose51=pose[0].tolist(),public_invalid_pose51=bad[0].tolist(),beta0=True,K_source=K[0].tolist(),render_scale=float(static.base.render_scale)),
        field=field_meta,rows=rows,algebra=dict(identity='z[t,a,+]=z[t,a,-] implies Wplus*z+Wminus*z=(Wplus+Wminus)*z, before nonlinear head.',
            static_declared_columns=5760,static_distinct_feature_coordinates_upper_bound=2880,
            first_linear_weight_parameters=737280,guaranteed_antisymmetric_null_parameters=368640,
            first_linear_columns_collapsible=2880,conclusion='Same nominal parameters do not establish equal functional input capacity. Independent sign weights cannot recover a distinction absent from their inputs.'),
        limits=['Structural invariant for static slots, independent of learned encoder/head weights; randomfield is an engineering witness only.',
            'Box/fallback90 distinct observed locations do not prove a full5760rank input distribution or twice the total model/task capacity.',
            'This is not evidence that duplication caused a task outcome. Full-image/global branch and all frozen training arms remain unchanged.',
            'FP64 merged-first-linear differences are finite summation roundoff, not a claim that reordering FP32 sums is bitwise.',
            'Last51head is zero initialized; audit uses nonzero first-linear preactivation to avoid a trivial zero-output equivalence.'],
        GT_or_real_task_scores_read=False,GPU_used=False,optimizer_updates=0,frozen_sources_changed=False,
        threadpools_observed=threadpool_info(),torch_CPU_threads=torch.get_num_threads())
    write(a.out/'receipt.json',result)
    text='Static槽位代数审计：同名参数量不等于相同功能输入容量。\n\n公共姿态、β0、公共K和随机合法r2计数场下，static90槽由45个锚点的重复sign组成；45对coords/flags/patches/共享CNN输出均逐位相同。首个Linear的5760输入列可按sign求和合并到2880列，737280个weight中保证存在368640维反对称冗余方向；独立W+、W−只能贡献其和。\n\nhand_box与R1 fullimage fallback各90个位置在该公共条件下均不同，随机场的编码也90个不同；fallback关联flag3为0但仍读取真实2D值与插值flag。它们没有static这种强制sign配对重复，不代表5760维分布满秩或整模型容量翻倍。\n\n这只是冻结结构的代数边界，无GT/GPU/训练或任务分数；不证明既有任务差异由此造成，也不修改正在运行的任何arm、表示或源码。\n'
    (a.out/'receipt.md').write_text(text)
    static.close_geometry_workers();box.close_geometry_workers()
    print(json.dumps(dict(status=result['status'],unique={k:v['unique_centers_per_time'] for k,v in rows.items()},merge_error=s['merge_sign_first_linear_max_abs_FP64'],null_error=s['arbitrary_antisymmetric_weight_change_max_abs_FP64'],out=str(a.out))))


if __name__=='__main__':main()
