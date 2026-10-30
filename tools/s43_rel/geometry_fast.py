"""Opt-in exact-sparsity MANO action Jacobian engineering backend.

Forward MANO, visibility and binding remain the frozen reference definitions.
Only structurally zero derivative work is removed: joint-local pose blend
derivatives and ALL nonzero skinning weights are evaluated exactly. This is
not ROI selection, weight thresholding, pruning or a new scientific reader.

An isolated module alias owns the replacement global, so importing this file
never changes geometry.py or a training process using its original module.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.sparse import csr_matrix

from . import geometry as reference

REFERENCE_SHA256=hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest()
_ORIGINAL_FUNCTION=reference.mano_kinematics


@lru_cache(maxsize=1)
def _derivative_constants():
    c=reference._constants()
    sparse=csr_matrix(c['weights'])
    if sparse.nnz!=np.count_nonzero(c['weights']):
        raise AssertionError('sparse conversion removed a nonzero skinning weight')
    pose=c['posedirs'].reshape(778*3,135)
    blocks=np.ascontiguousarray(pose.T.reshape(15,9,778*3))
    generators=np.array([[[0,0,0],[0,0,-1],[0,1,0]],
                         [[0,0,1],[0,0,0],[-1,0,0]],
                         [[0,-1,0],[1,0,0],[0,0,0]]],np.float64)
    return sparse,blocks,generators


@lru_cache(maxsize=16)
def _shape_exact(beta_bytes):
    """Bounded cache, keyed by all exact validated float64 beta bytes."""
    beta=np.frombuffer(beta_bytes,dtype=np.float64)
    c=reference._constants()
    vs=c['v_template']+np.einsum('vkl,l->vk',c['shapedirs'],beta)
    J=c['J_regressor']@vs
    vs.flags.writeable=False;J.flags.writeable=False
    return vs,J


def mano_kinematics(prev51,betas,config=reference.GeometryConfig(),*,delta=None,jacobian=True):
    """Frozen-reference FK plus sparse exact right-SO3 analytic derivatives.

    The 51 output axes retain translation5mm and rotation0.1rad per unit.
    Root rotations do not affect pose-corrective blend shapes, exactly as in
    the reference. Translation derivative axes are filled analytically at end.
    """
    prev=reference._finite(prev51,(51,),'prev51')
    beta=reference._finite(betas,(10,),'betas')
    t,R=reference._decode(prev)
    if delta is not None:
        delta=reference._finite(delta,(51,),'delta')
        t+=config.translation_scale_m*delta[:3]
        R=R@reference.rotation_exp(config.rotation_scale_rad*delta[3:].reshape(16,3))
    c=reference._constants()
    vs,J=_shape_exact(np.ascontiguousarray(beta,dtype=np.float64).tobytes())
    pose_dirs=c['posedirs'].reshape(778*3,135)
    vp=vs+(pose_dirs@(R[1:]-np.eye(3)).reshape(135)).reshape(778,3)
    rel=J.copy();rel[1:]-=J[np.asarray(c['parents'][1:])]
    local=np.broadcast_to(np.eye(4),(16,4,4)).copy()
    local[:,:3,:3],local[:,:3,3]=R,rel
    T=np.empty_like(local);T[0]=local[0]
    if jacobian:
        weights,blocks,generators=_derivative_constants()
        joint_dR=config.rotation_scale_rad*(R[:,None]@generators[None])
        # 15 independent 3x9 by 9x2334 products; no 51x135 zero-filled GEMM.
        dvp=np.zeros((48,778,3),np.float64)
        dvp[3:]=(joint_dR[1:].reshape(15,3,9)@blocks).reshape(45,778,3)
        dlocal=np.zeros((48,16,4,4),np.float64)
        for joint in range(16):
            dlocal[3*joint:3*joint+3,joint,:3,:3]=joint_dR[joint]
        dT=np.empty_like(dlocal);dT[:,0]=dlocal[:,0]
    for joint in range(1,16):
        parent=c['parents'][joint]
        T[joint]=T[parent]@local[joint]
        if jacobian:
            dT[:,joint]=dT[:,parent]@local[joint]+T[parent]@dlocal[:,joint]
    joint_rel=T.copy()
    joint_rel[:,:3,3]-=np.einsum('jkl,jl->jk',T[:,:3,:3],J)
    # Forward LBS deliberately retains the same dense operations/order.
    blended=(c['weights']@joint_rel.reshape(16,16)).reshape(778,4,4)
    vertices=np.einsum('vkl,vl->vk',blended[:,:3,:3],vp)+blended[:,:3,3]+t
    if not jacobian:return vertices
    drel=dT.copy()
    drel[:,:,:3,3]-=np.einsum('qjkl,jl->qjk',dT[:,:,:3,:3],J)
    # Homogeneous bottom derivative row and all translation-action rows are
    # identically zero. Keep all48 rotation axes and all2028 nonzero weights.
    top=np.ascontiguousarray(drel[:,:,:3,:].transpose(1,0,2,3)).reshape(16,48*12)
    dblend=(weights@top).reshape(778,48,3,4).transpose(1,0,2,3)
    # The reference's broadcast einsum iterates a strided 3-vector product
    # inefficiently. Write those same three scalar products explicitly; this
    # does not approximate or omit any derivative contribution.
    posed_rotation=np.empty_like(dvp)
    for axis in range(3):
        posed_rotation[:,:,axis]=(blended[None,:,axis,0]*dvp[:,:,0]+
                                  blended[None,:,axis,1]*dvp[:,:,1]+
                                  blended[None,:,axis,2]*dvp[:,:,2])
    rotation_dv=(np.einsum('qvkl,vl->qvk',dblend[:,:,:,:3],vp)+
                 posed_rotation+dblend[:,:,:,3])
    dv=np.zeros((51,778,3),np.float64)
    dv[3:]=rotation_dv
    dv[np.arange(3),:,np.arange(3)]=config.translation_scale_m
    return vertices,dv


@lru_cache(maxsize=1)
def _isolated_reference():
    name='_s43_rel_geometry_exact_sparse_alias_'+REFERENCE_SHA256[:16]
    if name in sys.modules:return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,reference.__file__)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    module.mano_kinematics=mano_kinematics
    if reference.mano_kinematics is not _ORIGINAL_FUNCTION:
        raise AssertionError('original geometry module was modified')
    return module


class FastManoGeometry(_isolated_reference().ManoGeometry):
    """Independent opt-in binding with the unchanged silhouette/spatial code."""
    def bind(self,*args,**kwargs):
        binding=super().bind(*args,**kwargs)
        binding.metadata['engineering_backend']='exact_sparse_action_jacobian'
        binding.metadata['reference_source_sha256']=REFERENCE_SHA256
        binding.metadata['fast_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return binding


def provider_metadata():
    weights,blocks,_=_derivative_constants()
    return dict(backend='exact_sparse_action_jacobian',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reference_source_sha256=REFERENCE_SHA256,reference_module_unchanged=reference.mano_kinematics is _ORIGINAL_FUNCTION,
        all_nonzero_weights_retained=weights.nnz,minimum_retained_weight=float(weights.data.min()),
        pose_derivative_blocks_shape=list(blocks.shape),shape_cache_max_entries=16,
        shape_cache_key='validated float64 beta exactbytes',
        removed_work='structural zero pose-derivative columns and homogeneous/translation zero derivative rows only',
        changed_science=False,changed_training_backend=False,approximate_pruning=False)


def engineering_benchmark(out=None):
    """Six saved causal ch contexts, numerical parity and measured CPU service.

    Read only cached deployment context, recording IDs and source K/beta. No
    targets, GT visibility, zgz data or accuracy metric are loaded.
    """
    import json
    import os
    import time
    from .core import RelConfig,RelationMemory
    from tools.s43_br.data import SequenceReader,DATA_ROOT
    root=Path(__file__).resolve().parents[2]
    output=Path(out) if out else root/'outputs/s43_rel_20261027/engineering/geometry_fast.json'
    cache=root/'outputs/s43_rel_20261027/mechanisms/e12/existing_features.npz'
    with np.load(cache,allow_pickle=False) as z:
        contexts=z['context'];recordings=z['recording'];ids=z['sample_id']
    selected=np.flatnonzero(recordings=='ch_local')[[0,1,3,7,11,15]]
    reader=SequenceReader('ch_local',DATA_ROOT)
    K=reference.event_intrinsics(reader.K_cam);beta=reader.betas
    mem=RelationMemory(RelConfig());topology=mem.topology
    old_provider=reference.ManoGeometry();fast_provider=FastManoGeometry()
    rng=np.random.default_rng(3407);cases=[]
    for index in selected:
        pose=contexts[index,:51]
        ov,oj=reference.mano_kinematics(pose,beta)
        fv,fj=mano_kinematics(pose,beta)
        if not np.array_equal(ov,fv):raise AssertionError('causal ch forward changed')
        np.testing.assert_allclose(oj,fj,rtol=2e-13,atol=2e-15)
        fd_error=0.
        for axis in range(51):
            step=np.eye(51)[axis]*1e-4
            plus=reference.mano_kinematics(pose,beta,delta=step,jacobian=False)
            minus=reference.mano_kinematics(pose,beta,delta=-step,jacobian=False)
            fd=(plus-minus)/(2e-4)
            np.testing.assert_allclose(fd,fj[axis],rtol=3e-6,atol=4e-10)
            fd_error=max(fd_error,float(np.abs(fd-fj[axis]).max()))
        ob=old_provider.bind(pose,beta,K,topology)
        fb=fast_provider.bind(pose,beta,K,topology)
        for key in ('uv','n','edge_ids','edge_slot','point_index','weights','normal_dot_delta','branch_regular'):
            np.testing.assert_array_equal(getattr(ob,key),getattr(fb,key))
        np.testing.assert_allclose(ob.a,fb.a,rtol=3e-13,atol=3e-13)
        from types import SimpleNamespace
        Z=rng.uniform(0,1,(len(ob.edge_ids),2,2,3,2)).astype(np.float32)
        snapshot=SimpleNamespace(Z=Z,edge_ids=ob.edge_ids)
        op,fp=ob.read(snapshot),fb.read(snapshot)
        np.testing.assert_array_equal(op['lam'],fp['lam'])
        np.testing.assert_allclose(op['f'],fp['f'],rtol=2e-12,atol=1e-10)
        old_diag=(ob.a**2).T@op['lam'];fast_diag=(fb.a**2).T@fp['lam']
        old_C=ob.a.T@(ob.a*op['lam'].sum(-1,keepdims=True))
        fast_C=fb.a.T@(fb.a*fp['lam'].sum(-1,keepdims=True))
        np.testing.assert_allclose(old_diag,fast_diag,rtol=3e-13,atol=3e-10)
        np.testing.assert_allclose(old_C,fast_C,rtol=3e-13,atol=3e-9)
        cases.append(dict(sample_id=str(ids[index]),forward_bitwise=True,
            all51_FD_max_m=fd_error,jacobian_reference_max_m=float(np.abs(oj-fj).max()),
            binding_a_max=float(np.abs(ob.a-fb.a).max()),f_max=float(np.abs(op['f']-fp['f']).max()),
            diagG_max=float(np.abs(old_diag-fast_diag).max()),C_max=float(np.abs(old_C-fast_C).max()),
            selected_edges=len(ob.edge_ids),visibility_and_sparse_links_bitwise=True))
    costs={name:dict(fk=[],bind=[],silhouette=[],spatial=[]) for name in ('reference','fast')}
    for repeat in range(4):
        for index in selected:
            pose=contexts[index,:51]
            order=('reference','fast') if repeat%2==0 else ('fast','reference')
            for name in order:
                fn=reference.mano_kinematics if name=='reference' else mano_kinematics
                provider=old_provider if name=='reference' else fast_provider
                start=time.perf_counter();fn(pose,beta);costs[name]['fk'].append(time.perf_counter()-start)
                start=time.perf_counter();binding=provider.bind(pose,beta,K,topology)
                bind_wall=time.perf_counter()-start
                t=binding.metadata['seconds']
                costs[name]['bind'].append(bind_wall)
                costs[name]['silhouette'].append(t['silhouette'])
                costs[name]['spatial'].append(t['spatial_binding'])
    stats={name:{part:dict(n=len(v),median_ms=float(np.median(v)*1000),p95_ms=float(np.percentile(v,95)*1000),
        min_ms=float(np.min(v)*1000),max_ms=float(np.max(v)*1000)) for part,v in row.items()} for name,row in costs.items()}
    result=dict(schema='s43_relation_fast_geometry_engineering_v1',scope='CPU arithmetic and service cost only; no GT/zgz accuracy or training changes',
        provider=provider_metadata(),cache_source=dict(path=str(cache),sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
            arrays_read=['context','recording','sample_id'],history='frozen original S43 two causal W50 from public initial state',selected_indices=selected.tolist()),
        cases=cases,costs=stats,speedup=dict(fk=stats['reference']['fk']['median_ms']/stats['fast']['fk']['median_ms'],
            full_bind=stats['reference']['bind']['median_ms']/stats['fast']['bind']['median_ms']),
        load_average=os.getloadavg(),thread_environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','CUDA_VISIBLE_DEVICES')},
        shape_cache_info=str(_shape_exact.cache_info()),
        bind_timing_scope='actual provider.bind wall, including metadata/fingerprint and fast backend source identity',
        exclusions=['sensor ingestion/seal/readout','neural model decode/commit','transfer','cold compilation/static spatial index','full asynchronous scheduler'],
        interpretation='Only an opt-in equivalent inference backend candidate; old training backend stays frozen. Whole-chain speed must be measured separately.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),speedup=result['speedup'],costs=stats)),flush=True)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--benchmark',action='store_true');parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.benchmark:engineering_benchmark(args.out)
