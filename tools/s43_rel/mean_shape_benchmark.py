"""Separate public-beta0 resource wrapper around the frozen surface benchmark.

Run in its own process: temporary module globals are restored in finally.
The original benchmark/reader/training/deployment files are never edited.
Formal measurements wait for BOTH main postprocess and finalization receipts.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
from pathlib import Path
import os
import sys
import tempfile
import threading

import numpy as np

from . import benchmark as B
from . import report as R

ARMS=('canonical',)+R.ARMS
SCHEMA='s43_relation_mean_shape_benchmark_v1'
_PATCH_LOCK=threading.Lock()


class MeanShapeObservationReader(B.ObservationReader):
    """Shared beta0 input reader; never call conditional parent initialization."""
    def __init__(self,recording,root):
        api=importlib.import_module('tools.s43_rel.mean_shape_replay')
        inner=api.MeanShapeInputReader(recording,root)
        self._mean_shape_reader=inner
        for name in ('root','base','events','offsets','K_cam','runs'):
            setattr(self,name,getattr(inner,name))
        self.betas=np.zeros(10,np.float32)
        if not np.array_equal(inner.betas,self.betas):
            raise AssertionError('shared public-mean input reader returned a nonzero beta')


@contextmanager
def public_mean_context(extension,engineering=False):
    """Controlled process-local replacement plus actual beta0 call guards."""
    api=importlib.import_module('tools.s43_rel.mean_shape_replay')
    if not _PATCH_LOCK.acquire(blocking=False):raise RuntimeError('mean-shape benchmark cannot run concurrent patched models')
    original_reader,original_identity=B.ObservationReader,B.source_identity
    original_tracker,original_decode=B.CausalRelationTracker,B.decode_surface
    trace=dict(tracker_resets=0,query_context_checks=0,mesh_decode_checks=0,
        all_tracker_betas_exact_zero=True,all_mesh_betas_exact_zero=True,poses=[],vertices=[],joints=[],betas=[])
    tsub_guard={'active':True,'blocked':0}
    def reject_tsub(event,args):
        if tsub_guard['active'] and event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)) and os.fsdecode(args[0]).endswith('_tsub.npy'):
            tsub_guard['blocked']+=1
            raise RuntimeError('mean-shape resource wrapper forbids opening unused tsub timestamps')
    sys.addaudithook(reject_tsub)
    class CheckedTracker(original_tracker):
        def reset(self,seq_id,boundary_ms,pose,betas,K_cam):
            if not np.array_equal(np.asarray(betas),np.zeros(10)):
                trace['all_tracker_betas_exact_zero']=False;raise AssertionError('nonzero inference shape reached tracker')
            trace['tracker_resets']+=1
            return super().reset(seq_id,boundary_ms,pose,betas,K_cam)
        def representation(self,cutoff_ms):
            if not np.array_equal(self.betas,np.zeros(10)):
                trace['all_tracker_betas_exact_zero']=False;raise AssertionError('nonzero shape reached model/query geometry')
            trace['query_context_checks']+=1
            return super().representation(cutoff_ms)
    def checked_decode(model,pose,betas):
        if not np.array_equal(np.asarray(betas),np.zeros(10)):
            trace['all_mesh_betas_exact_zero']=False;raise AssertionError('nonzero shape reached output MANO')
        surface,costs=original_decode(model,pose,betas)
        trace['mesh_decode_checks']+=1
        if engineering:
            trace['poses'].append(np.asarray(pose).copy());trace['betas'].append(np.asarray(betas).copy())
            trace['vertices'].append(surface['vertices_m'].copy());trace['joints'].append(surface['joints_m'].copy())
        return surface,costs
    def additional_identity(config,checkpoint,cfg,reader):
        identities=original_identity(config,checkpoint,cfg,reader)
        for path in (Path(__file__),Path(api.__file__),Path(R.__file__),Path(extension)):
            path=path.resolve();identities[str(path)]={'sha256':B.stream_sha(path),'bytes':path.stat().st_size}
        return identities
    try:
        B.ObservationReader=MeanShapeObservationReader;B.source_identity=additional_identity
        B.CausalRelationTracker=CheckedTracker;B.decode_surface=checked_decode
        with api.InferenceAccessGuard() as guard:
            trace['tsub_guard']=tsub_guard
            yield trace,guard
    finally:
        tsub_guard['active']=False
        B.ObservationReader=original_reader;B.source_identity=original_identity
        B.CausalRelationTracker=original_tracker;B.decode_surface=original_decode
        _PATCH_LOCK.release()


def formal_gate(arm,checkpoint,extension):
    R.require(extension.exists(),'public-mean extension must be frozen before formal run')
    doc=R.read(extension)
    R.require(doc.get('schema')=='s43_rel_mean_shape_deployment_extension_v1','public-mean extension identity mismatch')
    R.require(doc.get('frozen_main_protocol_sha256')==R.sha(R.DEFAULT/'protocol.json'),'public-mean extension no longer binds main protocol')
    for name,key in (('postprocess_complete.json','complete'),('finalization_complete.json','execution_complete')):
        path=R.DEFAULT/name
        R.require(path.exists() and R.read(path).get(key) is True,'wait for '+name+' before formal public-mean resources')
    if arm=='canonical':R.require(checkpoint is None,'canonical public-mean resource must retain original frozen weights')
    else:
        directory=R.DEFAULT/f'train/full_r1/{arm}'
        receipt=R.read(directory/'complete.json')
        R.require(receipt.get('completed_steps')==6000 and receipt.get('actual_outputs')==192000,'fixed final6000 training incomplete')
        R.require(checkpoint is not None and checkpoint.resolve()==(directory/'step=6000.ckpt').resolve(),'use fixed final weights-only deployment artifact')


def run(arm,config,checkpoint,out,*,device='cpu',engineering=False,extension=None,threads=2):
    extension=Path(extension or R.DEFAULT/'mean_shape_deployment_extension.json')
    out=Path(out)
    if out.exists():raise FileExistsError(out)
    R.require(arm in ARMS,'unknown public-mean resource arm')
    R.require(extension.exists(),'extension identity required even for read-only engineering receipt')
    if not engineering:formal_gate(arm,checkpoint,extension)
    elif str(device)!='cpu':raise ValueError('three-query engineering smoke is CPU-only')
    queries=3 if engineering else 64
    before=dict(reader=B.ObservationReader,identity=B.source_identity,tracker=B.CausalRelationTracker,decode=B.decode_surface)
    with tempfile.TemporaryDirectory(prefix='s43_public_mean_benchmark_') as tmp:
        with public_mean_context(extension,engineering) as (trace,guard):
            raw=B.benchmark(config,checkpoint,Path(tmp)/'surface.json',device=device,
                recording='ch_local',hop=50,queries=queries,threads=threads,root=B.DATA_ROOT)
            receipt=guard.receipt()
        for name,attribute in (('reader','ObservationReader'),('identity','source_identity'),('tracker','CausalRelationTracker'),('decode','decode_surface')):
            if getattr(B,attribute) is not before[name]:raise AssertionError('original benchmark globals were not restored')
    R.require(not receipt.get('aux_betas_accessed') and not receipt.get('pose_metadata_opened'),'forbidden public-mean inference field accessed')
    R.require(receipt.get('blocked_aux_beta_attempts',0)==0 and receipt.get('blocked_pose_meta_attempts',0)==0,'forbidden input access attempt during public-mean benchmark')
    R.require(trace['tsub_guard']['blocked']==0,'tsub field was accessed during public-mean resource measurement')
    R.require(trace['mesh_decode_checks']==queries and trace['all_tracker_betas_exact_zero'] and trace['all_mesh_betas_exact_zero'],'public-mean shape guard/call count mismatch')
    R.require(raw['arm']==('s43' if arm=='canonical' else arm),'public-mean label/config model arm mismatch')
    result=dict(raw)
    result.update(schema=SCHEMA,base_schema=raw['schema'],arm=arm,model_arm=raw['arm'],
        shape_condition='public_mean_beta0',inference_beta=[0.]*10,engineering_only=engineering,
        aux_betas_accessed=False,pose_metadata_opened=False,tsub_opened=False,access_guard=receipt,
        access_guard_receipt_captured_inside_protected_measurement=True,
        extension_sha256=R.sha(extension),wrapper_sha256=R.sha(__file__),
        wrapper_source=str(Path(__file__).resolve()),reader_source=str(Path(importlib.import_module('tools.s43_rel.mean_shape_replay').__file__).resolve()),
        module_globals_restored=True,tracker_beta0_checks=trace['query_context_checks'],mesh_beta0_checks=trace['mesh_decode_checks'],
        calibration_condition='Public zeros10 shape for model, canonical warmup, query geometry and output MANO; external K and valid-run metadata retained.',
        guard_cost_scope='Process-local beta assertions/access guards are included in observed wall time; no frozen implementation edited.',
        source_hash_interpretation='Auxiliary NPZ bytes are hashed as an opaque source identity; aux betas member is never accessed or decoded during this resource measurement.')
    if engineering:
        arrays=out.with_name(out.stem+'.engineering_outputs.npz')
        arrays.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(arrays,pred=np.asarray(trace['poses']),inference_beta=np.asarray(trace['betas']),
            vertices_m=np.asarray(trace['vertices']),joints_m=np.asarray(trace['joints']))
        result['engineering_surface_outputs']=str(arrays)
    R.dump(out,result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--config',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--device');p.add_argument('--threads',type=int,default=2);p.add_argument('--engineering',action='store_true')
    p.add_argument('--extension',type=Path,default=R.DEFAULT/'mean_shape_deployment_extension.json')
    args=p.parse_args();model_arm='s43' if args.arm=='canonical' else args.arm
    config=args.config or R.ROOT/f'configs/s43_rel/r1/{model_arm}.yaml'
    checkpoint=args.checkpoint
    if checkpoint is None and args.arm!='canonical':
        candidate=R.DEFAULT/f'train/full_r1/{args.arm}/step=6000.ckpt'
        checkpoint=candidate if candidate.exists() else None
    output=args.out or (R.DEFAULT/f'engineering/mean_shape_{args.arm}_cpu3.json' if args.engineering else R.DEFAULT/f'mean_shape/resources/{args.arm}_h50.json')
    device=args.device or ('cpu' if args.engineering else 'cuda:0')
    result=run(args.arm,config,checkpoint,output,device=device,engineering=args.engineering,extension=args.extension,threads=args.threads)
    print({'schema':result['schema'],'out':str(output),'queries':result['queries'],'beta0_mesh_checks':result['mesh_beta0_checks']})


if __name__=='__main__':main()
