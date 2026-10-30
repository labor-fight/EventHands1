"""Stage5-only invalid-geometry R1 wrapper, never edits frozen query/model.

Finite bad deployed poses retain their value. Invalid MANO local queries use
fixed full-image 9x10 patches of the same observation field with association
flag zero. No GT, pose clamp/reset/hold, loss, parameter or head change.
"""
from __future__ import annotations
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass,replace
import hashlib
from pathlib import Path
import time
import numpy as np
import torch

from .s43_field_model import S43FieldModel
from .field_query import FieldQueryBuilder,FieldQueryConfig,sample_query_patches
from . import witness_geometry as G
from tools.s43_integration.model import DEFAULT_CKPT,DEFAULT_CFG,load_legacy

VARIANT='invalid_geometry_uniform_r1'


@dataclass(frozen=True)
class QueryPlan:
    query:object
    association_flag:torch.Tensor
    metadata:dict


def prepare_query_cpu(task):
    """Independent lane geometry only; called serially or by ordered map.

    No neural module, RNG, input event field or mutable model state is touched.
    """
    p,beta,K,ref,T,history,mode,scale=task
    eventK=np.array(K,np.float64,copy=True);eventK[:2]*=scale
    builder=FieldQueryBuilder(p,beta,eventK,int(ref),history=history,
                              config=FieldQueryConfig(field_resolution=2))
    # Constructor validates all finite conditions/K/history before fallback.
    geometry=builder._ensure_geometry()
    joints=geometry['joints']
    if not np.isfinite(joints).all():raise ArithmeticError('nonfinite MANO query FK; no fallback for NaN/Inf')
    z=joints[1:,2];bad=z<=builder.config.near_plane_m
    reason='initial_finger_joint_z_at_or_below_near_plane' if bad.any() else None
    normal_valid=None
    if reason is None:
        query=builder.build(int(T),mode=mode)
        normal_valid=int(query.valid.sum())
        if normal_valid==0:reason='query_has_zero_valid_2d_interpolation_samples'
    if reason is not None:
        # event_grid with no event_counts is fixed full-image uniform; it
        # does not use MANO coordinates, an observed bbox, or target evidence.
        query=builder.build(int(T),mode='event_grid',event_counts=None)
        association=torch.zeros_like(query.in_view)
    else:association=query.in_view
    metadata=dict(query.meta,model_variant=VARIANT,requested_mode=mode,
        used_mode='fixed_full_image_uniform9x10' if reason is not None else mode,
        used_uniform_reacquisition=reason is not None,invalid_reason=reason,
        initial_finger_z_min_m=float(z.min()),initial_finger_z_max_m=float(z.max()),
        initial_finger_invalid_count=int(bad.sum()),declared_near_plane_m=builder.config.near_plane_m,
        normal_valid_sample_count=normal_valid,actual_2d_valid_samples=int(query.valid.sum()),
        actual_2d_in_view_samples=int(query.in_view.sum()),
        association_flag_nonzero=int(association.sum()),
        local_flag3='MANO association/in_view; forced0 in uniform reacquisition',
        fallback_finger_identity=False if reason is not None else query.meta.get('finger_identity_used',mode=='static'),
        reacquisition='read same field via fixed uniform centres; no pose/filter/reset/clamp/GT/eventbbox',
        parent_query_source_sha256=query.meta['source_sha256'])
    return QueryPlan(replace(query,meta=metadata),association,metadata)


class S43FieldReacquireModel(S43FieldModel):
    """Same model tensors/initialization; total local query for finite poses.

    workers=1 is the numeric reference; fixed16 only parallelizes CPU query
    preparation. B1 always serial. There is no NN or random work in workers.
    """
    def __init__(self,base,arm='static',*,use_prev_mlp=True,geometry_workers=16):
        if geometry_workers not in (1,16):raise ValueError('R1 permits fixed16 or serial1 geometry workers')
        super().__init__(base,arm,use_prev_mlp=use_prev_mlp)
        self.geometry_workers=int(geometry_workers);self._geometry_pool=None
        self.last_query_metadata=()
        self._observation_stats=dict(query_batches=0,query_rows=0,uniform_reacquisition_rows=0,
            initial_camera_invalid_rows=0,zero_valid_query_rows=0,
            initial_finger_z_min_m=None,initial_finger_z_max_m=None,
            serial_query_rows=0,parallel_query_rows=0,geometry_prepare_wall_seconds=0.)

    def prepare_queries_cpu(self,prev,beta,K,refs,cutoffs,histories):
        """One batched D2H then ordered independent CPU-only geometry tasks."""
        started=time.perf_counter();B=len(prev)
        conditions=torch.cat((prev.detach(),beta.detach(),K.detach().reshape(B,9)),dim=1).cpu().numpy().astype(np.float64)
        # Warm read-only constants once, avoiding duplicate first-use asset I/O
        # when several workers enter frozen FK caches together.
        G._joint_constants()
        for row in np.unique(conditions[:,51:61],axis=0):G._shape_terms(np.ascontiguousarray(row).tobytes())
        tasks=[(c[:51],c[51:61],c[61:].reshape(3,3),int(refs[i]),int(cutoffs[i]),histories[i],
                self.arm,float(self.base.render_scale)) for i,c in enumerate(conditions)]
        parallel=B>1 and self.geometry_workers>1
        if parallel:
            if self._geometry_pool is None:
                self._geometry_pool=ThreadPoolExecutor(max_workers=self.geometry_workers,thread_name_prefix='s43-r1-geometry')
            plans=list(self._geometry_pool.map(prepare_query_cpu,tasks))
        else:plans=[prepare_query_cpu(t) for t in tasks]
        self._observation_stats['geometry_prepare_wall_seconds']+=time.perf_counter()-started
        self._observation_stats['parallel_query_rows' if parallel else 'serial_query_rows']+=B
        return plans

    def _local_patches(self,field,prev,beta,K,refs,cutoffs,histories):
        B=len(field)
        fine=field[:,:16].reshape(B,8,2,180,240)[:,:,:,::2,::2]
        coverage=field[:,16:24,::2,::2].unsqueeze(2);cell=torch.cat((fine,coverage),dim=2)
        plans=self.prepare_queries_cpu(prev,beta,K,refs,cutoffs,histories)
        patches=[];metadata=[]
        for lane,plan in enumerate(plans):
            q=plan.query
            sampled=sample_query_patches(cell[lane],q).reshape(90,8,7,7,3).permute(1,0,4,2,3).contiguous()
            # Values/finecoverage follow actual2D valid support. Channel3 is
            # association, not physical sampling validity, during fallback.
            flags=torch.stack((plan.association_flag,q.full_interp_support),dim=2).reshape(8,90,2,7,7).to(field)
            patches.append(torch.cat((sampled,flags),dim=2));metadata.append(plan.metadata)
        stats=self._observation_stats;stats['query_batches']+=1;stats['query_rows']+=B
        stats['uniform_reacquisition_rows']+=sum(m['used_uniform_reacquisition'] for m in metadata)
        stats['initial_camera_invalid_rows']+=sum(m['initial_finger_invalid_count']>0 for m in metadata)
        stats['zero_valid_query_rows']+=sum(m['invalid_reason']=='query_has_zero_valid_2d_interpolation_samples' for m in metadata)
        for m in metadata:
            stats['initial_finger_z_min_m']=m['initial_finger_z_min_m'] if stats['initial_finger_z_min_m'] is None else min(stats['initial_finger_z_min_m'],m['initial_finger_z_min_m'])
            stats['initial_finger_z_max_m']=m['initial_finger_z_max_m'] if stats['initial_finger_z_max_m'] is None else max(stats['initial_finger_z_max_m'],m['initial_finger_z_max_m'])
        self.last_query_metadata=tuple(copy.deepcopy(metadata))
        return torch.stack(patches),metadata

    @property
    def observation_stats(self):return copy.deepcopy(self._observation_stats)

    def diagnostic_state_dict(self):return dict(stats=self.observation_stats,last_query_metadata=copy.deepcopy(self.last_query_metadata))

    def load_diagnostic_state_dict(self,state):
        if set(state['stats'])!=set(self._observation_stats):raise ValueError('R1 diagnostic schema differs')
        self._observation_stats=copy.deepcopy(state['stats']);self.last_query_metadata=copy.deepcopy(state['last_query_metadata'])

    def close_geometry_workers(self):
        if self._geometry_pool is not None:self._geometry_pool.shutdown(wait=True);self._geometry_pool=None

    def metadata(self):
        return dict(super().metadata(),schema='s43_field_frontend_reacquisition_R1_v1',model_variant=VARIANT,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            frozen_parent_source_sha256=hashlib.sha256(Path(__file__).with_name('s43_field_model.py').read_bytes()).hexdigest(),
            geometry_workers=self.geometry_workers,B1_geometry_serial=True,
            geometry_parallel_scope='ordered CPUquery only; one batch conditionsD2H; no NN/RNG in workers',
            fallback='any20fingerjointz<=.03 or zero2Dvalid samples -> fixedfullimageuniform9x10',
            association_flag3='zero in fallback; sameF2/finecoverage and actual2Dinterp flag retained',
            observation_stats=self.observation_stats)


def load_s43_field_reacquire_model(*,arm='static',use_prev_mlp=True,geometry_workers=16,device='cpu',
        base_checkpoint=DEFAULT_CKPT,base_config=DEFAULT_CFG,checkpoint=None):
    base,cfg=load_legacy(base_checkpoint,base_config,device)
    model=S43FieldReacquireModel(base,arm,use_prev_mlp=use_prev_mlp,geometry_workers=geometry_workers).to(device)
    if checkpoint is not None:
        p=torch.load(checkpoint,map_location=device)
        if (p.get('schema')!='s43_field_reacquisition_checkpoint_v1' or p.get('model_variant')!=VARIANT
                or p.get('arm')!=arm or p.get('use_prev_mlp')!=use_prev_mlp):raise ValueError('explicit R1 checkpoint variant required')
        model.load_state_dict(p['model_state_dict'],strict=True)
    return model,dict(model.metadata(),original_base_config=copy.deepcopy(cfg))
