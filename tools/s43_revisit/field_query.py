"""Causal field2 patch coordinates from deployed/public MANO conditions.

This prototype uses15finger bones,3interior anchors,2legal MCP actions,
8native-time bins and7x7 patches. It performs no event likelihood, visibility,
target access, memory update or feedback. Full-image field2 remains a separate
reader input; narrow queries must not delete the observation field.
"""
from __future__ import annotations
from dataclasses import dataclass,asdict
import hashlib
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from . import witness_geometry as G

MODES=('anatomical','static','generic_trajectory','event_grid','hand_box')


def _array(x,shape,name):
    if hasattr(x,'detach'):x=x.detach().cpu().numpy()
    a=np.asarray(x,np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'finite {name}{shape} required')
    a=a.copy();a.flags.writeable=False;return a


def _integer(x,name):
    if isinstance(x,(bool,np.bool_)) or not np.isscalar(x) or not np.isfinite(x) or int(x)!=x:
        raise ValueError(f'integer native-ms {name} required')
    return int(x)


def _tensor(x,dtype=None):return torch.tensor(np.array(x,copy=True),dtype=dtype)


@dataclass(frozen=True)
class FieldQueryConfig:
    width:int=240
    height:int=180
    field_resolution:int=2
    window_ms:int=50
    time_bins:int=8
    patch_size:int=7
    patch_spacing_px:float=2.
    anchor_fractions:tuple=(.25,.5,.75)
    near_plane_m:float=.03
    projected_bone_epsilon_px:float=1e-6
    hand_box_margin_px:float=8.
    hand_box_min_extent_px:float=32.
    event_layout:str='bbox'

    def __post_init__(self):
        if (self.width,self.height,self.window_ms,self.time_bins,self.patch_size)!=(240,180,50,8,7):
            raise ValueError('prototype fixes240x180,W50,8bins,7x7patches')
        if self.field_resolution not in (1,2) or self.anchor_fractions!=(.25,.5,.75):
            raise ValueError('resolution1/2 and3fixed interior anchors required')
        if self.event_layout not in ('bbox','uniform'):raise ValueError('event layout bbox/uniform required')
        values=(self.patch_spacing_px,self.near_plane_m,self.projected_bone_epsilon_px,
                self.hand_box_margin_px,self.hand_box_min_extent_px)
        if not np.isfinite(values).all() or min(values)<=0:raise ValueError('positive finite query scales required')


@dataclass(frozen=True)
class QueryCoordinates:
    grid:torch.Tensor                   # [8,90,49,2],normalized align_corners=False
    valid:torch.Tensor                  # geometry AND full cell interpolation support
    in_view:torch.Tensor                # geometry AND physical pixel-center bounds
    full_interp_support:torch.Tensor   # no coarse-cell zero-padding needed
    centers_px:torch.Tensor            # [8,90,2], raw image pixels
    frame_degenerate:torch.Tensor       # [90],fallback image axes
    depth_valid:torch.Tensor            # [8,90]
    bin_edges_ms:torch.Tensor           # [9],halfopen native intervals
    bin_times_ms:torch.Tensor           # [8],mean of native integer stamps
    history_supported:torch.Tensor     # [8],inside available deployed-history knots
    meta:dict

    def sample(self,binned_field):return sample_query_patches(binned_field,self)


def sample_query_patches(binned_field,query):
    """Bilinear cell-grid sample; result[...,5,3,3,2,8,7,7,C].

    Input[8,C,H/r,W/r] or[B,8,C,H/r,W/r]; field2 density/count channels
    must be the actual90x120cells, not the nearest-upsampled240pximage.
    Geometry is detached; gradients may flow to the field/reader tensors.
    """
    if not isinstance(binned_field,torch.Tensor) or not binned_field.is_floating_point():
        raise ValueError('floating torch cell-grid tensor required')
    x=binned_field;batched=x.ndim==5
    if not batched:x=x.unsqueeze(0)
    if x.ndim!=5 or x.shape[1]!=8 or tuple(x.shape[-2:])!=tuple(query.meta['field_shape']):
        raise ValueError('input[B,8,C,fieldH,fieldW] does not match query resolution')
    B,_,C,H,W=x.shape
    grid=query.grid.to(device=x.device,dtype=x.dtype).unsqueeze(0).expand(B,-1,-1,-1,-1).reshape(B*8,90,49,2)
    sampled=F.grid_sample(x.reshape(B*8,C,H,W),grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    sampled=sampled.reshape(B,8,C,90,49)*query.valid.to(device=x.device)[None,:,None]
    sampled=sampled.permute(0,3,1,4,2).reshape(B,5,3,3,2,8,7,7,C)
    return sampled if batched else sampled[0]


def _rotations(p):
    a=np.asarray(p)[3:].copy();a[3:]+=G._joint_constants()['mean']
    return G.rotation_exp(a.reshape(16,3))


def interpolate_deployed_pose(a,b,fraction):
    """Translation linear,16 actual rotations on right-local SO(3) geodesics."""
    a=_array(a,(51,),'posea');b=_array(b,(51,),'poseb');u=float(fraction)
    if not np.isfinite(u) or not 0<=u<=1:raise ValueError('interpolation fraction in[0,1] required')
    ra,rb=_rotations(a),_rotations(b)
    omega=Rotation.from_matrix(ra.transpose(0,2,1)@rb).as_rotvec()
    r=ra@G.rotation_exp(u*omega);aa=Rotation.from_matrix(r).as_rotvec().reshape(48)
    aa[3:]-=G._joint_constants()['mean']
    return np.r_[(1-u)*a[:3]+u*b[:3],aa]


def _anchors(joints,fractions):
    # joints[...,21,3]→[...,5,3bone,3anchor,3XYZ].
    finger=np.take(joints,G.FINGER_JOINTS,axis=-2)
    alpha=np.asarray(fractions)[:,None]
    return finger[...,:-1,None,:]*(1-alpha)+finger[...,1:,None,:]*alpha


def _frame(start,end,epsilon):
    d=end-start;length=np.linalg.norm(d,axis=-1);bad=(length<epsilon)|~np.isfinite(length)
    tangent=d/np.maximum(length[...,None],epsilon)
    tangent=np.where(bad[...,None],np.asarray([1.,0.]),tangent)
    normal=np.stack((-tangent[...,1],tangent[...,0]),-1)
    return np.stack((tangent,normal),-2),bad


def _uniform_box(box):
    left,top,right,bottom=map(float,box)
    x=left+(np.arange(10)+.5)/10*(right-left)
    y=top+(np.arange(9)+.5)/9*(bottom-top)
    xx,yy=np.meshgrid(x,y);return np.stack((xx,yy),-1).reshape(90,2)


class FieldQueryBuilder:
    def __init__(self,anchor51,beta,K_event,ref_ms,*,history=(),config=FieldQueryConfig()):
        self.anchor=_array(anchor51,(51,),'anchor');self.beta=_array(beta,(10,),'beta')
        self.K=_array(K_event,(3,3),'eventK');self.ref_ms=_integer(ref_ms,'ref_ms');self.config=config
        if self.K[0,0]<=0 or self.K[1,1]<=0 or not np.allclose(self.K[2],[0,0,1],atol=1e-12,rtol=0):
            raise ValueError('positive-focal pinhole event-grid K required; no internal scaling')
        records=[]
        previous_time=None
        for t,p in history:
            t=_integer(t,'history time');pose=_array(p,(51,),'deployed history pose')
            if t>self.ref_ms or (previous_time is not None and t<=previous_time):raise ValueError('strictly ordered deployed history, no future commits')
            previous_time=t
            if t==self.ref_ms:
                if not np.array_equal(pose,self.anchor):raise ValueError('history atref disagrees with provided anchor')
            else:records.append((t,pose))
        self.history=tuple(records+[(self.ref_ms,self.anchor)])
        self._geo=None;self._calibration=None;self._endpoint_states=None

    def _project(self,xyz):
        h=xyz@self.K.T;z=xyz[...,2];safe=np.where(np.abs(h[...,2])>1e-12,h[...,2],1.)
        xy=h[...,:2]/safe[...,None]
        good=(z>self.config.near_plane_m)&np.isfinite(xy).all(-1)
        return xy,good

    def _ensure_geometry(self,motion=False):
        if self._geo is None:
            j=G.fk_joints(self.anchor,self.beta);xy,valid=self._project(j)
            f=np.take(xy,G.FINGER_JOINTS,axis=-2)
            basis,degenerate=_frame(f[:,:-1],f[:,1:],self.config.projected_bone_epsilon_px)
            fvalid=np.take(valid,G.FINGER_JOINTS)
            centers,center_valid=self._project(_anchors(j,self.config.anchor_fractions))
            self._geo=dict(joints=j,projected_joints=xy,joint_valid=valid,centers=centers,center_valid=center_valid,
                basis=basis,degenerate=degenerate,frame_valid=fvalid[:,:-1]&fvalid[:,1:])
        if motion and self._endpoint_states is None:
            self._endpoint_states,self._calibration=G.motion_endpoints(self.anchor,self.beta)
        return self._geo

    def _past(self,s):
        records=self.history
        if s<=records[0][0]:return records[0][1].copy(),s==records[0][0]
        for (ta,a),(tb,b) in zip(records[:-1],records[1:]):
            if ta<=s<=tb:return interpolate_deployed_pose(a,b,(s-ta)/(tb-ta)),True
        return self.anchor.copy(),False

    def _causal_centers(self,times,mode):
        self._ensure_geometry(motion=True)
        bases=[];supported=[]
        axes=G.public_flexion_axes()
        for s in times:
            base,known=self._past(s);supported.append(known);bases.append(base)
        bases=np.asarray(bases);alpha=np.asarray(self.config.anchor_fractions)[:,None]
        if mode=='anatomical':
            # Compose actual matrices and batch sparse21-joint FK, without80
            # separate Exp/Log chart roundtrips or complete mesh renderings.
            rotation=np.asarray([_rotations(p) for p in bases])
            candidate=np.broadcast_to(rotation[:,None,None],(8,5,2,16,3,3)).copy()
            phase=np.clip((times-self.ref_ms)/50.,0.,1.)
            for j,row in enumerate(axes):
                direction=np.eye(3)[row['axis']]*row['sign']
                angles=np.asarray(self._calibration['angles_rad'][j])*np.asarray([1,-1])
                delta=G.rotation_exp(direction[None,None]*phase[:,None,None]*angles[None,:,None])
                candidate[:,j,:,row['mcp_mano']]=rotation[:,None,row['mcp_mano']]@delta
            translation=np.broadcast_to(bases[:,None,None,:3],(8,5,2,3))
            joints=G._fk_rotations(candidate.reshape(80,16,3,3),translation.reshape(80,3),self.beta).reshape(8,5,2,21,3)
            own=np.stack([np.take(joints[:,j],G.FINGER_JOINTS[j],axis=2) for j in range(5)],axis=1)
            centers3d=own[...,:-1,None,:]*(1-alpha)+own[...,1:,None,:]*alpha
            centers,valid=self._project(centers3d.transpose(0,1,3,4,2,5))
        else:
            past=G.fk_joints(bases,self.beta)
            xy,ok=self._project(_anchors(past,self.config.anchor_fractions))
            centers=np.repeat(xy[...,None,:],2,axis=-2);valid=np.repeat(ok[...,None],2,axis=-1)
        if mode=='generic_trajectory':
            endpoint_j=G.fk_joints(self._endpoint_states,self.beta)
            endpoint_own=np.stack([np.take(endpoint_j[j],G.FINGER_JOINTS[j],axis=1) for j in range(5)])
            end3d=endpoint_own[...,:-1,None,:]*(1-alpha)+endpoint_own[...,1:,None,:]*alpha
            endpoint_xy,endvalid=self._project(end3d.transpose(0,2,3,1,4))
            start=self._geo['centers'][...,None,:]
            start_valid=self._geo['center_valid'][...,None]
            for t,s in enumerate(times):
                if s>self.ref_ms:
                    phase=np.clip((s-self.ref_ms)/50.,0.,1.)
                    centers[t]=start+phase*(endpoint_xy-start)
                    valid[t]=start_valid&endvalid
        return centers,valid,np.asarray(supported,bool)

    def _event_box(self,event_counts):
        c=self.config
        default=(-.5,-.5,c.width-.5,c.height-.5)
        if event_counts is None or c.event_layout=='uniform':return default,'uniform_full_image'
        if hasattr(event_counts,'detach'):event_counts=event_counts.detach().cpu().numpy()
        a=np.asarray(event_counts)
        if not np.isfinite(a).all() or np.any(a<0):raise ValueError('nonnegative finite observed event counts/support required')
        if a.ndim>=3 and a.shape[-3:]==(90,120,2):support=np.any(a>0,axis=tuple(range(a.ndim-3))+(a.ndim-1,));r=2
        elif a.ndim>=2 and a.shape[-2:] in ((90,120),(180,240)):
            support=np.any(a>0,axis=tuple(range(a.ndim-2))) if a.ndim>2 else a>0
            r=2 if support.shape==(90,120) else 1
        else:raise ValueError('event counts must expose90x120cells(or180x240pixels), optionaltime/polarity axes')
        yy,xx=np.nonzero(support)
        if not len(xx):return default,'empty_fallback_full_image'
        return (float(xx.min()*r-.5),float(yy.min()*r-.5),float((xx.max()+1)*r-.5),float((yy.max()+1)*r-.5)),'observed_event_bbox'

    def build(self,cutoff_ms,*,mode='anatomical',event_counts=None):
        started=time.perf_counter();T=_integer(cutoff_ms,'cutoff_ms');c=self.config
        if mode not in MODES or T<self.ref_ms:raise ValueError('known querymode and cutoff>=last deployed commit required')
        edges=T-50+(np.arange(9)*50+7)//8
        times=(edges[:-1]+edges[1:]-1)/2.  # mean native integer stamps; not interval midpoint
        supported=np.asarray([self.history[0][0]<=s<=self.ref_ms for s in times])
        auxiliary={}
        if mode=='event_grid':
            box,reason=self._event_box(event_counts);flat=_uniform_box(box)
            centers=np.broadcast_to(flat,(8,90,2)).copy();depth=np.ones((8,90),bool)
            basis=np.broadcast_to(np.eye(2),(90,2,2));degen=np.zeros(90,bool)
            frame_valid=np.ones(90,bool);auxiliary=dict(query_box=list(box),bbox_source=reason,MANO_used=False)
        elif mode=='hand_box':
            g=self._ensure_geometry();xy=g['projected_joints'][1:];good=g['joint_valid'][1:]
            if not good.all():raise ValueError('hand_box requires all initial finger joints infront of nearplane')
            lo=xy.min(0);hi=xy.max(0);center=(lo+hi)/2
            half=np.maximum((hi-lo)/2+c.hand_box_margin_px,c.hand_box_min_extent_px/2)
            box=(*list(center-half),*list(center+half));flat=_uniform_box(box)
            centers=np.broadcast_to(flat,(8,90,2)).copy();depth=np.ones((8,90),bool)
            basis=np.broadcast_to(np.eye(2),(90,2,2));degen=np.zeros(90,bool);frame_valid=np.ones(90,bool)
            auxiliary=dict(query_box=list(box),MANO_used=True,finger_identity_used=False)
        else:
            g=self._ensure_geometry()
            if mode=='static':
                centers=np.broadcast_to(g['centers'][...,None,:],(8,5,3,3,2,2)).copy()
                depth=np.broadcast_to(g['center_valid'][...,None],(8,5,3,3,2)).copy()
            else:centers,depth,supported=self._causal_centers(times,mode)
            centers=centers.reshape(8,90,2);depth=depth.reshape(8,90)
            basis=np.broadcast_to(g['basis'][:,:,None,None],(5,3,3,2,2,2)).reshape(90,2,2)
            degen=np.broadcast_to(g['degenerate'][:,:,None,None],(5,3,3,2)).reshape(90)
            frame_valid=np.broadcast_to(g['frame_valid'][:,:,None,None],(5,3,3,2)).reshape(90)
            auxiliary=dict(MANO_used=True,calibration=self._calibration if mode!='static' else None)
        offset=(np.arange(7)-3)*c.patch_spacing_px;oy,ox=np.meshgrid(offset,offset,indexing='ij')
        patch=np.stack((ox,oy),-1).reshape(49,2)
        xy=centers[:,:,None]+np.einsum('pk,nkd->npd',patch,basis)[None]
        finite=np.isfinite(xy).all(-1);geometric=depth[:,:,None]&frame_valid[None,:,None]&finite
        in_view=geometric&(xy[...,0]>=0)&(xy[...,0]<=239)&(xy[...,1]>=0)&(xy[...,1]<=179)
        fringe=(c.field_resolution-1)/2
        support=geometric&(xy[...,0]>=fringe)&(xy[...,0]<=239-fringe)&(xy[...,1]>=fringe)&(xy[...,1]<=179-fringe)
        normalized=2*(xy+.5)/np.asarray([240,180])-1
        normalized=np.where(geometric[...,None],normalized,2.)
        tensors=dict(grid=_tensor(normalized,torch.float32),valid=_tensor(support,torch.bool),in_view=_tensor(in_view,torch.bool),
            full_interp_support=_tensor(support,torch.bool),centers_px=_tensor(np.where(np.isfinite(centers),centers,0.),torch.float32),
            frame_degenerate=_tensor(degen,torch.bool),depth_valid=_tensor(depth,torch.bool),
            bin_edges_ms=_tensor(edges,torch.int64),bin_times_ms=_tensor(times,torch.float64),history_supported=_tensor(supported,torch.bool))
        h=hashlib.sha256()
        for a in (self.anchor,self.beta,self.K):h.update(a.tobytes())
        for t,p in self.history:h.update(str(t-self.ref_ms).encode());h.update(p.tobytes())
        meta=dict(mode=mode,config=asdict(c),field_shape=[180//c.field_resolution,240//c.field_resolution],
            axes='finger5,bone3,interior_anchor3,hypothesis(+,-)2,time8,patch_y7,patch_x7,channelC',
            grid_axes='time8,flatten(finger,bone,anchor,hypothesis)90,flatten(patch_y,patch_x)49,xy2',
            K_space='event240x180; no internal scaling',align_corners=False,
            pixel_to_normalized='2*(eventxy+.5)/[240,180]-1; inputcell center r*(index+.5)-.5',
            time_representative='arithmetic mean of native integer stamps[left,right); .5 permitted, not invented sub-ms event order',
            path='past deployed commits interpolate translation and actual SO3; before first knot hold+markunsupported; afterref hypothetical MCP actionphase clip0..1, other velocityzero',
            patch_frame='fixed at latest anchor projectedbone tangent/normal; image axes for degeneratebone or boxcontrols',
            generic='same actual MANO projected endpoints, straight image-space center path; past path shared with anatomical',
            static='same latest anchor centers at every time; bothhyp slots retained, no forced logit sharing',
            validity='independent geometry/pixel-view/fullcoarse-interpolation flags; exposure comes from observationfield separately',
            near_plane='unsupported geometry masked, not a held prediction; no physicalvisibility/occlusion claim',
            coarse_branch_required='reader also consumes unchanged full-image field2; narrowquery does not filter observationmemory',
            decoder='independent10class/63proxy head; no scoregate hardwired to hypothesis or sign',
            shape_condition='beta0' if not np.any(self.beta) else 'external supplied beta conditional',
            ref_ms=self.ref_ms,cutoff_ms=T,history_knots=[t for t,_ in self.history],condition_sha256=h.hexdigest(),
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            witness_geometry_source_sha256=hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest(),
            mano_asset_sha256=G._asset_sha() if auxiliary.get('MANO_used') else None,
            coordinate_tensor_bytes=sum(v.numel()*v.element_size() for v in tensors.values()),
            two_polarity_sample_values=70560,two_polarity_float32_payload_bytes=282240,
            build_seconds=time.perf_counter()-started,**auxiliary)
        return QueryCoordinates(**tensors,meta=meta)
