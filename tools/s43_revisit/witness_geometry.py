"""Public/deployed MANO motion kernels for bounded episode selection.

These are geometric selection responses, never likelihoods.  Joint/tip FK is
exact MANO LBS restricted to the five required tip vertices.  The subsequent
kernel is an explicit approximation: project only trajectory endpoints and
linearly interpolate the *image* bone endpoints, sampled on a 4px grid.
No visible-surface or event-emission interpretation is attached to that tube.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
from tools.s43_rel.geometry import rotation_exp

ROOT=Path(__file__).resolve().parents[2]
ASSET=ROOT/'assets/mano_right.npz'
FINGER_NAMES=('thumb','index','middle','ring','pinky')
MCP_MANO=(13,1,4,10,7)
FINGER_JOINTS=np.asarray([np.arange(1+4*j,5+4*j) for j in range(5)])
PUBLIC_INITIAL=np.asarray([0.,0.,.6]+[0.]*48,np.float64)


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _finite(x,shape,name):
    if hasattr(x,'detach'):x=x.detach().cpu().numpy()
    a=np.asarray(x,np.float64)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError(f'{name}: finite {shape} required')
    return a.copy()
def _ro(a):
    a=np.ascontiguousarray(a);a.flags.writeable=False;return a


@lru_cache(maxsize=1)
def _joint_constants():
    """Keep only exact rest joints and five fingertip dependencies, not778v."""
    with np.load(ASSET,allow_pickle=False) as z:
        tips=np.asarray(z['fingertip_vertex_ids'],np.int64)
        reg=np.asarray(z['J_regressor'],np.float64)
        vt=np.asarray(z['v_template'],np.float64);sd=np.asarray(z['shapedirs'],np.float64)
        c=dict(J_template=reg@vt,J_dirs=np.einsum('jv,vkl->jkl',reg,sd),
            tip_template=vt[tips],tip_dirs=sd[tips],
            tip_pose=np.asarray(z['posedirs'],np.float64)[tips].reshape(15,135),
            weights=np.asarray(z['weights'],np.float64)[tips],
            mean=np.asarray(z['hands_mean'],np.float64),
            parents=np.asarray(z['kintree_table'][0],np.int64),
            reorder=np.asarray(z['mano16_tips_to_openpose21'],np.int64))
    c['parents'][0]=-1
    return {k:_ro(v) for k,v in c.items()}


@lru_cache(maxsize=16)
def _shape_terms(beta_bytes):
    beta=np.frombuffer(beta_bytes,np.float64);c=_joint_constants()
    return (_ro(c['J_template']+np.einsum('jkl,l->jk',c['J_dirs'],beta)),
            _ro(c['tip_template']+np.einsum('vkl,l->vk',c['tip_dirs'],beta)))


def fk_joints(states,beta):
    """Exact matrix-MANO21 joints, residual AA + hand mean exactly once.

    Only five fingertip LBS rows are needed; all nonzero influences and pose
    correctives for these rows are retained.  Returns metres, float64.
    """
    a=np.asarray(states,np.float64);b=_finite(beta,(10,),'beta')
    if a.shape[-1:]!=(51,) or not np.isfinite(a).all():raise ValueError('finite states[...,51] required')
    lead=a.shape[:-1];p=a.reshape(-1,51);c=_joint_constants()
    angles=p[:,3:].copy();angles[:,3:]+=c['mean']
    rot=rotation_exp(angles.reshape(-1,16,3))
    return _fk_rotations(rot,p[:,:3],b).reshape(lead+(21,3))


def _fk_rotations(rot,translation,beta):
    """Matrix input also avoids Exp/Log roundtrips in scale calibration."""
    c=_joint_constants();J,tips=_shape_terms(beta.tobytes());n=len(rot)
    rel=J.copy();rel[1:]-=J[c['parents'][1:]]
    local=np.broadcast_to(np.eye(4),(n,16,4,4)).copy()
    local[:,:,:3,:3]=rot;local[:,:,:3,3]=rel
    T=np.empty_like(local);T[:,0]=local[:,0]
    for j in range(1,16):T[:,j]=T[:,c['parents'][j]]@local[:,j]
    posed=T[:,:,:3,3].copy()
    T[:,:,:3,3]-=np.einsum('bjkl,jl->bjk',T[:,:,:3,:3],J)
    vp=tips+(rot[:,1:]-np.eye(3)).reshape(n,135).dot(c['tip_pose'].T).reshape(n,5,3)
    blend=np.einsum('vj,bjkl->bvkl',c['weights'],T)
    tip=np.einsum('bvkl,bvl->bvk',blend[:,:,:3,:3],vp)+blend[:,:,:3,3]
    return np.concatenate((posed,tip),axis=1)[:,c['reorder']]+translation[:,None,:]


def right_retract(state,joint,axis,angle):
    """Right-local SO(3) action, with robust principal Log and mean removal."""
    p=_finite(state,(51,),'state');v=_finite(axis,(3,),'axis')*float(angle)
    if not np.isfinite(v).all() or not 0<=int(joint)<=15:raise ValueError('invalid right action')
    c=_joint_constants();start=3+3*int(joint)
    mean=np.zeros(3) if joint==0 else c['mean'][3*(joint-1):3*joint]
    old=rotation_exp(p[start:start+3]+mean)
    p[start:start+3]=Rotation.from_matrix(old@rotation_exp(v)).as_rotvec()-mean
    return p


@lru_cache(maxsize=1)
def public_flexion_axes():
    """Choose axes from public beta0 mean only, by wrist-tip shortening."""
    base=fk_joints(PUBLIC_INITIAL,np.zeros(10));rows=[]
    for finger,joint in enumerate(MCP_MANO):
        candidates=[]
        for axis in range(3):
            for sign in (-1,1):
                direction=np.eye(3)[axis]*sign
                altered=fk_joints(right_retract(PUBLIC_INITIAL,joint,direction,.3),np.zeros(10))
                idx=FINGER_JOINTS[finger]
                shortening=(np.linalg.norm(base[idx[-1]]-base[0])-np.linalg.norm(altered[idx[-1]]-altered[0]))*1000
                movement=np.linalg.norm((altered[idx]-altered[0])-(base[idx]-base[0]),axis=-1).mean()*1000
                candidates.append(dict(axis=axis,sign=sign,shortening_mm=float(shortening),mean_joint_displacement_mm=float(movement)))
        best=max(candidates,key=lambda x:x['shortening_mm'])
        if best['shortening_mm']<=0 or best['mean_joint_displacement_mm']<.5:raise RuntimeError('public MCP flexion check failed')
        rows.append(dict(finger=FINGER_NAMES[finger],mcp_mano=joint,axis=best['axis'],sign=best['sign'],
                         public_check_angle_rad=.3,candidates=candidates))
    return tuple(rows)


def motion_endpoints(state,beta,target_mm=5.,max_angle_rad=.6):
    """Ten real MCP actions, separately FK-scaled to target or recorded cap."""
    p=_finite(state,(51,),'state');beta=_finite(beta,(10,),'beta')
    if not np.isfinite([target_mm,max_angle_rad]).all() or min(target_mm,max_angle_rad)<=0:raise ValueError('positive action scale required')
    public=public_flexion_axes();initial=fk_joints(p,beta)
    angles0=p[3:].copy();angles0[3:]+=_joint_constants()['mean']
    base=rotation_exp(angles0.reshape(16,3));mcp=np.repeat(MCP_MANO,2)
    axes=np.asarray([np.eye(3)[row['axis']]*row['sign']*sign for row in public for sign in (1,-1)])
    def rotations_at(angles):
        rot=np.broadcast_to(base,(10,16,3,3)).copy()
        rot[np.arange(10),mcp]=base[mcp]@rotation_exp(axes*angles[:,None])
        return rot
    def displacement(angles):
        jt=_fk_rotations(rotations_at(angles),np.broadcast_to(p[:3],(10,3)),beta).reshape(5,2,21,3)
        return np.asarray([[np.linalg.norm((jt[j,k,FINGER_JOINTS[j]]-jt[j,k,0])-
                         (initial[FINGER_JOINTS[j]]-initial[0]),axis=-1).mean()*1000
                         for k in range(2)] for j in range(5)]).ravel()
    cap=displacement(np.full(10,max_angle_rad));reachable=cap>=target_mm
    low=np.zeros(10);high=np.full(10,max_angle_rad)
    for _ in range(16):
        mid=(low+high)/2;distance=displacement(mid)
        below=distance<target_mm;low=np.where(below,mid,low);high=np.where(below,high,mid)
    angles=np.where(reachable,(low+high)/2,max_angle_rad)
    rotations=rotations_at(angles);endpoints=np.broadcast_to(p,(10,51)).copy()
    chart=Rotation.from_matrix(rotations[np.arange(10),mcp]).as_rotvec()
    mean=_joint_constants()['mean'].reshape(15,3)
    for k,j in enumerate(mcp):endpoints[k,3+3*j:6+3*j]=chart[k]-mean[j-1]
    actual=displacement(angles)
    return endpoints.reshape(5,2,51),dict(target_mean_joint_displacement_mm=float(target_mm),
        max_angle_rad=float(max_angle_rad),angles_rad=angles.reshape(5,2).tolist(),
        actual_mm=actual.reshape(5,2).tolist(),target_reached=reachable.reshape(5,2).tolist(),
        calibration='exact21-joint FK; mean displacement of four wrist-relative joints; fixed public MCP axis, no events or labels')


@dataclass(frozen=True)
class MotionConfig:
    width:int=240
    height:int=180
    epoch_ms:int=10
    trajectory_ms:int=50
    sigma_px:float=4.
    coarse_cell:int=4
    target_mm:float=5.
    max_angle_rad:float=.6

    def __post_init__(self):
        if (self.width,self.height,self.epoch_ms,self.trajectory_ms,self.coarse_cell)!=(240,180,10,50,4):
            raise ValueError('frozen prototype uses240x180,epoch10ms,trajectory50ms,grid4px')
        if not np.isfinite([self.sigma_px,self.target_mm,self.max_angle_rad]).all() or min(self.sigma_px,self.target_mm,self.max_angle_rad)<=0:
            raise ValueError('positive finite scales required')


def _segment_kernel(points,bones,sigma):
    # bones[...,4,2], three MCP→PIP→DIP→tip segments, no wrist spoke.
    # Separate3 segments bounds scratch; no [hyp,t,H,W,3,2] allocation.
    # Kernel arithmetic is float32, separately declared from float64 exactFK.
    bones=np.asarray(bones,np.float32);xx=points[...,0,0].astype(np.float32);yy=points[...,0,1].astype(np.float32)
    dist=None
    for k in range(3):
        ax=bones[...,k,0,None,None];ay=bones[...,k,1,None,None]
        dx=bones[...,k+1,0,None,None]-ax;dy=bones[...,k+1,1,None,None]-ay
        ox=xx-ax;oy=yy-ay
        u=np.clip((ox*dx+oy*dy)/np.maximum(dx*dx+dy*dy,1e-12),0.,1.)
        d=(ox-u*dx)**2+(oy-u*dy)**2
        dist=d if dist is None else np.minimum(dist,d)
    return np.exp(-dist/(2*sigma*sigma))


class MotionPolicy:
    """Immutable deployed anchor, plus one bounded derived epoch-LUT cache.

    K is already on240x180 pixels.  Creating a new policy is an explicit pose
    commit; episode_scores never changes the anchor.  Cache replacement is
    computational only and does not change any observation/selection state.
    """
    def __init__(self,prev51,beta,K_event,ref_ms,*,mode='witness',config=MotionConfig()):
        started=time.perf_counter()
        if mode not in ('witness','quota','generic_temporal'):raise ValueError('unknown motion policy')
        if not np.isfinite(ref_ms) or int(ref_ms)!=ref_ms:raise ValueError('integer native-ms reference required')
        self.prev=_ro(_finite(prev51,(51,),'prev51'));self.beta=_ro(_finite(beta,(10,),'beta'))
        self.K=_ro(_finite(K_event,(3,3),'event-grid K'));self.ref_ms=int(ref_ms);self.mode=mode;self.config=config
        if self.K[0,0]<=0 or self.K[1,1]<=0 or not np.allclose(self.K[2],[0,0,1],atol=1e-12,rtol=0):
            raise ValueError('positive-focal pinhole event-grid K required')
        self._lut=None;self._epoch=None;self._kernel_refresh_seconds=0.;self._kernel_refreshes=0
        self._endpoints=None;self._static_lut=None
        self.metadata=dict(mode=mode,config=asdict(config),ref_ms=self.ref_ms,camera_space='event240x180; no internal scaling',
            shape_condition='beta0' if not np.any(self.beta) else 'externally supplied beta',
            kernel_semantics='geometric selection response; not likelihood/probability/emission/visibility',
            temporal_phase='native interval midpoint s=t+.5; clip((s-ref_ms)/50,0,1), hold endpoint after50ms',
            geometry_approximation='float64 exact FK endpoints0/50ms then linear image-space bones; float32 distance/exp nearest4px-grid response',
            quota='static initial3-segment distance kernel, no temporal disagreement',
            generic_temporal='no MANO access; local centered time-space covariance norm, tilehash buckets without anatomical meaning',
            sign_semantics='witness: signedv_owner; quota/generic: deterministic hash capacity buckets only, not motion direction',
            tie_rule='deterministic tile+epoch hash, independent of counts arrival/query order',
            source_sha256=_sha(__file__))
        if mode!='generic_temporal':
            if mode=='witness':
                endpoint,cal=motion_endpoints(self.prev,self.beta,config.target_mm,config.max_angle_rad)
                joints=fk_joints(np.r_[self.prev[None],endpoint.reshape(10,51)],self.beta)
                self.metadata.update(calibration=cal,public_axes=[{k:r[k] for k in ('finger','mcp_mano','axis','sign')} for r in public_flexion_axes()])
            else:joints=fk_joints(self.prev[None],self.beta)
            if joints[...,2].min()<=.03:raise ValueError('joint/tip geometry intersects near plane')
            h=joints@self.K.T;xy=h[...,:2]/h[...,2:]
            initial=xy[0,FINGER_JOINTS]
            self.metadata.update(mano_asset_sha256=_asset_sha())
            if mode=='witness':
                moved=xy[1:].reshape(5,2,21,2)
                moved=np.asarray([[moved[j,k,FINGER_JOINTS[j]] for k in range(2)] for j in range(5)])
                self._endpoints=_ro(np.stack((np.repeat(initial[:,None],2,axis=1),moved)))
            if mode=='quota':
                self._static_lut=_ro(self._kernel(initial).astype(np.float32))
        self._refresh_seconds=time.perf_counter()-started

    def _kernel(self,bones):
        c=self.config
        yy,xx=np.meshgrid((np.arange(c.height//4)+.5)*4-.5,(np.arange(c.width//4)+.5)*4-.5,indexing='ij')
        points=np.stack((xx,yy),-1)[...,None,:]
        return _segment_kernel(points,bones,c.sigma_px)

    def _epoch_kernel(self,epoch):
        if self._epoch!=epoch:
            started=time.perf_counter();s=epoch+np.arange(10)+.5
            alpha=np.clip((s-self.ref_ms)/50.,0.,1.)
            # lead5finger×2sign×10time; output5×10×45×60.
            bones=self._endpoints[0,:,:,None]+alpha[None,None,:,None,None]*(self._endpoints[1,:,:,None]-self._endpoints[0,:,:,None])
            if np.ptp(alpha)==0:
                delta=np.zeros((5,10,45,60),np.float32)
            else:
                response=self._kernel(bones);delta=response[:,0]-response[:,1]
                delta-=delta.mean(axis=1,keepdims=True)
            self._lut=_ro(delta.astype(np.float32));self._epoch=int(epoch)
            self._kernel_refresh_seconds+=time.perf_counter()-started;self._kernel_refreshes+=1
        return self._lut

    @staticmethod
    def _hash(tile,epoch):
        return int.from_bytes(hashlib.blake2b(f'{int(tile[0])},{int(tile[1])},{int(epoch)}'.encode(),digest_size=8).digest(),'little')

    def episode_score_details(self,counts,tile_xy,epoch_start_ms):
        b=np.asarray(counts)
        if b.shape!=(10,8,8,2) or not np.issubdtype(b.dtype,np.integer) or np.any(b<0):raise ValueError('nonnegative integer counts[10,8,8,2] required')
        if np.any(b>np.iinfo(np.uint32).max):raise OverflowError('episode count exceeds uint32')
        tile=np.asarray(tile_xy)
        if tile.shape!=(2,) or not np.isfinite(tile).all() or np.any(tile!=np.floor(tile)) or np.any(tile%8):raise ValueError('tile_xy: integer8px-aligned left/top required')
        x,y=map(int,tile)
        if x<0 or y<0 or x>=240 or y>=180:raise ValueError('tile origin outside sensor')
        if not np.isfinite(epoch_start_ms) or int(epoch_start_ms)!=epoch_start_ms:raise ValueError('integer epoch start required')
        epoch=int(epoch_start_ms)
        if epoch<self.ref_ms:raise ValueError('cannot select old epoch using a future committed anchor')
        n=b.astype(np.float64).sum(-1)
        valid=(np.arange(8)[None,:]+x<240)&(np.arange(8)[:,None]+y<180)
        n=n*valid;total=float(n.sum());key=self._hash(tile,epoch)
        if self.mode=='generic_temporal':
            if total:
                t=(np.arange(10)-4.5)/5.;yy,xx=np.mgrid[:8,:8];xx=(xx-3.5)/4;yy=(yy-3.5)/4
                mt=np.sum(n*t[:,None,None])/total
                cov=np.array([np.sum(n*(t[:,None,None]-mt)*u) for u in (xx,yy)])
                priority=float(np.linalg.norm(cov)/np.sqrt(1+total))
            else:priority=0.
            sign=(1 if key&8 else -1) if priority else 0
            return dict(owner=int(key%5),priority=priority,sign=sign,signed_score=sign*priority)
        rows=(y+np.arange(8))//4;cols=(x+np.arange(8))//4
        if self.mode=='quota':
            lut=self._static_lut[:,np.minimum(rows,44)[:,None],cols[None,:]]
            v=np.sum(lut*n.sum(0)[None],axis=(1,2))/np.sqrt(1+total)
        else:
            lut=self._epoch_kernel(epoch)[:,:,np.minimum(rows,44)[:,None],cols[None,:]]
            # Equivalent with mean-zero D, and exactly zero for static counts.
            centered=n-n.mean(axis=0,keepdims=True)
            v=np.sum(lut*centered[None],axis=(1,2,3))/np.sqrt(1+total)
        best=float(np.abs(v).max());ties=np.flatnonzero(np.abs(v)==best)
        owner=int(ties[key%len(ties)])
        sign=int(np.sign(v[owner])) if self.mode=='witness' else ((1 if key&8 else -1) if best else 0)
        return dict(owner=owner,priority=best,sign=sign,signed_score=float(v[owner]) if self.mode=='witness' else sign*best)

    def episode_scores(self,counts,tile_xy,epoch_start_ms):
        d=self.episode_score_details(counts,tile_xy,epoch_start_ms)
        return d['owner'],d['priority']

    @property
    def state_nbytes(self):
        return sum(a.nbytes for a in (self.prev,self.beta,self.K,self._endpoints,self._static_lut,self._lut) if a is not None)
    @property
    def nbytes(self):return self.state_nbytes

    def receipt(self):
        # Generic policy deliberately does not initialize MANO/shape caches.
        shared=sum(a.nbytes for a in _joint_constants().values()) if self.mode!='generic_temporal' else 0
        return dict(metadata=self.metadata,state_nbytes=self.state_nbytes,owned_array_bytes=self.state_nbytes,
            cache_bytes=0 if self._lut is None else self._lut.nbytes,
            static_kernel_bytes=0 if self._static_lut is None else self._static_lut.nbytes,
            shared_joint_asset_array_bytes=shared,shape_cache_max_entries=16,
            shape_cache_entry_array_bytes=0 if self.mode=='generic_temporal' else 504,
            refresh_seconds=self._refresh_seconds,kernel_refresh_seconds=self._kernel_refresh_seconds,
            kernel_refreshes=self._kernel_refreshes,cached_epoch=self._epoch,
            exclusions='Python object/metadata overhead, caller events and FK scratch separate; process RSS is not policy bytes')


@lru_cache(maxsize=1)
def _asset_sha():return _sha(ASSET)


@dataclass(frozen=True)
class SyntheticCondition:
    finger:int
    sign:int
    root_y_rad:float=0.
    root_dx_m:float=0.
    moving_background:bool=False
    seed:int=3407


def synthetic_conditions(include_background=False):
    return tuple(SyntheticCondition(j,s,y,dx,bg) for bg in ((False,True) if include_background else (False,))
        for y in (0.,.3) for j in range(5) for s in (1,-1) for dx in (0.,.01))


def generate_synthetic_case(condition,out_prefix,*,K_source=None):
    """Write native observations and targets to separate files; no scoring.

    Known synthetic boundary-state condition is public to every policy.  The
    condition/target files are generator/scoring provenance, never selector
    input.  Foreground is constant .70; optional moving background is a
    radiance step on a plane behind the hand (.15/.30, x30→210 in100ms).
    No random point events are added or described as a physical background.
    """
    from tools.s43_rel.mechanisms import full_render
    c=condition
    if not isinstance(c,SyntheticCondition) or c.finger not in range(5) or c.sign not in (-1,1):raise ValueError('invalid fixed synthetic condition')
    if c.root_y_rad not in (0.,.3) or c.root_dx_m not in (0.,.01):raise ValueError('condition outside declared synthetic domain')
    out=Path(out_prefix);out.parent.mkdir(parents=True,exist_ok=True)
    files=[Path(str(out)+suffix) for suffix in ('.events.npz','.targets.npz','.json')]
    if any(p.exists() for p in files):raise FileExistsError('preserve existing synthetic artifact')
    camera=np.asarray([[600.,0.,320.],[0.,600.,240.],[0.,0.,1.]]) if K_source is None else _finite(K_source,(3,3),'K_source')
    event=camera.copy();event[:2]*=.375
    public=PUBLIC_INITIAL.copy();public[4]=c.root_y_rad;beta=np.zeros(10)
    endpoints,cal=motion_endpoints(public,beta);row=public_flexion_axes()[c.finger]
    angle=cal['angles_rad'][c.finger][0 if c.sign==1 else 1]
    states=[]
    for ms in range(101):
        alpha=min(ms/50.,1.)
        p=right_retract(public,row['mcp_mano'],np.eye(3)[row['axis']]*row['sign']*c.sign,alpha*angle)
        p[0]+=alpha*c.root_dx_m;states.append(p)
    states=np.asarray(states);joints=fk_joints(states,beta)
    masks=full_render(states,beta,camera)
    xx=np.arange(240)[None,None,:]
    if c.moving_background:
        edge=30+180*np.arange(101)/100.
        background=np.where(xx<edge[:,None,None],.15,.30)
    else:background=np.full((101,1,240),.15)
    intensity=np.where(masks,.70,background)
    threshold=.35;rng=np.random.default_rng(c.seed)
    reference=np.log(intensity[0])+rng.uniform(-.1*threshold,.1*threshold,(180,240))
    output=[]
    for ms in range(100):
        delta=np.log(intensity[ms+1])-reference
        count=np.floor(np.abs(delta)/threshold).astype(np.int64)
        polarity=delta>0;reference+=np.sign(delta)*count*threshold
        yy,x=np.nonzero(count)
        if len(x):output.append(np.column_stack((np.repeat(x,count[yy,x]),np.repeat(yy,count[yy,x]),
            np.full(int(count.sum()),ms,np.int64),np.repeat(polarity[yy,x],count[yy,x]))))
    events=np.concatenate(output).astype(np.int64) if output else np.empty((0,4),np.int64)
    offsets=np.searchsorted(events[:,2],np.arange(101))
    np.savez_compressed(files[0],events=events,offsets=offsets,public_initial51=public,beta=beta,K_event=event,K_source=camera,
        boundary_ms=np.asarray(0,np.int64),stop_ms=np.asarray(100,np.int64))
    np.savez_compressed(files[1],target51=states,joints_m=joints,time_ms=np.arange(101),finger=np.asarray(c.finger),sign=np.asarray(c.sign))
    manifest=dict(schema='s43_witness_synthetic_source_v1',condition=asdict(c),source_sha256=_sha(__file__),
        renderer_source_sha256=_sha(ROOT/'tools/s43_rel/mechanisms.py'),mano_asset_sha256=_asset_sha(),
        files=dict(observations=str(files[0]),targets=str(files[1])),digests=dict(observations=_sha(files[0]),targets=_sha(files[1])),
        native_ms=1,cutoff_contract='event t is interval[t,t+1), pose target at integer endpoint',events=len(events),
        public_condition='known synthetic generator boundary state only; not actual-deployment oracle init',
        threshold=threshold,phase='independent pixel uniform[-.035,.035], same seed across motions',
        radiance=dict(foreground=.70,background=(.15,.30) if c.moving_background else (.15,),
                      background_geometry='moving vertical radiance edge on plane behind foreground; x30→210px/100ms' if c.moving_background else 'constant plane behind hand'),
        approximation='binary polygon-union pixel coverage, public constant radiances, threshold crossing quantized to native1ms; not textured real sensor',
        calibration=cal,score_read=False,
        target_access='target51/FK/actions are separate files; input builders must not read them before observations/selection are constructed; final scoring after predictions saved')
    files[2].write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    return manifest


def generate_suite(output_dir,protocol_path,engineering_receipt):
    """Formal80-source generation only; source-freeze precedes any rendering."""
    out=Path(output_dir);protocol_path=Path(protocol_path);test=Path(engineering_receipt)
    protocol=json.loads(protocol_path.read_text())
    if protocol.get('schema')!='s43_witness_prototype_v1' or protocol['synthetic']['sequences']!=80:
        raise ValueError('requires frozen80-sequence stage2 protocol')
    if out.exists() and any(out.iterdir()):raise FileExistsError('formal generation never overwrites artifacts')
    out.mkdir(parents=True,exist_ok=True)
    dependencies=[Path(__file__),ROOT/'tools/s43_rel/geometry.py',ROOT/'tools/s43_rel/mechanisms.py',
        ROOT/'tools/s43_integration/fast_geometry.py',ROOT/'tools/s43_integration/raster.cpp',
        ROOT/'model/mano_layer.py',ASSET]
    frozen={str(p.relative_to(ROOT)):_sha(p) for p in dependencies}
    receipt=json.loads(test.read_text())
    if receipt.get('geometry_source_sha256')!=_sha(__file__) or receipt.get('unit_tests_passed') is not True:
        raise ValueError('engineering test receipt must bind actual geometry/generator source')
    freeze=dict(schema='s43_witness_geometry_sourcefreeze_v1',protocol=str(protocol_path),protocol_sha256=_sha(protocol_path),
        dependencies=frozen,engineering_receipt=str(test),engineering_receipt_sha256=_sha(test),
        score_read=False,formal_source_count=80,phase='saved before formal rendering; no task reader run')
    (out/'sourcefreeze.json').write_text(json.dumps(freeze,indent=2)+'\n')
    cases=[];started=time.perf_counter()
    for i,condition in enumerate(synthetic_conditions(include_background=True)):
        prefix=out/f'case{i:03d}';d=generate_synthetic_case(condition,prefix)
        cases.append(dict(id=f'case{i:03d}',observations=d['files']['observations'],targets=d['files']['targets'],
            manifest=str(prefix)+'.json',observation_sha256=d['digests']['observations'],target_sha256=d['digests']['targets'],
            events=d['events']))
        print(json.dumps(dict(generated=i+1,total=80,events=d['events'])),flush=True)
    if any(_sha(ROOT/k)!=v for k,v in frozen.items()):raise RuntimeError('source changed during formal source generation')
    summary=dict(schema='s43_witness_synthetic_suite_v1',protocol_sha256=_sha(protocol_path),
        geometry_source_sha256=_sha(__file__),sourcefreeze_sha256=_sha(out/'sourcefreeze.json'),
        cases=cases,source_count=80,clean40_moving_background40=True,score_read=False,
        seconds=time.perf_counter()-started,sources_unchanged=True,
        input_rule='selectors/readers get observation files only; condition/action/targets are separate generator provenance or post-input supervision/scoring')
    (out/'manifest.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description='Generate frozen native-ms physical synthetic sources; never score.')
    parser.add_argument('--generate-dir',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,default=ROOT/'outputs/s43_revisit_20261028/stage2_protocol.json')
    parser.add_argument('--engineering-receipt',type=Path,required=True)
    args=parser.parse_args();generate_suite(args.generate_dir,args.protocol,args.engineering_receipt)
