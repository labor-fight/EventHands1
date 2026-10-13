"""XD physical SO(3) interventions and finite, paired measurement contracts.

Geometry is measured in float64; network states retain their original float32 dtype.
Root/finger directional scores are conditional left/right transport diagnostics.
"""
from __future__ import annotations
import hashlib
import math
import numpy as np
from scipy.spatial.transform import Rotation

STUDY = 'XD-RECOVERY-COST-V2-20261013'
SCALES = {'small': (.005, math.radians(5)), 'large': (.050, math.radians(20))}
BLOCKS = ('translation', 'root', 'finger', 'mixed')


def stable_int(value):
    return int(hashlib.sha256(str(value).encode()).hexdigest(), 16)


def vectors(packet_id, label, count=1):
    rng = np.random.default_rng(stable_int(f'{STUDY}|{packet_id}|{label}') % (2**64))
    value = rng.standard_normal((count, 3))
    return value / np.linalg.norm(value, axis=-1, keepdims=True)


def _spec(block, scale=None, sign=None, **kw):
    value = dict(block=block, **kw)
    if scale is not None: value.update(scale=scale, sign=sign)
    value['id'] = '|'.join(f'{k}={v}' for k,v in value.items())
    return value


def probe_specs(packet_id, seen_ordinal=None):
    specs = [_spec('clean'), _spec('natural')]
    for block in ('translation', 'root'):
        for scale in SCALES:
            for axis in range(3):
                for sign in (-1, 1): specs.append(_spec(block, scale, sign, axis=axis))
    for scale in SCALES:
        for joint in range(15):
            axes = range(3) if seen_ordinal is None else [(seen_ordinal+joint)%3]
            for axis in axes:
                for sign in (-1, 1): specs.append(_spec('finger', scale, sign, joint=joint, axis=axis))
    for scale in SCALES:
        for mix_id in range(2):
            for sign in (-1,1): specs.append(_spec('mixed', scale, sign, mix_id=mix_id, packet_id=packet_id))
    return specs


def recovery_specs(packet_id, small_only=False):
    specs = [_spec('clean')] if small_only else [_spec('clean'), _spec('natural')]
    for block in BLOCKS:
        for scale in (['small'] if small_only else SCALES):
            for sign in (-1,1): specs.append(_spec(block, scale, sign, vector_id=packet_id))
    return specs


def control_specs(specs, packet_id):
    result = specs[:2]
    for block in BLOCKS:
        candidates = [s for s in specs if s['block']==block and s.get('scale')=='large' and s['sign']==1]
        positive = candidates[stable_int(f'{STUDY}|{packet_id}|{block}') % len(candidates)]
        negative = next(s for s in specs if s['block']==block and s.get('scale')=='large' and s['sign']==-1
                        and all(s.get(k)==v for k,v in positive.items() if k not in ('id','sign')))
        result.extend([negative,positive])
    return result


def rotations(state, mean):
    state=np.asarray(state,np.float64)
    return Rotation.from_rotvec(state[...,3:6].reshape(-1,3)).as_matrix().reshape(state.shape[:-1]+(3,3)), \
        Rotation.from_rotvec((state[...,6:]+np.asarray(mean)).reshape(-1,3)).as_matrix().reshape(state.shape[:-1]+(15,3,3))


def perturb_state(state, spec, mean, wrist_fn=None, natural=None):
    original=np.asarray(state)
    result=np.asarray(state,np.float64).copy()
    block=spec['block']
    if block=='clean': return result.astype(original.dtype)
    if block=='natural':
        if natural is None: raise ValueError('natural intervention requires common frozen state')
        return np.asarray(natural,dtype=original.dtype).copy()
    distance, angle=SCALES[spec['scale']]; sign=spec['sign']
    if 'axis' in spec:
        axis=np.eye(3)[spec['axis']]
    else:
        axis=vectors(spec.get('vector_id',spec.get('packet_id')),f"{block}|{spec.get('mix_id',0)}|root")[0]
    wrist_before = np.asarray(wrist_fn(result)) if wrist_fn is not None else None
    if block in ('root','mixed'):
        changed=Rotation.from_rotvec(axis*angle*sign)*Rotation.from_rotvec(result[3:6])
        result[3:6]=changed.as_rotvec()
        if wrist_fn is not None: result[:3]+=wrist_before-np.asarray(wrist_fn(result))
    if block in ('finger','mixed'):
        full=(result[6:]+np.asarray(mean,np.float64)).reshape(15,3)
        selected=[spec['joint']] if 'joint' in spec else range(15)
        axes=np.tile(axis,(15,1)) if 'joint' in spec else vectors(spec.get('vector_id',spec.get('packet_id')),f"{block}|{spec.get('mix_id',0)}|finger",15)
        for j in selected: full[j]=(Rotation.from_rotvec(axes[j]*angle*sign)*Rotation.from_rotvec(full[j])).as_rotvec()
        result[6:]=full.reshape(45)-np.asarray(mean)
        # Local rotations should preserve the actual wrist; compensate harmless FK roundoff.
        if wrist_fn is not None: result[:3]+=wrist_before-np.asarray(wrist_fn(result))
    if block in ('translation','mixed'):
        t_axis=axis if block=='translation' else vectors(spec.get('vector_id',spec.get('packet_id')),f"mixed|{spec.get('mix_id',0)}|translation")[0]
        result[:3]+=sign*distance*t_axis
    return result.astype(original.dtype)


def _correction(d0,d1,transport=None):
    d0=np.asarray(d0,np.float64); d1=np.asarray(d1,np.float64)
    energy=float(np.sum(d0*d0))
    if not np.isfinite(d0).all() or not np.isfinite(d1).all():
        return {'rho':float('nan'),'gain':float('nan'),'cosine':float('nan'),'input_norm':float('nan'),'output_norm':float('nan')}
    if energy<=1e-18: return {'rho':None,'gain':None,'cosine':None,'input_norm':math.sqrt(energy),'output_norm':float(np.linalg.norm(d1))}
    a=d0 if transport is None else np.asarray(transport)
    correction=d1-a; target=-a
    numerator=float(np.sum(correction*target)); corr_energy=float(np.sum(correction*correction))
    return {'rho':float(np.linalg.norm(d1)/math.sqrt(energy)), 'gain':numerator/float(np.sum(a*a)),
            'cosine':None if corr_energy<=1e-18 else numerator/math.sqrt(corr_energy*float(np.sum(a*a))),
            'input_norm':math.sqrt(energy),'output_norm':float(np.linalg.norm(d1))}


def vector_pair_metrics(p0,pd,y0,yd):
    return _correction(np.asarray(pd)-p0,np.asarray(yd)-y0)


def rotation_pair_metrics(p0,pd,y0,yd):
    p0,pd,y0,yd=map(lambda x:np.asarray(x,np.float64),(p0,pd,y0,yd))
    d0=pd-p0; d1=yd-y0
    left=(y0@np.swapaxes(p0,-1,-2))@d0
    right=d0@(np.swapaxes(p0,-1,-2)@y0)
    result={'rho':_correction(d0,d1)['rho'], 'raw':_correction(d0,d1),
            'left':_correction(d0,d1,left),'right':_correction(d0,d1,right)}
    if d0.ndim==3:
        result['by_joint']=[rotation_pair_metrics(a,b,c,d) for a,b,c,d in zip(p0,pd,y0,yd)]
    return result


def geodesic_deg(r0,r1):
    r0,r1=np.asarray(r0),np.asarray(r1)
    if not np.isfinite(r0).all() or not np.isfinite(r1).all(): return np.full(r0.shape[:-2],np.nan)
    relative=np.swapaxes(r0,-1,-2)@r1
    return Rotation.from_matrix(relative.reshape(-1,3,3)).magnitude().reshape(relative.shape[:-2])*180/np.pi


def time_auc(errors,times_ms):
    errors=np.asarray(errors,np.float64); times=np.asarray(times_ms,np.float64)
    if len(errors)!=len(times) or len(errors)<2 or not np.all(np.diff(times)>0): raise ValueError('AUC requires paired strictly increasing timestamps')
    if not np.isfinite(errors).all(): return None
    return float(np.sum(np.diff(times)*(errors[1:]+errors[:-1])*.5)/(times[-1]-times[0]))


def sustained_recovery(abs_mm,ra_mm,times_ms,hold=3):
    a=np.asarray(abs_mm); r=np.asarray(ra_mm); t=np.asarray(times_ms)
    if not(len(a)==len(r)==len(t)): raise ValueError('recovery arrays must align')
    good=np.isfinite(a)&np.isfinite(r)&(a<=20)&(r<=10)
    initially=bool(good[0]); hit=None
    if not initially:
        for i in range(1,len(good)-hold+1):
            if good[i:i+hold].all(): hit=i;break
    return {'initially_satisfied':initially,'hit_step':hit,'time_ms':None if hit is None else int(t[hit]-t[0]),
            'right_censored':not initially and hit is None,'horizon_ms':int(t[-1]-t[0]),
            'trajectory_finite':bool(np.isfinite(a).all() and np.isfinite(r).all())}


def sustained_half(distances,times_ms,hold=3):
    d=np.asarray(distances,np.float64); t=np.asarray(times_ms)
    if d[0]<=1e-9 or not np.isfinite(d[0]): return {'defined':False,'hit_step':None,'right_censored':None}
    good=np.isfinite(d)&(d<=.5*d[0]); hit=None
    for i in range(1,len(d)-hold+1):
        if good[i:i+hold].all():hit=i;break
    return {'defined':True,'hit_step':hit,'time_ms':None if hit is None else int(t[hit]-t[0]),
            'right_censored':hit is None,'horizon_ms':int(t[-1]-t[0])}


def transform_events(ev5,mode,width=240):
    old=np.asarray(ev5); result=old.copy()
    if mode=='mirror':result[:,1]=width-1-old[:,1]
    elif mode=='swap':
        group=np.clip(np.floor(old[:,3]/.05*4).astype(int),0,3)*2+(old[:,4]>0).astype(int)
        for label in np.unique(group):
            idx=np.flatnonzero(group==label)
            if len(idx)>1:result[idx,1:3]=old[np.roll(idx,1),1:3]
    elif mode=='empty':result=np.zeros((0,5),dtype=old.dtype)
    else:raise ValueError(mode)
    changed=float(np.mean(np.any(result[:,1:3]!=old[:,1:3],axis=-1))) if len(old) and mode!='empty' else None
    return result, {'mode':mode,'count_original':len(old),'count_transformed':len(result),'changed_fraction':changed,
                    'weak_swap':mode=='swap' and (changed is None or changed<.01)}


def motion_response(pred,truth,times_ms,root_relative=True,horizon_ms=200,smooth_gt=True,active_floor_mm=1.):
    """Causal model trajectory; future GT smoothing exists solely inside offline score.

    Use centered 5-point GT means and exact 200ms endpoints. Edge windows without
    all 5 labels are undefined. Report signed gain; low energy is undefined.
    """
    pred=np.asarray(pred,np.float64); gt=np.asarray(truth,np.float64);times=np.asarray(times_ms,np.int64)
    if root_relative:pred=pred-pred[:,:1];gt=gt-gt[:,:1]
    smooth=gt.copy()
    if smooth_gt:
        smooth[:]=np.nan
        for i in range(2,len(gt)-2):smooth[i]=gt[i-2:i+3].mean(0)
    pv=[];gv=[];start=[]
    for i,time in enumerate(times):
        match=np.flatnonzero(times==time+horizon_ms)
        if not len(match):continue
        j=match[0]
        p=pred[j]-pred[i];g=smooth[j]-smooth[i]
        if np.isfinite(p).all() and np.isfinite(g).all() and math.sqrt(float(np.mean(np.sum(g*g,axis=-1))))*1000>active_floor_mm:
            pv.append(p);gv.append(g);start.append(i)
    base={'horizon_ms':horizon_ms,'gt_smoothing_points':5 if smooth_gt else 1,'active_floor_mm':active_floor_mm,'active_windows':len(pv),'eligible_time_windows':int(sum(t+horizon_ms in set(times.tolist()) for t in times)),
          'gain':None,'gain_bias':None,'cosine':None,'lag_ms':None,'lag_correlations':{}}
    if not pv:return base
    p=np.asarray(pv);g=np.asarray(gv);energy=np.sum(g*g);pe=np.sum(p*p);dot=np.sum(p*g)
    base.update(gain=float(dot/energy),gain_bias=abs(float(dot/energy)-1),cosine=None if pe<=1e-18 else float(dot/math.sqrt(pe*energy)))
    # Fixed positive lag means predicted movement follows GT movement.
    scores={}
    for lag in (0,50,100,150,200):
        ps=[];gs=[]
        for i in start:
            j=np.flatnonzero(times==times[i]+horizon_ms)
            a=np.flatnonzero(times==times[i]+lag);b=np.flatnonzero(times==times[i]+horizon_ms+lag)
            if len(j) and len(a) and len(b):
                pp=pred[b[0]]-pred[a[0]];gg=smooth[j[0]]-smooth[i]
                if np.isfinite(pp).all() and np.isfinite(gg).all():ps.append(pp);gs.append(gg)
        if ps:
            pp=np.asarray(ps);gg=np.asarray(gs);den=np.sqrt(np.sum(pp*pp)*np.sum(gg*gg))
            scores[str(lag)]=None if den<=1e-18 else float(np.sum(pp*gg)/den)
    defined={int(k):v for k,v in scores.items() if v is not None}
    best=max(defined.values()) if defined else None
    maxima=[k for k,v in defined.items() if best-v<=1e-8] if defined else []
    base.update(lag_ms=maxima[0] if len(maxima)==1 else None,lag_correlations=scores,lag_identifiable=len(maxima)==1)
    return base


def sanitize(value,location='$',invalid=None):
    if invalid is None:invalid=[]
    if isinstance(value,np.ndarray):value=value.tolist()
    elif isinstance(value,np.generic):value=value.item()
    if isinstance(value,float) and not math.isfinite(value):invalid.append(location);return None
    if isinstance(value,dict):return {str(k):sanitize(v,f'{location}.{k}',invalid) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [sanitize(v,f'{location}[{i}]',invalid) for i,v in enumerate(value)]
    return value


def rotation_motion_response(pred,truth,times_ms,horizon_ms=200,smooth_gt=True,active_floor_deg=1.):
    """Signed angular displacement, with per-joint SO(3) GT means offline only.

    Root uses the camera frame and local rotations each parent's coordinate frame.
    A 5-point chordal SO(3) mean replaces invalid linear AA smoothing.
    """
    p=np.asarray(pred,np.float64);g=np.asarray(truth,np.float64);times=np.asarray(times_ms,np.int64)
    if p.ndim==3:p=p[:,None];g=g[:,None]
    smooth=g.copy()
    if smooth_gt:
        smooth[:]=np.nan
        for i in range(2,len(g)-2):
            for joint in range(g.shape[1]):
                values=g[i-2:i+3,joint]
                if np.isfinite(values).all():smooth[i,joint]=Rotation.from_matrix(values.copy()).mean().as_matrix()
    def delta(values,i,j):
        relative=values[j]@np.swapaxes(values[i],-1,-2)
        return Rotation.from_matrix(relative).as_rotvec() if np.isfinite(relative).all() else np.full((g.shape[1],3),np.nan)
    indices=[];pv=[];gv=[]
    for i,t in enumerate(times):
        match=np.flatnonzero(times==t+horizon_ms)
        if not len(match):continue
        j=int(match[0]);pp=delta(p,i,j);gg=delta(smooth,i,j)
        if np.isfinite(pp).all() and np.isfinite(gg).all() and math.sqrt(float(np.mean(np.sum(gg*gg,axis=-1))))>math.radians(active_floor_deg):
            indices.append(i);pv.append(pp);gv.append(gg)
    result={'horizon_ms':horizon_ms,'gt_smoothing_points':5 if smooth_gt else 1,'gt_smoothing':'SO3 chordal mean','active_floor_deg':active_floor_deg,
            'active_windows':len(pv),'gain':None,'gain_bias':None,'cosine':None,'lag_ms':None,'lag_correlations':{},'lag_identifiable':False}
    if not pv:return result
    pp=np.asarray(pv);gg=np.asarray(gv);dot=np.sum(pp*gg);energy=np.sum(gg*gg);pe=np.sum(pp*pp)
    result.update(gain=float(dot/energy),gain_bias=abs(float(dot/energy)-1),cosine=None if pe<=1e-18 else float(dot/math.sqrt(energy*pe)))
    scores={}
    for lag in (0,50,100,150,200):
        ps=[];gs=[]
        for i in indices:
            a=np.flatnonzero(times==times[i]+lag);b=np.flatnonzero(times==times[i]+lag+horizon_ms);j=np.flatnonzero(times==times[i]+horizon_ms)
            if len(a) and len(b) and len(j):
                pp=delta(p,int(a[0]),int(b[0]));gg=delta(smooth,i,int(j[0]))
                if np.isfinite(pp).all() and np.isfinite(gg).all():ps.append(pp);gs.append(gg)
        if ps:
            pp=np.asarray(ps);gg=np.asarray(gs);den=np.sqrt(np.sum(pp*pp)*np.sum(gg*gg))
            scores[str(lag)]=None if den<=1e-18 else float(np.sum(pp*gg)/den)
    defined={int(k):v for k,v in scores.items() if v is not None};best=max(defined.values()) if defined else None
    maxima=[k for k,v in defined.items() if best-v<=1e-8] if defined else []
    result.update(lag_ms=maxima[0] if len(maxima)==1 else None,lag_correlations=scores,lag_identifiable=len(maxima)==1)
    return result
