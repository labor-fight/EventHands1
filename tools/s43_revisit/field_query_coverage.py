"""Post-prediction GT projection coverage only; no fitting or source attribution."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_revisit_20261028'
ARMS=('event_grid','hand_box','static','generic_trajectory','anatomical')
ALL_ARMS=('global_only',*ARMS,'anatomical_temporal_collapse','witness_anatomical','quota_anatomical')
CONDITIONS=('none','wrong_x','wrong_yaw','corrected_from_x','corrected_from_yaw')
ANATOMICAL=('static','generic_trajectory','anatomical')
TOLERANCE_PX=1e-5


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def dump(path,x):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False,
        default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist())+'\n')


def prediction_gate(base,stage_freeze):
    """All fixed predictions exist/hash-match before any diagnostic GT decode."""
    identities={};stage_identity=sha(stage_freeze)
    for domain in ('synthetic','existing'):
        feature=base/'features'/domain/'complete.json';complete_identity=sha(feature)
        for arm in ALL_ARMS:
            folder=base/'readers'/domain/arm;path=folder/'predictions_complete.json'
            data=json.loads(path.read_text());identities[str(path.resolve())]=sha(path)
            expected=list(range(4)) if domain=='synthetic' else [-1]
            if data['folds']!=expected or data['source_freeze_sha256']!=stage_identity or data['feature_complete_sha256']!=complete_identity:
                raise RuntimeError('prediction/feature/fold identity mismatch')
            for fold in expected:
                for kind in ('cnn','rbf'):
                    path=folder/f'fold{fold}'/f'{kind}_predictions.npz';name=str(path.resolve())
                    if name not in data['artifacts'] or sha(path)!=data['artifacts'][name]:
                        raise RuntimeError('incomplete/changed fixed prediction')
                    identities[name]=data['artifacts'][name]
    return identities


def freeze(base,out,plan,stage_freeze):
    if (out/'freeze.json').exists():raise FileExistsError(out/'freeze.json')
    # rows contain observed query contexts/priors; labels are HASHED ONLY here.
    files={Path(__file__),ROOT/'tests/test_s43_field_query_coverage.py',plan,stage_freeze,
           OUT/'stage3_field_query_protocol.json'}
    for domain in ('synthetic','existing'):
        folder=base/'features'/domain
        for name in ('rows.json','labels.json','complete.json'):files.add(folder/name)
        rows=json.loads((folder/'rows.json').read_text());f=json.loads((folder/'complete.json').read_text())
        if sha(folder/'rows.json')!=f['rows_sha256'] or sha(folder/'labels.json')!=f['labels_sha256']:
            raise RuntimeError('prepared metadata changed')
        seen=set()
        for r in rows:
            if r['arm'] not in ARMS:continue
            key=(r['case_id'],r['arm'],r['condition'])
            if key in seen:raise RuntimeError('duplicate case/arm/condition')
            seen.add(key);p=Path(r['coordinate_file'])
            if sha(p)!=r['coordinate_sha256']:raise RuntimeError('sealed coordinate changed')
            files.add(p)
        cases={r['case_id'] for r in rows}
        if len(seen)!=len(cases)*len(ARMS)*len(CONDITIONS):raise RuntimeError('missing diagnostic view')
    predictions=prediction_gate(base,stage_freeze)
    dump(out/'freeze.json',dict(schema='s43_field_query_coverage_freeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        base=str(base.resolve()),stage_freeze=str(stage_freeze.resolve()),plan=str(plan.resolve()),
        identities={str(p.resolve()):sha(p) for p in sorted(files)},prediction_identities=predictions,
        labels_decoded=False,task_scores_computed_or_read=False,all_predictions_complete=True,
        arms=list(ARMS),conditions=list(CONDITIONS),tolerance_px=TOLERANCE_PX,
        box_controls='only any-patch union; index slots have no finger/bone identity',
        label_use='post-prediction GT projection only; no visibility/sourceownership/information-loss inference'))


def gt_points(joints):
    joints=np.asarray(joints,np.float64)
    if joints.shape!=(21,3) or not np.isfinite(joints).all():raise ValueError('finite21x3 GT joints required')
    finger=joints[1:].reshape(5,4,3);a=np.asarray([.25,.5,.75])[:,None]
    anchors=finger[:,:-1,None]*(1-a)+finger[:,1:,None]*a
    xyz=np.concatenate((finger.reshape(20,3),anchors.reshape(45,3)))
    return xyz,np.r_[np.repeat(np.arange(5),4),np.repeat(np.arange(5),9)],np.r_[np.full(20,-1),np.tile(np.repeat(np.arange(3),3),5)]


def project(xyz,K):
    K=np.asarray(K,np.float64).copy()
    if K.shape!=(3,3) or not np.isfinite(K).all():raise ValueError('finiteK required')
    K[:2]*=.375;h=xyz@K.T;good=(xyz[:,2]>.03)&(np.abs(h[:,2])>1e-12)
    uv=np.divide(h[:,:2],h[:,2,None],out=np.zeros((len(xyz),2)),where=good[:,None])
    good&=np.isfinite(uv).all(1)
    inview=good&(uv[:,0]>=0)&(uv[:,0]<=239)&(uv[:,1]>=0)&(uv[:,1]<=179)
    return uv,good,inview


def geometry(coords,uv,projectable,inview,finger,bone,arm):
    grid=np.asarray(coords['grid'],np.float64)
    if grid.shape!=(8,90,49,2) or not np.isfinite(grid).all():raise ValueError('sealed grid shape/value')
    node=np.asarray(coords['valid'],bool)&np.asarray(coords['full_interp_support'],bool)
    if node.shape!=(8,90,49):raise ValueError('sealed validity shape')
    px=((grid+1)*np.asarray([240,180])/2-.5).reshape(8,90,7,7,2)
    center=px[:,:,3,3];ex=px[:,:,3,4]-center;ey=px[:,:,4,3]-center
    det=ex[...,0]*ey[...,1]-ex[...,1]*ey[...,0];ok=np.abs(det)>1e-12
    det=np.where(ok,det,1);d=uv[None,None]-center[:,:,None]
    u=(d[...,0]*ey[...,1,None]-d[...,1]*ey[...,0,None])/det[...,None]
    v=(d[...,1]*ex[...,0,None]-d[...,0]*ex[...,1,None])/det[...,None]
    # Convert the fixed pixel tolerance to local step coordinates.
    tol=TOLERANCE_PX/np.maximum(np.minimum(np.linalg.norm(ex,axis=-1),np.linalg.norm(ey,axis=-1)),1e-12)
    inside=(np.abs(u)<=3+tol[...,None])&(np.abs(v)<=3+tol[...,None])&ok[...,None]&projectable[None,None]
    ix=np.clip(np.floor(u+3),0,5).astype(np.int64);iy=np.clip(np.floor(v+3),0,5).astype(np.int64)
    quad=np.ones(inside.shape,bool)
    for dx,dy in ((0,0),(1,0),(0,1),(1,1)):
        quad&=np.take_along_axis(node,(iy+dy)*7+ix+dx,axis=2)
    supported=inside&quad&inview[None,None]
    patchfinger=np.repeat(np.arange(5),18);patchbone=np.tile(np.repeat(np.arange(3),6),5)
    samefinger=patchfinger[:,None]==finger[None]
    matched=samefinger&((bone[None]<0)|(patchbone[:,None]==bone[None]))
    # Uniform boxes do not have anatomical slot identities.
    if arm not in ANATOMICAL:matched=np.ones_like(matched);samefinger=np.ones_like(matched)
    result={}
    for name,mask in (('matched',matched),('samefinger',samefinger),('any',np.ones_like(matched))):
        result[name+'_geometry']=np.any(inside&mask[None],axis=1)
        result[name+'_supported']=np.any(supported&mask[None],axis=1)
    clipped=center[:,:,None]+np.clip(u,-3,3)[...,None]*ex[:,:,None]+np.clip(v,-3,3)[...,None]*ey[:,:,None]
    distance=np.linalg.norm(uv[None,None]-clipped,axis=-1)
    distance=np.where(ok[...,None]&matched[None]&projectable[None,None],distance,np.inf)
    distance=distance.min(1)
    result['last_distance_px']=distance[-1].astype(np.float32)
    result['union_distance_px']=distance.min(0).astype(np.float32)
    return result


def summarize(records,arrays,selected):
    ix=np.flatnonzero(selected);N=len(ix)
    if not N:return dict(n=0,status='NO_EVIDENCE')
    output=dict(n=N,unique_cases=len({records[i]['case_id'] for i in ix}),metrics={})
    for point_name,sl in (('joints',slice(0,20)),('bone_anchors',slice(20,65))):
        valid=arrays['in_view'][ix,sl];denom=valid.sum()
        x=dict(points=N*(20 if point_name=='joints' else 45),projectable=int(arrays['projectable'][ix,sl].sum()),
               in_view=int(denom),in_view_fraction=float(valid.mean()))
        for match in ('matched','samefinger','any'):
            for semantic in ('geometry','supported'):
                a=arrays[match+'_'+semantic][ix,:,sl]
                for scope,value in (('last',a[:,-1]),('union8_spatial',a.any(1))):
                    key=f'{match}_{semantic}_{scope}'
                    x[key]=float(value.mean())
                    x[key+'_conditional_inview']=float((value&valid).sum()/denom) if denom else None
        for scope in ('last','union'):
            d=arrays[scope+'_distance_px'][ix,sl];finite=d[np.isfinite(d)]
            x[scope+'_distance_finite_n']=len(finite)
            x[scope+'_distance_P50_px']=float(np.percentile(finite,50)) if len(finite) else None
            x[scope+'_distance_P95_px']=float(np.percentile(finite,95)) if len(finite) else None
        output['metrics'][point_name]=x
    output['per_finger']={}
    for j in range(5):
        p=np.r_[np.arange(j*4,j*4+4),np.arange(20+j*9,20+j*9+9)]
        a=arrays['matched_supported'][ix][:,:,p]
        output['per_finger'][str(j)]=dict(joint_last=float(a[:,-1,:4].mean()),anchor_last=float(a[:,-1,4:].mean()),
           joint_union=float(a[:,:,:4].any(1).mean()),anchor_union=float(a[:,:,4:].any(1).mean()))
    return output


def run(out):
    freeze_path=out/'freeze.json';f=json.loads(freeze_path.read_text())
    if (out/'results.json').exists():raise FileExistsError(out/'results.json')
    identities={**f['identities'],**f['prediction_identities']}
    for path,identity in identities.items():
        if sha(path)!=identity:raise RuntimeError('frozen diagnostic input changed: '+path)
    base=Path(f['base']);prediction_gate(base,Path(f['stage_freeze']))
    # First semantic GT read occurs only after every identity and prediction gate.
    records=[];values=[];cache={};started=time.perf_counter();gt_opened=datetime.now(timezone.utc).isoformat()
    for domain in ('synthetic','existing'):
        folder=base/'features'/domain;rows=json.loads((folder/'rows.json').read_text());labels=json.loads((folder/'labels.json').read_text())
        wanted=[r for r in rows if r['arm'] in ARMS]
        for n,r in enumerate(wanted):
            cid=r['case_id'];label=labels[cid];key=(domain,cid,r['coordinate_file'])
            if key not in cache:
                T=int(r['cutoff_ms'])
                if domain=='existing':
                    if label['target_time_ms']!=T-1:raise RuntimeError('existing endpoint mismatch')
                    joint=label['target']
                else:joint=label['target_joints'][T]
                xyz,finger,bone=gt_points(joint);uv,good,inview=project(xyz,np.asarray(r['context'])[61:].reshape(3,3))
                with np.load(r['coordinate_file'],allow_pickle=False) as z:coords={k:z[k] for k in ('grid','valid','full_interp_support')}
                g=geometry(coords,uv,good,inview,finger,bone,r['arm']);g.update(projectable=good,in_view=inview)
                cache[key]=g
            records.append(dict(domain=domain,case_id=cid,arm=r['arm'],condition=r['condition'],
                recording=r.get('recording'),fold=r['fold'],partition=r.get('partition'),
                background=label.get('moving_background'),target_ms=int(r['cutoff_ms'])-(domain=='existing'),
                anatomical_slot_semantics=r['arm'] in ANATOMICAL))
            values.append(cache[key])
            if n%1000==0:print(json.dumps(dict(domain=domain,rows_done=n,GT_use='projectiononly')),flush=True)
    arrays={k:np.stack([v[k] for v in values]) for k in values[0]}
    np.savez_compressed(out/'percase.npz',**arrays)
    dump(out/'rows.json',records);result={}
    for domain in ('synthetic','existing'):
        result[domain]={}
        for arm in ARMS:
            result[domain][arm]={}
            for condition in CONDITIONS:
                select=np.asarray([r['domain']==domain and r['arm']==arm and r['condition']==condition for r in records])
                section=dict(all=summarize(records,arrays,select),by_recording={},by_context={},by_background={})
                for field,dest in (('recording','by_recording'),('fold','by_context'),('background','by_background'),('partition','by_partition')):
                    section.setdefault(dest,{})
                    for value in sorted({str(r[field]) for r,k in zip(records,select) if k and r[field] is not None}):
                        chosen=select&np.asarray([str(r[field])==value for r in records])
                        section[dest][value]=summarize(records,arrays,chosen)
                result[domain][arm][condition]=section
    after={p:sha(p)==h for p,h in identities.items()}
    if not all(after.values()):raise RuntimeError('source changed during coverage diagnosis')
    dump(out/'results.json',dict(schema='s43_field_query_gt_projection_coverage_v1',status='PASS',
        freeze_sha256=sha(freeze_path),GT_semantically_opened_utc=gt_opened,GT_opened_after_freeze_and_all_predictions=True,
        rows=len(records),unique_cases=len({(r['domain'],r['case_id']) for r in records}),
        results=result,source_files_unchanged=after,seconds=time.perf_counter()-started,
        percase_sha256=sha(out/'percase.npz'),rows_sha256=sha(out/'rows.json'),
        aliases=dict(anatomical_temporal_collapse='same sealed geometry as anatomical; not an independent coverage observation',global_only='no local query:N/A'),
        claims='GT point projection reach only; not physical contour, visibility, foreground ownership, event existence, task sufficiency or a changed Stage3 gate'))
    print(json.dumps(dict(status='PASS',rows=len(records),unique_cases=368)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=('freeze','run'))
    p.add_argument('--base',type=Path,default=OUT/'field_atomic_v1');p.add_argument('--out',type=Path,default=OUT/'field_query_coverage_v1')
    p.add_argument('--plan',type=Path,default=OUT/'field_query_coverage_plan.md')
    p.add_argument('--stage-freeze',type=Path,default=OUT/'field_query_engineering/actual_sourcefreeze.json')
    a=p.parse_args()
    if a.command=='freeze':freeze(a.base,a.out,a.plan,a.stage_freeze)
    else:run(a.out)


if __name__=='__main__':main()
