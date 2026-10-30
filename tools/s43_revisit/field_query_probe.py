"""Stage3 matched independent field-query readers; frozen B is read-only.

Query coordinates never receive current target finger/state. All coordinates
and source tensors are sealed before labels are attached. This is a finite
information diagnostic, not full51D MANO training or deployment accuracy.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json,os,time
from pathlib import Path
import numpy as np
from . import witness_probe as old

ROOT=old.ROOT;OUT=old.OUT;PROTOCOL=OUT/'stage3_field_query_protocol.json'
PROTOCOL_SHA='890568967af9bf882e78f7b16e6d08d32c1eeee93c823f1d6ea863153e3ea5b7'
ARMS=('global_only','event_grid','hand_box','static','generic_trajectory','anatomical',
      'anatomical_temporal_collapse','witness_anatomical','quota_anatomical')
CONDITIONS={
 'none':('none','none',True),
 'wrong_x':('root_x10mm','root_x10mm',True),
 'wrong_yaw':('root_yaw0.1','root_yaw0.1',True),
 'corrected_from_x':('root_x10mm','none',False),
 'corrected_from_yaw':('root_yaw0.1','none',False)}
SEED=3407;BIN_WIDTH=np.asarray([7,6,6,6,7,6,6,6],np.float32)
sha=old.sha;dump=old.dump


def collapse_time(grid):
    """Uniform native-ms intensity: preserve every spatial/polarity total."""
    x=np.asarray(grid,np.float32);out=x.copy()
    if x.shape!=(40,180,240):raise ValueError('grid40 required')
    for a,b in ((0,16),(24,40)):
        raw=np.expm1(x[a:b].astype(np.float64)).reshape(8,2,180,240)
        total=raw.sum(0)
        out[a:b]=np.log1p(total[None]*BIN_WIDTH[:,None,None,None]/50.).reshape(16,180,240)
    return out


def source_rows(domain):
    p=OUT/'witness_atomic_v1'/f'{domain}_features'
    rows=json.loads((p/'rows.json').read_text());by={}
    for row in rows:
        if row['arm'] not in ('field2','witness','quota'):continue
        by.setdefault(row['case_id'],{})[(row['arm'],row['selection_error'])]=row
    return p,by


def dependencies():
    names=['tools/s43_revisit/field_query_probe.py','tools/s43_revisit/field_query.py',
      'tools/s43_revisit/field_core.py']
    return dict(old.dependencies(),**{str((ROOT/n).resolve()):sha(ROOT/n) for n in names})


def require_gate(freeze):
    x=json.loads(Path(freeze).read_text())
    if x.get('schema')!='s43_field_query_actual_sourcefreeze_v1' or not x.get('engineering_passed'):
        raise RuntimeError('Stage3 actual engineering/source freeze required')
    if sha(PROTOCOL)!=PROTOCOL_SHA or x['protocol_sha256']!=PROTOCOL_SHA or dependencies()!=x['dependencies']:
        raise RuntimeError('Stage3 frozen implementation/protocol changed')
    for path,identity in {**x['sources'],**x['engineering']}.items():
        if sha(path)!=identity:raise RuntimeError('Stage3 frozen input changed: '+path)
    return x


def sourcefreeze(dest,engineering,commands):
    if sha(PROTOCOL)!=PROTOCOL_SHA:raise RuntimeError('wrong frozen Stage3 protocol')
    if Path(dest).exists():raise FileExistsError(dest)
    checks={}
    for name in engineering:
        x=json.loads(Path(name).read_text())
        if not (x.get('status')=='PASS' or x.get('passed') is True or x.get('unit_tests_passed') is True):raise RuntimeError('nonpassing engineering')
        checks[str(Path(name).resolve())]=sha(name)
    if not checks:raise RuntimeError('engineering required')
    paths={PROTOCOL,OUT/'field_query_engineering/implementation_extension.json',
      OUT/'witness_probe_engineering/actual_sourcefreeze.json',
      OUT/'witness_probe_engineering/environment_dependency_amendment.json',
      OUT/'witness_probe_engineering/environment_dependency_followup.json'}
    protocol=json.loads(PROTOCOL.read_text())
    for name,identity in protocol['prior_identity'].items():
        if sha(ROOT/name)!=identity:raise RuntimeError('preregistered prior artifact changed')
        paths.add(ROOT/name)
    import threadpoolctl
    paths.add(Path(threadpoolctl.__file__))
    for domain in ('synthetic','existing'):
        base,by=source_rows(domain);paths.update(base/f for f in ('rows.json','labels.json','complete.json'))
        for case in by.values():
            for row in case.values():
                path=Path(row['feature_file'])
                if sha(path)!=row['feature_sha256']:raise RuntimeError('sealed B tensor changed')
                paths.add(path)
    dump(dest,dict(schema='s43_field_query_actual_sourcefreeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),
      protocol_sha256=sha(PROTOCOL),dependencies=dependencies(),sources={str(p.resolve()):sha(p) for p in sorted(paths)},
      engineering=checks,engineering_passed=True,commands=commands,new_A_task_scores_read=False))


def _query_arrays(query):
    return dict(grid=np.asarray(query.grid,np.float32),valid=np.asarray(query.valid,bool),
      in_view=np.asarray(query.in_view,bool),full_interp_support=np.asarray(query.full_interp_support,bool),
      frame_degenerate=np.asarray(query.frame_degenerate,bool),depth_valid=np.asarray(query.depth_valid,bool),
      history_supported=np.asarray(query.history_supported,bool),centers_px=np.asarray(query.centers_px,np.float32),
      bin_edges_ms=np.asarray(query.bin_edges_ms,np.int64),bin_times_ms=np.asarray(query.bin_times_ms,np.float64))


def prepare(domain,dest,freeze):
    require_gate(freeze)
    from .field_query import FieldQueryBuilder,FieldQueryConfig
    from .witness_geometry import fk_joints
    dest=Path(dest)
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True);(dest/'coordinates').mkdir()
    base,by=source_rows(domain);rows=[];started=time.perf_counter()
    for cid,sources in by.items():
        source=sources[('field2','none')];c=np.asarray(source['context'],np.float64)
        anchor=c[:51];beta=c[51:61];K_source=c[61:].reshape(3,3);K=K_source.copy();K[:2]*=.375
        T=int(source['cutoff_ms']);ref=T-50
        with np.load(source['feature_file'],allow_pickle=False) as z:complete=np.asarray(z['grid'],np.float32)
        support=complete[:16].reshape(8,2,180,240)[:,:,::2,::2].any((0,1))
        coords={}
        for arm in ARMS:
            storage='witness' if arm=='witness_anatomical' else 'quota' if arm=='quota_anatomical' else 'field2'
            mode='anatomical' if arm in ('anatomical_temporal_collapse','witness_anatomical','quota_anatomical') else arm
            for condition,(selected_error,query_error,training_view) in CONDITIONS.items():
                q=old.inject_anchor(anchor,query_error);context=old.condition70(q,beta,K_source)
                resolution=2 if storage=='field2' else 1
                key=(mode,query_error,resolution)
                if key not in coords:
                    if mode=='global_only':
                        arrays=dict(grid=np.zeros((8,90,49,2),np.float32),valid=np.zeros((8,90,49),bool),
                          in_view=np.zeros((8,90,49),bool),full_interp_support=np.zeros((8,90,49),bool));meta=dict(mode='global_only')
                    else:
                        query=FieldQueryBuilder(q,beta,K,ref_ms=ref,history=(),
                          config=FieldQueryConfig(field_resolution=resolution)).build(T,mode=mode,event_counts=support)
                        arrays=_query_arrays(query);meta=query.meta
                    path=dest/'coordinates'/f'{cid}_{mode}_{query_error}_r{resolution}.npz';np.savez_compressed(path,**arrays)
                    coords[key]=(str(path.resolve()),sha(path),meta)
                file,identity,meta=coords[key]
                selected=sources[(storage,selected_error)]
                rows.append(dict(case_id=cid,condition=condition,arm=arm,training_view=training_view,
                  selected_error=selected_error,query_error=query_error,cutoff_ms=T,fold=source['fold'],
                  partition=source.get('partition'),recording=source.get('recording'),
                  source_file=selected['feature_file'],source_sha256=selected['feature_sha256'],resolution=2 if storage=='field2' else 1,
                  transform='collapse' if arm=='anatomical_temporal_collapse' else 'none',
                  coordinate_file=file,coordinate_sha256=identity,coordinate_metadata=meta,context=context.tolist(),
                  query_anchor=q.tolist(),query_prior=fk_joints(q,beta).tolist(),target_not_used=True))
        print(json.dumps(dict(case=cid,coordinates_sealed=True,targets_not_read=True)),flush=True)
    dump(dest/'observations_complete.json',dict(domain=domain,cases=len(by),rows=len(rows),targets_read=False,
      source_freeze_sha256=sha(freeze),seconds=time.perf_counter()-started))
    labels=json.loads((base/'labels.json').read_text())
    if domain=='existing':
        maximum=max(float(np.max(np.abs(np.asarray(r['query_prior'])-np.asarray(labels[r['case_id']]['prior']))))
          for r in rows if r['condition']=='none')
        if maximum>1e-5:raise RuntimeError('current FK vs old cached prior mismatch >1e-5m')
    else:maximum=None
    dump(dest/'rows.json',rows);dump(dest/'labels.json',labels)
    # A correction queries unchanged complete evidence; B correction keeps the wrong selected evidence.
    for cid in by:
      for arm in ARMS[:7]:
        rr={r['condition']:r for r in rows if r['case_id']==cid and r['arm']==arm}
        for error in ('corrected_from_x','corrected_from_yaw'):
            assert rr[error]['coordinate_sha256']==rr['none']['coordinate_sha256']
            assert rr[error]['source_sha256']==rr['none']['source_sha256'] and rr[error]['context']==rr['none']['context']
    dump(dest/'complete.json',dict(schema='s43_field_query_features_v1',domain=domain,cases=len(by),rows=len(rows),
      source_freeze_sha256=sha(freeze),dependencies=dependencies(),source_then_label=True,
      normal_FK_vs_oldprior_max_m=maximum,corrected_complete_field_requery_exact_normal=True,
      rows_sha256=sha(dest/'rows.json'),labels_sha256=sha(dest/'labels.json')))
    require_gate(freeze)


def make_cnn(output,local_enabled=True):
    import torch
    class Reader(torch.nn.Module):
        def __init__(self):
            super().__init__();self.local_enabled=local_enabled
            self.global_encoder=torch.nn.Sequential(torch.nn.Conv2d(40,8,3,padding=1),torch.nn.SiLU(),torch.nn.AvgPool2d(4),
              torch.nn.Conv2d(8,16,3,padding=1),torch.nn.SiLU(),torch.nn.AdaptiveAvgPool2d((6,8)))
            self.local_encoder=torch.nn.Sequential(torch.nn.Conv2d(5,8,3,padding=1),torch.nn.SiLU(),
              torch.nn.Conv2d(8,8,3,padding=1),torch.nn.SiLU(),torch.nn.AdaptiveAvgPool2d(1))
            self.head=torch.nn.Sequential(torch.nn.Linear(768+5760+70,128),torch.nn.SiLU(),
              torch.nn.Linear(128,128),torch.nn.SiLU(),torch.nn.Linear(128,output))
        def forward(self,global_grid,local_patch,context):
            g=self.global_encoder(global_grid).flatten(1)
            if self.local_enabled:
                l=self.local_encoder(local_patch.reshape(-1,5,7,7)).reshape(len(g),5760)
            else:l=torch.zeros((len(g),5760),device=g.device,dtype=g.dtype)
            return self.head(torch.cat((g,l,context),1))
    return Reader()


def sample_patches(global_grid,coordinates,resolution):
    """5 channels F2 + observedfinecoverage + inview + fullsupport, no coords-as-features."""
    import torch
    import torch.nn.functional as F
    x=global_grid
    fine=x[:16].reshape(8,2,180,240);coverage=x[16:24].reshape(8,1,180,240)
    data=torch.cat((fine,coverage),1)
    if resolution==2:data=data[:,:,::2,::2]
    elif resolution!=1:raise ValueError('declared native1px or cell2px resolution')
    grid=torch.as_tensor(coordinates['grid'],device=x.device).reshape(8,90,49,2)
    sampled=F.grid_sample(data,grid,mode='bilinear',padding_mode='zeros',align_corners=False)
    valid=torch.as_tensor(coordinates['valid'],device=x.device)
    sampled=sampled*valid[:,None]
    inview=torch.as_tensor(coordinates['in_view'],device=x.device,dtype=x.dtype)
    support=torch.as_tensor(coordinates['full_interp_support'],device=x.device,dtype=x.dtype)
    return torch.cat((sampled,inview[:,None],support[:,None]),1).permute(0,2,1,3).reshape(8,90,5,7,7).contiguous()


def load_banks(rows,device):
    import torch
    globals={};locals={};gx=[];lx=[]
    for row in rows:
        key=(row['source_file'],row['transform'])
        if key not in globals:globals[key]=len(globals)
        gx.append(globals[key])
        key2=(key,row['coordinate_file'],row['resolution'],row['arm']=='global_only')
        if key2 not in locals:locals[key2]=len(locals)
        lx.append(locals[key2])
    gb=torch.empty((len(globals),40,180,240),device=device,dtype=torch.float32)
    for (file,transform),i in globals.items():
        with np.load(file,allow_pickle=False) as z:x=np.asarray(z['grid'],np.float32)
        if transform=='collapse':x=collapse_time(x)
        gb[i]=torch.as_tensor(x,device=device)
    lb=torch.empty((len(locals),8,90,5,7,7),device=device,dtype=torch.float32)
    for (key,file,resolution,disabled),i in locals.items():
        if disabled:lb[i].zero_();continue
        with np.load(file,allow_pickle=False) as z:coords={k:z[k] for k in ('grid','valid','in_view','full_interp_support')}
        lb[i]=sample_patches(gb[globals[key]],coords,resolution)
    return gb,lb,np.asarray(gx,np.int64),np.asarray(lx,np.int64)


class LowCapacityRBF:
    def fit(self,x,c,y):
        self.scale=np.maximum(np.sqrt(np.mean(x.astype(np.float64)**2,0)),1.)
        self.cm=c.mean(0);self.cs=c.std(0);n=int(np.sum(self.cs>=1e-6));self.cs[self.cs<1e-6]=1.;self.context_scale=np.sqrt(max(n,1))
        self.library=self.transform(x,c);d=old.RBFReader.distance(self.library,self.library)
        v=d[np.triu_indices(len(d),1)];v=v[v>1e-12];self.bandwidth=max(float(np.median(v)) if len(v) else 1.,1e-6)
        K=np.exp(-d/self.bandwidth);K.flat[::len(K)+1]+=.01;self.coef=np.linalg.solve(K,y.astype(np.float64));self.kernel_bytes=K.nbytes+d.nbytes
        return self
    def transform(self,x,c):
        xx=np.clip(x/self.scale,-8,8).astype(np.float64)
        xx[:,:1920]*=np.sqrt((1/2)/1920);xx[:,1920:]*=np.sqrt((1/2)/(x.shape[1]-1920))
        cc=np.clip((c-self.cm)/self.cs,-8,8)/self.context_scale
        return np.concatenate((xx,cc),1)
    def predict(self,x,c):return np.exp(-old.RBFReader.distance(self.transform(x,c),self.library)/self.bandwidth)@self.coef
    def save(self,path):np.savez_compressed(path,**self.__dict__)
    def accounting(self):return dict(library_rows=len(self.library),coefficient_parameters=self.coef.size,
      model_array_bytes=sum(v.nbytes for v in self.__dict__.values() if isinstance(v,np.ndarray)),offline_kernel_bytes=self.kernel_bytes)


def target_array(rows,labels,domain):
    if domain=='synthetic':return np.eye(10,dtype=np.float32)[[labels[r['case_id']]['finger']*2+int(labels[r['case_id']]['sign']<0) for r in rows]]
    return np.asarray([(np.asarray(labels[r['case_id']]['target'])-np.asarray(r['query_prior'])).reshape(63)*1000 for r in rows],np.float32)


def fit(domain,arm,features,dest,freeze,device):
    require_gate(freeze);old.validate_device(device)
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    features=Path(features);dest=Path(dest)
    if dest.exists():raise FileExistsError(dest)
    complete=json.loads((features/'complete.json').read_text())
    if complete['source_freeze_sha256']!=sha(freeze):raise RuntimeError('wrong feature sourcefreeze')
    if sha(features/'rows.json')!=complete['rows_sha256'] or sha(features/'labels.json')!=complete['labels_sha256']:
        raise RuntimeError('prepared inputs/labels changed')
    rows=[r for r in json.loads((features/'rows.json').read_text()) if r['arm']==arm];labels=json.loads((features/'labels.json').read_text())
    for file,identity in {(r['coordinate_file'],r['coordinate_sha256']) for r in rows}|{(r['source_file'],r['source_sha256']) for r in rows}:
        if sha(file)!=identity:raise RuntimeError('fixed tensor/coordinate changed')
    dest.mkdir(parents=True);gb,lb,gi,li=load_banks(rows,device)
    context=np.asarray([r['context'] for r in rows],np.float32);y=target_array(rows,labels,domain)
    # Fixed low capacity features keep every token/time but average its49 spatial samples.
    pooled_global=gb.reshape(len(gb),40,6,30,8,30).mean((3,5)).reshape(len(gb),1920).cpu().numpy()
    pooled_local=lb.mean((-1,-2)).reshape(len(lb),3600).cpu().numpy()
    x=np.concatenate((pooled_global[gi],pooled_local[li]),1)
    folds=range(4) if domain=='synthetic' else (-1,)
    accounts={}
    for fold in folds:
        foldfit=np.asarray([r['fold']!=fold if domain=='synthetic' else r['partition']=='FIT7' for r in rows])
        eligible=np.asarray([r['training_view'] for r in rows]);train=foldfit&eligible;ids=np.flatnonzero(train)
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);rng=np.random.default_rng(SEED)
        model=make_cnn(y.shape[1],arm!='global_only').to(device);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        gweight=np.bincount(gi[train],minlength=len(gb));lweight=np.bincount(li[train],minlength=len(lb))
        gs=np.zeros(40);ls=np.zeros(5)
        for i,n in enumerate(gweight):
            if n:gs+=n*gb[i].double().square().mean((1,2)).cpu().numpy()
        for i,n in enumerate(lweight):
            if n:ls+=n*lb[i].double().square().mean((0,1,3,4)).cpu().numpy()
        gs=np.maximum(np.sqrt(gs/len(ids)),1.).astype(np.float32);ls=np.maximum(np.sqrt(ls/len(ids)),1.).astype(np.float32)
        gst=torch.as_tensor(gs[None,:,None,None],device=device);lst=torch.as_tensor(ls[None,None,None,:,None,None],device=device)
        cm=context[train].mean(0);cs=context[train].std(0);cs[cs<1e-6]=1.;ct=torch.as_tensor(np.clip((context-cm)/cs,-8,8),device=device)
        yt=torch.as_tensor(y,device=device);ym=y[train].mean(0);ys=np.maximum(y[train].std(0),1.)
        ymt=torch.as_tensor(ym,device=device);yst=torch.as_tensor(ys,device=device)
        gt=torch.as_tensor(gi,device=device);lt=torch.as_tensor(li,device=device);curve=[];started=time.perf_counter()
        for step in range(2000):
            ii=torch.as_tensor(rng.choice(ids,32,replace=True),device=device)
            optimizer.zero_grad(set_to_none=True)
            pp=model((gb[gt[ii]]/gst).clamp(-8,8),(lb[lt[ii]]/lst).clamp(-8,8),ct[ii])
            loss=torch.nn.functional.cross_entropy(pp,yt[ii].argmax(1)) if domain=='synthetic' else torch.nn.functional.mse_loss(pp,(yt[ii]-ymt)/yst)
            if not bool(torch.isfinite(loss).item()):raise RuntimeError('nonfinite training loss; no task score')
            loss.backward();optimizer.step()
            if step%500==0 or step==1999:
                curve.append([step+1,float(loss.detach().cpu())]);print(json.dumps(dict(arm=arm,fold=fold,step=step+1,FIT_loss=curve[-1][1],SELECT_not_scored=True)),flush=True)
        if device!='cpu':torch.cuda.synchronize()
        elapsed=time.perf_counter()-started;pred=[];model.eval()
        with torch.inference_mode():
            for a in range(0,len(rows),32):
                ii=slice(a,a+32);pp=model((gb[gt[ii]]/gst).clamp(-8,8),(lb[lt[ii]]/lst).clamp(-8,8),ct[ii])
                if domain!='synthetic':pp=pp*yst+ymt
                pred.append(pp.cpu().numpy())
        pred=np.concatenate(pred)
        if not np.isfinite(pred).all():raise RuntimeError('nonfinite CNN prediction')
        folder=dest/f'fold{fold}';folder.mkdir()
        torch.save(dict(model=model.state_dict(),steps=2000,seed=SEED,output=y.shape[1],arm=arm,global_scale=gs,local_scale=ls,context_mean=cm,context_std=cs,target_mean=ym,target_std=ys),folder/'cnn.pt')
        np.savez_compressed(folder/'cnn_predictions.npz',prediction=pred,fit=train,foldfit=foldfit)
        params=sum(p.numel() for p in model.parameters());localparams=sum(p.numel() for p in model.local_encoder.parameters())
        cnn=dict(parameters=params,structurally_active_parameters=params if arm!='global_only' else params-localparams-5760*128,
          steps=2000,batch=32,seed=SEED,optimizer_loop_seconds=elapsed,FIT_curve=curve,device=device,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
          global_offline_dataset_cache_bytes=gb.numel()*4,local_offline_dataset_cache_bytes=lb.numel()*4,
          precision='FP32 noTF32 noautocast',train_views=len(ids),torch_version=torch.__version__,
          cudnn_deterministic=True,device_name='cpu' if device=='cpu' else torch.cuda.get_device_name(),CPU_threads=8)
        started=time.perf_counter()
        with threadpool_limits(8):
            head=LowCapacityRBF().fit(x[train],context[train],y[train]);p=head.predict(x,context)
        if not np.isfinite(p).all():raise RuntimeError('nonfinite RBF prediction')
        head.save(folder/'rbf.npz');np.savez_compressed(folder/'rbf_predictions.npz',prediction=p,fit=train,foldfit=foldfit)
        accounts[str(fold)]=dict(cnn=cnn,rbf=dict(head.accounting(),fit_prediction_seconds=time.perf_counter()-started))
        del model,optimizer
        if device!='cpu':torch.cuda.empty_cache()
    artifacts={str(file.resolve()):sha(file) for fold in folds for file in (dest/f'fold{fold}').iterdir() if file.is_file()}
    dump(dest/'predictions_complete.json',dict(domain=domain,arm=arm,folds=list(folds),accounts=accounts,
      artifacts=artifacts,feature_complete_sha256=sha(features/'complete.json'),
      source_freeze_sha256=sha(freeze),SELECT_metrics_not_computed=True))
    require_gate(freeze)


def _query_metrics(pred,rows,labels,domain):
    """All GT access is scoring/supervision, never coordinate construction."""
    if domain=='existing':
        rr=[];ll={}
        for r in rows:
            key=r['case_id']+'/'+r['condition'];item=dict(r,case_id=key);rr.append(item)
            ll[key]=dict(labels[r['case_id']],prior=r['query_prior'])
        return old.metrics(pred,rr,ll,'existing')
    canonical=old.metrics(pred,rows,labels,'synthetic')
    rr=[];ll={}
    for r in rows:
        key=r['case_id']+'/'+r['condition'];rr.append(dict(r,case_id=key))
        ll[key]=dict(labels[r['case_id']],public_anchor=r['query_anchor'])
    deployed=old.metrics(pred,rr,ll,'synthetic')
    canonical['query_anchor_candidate_decode']={k:v for k,v in deployed.items() if k.endswith('_mm')}
    canonical['canonical_decode_scope']='known synthetic public boundary finite-class diagnostic; not root correction/deployment accuracy'
    by={}
    for bg in (False,True):
        take=np.asarray([labels[r['case_id']]['moving_background']==bg for r in rows])
        if take.any():by[str(bg)]=old.metrics(pred[take],[r for r,k in zip(rows,take) if k],labels,'synthetic')
    canonical['by_background']=by
    return canonical


def score(features_root,readers_root,dest,freeze):
    """Require every fixed arm/fold prediction before any Stage3 task metric."""
    require_gate(freeze);features_root=Path(features_root);readers_root=Path(readers_root);dest=Path(dest)
    if dest.exists():raise FileExistsError(dest)
    for domain in ('synthetic','existing'):
      fc=json.loads((features_root/domain/'complete.json').read_text())
      if sha(features_root/domain/'rows.json')!=fc['rows_sha256'] or sha(features_root/domain/'labels.json')!=fc['labels_sha256']:
          raise RuntimeError('prepared inputs/labels changed before scoring')
      for arm in ARMS:
        complete=json.loads((readers_root/domain/arm/'predictions_complete.json').read_text())
        if complete['source_freeze_sha256']!=sha(freeze):raise RuntimeError('wrong predictions freeze')
        if complete['feature_complete_sha256']!=sha(features_root/domain/'complete.json'):raise RuntimeError('feature/prediction identity mismatch')
        for file,identity in complete['artifacts'].items():
            if sha(file)!=identity:raise RuntimeError('fixed fitted artifact changed')
        expected=list(range(4)) if domain=='synthetic' else [-1]
        if complete['folds']!=expected:raise RuntimeError('incomplete fold list')
        for fold in expected:
          for kind in ('cnn','rbf'):
            file=readers_root/domain/arm/f'fold{fold}'/f'{kind}_predictions.npz'
            if str(file.resolve()) not in complete['artifacts']:raise RuntimeError('missing fixed prediction artifact')
            with np.load(file,allow_pickle=False) as z:
                expected_shape=(400,10) if domain=='synthetic' else (1440,63)
                if z['prediction'].shape!=expected_shape or not np.isfinite(z['prediction']).all():raise RuntimeError('bad fixed prediction shape/value')
    dest.mkdir(parents=True);result={}
    for domain in ('synthetic','existing'):
        allrows=json.loads((features_root/domain/'rows.json').read_text());labels=json.loads((features_root/domain/'labels.json').read_text())
        result[domain]={};folds=range(4) if domain=='synthetic' else (-1,)
        for arm in ARMS:
            rows=[r for r in allrows if r['arm']==arm];result[domain][arm]={}
            for kind in ('cnn','rbf'):
                per_fold={};oof=None
                for fold in folds:
                    path=readers_root/domain/arm/f'fold{fold}'/f'{kind}_predictions.npz'
                    with np.load(path,allow_pickle=False) as z:prediction=z['prediction'];foldfit=z['foldfit']
                    if not np.isfinite(prediction).all():raise RuntimeError('nonfinite fixed prediction')
                    if oof is None:oof=np.zeros_like(prediction)
                    oof[~foldfit]=prediction[~foldfit]
                    metrics={}
                    for split,mask in [('FIT',foldfit),('SELECT',~foldfit)]:
                      for condition in CONDITIONS:
                        chosen=mask&np.asarray([r['condition']==condition for r in rows])
                        rr=[r for r,k in zip(rows,chosen) if k]
                        metrics[f'{split}/{condition}']=_query_metrics(prediction[chosen],rr,labels,domain)
                    per_fold[str(fold)]=metrics
                pooled={}
                for condition in CONDITIONS:
                    chosen=np.asarray([r['condition']==condition for r in rows])
                    if domain=='existing':chosen&=np.asarray([r['partition']=='SELECT' for r in rows])
                    rr=[r for r,k in zip(rows,chosen) if k];pooled[condition]=_query_metrics(oof[chosen],rr,labels,domain)
                result[domain][arm][kind]=dict(per_fold=per_fold,held_out_pooled=pooled)
        if domain=='existing':
            prior={}
            rows=[r for r in allrows if r['arm']=='global_only']
            for condition in CONDITIONS:
                rr=[r for r in rows if r['condition']==condition and r['partition']=='SELECT']
                prior[condition]=_query_metrics(np.zeros((len(rr),63)),rr,labels,domain)
            result[domain]['unchanged_query_prior']=prior
    dump(dest/'results.json',dict(schema='s43_field_query_atomic_results_v1',source_freeze_sha256=sha(freeze),
      protocol_sha256=sha(PROTOCOL),results=result,all_fixed_predictions_saved_before_scoring=True,
      independent_matched_heads=True,not_full_MANO_system=True,no_novelty_certification=True))
    require_gate(freeze)


def engineering(dest):
    import torch
    torch.set_num_threads(8);torch.manual_seed(SEED)
    model=make_cnn(63);g=torch.zeros(2,40,180,240);l=torch.ones(2,8,90,5,7,7);c=torch.zeros(2,70)
    pred=model(g,l,c);pred.square().mean().backward()
    assert pred.shape==(2,63) and model.local_encoder[0].weight.grad.abs().sum()>0
    values=np.zeros((40,180,240),np.float32);values[0,4,4]=np.log1p(3);values[6,4,4]=np.log1p(5);values[16:24]=1
    transformed=collapse_time(values)
    np.testing.assert_allclose(np.expm1(transformed[:16].astype(float)).reshape(8,2,180,240).sum(0),
      np.expm1(values[:16].astype(float)).reshape(8,2,180,240).sum(0),rtol=1e-6,atol=1e-7)
    np.testing.assert_array_equal(values[16:24],transformed[16:24])
    dump(dest,dict(status='PASS',schema='s43_field_query_reader_engineering_v1',created_utc=datetime.now(timezone.utc).isoformat(),
      source_sha256=sha(__file__),task_data_opened=False,new_task_scores_read=False,parameters_regression=sum(p.numel() for p in model.parameters()),
      parameters_classification=sum(p.numel() for p in make_cnn(10).parameters()),
      matched_architecture_all_modes=True,local_shape=[8,90,5,7,7],local_sign_slots_independent=True,
      temporal_collapse_preserves_spatial_polarity_totals_with_float32_tolerance=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('freeze');a.add_argument('--out',type=Path,required=True);a.add_argument('--engineering',nargs='+',required=True);a.add_argument('--commands',nargs='+',required=True)
    a=sub.add_parser('prepare');a.add_argument('--domain',choices=('synthetic','existing'),required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--source-freeze',type=Path,required=True)
    a=sub.add_parser('engineering');a.add_argument('--out',type=Path,required=True)
    a=sub.add_parser('fit');a.add_argument('--domain',choices=('synthetic','existing'),required=True);a.add_argument('--arm',choices=ARMS,required=True);a.add_argument('--features',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--source-freeze',type=Path,required=True);a.add_argument('--device',default='cpu')
    a=sub.add_parser('score');a.add_argument('--features-root',type=Path,required=True);a.add_argument('--readers-root',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--source-freeze',type=Path,required=True)
    args=p.parse_args()
    if args.command=='freeze':sourcefreeze(args.out,args.engineering,args.commands)
    elif args.command=='prepare':prepare(args.domain,args.out,args.source_freeze)
    elif args.command=='engineering':engineering(args.out)
    elif args.command=='score':score(args.features_root,args.readers_root,args.out,args.source_freeze)
    else:fit(args.domain,args.arm,args.features,args.out,args.source_freeze,args.device)


if __name__=='__main__':main()
