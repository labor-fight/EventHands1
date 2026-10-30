"""Stage4 identity-initialized MANO51 vs free-joint63 readers; never full training.

Every fixed prediction is saved before scoring. Both readers share inputs,
supervision, hidden architecture, normalization and optimization budget.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from . import field_query_probe as P

ROOT=P.ROOT;OUT=P.OUT;PROTOCOL=OUT/'stage4_mano_reader_protocol.json'
PROTOCOL_SHA='abbc69ebccc791974ef340d9498933cd5feb2e79e46542c11c393a00a4f235a1'
ADDENDUM=OUT/'stage4_nominal_guard_addendum.json'
ADDENDUM_SHA='7dad0bb50db059156ee29c43a7b186122a02db5b2183c11c6aa520c77dc8ea7f'
ARMS=('context_only','global_only','hand_box','static','static_temporal_collapse','anatomical','quota_anatomical')
READERS=('joint63','mano51');SEED=3407
SOURCE_ARM={a:('global_only' if a=='context_only' else 'static' if a=='static_temporal_collapse' else a) for a in ARMS}
sha=P.sha


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,
      default=lambda v:v.tolist() if isinstance(v,np.ndarray) else v.item() if isinstance(v,np.generic) else str(v))+'\n',encoding='utf-8')


class IdentityFeatureReader(nn.Module):
    def __init__(self,arm,reader):
        super().__init__()
        if arm not in ARMS or reader not in READERS:raise ValueError('unknown arm/reader')
        self.arm,self.reader=arm,reader;self.use_global=arm!='context_only';self.use_local=arm not in ('context_only','global_only')
        same=P.make_cnn(51 if reader=='mano51' else 63,self.use_local)
        self.global_encoder,self.local_encoder,self.head=same.global_encoder,same.local_encoder,same.head
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
        for module,enabled in ((self.global_encoder,self.use_global),(self.local_encoder,self.use_local)):
            for p in module.parameters():p.requires_grad_(enabled)

    def forward(self,grid,local,context):
        B=len(context)
        g=self.global_encoder(grid).flatten(1) if self.use_global else context.new_zeros((B,768))
        l=self.local_encoder(local.reshape(-1,5,7,7)).reshape(B,5760) if self.use_local else context.new_zeros((B,5760))
        return self.head(torch.cat((g,l,context),1))

    def accounting(self):
        total=sum(p.numel() for p in self.parameters());inactive=0
        for m,on in ((self.global_encoder,self.use_global),(self.local_encoder,self.use_local)):
            if not on:inactive+=sum(p.numel() for p in m.parameters())
        inactive+=(0 if self.use_global else 768*128)+(0 if self.use_local else 5760*128)
        return dict(parameters=total,structurally_active_parameters=total-inactive,
          trainable_parameters=sum(p.numel() for p in self.parameters() if p.requires_grad),
          output_width=51 if self.reader=='mano51' else 63,final_zero_initialized=True)


def joint_mse(prediction_m,target_m):
    return ((prediction_m-target_m)/.01).square().mean()


def target_joints(rows,labels,domain):
    return np.asarray([labels[r['case_id']]['target'] if domain=='existing' else
        labels[r['case_id']]['target_joints'][int(r['cutoff_ms'])] for r in rows],np.float32)


def decoder_instance(device):
    from .field_mano_decoder import FieldManoDecoder
    return FieldManoDecoder().to(device)


def decode_joints(raw,anchor,beta,reader,decoder,identity_joints):
    if reader=='joint63':return identity_joints+.01*raw.reshape(-1,21,3),None
    value=decoder(anchor,beta,raw)
    return value.joints,value


def dependencies():
    names=['tools/s43_revisit/newfield_mano_probe.py','tools/s43_revisit/field_mano_decoder.py']
    return dict(P.dependencies(),**{str((ROOT/n).resolve()):sha(ROOT/n) for n in names})


def require_gate(freeze):
    f=json.loads(Path(freeze).read_text(encoding='utf-8'))
    if sha(PROTOCOL)!=PROTOCOL_SHA or sha(ADDENDUM)!=ADDENDUM_SHA or f['protocol_sha256']!=PROTOCOL_SHA or f['dependencies']!=dependencies():
        raise RuntimeError('Stage4 frozen source/protocol changed')
    if f.get('schema')!='s43_field_identity_mano_actual_sourcefreeze_v1' or not f['engineering_passed']:
        raise RuntimeError('actual Stage4 sourcefreeze required')
    for path,identity in {**f['sources'],**f['engineering']}.items():
        if sha(path)!=identity:raise RuntimeError('Stage4 input identity changed: '+path)
    return f


def sourcefreeze(dest,engineering,commands):
    if Path(dest).exists():raise FileExistsError(dest)
    if sha(PROTOCOL)!=PROTOCOL_SHA:raise RuntimeError('wrong Stage4 protocol')
    checks={};current_probe=sha(__file__);current_decoder=sha(ROOT/'tools/s43_revisit/field_mano_decoder.py');own=False
    for file in engineering:
        x=json.loads(Path(file).read_text(encoding='utf-8'))
        if not (x.get('status')=='PASS' or x.get('unit_tests_passed') is True):raise RuntimeError('nonpassing engineering')
        if x.get('schema')=='s43_field_identity_reader_engineering_v1':
            if x['probe_sha256']!=current_probe or x['decoder_sha256']!=current_decoder:raise RuntimeError('stale reader engineering')
            own=True
        checks[str(Path(file).resolve())]=sha(file)
    if not own:raise RuntimeError('current two-reader identity/FK gradient engineering required')
    if sha(ADDENDUM)!=ADDENDUM_SHA:raise RuntimeError('wrong prefit nominal guard addendum')
    files={PROTOCOL,ADDENDUM,OUT/'field_mano_engineering/implementation_extension.json',
      OUT/'field_mano_engineering/field_mano_fitqueue.py',ROOT/'tests/test_s43_field_mano_probe.py'}
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    for name,identity in protocol['prior_identity'].items():
        if sha(ROOT/name)!=identity:raise RuntimeError('prior artifact changed')
        files.add(ROOT/name)
    for name,identity in json.loads(ADDENDUM.read_text(encoding='utf-8'))['prior_identity'].items():
        if sha(ROOT/name)!=identity:raise RuntimeError('guard prior artifact changed')
        files.add(ROOT/name)
    for domain in ('synthetic','existing'):
        d=OUT/'field_atomic_v1/features'/domain
        files.update(d/n for n in ('rows.json','labels.json','complete.json'))
        rows=json.loads((d/'rows.json').read_text(encoding='utf-8'))
        for r in rows:
            if r['arm'] not in set(SOURCE_ARM.values()):continue
            for path,key in ((r['source_file'],'source_sha256'),(r['coordinate_file'],'coordinate_sha256')):
                if sha(path)!=r[key]:raise RuntimeError('sealed Stage3 input changed')
                files.add(Path(path))
    dump(dest,dict(schema='s43_field_identity_mano_actual_sourcefreeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),
      protocol_sha256=PROTOCOL_SHA,nominal_guard_addendum_sha256=ADDENDUM_SHA,dependencies=dependencies(),sources={str(p.resolve()):sha(p) for p in sorted(files)},
      engineering=checks,engineering_passed=True,commands=commands,new_Stage4_task_scores_read=False))


def prepare(domain,dest,freeze):
    require_gate(freeze);dest=Path(dest)
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True);source=OUT/'field_atomic_v1/features'/domain
    oldrows=json.loads((source/'rows.json').read_text(encoding='utf-8'));rows=[]
    for arm in ARMS:
        for original in oldrows:
            if original['arm']!=SOURCE_ARM[arm]:continue
            row=dict(original,arm=arm,source_arm=original['arm'])
            if arm=='static_temporal_collapse':row['transform']='collapse'
            rows.append(row)
    expected=(80 if domain=='synthetic' else 288)*len(ARMS)*5
    if len(rows)!=expected:raise RuntimeError('prepared row count mismatch')
    dump(dest/'rows.json',rows)
    dump(dest/'observations_complete.json',dict(domain=domain,rows=len(rows),targets_decoded=False,
      input_then_target=True,source_freeze_sha256=sha(freeze)))
    shutil.copyfile(source/'labels.json',dest/'labels.json')
    dump(dest/'complete.json',dict(schema='s43_field_identity_reader_features_v1',domain=domain,rows=len(rows),
      source_freeze_sha256=sha(freeze),rows_sha256=sha(dest/'rows.json'),labels_sha256=sha(dest/'labels.json'),
      labels_copied_only_after_observations_sealed=True,source_then_label=True))
    require_gate(freeze)


def input_normalization(gb,lb,gi,li,context,train):
    ids=np.flatnonzero(train);gw=np.bincount(gi[train],minlength=len(gb));lw=np.bincount(li[train],minlength=len(lb))
    gs=np.zeros(40);ls=np.zeros(5)
    for i,n in enumerate(gw):
        if n:gs+=n*gb[i].double().square().mean((1,2)).cpu().numpy()
    for i,n in enumerate(lw):
        if n:ls+=n*lb[i].double().square().mean((0,1,3,4)).cpu().numpy()
    gs=np.maximum(np.sqrt(gs/len(ids)),1.).astype(np.float32);ls=np.maximum(np.sqrt(ls/len(ids)),1.).astype(np.float32)
    cm=context[train].mean(0);cs=context[train].std(0);cs[cs<1e-6]=1
    return gs,ls,cm,cs


def training_masks(rows,domain,fold,regime):
    if regime not in ('all3','nominal_guard'):raise ValueError('training regime')
    foldfit=np.asarray([r['fold']!=fold if domain=='synthetic' else r['partition']=='FIT7' for r in rows])
    train=foldfit&np.asarray([r['training_view'] for r in rows])
    if regime=='nominal_guard':train&=np.asarray([r['condition']=='none' for r in rows])
    return foldfit,train


def fit(domain,arm,reader,features,dest,freeze,device,regime='all3'):
    if regime=='nominal_guard' and (domain!='existing' or arm not in ('hand_box','static')):raise ValueError('guard outside frozenfourfits')
    require_gate(freeze);P.old.validate_device(device)
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    features,dest=Path(features),Path(dest)
    if dest.exists():raise FileExistsError(dest)
    f=json.loads((features/'complete.json').read_text(encoding='utf-8'))
    if f['source_freeze_sha256']!=sha(freeze):raise RuntimeError('wrong prepared sourcefreeze')
    for key,file in [('rows_sha256','rows.json'),('labels_sha256','labels.json')]:
        if f[key]!=sha(features/file):raise RuntimeError('prepared inputs changed')
    rows=[r for r in json.loads((features/'rows.json').read_text(encoding='utf-8')) if r['arm']==arm]
    for file,identity in {(r['source_file'],r['source_sha256']) for r in rows}|{(r['coordinate_file'],r['coordinate_sha256']) for r in rows}:
        if sha(file)!=identity:raise RuntimeError('fixed observation/coordinate changed')
    dest.mkdir(parents=True);gb,lb,gi,li=P.load_banks(rows,device)
    context=np.asarray([r['context'] for r in rows],np.float32)
    anchor=torch.as_tensor(context[:,:51],device=device);beta=torch.as_tensor(context[:,51:61],device=device)
    decoder=decoder_instance(device);identity=[]
    with torch.no_grad():
        for a in range(0,len(rows),32):identity.append(decoder(anchor[a:a+32],beta[a:a+32],torch.zeros_like(anchor[a:a+32])).joints)
    identity=torch.cat(identity)
    # Target materialization follows every observation tensor and anchor FK.
    labels=json.loads((features/'labels.json').read_text(encoding='utf-8'));target=torch.as_tensor(target_joints(rows,labels,domain),device=device)
    if target.shape!=(len(rows),21,3) or not torch.isfinite(target).all():raise RuntimeError('target shape/value')
    gt,lt=torch.as_tensor(gi,device=device),torch.as_tensor(li,device=device)
    folds=range(4) if domain=='synthetic' else (-1,);accounts={}
    for fold in folds:
        foldfit,train=training_masks(rows,domain,fold,regime);ids=np.flatnonzero(train)
        torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);rng=np.random.default_rng(SEED)
        model=IdentityFeatureReader(arm,reader).to(device);opt=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.001)
        gs,ls,cm,cs=input_normalization(gb,lb,gi,li,context,train)
        gst=torch.as_tensor(gs[None,:,None,None],device=device);lst=torch.as_tensor(ls[None,None,None,:,None,None],device=device)
        ct=torch.as_tensor(np.clip((context-cm)/cs,-8,8),device=device)
        curve=[];start=time.perf_counter();step_digest=hashlib.sha256();initial_max=0.
        for step in range(2000):
            draw=rng.choice(ids,32,replace=True);step_digest.update(draw.astype('<i8').tobytes());ii=torch.as_tensor(draw,device=device)
            opt.zero_grad(set_to_none=True)
            raw=model((gb[gt[ii]]/gst).clamp(-8,8),(lb[lt[ii]]/lst).clamp(-8,8),ct[ii])
            predicted,_=decode_joints(raw,anchor[ii],beta[ii],reader,decoder,identity[ii])
            if step==0:
                if torch.count_nonzero(raw):raise AssertionError('initial action is not zero')
                initial_max=float((predicted-identity[ii]).abs().max().detach().cpu())
                if initial_max>1e-6:raise AssertionError('heads start from different physical identity')
            loss=joint_mse(predicted,target[ii])
            if not torch.isfinite(loss):raise ArithmeticError('nonfinite joint MSE')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ArithmeticError('nonfinite gradient')
            opt.step()
            if step%500==0 or step==1999:
                curve.append([step+1,float(loss.detach().cpu())]);print(json.dumps(dict(domain=domain,arm=arm,reader=reader,fold=fold,step=step+1,FIT_loss=curve[-1][1],SELECT_not_scored=True)),flush=True)
        if device!='cpu':torch.cuda.synchronize()
        elapsed=time.perf_counter()-start;prediction=[];raws=[];states=[];model.eval()
        with torch.inference_mode():
            for a in range(0,len(rows),32):
                s=slice(a,a+32);raw=model((gb[gt[s]]/gst).clamp(-8,8),(lb[lt[s]]/lst).clamp(-8,8),ct[s])
                predicted,decoded=decode_joints(raw,anchor[s],beta[s],reader,decoder,identity[s])
                prediction.append(predicted.float().cpu().numpy());raws.append(raw.float().cpu().numpy())
                if decoded is not None:states.append(decoded.commit_pose51().float().cpu().numpy())
        prediction=np.concatenate(prediction);raws=np.concatenate(raws)
        if not np.isfinite(prediction).all() or not np.isfinite(raws).all():raise ArithmeticError('nonfinite fixed prediction')
        folder=dest/f'fold{fold}';folder.mkdir()
        npz=dict(prediction_joints_m=prediction,anchor_joints_m=identity.cpu().numpy(),raw_output=raws,fit=train,foldfit=foldfit)
        if states:
            state=np.concatenate(states)
            if not np.isfinite(state).all() or state.shape!=(len(rows),51):raise ArithmeticError('invalid detached MANO state')
            npz['state51']=state
        np.savez_compressed(folder/'predictions.npz',**npz)
        torch.save(dict(schema='s43_field_identity_mano_head_v1',arm=arm,reader=reader,regime=regime,fold=fold,seed=SEED,steps=2000,
          model=model.state_dict(),optimizer=opt.state_dict(),global_scale=gs,local_scale=ls,context_mean=cm,context_std=cs,
          output_target_mean_added=False,per_axis_target_std=False,joint_loss_scale_m=.01,
          decoder_metadata=decoder.metadata(),network_source_sha256=sha(__file__),
          inference_input_contract=dict(global_shape=[40,180,240],local_shape=[8,90,5,7,7],context_width=70,
            global_scale='FITRMSfloor1 clip8',local_scale='FITRMSfloor1 clip8 after sampling',
            context='FITmean/std,std<1e-6→1,clip8;anchor51,beta10,Ksource9',
            features='loggrid40 once;staticcollapse before bothbranches;actualcellgrid sampling',
            empty_gate=False,action_clip=False,previous_pose_skip='decoder anchor + rightlocalaction'),
          source_freeze_sha256=sha(freeze)),folder/'head.pt')
        accounts[str(fold)]=dict(model.accounting(),steps=2000,batch=32,supervised_views=64000,train_views=len(ids),
          seed=SEED,draw_digest=step_digest.hexdigest(),FIT_curve=curve,optimizer_loop_seconds=elapsed,
          initial_identity_max_m=initial_max,device=device,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
          precision='FP32 noTF32 noautocast',global_dataset_cache_bytes=gb.numel()*4,local_dataset_cache_bytes=lb.numel()*4,
          same_fixed_joint_MSE=True,loss_scale_m=.01)
        del model,opt
        if device!='cpu':torch.cuda.empty_cache()
    artifacts={str(p.resolve()):sha(p) for fold in folds for p in (dest/f'fold{fold}').iterdir()}
    dump(dest/'predictions_complete.json',dict(domain=domain,arm=arm,reader=reader,regime=regime,folds=list(folds),accounts=accounts,
      artifacts=artifacts,source_freeze_sha256=sha(freeze),feature_complete_sha256=sha(features/'complete.json'),
      SELECT_metrics_not_computed=True))
    require_gate(freeze)


def metrics(pred,truth,identity,rows,labels,domain):
    error=pred-truth;ra=(pred-pred[:,:1])-(truth-truth[:,:1])
    arrays=dict(abs_mm=np.linalg.norm(error,axis=-1).mean(1)*1000,J0_mm=np.linalg.norm(ra,axis=-1).mean(1)*1000,
      root_mm=np.linalg.norm(error[:,0],axis=-1)*1000)
    if domain=='synthetic':
        ids=np.asarray([np.arange(1+4*labels[r['case_id']]['finger'],5+4*labels[r['case_id']]['finger']) for r in rows])
        arrays['active_finger_J0_mm']=np.take_along_axis(np.linalg.norm(ra,axis=-1),ids,axis=1).mean(1)*1000
    result=dict(n=len(rows),unique_cases=len({r['case_id'] for r in rows}),**{k:float(v.mean()) for k,v in arrays.items()})
    for key,out in (('recording','per_recording'),('fold','per_context')):
        result[out]={}
        for value in sorted({str(r.get(key)) for r in rows if r.get(key) is not None}):
            select=np.asarray([str(r.get(key))==value for r in rows]);result[out][value]=dict(n=int(select.sum()),**{k:float(v[select].mean()) for k,v in arrays.items()})
    if domain=='synthetic':
        result['per_background']={}
        for bg in (False,True):
            take=np.asarray([labels[r['case_id']]['moving_background']==bg for r in rows])
            result['per_background'][str(bg)]=dict(n=int(take.sum()),**{k:float(v[take].mean()) for k,v in arrays.items()})
    return result


def check_predictions(folder,feature,domain,reader,freeze,regime):
    c=json.loads((folder/'predictions_complete.json').read_text(encoding='utf-8'));expected=list(range(4)) if domain=='synthetic' else [-1]
    if c['regime']!=regime or c['folds']!=expected or c['source_freeze_sha256']!=sha(freeze) or c['feature_complete_sha256']!=sha(feature/'complete.json'):
        raise RuntimeError('fit identity/regime mismatch')
    for path,identity in c['artifacts'].items():
        if sha(path)!=identity:raise RuntimeError('fixed fitted artifact changed')
    for fold in expected:
        path=folder/f'fold{fold}/predictions.npz'
        if str(path.resolve()) not in c['artifacts']:raise RuntimeError('missing fixed prediction')
        with np.load(path,allow_pickle=False) as z:
            N=400 if domain=='synthetic' else 1440
            if z['prediction_joints_m'].shape!=(N,21,3) or not np.isfinite(z['prediction_joints_m']).all():raise RuntimeError('bad fixed joints')
            if reader=='mano51' and (z['state51'].shape!=(N,51) or not np.isfinite(z['state51']).all()):raise RuntimeError('invalid MANO state')


def reader_scores(folder,domain,rows,truth,labels):
    per_fold={};oof=np.zeros_like(truth);prior=None
    for fold in (range(4) if domain=='synthetic' else [-1]):
        with np.load(folder/f'fold{fold}/predictions.npz',allow_pickle=False) as z:
            pred=z['prediction_joints_m'];prior=z['anchor_joints_m'];foldfit=z['foldfit']
        oof[~foldfit]=pred[~foldfit];mm={}
        for split,mask in [('FIT',foldfit),('SELECT',~foldfit)]:
            for condition in P.CONDITIONS:
                chosen=mask&np.asarray([r['condition']==condition for r in rows]);rr=[r for r,k in zip(rows,chosen) if k]
                mm[split+'/'+condition]=metrics(pred[chosen],truth[chosen],prior[chosen],rr,labels,domain)
        per_fold[str(fold)]=mm
    pooled={};base={}
    for condition in P.CONDITIONS:
        chosen=np.asarray([r['condition']==condition and (domain=='synthetic' or r['partition']=='SELECT') for r in rows]);rr=[r for r,k in zip(rows,chosen) if k]
        pooled[condition]=metrics(oof[chosen],truth[chosen],prior[chosen],rr,labels,domain)
        base[condition]=metrics(prior[chosen],truth[chosen],prior[chosen],rr,labels,domain)
    return dict(per_fold=per_fold,held_out_pooled=pooled),base


def score(features_root,readers_root,dest,freeze,guards_root=None):
    require_gate(freeze);features_root,readers_root,dest=Path(features_root),Path(readers_root),Path(dest)
    guards_root=Path(guards_root) if guards_root else readers_root.parent/'readers_nominal_guard'
    if dest.exists():raise FileExistsError(dest)
    # All70original fits + four prefit guards precede every new task metric.
    for domain in ('synthetic','existing'):
        feature=features_root/domain;f=json.loads((feature/'complete.json').read_text(encoding='utf-8'))
        if f['rows_sha256']!=sha(feature/'rows.json') or f['labels_sha256']!=sha(feature/'labels.json'):raise RuntimeError('feature metadata changed')
        for arm in ARMS:
            for reader in READERS:check_predictions(readers_root/domain/arm/reader,feature,domain,reader,freeze,'all3')
    for arm in ('hand_box','static'):
        for reader in READERS:check_predictions(guards_root/'existing'/arm/reader,features_root/'existing','existing',reader,freeze,'nominal_guard')
    result={};identity_results={};guard_results={};dest.mkdir(parents=True)
    for domain in ('synthetic','existing'):
        feature=features_root/domain;allrows=json.loads((feature/'rows.json').read_text(encoding='utf-8'));labels=json.loads((feature/'labels.json').read_text(encoding='utf-8'));result[domain]={}
        for arm in ARMS:
            rows=[r for r in allrows if r['arm']==arm];truth=target_joints(rows,labels,domain);result[domain][arm]={}
            for reader in READERS:
                scores,base=reader_scores(readers_root/domain/arm/reader,domain,rows,truth,labels)
                result[domain][arm][reader]=scores
                if arm=='global_only' and reader=='joint63':identity_results[domain]=base
            if domain=='existing' and arm in ('hand_box','static'):
                guard_results[arm]={}
                for reader in READERS:guard_results[arm][reader]=reader_scores(guards_root/domain/arm/reader,domain,rows,truth,labels)[0]
    dump(dest/'results.json',dict(schema='s43_field_identity_mano_reader_results_v1',protocol_sha256=PROTOCOL_SHA,
      nominal_guard_addendum_sha256=ADDENDUM_SHA,source_freeze_sha256=sha(freeze),results=result,identity=identity_results,
      nominal_guard=guard_results,all74_fits_saved_before_scoring=True,supervision='same continuous21joint absoluteMSE /(.01m)^2',
      new_zgz_read=False,full_MANO_training=False,no_MPVPE_claim=True))
    require_gate(freeze)


def engineering(dest,device='cpu',batch=2):
    torch.set_num_threads(2);torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if device!='cpu':P.old.validate_device(device);torch.cuda.reset_peak_memory_stats()
    decoder=decoder_instance(device)
    anchor=torch.tensor([[0,0,.6]+[0.]*48]*batch,device=device);beta=torch.zeros(batch,10,device=device)
    g=torch.zeros(batch,40,180,240,device=device);local=torch.ones(batch,8,90,5,7,7,device=device);ctx=torch.zeros(batch,70,device=device)
    identity=decoder(anchor,beta,torch.zeros_like(anchor)).joints.detach();checks={}
    for reader in READERS:
        torch.manual_seed(SEED);model=IdentityFeatureReader('static',reader).to(device);raw=model(g,local,ctx)
        pred,_=decode_joints(raw,anchor,beta,reader,decoder,identity)
        assert not torch.count_nonzero(raw) and float((pred-identity).abs().max())<=1e-6
        joint_mse(pred,identity+torch.tensor([.001,0,0],device=device)).backward()
        assert model.head[-1].weight.grad.abs().sum()>0
        assert model.local_encoder[0].weight.grad.abs().sum()==0
        model.zero_grad(set_to_none=True)
        with torch.no_grad():model.head[-1].weight.fill_(.001)
        raw=model(g,local,ctx);pred,_=decode_joints(raw,anchor,beta,reader,decoder,identity)
        joint_mse(pred,identity+torch.tensor([.001,0,0],device=device)).backward()
        assert model.local_encoder[0].weight.grad.abs().sum()>0
        checks[reader]=model.accounting()
    dump(dest,dict(status='PASS',schema='s43_field_identity_reader_engineering_v1',probe_sha256=sha(__file__),
      decoder_sha256=sha(ROOT/'tools/s43_revisit/field_mano_decoder.py'),scientific_data_opened=False,new_Stage4_task_scores_read=False,
      same_zero_identity=True,zero_head_first_step_role_tested=True,nonzero_head_encoder_gradient_tested=True,readers=checks,
      device=device,batch=batch,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
      peak_allocated_bytes=0 if device=='cpu' else torch.cuda.max_memory_allocated(),
      peak_reserved_bytes=0 if device=='cpu' else torch.cuda.max_memory_reserved()))
    dump(Path(dest).parent/'implementation_extension.json',dict(schema='s43_field_identity_pre_task_implementation_v1',
      protocol_sha256=PROTOCOL_SHA,probe_sha256=sha(__file__),decoder_sha256=sha(ROOT/'tools/s43_revisit/field_mano_decoder.py'),
      input_normalization='exact Stage3 actual rule: global/local FITRMSfloor1; context FITmean/std,std<1e-6→1,clip8',
      clarification='protocol contextRMS wording was imprecise; root confirmed exact Stage3 rule before anyStage4 score',
      output_normalization='none: bothfixed10mm jointMSE; no targetmean/std',arms=list(ARMS),readers=list(READERS),
      nominal_guard_addendum_sha256=ADDENDUM_SHA,original_fits=70,nominal_guard_fits=4,
      fits=74,steps_perfit=2000,batch=32,total_updates=148000,supervised_views=4736000,
      scope='identity-init51Dvs63D atomicdecoder study, not fullS43 or beta0deployment, no nulltraining',new_Stage4_task_scores_read=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('freeze');a.add_argument('--out',type=Path,required=True);a.add_argument('--engineering',nargs='+',required=True);a.add_argument('--commands',nargs='+',required=True)
    a=sub.add_parser('prepare');a.add_argument('--domain',choices=('synthetic','existing'),required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--source-freeze',type=Path,required=True)
    a=sub.add_parser('engineering');a.add_argument('--out',type=Path,required=True);a.add_argument('--device',default='cpu');a.add_argument('--batch',type=int,default=2)
    a=sub.add_parser('fit');a.add_argument('--domain',choices=('synthetic','existing'),required=True);a.add_argument('--arm',choices=ARMS,required=True);a.add_argument('--reader',choices=READERS,required=True);a.add_argument('--features',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.add_argument('--source-freeze',type=Path,required=True);a.add_argument('--device',default='cpu');a.add_argument('--regime',choices=('all3','nominal_guard'),default='all3')
    a=sub.add_parser('score');a.add_argument('--features-root',type=Path,required=True);a.add_argument('--readers-root',type=Path,required=True);a.add_argument('--guards-root',type=Path);a.add_argument('--out',type=Path,required=True);a.add_argument('--source-freeze',type=Path,required=True)
    a=p.parse_args()
    if a.command=='freeze':sourcefreeze(a.out,a.engineering,a.commands)
    elif a.command=='prepare':prepare(a.domain,a.out,a.source_freeze)
    elif a.command=='engineering':engineering(a.out,a.device,a.batch)
    elif a.command=='score':score(a.features_root,a.readers_root,a.out,a.source_freeze,a.guards_root)
    else:fit(a.domain,a.arm,a.reader,a.features,a.out,a.source_freeze,a.device,a.regime)


if __name__=='__main__':main()
