"""Pre-registered nominal-only R1; old Stage2/3 files remain read-only."""
from __future__ import annotations
import argparse,json,os,time,types
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from . import field_query_probe as ref
ROOT=ref.ROOT;OUT=ref.OUT;OLD=OUT/'field_atomic_v1';PROTOCOL=OUT/'stage3_reader_R1_protocol.json'
PROTOCOL_SHA='4fbbf4df4b689f9737ec7fa27691b670eca3f2c17ff9b7c82cb1db3932750705'
ARMS=('global_only','hand_box','static','anatomical','anatomical_temporal_collapse','quota_anatomical','context_only')
SETTINGS=('nominal_fixed_all3_statistics','nominal_own_statistics')
sha=ref.sha;dump=ref.dump


def dependencies():return dict(ref.dependencies(),**{str(Path(__file__).resolve()):sha(__file__)})
def input_rows(arm):
    rows=json.loads((OLD/'features/existing/rows.json').read_text());source='global_only' if arm=='context_only' else arm
    return [dict(r,arm=arm) for r in rows if r['arm']==source]

def gate(path):
    x=json.loads(Path(path).read_text())
    if x.get('schema')!='s43_field_query_reader_R1_actual_freeze_v1' or x['dependencies']!=dependencies() or sha(PROTOCOL)!=PROTOCOL_SHA:raise RuntimeError('wrong R1 actualfreeze')
    for file,identity in {**x['sources'],**x['engineering']}.items():
        if sha(file)!=identity:raise RuntimeError('R1 source changed: '+file)
    return x


def freeze(path,receipts,commands):
    if Path(path).exists():raise FileExistsError(path)
    if sha(PROTOCOL)!=PROTOCOL_SHA:raise RuntimeError('wrong R1 protocol')
    sources={PROTOCOL,OUT/'reader_r1_engineering/implementation_extension.json'}
    for file,identity in json.loads(PROTOCOL.read_text())['prior_identity'].items():
        if sha(ROOT/file)!=identity:raise RuntimeError('preregistered prior artifact changed')
        sources.add(ROOT/file)
    for arm in ARMS:
        source='global_only' if arm=='context_only' else arm
        sources.update((OLD/'readers/existing'/source/'fold-1'/name) for name in ('cnn.pt','rbf.npz'))
        for row in input_rows(arm):sources.update((Path(row['source_file']),Path(row['coordinate_file'])))
    extension=json.loads((OUT/'reader_r1_engineering/implementation_extension.json').read_text())
    for name,identity in extension['commands_sha256'].items():
        if sha(name)!=identity:raise RuntimeError('R1 command changed')
        sources.add(Path(name))
    engineering={}
    for file in receipts:
        x=json.loads(Path(file).read_text())
        if x.get('status')!='PASS':raise RuntimeError('R1 engineering incomplete')
        engineering[str(Path(file).resolve())]=sha(file)
    dump(path,dict(schema='s43_field_query_reader_R1_actual_freeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=PROTOCOL_SHA,dependencies=dependencies(),sources={str(p.resolve()):sha(p) for p in sorted(sources)},engineering=engineering,commands=commands,R1_task_scores_seen=False))


def make_reader(arm):
    import torch
    model=ref.make_cnn(63,arm not in ('global_only','context_only'))
    if arm=='context_only':
        def forward(self,g,l,c):return self.head(torch.cat((torch.zeros((len(c),6528),device=c.device,dtype=c.dtype),c),1))
        model.forward=types.MethodType(forward,model)
    return model


def loaded_cnn_stats(arm):
    import torch
    source='global_only' if arm=='context_only' else arm
    p=OLD/'readers/existing'/source/'fold-1/cnn.pt';x=torch.load(p,map_location='cpu',weights_only=False)
    return {k:x[k] for k in ('global_scale','local_scale','context_mean','context_std','target_mean','target_std')}


def fit_kernel(x,c,y,stats):
    head=ref.LowCapacityRBF()
    for k,v in stats.items():setattr(head,k,v)
    head.library=head.transform(x,c);d=ref.old.RBFReader.distance(head.library,head.library)
    K=np.exp(-d/head.bandwidth);K.flat[::len(K)+1]+=.01;head.coef=np.linalg.solve(K,y.astype(float));head.kernel_bytes=K.nbytes+d.nbytes
    return head


def fixed_rbf_stats(arm,x,c,all3):
    source='global_only' if arm=='context_only' else arm
    with np.load(OLD/'readers/existing'/source/'fold-1/rbf.npz',allow_pickle=False) as z:
        out={k:np.array(z[k]) for k in ('scale','cm','cs','context_scale','bandwidth')}
    if arm=='context_only':
        # No old all3 context-only head exists: derive its input-only statistic,
        # using common all3 context transform, never a generic-image bandwidth.
        out['scale']=np.ones(x.shape[1]);head=ref.LowCapacityRBF()
        for k,v in out.items():setattr(head,k,v)
        lib=head.transform(x[all3],c[all3]);d=ref.old.RBFReader.distance(lib,lib)
        v=d[np.triu_indices(len(d),1)];v=v[v>1e-12];out['bandwidth']=max(float(np.median(v)) if len(v) else 1.,1e-6)
    return out


def fit(arm,setting,dest,freeze_path,device):
    gate(freeze_path);ref.old.validate_device(device)
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(8);torch.manual_seed(3407);torch.cuda.manual_seed_all(3407)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    dest=Path(dest)
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True);rows=input_rows(arm);labels=json.loads((OLD/'features/existing/labels.json').read_text())
    gb,lb,gi,li=ref.load_banks(rows,device)
    if arm=='context_only':gb.zero_();lb.zero_()
    context=np.asarray([r['context'] for r in rows],np.float32);y=ref.target_array(rows,labels,'existing')
    foldfit=np.asarray([r['partition']=='FIT7' for r in rows]);all3=foldfit&np.asarray([r['training_view'] for r in rows]);train=foldfit&np.asarray([r['condition']=='none' for r in rows]);ids=np.flatnonzero(train)
    assert len(ids)==224 and int(all3.sum())==672
    x=np.concatenate((gb.reshape(len(gb),40,6,30,8,30).mean((3,5)).reshape(len(gb),1920).cpu().numpy()[gi],lb.mean((-1,-2)).reshape(len(lb),3600).cpu().numpy()[li]),1)
    stats=loaded_cnn_stats(arm)
    if setting==SETTINGS[1]:
        gs=np.zeros(40);ls=np.zeros(5);gw=np.bincount(gi[train],minlength=len(gb));lw=np.bincount(li[train],minlength=len(lb))
        for i,n in enumerate(gw):
            if n:gs+=n*gb[i].double().square().mean((1,2)).cpu().numpy()
        for i,n in enumerate(lw):
            if n:ls+=n*lb[i].double().square().mean((0,1,3,4)).cpu().numpy()
        cs=context[train].std(0);cs[cs<1e-6]=1
        stats=dict(global_scale=np.maximum(np.sqrt(gs/224),1).astype(np.float32),local_scale=np.maximum(np.sqrt(ls/224),1).astype(np.float32),context_mean=context[train].mean(0),context_std=cs,target_mean=y[train].mean(0),target_std=np.maximum(y[train].std(0),1))
    gs=torch.as_tensor(stats['global_scale'][None,:,None,None],device=device);ls=torch.as_tensor(stats['local_scale'][None,None,None,:,None,None],device=device)
    ct=torch.as_tensor(np.clip((context-stats['context_mean'])/stats['context_std'],-8,8),device=device)
    yt=torch.as_tensor(y,device=device);ym=torch.as_tensor(stats['target_mean'],device=device);ys=torch.as_tensor(stats['target_std'],device=device)
    git=torch.as_tensor(gi,device=device);lit=torch.as_tensor(li,device=device)
    # Seeding after all deterministic input/stat preparation exactly matches
    # Stage3 construction and pairs both R1 settings' initial weights/draws.
    torch.manual_seed(3407);torch.cuda.manual_seed_all(3407);model=make_reader(arm).to(device);opt=torch.optim.Adam(model.parameters(),lr=.001);rng=np.random.default_rng(3407);curve=[];begin=time.perf_counter()
    for step in range(2000):
        ii=torch.as_tensor(rng.choice(ids,32,replace=True),device=device);opt.zero_grad(set_to_none=True)
        pred=model((gb[git[ii]]/gs).clamp(-8,8),(lb[lit[ii]]/ls).clamp(-8,8),ct[ii]);loss=torch.nn.functional.mse_loss(pred,(yt[ii]-ym)/ys)
        if not bool(torch.isfinite(loss).item()):raise RuntimeError('nonfinite R1 loss')
        loss.backward();opt.step()
        if step%500==0 or step==1999:
            curve.append([step+1,float(loss.detach().cpu())]);print(json.dumps(dict(arm=arm,setting=setting,step=step+1,FIT_loss=curve[-1][1],SELECT_not_scored=True)),flush=True)
    if device!='cpu':torch.cuda.synchronize()
    seconds=time.perf_counter()-begin;model.eval();outputs=[]
    with torch.inference_mode():
        for a in range(0,len(rows),32):
            i=slice(a,a+32);pp=model((gb[git[i]]/gs).clamp(-8,8),(lb[lit[i]]/ls).clamp(-8,8),ct[i])*ys+ym;outputs.append(pp.cpu().numpy())
    pred=np.concatenate(outputs)
    if not np.isfinite(pred).all():raise RuntimeError('nonfinite R1 prediction')
    torch.save(dict(model=model.state_dict(),stats=stats,steps=2000,seed=3407,arm=arm,setting=setting),dest/'cnn.pt');np.savez_compressed(dest/'cnn_predictions.npz',prediction=pred,fit=train,foldfit=foldfit)
    begin=time.perf_counter()
    with threadpool_limits(8):
        if setting==SETTINGS[0]:head=fit_kernel(x[train],context[train],y[train],fixed_rbf_stats(arm,x,context,all3))
        else:head=ref.LowCapacityRBF().fit(x[train],context[train],y[train])
        rpred=head.predict(x,context)
    if not np.isfinite(rpred).all():raise RuntimeError('nonfinite R1 RBF prediction')
    head.save(dest/'rbf.npz');np.savez_compressed(dest/'rbf_predictions.npz',prediction=rpred,fit=train,foldfit=foldfit)
    artifacts={str((dest/name).resolve()):sha(dest/name) for name in ('cnn.pt','rbf.npz','cnn_predictions.npz','rbf_predictions.npz')}
    total=sum(v.numel() for v in model.parameters());active=33727 if arm=='context_only' else 136087 if arm=='global_only' else total
    dump(dest/'predictions_complete.json',dict(arm=arm,setting=setting,source_freeze_sha256=sha(freeze_path),artifacts=artifacts,steps=2000,batch=32,seed=3407,nominal_FIT=224,all3_FIT_for_fixed_statistics=672,allocated_parameters=total,structurally_active_parameters=active,optimizer_loop_seconds=seconds,FIT_curve=curve,rbf=dict(head.accounting(),fit_prediction_seconds=time.perf_counter()-begin),device=device,CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),precision='FP32 noTF32 noautocast',SELECT_scores_not_computed=True))
    gate(freeze_path)


def score(readers,dest,freeze_path):
    gate(freeze_path);readers=Path(readers);dest=Path(dest)
    if dest.exists():raise FileExistsError(dest)
    for arm in ARMS:
      for setting in SETTINGS:
        marker=json.loads((readers/arm/setting/'predictions_complete.json').read_text())
        if marker['source_freeze_sha256']!=sha(freeze_path):raise RuntimeError('R1 markerfreeze mismatch')
        for file,h in marker['artifacts'].items():
            if sha(file)!=h:raise RuntimeError('fixed R1 artifact changed')
        for kind in ('cnn','rbf'):
            with np.load(readers/arm/setting/f'{kind}_predictions.npz',allow_pickle=False) as z:
                if z['prediction'].shape!=(1440,63) or not np.isfinite(z['prediction']).all() or int(z['fit'].sum())!=224:raise RuntimeError('bad R1 fixed output')
    labels=json.loads((OLD/'features/existing/labels.json').read_text());results={}
    for arm in ARMS:
      rows=input_rows(arm);results[arm]={}
      for setting in SETTINGS:
        results[arm][setting]={}
        for kind in ('cnn','rbf'):
            with np.load(readers/arm/setting/f'{kind}_predictions.npz',allow_pickle=False) as z:pred=z['prediction'];foldfit=z['foldfit']
            metrics={}
            for split,mask in [('FIT',foldfit),('SELECT',~foldfit)]:
              for condition in ref.CONDITIONS:
                choose=mask&np.asarray([r['condition']==condition for r in rows]);rr=[r for r,k in zip(rows,choose) if k]
                metrics[f'{split}/{condition}']=ref._query_metrics(pred[choose],rr,labels,'existing')
            results[arm][setting][kind]=metrics
    dest.mkdir(parents=True);dump(dest/'results.json',dict(schema='s43_reader_R1_results_v1',results=results,protocol_sha256=PROTOCOL_SHA,source_freeze_sha256=sha(freeze_path),all14_predictions_saved_before_metrics=True,old_gates_unchanged=True,no_automatic_full_training=True));gate(freeze_path)


def engineering(dest):
    import torch
    torch.set_num_threads(2);torch.manual_seed(3407);m=make_reader('context_only');g=torch.randn(2,40,180,240);l=torch.randn(2,8,90,5,7,7);c=torch.randn(2,70)
    a=m(g,l,c);b=m(torch.zeros_like(g),torch.zeros_like(l),c);torch.testing.assert_close(a,b,rtol=0,atol=0);a.square().mean().backward()
    assert all(t.grad is None for t in m.global_encoder.parameters()) and all(t.grad is None for t in m.local_encoder.parameters())
    rng=np.random.default_rng(1);x=rng.random((8,5520)).astype(np.float32);ctx=rng.random((8,70)).astype(np.float32);y=rng.random((8,63)).astype(np.float32)
    initial=ref.LowCapacityRBF().fit(x,ctx,y);stats={k:getattr(initial,k) for k in ('scale','cm','cs','context_scale','bandwidth')}
    rebuilt=fit_kernel(x,ctx,y,stats);np.testing.assert_array_equal(initial.predict(x,ctx),rebuilt.predict(x,ctx))
    dump(dest,dict(status='PASS',source_sha256=sha(__file__),scientific_data_opened=False,R1_task_scores_seen=False,context_has_no_measurement_bias_leak=True,fixed_RBF_transform_solve_matches_original=True,parameters=sum(v.numel() for v in m.parameters()),steps_formal=2000))


def main():
    a=argparse.ArgumentParser();s=a.add_subparsers(dest='cmd',required=True)
    q=s.add_parser('engineering');q.add_argument('--out',required=True)
    q=s.add_parser('freeze');q.add_argument('--out',required=True);q.add_argument('--receipts',nargs='+',required=True);q.add_argument('--commands',nargs='+',required=True)
    q=s.add_parser('fit');q.add_argument('--arm',choices=ARMS,required=True);q.add_argument('--setting',choices=SETTINGS,required=True);q.add_argument('--out',required=True);q.add_argument('--freeze',required=True);q.add_argument('--device',default='cpu')
    q=s.add_parser('score');q.add_argument('--readers',required=True);q.add_argument('--out',required=True);q.add_argument('--freeze',required=True)
    args=a.parse_args()
    if args.cmd=='engineering':engineering(args.out)
    elif args.cmd=='freeze':freeze(args.out,args.receipts,args.commands)
    elif args.cmd=='fit':fit(args.arm,args.setting,args.out,args.freeze,args.device)
    else:score(args.readers,args.out,args.freeze)


if __name__=='__main__':main()
