"""Post-score finite synthetic collision/visibility/reader receipt audit.

No fitting, checkpoint selection, gate change or task input construction. GT
and generator conditions here only group/score already-sealed observations.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/s43_revisit_20261028'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def summary(x):
    a=np.asarray(x,float)
    if not len(a):return dict(n=0)
    return dict(n=len(a),min=float(a.min()),q25=float(np.quantile(a,.25)),median=float(np.median(a)),q75=float(np.quantile(a,.75)),max=float(a.max()))


def sparse_counts(events,T=50,r=1,bins=50):
    e=events[(events[:,2]>=0)&(events[:,2]<T)]
    key=(((e[:,2]*bins//T*(180//r)+e[:,1]//r)*(240//r)+e[:,0]//r)*2+e[:,3])
    keys,n=np.unique(key,return_counts=True)
    return keys,n.astype(np.int64)


def vector_hash(view):
    h=hashlib.sha256()
    for v in view:h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


def distance(a,b):
    ka,va=a;kb,vb=b;k=np.union1d(ka,kb)
    x=np.zeros(len(k),np.int64);y=x.copy()
    x[np.searchsorted(k,ka)]=va;y[np.searchsorted(k,kb)]=vb;d=x-y
    l1=int(np.abs(d).sum());den=int(x.sum()+y.sum())
    return dict(exact_equal=bool(l1==0),L1_count=l1,L2_count=float(np.sqrt(np.dot(d.astype(float),d))),
        relative_L1_to_combined_mass=float(l1/den) if den else 0.,
        differing_coordinates=int(np.count_nonzero(d)),mass_a=int(x.sum()),mass_b=int(y.sum()))


def grid_from_sparse(view,r):
    key,count=view;fine=np.zeros(8*(180//r)*(240//r)*2,np.uint64);fine[key]=count
    fine=fine.reshape(8,180//r,240//r,2)
    d=fine.astype(np.float32)/(r*r);d=d.repeat(r,1).repeat(r,2)
    c=4//r;coarse=fine.reshape(8,45,c,60,c,2).sum((2,4)).astype(np.float32).repeat(4,1).repeat(4,2)
    return np.concatenate((np.log1p(d).transpose(0,3,1,2).reshape(16,180,240),np.ones((8,180,240),np.float32),
                           np.log1p(coarse).transpose(0,3,1,2).reshape(16,180,240))).astype(np.float32)


def pair_records(cases):
    groups={}
    for c in cases:groups.setdefault(tuple(c['condition'][k] for k in ('root_y_rad','root_dx_m','moving_background')),[]).append(c)
    pairs=[]
    for context,group in groups.items():
        assert len(group)==10 and len({c['class'] for c in group})==10
        for a,b in itertools.combinations(group,2):
            p=dict(case_a=a['id'],case_b=b['id'],context=list(context),class_a=a['class'],class_b=b['class'],
                opposite_same_finger=bool(a['finger']==b['finger'] and a['sign']==-b['sign']),
                target_J0_separation_mm=float(np.linalg.norm((a['joints']-a['joints'][:1])-(b['joints']-b['joints'][:1]),axis=-1).mean()*1000),views={})
            for name in ('raw50','raw100','fullfine8','field2_8'):
                p['views'][name]=distance(a['views'][name],b['views'][name])
            for name in ('fullfine','field2'):
                x=a['pooled'][name]-b['pooled'][name]
                p['views']['RBF_pool_'+name]=dict(exact_equal=bool(np.all(x==0)),L2=float(np.linalg.norm(x.astype(float))))
            pairs.append(p)
    return groups,pairs


def reader_audit(readers,rows,labels):
    report=json.loads((readers/'results.json').read_text());N=len(rows)
    y=np.asarray([labels[r['case_id']]['finger']*2+int(labels[r['case_id']]['sign']<0) for r in rows]);fold=np.asarray([r['fold'] for r in rows])
    arm=np.asarray([r['arm'] for r in rows]);error=np.asarray([r['selection_error'] for r in rows]);T=np.asarray([r['cutoff_ms'] for r in rows])
    from tools.s43_revisit.witness_geometry import motion_endpoints,fk_joints,FINGER_JOINTS
    tables={}
    for cid,label in labels.items():
        public=np.asarray(label['public_anchor']);beta=np.asarray(label['beta']);key=(public.tobytes(),beta.tobytes())
        if key not in tables:tables[key]=fk_joints(motion_endpoints(public,beta)[0].reshape(10,51),beta)
    candidate=np.asarray([tables[(np.asarray(labels[r['case_id']]['public_anchor']).tobytes(),np.asarray(labels[r['case_id']]['beta']).tobytes())] for r in rows])
    truth=np.asarray([labels[r['case_id']]['target_joints'][r['cutoff_ms']] for r in rows]);fingers=FINGER_JOINTS[y//2]
    def risks(pred):
        P=candidate[np.arange(N),pred.argmax(1)];relative=(P-P[:,:1])-(truth-truth[:,:1]);d=np.linalg.norm(relative,axis=-1)*1000
        return dict(mode_accuracy=(pred.argmax(1)==y).astype(float),J0_mm=d.mean(1),
            active_finger_J0_mm=d[np.arange(N)[:,None],fingers].mean(1),
            abs_mm=np.linalg.norm(P-truth,axis=-1).mean(1)*1000,root_mm=np.linalg.norm(P[:,0]-truth[:,0],axis=-1)*1000)
    fold_records={};max_metric_error=0.;max_risk_error=0.;aggregated={};all_masks=True;all_oof=True
    for reader in ('cnn','rbf'):
        oof=np.zeros((N,10),np.float32);seen=np.zeros(N,int)
        for f in range(4):
            with np.load(readers/f'fold{f}/{reader}_predictions.npz') as z:
                pred=z['prediction'];fit=z['fit'];select=z['select']
            masks=np.array_equal(fit,fold!=f) and np.array_equal(select,fold==f);all_masks&=masks
            oof[select]=pred[select];seen[select]+=1
            account=report['results'][f'{reader}/fold{f}']['accounting']
            risk=risks(pred)
            values={}
            for a in np.unique(arm):
                for split,mask in [('FIT',fit),('SELECT',select)]:
                    ids=mask&(arm==a)&(error=='none')&(T==50);acc=float(np.mean(pred[ids].argmax(1)==y[ids]))
                    stored=report['results'][f'{reader}/fold{f}']['metrics'][f'{split}/{a}/none/t50']['mode_accuracy']
                    max_metric_error=max(max_metric_error,abs(acc-stored));values[f'{split}/{a}']=dict(n=int(ids.sum()),accuracy=acc)
            for split,mask in [('FIT',fit),('SELECT',select)]:
                for a in np.unique(arm):
                    for e in np.unique(error):
                        ids=mask&(arm==a)&(error==e)&(T==50)
                        if not ids.any():continue
                        stored=report['results'][f'{reader}/fold{f}']['metrics'][f'{split}/{a}/{e}/t50']
                        for metric,v in risk.items():max_risk_error=max(max_risk_error,abs(float(v[ids].mean())-stored[metric]))
            ckpt={}
            if reader=='cnn':
                import torch
                state=torch.load(readers/f'fold{f}/cnn.pt',map_location='cpu',weights_only=False)
                ckpt=dict(steps=state['steps'],seed=state['seed'],precision=state['precision'],device=state['device'],
                    tensor_parameters=int(sum(v.numel() for v in state['model'].values())),dtypes=sorted({str(v.dtype) for v in state['model'].values()}))
                assert ckpt['steps']==account['steps']==2000 and ckpt['seed']==3407
                assert account['curve'][0][0]==1 and account['curve'][-1][0]==2000
                assert ckpt['tensor_parameters']==account['parameters']
            else:
                with np.load(readers/f'fold{f}/rbf.npz') as z:
                    ckpt=dict(library_rows=len(z['library']),coef_shape=list(z['coef'].shape),bandwidth=float(z['bandwidth']))
                assert ckpt['library_rows']==int(fit.sum())==account['library_rows']
            fold_records[f'{reader}/fold{f}']=dict(fit_rows=int(fit.sum()),select_rows=int(select.sum()),
                fit_cases=len({rows[i]['case_id'] for i in np.flatnonzero(fit)}),select_cases=len({rows[i]['case_id'] for i in np.flatnonzero(select)}),
                masks_correct=masks,checkpoint=ckpt,accounting=account,independent_mode_accuracies=values)
        with np.load(readers/'out_of_fold_predictions.npz') as z:saved=z[reader]
        equal=np.array_equal(oof,saved);all_oof&=equal and np.all(seen==1)
        risk=risks(oof)
        aggregate={}
        for a in np.unique(arm):
            ids=(arm==a)&(error=='none')&(T==50);acc=float(np.mean(oof[ids].argmax(1)==y[ids]))
            stored=report['out_of_fold'][reader][f'SELECT/{a}/none/t50']['mode_accuracy']
            max_metric_error=max(max_metric_error,abs(acc-stored));aggregate[a]=dict(n=int(ids.sum()),accuracy=acc,
                by_fold={str(f):float(np.mean(oof[ids&(fold==f)].argmax(1)==y[ids&(fold==f)])) for f in range(4)})
        for a in np.unique(arm):
            for e in np.unique(error):
                ids=(arm==a)&(error==e)&(T==50)
                if not ids.any():continue
                stored=report['out_of_fold'][reader][f'SELECT/{a}/{e}/t50']
                for metric,v in risk.items():max_risk_error=max(max_risk_error,abs(float(v[ids].mean())-stored[metric]))
        aggregated[reader]=aggregate
    return dict(rows=N,cases=len(labels),all_fold_masks_correct=bool(all_masks),each_row_OOF_once_and_saved_logits_exact=bool(all_oof),
        maximum_independent_mode_accuracy_difference=max_metric_error,maximum_independent_all_FK_metric_difference=max_risk_error,folds=fold_records,out_of_fold=aggregated,
        limitation='Checkpoint steps and curve endpoints corroborate saved2000-step metadata/source loop, not an independent audit of every unlogged optimizer update; shared model trained over30views/case, not independent2400examples.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=BASE/'witness_posthoc/synthetic_diagnosis_v2');args=p.parse_args()
    out=args.out
    if out.exists():raise FileExistsError('preserve existing posthoc output')
    out.mkdir(parents=True)
    sources=BASE/'witness_synthetic_v1';features=BASE/'witness_atomic_v1/synthetic_features';readers=BASE/'witness_atomic_v1/synthetic_readers'
    rows=json.loads((features/'rows.json').read_text());labels=json.loads((features/'labels.json').read_text())
    manifest=json.loads((sources/'manifest.json').read_text());identities={str(p):sha(p) for p in [Path(__file__),sources/'manifest.json',features/'rows.json',features/'labels.json',readers/'results.json',readers/'out_of_fold_predictions.npz',ROOT/'tools/s43_revisit/witness_geometry.py',ROOT/'tools/s43_revisit/witness_probe.py']}
    frozen=json.loads((sources/'sourcefreeze.json').read_text())
    for name,value in frozen['dependencies'].items():
        assert sha(ROOT/name)==value;identities[str(ROOT/name)]=value
    for item in manifest['cases']:
        for name in ('observations','targets','manifest'):identities[str(Path(item[name]))]=sha(item[name])
    for f in range(4):
        for name in ('cnn_predictions.npz','rbf_predictions.npz','cnn.pt','rbf.npz'):
            path=readers/f'fold{f}'/name;identities[str(path)]=sha(path)
    lookup={(r['case_id'],r['arm']):r for r in rows if r['selection_error']=='none' and r['cutoff_ms']==50}
    import torch
    torch.set_num_threads(1)
    from tools.s43_rel.mechanisms import full_render
    cases=[];root_masks={};grid_checks=[];opposite_grid=[]
    for item in manifest['cases']:
        c=json.loads(Path(item['manifest']).read_text())['condition'];cid=item['id']
        assert sha(item['observations'])==item['observation_sha256'] and sha(item['targets'])==item['target_sha256']
        with np.load(item['observations']) as z:events=z['events'];K=z['K_source'];beta=z['beta'];public=z['public_initial51']
        with np.load(item['targets']) as z:states=z['target51'];joints=z['joints_m'];finger=int(z['finger']);sign=int(z['sign'])
        assert finger==labels[cid]['finger'] and sign==labels[cid]['sign']
        views=dict(raw50=sparse_counts(events),raw100=sparse_counts(events,100,bins=100),
                   fullfine8=sparse_counts(events,bins=8),field2_8=sparse_counts(events,r=2,bins=8))
        pooled={};grid_hash={}
        for arm,r in [('fullfine',1),('field2',2)]:
            row=lookup[cid,arm];assert sha(row['feature_file'])==row['feature_sha256']
            with np.load(row['feature_file']) as z:grid=z['grid']
            expect=grid_from_sparse(views['fullfine8' if r==1 else 'field2_8'],r)
            grid_checks.append(dict(case=cid,arm=arm,exact=np.array_equal(grid,expect),max_abs=float(np.max(np.abs(grid-expect)))))
            pooled[arm]=grid.reshape(40,6,30,8,30).mean((2,4))
            grid_hash[arm]=hashlib.sha256(grid.tobytes()).hexdigest()
        mask=full_render(states,beta,K).astype(bool)
        key=(c['root_y_rad'],c['root_dx_m'])
        if key not in root_masks:
            root_only=states.copy();root_only[:,6:]=public[6:]
            root_masks[key]=full_render(root_only,beta,K).astype(bool)
        changes=np.count_nonzero(mask[1:51]!=mask[:50],axis=(1,2));finger_delta=mask[:51]!=root_masks[key][:51]
        item=dict(id=cid,condition=c,finger=finger,sign=sign,**{'class':finger*2+int(sign<0)},joints=joints[50],views=views,
            pooled=pooled,grid_hash=grid_hash,view_hash={k:vector_hash(v) for k,v in views.items()},
            mass50=int(views['raw50'][1].sum()),mass100=len(events),
            visible=dict(terminal_xor_from_initial_pixels=int(np.count_nonzero(mask[50]!=mask[0])),
                ever_changed_from_initial_pixels=int(np.any(mask[:51]!=mask[0],axis=0).sum()),consecutive_pixel_transitions=int(changes.sum()),
                first_visible_interval_ms=int(np.flatnonzero(changes)[0]) if changes.any() else None,
                finger_given_root_ever_different_pixels=int(np.any(finger_delta,axis=0).sum()),finger_given_root_terminal_different_pixels=int(finger_delta[-1].sum())))
        cases.append(item)
    groups,pairs=pair_records(cases)
    # Saved log-grid differences only for40opposite pairs; canonical integer fields cover all360pairs.
    byid={c['id']:c for c in cases}
    for pair in pairs:
        if not pair['opposite_same_finger']:continue
        result=dict(case_a=pair['case_a'],case_b=pair['case_b'],views={})
        for arm in ('fullfine','field2'):
            with np.load(lookup[pair['case_a'],arm]['feature_file']) as z:a=z['grid']
            with np.load(lookup[pair['case_b'],arm]['feature_file']) as z:b=z['grid']
            d=a.astype(float)-b
            result['views'][arm]=dict(L1_grid40=float(np.abs(d).sum()),L2_grid40=float(np.linalg.norm(d)),exact_equal=bool(not np.any(d)))
        opposite_grid.append(result)
    cross=[]
    for a,b in itertools.combinations(cases,2):
        if a['class']!=b['class'] or a['condition']['moving_background']!=b['condition']['moving_background']:continue
        if a['condition']['root_y_rad']==b['condition']['root_y_rad'] and a['condition']['root_dx_m']==b['condition']['root_dx_m']:continue
        cross.append(dict(case_a=a['id'],case_b=b['id'],views={k:distance(a['views'][k],b['views'][k]) for k in ('raw50','fullfine8','field2_8')}))
    group_summary=[]
    for context,group in groups.items():
        pp=[p for p in pairs if p['context']==list(context)]
        group_summary.append(dict(context=list(context),n=10,zero_events_cases=[c['id'] for c in group if c['mass50']==0],
            events50=summary([c['mass50'] for c in group]),
            raw_finger_visibility_pixels=summary([c['visible']['finger_given_root_ever_different_pixels'] for c in group]),
            views={view:dict(exact_colliding_pairs=sum(p['views'][view]['exact_equal'] for p in pp),
                relative_L1=summary([p['views'][view]['relative_L1_to_combined_mass'] for p in pp]),
                absolute_L1=summary([p['views'][view]['L1_count'] for p in pp]),
                opposite_L1=summary([p['views'][view]['L1_count'] for p in pp if p['opposite_same_finger']])) for view in ('raw50','raw100','fullfine8','field2_8')}))
    audit=reader_audit(readers,rows,labels)
    result=dict(schema='s43_witness_synthetic_posthoc_diagnosis_v1',posthoc=True,new_gate=False,known_before='B CNN OOF21.25%,fullfine22.5%,RBF near chance and FIT fullfine86-100% were read before this diagnosis',
        source_identities=identities,primary_T_ms=50,secondary_raw100='whole source only; not used to select a new reader/query or repair gate',
        groups=group_summary,pairs=pairs,opposite_saved_grid_norms=opposite_grid,cross_context_same_mode=cross,
        cross_context_summary={view:dict(relative_L1=summary([p['views'][view]['relative_L1_to_combined_mass'] for p in cross]),
            L1_count=summary([p['views'][view]['L1_count'] for p in cross])) for view in ('raw50','fullfine8','field2_8')},
        cases=[{k:v for k,v in c.items() if k not in ('views','joints','pooled')} for c in cases],
        grid_reconstruction_checks=grid_checks,all_saved_full_grids_reconstructed_exact=all(x['exact'] for x in grid_checks),
        reader_audit=audit,limits=[
            'Exact uniqueness only rules out exact collisions in this finite noiseless controlled family; not robust sufficiency or a bound on real-event Bayes error.',
            'Known synthetic public boundary/orientation is a conditional generator input, not GT initialization available in real deployment; no raw-information upper bound is claimed.',
            'Pair normalizedL1 can be small because a common moving rear edge dominates count mass; report absolute differing counts and pixel support together, no empirical sensor-noise margin measured.',
            'Root-only visibility mask is posthoc known-generator root trajectory attribution, not an input or oracle deployed estimator.',
            'Four physical contexts and common threshold phase, not80independent people/noise replicates; group comparisons cannot uniquely isolate every source of OOF failure.'])
    result['source_identities_unchanged']=identities=={p:sha(p) for p in identities};assert result['source_identities_unchanged']
    dump(out/'results.json',result)
    print(json.dumps(dict(out=str(out),groups=len(groups),pairs=len(pairs),raw50_collisions=sum(p['views']['raw50']['exact_equal'] for p in pairs),
        fullfine_collisions=sum(p['views']['fullfine8']['exact_equal'] for p in pairs),field2_collisions=sum(p['views']['field2_8']['exact_equal'] for p in pairs),
        zero_events=sum(c['mass50']==0 for c in cases),grid_exact=result['all_saved_full_grids_reconstructed_exact'],reader_aggregate=audit['out_of_fold'])))


if __name__=='__main__':main()
