"""Preregistered posthoc null-finger controlled-source ambiguity diagnostic.

Calls the frozen renderer. New threshold-source adapter matches the original
frozen witness generator; no selector/reader/fit or scientific gate is changed.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
DEFAULT=ROOT/'outputs/s43_revisit_20261028/witness_posthoc/null_control_v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def images_from_masks(masks,moving):
    T,H,W=masks.shape
    if (T,H,W)!=(101,180,240):raise ValueError('frozen source contract101x180x240')
    if moving:
        edge=30+180*np.arange(101)/100.
        background=np.where(np.arange(240)[None,None,:]<edge[:,None,None],.15,.30)
    else:background=np.full((101,1,240),.15)
    return np.where(masks,.70,background)


def threshold_native(intensity,seed=3407,threshold=.35):
    """Original frozen source semantics: commonphase, real multiplicity,no noise."""
    levels=np.log(np.asarray(intensity,np.float64));rng=np.random.default_rng(seed)
    reference=levels[0]+rng.uniform(-.1*threshold,.1*threshold,levels.shape[1:]);out=[]
    for ms in range(len(levels)-1):
        delta=levels[ms+1]-reference;count=np.floor(np.abs(delta)/threshold).astype(np.int64)
        polarity=delta>0;reference+=np.sign(delta)*count*threshold
        y,x=np.nonzero(count)
        if len(x):out.append(np.column_stack((np.repeat(x,count[y,x]),np.repeat(y,count[y,x]),
            np.full(int(count.sum()),ms,np.int64),np.repeat(polarity[y,x],count[y,x]))))
    return np.concatenate(out).astype(np.int64) if out else np.empty((0,4),np.int64)


def canonical(events):
    e=np.asarray(events);e=e[(e[:,2]>=0)&(e[:,2]<50)]
    key=(((e[:,2]*180+e[:,1])*240+e[:,0])*2+e[:,3])
    return np.unique(key,return_counts=True)


def difference(a,b):
    ka,va=canonical(a);kb,vb=canonical(b);k=np.union1d(ka,kb)
    x=np.zeros(len(k),np.int64);y=x.copy();x[np.searchsorted(k,ka)]=va;y[np.searchsorted(k,kb)]=vb
    d=x-y;mass=int(x.sum()+y.sum());l1=int(np.abs(d).sum())
    return dict(exact_collision=l1==0,L1_count=l1,L2_count=float(np.linalg.norm(d.astype(float))),
        differing_native_coordinates=int(np.count_nonzero(d)),relative_L1_to_combined_mass=l1/mass if mass else 0.,
        null_events=int(x.sum()),action_events=int(y.sum()),both_event_streams_empty=mass==0)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=DEFAULT);args=parser.parse_args();out=args.out
    prereg=out/'prereg.json';p=json.loads(prereg.read_text())
    if p.get('schema')!='s43_witness_null_control_prereg_v1':raise ValueError('independent prereg required')
    if (out/'sourcefreeze.json').exists():raise FileExistsError('preserve previous nullrun, including failures')
    for path,value in p['existing_source_identities'].items():
        if sha(path)!=value:raise RuntimeError('frozen prior source changed:'+path)
    original=ROOT/'outputs/s43_revisit_20261028/witness_synthetic_v1'
    manifest=json.loads((original/'manifest.json').read_text());groups={};identities=dict(p['existing_source_identities'])
    for row in manifest['cases']:
        c=json.loads(Path(row['manifest']).read_text())['condition'];key=(c['root_y_rad'],c['root_dx_m'],c['moving_background'])
        groups.setdefault(key,[]).append((row,c))
        for field in ('observations','targets','manifest'):identities[str(Path(row[field]).resolve())]=sha(row[field])
        assert sha(row['observations'])==row['observation_sha256'] and sha(row['targets'])==row['target_sha256']
    assert len(groups)==8 and all(len(g)==10 for g in groups.values())
    identities[str(Path(__file__).resolve())]=sha(__file__);identities[str(prereg.resolve())]=sha(prereg)
    dump(out/'sourcefreeze.json',dict(schema='s43_witness_null_sourcefreeze_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        prereg_sha256=sha(prereg),sources=identities,new_null_render_performed=False,new_null_results_read=False,
        implementation='Frozen full_render reused; minimal new radiance/threshold adapter independently checked against oldsources'))
    source_dir=out/'sources';source_dir.mkdir()
    import torch
    torch.set_num_threads(1)
    from tools.s43_rel.mechanisms import full_render
    from tools.s43_revisit.witness_geometry import fk_joints,FINGER_JOINTS
    cases=[];pairs=[];regeneration=[]
    for index,(context,group) in enumerate(sorted(groups.items())):
        yaw,dx,moving=context;first=group[0][0]
        with np.load(first['observations']) as z:public=z['public_initial51'];beta=z['beta'];K=z['K_source'];K_event=z['K_event']
        assert not beta.any()
        states=np.repeat(public[None],101,axis=0);states[:,0]+=np.clip(np.arange(101)/50,0,1)*dx
        joints=fk_joints(states,beta);mask=full_render(states,beta,K).astype(bool)
        events=threshold_native(images_from_masks(mask,moving));prefix=source_dir/f'null{index:02d}'
        np.savez_compressed(str(prefix)+'.events.npz',events=events,offsets=np.searchsorted(events[:,2],np.arange(101)),
            public_initial51=public,beta=beta,K_source=K,K_event=K_event,boundary_ms=0,stop_ms=100)
        np.savez_compressed(str(prefix)+'.targets.npz',target51=states,joints_m=joints,time_ms=np.arange(101))
        manifest_null=dict(id=f'null{index:02d}',root_y_rad=yaw,root_dx_m=dx,moving_background=moving,local_MCP_action='none',
            observations=str(prefix)+'.events.npz',targets=str(prefix)+'.targets.npz',events100=len(events),events50=int(np.sum(events[:,2]<50)),
            observed_sha256=sha(str(prefix)+'.events.npz'),target_sha256=sha(str(prefix)+'.targets.npz'),prereg_sha256=sha(prereg))
        dump(str(prefix)+'.json',manifest_null);cases.append(manifest_null)
        for j,(row,c) in enumerate(group):
            with np.load(row['observations']) as z:action_events=z['events']
            with np.load(row['targets']) as z:action_joints=z['joints_m'];action_states=z['target51']
            action_mask=full_render(action_states,beta,K).astype(bool)
            if j==0:
                regenerated=threshold_native(images_from_masks(action_mask,moving))
                same=np.array_equal(regenerated,action_events)
                regeneration.append(dict(context=list(context),old_case=row['id'],raw100_bitwise=same,
                    old_events=len(action_events),regenerated_events=len(regenerated)))
                if not same:raise RuntimeError('new threshold adapter doesnot reproduce original fixedsource')
            raw=difference(events,action_events);relative=(action_joints[50]-action_joints[50,:1])-(joints[50]-joints[50,:1])
            fk=np.linalg.norm(relative,axis=-1)*1000;idx=FINGER_JOINTS[c['finger']]
            delta=action_mask[:51]!=mask[:51];frame_changed=np.any(delta,axis=(1,2))
            pair=dict(null_id=manifest_null['id'],action_id=row['id'],context=list(context),finger=c['finger'],sign=c['sign'],
                **raw,FK_J0_mm=float(fk.mean()),active_finger_J0_mm=float(fk[idx].mean()),
                wrist_abs_mm=float(np.linalg.norm(action_joints[50,0]-joints[50,0])*1000),
                mask_given_same_root=dict(ever_different_pixels=int(np.any(delta,axis=0).sum()),
                    terminal_different_pixels=int(delta[-1].sum()),sum_differing_pose_samples_pixels=int(delta.sum()),
                    first_different_pose_ms=int(np.flatnonzero(frame_changed)[0]) if frame_changed.any() else None),
                exposure='Known generator root/background/boundary heldpaired; no targets or nullclass provided to any originalreader')
            pairs.append(pair)
        print(json.dumps(dict(null=index+1,total=8,events50=manifest_null['events50'])),flush=True)
    exact=[r for r in pairs if r['exact_collision']]
    positive=[r for r in pairs if r['L1_count']>0]
    aggregate=dict(pairs=len(pairs),exact_collision_count=len(exact),exact_pairs=[dict(null_id=r['null_id'],action_id=r['action_id'],context=r['context'],
        finger=r['finger'],sign=r['sign'],active_finger_J0_mm=r['active_finger_J0_mm'],FK_J0_mm=r['FK_J0_mm'],null_events=r['null_events'],action_events=r['action_events']) for r in exact],
        positive_L1_atmost={str(n):sum(r['L1_count']<=n for r in positive) for n in (4,16)},
        positive_relative_L1_atmost={str(t):sum(r['relative_L1_to_combined_mass']<=t for r in positive) for t in (.0001,.001)},
        no_finger_visible_change=sum(r['mask_given_same_root']['ever_different_pixels']==0 for r in pairs))
    unchanged=all(sha(path)==value for path,value in identities.items());assert unchanged
    dump(out/'results.json',dict(schema='s43_witness_null_diagnosis_v1',preregistered_posthoc=True,prereg_sha256=sha(prereg),
        sourcefreeze_sha256=sha(out/'sourcefreeze.json'),aggregate=aggregate,null_sources=cases,pairs=pairs,
        adapter_regeneration_checks=regeneration,all8oldsource_regenerations_bitwise=all(r['raw100_bitwise'] for r in regeneration),
        original_sources_and_reader_unchanged=unchanged,new_fits=0,new_accuracy_scores=0,new_thresholds_tuned=False,
        scope='Only eight known synthetic constant-radiance/phase/native-ms contexts; nullomission limitation of closed10mode family,not a universal realdata information claim'))
    print(json.dumps(aggregate))


if __name__=='__main__':main()
