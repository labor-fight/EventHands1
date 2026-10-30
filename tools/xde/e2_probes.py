#!/usr/bin/env python3
"""XDE E2: frozen Q1 bank 34->4 conditional decoding, no production edits.

Prepare freezes sampling before extraction. CUDA work is restricted to physical
devices 4/5 through CUDA_VISIBLE_DEVICES and batch-one source forwards.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'model'))
OUT = ROOT / 'outputs/xde/e2'
RUN = ROOT / 'outputs/semkine/xdl_q1_s3407'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    temp = path.with_suffix(path.suffix + '.tmp'); temp.write_text(text); temp.replace(path)


def fixed_midpoints(items, n):
    if len(items) < n:
        raise RuntimeError(f'not enough disjoint packets: {len(items)} < {n}')
    return [items[int((i + .5) * len(items) / n)] for i in range(n)]


def prepare():
    from config import load_config
    from semkine.dataset import sequences_for_split, splits_manifest
    cfg = load_config(RUN / 'config_resolved.yaml')
    source = ROOT / 'outputs/xdl/packet_bank.json'
    bank = json.loads(source.read_text())
    official = {x[0] for x in sequences_for_split(Path(cfg['DATA']['ROOT']), 'train', splits_manifest(cfg))}
    assert len(official) == 72
    selected = []
    for name, meta in sorted(bank['sequences'].items()):
        rows = meta['rows']
        if name in official:
            # Whole valid runs shorter than 5s stay intact; longer runs split
            # into fixed disjoint 5s blocks. All perturbations share the fold.
            keys = [(r['run_id'], (r['event_start_ms'] - meta['valid_runs_ms'][r['run_id']][0]) // 5000) for r in rows]
            unique = sorted(set(keys)); folds = {k: ('check' if i % 4 == 3 else 'fit') for i, k in enumerate(unique)}
            if len(unique) < 4:
                raise RuntimeError(f'{name}: fewer than four independent temporal blocks')
            pools = {split: [dict(r, segment=f'{name}:{key[0]}:{key[1]}') for r, key in zip(rows, keys) if folds[key] == split] for split in ('fit', 'check')}
            requests = [('fit', 24), ('check', 8)]
        elif name in ('zgz_global', 'zgz_local'):
            pools = {'zgz': [dict(r, segment=f'{name}:{r["run_id"]}') for r in rows]}; requests = [('zgz', 128)]
        else:
            raise RuntimeError('undeclared sequence')
        for split, count in requests:
            pool = pools[split]
            for row in fixed_midpoints(pool, count):
                ordinal = next(i for i, r in enumerate(pool) if r['index'] == row['index'])
                donor = pool[(ordinal + len(pool) // 2) % len(pool)]
                selected.append({'sequence': name, 'directory': meta['directory'], 'split': split,
                                 'row': row, 'donor_row': donor})
    checkpoint = next(RUN.glob('*-step=6000.ckpt'))
    p = {'study': 'XDE-E2-Q1-BANK34TO4-20261014', 'created_before_extraction': True,
         'source_bank': str(source), 'source_bank_sha256': sha(source),
         'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
         'config': str(RUN / 'config_resolved.yaml'), 'config_sha256': sha(RUN / 'config_resolved.yaml'),
         'script_at_freeze_sha256': sha(__file__), 'official_split': str(splits_manifest(cfg)),
         'official_split_sha256': sha(splits_manifest(cfg)),
         'packet_sampling': '72 sequences each24fit+8check;zgz128 each;chronological midpoint ranks',
         'fit_check_partition': 'whole valid runs if <5s,else disjoint5s blocks;chronological segment index mod4==3 check;50ms packets never cross5s block',
         'perturb_z_mm': [0., -50., 50.], 'all_other_state_blocks': 'clean preceding GT; root/finger not ranked',
         'target': 'actual current GT transl minus perturbed incoming transl, xyz millimetres; not injected delta alone',
         'wrong_content': 'same-sequence same-fold donor halfway round chronology; resample donor xy separately by polarity to original per-polarity count; keep original timestamps,p,count; recompute full model at identical incoming state',
         'wrong_limit': 'OOD content counterfactual, not a physically consistent alternative stream; same pose donor can make it weak',
         'interfaces': ['state', 'pre', 'post', 'wrong_pre', 'wrong_post'],
         'head': {'input': 643, 'state': 99, 'bank': '16x34;post4zero-padded within each slot;state-only all bank zero',
                  'widths': [128, 512], 'seeds': [1701, 1702], 'architecture': 'Linear643,H;ReLU;LinearH,H;ReLU;LinearH,3',
                  'optimizer': 'AdamW', 'lr': .001, 'weight_decay': .0001, 'epochs': 300, 'batch_size': 256,
                  'selection': 'minimum check z-MAE per head; no zgz fitting, tuning or checkpoint selection',
                  'normalization': 'fit-only mean/std independently per interface;std floor1e-3;targets divided by50'},
         'gate': {'primary': 'check z-MAE vs state and matching wrong interface;bothseeds', 'relative': .10, 'absolute_mm': 2.,
                  'wide_head': 'prespecified512;reported separately,not selected using zgz'},
         'physical_gpus': [4, 5], 'source_forward_batch': 1, 'max_gpu_hours': 6.,
         'selected_packets': selected, 'packets': len(selected), 'feature_rows': len(selected) * 3,
         'source_forwards': len(selected) * 6}
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text())
        # Script fixes are independently recorded in cache provenance.
        p['script_at_freeze_sha256'] = old['script_at_freeze_sha256']
        if old != p: raise RuntimeError('frozen protocol differs')
    else: save_json(path, p)
    (OUT / 'protocol.sha256').write_text(sha(path) + '\n')
    print(json.dumps({'protocol_sha256': sha(path), 'packets': p['packets'], 'rows': p['feature_rows'], 'forwards': p['source_forwards']}), flush=True)


def gpu_contract():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in ('4', '5'): raise RuntimeError('set CUDA_VISIBLE_DEVICES to exactly4 or5')
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(3407)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return torch.device('cuda:0')


def wrong_events(events, donor):
    result = np.array(events, copy=True)
    for pol in (0, 1):
        dest = np.flatnonzero((events[:, 4] > .5) == bool(pol))
        source = np.flatnonzero((donor[:, 4] > .5) == bool(pol))
        if len(dest) and not len(source): source = np.arange(len(donor))
        if len(dest) and not len(source):
            # Explicit retained degenerate control, not silently real content.
            result[dest, 1] = 239. - result[dest, 1]
        elif len(dest):
            source = source[np.minimum((np.arange(len(dest)) * len(source) / len(dest)).astype(int), len(source)-1)]
            result[dest, 1:3] = donor[source, 1:3]
    assert np.array_equal(result[:, [0, 3, 4]], events[:, [0, 3, 4]])
    return result


def extract(shard):
    import torch
    from tools.xdl import diagnostics as D
    device = gpu_contract(); protocol_path = OUT / 'protocol.json'
    p = json.loads(protocol_path.read_text()); ph = sha(protocol_path)
    model, cfg, identity = D.load_model(str(RUN), device)
    model.xyz_keep_aux = True
    for parameter in model.parameters(): parameter.requires_grad_(False)
    captured = {}
    def hook(module, inputs, output):
        captured['pre'] = inputs[0].detach().float().cpu().numpy()[0]
        captured['post'] = output.detach().float().cpu().numpy()[0]
    handle = model.observation_bank.compress.register_forward_hook(hook)
    selected = p['selected_packets']; names = sorted({r['sequence'] for r in selected})[shard::2]
    bank = json.loads(Path(p['source_bank']).read_text())
    start = time.perf_counter(); calls = 0; completed = 0
    for name in names:
        path = OUT / 'cache' / f'{name}.npz'; audit_path = path.with_suffix('.json')
        if path.exists():
            if json.loads(audit_path.read_text())['protocol_sha256'] != ph: raise RuntimeError('cache protocol differs')
            continue
        data = D.load_data(cfg, name, bank['sequences'][name]); beta, camera = D.context(data, device)
        model.set_hand_context(beta, camera); D.reset(model)
        arrays = {k: [] for k in ('pre', 'post', 'wrong_pre', 'wrong_post', 'state', 'target_xyz_mm', 'incoming51', 'target51', 'prediction51')}
        metadata = []; query_seconds = 0.; wrong_strength = []
        for item in [r for r in selected if r['sequence'] == name]:
            row = item['row']; ev = D.raw_events(data, row); donor = D.raw_events(data, item['donor_row']); wrong = wrong_events(ev, donor)
            changed = float(np.any(ev[:, 1:3] != wrong[:, 1:3], -1).mean()) if len(ev) else 0.
            wrong_strength.append(changed)
            for dz in p['perturb_z_mm']:
                incoming = data['gt'][row['previous_end_ms']].copy(); incoming[2] += dz / 1000.
                truth = data['gt'][row['end_ms']].copy()
                for label, raw in (('', ev), ('wrong_', wrong)):
                    captured.clear(); packet = D.make_packet(raw, incoming[None], beta, camera, device, row)
                    if packet.target.any(): raise RuntimeError('target leakage')
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    with torch.no_grad(): output = model.forward_packet(packet)
                    torch.cuda.synchronize(); query_seconds += time.perf_counter()-t0; calls += 1
                    if not len(raw):
                        raise RuntimeError('empty source packet: explicit protocol amendment required')
                    pre = captured['pre']; post = np.zeros((16,34), np.float32); post[:, :4] = model.xyz_last['xdl_bank'].float().cpu().numpy()[0].reshape(16,4)
                    state = model.xyz_last['xdl_state'].float().cpu().numpy()[0]
                    if not all(np.isfinite(v).all() for v in (pre, post, state, output.cpu().numpy())): raise FloatingPointError('nonfinite feature')
                    arrays[label+'pre'].append(pre); arrays[label+'post'].append(post)
                    if not label:
                        arrays['state'].append(state); arrays['prediction51'].append(output.cpu().numpy()[0])
                    elif not np.array_equal(state, arrays['state'][-1]): raise RuntimeError('wrong control changed state99')
                arrays['incoming51'].append(incoming); arrays['target51'].append(truth); arrays['target_xyz_mm'].append((truth[:3]-incoming[:3])*1000.)
                metadata.append({'sequence': name, 'split': item['split'], 'segment': row['segment'], 'index': row['index'], 'run_id': row['run_id'], 'end_ms': row['end_ms'], 'previous_end_ms': row['previous_end_ms'], 'dz_mm': dz, 'raw_count': len(ev), 'donor_end_ms': item['donor_row']['end_ms'], 'wrong_xy_fraction': changed, 'event_sha256': hashlib.sha256(ev.tobytes()).hexdigest()})
            if calls % 600 == 0:
                print(json.dumps({'phase':'extract','shard':shard,'calls':calls,'seconds':time.perf_counter()-start,'sequence':name}),flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{k: np.stack(v) for k,v in arrays.items()})
        save_json(audit_path, {'protocol_sha256': ph, 'executed_script_sha256':sha(__file__), 'identity':identity,
                              'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'], 'source_batch_size':1, 'query_seconds':query_seconds,
                              'rows':metadata,'wrong_xy_fraction_mean':float(np.mean(wrong_strength)), 'cache_sha256':sha(path)})
        completed += len(metadata)
        print(json.dumps({'phase':'sequence_done','shard':shard,'sequence':name,'feature_rows':len(metadata),'query_seconds':query_seconds,'calls':calls,'wall_seconds':time.perf_counter()-start}),flush=True)
    handle.remove()
    save_json(OUT/f'extract_shard{shard}.json', {'protocol_sha256':ph,'wall_seconds':time.perf_counter()-start,'calls':calls,'rows_new':completed,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES']})


def load_cache():
    p = json.loads((OUT/'protocol.json').read_text()); ph = sha(OUT/'protocol.json')
    names = sorted({r['sequence'] for r in p['selected_packets']}); arrays = {}; meta=[]
    for name in names:
        path=OUT/'cache'/f'{name}.npz'; a=json.loads(path.with_suffix('.json').read_text())
        if a['protocol_sha256']!=ph or a['cache_sha256']!=sha(path):raise RuntimeError('cache provenance changed')
        with np.load(path) as data:
            for k in data.files: arrays.setdefault(k,[]).append(data[k])
        meta.extend(a['rows'])
    return {k:np.concatenate(v) for k,v in arrays.items()},meta,p


def metrics(pred,target,metadata):
    err=pred-target; zerr=np.abs(err[:,2]); truez=target[:,2]; active=np.abs(truez)>=1.
    return {'n':len(pred),'z_mae_mm':float(zerr.mean()),'z_signed_bias_mm':float(err[:,2].mean()),
            'xyz_endpoint_mm':float(np.linalg.norm(err,axis=-1).mean()),
            'hold_z_mae_mm':float(np.abs(target[:,2]).mean()),'hold_xyz_mm':float(np.linalg.norm(target,axis=-1).mean()),
            'z_direction_n':int(active.sum()),'z_direction_accuracy':float((pred[active,2]*truez[active]>0).mean()) if active.any() else None,
            'z_better_than_hold':float((zerr<np.abs(target[:,2])).mean())}


def fit(seed):
    import torch
    from torch import nn
    device=gpu_contract(); data,metadata,p=load_cache(); start=time.perf_counter()
    split=np.array([r['split'] for r in metadata]); fit_mask=split=='fit'; check_mask=split=='check'
    # Immutable disjoint segment validation including all perturbation replicas.
    fit_segments={r['segment'] for r in metadata if r['split']=='fit'}; check_segments={r['segment'] for r in metadata if r['split']=='check'}
    assert not (fit_segments & check_segments)
    y=data['target_xyz_mm'].astype(np.float32)
    summaries=[]
    for width in p['head']['widths']:
        for interface in p['interfaces']:
            torch.manual_seed(seed); np.random.seed(seed)
            bank=np.zeros_like(data['pre']) if interface=='state' else data[interface]
            x=np.concatenate([data['state'],bank.reshape(len(bank),-1)],-1).astype(np.float32)
            variable_inputs=int((np.ptp(x[fit_mask],axis=0)>0).sum())
            mean=x[fit_mask].mean(0); std=np.maximum(x[fit_mask].std(0),.001)
            x=(x-mean)/std
            xt=torch.from_numpy(x).to(device);yt=torch.from_numpy(y/50.).to(device)
            fi=torch.from_numpy(np.flatnonzero(fit_mask)).to(device);ci=torch.from_numpy(np.flatnonzero(check_mask)).to(device)
            head=nn.Sequential(nn.Linear(643,width),nn.ReLU(),nn.Linear(width,width),nn.ReLU(),nn.Linear(width,3)).to(device)
            optimizer=torch.optim.AdamW(head.parameters(),lr=p['head']['lr'],weight_decay=p['head']['weight_decay'])
            best=float('inf'); best_state=None; best_epoch=None; curve=[]; train_start=time.perf_counter()
            for epoch in range(p['head']['epochs']):
                head.train(); order=fi[torch.randperm(len(fi),device=device)]
                for indices in order.split(p['head']['batch_size']):
                    optimizer.zero_grad(set_to_none=True); loss=(head(xt[indices])-yt[indices]).square().mean();loss.backward();optimizer.step()
                head.eval()
                with torch.no_grad():
                    score=float((head(xt[ci])[:,2]-yt[ci,2]).abs().mean()*50.)
                if score<best:
                    best=score;best_epoch=epoch+1;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
                if (epoch+1)%25==0:curve.append({'epoch':epoch+1,'check_z_mae_mm':score,'loss':float(loss)})
            head.load_state_dict(best_state);head.eval()
            with torch.no_grad(): predictions=torch.cat([head(t) for t in xt.split(1024)]).cpu().numpy()*50.
            report={'seed':seed,'width':width,'interface':interface,'best_epoch':best_epoch,'nominal_parameters':sum(v.numel() for v in head.parameters()),
                    'variable_input_dimensions':variable_inputs,'active_first_layer_parameters':variable_inputs*width,
                    'effective_parameters_nonconstant_inputs':sum(v.numel() for v in head.parameters())-(643-variable_inputs)*width,
                    'train_seconds':time.perf_counter()-train_start,'metrics':{},'curve':curve}
            groups={key:split==key for key in ('fit','check','zgz')}
            for mode in ('local','global'):
                groups[f'check_{mode}']=check_mask&np.array([f'_{mode}' in r['sequence'] for r in metadata])
                groups[f'zgz_{mode}']=np.array([r['sequence']==f'zgz_{mode}' for r in metadata])
            for key,mask in groups.items():
                report['metrics'][key]=metrics(predictions[mask],y[mask],[r for r,m in zip(metadata,mask) if m])
                if key.startswith('zgz_') or key.startswith('check_'):
                    for dz in p['perturb_z_mm']:
                        sub=mask&np.array([r['dz_mm']==dz for r in metadata])
                        report['metrics'][f'{key}_dz{int(dz)}']=metrics(predictions[sub],y[sub],[])
            path=OUT/'heads'/f'{interface}_w{width}_s{seed}.pt';path.parent.mkdir(parents=True,exist_ok=True)
            torch.save({'state_dict':best_state,'mean':mean,'std':std,'width':width,'interface':interface,'seed':seed,'protocol_sha256':sha(OUT/'protocol.json')},path)
            np.savez_compressed(path.with_suffix('.predictions.npz'),delta_xyz_mm=predictions,target_xyz_mm=y)
            save_json(path.with_suffix('.json'),report);summaries.append(report)
            print(json.dumps({k:report[k] for k in ('seed','width','interface','best_epoch','train_seconds')}|{'check_z':report['metrics']['check']['z_mae_mm'],'zgz_z':report['metrics']['zgz']['z_mae_mm']}),flush=True)
    save_json(OUT/f'fit_s{seed}.json',{'protocol_sha256':sha(OUT/'protocol.json'),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'wall_seconds':time.perf_counter()-start,'results':summaries})


def summarize():
    p=json.loads((OUT/'protocol.json').read_text());reports={}
    for seed in p['head']['seeds']:
        value=json.loads((OUT/f'fit_s{seed}.json').read_text())
        for r in value['results']:reports[(r['width'],r['seed'],r['interface'])]=r
    gates=[]
    for width in p['head']['widths']:
        for interface in ('pre','post'):
            checks=[]
            for seed in p['head']['seeds']:
                true=reports[(width,seed,interface)]['metrics']['check']['z_mae_mm']
                for reference in ('state','wrong_'+interface):
                    baseline=reports[(width,seed,reference)]['metrics']['check']['z_mae_mm'];improve=baseline-true
                    checks.append({'seed':seed,'reference':reference,'true_z_mae_mm':true,'baseline_z_mae_mm':baseline,'improvement_mm':improve,'pass':improve>=2. and improve>=.1*baseline})
            gates.append({'width':width,'interface':interface,'pass':all(r['pass'] for r in checks),'comparisons':checks})
    save_json(OUT/'verdict.json',{'protocol_sha256':sha(OUT/'protocol.json'),'gates':gates,'scope':'conditional xyz decoding only; frozen pretrained Q1; perturbations clean,+/-50mmz;no re-trained encoder;negative not information-theoretic result','needs_short_rollout':any(g['pass'] for g in gates)})
    print(json.dumps({'gates':gates},ensure_ascii=False),flush=True)


def prepare_rollout():
    """Parent-authorized expansion; keep the original smaller plan unchanged."""
    e1path=ROOT/'outputs/xde/e1/protocol.json';e1=json.loads(e1path.read_text())
    gates=json.loads((OUT/'verdict.json').read_text())['gates']
    assert all(next(g['pass'] for g in gates if g['width']==128 and g['interface']==i) for i in ('pre','post'))
    modes=['q1','state_s1701','state_s1702','pre_s1701','pre_s1702','post_s1701','post_s1702','hold']
    p={'study':'XDE-E2-short-own-rollout-v2','amendment_reason':'parent instruction: use E1 fixed own_selection24x20, evaluate pre and post width128 both seeds; add clean same-clip starts; no zgz-based head selection',
       'original_short_plan_sha256':sha(OUT/'rollout_protocol.json'),'e2_protocol_sha256':sha(OUT/'protocol.json'),
       'e1_protocol_sha256':sha(e1path),'selected':e1['own_selection']['selected'],'steps':20,
       'initials':['own','clean'],'modes':modes,'width':128,'seeds':[1701,1702],
       'update':'only replace translation3;root/fingers use Q1 output at each branch own state; no GT resets; source batch1',
       'score':'all queries excluding initial; physical wrist,absolute/RA MPJPE,root/finger,existing XDL recovery/motion; clean first-step and signed z separately',
       'scope':'small composite-head diagnostic, no new full-model claim; check selected heads, no zgz tuning',
       'physical_gpus':[4,5],'expected_forwards':2*24*2*7*20,
       'head_sha256':{mode:sha(OUT/'heads'/f'{mode.split("_s")[0]}_w128_s{mode.split("_s")[1]}.pt') for mode in modes if '_s' in mode}}
    path=OUT/'rollout_protocol_v2.json'
    if path.exists():assert json.loads(path.read_text())==p
    else:save_json(path,p)
    print(json.dumps({'rollout_protocol_sha256':sha(path),'forwards':p['expected_forwards']}),flush=True)


def rollout(sequence):
    import gzip
    import torch
    from torch import nn
    from tools.xdl import diagnostics as D
    from tools.xde.e1_diagnostics import clean_json
    device=gpu_contract();torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    protocol_path=OUT/'rollout_protocol_v2.json';p=json.loads(protocol_path.read_text());ph=sha(protocol_path)
    model,cfg,identity=D.load_model(str(RUN),device);model.xyz_keep_aux=True
    mano=D.ManoLayer(cfg['MANO']['NPZ'],add_mean=False).to(device).eval();mano.fast_chain=True
    mean=mano.hands_mean.cpu().numpy();bank=json.loads((ROOT/'outputs/xdl/packet_bank.json').read_text())
    data=D.load_data(cfg,sequence,bank['sequences'][sequence]);beta,camera=D.context(data,device);model.set_hand_context(beta,camera)
    captured={}
    handle=model.observation_bank.compress.register_forward_hook(lambda _m,inputs,_o:captured.update(pre=inputs[0].detach()))
    heads={}
    for mode,digest in p['head_sha256'].items():
        interface,seed=mode.split('_s');path=OUT/'heads'/f'{interface}_w128_s{seed}.pt';assert sha(path)==digest
        saved=torch.load(path,map_location='cpu');head=nn.Sequential(nn.Linear(643,128),nn.ReLU(),nn.Linear(128,128),nn.ReLU(),nn.Linear(128,3)).to(device)
        head.load_state_dict(saved['state_dict']);head.eval()
        heads[mode]=(head,torch.as_tensor(saved['mean'],device=device),torch.as_tensor(saved['std'],device=device),interface)
    start=time.perf_counter();calls=0;records=[];replay=[]
    with gzip.open(ROOT/f'outputs/xde/e1/q1_own_{sequence}.pt.gz','rb') as f:e1=torch.load(f,map_location='cpu')
    e1real={r['clip_id']:r['predictions'] for r in e1['trajectories'] if r['spec']['block']=='real'}
    for clip in p['selected'][sequence]:
        rows=[bank['sequences'][sequence]['rows'][i] for i in clip['end_indices']];assert len(rows)==20
        raw=[D.raw_events(data,r) for r in rows]
        times=np.array([clip['initial_time_ms']]+[r['end_ms'] for r in rows]);truth=data['gt'][times]
        for initialization in p['initials']:
            initial=np.array(clip['initial'] if initialization=='own' else truth[0],np.float32)
            predictions=[]
            for mode in p['modes']:
                states=[initial.copy()];D.reset(model)
                for events,row in zip(raw,rows):
                    if mode=='hold':states.append(initial.copy());continue
                    previous=states[-1];captured.clear();packet=D.make_packet(events,previous[None],beta,camera,device,row)
                    assert not bool(packet.target.any())
                    with torch.no_grad():
                        output=model.forward_packet(packet);calls+=1
                        if mode!='q1' and len(events):
                            head,center,scale,interface=heads[mode]
                            features=torch.zeros((1,16,34),device=device)
                            if interface=='pre':features=captured['pre']
                            elif interface=='post':features[:,:,:4]=model.xyz_last['xdl_bank'].reshape(1,16,4)
                            x=torch.cat([model.xyz_last['xdl_state'],features.reshape(1,-1)],-1)
                            delta=head((x-center)/scale)*.05
                            output=output.clone();output[:,:3]=packet.prev_state[:,:3]+delta
                    value=output.float().cpu().numpy()[0]
                    if not np.isfinite(value).all():raise FloatingPointError(f'nonfinite rollout {sequence} {clip["clip_id"]} {mode}')
                    states.append(value)
                predictions.append(np.stack(states))
            predictions=np.stack(predictions)
            if initialization=='own':replay.append(float(np.max(np.abs(predictions[0]-e1real[clip['clip_id']]))))
            specs=[{'id':mode,'block':mode} for mode in p['modes']]
            with torch.no_grad():scored=D.score_trajectories(mano,predictions,truth,times,specs,np.stack([initial]*len(specs)),beta,device,mean)
            for row in scored:
                row.update(sequence=sequence,clip_id=clip['clip_id'],initialization=initialization,truth=truth,
                           transl_z_error_mm=(row['predictions'][:,2]-truth[:,2])*1000.)
                records.append(row)
        print(json.dumps({'rollout_sequence':sequence,'clip':clip['clip_id'],'calls':calls,'wall_seconds':time.perf_counter()-start}),flush=True)
    handle.remove();path=OUT/f'rollout_{sequence}.pt.gz'
    payload={'protocol_sha256':ph,'identity':identity,'records':records,'q1_e1_replay_max_state_difference':max(replay),'calls':calls,'wall_seconds':time.perf_counter()-start,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'executed_script_sha256':sha(__file__)}
    with gzip.open(path,'wb',compresslevel=2) as f:torch.save(payload,f)
    result={k:v for k,v in payload.items() if k!='records'};result['raw_sha256']=sha(path);result['metrics']={}
    for initialization in p['initials']:
        for mode in p['modes']:
            rows=[r for r in records if r['initialization']==initialization and r['spec']['id']==mode]
            score={'n_clips':len(rows),'n_queries':sum(len(r['predictions'])-1 for r in rows)}
            for metric in rows[0]['errors']:
                values=np.concatenate([r['errors'][metric][1:] for r in rows]);score[metric]=float(values.mean())
                score['final_'+metric]=float(np.mean([r['errors'][metric][-1] for r in rows]))
                score['first_'+metric]=float(np.mean([r['errors'][metric][1] for r in rows]))
            ze=np.concatenate([r['transl_z_error_mm'][1:] for r in rows]);score['z_signed_bias_mm']=float(ze.mean());score['z_mae_mm']=float(np.abs(ze).mean())
            score['first_z_signed_bias_mm']=float(np.mean([r['transl_z_error_mm'][1] for r in rows]));score['final_z_signed_bias_mm']=float(np.mean([r['transl_z_error_mm'][-1] for r in rows]))
            score['initially_satisfied']=sum(r['recovery']['initially_satisfied'] for r in rows)
            score['new_sustained_recovery']=sum(not r['recovery']['initially_satisfied'] and r['recovery']['hit_step'] is not None for r in rows)
            score['motion']={space:{metric:float(np.nanmean([r['motion'][space][metric] if r['motion'][space][metric] is not None else np.nan for r in rows])) for metric in ('gain','gain_bias','cosine','lag_ms')} for space in ('absolute','root_relative','root','finger')}
            result['metrics'][f'{initialization}/{mode}']=score
    save_json(path.with_suffix('.json'),clean_json(result))
    print(json.dumps({'rollout_done':sequence,'calls':calls,'wall_seconds':result['wall_seconds'],'q1_e1_maxdiff':max(replay)}),flush=True)


def self_test():
    rng=np.random.default_rng(17);e=np.zeros((19,5),np.float32);e[:,1:3]=rng.integers(0,170,(19,2));e[:,3]=np.arange(19)*.001;e[:,4]=np.arange(19)%2
    d=e[:7].copy();w=wrong_events(e,d)
    assert np.array_equal(w[:,[0,3,4]],e[:,[0,3,4]]) and w.shape==e.shape
    post=rng.normal(size=(5,16,4)).astype(np.float32);padded=np.zeros((5,16,34),np.float32);padded[:,:,:4]=post
    assert np.array_equal(padded[:,:,:4],post) and not padded[:,:,4:].any()
    states=rng.normal(size=(5,51)).astype(np.float32);truth=states.copy();truth[:,:3]+=.003
    target=(truth[:,:3]-states[:,:3])*1000;assert np.allclose(target,3,atol=.001)
    save_json(OUT/'self_test.json',{'passed':True,'checks':['wrong_count_p_timestamp_exact','post4zero_padding','true_endpoint_target'],'script_sha256':sha(__file__)})
    print('self tests passed',flush=True)


def supplemental():
    """CPU only: audit fixed rollout outputs; never fit or select a head."""
    import gzip
    import torch
    torch.set_num_threads(4)
    from tools.xde.e1_diagnostics import clean_json
    result={'method':{'auc':'per-clip existing XDL time_auc, then equal mean of24clips; includes initial state and trapezoidal endpoint weights; distinct from20output mean',
                      'motion':'existing smoothed-GT200ms motion_response; explicit defined/undefined clip denominators; lag search0/50/100/150/200ms',
                      'acceleration_and_jerk':'post-hoc translation-only supplemental; use20 outputstates excluding initial, raw GT, exact50ms spacing; 2nd/3rd differences divided bydt^2/dt^3; mm/s^2 andmm/s^3; GT label noise remains; low roughness alone is not successful tracking',
                      'scope':'24 fixed stratified own-failure clips/sequence x20queries; no full-sequence performance claim'},
            'sequences':{},'audit':{'all_finite':True,'failures':0,'calls':0,'initial_angles_first_output_exact':True,
                                  'first_ra_mpjpe_max_difference_mm':0.,'first_root_angle_max_difference_deg':0.,'first_finger_angle_max_difference_deg':0.},
            'budget':{}}
    for sequence in ('zgz_local','zgz_global'):
        path=OUT/f'rollout_{sequence}.pt.gz'
        with gzip.open(path,'rb') as f:saved=torch.load(f,map_location='cpu')
        rows=saved['records'];result['audit']['calls']+=saved['calls'];cohort={}
        for row in rows:
            valid=bool(np.isfinite(row['predictions']).all())
            result['audit']['all_finite'] &= valid;result['audit']['failures']+=int(not valid)
            q1=next(x for x in rows if x['clip_id']==row['clip_id'] and x['initialization']==row['initialization'] and x['spec']['id']=='q1')
            if row['spec']['id']!='hold':
                result['audit']['initial_angles_first_output_exact'] &= np.array_equal(row['predictions'][1,3:],q1['predictions'][1,3:])
                for metric,key in [('mpjpe_ra_mm','first_ra_mpjpe_max_difference_mm'),('root_deg','first_root_angle_max_difference_deg'),('finger_deg','first_finger_angle_max_difference_deg')]:
                    result['audit'][key]=max(result['audit'][key],float(abs(row['errors'][metric][1]-q1['errors'][metric][1])))
        for initialization in ('own','clean'):
            for mode in json.loads((OUT/'rollout_protocol_v2.json').read_text())['modes']:
                chosen=[r for r in rows if r['initialization']==initialization and r['spec']['id']==mode]
                out={'n_clips':len(chosen),'auc':{},'motion':{},'roughness':{},'recovery':{},'first_all_angles_equal_q1':True}
                for metric in chosen[0]['auc']:
                    values=[r['auc'][metric] for r in chosen];good=[float(v) for v in values if v is not None and np.isfinite(v)]
                    out['auc'][metric]={'mean':float(np.mean(good)) if good else None,'defined':len(good),'undefined':len(values)-len(good)}
                for space in ('absolute','root_relative','root','finger'):
                    out['motion'][space]={}
                    for metric in ('gain','gain_bias','cosine','lag_ms'):
                        values=[r['motion'][space][metric] for r in chosen];good=[float(v) for v in values if v is not None and np.isfinite(v)]
                        out['motion'][space][metric]={'mean':float(np.mean(good)) if good else None,'defined':len(good),'undefined':len(values)-len(good)}
                rough={k:[] for k in ('pred_acceleration_mm_s2','gt_acceleration_mm_s2','acceleration_error_mm_s2','pred_jerk_mm_s3','gt_jerk_mm_s3','jerk_error_mm_s3')}
                for row in chosen:
                    assert np.all(np.diff(row['times_ms'])==50)
                    pred=row['predictions'][1:,:3].astype(np.float64)*1000.;truth=row['truth'][1:,:3].astype(np.float64)*1000.
                    for order,label,unit in ((2,'acceleration','s2'),(3,'jerk','s3')):
                        pd=np.diff(pred,n=order,axis=0)/(.05**order);gt=np.diff(truth,n=order,axis=0)/(.05**order)
                        rough[f'pred_{label}_mm_{unit}'].append(float(np.linalg.norm(pd,axis=-1).mean()))
                        rough[f'gt_{label}_mm_{unit}'].append(float(np.linalg.norm(gt,axis=-1).mean()))
                        rough[f'{label}_error_mm_{unit}'].append(float(np.linalg.norm(pd-gt,axis=-1).mean()))
                out['roughness']={k:{'mean':float(np.mean(v)),'defined':len(v),'undefined':0} for k,v in rough.items()}
                initially=sum(r['recovery']['initially_satisfied'] for r in chosen)
                out['recovery']={'initially_satisfied':initially,'eligible_new_recovery':len(chosen)-initially,
                                 'new_sustained_recovery':sum(not r['recovery']['initially_satisfied'] and r['recovery']['hit_step'] is not None for r in chosen),
                                 'condition':'absolute MPJPE<=20mm and RA<=10mm for3consecutivequeries; existing XDL rule',
                                 'final_good':sum(r['errors']['mpjpe_abs_mm'][-1]<=20 and r['errors']['mpjpe_ra_mm'][-1]<=10 for r in chosen)}
                q1s={r['clip_id']:r for r in rows if r['initialization']==initialization and r['spec']['id']=='q1'}
                if mode!='hold':out['first_all_angles_equal_q1']=all(np.array_equal(r['predictions'][1,3:],q1s[r['clip_id']]['predictions'][1,3:]) for r in chosen)
                else:out['first_all_angles_equal_q1']=None
                cohort[f'{initialization}/{mode}']=out
        result['sequences'][sequence]=cohort
    result['audit']['expected_calls']=13440;assert result['audit']['calls']==13440
    receipts={'extract':[json.loads((OUT/f'extract_shard{s}.json').read_text())['wall_seconds'] for s in (0,1)],
              'fit':[json.loads((OUT/f'fit_s{s}.json').read_text())['wall_seconds'] for s in (1701,1702)],
              'rollout':[json.loads((OUT/f'rollout_{s}.pt.json').read_text())['wall_seconds'] for s in ('zgz_local','zgz_global')]}
    result['budget']={'worker_allocated_seconds':receipts,'summed_worker_seconds':sum(sum(v) for v in receipts.values()),
                      'allocated_gpu_hours':sum(sum(v) for v in receipts.values())/3600.,'maximum_gpu_hours':6.,
                      'note':'sum wallclock for all extraction,20headfits,androllout workers; diagnostic forward timer is not deployment latency'}
    result['executed_source_sha256']=sha(__file__)
    save_json(OUT/'supplemental_audit.json',clean_json(result))
    print(json.dumps(clean_json({'audit':result['audit'],'budget':result['budget']})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','extract','fit','summarize','self-test','prepare-rollout','rollout','supplemental']);parser.add_argument('--shard',type=int,choices=[0,1]);parser.add_argument('--seed',type=int,choices=[1701,1702]);parser.add_argument('--sequence',choices=['zgz_local','zgz_global']);args=parser.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='extract':extract(args.shard)
    elif args.command=='fit':fit(args.seed)
    elif args.command=='summarize':summarize()
    elif args.command=='prepare-rollout':prepare_rollout()
    elif args.command=='rollout':rollout(args.sequence)
    elif args.command=='supplemental':supplemental()
    else:self_test()
