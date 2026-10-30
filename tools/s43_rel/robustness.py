"""Final-only common-context perturbations and posthoc nominal motion strata.

This is a diagnostic reader of completed W50/hop50 trajectories.  Nominal
paired-S43 predictions provide the shared deployed previous state; GT is not
opened until every diagnostic prediction has been persisted.  Diagnostic
queries never commit new feedback.  All four arms share the same sensor
prefix and previous context; fresh versus stale changes only the typed
geometry package within a given trained arm.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time

import numpy as np
import scipy
from scipy.spatial.transform import Rotation
import torch

from tools.s43_br.data import DATA_ROOT, PUBLIC_INITIAL_STATE, SequenceReader
from .core import RelConfig, RelationMemory
from .data import ContinuousRunLanes, last_lnes, representation_from_snapshot
from .geometry import GeometryConfig, ManoGeometry, event_intrinsics, rotation_exp
from .model import ARMS, load_relation_model
from .replay import gpu_sync, validate_device

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/s43_rel_20261027'
CONDITIONS=('none','root_x_5mm','root_x_15mm','index_MCP_right_z_0.1rad','index_MCP_right_z_0.3rad')
FINGERS={'thumb':(1,2,3,4),'index':(5,6,7,8),'middle':(9,10,11,12),
         'ring':(13,14,15,16),'pinky':(17,18,19,20)}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def arrayhash(*arrays):
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a);h.update(str((a.dtype.str,a.shape)).encode());h.update(a.tobytes())
    return h.hexdigest()


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False,ensure_ascii=False)+'\n')


def fixed_indices(n):
    """Exactly the declared evenly spaced 1..n-1 indices, no outcome filtering."""
    return np.unique(np.linspace(1,n-1,min(8,n-1)).astype(np.int64)) if n>1 else np.empty(0,np.int64)


def _mean_and_index():
    with np.load(ROOT/'assets/mano_right.npz',allow_pickle=False) as z:
        mean=z['hands_mean'].astype(np.float64).reshape(15,3)
        index_mcp=int(z['mano16_tips_to_openpose21'][5])
    if index_mcp!=1:raise ValueError('frozen MANO index-MCP mapping changed')
    return mean,index_mcp


def continuous_log(matrix,reference):
    """SO(3) log nearest the prior chart, robust through the pi branch."""
    w=Rotation.from_matrix(np.asarray(matrix,np.float64)).as_rotvec()
    angle=np.linalg.norm(w)
    if angle<1e-12:return w
    axis=w/angle
    candidates=np.stack([w+2*np.pi*k*axis for k in (-2,-1,0,1,2)])
    return candidates[np.argmin(np.linalg.norm(candidates-np.asarray(reference)[None],axis=1))]


def perturb_previous(previous,condition):
    p=np.asarray(previous,np.float64)
    if p.shape!=(51,) or not np.isfinite(p).all():raise ValueError('finite deployed previous51 required')
    out=p.copy()
    if condition=='none':return out.astype(np.float32)
    if condition in ('root_x_5mm','root_x_15mm'):
        out[0]+=.005 if condition=='root_x_5mm' else .015
    elif condition in ('index_MCP_right_z_0.1rad','index_MCP_right_z_0.3rad'):
        mean,joint=_mean_and_index();sl=slice(3+3*joint,6+3*joint)
        actual=p[sl]+mean[joint-1]
        angle=.1 if condition.endswith('0.1rad') else .3
        target=rotation_exp(actual)@rotation_exp(np.array([0.,0.,angle]))
        out[sl]=continuous_log(target,actual)-mean[joint-1]
        if not np.allclose(rotation_exp(out[sl]+mean[joint-1]),target,atol=1e-12,rtol=0):
            raise ArithmeticError('right-SO3 residual chart roundtrip failed')
    else:raise ValueError(condition)
    return out.astype(np.float32)


def trace_inputs(path):
    """Read only deployed predictions and addresses. Never load gt here."""
    with np.load(path,allow_pickle=False) as z:
        data={key:z[key].copy() for key in ('pred','end_ms','run_id')}
    n=len(data['pred'])
    if data['pred'].shape!=(n,51) or any(data[k].shape!=(n,) for k in ('end_ms','run_id')):
        raise ValueError('nominal trace shape mismatch')
    if not np.isfinite(data['pred']).all():raise ValueError('nonfinite nominal deployed prediction')
    return data


def trace_path(folder,arm,recording):return Path(folder)/f'{arm}_{recording}_h50.npz'


def _gate(configs,checkpoints,nominal,recordings):
    """Final weights and complete reports must exist before final-only access."""
    paths={};receipts={}
    for arm in ARMS:
        cfg=Path(configs)/f'{arm}.yaml';ckpt=Path(checkpoints)/arm/'last.ckpt'
        complete=ckpt.with_name('complete.json')
        receipt=json.loads(complete.read_text())
        if receipt.get('stage')!='full' or receipt.get('completed_steps')!=6000:
            raise RuntimeError('robustness requires completed final 6000-step weights')
        if receipt.get('checkpoint_sha256')!=sha(ckpt):raise RuntimeError('completed checkpoint identity mismatch')
        paths[arm]=(cfg,ckpt);receipts[arm]={'checkpoint':str(ckpt),'checkpoint_sha256':sha(ckpt),
                                          'config':str(cfg),'config_sha256':sha(cfg)}
        for recording in recordings:
            p=trace_path(nominal,arm,recording)
            report=json.loads(p.with_suffix('.json').read_text())
            if not (report.get('full_recording') and report.get('all_valid_events_consumed_once') and
                    report.get('hop_ms')==50 and report.get('commit_schedule_ms')==50):
                raise RuntimeError('complete nominal W50/hop50 report required')
            if Path(report['checkpoint']).resolve()!=ckpt.resolve():
                raise RuntimeError('nominal trajectory belongs to another checkpoint')
            if Path(report['config']).resolve()!=cfg.resolve():raise RuntimeError('nominal config identity mismatch')
            frozen=report.get('source_files',{}).get(str(ckpt.resolve()),{})
            if frozen.get('sha256')!=receipts[arm]['checkpoint_sha256']:
                raise RuntimeError('nominal trajectory was evaluated with different checkpoint bytes')
            receipts[arm].setdefault('nominal',{})[recording]={'path':str(p),'sha256':sha(p),
                'report_sha256':sha(p.with_suffix('.json'))}
    return paths,receipts


def snapshot_for_binding(memory,cutoff,binding,arm):
    if arm=='full':return memory.readout(cutoff,edge_ids=binding.edge_ids)
    if arm!='no_order':raise ValueError(arm)
    reverse=memory.topology.reverse[binding.edge_ids]
    union=np.union1d(binding.edge_ids,reverse)
    snap=memory.readout(cutoff,edge_ids=union)
    direct,back=np.searchsorted(union,binding.edge_ids),np.searchsorted(union,reverse)
    # The frozen arm is spatial symmetrization at fixed p/q, not physical F.
    return replace(snap,Z=.5*(snap.Z[direct]+snap.Z[back]),edge_ids=binding.edge_ids)


def evidence_hash(memory,cutoff):
    snapshot=memory.readout(cutoff)
    return arrayhash(snapshot.A,snapshot.Z)


@torch.inference_mode()
def prediction(model,rep,previous,beta,K,device):
    tensor=lambda value:torch.as_tensor(np.asarray(value,np.float32)[None],device=device)
    value=model([rep],tensor(previous),tensor(beta),tensor(K))[0].float()
    gpu_sync(device)
    result=value.cpu().numpy().copy()
    if result.shape!=(51,) or not np.isfinite(result).all():raise ArithmeticError('nonfinite diagnostic prediction')
    return result


def collect_recording(recording,nominal,models,cfg,device):
    traces={arm:trace_inputs(trace_path(nominal,arm,recording)) for arm in ARMS}
    reference=traces['s43']
    if not all(np.array_equal(v[k],reference[k]) for v in traces.values() for k in ('end_ms','run_id')):
        raise ValueError('paired nominal trace addresses differ')
    selected=fixed_indices(len(reference['pred']))
    selections=[]
    for i in selected:
        valid=bool(reference['run_id'][i]==reference['run_id'][i-1] and
                   reference['end_ms'][i]-reference['end_ms'][i-1]==50)
        selections.append({'index':int(i),'end_ms':int(reference['end_ms'][i]),
                           'run_id':int(reference['run_id'][i]),'eligible':valid,
                           'reason':None if valid else 'cross-run or non50ms pair; no replacement'})
    reader=SequenceReader(recording,Path(cfg['REL'].get('DATA_ROOT',DATA_ROOT)))
    memory=RelationMemory(RelConfig(**cfg.get('MEMORY',{})),seq_id=recording)
    provider=ManoGeometry(GeometryConfig(**cfg.get('GEOMETRY',{})))
    rows=[];predictions=[];shared_prevs=[];perturbed_prevs=[];audits=[]
    current_run=None;cursor=None;ordinal=0;count=0;source_digest=hashlib.sha256()
    with torch.inference_mode():
        for choice in selections:
            if not choice['eligible']:continue
            index,run=choice['index'],choice['run_id'];cutoff=choice['end_ms']+1
            a,b=map(int,reader.runs[run])
            if not a+100<cutoff<=b:raise ValueError('selected query outside its valid causal run')
            if current_run!=run:
                memory.reset(f'{recording}:run{run}',boundary_ms=a)
                current_run,cursor,ordinal=run,a,0
            while cursor<cutoff:
                end=min(cursor+50,cutoff)
                raw=ContinuousRunLanes._packet(reader,cursor,end)
                memory.ingest(raw['events'],watermark_ms=end,packet_ordinal=ordinal,
                              source_range=raw['source_range'])
                source_digest.update(raw['events'].tobytes());count+=len(raw['events'])
                cursor,ordinal=end,ordinal+1
            shared=reference['pred'][index-1].copy()
            observed=ContinuousRunLanes._packet(reader,cutoff-50,cutoff)
            item=dict(observed,lnes=last_lnes(observed['events'],cutoff-50,cutoff),
                      seq_id=recording,start_ms=cutoff-50,cutoff_ms=cutoff,packet_ordinal=ordinal-1)
            before=evidence_hash(memory,cutoff)
            baseline=provider.bind(shared,reader.betas,event_intrinsics(reader.K_cam),memory.topology)
            for condition in CONDITIONS:
                prev=perturb_previous(shared,condition)
                fresh=baseline if condition=='none' else provider.bind(prev,reader.betas,event_intrinsics(reader.K_cam),memory.topology)
                # Actual render of the shared changed context is hashed once;
                # both geometry modes call the same trained base with this prev.
                render=models['full'].base._render_prev(torch.as_tensor(prev[None],device=device),
                    torch.as_tensor(reader.betas[None],device=device),torch.as_tensor(reader.K_cam[None],device=device))
                gpu_sync(device);render_digest=arrayhash(render.float().cpu().numpy())
                A=memory.readout(cutoff,edge_ids=[]).A
                for arm in ARMS:
                    modes=('fresh','stale') if arm in ('full','no_order') else ('not_applicable',)
                    pair={}
                    for mode in modes:
                        binding=fresh if mode!='stale' else baseline
                        snap=snapshot_for_binding(memory,cutoff,binding,arm) if arm in ('full','no_order') else memory.readout(cutoff,edge_ids=[])
                        geom=binding.read(snap) if arm in ('full','no_order') else None
                        rep=representation_from_snapshot(item,snap,arm,geom)
                        rep.update(delta_commit_ms=50.,output_lag_ms=1.,missing=False)
                        pred=prediction(models[arm],rep,prev,reader.betas,reader.K_cam,device)
                        pair[mode]=pred
                        rows.append({'recording':recording,'sample_index':index,'run_id':run,
                                     'end_ms':cutoff-1,'condition':condition,'arm':arm,'geometry_mode':mode,
                                     'common_previous_source_index':index-1,'delta_commit_ms':50.,
                                     'previous_source':'paired trained-S43 nominal deployed prediction',
                                     'network_previous_sha256':arrayhash(prev),
                                     'lnes_sha256':arrayhash(item['lnes']),'shared_render_sha256':render_digest,
                                     'sensor_A_sha256':arrayhash(A),'sensor_evidence_sha256':before,
                                     'geometry_version':binding.metadata['geometry_version'] if geom is not None else None})
                        predictions.append(pred);shared_prevs.append(shared);perturbed_prevs.append(prev)
                    if condition=='none' and len(pair)==2 and not np.array_equal(pair['fresh'],pair['stale']):
                        raise AssertionError('unperturbed fresh/stale parity failed')
            after=evidence_hash(memory,cutoff)
            if after!=before or memory.commit_count!=0:raise AssertionError('diagnostic query changed sensor state/feedback')
            audits.append({'sample_index':index,'cutoff_ms':cutoff,'A_Z_unchanged':True,
                           'A_Z_sha256':before,'feedback_commits':memory.commit_count,
                           'ingested_events_since_current_boundary':memory.stats()['events']})
    return {'rows':rows,'pred':np.asarray(predictions,np.float32).reshape(-1,51),
            'shared_prev':np.asarray(shared_prevs,np.float32).reshape(-1,51),
            'perturbed_prev':np.asarray(perturbed_prevs,np.float32).reshape(-1,51),
            'selections':selections,'audits':audits,'source':{'base':str(reader.base),
             'ingested_event_count_across_selected_runs':count,'ingested_event_payload_sha256':source_digest.hexdigest(),
             'context_betas_sha256':arrayhash(reader.betas),'context_K_cam_sha256':arrayhash(reader.K_cam)},
            'beta':reader.betas.copy()}


def _mano():
    name='_s43_rel_robustness_readonly_mano'
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,ROOT/'model/mano_layer.py')
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return sys.modules[name].ManoLayer(ROOT/'assets/mano_right.npz',add_mean=True).cpu().eval()


@torch.inference_mode()
def fk_joints(mano,poses,beta):
    poses=np.asarray(poses,np.float32).reshape(-1,51);result=[]
    for start in range(0,len(poses),256):
        p=torch.from_numpy(poses[start:start+256]);b=torch.from_numpy(np.asarray(beta,np.float32)[None]).expand(len(p),-1)
        _,j=mano(b,p[:,3:6],p[:,6:],p[:,:3]);result.append(j.numpy())
    return np.concatenate(result) if result else np.empty((0,21,3),np.float32)


def errors(predicted,truth):
    absolute=np.linalg.norm(predicted-truth,axis=-1)*1000.
    rp=predicted-predicted[:,:1];rt=truth-truth[:,:1]
    aligned=np.linalg.norm(rp-rt,axis=-1)*1000.
    return {'all_abs_mm':absolute.mean(1),'all_root_aligned_mm':aligned.mean(1),
            'index_abs_mm':absolute[:,FINGERS['index']].mean(1),
            'index_root_aligned_mm':aligned[:,FINGERS['index']].mean(1)}


def averages(values,mask):
    n=int(np.asarray(mask).sum())
    return {'n':n,**({k:float(v[mask].mean()) for k,v in values.items()} if n else
                     {'status':'no samples; evidence missing'})}


def nominal_motion_masks(gt,target,end_ms,run_id):
    """Pure posthoc thresholds, independent of prediction errors."""
    gt=np.asarray(gt);target=np.asarray(target)
    n=len(gt)
    if target.shape!=(n,21,3):raise ValueError('GT FK joint shape mismatch')
    relative=target-target[:,:1]
    valid=np.zeros(n,bool)
    valid[1:]=(np.diff(end_ms)==50)&(np.diff(run_id)==0)
    displacement=np.zeros((n,5))
    for f,ids in enumerate(FINGERS.values()):
        displacement[1:,f]=np.linalg.norm(relative[1:,ids]-relative[:-1,ids],axis=-1).mean(1)*1000.
    root_translation=np.zeros(n);root_rotation=np.zeros(n)
    root_translation[1:]=np.linalg.norm(np.diff(gt[:,:3],axis=0),axis=1)*1000.
    if n>1:
        rotations=rotation_exp(gt[:,3:6])
        root_rotation[1:]=Rotation.from_matrix(rotations[:-1].transpose(0,2,1)@rotations[1:]).magnitude()*180./np.pi
    quiet=(root_translation<2.)&(root_rotation<2.)
    max_finger=displacement.max(1);mean20=displacement.mean(1);index=displacement[:,1]
    masks={'all_outputs':np.ones(n,bool),'valid_50ms_pairs':valid,
           'local_finger_primary':valid&quiet&(max_finger>5.),
           'local_mean20_secondary':valid&quiet&(mean20>5.),
           'local_index_secondary':valid&quiet&(index>5.)}
    quantiles=np.quantile(max_finger[valid],[0,.25,.5,.75,1.]) if valid.any() else np.zeros(5)
    for k in range(4):
        masks[f'finger_motion_quartile_{k+1}']=valid&(max_finger>=quantiles[k])&(
            max_finger<=quantiles[k+1] if k==3 else max_finger<quantiles[k+1])
    return masks,displacement,root_translation,root_rotation,quantiles


def score_diagnostics(collected,nominal,out):
    """First semantic target access, after all query predictions are persisted."""
    mano=_mano();reports={};all_per=[]
    for recording,data in collected.items():
        with np.load(trace_path(nominal,'s43',recording),allow_pickle=False) as z:truth=z['gt'].copy()
        indices=np.array([r['sample_index'] for r in data['rows']],np.int64)
        target=truth[indices]
        J=fk_joints(mano,data['pred'],data['beta']);T=fk_joints(mano,target,data['beta'])
        values=errors(J,T)
        input_values=errors(fk_joints(mano,data['perturbed_prev'],data['beta']),T)
        rows=data['rows'];summary=[]
        for condition in CONDITIONS:
            for arm in ARMS:
                for mode in (('fresh','stale') if arm in ('full','no_order') else ('not_applicable',)):
                    mask=np.array([r['condition']==condition and r['arm']==arm and r['geometry_mode']==mode for r in rows])
                    summary.append(dict(condition=condition,arm=arm,geometry_mode=mode,
                                        output=averages(values,mask),perturbed_input=averages(input_values,mask)))
        contrast=[]
        for condition in CONDITIONS:
            for arm in ('full','no_order'):
                fresh=np.array([i for i,r in enumerate(rows) if r['condition']==condition and r['arm']==arm and r['geometry_mode']=='fresh'])
                stale=np.array([i for i,r in enumerate(rows) if r['condition']==condition and r['arm']==arm and r['geometry_mode']=='stale'])
                if len(fresh)!=len(stale):raise AssertionError('unpaired fresh/stale diagnostics')
                contrast.append({'condition':condition,'arm':arm,'n':len(fresh),
                    'fresh_minus_stale_mm':{k:float(np.mean(v[fresh]-v[stale])) if len(fresh) else None for k,v in values.items()}})
        np.savez_compressed(Path(out)/f'{recording}_perturbation_scores.npz',
                            **values,**{'input_'+k:v for k,v in input_values.items()})
        reports[recording]={'summaries':summary,'paired_geometry_contrasts':contrast,'fixed_indices':data['selections'],
                             'scope':'external common-deployed-context queries; not a new feedback trajectory or universal recovery basin'}
    return reports


def motion_strata(nominal,recordings,betas,out):
    """Posthoc GT strata on complete nominal outputs, never query selection."""
    mano=_mano();report={}
    for recording in recordings:
        ref=trace_inputs(trace_path(nominal,'s43',recording))
        with np.load(trace_path(nominal,'s43',recording),allow_pickle=False) as z:gt=z['gt'].copy()
        target=fk_joints(mano,gt,betas[recording])
        masks,displacement,root_translation,root_rotation,quantiles=nominal_motion_masks(gt,target,ref['end_ms'],ref['run_id'])
        arms={}
        for arm in ARMS:
            trace=trace_inputs(trace_path(nominal,arm,recording))
            with np.load(trace_path(nominal,arm,recording),allow_pickle=False) as z:arm_gt=z['gt']
            if not np.array_equal(arm_gt,gt):raise AssertionError('paired nominal GT differs')
            values=errors(fk_joints(mano,trace['pred'],betas[recording]),target)
            arms[arm]={name:averages(values,mask) for name,mask in masks.items()}
        np.savez_compressed(Path(out)/f'{recording}_nominal_motion_strata.npz',
            finger_displacement_mm=displacement,root_translation_mm=root_translation,
            root_rotation_deg=root_rotation,end_ms=ref['end_ms'],run_id=ref['run_id'],**masks)
        report[recording]={'thresholds':{'finger_mm_per_50ms':5.,'root_translation_mm':2.,'root_rotation_deg':2.},
            'primary_definition':'max over five fingers of mean wrist-relative FK displacement of its four OpenPose joints >5mm, quiet root; valid adjacent50ms same-run outputs only',
            'secondary_definitions':['mean20 nonwrist displacement >5mm','index four-joint mean displacement >5mm'],
            'primary_n':int(masks['local_finger_primary'].sum()),
            'evidence_status':'available' if masks['local_finger_primary'].any() else 'no samples; evidence missing',
            'motion_quantile_edges_mm':quantiles.tolist(),'arms':arms,
            'interpretation':'posthoc descriptive association, no causal claim or real high-frequency GT validation'}
    return report


def run(configs,checkpoints,nominal,recordings,out,device='cpu',threads=1):
    validate_device(device);torch.set_num_threads(int(threads))
    out=Path(out)
    if (out/'report.json').exists() or (out/'predictions_before_scoring.npz').exists():raise FileExistsError(out)
    paths,receipts=_gate(configs,checkpoints,nominal,recordings)
    out.mkdir(parents=True,exist_ok=True)
    sources={str(p):sha(p) for p in [Path(__file__),Path(__file__).with_name('core.py'),
        Path(__file__).with_name('core_cpu.cpp'),Path(__file__).with_name('geometry.py'),
        Path(__file__).with_name('geometry_cpu.cpp'),Path(__file__).with_name('data.py'),
        Path(__file__).with_name('model.py'),Path(__file__).with_name('joint_reader.py'),
        ROOT/'tools/s43_integration/model.py',ROOT/'model/model.py',ROOT/'model/pose_repr.py',
        ROOT/'model/mano_layer.py',ROOT/'assets/mano_right.npz']}
    manifest={'schema':'s43_rel_final_robustness_v1','inputs':receipts,'sources':sources,
        'recordings':list(recordings),'selection':'8 fixed evenly spaced indices in1..n-1; invalid cross-run pairs not replaced',
        'conditions':list(CONDITIONS),'previous_context':'paired trained-S43 nominal pred[index-1], shared across all four arms',
        'SO3':'index MCP MANO joint1, actual local AA=residual+mean once, R right Exp(z angle), nearest equivalent log chart minus same mean',
        'geometry_modes':'full/no_order fresh=perturbed prev geometry; stale=nominal shared prev geometry; same perturbed network prev, LNES, beta/K, delta_commit50ms',
        'sensor_history':'one shared sensor memory per current recording/run, continuous from valid boundary; no diagnostic commits',
        'GT_role':'GT arrays opened only after every perturbation prediction is persisted, then separate posthoc nominal strata',
        'motion_rule_before_readout':{'primary':'max over five four-joint wrist-relative means >5mm per50ms',
            'quiet_root_translation_mm':2.,'quiet_root_rotation_deg':2.,'secondary':['mean20>5mm','index4>5mm']},
        'device':device,'dtype':'FP32','self_feedback_diagnostic':False,'no_training':True,
        'versions':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'torch':torch.__version__}}
    write(out/'manifest_before_readouts.json',manifest)
    models={};configs={}
    for arm,(conf,ckpt) in paths.items():models[arm],configs[arm]=load_relation_model(conf,ckpt,device=device);models[arm].eval()
    config=configs['full']
    if any(configs[a].get('MEMORY',{})!=config.get('MEMORY',{}) for a in ARMS):raise ValueError('arms have different sensor dictionaries')
    if any(configs[a].get('GEOMETRY',{})!=config.get('GEOMETRY',{}) for a in ARMS):raise ValueError('arms have different query geometry configurations')
    for key in ('BASE_CHECKPOINT','BASE_CONFIG'):
        if any(configs[a]['REL'][key]!=config['REL'][key] for a in ARMS):raise ValueError('arms have different original MANO/render contexts')
    collected={recording:collect_recording(recording,nominal,models,config,device) for recording in recordings}
    rows=[];pred=[];previous=[];perturbed=[]
    for recording,data in collected.items():rows+=data['rows'];pred.extend(data['pred']);previous.extend(data['shared_prev']);perturbed.extend(data['perturbed_prev'])
    # No GT or GT-derived mask enters either of these persisted input outputs.
    np.savez_compressed(out/'predictions_before_scoring.npz',pred=np.asarray(pred,np.float32).reshape(-1,51),
                        shared_previous=np.asarray(previous,np.float32).reshape(-1,51),
                        perturbed_previous=np.asarray(perturbed,np.float32).reshape(-1,51))
    write(out/'prediction_addresses_before_scoring.json',rows)
    write(out/'sensor_prefix_audits.json',{r:{k:v for k,v in d.items() if k in ('selections','audits','source')} for r,d in collected.items()})
    diagnostic=score_diagnostics(collected,nominal,out)
    strata=motion_strata(nominal,recordings,{r:d['beta'] for r,d in collected.items()},out)
    unchanged={p:sha(p)==h for p,h in sources.items()}
    if not all(unchanged.values()):raise RuntimeError('robustness sources changed during execution')
    report={'schema':'s43_rel_final_robustness_results_v1','diagnostic_perturbations':diagnostic,
            'posthoc_complete_nominal_strata':strata,'source_files_unchanged':unchanged,
            'GT_input_used':False,'sensor_queries_committed':False,
            'limits':['8 fixed queries per recording, descriptive one-seed diagnostic.',
                      'Shared deployed prior controls context; this is not each arm own-feedback robustness.',
                      'Current-geometry read of old evidence remains approximate; no calibrated covariance is inferred.',
                      'GT strata are posthoc W50 pseudo-label descriptors, not high-frequency physical truth.']}
    write(out/'report.json',report)
    return report


def self_test(out):
    """Actual ch input geometry plus near-pi toy actions; no checkpoint/GT/score."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    mean,j=_mean_and_index();p=PUBLIC_INITIAL_STATE.astype(float)
    axis=np.array([.2,.3,.4]);axis/=np.linalg.norm(axis)
    p[6:9]=axis*(np.pi-1e-7)-mean[0]
    checks={}
    for name in CONDITIONS:
        q=perturb_previous(p,name)
        if name.startswith('index_'):
            angle=.1 if name.endswith('0.1rad') else .3
            expected=rotation_exp(p[6:9]+mean[0])@rotation_exp(np.array([0.,0.,angle]))
            error=float(np.max(np.abs(rotation_exp(q[6:9].astype(float)+mean[0])-expected)))
            if error>3e-7:raise AssertionError('FP32 near-pi right-action roundtrip')
            checks[name]={'matrix_roundtrip_max_abs':error,'mean_added_once':True}
        elif name.startswith('root_'):
            requested=.005 if name=='root_x_5mm' else .015
            error=abs(float(q[0]-p[0])-requested)
            if error>1e-8:raise AssertionError('translation perturbation')
            checks[name]={'translation_error_m':error}
    reader=SequenceReader('ch_local');mem=RelationMemory(RelConfig(),seq_id='selftest')
    mem.reset('selftest',boundary_ms=0);raw=ContinuousRunLanes._packet(reader,0,100)
    mem.ingest(raw['events'],watermark_ms=100,packet_ordinal=0,source_range=raw['source_range'])
    before=evidence_hash(mem,100);g=ManoGeometry();K=event_intrinsics(reader.K_cam)
    b=g.bind(PUBLIC_INITIAL_STATE,reader.betas,K,mem.topology)
    changed=g.bind(perturb_previous(PUBLIC_INITIAL_STATE,'root_x_5mm'),reader.betas,K,mem.topology)
    for arm in ('full','no_order'):
        fresh=changed.read(snapshot_for_binding(mem,100,changed,arm))
        stale=b.read(snapshot_for_binding(mem,100,b,arm))
        if not all(np.isfinite(v).all() for v in (fresh['f'],fresh['lam'],stale['f'],stale['lam'])):raise AssertionError('nonfinite packs')
        checks[arm]={'fresh_f_norm':float(np.linalg.norm(fresh['f'])),'stale_f_norm':float(np.linalg.norm(stale['f'])),
                     'sensor_A_Z_unchanged':before==evidence_hash(mem,100),'feedback_commits':mem.commit_count}
    # Known toy translations validate scorer units and wrist alignment without
    # computing a task score from the existing recording.
    toy=np.zeros((2,21,3));shifted=toy.copy();shifted[:,:,0]=.001
    toy_errors=errors(shifted,toy)
    if not (np.allclose(toy_errors['all_abs_mm'],1.) and np.allclose(toy_errors['all_root_aligned_mm'],0.)):
        raise AssertionError('known toy scoring identity')
    poses=np.zeros((3,51));joints=np.zeros((3,21,3));joints[1:,FINGERS['index'],0]=.006
    poses[2,0]=.003
    masks,*_=nominal_motion_masks(poses,joints,np.array([149,199,249]),np.zeros(3,int))
    if masks['local_finger_primary'].tolist()!=[False,True,False] or masks['local_mean20_secondary'].any():
        raise AssertionError('known toy local-finger/root-motion strata')
    # Exercise every query-builder arm on real ch source events using a named
    # trivial model, not actual weights, targets, or accuracy measurements.
    class ToyBase(torch.nn.Module):
        def _render_prev(self,prev,beta,K):return torch.zeros((len(prev),180,240),device=prev.device)
    class ToyModel(torch.nn.Module):
        def __init__(self):super().__init__();self.base=ToyBase()
        def forward(self,reps,prev,betas=None,camera_K=None):
            if len(reps)!=len(prev):raise AssertionError('toy input batch mismatch')
            return prev.clone()
    with tempfile.TemporaryDirectory(prefix='s43_robustness_selftest_') as temp:
        for arm in ARMS:
            np.savez(Path(temp)/f'{arm}_ch_local_h50.npz',pred=np.tile(PUBLIC_INITIAL_STATE,(20,1)),
                end_ms=np.arange(149,1100,50),run_id=np.zeros(20,np.int64),gt=np.full((20,51),np.nan))
        collection=collect_recording('ch_local',temp,{arm:ToyModel() for arm in ARMS},
            {'REL':{'DATA_ROOT':str(DATA_ROOT)},'MEMORY':asdict(RelConfig()),'GEOMETRY':asdict(GeometryConfig())},'cpu')
        if len(collection['rows'])!=240 or not all(a['A_Z_unchanged'] for a in collection['audits']):
            raise AssertionError('all-arm real-source query collection contract')
    checks['all_arm_collection']={'actual_source':'ch_local','toy_forward_only':True,'queries':240,
        'selected_prefixes':len(collection['audits']),'all_A_Z_unchanged':True,
        'max_native_cutoff_ms':1100,'actual_ingested_events':collection['source']['ingested_event_count_across_selected_runs'],
        'gt_arrays_contain_NaN_and_are_not_read_by_input_collection':True}
    result={'schema':'robustness_implementation_selftest_v1','status':'PASS','source_sha256':sha(__file__),
            'recording':'ch_local','native_prefix_ms':100,'events':len(raw['events']),
            'fixed_indices_example_n20':fixed_indices(20).tolist(),'checks':checks,
            'targets_opened':False,'trained_model_or_checkpoint_used':False,'toy_model_used_for_input_routing':True,
            'zgz_accessed':False,'recorded_task_scores_computed':False}
    write(out/'self_test.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--configs-dir',type=Path,default=ROOT/'configs/s43_rel/r1')
    p.add_argument('--checkpoint-dir',type=Path,default=OUT/'train/full_r1')
    p.add_argument('--nominal-eval-dir',type=Path,default=OUT/'eval/full_final_h50')
    p.add_argument('--recordings',nargs='+',default=['zgz_local','zgz_global'])
    p.add_argument('--out',type=Path,default=OUT/'robustness/full_final');p.add_argument('--device',default='cpu')
    p.add_argument('--threads',type=int,default=1);p.add_argument('--self-test',action='store_true')
    a=p.parse_args()
    result=self_test(a.out) if a.self_test else run(a.configs_dir,a.checkpoint_dir,a.nominal_eval_dir,a.recordings,a.out,a.device,a.threads)
    print(json.dumps({'status':result.get('status','complete'),'out':str(a.out)},indent=2))


if __name__=='__main__':main()
