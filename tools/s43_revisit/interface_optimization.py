"""Fixed CPU checkpoint diagnostic authorized by Stage1 optimization extension."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from tools.s43_rel.mean_shape_replay import MeanShapeInputReader,PUBLIC_INITIAL,_mano,fk
from tools.s43_rel.model import load_relation_model
from tools.s43_rel.replay import CausalRelationTracker
from tools.s43_rel.core import RelConfig
from tools.s43_rel.geometry import GeometryConfig,rotation_exp
from tools.s43_rel.data import last_lnes
from tools.s43_rel.training import _source_files
from .interface_audit import guards

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/s43_revisit_20261028'
FINGERS={'thumb':[1,2,3,4],'index':[5,6,7,8],'middle':[9,10,11,12],'ring':[13,14,15,16],'pinky':[17,18,19,20]}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def raw_target(base,ends):
    path=Path(str(base)+'.meta')
    with path.open('rb') as f:n,=struct.unpack('<i',f.read(4))
    if n!=51:raise ValueError('target shape')
    dt=np.dtype([('data','<f8',51),('m0','u1'),('m1','u1')]);x=np.memmap(path,dtype=dt,mode='r',offset=4)[ends]
    if np.any(x['m0']!=4) or np.any(x['m1']!=13):raise ValueError('invalid score marker')
    return np.asarray(x['data'],np.float32).copy()
def metrics(mano,pred,truth,beta):
    pj,_=fk(mano,pred,beta);gj,_=fk(mano,truth,beta)
    absolute=np.linalg.norm(pj-gj,axis=-1)*1000.;aligned=np.linalg.norm((pj-pj[:,:1])-(gj-gj[:,:1]),axis=-1)*1000.
    Rp,Rg=rotation_exp(pred[:,3:6]),rotation_exp(truth[:,3:6])
    values={'MPJPE_abs_mm':absolute.mean(1),'MPJPE_J0_mm':aligned.mean(1),'wrist_abs_mm':absolute[:,0],
        'translation_parameter_error_mm':np.linalg.norm(pred[:,:3]-truth[:,:3],axis=1)*1000.,
        'root_rotation_error_deg':Rotation.from_matrix(Rp.transpose(0,2,1)@Rg).magnitude()*180/np.pi,
        'nonwrist_J0_mm':aligned[:,1:].mean(1)}
    for finger,ids in FINGERS.items():values[finger+'_abs_mm']=absolute[:,ids].mean(1);values[finger+'_J0_mm']=aligned[:,ids].mean(1)
    return values

def run(protocol,extension,out):
    torch.set_num_threads(1);guards(protocol)
    ext=json.loads(Path(extension).read_text())
    if ext['parent_sha256']!=sha(protocol):raise RuntimeError('extension binding mismatch')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'results.json').exists() or (out/'predictions_before_scoring.npz').exists():raise FileExistsError(out)
    sources=_source_files();config=Path(ext['config']['path'])
    if sha(config)!=ext['config']['sha256']:raise RuntimeError('config changed')
    models={};configs={}
    for label,checkpoint in [('canonical',None)]+[(f's43_step{s}',ext['checkpoints'][str(s)]['path']) for s in (500,2000,6000)]:
        if checkpoint and sha(checkpoint)!=ext['checkpoints'][label.replace('s43_step','')]['sha256']:raise RuntimeError('checkpoint changed')
        models[label],configs[label]=load_relation_model(config,checkpoint,device='cpu');models[label].eval()
    canonical=models['canonical'].base
    records=[];predictions=[];input_metadata={};betas={}
    with torch.inference_mode():
        for choice in ext['selection']:
            name=choice['recording'];reader=MeanShapeInputReader(name)
            # Explicitly known-shape conditional diagnostic. No pose is opened.
            with np.load(str(reader.base)+'_aux.npz',allow_pickle=False) as aux:beta=np.asarray(aux['betas'],np.float32).reshape(10).copy()
            betas[name]=beta;start,stop=choice['analyzed_prefix'];warm_states=[];state=PUBLIC_INITIAL.copy()
            for a in (start,start+50):
                observed=reader.packet(a,a+50)['events'];image=last_lnes(observed,a,a+50)
                state=canonical(torch.tensor(image[None]),torch.tensor(state[None]),torch.tensor(beta[None]),torch.tensor(reader.K_cam[None]))[0].float().numpy()
                warm_states.append(state.copy())
            input_metadata[name]={'source_base':str(reader.base),'beta_given':True,'beta_source':'aux betas, conditional diagnostic',
                'source_rows':[int(reader.offsets[start]),int(reader.offsets[stop])],
                'warm_states_sha256':hashlib.sha256(np.stack(warm_states).tobytes()).hexdigest(),'target_pose_read_before_prediction_save':False}
            for label,model in models.items():
                cfg=configs[label];tracker=CausalRelationTracker(model,RelConfig(**cfg.get('MEMORY',{})),GeometryConfig(**cfg.get('GEOMETRY',{})))
                tracker.reset(f'{name}:{label}',start,PUBLIC_INITIAL,beta,reader.K_cam)
                ordinal=0;count=0
                for cursor in range(start,stop,50):
                    cutoff=min(cursor+50,stop)
                    for ms in range(cursor,cutoff):
                        raw=reader.packet(ms,ms+1);tracker.ingest(raw['events'],watermark_ms=ms+1,packet_ordinal=ordinal,source_range=raw['source_range'])
                        ordinal+=1;count+=len(raw['events'])
                    if cutoff-start in (50,100):tracker.canonical_commit(cutoff,warm_states[(cutoff-start)//50-1]);continue
                    if cutoff-cursor<50:continue
                    result=tracker.query(cutoff);tracker.commit(result)
                    records.append({'recording':name,'model':label,'run_id':choice['run_id'],'end_ms':cutoff-1,
                        'second_half':cutoff>start+1000,'head_partition':choice['mechanism_head_partition']})
                    predictions.append(result['state'])
                expected=int(reader.offsets[stop])-int(reader.offsets[start])
                if count!=expected:raise AssertionError('prefix source count mismatch')
                tracker.release();print(json.dumps({'recording':name,'model':label,'source_events':count}),flush=True)
    pred=np.asarray(predictions,np.float32).reshape(-1,51)
    prediction_path=out/'predictions_before_scoring.npz'
    np.savez_compressed(prediction_path,pred=pred,recording=np.array([r['recording'] for r in records]),
        model=np.array([r['model'] for r in records]),end_ms=np.array([r['end_ms'] for r in records]),
        second_half=np.array([r['second_half'] for r in records]))
    write(out/'input_addresses_before_scoring.json',{'records':records,'inputs':input_metadata,'extension_sha256':sha(extension)})
    # First pose-target access after every one of the fixed16 streams finished.
    mano=_mano();result={};per_values={};truth_all=np.empty_like(pred)
    for choice in ext['selection']:
        name=choice['recording'];mask=np.array([r['recording']==name for r in records]);idx=np.flatnonzero(mask)
        ends=np.array([records[i]['end_ms'] for i in idx]);truth=raw_target(choice['source_base'],ends);truth_all[idx]=truth
        values=metrics(mano,pred[idx],truth,betas[name]);per_values[name]=values;result[name]={}
        for label in models:
            local=np.array([records[i]['model']==label for i in idx]);half=np.array([records[i]['second_half'] for i in idx])
            result[name][label]={}
            for scope,keep in [('whole_scored_prefix',local),('second_half',local&half)]:
                result[name][label][scope]={'n':int(keep.sum()),**{k:float(v[keep].mean()) for k,v in values.items()}}
    np.savez_compressed(out/'scored_predictions.npz',pred=pred,gt=truth_all)
    guards(protocol)
    unchanged={p:sha(p)==h for p,h in sources.items()}
    if not all(unchanged.values()):raise RuntimeError('frozen source changed')
    report={'schema':'s43_revisit_O1_fixed_checkpoint_conditional_diagnostic_v1','protocol_sha256':sha(protocol),
        'extension_sha256':sha(extension),'source_sha256':sha(__file__),'prediction_archive_sha256':sha(prediction_path),
        'shape_condition':'known aux beta conditional diagnostic, not deployable mean-shape result',
        'all_predictions_saved_before_pose_GT':True,'results':result,'sources_unchanged':unchanged,
        'checkpoint_selection_from_scores':False,'zgz_accessed':False,'new_training':False,
        'limitations':['All4 recordings were in original/full9-subject training; SELECT is only diagnostic-head partition.',
            'These fixed prefixes assess fitting/cold-start/selfhistory sensitivity, not unseen-domain overfit causation.',
            'No checkpoint is selected from the new scores. No inference about adequate optimization follows solely from a flat or nonsignificant difference.']}
    write(out/'results.json',report);print(json.dumps({'complete':True,'out':str(out/'results.json')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,default=OUT/'stage1_protocol.json')
    p.add_argument('--extension',type=Path,default=OUT/'stage1_optimization_extension.json')
    p.add_argument('--out',type=Path,default=OUT/'interface_optimization');a=p.parse_args();run(a.protocol,a.extension,a.out)
