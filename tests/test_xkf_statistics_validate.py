"""Adversarial provenance and held-out-family checks for the new validator."""
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'model')]
import numpy as np
import pytest
from tools.xkf_statistics import validate as V

@pytest.fixture
def evidence(tmp_path):
 rows=[];indices=[]
 for person in ('ch','lfz','lpc','lr','ly','lyh','lyq','ycy','ylf'):
  for movement in ('local','global'):
   for version in ('','_v2','_v3','_v4'):
    for j in range(32):
     if j>=4:indices.append(len(rows))
     rows.append({'sequence':f'{person}_{movement}{version}'})
 rng=np.random.default_rng(13);ei=rng.normal(size=(2016,51));ed=rng.normal(size=(2016,51));gate=rng.uniform(size=2016);speed=rng.uniform(0,1000,size=2016)
 rawpath=tmp_path/'raw.npz';np.savez(rawpath,row_indices=indices,sequence=[rows[i]['sequence'] for i in indices],e_i=ei,e_d=ed,gate=gate,gt_joint_speed_mm_s=speed)
 source=tmp_path/'producer.py';source.write_text('fixed producer\n')
 resultpath=tmp_path/'result.json';V.write(resultpath,dict(raw_sha256=V.sha(rawpath),partial_smoke=False,no_development_data_loaded=True,all_training_data_file_stats_unchanged=True))
 identity=dict(seed=3407,global_step=2600,checkpoint_sha256='checkpoint',config_sha256='config')
 declaration=dict(fit_row_indices=indices,models={'3407':identity.copy()},fitting=dict(denominator_floor=1e-18,psd_tolerance=1e-12))
 summary=dict(status='COMPLETE',partial_smoke=False,rows=2016,actual_forward_calls=8064,all_buffers_unchanged=True,seed=3407,declaration_sha256='decl',identity=identity,source_sha256={str(source):V.sha(source)},input_sha256={},raw_path=str(rawpath),raw_sha256=V.sha(rawpath),result_path=str(resultpath),result_sha256=V.sha(resultpath),baseline_effective_gains=[.1]*19)
 collection=tmp_path/'summary.json';V.write(collection,summary)
 ids=np.array([rows[i]['sequence'].split('_')[0] for i in indices]);families=np.array([rows[i]['sequence'].split('_v')[0] for i in indices]);ix=np.asarray(indices)
 def fit(mask,name):
  value=V.fit_paired_gains(ei[mask],ed[mask],gate[mask],[.1]*19,**declaration['fitting'])
  value.update(index_space='canonical2304',train_indices=ix[mask].tolist(),excluded_indices=ix[~mask].tolist(),fold_family=name,high_motion_threshold_mm_s=float(np.quantile(speed[mask],.75)))
  return value
 family={name:fit(families!=name,name) for name in sorted(set(families))}
 folds=dict(kind='PAIRED_STATISTICS_CROSSFIT_CANDIDATES',status='FITTED_NOT_ACCEPTED',accepted_for_dev=False,automatic_full_training=False,seed=3407,declaration_sha256='decl',collection_summary_path=str(collection),collection_summary_sha256=V.sha(collection),collection_raw_path=str(rawpath),collection_raw_sha256=V.sha(rawpath),source_sha256={str(source):V.sha(source)},input_sha256={},baseline_effective_gains=[.1]*19,identity_folds=9,motion_family_folds=18,sequence_lookup_aliases=72,identity={name:fit(ids!=name,name) for name in sorted(set(ids))},motion_family=family,trial={row['sequence']:family[row['sequence'].split('_v')[0]] for row in rows})
 foldpath=tmp_path/'folds.json';V.write(foldpath,folds)
 return declaration,rows,foldpath,folds,collection,summary,source

def test_full_family_held_out_fits_are_verified(evidence):
 declaration,rows,path,*_=evidence
 value,hashes=V.load_folds(path,declaration,'decl',3407,rows)
 assert len(value['identity'])==9 and len(value['motion_family'])==18 and len(value['trial'])==72
 assert len(hashes)>=5

@pytest.mark.parametrize('attack',('family_leak','forged_coefficients','partial_collection','source_drift'))
def test_false_complete_or_leaky_evidence_is_rejected(evidence,attack):
 declaration,rows,path,folds,collection,summary,source=evidence
 if attack=='family_leak':
  fold=folds['trial']['ch_local'];held=fold['excluded_indices'].pop();fold['train_indices'].append(held)
 elif attack=='forged_coefficients':folds['identity']['ch']['coefficients'][0]=.999
 elif attack=='partial_collection':
  summary['partial_smoke']=True;V.write(collection,summary);folds['collection_summary_sha256']=V.sha(collection)
 else:source.write_text('changed producer\n')
 V.write(path,folds)
 with pytest.raises(ValueError):V.load_folds(path,declaration,'decl',3407,rows)

def test_short_main_and_flow_projection_use_actual_intervals():
 t=4;truth=np.zeros((t,21,3));truth[:,0,0]=[1,2,3,4]
 predicted=np.broadcast_to(truth[:,None,None],(t,1,3,21,3)).copy()*2
 raw=dict(end_ms=np.arange(t),metrics=np.ones((t,1,3,len(V.R.METRICS))),joint_motion_error_mm=predicted-truth[:,None,None],joint_motion_mm=predicted,true_joint_motion_mm=truth,true_joint_speed_mm_s=np.array([20,40,60,80]))
 clip=dict(sequence='ch_local',training_identity='ch');folds={name:dict(high_motion_threshold_mm_s=50) for name in V.MODES[1:]}
 records=V.clip_records(clip,raw,[9],['x']*9+['small_gaussian'],folds)
 main=[r for r in records if r['window']=='main_onward']
 assert len(records)==6 and all(r['censored'] and not r['full10'] for r in records)
 assert all(r['scored_steps']==1 and r['joint_motion_projection']==2 for r in main)
 assert all(r['high_motion']['identity_cv']['n']==1 for r in main)
