from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'model')]
import torch,pytest
from semkine.xkf_motion_gate import MotionCountFusionPolicy,interpolate_motion
from semkine.xkf_centroid import CentroidFusionPolicy
from semkine.xkf_calibrated_motion_gate import CalibratedMotionCountPolicy
SPEC=dict(id='p',gain_scale=[.5,.25,.5],count_gate='ramp1000',translation_reference='joint_centroid',motion_count_gate='ramp1000')
def fk(x,b):
 j=x[:,None,:3]+torch.arange(63).reshape(1,21,3)/1000+x[:,None,3:6]*.05
 return j,j

def values():
 torch.manual_seed(71);p=torch.randn(3,51)*.03;i=p+torch.randn(3,51)*.02;d=p+torch.randn(3,51)*.04;p[:,:3]+=.6;i[:,:3]+=.6;d[:,:3]+=.6
 return p,i,d,torch.tensor([0,200,2000]),torch.full((19,),.3),torch.zeros(3,10)

def test_n1000_exact_original_and_trace():
 p,i,d,c,k,b=values();old=MotionCountFusionPolicy(SPEC).bind_geometry(fk,b);new=CalibratedMotionCountPolicy(dict(SPEC,motion_count_threshold=1000.)).bind_geometry(fk,b)
 new.capture=True
 for a,z in zip(new.fuse(i,d,{},p,c,k),old.fuse(i,d,{},p,c,k)):assert torch.equal(a,z)
 assert len(new.trace)==1;assert torch.equal(new.trace[0]['motion'],i)
 assert torch.equal(new.trace[0]['gated_motion'][0],p[0]);assert torch.equal(new.trace[0]['gated_motion'][2],i[2])

def test_only_motion_scale_changes_not_D_gain():
 p,i,d,c,k,b=values();n=4000.;new=CalibratedMotionCountPolicy(dict(SPEC,motion_count_threshold=n)).bind_geometry(fk,b)
 result,zd,gains=new.fuse(i,d,{},p,c,k)
 old=CentroidFusionPolicy({key:v for key,v in SPEC.items() if key!='motion_count_gate'}).bind_geometry(fk,b)
 expected=old.fuse(interpolate_motion(p,i,(c/n).clamp(0,1)),d,{},p,c,k)
 for a,z in zip((result,zd,gains),expected):assert torch.equal(a,z)
 assert torch.equal(result[0],p[0]);assert torch.equal(gains,old.gains(k,c))

@pytest.mark.parametrize('threshold',[0.,-1.,True,float('nan'),float('inf')])
def test_bad_threshold_rejected(threshold):
 with pytest.raises(ValueError):CalibratedMotionCountPolicy(dict(SPEC,motion_count_threshold=threshold))

def test_frozen_policy_contract():
 for changed in [dict(count_gate=None),dict(gain_scale=[1,.25,.5]),dict(translation_reference='state_t'),dict(direct_correction={'kind':'identity'}),dict(temperature_multiplier=2),dict(update_count_gate='ramp1000')]:
  with pytest.raises(ValueError):CalibratedMotionCountPolicy(dict(SPEC,**changed))

def _fake_runtime(tmp_path,fit_patch):
 import json,hashlib
 def write(name,value):
  p=tmp_path/name;p.write_text(json.dumps(value));return p,hashlib.sha256(p.read_bytes()).hexdigest()
 sources={}
 for name in ('a.json','b.json','c.json'):
  p,h=write(name,{});sources[str(p)]=h
 prereg,ph=write('prereg.json',dict(read_development_data=False,rows_per_seed=2304,conditions=10))
 fit=dict(kind='TRAIN_ONLY_I_RAMP_FIT',accepted_for_dev=True,status='ACCEPTED_FOR_DEV_EVALUATION',selected_family='shared',thresholds={'3407':2000.,'3408':2000.,'3409':2000.},training_source_sha256=sources,prereg_sha256=ph,metrics={'shared':{'eligible':True}})
 fit.update(fit_patch);fp,fh=write('fit.json',fit)
 runtime=dict(kind='motion_count_ramp',threshold=2000.,seed=3407,checkpoint_sha256='checkpoint',fit_path=str(fp),fit_sha256=fh,prereg_path=str(prereg),prereg_sha256=ph,training_source_sha256=sources,selection_family='shared',accepted_for_dev=True)
 return write('runtime.json',runtime)[0]

@pytest.mark.parametrize('patch',[dict(accepted_for_dev=False),dict(status='TRAIN_EVIDENCE_NOT_SUPPORTIVE'),dict(selected_family='fixed1000'),dict(thresholds={'3407':3000.,'3408':2000.,'3409':2000.}),dict(metrics={'shared':{'eligible':False}})])
def test_runtime_cannot_override_rejected_or_different_fit(tmp_path,patch):
 from tools.xkf_fusion.evaluate_calibrated_motion import load_runtime
 path=_fake_runtime(tmp_path,patch)
 with pytest.raises(ValueError,match='actual accepted fit evidence'):
  load_runtime(path,3407,'checkpoint')
