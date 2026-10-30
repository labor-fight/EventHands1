from pathlib import Path
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'model')]
import torch,pytest
from semkine.lie import so3_exp
from semkine.xkf_centroid import CentroidFusionPolicy
from semkine.xkf_motion_gate import MotionCountFusionPolicy,interpolate_motion

def states():
    torch.manual_seed(723);a=torch.randn(3,51)*.03;b=torch.randn(3,51)*.03
    a[:,:3]+=.5;b[:,:3]+=.6;a[:,3:6]=0;b[:,3:6]=torch.tensor([2.,0,0]);return a,b

def fk(x,b):
    j=x[:,None,:3]+torch.arange(63).reshape(1,21,3)/1000+x[:,None,3:6]*.05
    return j,j

def test_exact_endpoints_and_no_root_clip():
    a,b=states();g=torch.tensor([0.,.5,1.]);v=interpolate_motion(a,b,g)
    assert torch.equal(v[0],a[0]);assert torch.equal(v[2],b[2])
    assert torch.allclose(so3_exp(v[1,3:6]),so3_exp(torch.tensor([1.,0,0])),atol=1e-6)
    assert torch.equal(interpolate_motion(a,b,torch.ones(3)),b)
    assert torch.equal(interpolate_motion(a,b,torch.zeros(3)),a)

@pytest.mark.parametrize('reference',['state_t','joint_centroid'])
def test_disabled_exact_legacy_and_gated_endpoints(reference):
    a,b=states();d=b*.9;k=torch.full((19,),.2);beta=torch.zeros(3,10);counts=torch.tensor([0,1000,3000])
    spec=dict(id='p',translation_reference=reference,count_gate='ramp1000')
    old=CentroidFusionPolicy(spec).bind_geometry(fk,beta);disabled=MotionCountFusionPolicy(spec).bind_geometry(fk,beta)
    actual=MotionCountFusionPolicy(dict(spec,motion_count_gate='ramp1000')).bind_geometry(fk,beta)
    baseline=old.fuse(b,d,{},a,counts,k)
    for got,wanted in zip(disabled.fuse(b,d,{},a,counts,k),baseline):assert torch.equal(got,wanted)
    result=actual.fuse(b,d,{},a,counts,k)[0]
    assert torch.equal(result[0],a[0]);assert torch.equal(result[1:],baseline[0][1:])

def test_motion_and_whole_update_are_distinct_and_declared():
    a,b=states();d=b*.9;k=torch.full((19,),.2);beta=torch.zeros(3,10);counts=torch.full((3,),100.)
    common=dict(id='p',translation_reference='joint_centroid')
    m=MotionCountFusionPolicy(dict(common,count_gate='ramp1000',motion_count_gate='ramp1000')).bind_geometry(fk,beta)
    w=MotionCountFusionPolicy(dict(common,update_count_gate='ramp1000')).bind_geometry(fk,beta)
    assert not torch.equal(m.fuse(b,d,{},a,counts,k)[0],w.fuse(b,d,{},a,counts,k)[0])
    with pytest.raises(ValueError):MotionCountFusionPolicy(dict(common,count_gate='ramp1000',update_count_gate='ramp1000'))
    with pytest.raises(ValueError):MotionCountFusionPolicy(dict(common,motion_count_gate='ramp1000',update_count_gate='ramp1000'))
