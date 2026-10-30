"""Target-free centroid fusion geometry and independent own-loop contracts."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'model')]
from semkine.xkf import fuse_states
from semkine.xkf_centroid import CentroidFusionPolicy, XKFCentroidModel, fuse_joint_centroids, joint_centroid_offsets
from semkine.xkf_fusion import FusionPolicy, XKFFusionModel
from tools.xkf_fusion import evaluate as Q
from tools.xkf_fusion.evaluate_centroid import rollout_with_geometry


def fk(state, beta):
    # Nonzero wrist and nontrivial all-joint centroid; both pose and beta matter.
    offsets = torch.arange(63, device=state.device).view(1, 21, 3) / 1000
    offsets = offsets + state[:, None, 3:6] * .08 + state[:, None, 6:9] * .02
    offsets = offsets + beta[:, None, :3] * .005
    joints = offsets + state[:, None, :3]
    return joints, joints


def values():
    generator = torch.Generator().manual_seed(73)
    a = torch.randn(2, 51, generator=generator) * .05
    b = torch.randn(2, 51, generator=generator) * .05
    a[:, :3] += torch.tensor([.01, .02, .6])
    b[:, :3] += torch.tensor([-.03, -.01, .8])
    beta = torch.tensor([[0.] * 10, [1.] * 10])
    k = torch.linspace(.1, .8, 19).repeat(2, 1)
    return a, b, beta, k


def centroid(state, beta, geometry=fk):
    return geometry(state, beta)[1].mean(1)


@pytest.mark.parametrize('translation_gain', [0., .3, 1.])
def test_centroid_anchor_with_independent_pose_gains(translation_gain):
    a, b, beta, k = values(); k[:, :3] = translation_gain
    actual = fuse_joint_centroids(a, b, k, beta, fk)
    wanted = (1-k[:, :3])*centroid(a,beta)+k[:, :3]*centroid(b,beta)
    assert torch.allclose(centroid(actual,beta), wanted, atol=1e-7)
    assert torch.equal(actual[:,3:],fuse_states(a,b,k)[:,3:])
    assert not torch.equal(actual[:,:3],fuse_states(a,b,k)[:,:3])


def test_zero_and_one_endpoints_and_mixed_beta():
    a,b,beta,k=values()
    assert torch.equal(fuse_joint_centroids(a,b,torch.zeros_like(k),beta,fk),a)
    assert torch.allclose(fuse_joint_centroids(a,b,torch.ones_like(k),beta,fk),b,atol=2e-7)
    batched=fuse_joint_centroids(a,b,k,beta,fk)
    separate=torch.cat([fuse_joint_centroids(a[i:i+1],b[i:i+1],k[i:i+1],beta[i:i+1],fk) for i in range(2)])
    assert torch.equal(batched,separate)


def test_anisotropic_gains_translation_equivariance_and_state_t_offset():
    a,b,beta,k=values();k[:,:3]=torch.tensor([.1,.4,.8])
    actual=fuse_joint_centroids(a,b,k,beta,fk)
    expected=(1-k[:,:3])*centroid(a,beta)+k[:,:3]*centroid(b,beta)
    assert torch.allclose(centroid(actual,beta),expected,atol=1e-7)
    shift=torch.tensor([.1,-.2,.3]);aa=a.clone();bb=b.clone();aa[:,:3]+=shift;bb[:,:3]+=shift
    shifted=fuse_joint_centroids(aa,bb,k,beta,fk)
    assert torch.allclose(shifted[:,:3],actual[:,:3]+shift,atol=1e-7)
    assert torch.equal(shifted[:,3:],actual[:,3:])
    assert torch.allclose(joint_centroid_offsets(a,beta,fk),centroid(a,beta)-a[:,:3],atol=1e-7)
    assert not torch.allclose(joint_centroid_offsets(a,beta,fk),centroid(a,beta)-fk(a,beta)[1][:,0])


def test_original_root_clipping_is_preserved():
    a,b,beta,k=values();a[:,3:6]=0.;b[:,3:6]=torch.tensor([2.,0.,0.]);k[:]=1.
    actual=fuse_joint_centroids(a,b,k,beta,fk)
    assert torch.equal(actual[:,3:],fuse_states(a,b,k)[:,3:])
    assert not torch.allclose(actual[:,3:6],b[:,3:6])
    assert torch.allclose(centroid(actual,beta),centroid(b,beta),atol=1e-7)


def test_empty_exact_hold_without_fk_and_mixed_empty():
    a,b,beta,k=values();prev=a+1
    policy=CentroidFusionPolicy({'id':'c','translation_reference':'joint_centroid'})
    actual,_,_=policy.fuse(a,b,{},prev,torch.zeros(2),k[0])
    assert torch.equal(actual,prev)
    policy.bind_geometry(fk,beta)
    actual,_,_=policy.fuse(a,b,{},prev,torch.tensor([0,20]),k[0])
    assert torch.equal(actual[0],prev[0])


def test_corrected_direct_used_consistently_and_no_target_key():
    a,b,beta,k=values()
    spec={'id':'c','translation_reference':'joint_centroid','direct_correction':
          {'kind':'translation_affine','matrix':torch.eye(3).tolist(),'bias':[.1,.2,.3]}}
    policy=CentroidFusionPolicy(spec).bind_geometry(fk,beta)
    result,direct,gain=policy.fuse(a,b,{},a,torch.ones(2),k[0])
    assert torch.allclose(direct[:,:3],b[:,:3]+b.new_tensor([.1,.2,.3]))
    assert torch.equal(result,fuse_joint_centroids(a,direct,gain,beta,fk))
    with pytest.raises(ValueError,match='unknown'):
        CentroidFusionPolicy(dict(spec,target=torch.zeros(51)))


def test_disabled_model_and_state_t_policy_exact_parity(monkeypatch):
    model=XKFCentroidModel.__new__(XKFCentroidModel);torch.nn.Module.__init__(model);model.configure_fusion(None)
    token=object();monkeypatch.setattr(XKFFusionModel,'forward_packet',lambda self,batch:token)
    assert model.forward_packet(None) is token and model.state_dict()=={}
    a,b,beta,k=values();counts=torch.tensor([100,1000]);spec={'id':'plain','count_gate':'ramp1000'}
    actual=CentroidFusionPolicy(spec).fuse(a,b,{},a,counts,k[0])
    expected=FusionPolicy(spec).fuse(a,b,{},a,counts,k[0])
    assert all(torch.equal(x,y) for x,y in zip(actual,expected))


def test_wrapped_evaluator_keeps_each_candidates_own_previous_state(monkeypatch):
    class Model:
        _fk=staticmethod(fk)
        def incremental_prediction(self,packet):
            result=packet.prev_state.clone();result[:,0]=2*result[:,0]+1;return result
        def gains(self,mode):return torch.ones(19)*.5
    class Cache:
        device=torch.device('cpu');beta=torch.zeros(1,10);data={'gt':np.zeros((3,51),np.float32)}
        def packet(self,row,prev,temperatures):
            packet=SimpleNamespace(prev_state=prev,counts=torch.ones(len(prev),dtype=torch.long),lnes_window_ms=torch.full((len(prev),),200))
            return packet,{1.:(torch.zeros(1,51),{})}
    monkeypatch.setattr(Q,'FusionPolicy',CentroidFusionPolicy)
    monkeypatch.setattr(Q.R,'frame_metrics',lambda model,pred,truth,beta:(pred.new_zeros(len(pred),len(Q.R.METRICS)),pred.new_zeros(len(pred),3),pred.new_zeros(len(pred),3)))
    clip={'clip_id':'test','row_indices':[0,1],'initial_gt_ms':0}
    meta={'rows':[{'end_ms':1,'packet_id':'a'},{'end_ms':2,'packet_id':'b'}]}
    candidates=[Q.Candidate({'id':'centroid','translation_reference':'joint_centroid'},3407),Q.Candidate({'id':'no_d','translation_reference':'joint_centroid','gain_scale':[0.,0.,0.]},3407)]
    raw=rollout_with_geometry(Q.rollout,Model(),meta,clip,Cache(),candidates)
    assert raw['predictions'][1,0,-1,0]==3.
    assert raw['predictions'][1,0,-2,0]==pytest.approx(1.,abs=1e-6)


def test_real_mano_centroid_constraint_cpu():
    import yaml
    from mano_layer import ManoLayer
    from pose_repr import decode_to_mano_inputs
    cfg=yaml.safe_load((ROOT/'configs/xkf_repair/repair_s3407.yaml').read_text())
    mano=ManoLayer(cfg['MANO']['NPZ'],add_mean=False).float().eval();mano.fast_chain=True
    def geometry(state,beta):
        d=decode_to_mano_inputs(state,cfg['MODEL']['POSE_REPR'],mano.hands_components,mano.hands_mean)
        return mano(beta,d['global_orient'],d['local_full_aa'],d['transl'])
    a,b,beta,k=values()
    with torch.no_grad():
        actual=fuse_joint_centroids(a,b,k,beta,geometry)
        expected=(1-k[:,:3])*centroid(a,beta,geometry)+k[:,:3]*centroid(b,beta,geometry)
        assert torch.allclose(centroid(actual,beta,geometry),expected,atol=2e-7)
