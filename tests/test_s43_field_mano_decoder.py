"""Fixed MANO operator/parity/gradient engineering; no taskdata or fitting."""
from functools import lru_cache
import numpy as np
import pytest
import torch

from tools.s43_revisit.field_mano_decoder import FieldManoDecoder,rotation_exp
from tools.s43_revisit import witness_geometry as G
from tools.s43_rel.geometry import mano_kinematics


def inputs(dtype=torch.float64):
    return torch.tensor(G.PUBLIC_INITIAL[None],dtype=dtype),torch.zeros(1,10,dtype=dtype)


def test_zero_action_is_bitwise_identity_and_preserves_nonprincipal_chart():
    m=FieldManoDecoder(dtype=torch.float64);a,b=inputs();a[0,3]=2*np.pi+.1
    implicit=m.identity(a,b);raw=torch.zeros(1,51,dtype=torch.float64,requires_grad=True);out=m(a,b,raw)
    assert torch.equal(implicit.joints,out.joints) and torch.equal(implicit.vertices,out.vertices)
    assert torch.equal(out.commit_pose51(),a) and not out.commit_pose51().requires_grad
    assert sum(p.numel() for p in m.parameters())==0


def test_full_vertices_and21joints_match_frozen_exact_MANO():
    m=FieldManoDecoder(dtype=torch.float64);rng=np.random.default_rng(3407)
    anchors=np.repeat(G.PUBLIC_INITIAL[None],3,axis=0);anchors[:,3:]+=rng.normal(0,.15,(3,48));anchors[2,3]=np.pi-1e-8
    betas=rng.normal(0,.1,(3,10));action=rng.normal(0,.2,(3,51))
    out=m(torch.tensor(anchors),torch.tensor(betas),torch.tensor(action));pose=out.commit_pose51().numpy()
    for i in range(3):
        np.testing.assert_allclose(out.joints[i].detach().numpy(),G.fk_joints(pose[i],betas[i]),atol=2e-13,rtol=0)
        np.testing.assert_allclose(out.vertices[i].detach().numpy(),mano_kinematics(pose[i],betas[i],jacobian=False),atol=2e-13,rtol=0)


@lru_cache(maxsize=1)
def legacy():
    from tools.s43_integration.model import load_legacy
    base,_=load_legacy(device='cpu');return base


def test_original_S43_base_fk_parity_FP32_declared_tolerance():
    m=FieldManoDecoder();a,b=inputs(torch.float32);d=torch.zeros(1,51);d[0,3:]=torch.linspace(-.2,.2,48)
    out=m(a,b,d);state=out.commit_pose51()
    with torch.no_grad():v,j=legacy()._fk(state,b)
    np.testing.assert_allclose(out.joints.detach().numpy(),j.numpy(),atol=1e-6,rtol=0)
    np.testing.assert_allclose(out.vertices.detach().numpy(),v.numpy(),atol=1e-6,rtol=0)


def test_exp_exactzero_has_finite_autograd_without_epsilon_rotation_bias():
    v=torch.zeros(2,3,dtype=torch.float64,requires_grad=True);r=rotation_exp(v)
    assert torch.equal(r,torch.eye(3,dtype=torch.float64).expand(2,3,3))
    (r*torch.arange(9,dtype=torch.float64).reshape(3,3)).sum().backward()
    assert torch.isfinite(v.grad).all() and v.grad.norm()>0


@pytest.mark.parametrize('nonzero',[False,True])
def test_all51_action_gradients_match_finite_differences_including_tip_vertices(nonzero):
    m=FieldManoDecoder(dtype=torch.float64);a,b=inputs();a=a.clone();a[0,6:]=torch.linspace(-.03,.04,45)
    rng=np.random.default_rng(3407);start=rng.normal(0,.15,(1,51)) if nonzero else np.zeros((1,51))
    d=torch.tensor(start,requires_grad=True)
    out=m(a,b,d);sel=torch.cat((out.joints.reshape(-1),out.vertices[:,[72,120,320,444,744]].reshape(-1)))
    w=torch.tensor(rng.normal(size=len(sel)));w=w/w.norm()
    objective=(sel*w).sum()/.01;grad=torch.autograd.grad(objective,d)[0].detach().numpy()[0]
    def value(x):
        z=m(a,b,torch.tensor(x));s=torch.cat((z.joints.reshape(-1),z.vertices[:,[72,120,320,444,744]].reshape(-1)))
        return float((s*w).sum()/.01)
    eps=2e-6;fd=[]
    with torch.no_grad():
        for i in range(51):
            plus=start.copy();minus=start.copy();plus[0,i]+=eps;minus[0,i]-=eps;fd.append((value(plus)-value(minus))/(2*eps))
    assert np.isfinite(grad).all()
    np.testing.assert_allclose(grad,fd,atol=3e-8,rtol=3e-6)


def test_right_action_near_pi_and_mean_once_detached_commit_roundtrip():
    m=FieldManoDecoder(dtype=torch.float64);a,b=inputs();a[0,3]=np.pi-1e-10
    mean=m.hands_mean[:3];a[0,6:9]=torch.tensor([np.pi-1e-10,0.,0.])-mean
    d=torch.zeros(1,51,dtype=torch.float64,requires_grad=True)
    with torch.no_grad():d[0,4]=.3;d[0,7]=-.2
    out=m(a,b,d)
    expected=m.anchor_rotations(a)@rotation_exp(.1*d[:,3:].reshape(1,16,3))
    torch.testing.assert_close(out.rotations,expected,atol=1e-14,rtol=0)
    saved=out.commit_pose51();again=m.identity(saved,b)
    torch.testing.assert_close(again.joints,out.joints,atol=2e-13,rtol=0)
    assert not saved.requires_grad and out.joints.requires_grad


def test_translation_scale_batchbroadcast_and_joint63_common_identity():
    m=FieldManoDecoder(dtype=torch.float64);a,b=inputs();b=b.expand(3,-1);d=torch.zeros(3,51,dtype=torch.float64);d[:,0]=torch.arange(3)
    out=m(a[0],b,d);zero=m.identity(a,b)
    expected=torch.zeros_like(out.joints);expected[:,:,0]=torch.arange(3,dtype=torch.float64)[:,None]*.01
    torch.testing.assert_close(out.joints-zero.joints,expected,atol=2e-16,rtol=0)
    torch.testing.assert_close(zero.joints+.01*torch.zeros_like(zero.joints),zero.joints,atol=0,rtol=0)


def test_zero_initialized_head_can_start_learning_FK_loss():
    torch.manual_seed(3407);m=FieldManoDecoder(dtype=torch.float64);a,b=inputs();head=torch.nn.Linear(7,51,dtype=torch.float64)
    torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    features=torch.randn(1,7,dtype=torch.float64,requires_grad=True);pred=m(a,b,head(features))
    target=pred.joints.detach()+torch.tensor([.002,-.001,.003])
    ((pred.joints-target)/.01).square().mean().backward()
    assert torch.isfinite(head.weight.grad).all() and head.weight.grad.norm()>0
    assert torch.isfinite(head.bias.grad).all() and head.bias.grad.norm()>0
    assert features.grad.norm()==0  # expected first-step zero-outputhead hiddengradient


def test_beta_gradient_and_schema_nonfinite_rejection():
    m=FieldManoDecoder(dtype=torch.float64);a,b=inputs();b.requires_grad_();m(a,b).vertices.square().sum().backward()
    assert torch.isfinite(b.grad).all() and b.grad.norm()>0
    with pytest.raises(ValueError):m(a,b,torch.full((1,51),float('nan')))
    with pytest.raises(ValueError):m(a,b,torch.zeros(1,50))
    with pytest.raises(ValueError):m(a.expand(2,-1),b.expand(3,-1))
