"""Opt-in exact-sparsity Jacobian engineering parity; no task scores."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from tools.s43_rel import geometry as old
from tools.s43_rel import geometry_fast as fast
from tools.s43_rel.core import RelationMemory,RelConfig


def cases():
    rng=np.random.default_rng(3407);result=[]
    for i in range(6):
        p=np.zeros(51);p[:3]=[rng.uniform(-.02,.02),rng.uniform(-.02,.01),rng.uniform(.5,.7)]
        p[3:6]=rng.normal(0,.2,3);p[6:]=rng.normal(0,.15,45)
        if i>=3:
            axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
            p[3:6]=axis*(np.pi+(i-4)*1e-7)
            p[12:15]=axis*(np.pi-(i-4)*2e-7)-old._constants()['hands_mean'][6:9]
        result.append((p,rng.normal(0,.3,10)))
    return result


@pytest.mark.parametrize('index',range(6))
def test_all51_derivatives_and_forward_match_frozen_reference(index):
    p,b=cases()[index]
    original_fn=old.mano_kinematics
    v,j=fast.mano_kinematics(p,b);ov,oj=old.mano_kinematics(p,b)
    np.testing.assert_array_equal(v,ov)
    np.testing.assert_allclose(j,oj,rtol=2e-13,atol=2e-15)
    eps=1e-4
    for axis in range(51):
        d=np.eye(51)[axis]*eps
        plus=old.mano_kinematics(p,b,delta=d,jacobian=False)
        minus=old.mano_kinematics(p,b,delta=-d,jacobian=False)
        np.testing.assert_allclose((plus-minus)/(2*eps),j[axis],rtol=3e-6,atol=4e-10)
    assert old.mano_kinematics is original_fn


def test_alias_isolation_and_binding_readout_and_gram_parity():
    p,b=cases()[0];mem=RelationMemory(RelConfig())
    K=np.array([[300.,0,119.5],[0,300.,89.5],[0,0,1.]])
    original_fn=old.mano_kinematics
    a=old.ManoGeometry().bind(p,b,K,mem.topology)
    z=fast.FastManoGeometry().bind(p,b,K,mem.topology)
    assert old.mano_kinematics is original_fn
    for key in ('uv','n','edge_ids','edge_slot','point_index','weights','normal_dot_delta','branch_regular'):
        np.testing.assert_array_equal(getattr(a,key),getattr(z,key))
    np.testing.assert_allclose(a.a,z.a,rtol=3e-13,atol=3e-13)
    rng=np.random.default_rng(21)
    Z=rng.random((len(a.edge_ids),2,2,3,2)).astype(np.float32)
    snap=SimpleNamespace(Z=Z,edge_ids=a.edge_ids)
    pa,pz=a.read(snap),z.read(snap)
    np.testing.assert_array_equal(pa['lam'],pz['lam'])
    np.testing.assert_allclose(pa['f'],pz['f'],rtol=2e-12,atol=1e-10)
    ga=np.einsum('nh,nj,nj->jh',pa['lam'],pa['a'],pa['a'])
    gz=np.einsum('nh,nj,nj->jh',pz['lam'],pz['a'],pz['a'])
    np.testing.assert_allclose(ga,gz,rtol=3e-13,atol=3e-10)


def test_no_weight_pruning_exactbytes_cache_and_no_jacobian_forward():
    info=fast.provider_metadata()
    assert info['all_nonzero_weights_retained']==np.count_nonzero(old._constants()['weights'])
    assert info['minimum_retained_weight']>0
    p,b=cases()[1]
    np.testing.assert_array_equal(fast.mano_kinematics(p,b,jacobian=False),old.mano_kinematics(p,b,jacobian=False))
    for i in range(20):fast.mano_kinematics(p,b+i*1e-10,jacobian=False)
    assert fast._shape_exact.cache_info().currsize<=16


def test_near_pi_equivalent_chart_has_same_physical_actions():
    p,b=cases()[2];q=p.copy()
    axis=np.array([.3,.4,.5]);axis/=np.linalg.norm(axis)
    p[3:6]=axis*(np.pi+1e-7);q[3:6]=-axis*(np.pi-1e-7)
    mean=old._constants()['hands_mean']
    p[12:15]=axis*(np.pi+2e-7)-mean[6:9]
    q[12:15]=-axis*(np.pi-2e-7)-mean[6:9]
    vp,jp=fast.mano_kinematics(p,b);vq,jq=fast.mano_kinematics(q,b)
    np.testing.assert_allclose(vp,vq,rtol=0,atol=1e-14)
    np.testing.assert_allclose(jp,jq,rtol=0,atol=1e-14)
