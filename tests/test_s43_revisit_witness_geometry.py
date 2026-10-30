"""Geometry/source-fixture engineering only; no learner or task-risk scores."""
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tools.s43_revisit import witness_geometry as G

K=np.array([[225.,0.,120.],[0.,225.,90.],[0.,0.,1.]])


def test_exact_joint_fk_matches_complete_readonly_mano():
    from tools.s43_integration.fast_geometry import fk_joints
    torch.set_num_threads(1)
    p=np.repeat(G.PUBLIC_INITIAL[None],3,axis=0)
    p[1,3:6]=[.1,.2,-.1];p[1,6:]=np.linspace(-.05,.05,45)
    p[2,3]=np.pi-1e-8;p[2,0]=.01
    beta=np.linspace(-.1,.1,10)
    np.testing.assert_allclose(G.fk_joints(p,beta),fk_joints(p,beta),atol=1e-7,rtol=0)


def test_public_axes_are_real_shortening_and_ten_actual_actions_are5mm():
    rows=G.public_flexion_axes();assert len(rows)==5
    for r in rows:
        chosen=[x for x in r['candidates'] if x['axis']==r['axis'] and x['sign']==r['sign']][0]
        assert chosen['shortening_mm']>0
        assert chosen['mean_joint_displacement_mm']>.5
    states,cal=G.motion_endpoints(G.PUBLIC_INITIAL,np.zeros(10))
    assert states.shape==(5,2,51)
    np.testing.assert_allclose(cal['actual_mm'],5.,atol=.001,rtol=0)
    assert np.all(cal['target_reached'])
    assert np.max(cal['angles_rad'])<.6


def test_rotation_near_pi_is_right_local_and_mean_once():
    p=G.PUBLIC_INITIAL.copy();p[3:6]=[np.pi-1e-10,0,0]
    axis=np.array([0.,1.,0.]);q=G.right_retract(p,0,axis,.3)
    np.testing.assert_allclose(G.rotation_exp(q[3:6]),G.rotation_exp(p[3:6])@G.rotation_exp(axis*.3),atol=1e-14)
    q=G.right_retract(p,1,axis,.3);mean=G._joint_constants()['mean'][:3]
    np.testing.assert_allclose(G.rotation_exp(q[6:9]+mean),G.rotation_exp(p[6:9]+mean)@G.rotation_exp(axis*.3),atol=1e-14)
    np.testing.assert_array_equal(q[9:],p[9:])


def test_unreachable_scale_records_actual_cap_instead_of_clamping_state():
    _,cal=G.motion_endpoints(G.PUBLIC_INITIAL,np.zeros(10),target_mm=500.)
    assert not np.any(cal['target_reached'])
    np.testing.assert_array_equal(cal['angles_rad'],np.full((5,2),.6))
    assert np.max(cal['actual_mm'])<500


def test_closed_form_horizontal_bone_kernel():
    p=G.MotionPolicy(G.PUBLIC_INITIAL,np.zeros(10),K,0,mode='quota')
    bones=np.array([[0.,5.5],[80.,5.5],[160.,5.5],[240.,5.5]])
    kernel=p._kernel(bones)
    np.testing.assert_allclose(kernel[1],np.ones(60),atol=1e-7)
    np.testing.assert_allclose(kernel[0],np.exp(-16/32)*np.ones(60),atol=1e-7)


def test_current_epoch_cache_bytes_and_constant_counts_null():
    p=G.MotionPolicy(G.PUBLIC_INITIAL,np.zeros(10),K,0)
    d=p.episode_score_details(np.ones((10,8,8,2),np.uint32),(120,80),0)
    assert d['priority']==0 and d['sign']==0
    assert p.receipt()['cache_bytes']==540000
    assert p._lut.shape==(5,10,45,60) and p._lut.dtype==np.float32
    assert p.state_nbytes==541840
    np.testing.assert_allclose(p._lut.sum(1),0,atol=1e-6)
    owners={p.episode_score_details(np.zeros((10,8,8,2),np.uint32),(x,80),0)['owner'] for x in range(0,240,8)}
    assert len(owners)>1


def test_centered_time_contrast_has_both_signs_without_destroying_abspriority():
    p=G.MotionPolicy(G.PUBLIC_INITIAL,np.zeros(10),K,0)
    a=np.zeros((10,8,8,2),np.uint32);a[:,4,4,0]=np.arange(1,11)
    b=a[::-1].copy()
    x=p.episode_score_details(a,(120,80),0);y=p.episode_score_details(b,(120,80),0)
    assert x['priority']>0 and x['owner']==y['owner']
    assert x['sign']==-y['sign'] and x['sign']!=0
    assert x['signed_score']==-y['signed_score']
    assert x['priority']==y['priority']


def test_outside_sensor_padding_is_unobserved_and_ignored():
    p=G.MotionPolicy(G.PUBLIC_INITIAL,np.zeros(10),K,0)
    b=np.zeros((10,8,8,2),np.uint32);b[:,4:,:,:]=100
    assert p.episode_scores(b,(232,176),0)[1]==0
    with pytest.raises(ValueError):p.episode_scores(b,(232,184),0)


def test_generic_temporal_never_loads_mano_and_separable_rate_has_no_motion_covariance():
    with patch.object(G,'_joint_constants',side_effect=AssertionError('MANO touched')):
        p=G.MotionPolicy(G.PUBLIC_INITIAL,np.zeros(10),K,0,mode='generic_temporal')
        b=np.zeros((10,8,8,2),np.uint32);b[:,3,4,0]=np.arange(1,11)
        assert p.episode_scores(b,(120,80),0)[1]<1e-12
        assert p.receipt()['shared_joint_asset_array_bytes']==0
        assert p.receipt()['cache_bytes']==0


def test_epoch_hold_and_cache_does_not_change_anchor():
    p=G.MotionPolicy(G.PUBLIC_INITIAL,np.zeros(10),K,0)
    before=p.prev.copy();b=np.ones((10,8,8,2),np.uint32);b[2,2,2]=100
    p.episode_scores(b,(120,80),0);p.episode_scores(b,(120,80),10)
    assert p.receipt()['kernel_refreshes']==2
    assert p.episode_scores(b,(120,80),50)[1]==0
    np.testing.assert_array_equal(p.prev,before)
    with pytest.raises(ValueError):G.MotionPolicy(before,np.zeros(10),K,50).episode_scores(b,(120,80),40)


def test_generator_fixture_separates_raw_source_targets_and_terminal_states():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as folder:
        prefixes=[Path(folder)/str(i) for i in range(2)]
        targets=[]
        for i,sign in enumerate((1,-1)):
            d=G.generate_synthetic_case(G.SyntheticCondition(1,sign,moving_background=bool(i)),prefixes[i])
            assert d['score_read'] is False
            with np.load(str(prefixes[i])+'.events.npz') as z:
                assert not set(('target51','joints_m','finger','sign'))&set(z.files)
                e=z['events'];assert e.shape[1]==4 and len(e)>0
                assert e[:,2].min()>=0 and e[:,2].max()<100
                np.testing.assert_array_equal(z['public_initial51'],G.PUBLIC_INITIAL)
            # This is public-generator FK validation, not prediction scoring.
            with np.load(str(prefixes[i])+'.targets.npz') as z:targets.append(z['joints_m'][-1].copy())
        a,b=targets;idx=G.FINGER_JOINTS[1]
        assert np.linalg.norm((a[idx]-a[0])-(b[idx]-b[0]),axis=-1).mean()>.009
