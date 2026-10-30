"""A-query engineering only; no learned task scores or target data."""
from unittest.mock import patch
import numpy as np
import pytest
import torch

from tools.s43_revisit import field_query as Q
from tools.s43_revisit import witness_geometry as G

K=np.asarray([[225.,0,120],[0,225.,90],[0,0,1.]])


def builder(anchor=None,ref=0,**kwargs):
    return Q.FieldQueryBuilder(G.PUBLIC_INITIAL if anchor is None else anchor,np.zeros(10),K,ref,**kwargs)


def test_shapes_native_bin_mean_and_explicit_payload():
    q=builder().build(50)
    assert q.grid.shape==(8,90,49,2) and q.grid.dtype==torch.float32
    assert q.valid.shape==(8,90,49) and q.valid.dtype==torch.bool
    np.testing.assert_array_equal(q.bin_edges_ms,[0,7,13,19,25,32,38,44,50])
    np.testing.assert_array_equal(q.bin_times_ms,[3,9.5,15.5,21.5,28,34.5,40.5,46.5])
    assert q.meta['two_polarity_sample_values']==70560
    assert q.meta['two_polarity_float32_payload_bytes']==282240
    assert q.meta['coordinate_tensor_bytes']==sum(v.numel()*v.element_size() for v in q.__dict__.values() if isinstance(v,torch.Tensor))


def test_anatomical_centers_equal_actual_right_action_FK_not_image_chord():
    b=builder();q=b.build(50);actual=q.centers_px.numpy().reshape(8,5,3,3,2,2)
    angles=b._calibration['angles_rad']
    for time,s in enumerate(q.bin_times_ms.numpy()):
        for j,row in enumerate(G.public_flexion_axes()):
            for k,sign in enumerate((1,-1)):
                state=G.right_retract(G.PUBLIC_INITIAL,row['mcp_mano'],np.eye(3)[row['axis']]*row['sign']*sign,s/50*angles[j][k])
                own=G.fk_joints(state,np.zeros(10))[G.FINGER_JOINTS[j]]
                alpha=np.asarray([.25,.5,.75])[:,None]
                xyz=own[:-1,None]*(1-alpha)+own[1:,None]*alpha
                h=xyz@K.T;expected=h[...,:2]/h[...,2:]
                np.testing.assert_allclose(actual[time,j,:,:,k],expected,atol=1e-5,rtol=0)


def test_static_keeps_both_sign_slots_without_geometric_sign_gate():
    q=builder().build(50,mode='static')
    grid=q.grid.reshape(8,5,3,3,2,7,7,2)
    assert torch.equal(grid[0],grid[-1])
    assert torch.equal(grid[:,:,:,:,0],grid[:,:,:,:,1])
    tokens=q.sample(torch.ones(8,2,90,120))
    assert tokens.shape==(5,3,3,2,8,7,7,2)


def test_generic_matches_actual_endpoints_but_curved_intermediate_path_can_differ():
    b=builder();a=b.build(100,mode='anatomical');g=b.build(100,mode='generic_trajectory')
    assert torch.equal(a.centers_px,g.centers_px)
    a=b.build(50,mode='anatomical');g=b.build(50,mode='generic_trajectory')
    assert not torch.equal(a.centers_px,g.centers_px)
    assert float((a.centers_px-g.centers_px).abs().max())<.2


def test_cell_center_normalization_bilinear_linear_field_and_gradient():
    q=builder().build(50,mode='static')
    yy,xx=np.meshgrid(np.arange(90)*2+.5,np.arange(120)*2+.5,indexing='ij')
    value=np.stack((xx,yy),axis=0).astype(np.float32)
    x=torch.tensor(np.repeat(value[None],8,axis=0),requires_grad=True)
    out=q.sample(x)
    actual=out.permute(4,0,1,2,3,5,6,7).reshape(8,90,49,2)
    pixel=(q.grid+1)*torch.tensor([240.,180.])/2-.5
    np.testing.assert_allclose(actual.detach().numpy(),pixel.numpy(),atol=3e-5,rtol=0)
    out.sum().backward();assert x.grad.abs().sum()>0
    assert not q.grid.requires_grad
    with pytest.raises(ValueError):q.sample(torch.zeros(8,2,180,240))


def test_batch_sampling_and_native_resolution_share_normalized_coordinates():
    a=builder().build(50,mode='static')
    b=builder(config=Q.FieldQueryConfig(field_resolution=1)).build(50,mode='static')
    assert torch.equal(a.grid,b.grid)
    assert a.sample(torch.ones(2,8,3,90,120)).shape==(2,5,3,3,2,8,7,7,3)
    assert b.sample(torch.ones(8,3,180,240)).shape==(5,3,3,2,8,7,7,3)


def test_so3_interpolation_crosses_pi_geodesically_and_hand_mean_once():
    a=G.PUBLIC_INITIAL.copy();b=a.copy();a[3]=np.pi-.1;b[3]=-np.pi+.1
    mean=G._joint_constants()['mean'][:3]
    a[6:9]=np.array([np.pi-.1,0,0])-mean;b[6:9]=np.array([-np.pi+.1,0,0])-mean
    p=Q.interpolate_deployed_pose(a,b,.5)
    expected=G.rotation_exp([np.pi,0,0])
    np.testing.assert_allclose(G.rotation_exp(p[3:6]),expected,atol=1e-14)
    np.testing.assert_allclose(G.rotation_exp(p[6:9]+mean),expected,atol=1e-14)


def test_history_is_deployed_only_future_rejected_and_pre_history_marked():
    p=G.PUBLIC_INITIAL.copy();p[0]=.01
    b=builder(p,ref=20,history=[(0,G.PUBLIC_INITIAL)])
    q=b.build(50)
    np.testing.assert_array_equal(q.history_supported,[True,True,True,False,False,False,False,False])
    with pytest.raises(ValueError):builder(p,ref=20,history=[(21,p)])
    with pytest.raises(ValueError):builder(p,ref=20,history=[(20,p),(0,G.PUBLIC_INITIAL)])
    with pytest.raises(ValueError):b.build(19)
    empty=builder(ref=20).build(30)
    assert not empty.history_supported.any()


def test_corrected_requery_exact_and_no_observation_or_anchor_mutation():
    normal=G.PUBLIC_INITIAL.copy();wrong=normal.copy();wrong[0]+=.01
    wrong=G.right_retract(wrong,0,[0,1,0],.1)
    counts=np.zeros((8,90,120,2),np.uint32);counts[:,20:40,30:60]=1;copy=counts.copy()
    for mode in Q.MODES:
        a=builder(normal).build(50,mode=mode,event_counts=counts)
        w=builder(wrong).build(50,mode=mode,event_counts=counts)
        fixed=builder(normal).build(50,mode=mode,event_counts=counts)
        assert torch.equal(a.grid,fixed.grid)
        if mode=='event_grid':assert torch.equal(a.grid,w.grid)
        else:assert not torch.equal(a.grid,w.grid)
    np.testing.assert_array_equal(counts,copy)
    np.testing.assert_array_equal(normal,G.PUBLIC_INITIAL)


def test_event_only_bbox_empty_fallback_and_no_MANO_dependency():
    support=np.zeros((90,120),bool);support[10,20]=True;support[30,40]=True
    with patch.object(G,'fk_joints',side_effect=AssertionError('MANO called')), patch.object(G,'motion_endpoints',side_effect=AssertionError('MANO called')):
        q=builder().build(50,mode='event_grid',event_counts=support)
        assert q.meta['query_box']==[39.5,19.5,81.5,61.5]
        assert q.meta['MANO_used'] is False
        empty=builder().build(50,mode='event_grid',event_counts=np.zeros((90,120),bool))
        assert empty.meta['query_box']==[-.5,-.5,239.5,179.5]


def test_hand_box_is_static_uniform_ROI_without_finger_identity():
    q=builder().build(50,mode='hand_box')
    assert q.meta['finger_identity_used'] is False
    assert torch.equal(q.centers_px[0],q.centers_px[-1])
    assert len(torch.unique(q.centers_px[0],dim=0))==90
    x0,y0,x1,y1=q.meta['query_box'];assert x1-x0>=32 and y1-y0>=32


def test_clock_shift_purequery_and_output_does_not_alias_builder_cache():
    b=builder();a=b.build(50);b.build(75);again=b.build(50)
    assert torch.equal(a.grid,again.grid)
    shift=builder(ref=1000).build(1050)
    assert torch.equal(a.grid,shift.grid)
    a.centers_px.zero_();assert torch.equal(again.grid,b.build(50).grid)


def test_projection_degeneracy_and_unsupported_nearplane_are_explicit():
    frame,bad=Q._frame(np.array([[1.,2.]]),np.array([[1.,2.]]),1e-6)
    assert bad[0];np.testing.assert_array_equal(frame[0],np.eye(2))
    p=G.PUBLIC_INITIAL.copy();p[2]=.01
    q=builder(p).build(50,mode='static')
    assert torch.isfinite(q.grid).all() and not q.depth_valid.all()
    assert q.valid.sum()<q.valid.numel()
    with pytest.raises(ValueError):builder(p).build(50,mode='hand_box')


def test_schema_no_nan_and_event_counts_nonnegative():
    p=G.PUBLIC_INITIAL.copy();p[0]=np.nan
    with pytest.raises(ValueError):builder(p)
    with pytest.raises(ValueError):builder().build(50,mode='event_grid',event_counts=-np.ones((90,120)))


def test_beta_K_versions_rebind_and_event_query_remains_sensor_only():
    a=builder().build(50,mode='static')
    beta=np.zeros(10);beta[0]=.2
    b=Q.FieldQueryBuilder(G.PUBLIC_INITIAL,beta,K,0).build(50,mode='static')
    camera=K.copy();camera[0,0]*=1.01
    c=Q.FieldQueryBuilder(G.PUBLIC_INITIAL,np.zeros(10),camera,0).build(50,mode='static')
    assert not torch.equal(a.grid,b.grid) and not torch.equal(a.grid,c.grid)
    assert a.meta['condition_sha256']!=b.meta['condition_sha256']!=c.meta['condition_sha256']
