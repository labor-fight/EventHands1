"""Stage3 reader/intervention engineering, no scientific sources or targets."""
import numpy as np
import pytest
import torch
from tools.s43_revisit import field_query_probe as p


def test_collapse_keeps_spatial_polarity_totals_and_all_coverage():
    rng=np.random.default_rng(3407);x=np.zeros((40,180,240),np.float32)
    x[:16]=np.log1p(rng.integers(0,9,(16,180,240))*.25)
    x[24:]=np.log1p(rng.integers(0,20,(16,180,240)));x[16:24]=rng.random((8,180,240))
    y=p.collapse_time(x)
    for a,b in ((0,16),(24,40)):
        before=np.expm1(x[a:b].astype(float)).reshape(8,2,180,240).sum(0)
        after=np.expm1(y[a:b].astype(float)).reshape(8,2,180,240).sum(0)
        np.testing.assert_allclose(after,before,rtol=1e-6,atol=1e-6)
    np.testing.assert_array_equal(x[16:24],y[16:24])


def test_collapse_removes_only_time_allocation():
    a=np.zeros((40,180,240),np.float32);b=a.copy()
    a[0,3,4]=np.log1p(4);b[12,3,4]=np.log1p(4)
    np.testing.assert_array_equal(p.collapse_time(a),p.collapse_time(b))


@pytest.mark.parametrize('resolution,xy',[(1,(10.,20.)),(2,(10.5,20.5))])
def test_sampler_cell_phase_and_observation_mask(resolution,xy):
    torch.set_num_threads(2)
    x=torch.zeros(40,180,240);x[16:24]=1
    if resolution==2:x[:16,20:22,10:12]=2
    else:x[:16,20,10]=2
    g=np.zeros((8,90,49,2),np.float32);g[...,0]=2*(xy[0]+.5)/240-1;g[...,1]=2*(xy[1]+.5)/180-1
    arrays=dict(grid=g,valid=np.ones((8,90,49),bool),in_view=np.ones((8,90,49),bool),full_interp_support=np.ones((8,90,49),bool))
    local=p.sample_patches(x,arrays,resolution)
    assert local.shape==(8,90,5,7,7)
    torch.testing.assert_close(local[:,:,:2],torch.full_like(local[:,:,:2],2),atol=1e-5,rtol=0)
    assert torch.all(local[:,:,2:]==1)
    x[16:24]=0
    missing=p.sample_patches(x,arrays,resolution)
    assert not missing[:,:,2].any() and torch.all(missing[:,:,3:]==1)


def test_invalid_query_is_zero_evidence_with_separate_inview_flag():
    x=torch.ones(40,180,240)
    arrays=dict(grid=np.zeros((8,90,49,2),np.float32),valid=np.zeros((8,90,49),bool),
      in_view=np.ones((8,90,49),bool),full_interp_support=np.zeros((8,90,49),bool))
    local=p.sample_patches(x,arrays,2)
    assert not local[:,:,:3].any() and torch.all(local[:,:,3]==1) and not local[:,:,4].any()


def test_same_allocated_architecture_global_only_has_no_constant_local_leak():
    torch.set_num_threads(2);torch.manual_seed(3407);model=p.make_cnn(10,False)
    a=model(torch.ones(2,40,180,240),torch.zeros(2,8,90,5,7,7),torch.zeros(2,70))
    b=model(torch.ones(2,40,180,240),torch.randn(2,8,90,5,7,7),torch.zeros(2,70))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert sum(t.numel() for t in model.parameters())==sum(t.numel() for t in p.make_cnn(10,True).parameters())
    a.square().mean().backward();assert all(v.grad is None for v in model.local_encoder.parameters())


def test_static_sign_slots_do_not_force_identical_logits():
    torch.manual_seed(3407);model=p.make_cnn(10)
    # Every sign slot may have identical observations; each output remains independent.
    with torch.no_grad():model.head[-1].bias[0]=1;model.head[-1].bias[1]=-1
    logits=model(torch.zeros(1,40,180,240),torch.ones(1,8,90,5,7,7),torch.zeros(1,70))
    assert logits[0,0]!=logits[0,1]
    logits[0,0].backward();assert model.local_encoder[0].weight.grad.abs().sum()>0


def test_training_augmentation_excludes_corrected_requeries():
    assert sum(v[2] for v in p.CONDITIONS.values())==3
    assert p.CONDITIONS['corrected_from_x']==('root_x10mm','none',False)
    assert p.CONDITIONS['corrected_from_yaw']==('root_yaw0.1','none',False)


def test_prepare_toy_pipeline_keeps_complete_evidence_and_corrected_queries(tmp_path,monkeypatch):
    import json
    from tools.s43_revisit.witness_geometry import fk_joints
    anchor=np.asarray([0,0,.6]+[0.]*48);beta=np.zeros(10);K=np.asarray([[600.,0,320],[0,600,240],[0,0,1]])
    context=p.old.condition70(anchor,beta,K);field=np.zeros((40,180,240),np.float32);field[16:24]=1
    file=tmp_path/'toy_input.npz';np.savez_compressed(file,grid=field)
    row=dict(case_id='toy',context=context.tolist(),cutoff_ms=50,fold=0,partition='FIT7',recording='toy',feature_file=str(file),feature_sha256=p.sha(file))
    sources={(storage,error):dict(row,arm=storage,selection_error=error) for storage in ('field2','witness','quota') for error in p.old.ERRORS}
    base=tmp_path/'source';base.mkdir();p.dump(base/'labels.json',{'toy':{'prior':fk_joints(context[:51].astype(float),beta).tolist()}})
    monkeypatch.setattr(p,'source_rows',lambda domain:(base,{'toy':sources}))
    monkeypatch.setattr(p,'require_gate',lambda freeze:True)
    monkeypatch.setattr(p,'dependencies',lambda:{})
    gate=tmp_path/'dummy_gate.json';gate.write_text('{}')
    p.prepare('existing',tmp_path/'prepared',gate)
    rows=json.loads((tmp_path/'prepared/rows.json').read_text())
    assert len(rows)==45 and all(r['target_not_used'] for r in rows)
    assert all(r['query_prior']!=rows[0]['query_prior'] for r in rows if r['condition']=='wrong_x')
    assert (tmp_path/'prepared/labels.json').stat().st_mtime_ns>(tmp_path/'prepared/observations_complete.json').stat().st_mtime_ns


def test_low_capacity_rbf_real_tensor_dimensions_are_finite():
    from threadpoolctl import threadpool_limits
    rng=np.random.default_rng(3);x=rng.random((8,5520)).astype(np.float32);c=rng.random((8,70)).astype(np.float32)
    with threadpool_limits(1):head=p.LowCapacityRBF().fit(x,c,np.eye(8))
    out=head.predict(x,c)
    assert out.shape==(8,8) and np.isfinite(out).all()

