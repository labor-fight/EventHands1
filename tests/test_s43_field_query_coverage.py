"""Only arithmetic/poison-label fixtures; no scientific data or model fitting."""
import json
import numpy as np
import pytest
from tools.s43_revisit import field_query_coverage as c


def coords(center=(50.,50.)):
    centers=np.broadcast_to(np.asarray(center),(8,90,2)).copy()
    off=np.arange(-3,4)*2.;yy,xx=np.meshgrid(off,off,indexing='ij')
    px=centers[:,:,None,None]+np.stack((xx,yy),-1)
    return dict(grid=(2*(px+.5)/np.asarray([240,180])-1).reshape(8,90,49,2),
        valid=np.ones((8,90,49),bool),full_interp_support=np.ones((8,90,49),bool))


def geometry(q,uv,finger=None,bone=None,arm='anatomical',inview=None):
    uv=np.asarray(uv,float);n=len(uv)
    return c.geometry(q,uv,np.ones(n,bool),np.ones(n,bool) if inview is None else inview,
        np.zeros(n,int) if finger is None else finger,np.zeros(n,int) if bone is None else bone,arm)


def test_fixed_patch_hull_and_native_coordinate_inverse():
    r=geometry(coords(),[[50,50],[56,56],[56.01,50]])
    np.testing.assert_array_equal(r['matched_geometry'][-1],[True,True,False])
    np.testing.assert_array_equal(r['matched_supported'][-1],[True,True,False])
    np.testing.assert_allclose(r['last_distance_px'],[0,0,.01],atol=1e-8)


def test_supported_quad_and_gt_inview_are_separate_from_geometry():
    q=coords();q['valid'][:,:,24]=False
    r=geometry(q,[[50,50]])
    assert r['matched_geometry'].all() and not r['matched_supported'].any()
    r=geometry(coords(),[[50,50]],inview=np.array([False]))
    assert r['matched_geometry'].all() and not r['matched_supported'].any()


def test_samebone_samefinger_and_box_slots_have_distinct_semantics():
    q=coords((200,100));px=(q['grid']+1)*np.asarray([240,180])/2-.5
    px[:,:6]+=np.array([-150,-50]);px[:,6:18]+=np.array([-100,0])
    q['grid']=2*(px+.5)/np.asarray([240,180])-1
    r=geometry(q,[[100,100],[100,100]],bone=np.array([0,-1]))
    np.testing.assert_array_equal(r['matched_supported'][-1],[False,True])
    assert r['samefinger_supported'].all() and r['any_supported'].all()
    box=geometry(q,[[100,100]],arm='hand_box')
    assert box['matched_supported'].all()


def test_last_bin_is_not_eight_bin_spatial_union():
    q=coords();px=(q['grid']+1)*np.asarray([240,180])/2-.5;px[1:]+=np.array([50,50]);q['grid']=2*(px+.5)/np.asarray([240,180])-1
    r=geometry(q,[[50,50]])
    assert r['matched_supported'].any(0)[0] and not r['matched_supported'][-1,0]
    assert r['union_distance_px'][0]==0 and r['last_distance_px'][0]>0


def test_gt_points_and_k_scale_with_invalid_depth():
    j=np.arange(63,dtype=float).reshape(21,3);p,f,b=c.gt_points(j)
    assert p.shape==(65,3) and f.shape==b.shape==(65,)
    np.testing.assert_array_equal(p[:20],j[1:]);np.testing.assert_allclose(p[20],.75*j[1]+.25*j[2])
    K=np.array([[600,0,320],[0,600,240],[0,0,1]])
    uv,good,view=c.project(np.array([[0,0,.6],[0,0,.01]]),K)
    np.testing.assert_array_equal(uv[0],[120,90]);np.testing.assert_array_equal(good,[True,False]);np.testing.assert_array_equal(view,[True,False])


def test_freeze_hashes_poison_labels_without_decoding_and_gate_precedes_gt(tmp_path,monkeypatch):
    monkeypatch.setattr(c,'OUT',tmp_path);(tmp_path/'stage3_field_query_protocol.json').write_text('{}')
    base=tmp_path/'base';stage=tmp_path/'stage.json';stage.write_text('{}');plan=tmp_path/'plan.md';plan.write_text('fixture')
    for domain in ('synthetic','existing'):
        d=base/'features'/domain;d.mkdir(parents=True);np.savez(d/'coords.npz',**coords())
        rows=[dict(case_id='toy',arm=arm,condition=condition,coordinate_file=str(d/'coords.npz'),coordinate_sha256=c.sha(d/'coords.npz'))
              for arm in c.ARMS for condition in c.CONDITIONS]
        (d/'rows.json').write_text(json.dumps(rows));(d/'labels.json').write_bytes(b'\xffPOISON_NOT_JSON')
        (d/'complete.json').write_text(json.dumps(dict(rows_sha256=c.sha(d/'rows.json'),labels_sha256=c.sha(d/'labels.json'))))
    monkeypatch.setattr(c,'prediction_gate',lambda *a:{})
    out=tmp_path/'good';c.freeze(base,out,plan,stage)
    f=json.loads((out/'freeze.json').read_text());assert not f['labels_decoded']
    def fail(*a):raise RuntimeError('all predictions not complete')
    monkeypatch.setattr(c,'prediction_gate',fail)
    with pytest.raises(RuntimeError):c.freeze(base,tmp_path/'bad',plan,stage)
    assert not (tmp_path/'bad/freeze.json').exists()
