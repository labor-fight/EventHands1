"""R1 domain repair and serial/ordered CPU geometry, no fits or task scores."""
import copy
import numpy as np
import pytest
import torch
from tools.s43_integration.model import load_legacy
from tools.s43_revisit.s43_field_model import S43FieldModel
from tools.s43_revisit.s43_field_model_reacquire import S43FieldReacquireModel,prepare_query_cpu
from tools.s43_revisit.field_core import TimeFieldMemory
from tools.s43_revisit import witness_geometry as G


@pytest.fixture(scope='module')
def base():
    torch.set_num_threads(2);return load_legacy()[0]


def inputs(B=2,z=.6):
    field=torch.zeros(B,40,180,240);field[:,:16]=float(np.log1p(.25));field[:,16:24]=1.;field[:,24:]=float(np.log1p(4.))
    prev=torch.zeros(B,51);prev[:,2]=z
    K=torch.tensor([[600.,0,320],[0,600,240],[0,0,1.]])[None].repeat(B,1,1)
    return field,prev,torch.zeros(B,10),K,np.zeros(B,np.int64),np.full(B,50,np.int64),[()]*B


@pytest.mark.parametrize('arm',['hand_box','static'])
def test_same_parameter_keys_initialization_and_indomain_old_values(base,arm):
    torch.manual_seed(3407);old=S43FieldModel(copy.deepcopy(base),arm).eval()
    torch.manual_seed(3407);new=S43FieldReacquireModel(copy.deepcopy(base),arm,geometry_workers=1).eval()
    assert old.state_dict().keys()==new.state_dict().keys()
    for key,v in old.state_dict().items():torch.testing.assert_close(v,new.state_dict()[key],rtol=0,atol=0)
    args=inputs();a,ma=old._local_patches(*args);b,mb=new._local_patches(*args)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert all(not m['used_uniform_reacquisition'] for m in mb)
    assert [m['condition_sha256'] for m in ma]==[m['condition_sha256'] for m in mb]
    field,p,beta,K,refs,T,history=args
    with torch.no_grad():
        a=old(field,p,beta_input=beta,K_source=K,ref_ms=refs,cutoff_ms=T,histories=history)
        b=new(field,p,beta_input=beta,K_source=K,ref_ms=refs,cutoff_ms=T,histories=history)
    torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('arm',['hand_box','static'])
def test_nearplane_uniform_values_are_observed_but_association_flag_zero(base,arm):
    model=S43FieldReacquireModel(copy.deepcopy(base),arm,geometry_workers=1)
    args=inputs(z=-.2);before=args[1].clone();local,metadata=model._local_patches(*args)
    assert local.shape==(2,8,90,5,7,7)
    assert torch.all(local[:,:,:,:2]>0) and torch.all(local[:,:,:,2]==1)
    assert torch.count_nonzero(local[:,:,:,3])==0 and torch.all(local[:,:,:,4]==1)
    torch.testing.assert_close(args[1],before,rtol=0,atol=0)
    assert all(m['used_uniform_reacquisition'] and m['initial_finger_invalid_count']>0 for m in metadata)
    assert model.observation_stats['uniform_reacquisition_rows']==2


def test_any_single_invalid_finger_joint_triggers_common_fallback():
    p=np.array([0.,0.,.6]+[0.]*48);j=G.fk_joints(p,np.zeros(10))
    p[2]+=.03-float(j[1:,2].min())-1e-6
    j=G.fk_joints(p,np.zeros(10));assert (j[1:,2]<=.03).any() and (j[1:,2]>.03).any()
    task=(p,np.zeros(10),np.array([[600.,0,320],[0,600,240],[0,0,1.]]),0,50,(),'static',.375)
    plan=prepare_query_cpu(task)
    assert plan.metadata['used_uniform_reacquisition']
    assert 0<plan.metadata['initial_finger_invalid_count']<20


@pytest.mark.parametrize('arm',['hand_box','static'])
def test_zero_valid_2d_query_falls_back_without_nearplane(base,arm):
    model=S43FieldReacquireModel(copy.deepcopy(base),arm,geometry_workers=1)
    args=list(inputs());args[1][:,0]=100
    local,metadata=model._local_patches(*args)
    assert all(m['invalid_reason']=='query_has_zero_valid_2d_interpolation_samples' for m in metadata)
    assert all(m['initial_finger_invalid_count']==0 for m in metadata)
    assert local[:,:,:,:2].any() and not local[:,:,:,3].any()


def test_fallback_does_not_clip_hold_or_reset_finite_bad_pose(base):
    model=S43FieldReacquireModel(copy.deepcopy(base),'hand_box',geometry_workers=1).eval()
    with torch.no_grad():
        model.base.rn.fc.weight.zero_();model.base.rn.fc.bias.fill_(.001)
        for p in model.base.prev_mlp.parameters():p.zero_()
    field,p,beta,K,refs,T,hist=inputs(B=1,z=-.2)
    with torch.no_grad():out,aux=model(field,p,beta_input=beta,K_source=K,ref_ms=refs,cutoff_ms=T,return_aux=True)
    torch.testing.assert_close(out,p+.001,rtol=0,atol=0)
    assert out[0,2]<0 and aux['query_metadata'][0]['used_uniform_reacquisition']


def test_serial_parallel_order_numeric_source_rng_and_diagnostics(base):
    torch.manual_seed(3407);serial=S43FieldReacquireModel(copy.deepcopy(base),'static',geometry_workers=1)
    torch.manual_seed(3407);parallel=S43FieldReacquireModel(copy.deepcopy(base),'static',geometry_workers=16)
    args=list(inputs(B=4));args[1][1,2]=-.2;args[1][2,0]=100
    rng=torch.get_rng_state().clone();a,ma=serial._local_patches(*args);b,mb=parallel._local_patches(*args)
    torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    for x,y in zip(ma,mb):
        for key in ('requested_mode','used_uniform_reacquisition','invalid_reason','condition_sha256','initial_finger_z_min_m','initial_finger_z_max_m'):
            assert x[key]==y[key]
    assert parallel.observation_stats['parallel_query_rows']==4
    state=parallel.diagnostic_state_dict();parallel.load_diagnostic_state_dict(state)
    assert parallel.observation_stats==state['stats'];parallel.close_geometry_workers()
    # The source object is not passed into query; verify read-only field origin.
    memory=TimeFieldMemory();memory.ingest(np.array([[90,80,2,1]],np.int64),50,0,source_range=(0,1))
    snap=memory.readout(50);before=snap.content_sha256();v=memory.version
    q=list(inputs(B=1,z=-.2));q[0]=torch.tensor(snap.to_grid40ch()[None])
    serial._local_patches(*q);assert snap.content_sha256()==before and memory.version==v


@pytest.mark.parametrize('which',['pose','beta','K'])
def test_nonfinite_is_error_and_not_reacquisition(base,which):
    model=S43FieldReacquireModel(copy.deepcopy(base),'static',geometry_workers=1).eval()
    field,p,beta,K,ref,T,hist=inputs(B=1)
    {'pose':p,'beta':beta,'K':K}[which].flatten()[0]=float('nan')
    with pytest.raises(ValueError):model(field,p,beta_input=beta,K_source=K,ref_ms=ref,cutoff_ms=T)
    assert model.observation_stats['uniform_reacquisition_rows']==0


def test_invalid_history_is_not_swallowed_by_domain_fallback(base):
    model=S43FieldReacquireModel(copy.deepcopy(base),'hand_box',geometry_workers=1)
    args=list(inputs(B=1,z=-.2));args[-1]=[[(1,args[1][0].numpy())]]
    with pytest.raises(ValueError,match='future'):model._local_patches(*args)
