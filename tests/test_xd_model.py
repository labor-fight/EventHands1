"""Contract tests for the opt-in paired correction/recovery model."""
from __future__ import annotations
import copy
import sys
from pathlib import Path
import pytest
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/'model'))
from config import load_config
from semkine.events import EventPacketBatch


def recovery_modules():
    # A missing implementation is an explicit behavior failure during RED.
    try:
        from semkine.xyz_observation import ObservationBank, project_state_joints
        from semkine.xyz_recovery_study import RecoveryXYZModel
    except ImportError as exc:
        pytest.fail('missing XD model implementation: '+str(exc))
    return ObservationBank, project_state_joints, RecoveryXYZModel


def packet(B=2, n=12):
    ev=torch.zeros(B*n,5)
    for b in range(B):
        ev[b*n:(b+1)*n,0]=b
        ev[b*n:(b+1)*n,1]=torch.linspace(100,140,n)
        ev[b*n:(b+1)*n,2]=torch.linspace(80,110,n)
        ev[b*n:(b+1)*n,3]=torch.linspace(.001,.049,n)
        ev[b*n:(b+1)*n,4]=torch.arange(n)%2
    prev=torch.zeros(B,51); prev[:,2]=.45; prev[:,3:6]=torch.tensor([.2,-.1,.15])
    K=torch.tensor([[603.,0,325.],[0,603.,242.],[0,0,1.]]).expand(B,-1,-1).clone()
    return EventPacketBatch(ev,torch.arange(B+1)*n,torch.zeros(B,dtype=torch.long),
        torch.zeros(B,dtype=torch.long),torch.full((B,),50000,dtype=torch.long),
        torch.full((B,),.05),torch.zeros(B,dtype=torch.bool),torch.zeros(B,dtype=torch.bool),
        prev+.001,prev,torch.zeros(B,10),K)


def make_model(query='mano', null=False, self_group=False, burn_mode=None):
    _,_,Model=recovery_modules()
    cfg=load_config(str(ROOT/'configs/xa/xa_c0_s3407.yaml'))
    cfg['MANO']['NPZ']=str(ROOT/'assets/mano_right.npz')
    cfg['MODEL'].update(XDL_STUDY=True,XDL_QUERY=query,XDL_SELF=self_group,
                        XDL_OBSERVATION_NULL=null,ENCODER_HIDDEN=8,ENCODER_LAYERS=1,
                        XYZ_REL_HIDDEN=8,XYZ_READOUT_HIDDEN=16,ENCODER_MAX_NODES=32)
    if burn_mode is not None:
        cfg['MODEL']['XDL_BURN_MODE']=burn_mode
    m=Model(cfg).float()
    with torch.no_grad():
        m.xyz.fc2.weight.normal_(0,.002)
    return m


def test_projection_uses_event_frame_and_fallback_without_deleting_rows():
    _,project,_=recovery_modules()
    j=torch.tensor([[[.1,.2,1.],[0.,0.,-1.],[10.,0.,1.],[float('nan'),0.,1.]]])
    u,valid,raw,outside=project(j,(torch.tensor([225.]),torch.tensor([225.]),
                                  torch.tensor([120.]),torch.tensor([90.])),180,240)
    assert torch.allclose(u[0,0],torch.tensor([142.5,135.]))
    assert torch.equal(u[0,1],torch.tensor([120.,90.]))
    assert torch.equal(u[0,2],torch.tensor([239.,90.]))
    assert torch.equal(u[0,3],torch.tensor([120.,90.]))
    assert valid.tolist()==[[True,False,True,False]]
    assert outside.tolist()==[[False,False,True,False]]


def test_full_support_queries_change_only_centers_and_keep_empty_rows_safe():
    Bank,_,_=recovery_modules()
    bank=Bank(height=180,width=240,max_nodes=32)
    b=packet(2,12); b.ptr[1:]=12; b.events=b.events[:12]
    obs=bank.encode(b.events,b.ptr,b.delta_t_s)
    u=torch.full((2,16,2),100.)
    f,wf=bank.query(obs,u,'fixed'); q,wq=bank.query(obs,u,'mano')
    assert f.shape==(2,64) and q.shape==(2,64)
    assert torch.all(wf[0,:,obs['mask'][0]] >= .1/12-1e-7)
    assert torch.allclose(wf[0].sum(-1),torch.ones(16))
    assert torch.isfinite(q).all() and torch.count_nonzero(q[1])==0
    assert torch.count_nonzero(wq[1])==0
    assert not torch.allclose(wf[0],wq[0])


def test_bank_and_xyz_sampling_use_the_same_valid_raw_indices():
    Bank,_,_=recovery_modules()
    from semkine.xyz_mesh import stride_sample
    b=packet(2,50); bank=Bank(max_nodes=32)
    obs=bank.encode(b.events,b.ptr,b.delta_t_s)
    src,mask=stride_sample(b.ptr,32)
    assert torch.equal(obs['src'],src) and torch.equal(obs['mask'],mask)


def test_n1_removes_both_observation_paths_but_preserves_state_and_grad_graph():
    m=make_model(null=True).train(); b=packet()
    p=m.forward_packet(b); changed=copy.deepcopy(b)
    changed.events[:,1]=40; changed.events[:,2]=25; changed.events[:,4]=1-changed.events[:,4]
    p2=m.forward_packet(changed)
    assert torch.equal(p,p2)
    p.square().sum().backward()
    for name,param in m.named_parameters():
        if name.startswith(('xyz.','observation_bank.')):
            assert param.grad is not None,name
            if not name.startswith(('xyz.fc1.','xyz.fc2.')):
                assert torch.count_nonzero(param.grad)==0,name
    changed.prev_state[:,0]+=.01
    assert not torch.equal(m.forward_packet(changed),p)


def test_nonempty_observation_gradients_survive_bf16_prefix_no_grad_cache():
    m=make_model(self_group=True).train(); b=packet()
    batch={'main':b,'prefix':[copy.deepcopy(b) for _ in range(3)],'global_row':torch.arange(2)}
    for prefix in batch['prefix']: prefix.target.zero_()
    m._trainer=type('Trainer',(),{'global_step':2500,'global_rank':0})()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        pred,_,_,_=m._predict_batch(batch)
        pred.float().square().sum().backward()
    assert m.training
    assert m.observation_bank.encoder.embed.weight.grad.abs().sum()>0
    assert m.xyz.rel[0].weight.grad.abs().sum()>0
    assert m.xyz.fc1.weight.grad.abs().sum()>0


def test_empty_packet_holds_exactly_and_keeps_every_parameter_in_graph():
    m=make_model().train(); b=packet(2,0)
    with torch.autocast('cpu',dtype=torch.bfloat16):
        pred=m.forward_packet(b)
    assert torch.equal(pred,b.prev_state)
    pred.sum().backward()
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)
    assert all(torch.count_nonzero(p.grad)==0 for p in m.parameters() if p.requires_grad)


def test_teacher_skips_prefix_and_does_not_read_targets_for_state():
    m=make_model(self_group=False).train(); b=packet()
    batch={'main':b,'prefix':[None,None,None],'global_row':torch.arange(2)}
    pred,truth,_,packed=m._predict_batch(batch)
    assert torch.equal(packed.prev_state,b.prev_state)
    assert torch.equal(truth,b.target)
    assert m.xdl_burn_rows==0 and m.xdl_self_rows==0
    assert torch.equal(pred,m.forward_packet(b))


def test_final_null_does_not_hide_nonfinite_observations():
    m=make_model(null=True).eval(); b=packet()
    with torch.no_grad(): m.observation_bank.compress[0].weight.fill_(float('nan'))
    with pytest.raises(FloatingPointError): m.forward_packet(b)


def test_subset_burn_matches_full_burn_conditioning_loss_and_main_gradients():
    from dataclasses import replace
    a=make_model(self_group=True).train()
    b=copy.deepcopy(a).train()
    main=packet(8,12)
    prefixes=[packet(8,12) for _ in range(3)]
    for index,prefix in enumerate(prefixes):
        prefix.target.zero_(); prefix.events[:,1]+=index
    previous=None
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):
        b.eval()
        for prefix in prefixes:
            p=prefix if previous is None else replace(prefix,prev_state=previous)
            previous=b.forward_packet(p).detach()
    b.train()
    # Frozen SHA-counter decisions for seed3407, step2500, rows0..7.
    expected=main.prev_state.clone()
    selected=torch.tensor([0,2,3,4,7])
    expected.index_copy_(0,selected,previous.index_select(0,selected))
    a._trainer=type('Trainer',(),{'global_step':2500,'global_rank':0})()
    packed={'main':main,'prefix':prefixes,'global_row':torch.arange(8)}
    with torch.autocast('cpu',dtype=torch.bfloat16):
        pa,truth,_,actual=a._predict_batch(packed)
        pb=b.forward_packet(replace(main,prev_state=expected))
        la=(pa.float()-truth).square().mean()
        lb=(pb.float()-truth).square().mean()
    assert torch.allclose(actual.prev_state,expected,atol=2e-5,rtol=2e-4)
    assert torch.allclose(la,lb,atol=2e-6,rtol=2e-3)
    la.backward();lb.backward()
    for (na,aa),(nb,bb) in zip(a.named_parameters(),b.named_parameters()):
        assert na==nb
        assert aa.grad is not None and bb.grad is not None,na
        assert torch.allclose(aa.grad,bb.grad,atol=2e-5,rtol=2e-2),na
    assert a.xdl_self_rows==5 and a.xdl_burn_rows==15
    assert a.training


def test_all_five_arms_have_paired_initial_parameters_and_no_target_leak():
    _,_,Model=recovery_modules()
    states=[];predictions=[]
    for arm in ('f0','q0','f1','q1','n1'):
        cfg=load_config(str(ROOT/f'configs/xdl/xdl_{arm}_s3407.yaml'))
        cfg['MANO']['NPZ']=str(ROOT/'assets/mano_right.npz')
        torch.manual_seed(3407)
        m=Model(cfg).eval()
        states.append(m.state_dict())
        with torch.no_grad():
            m.xyz.fc2.weight.normal_(0,.0002)
            clean=packet(1,12);changed=copy.deepcopy(clean)
            changed.target.fill_(float('nan'))
            assert torch.equal(m.forward_packet(clean),m.forward_packet(changed))
            predictions.append(m.forward_packet(clean))
    for state in states[1:]:
        assert state.keys()==states[0].keys()
        assert all(torch.equal(state[k],states[0][k]) for k in state)
    # f0/f1 and q0/q1 share the same inference behavior at equal weights.
    assert torch.equal(predictions[0],predictions[2])
    assert torch.equal(predictions[1],predictions[3])


def test_ragged_subset_retains_raw_ids_clock_context_and_row_boundaries():
    from semkine.xyz_recovery_study import select_packet_rows
    original=packet(3,12)
    original.t_start_us=torch.tensor([0,50000,100000])
    original.betas[:,0]=torch.tensor([1.,2.,3.])
    chosen=select_packet_rows(original,torch.tensor([2,0]))
    chosen.validate()
    assert torch.equal(chosen.ptr,torch.tensor([0,12,24]))
    assert torch.equal(chosen.packet(0),original.packet(2))
    assert torch.equal(chosen.packet(1),original.packet(0))
    assert torch.equal(chosen.t_start_us,torch.tensor([100000,0]))
    assert torch.equal(chosen.betas[:,0],torch.tensor([3.,1.]))


def c0_high_response_model(burn_mode):
    """Read-only old C0 response mapped into the common new readout for a test.

    This is a diagnostic fixture, never a formal training initializer. Bank and
    new state columns are zeroed only in this test to retain the trained C0 map.
    """
    _,_,Model=recovery_modules()
    cfg=load_config(str(ROOT/'configs/xdl/xdl_q1_s3407.yaml'))
    cfg['MANO']['NPZ']=str(ROOT/'assets/mano_right.npz')
    cfg['MODEL']['XDL_BURN_MODE']=burn_mode
    model=Model(cfg).float()
    path=ROOT/'outputs/semkine/xa_c0_s3407/xa_c0_s3407-step=6000.ckpt'
    old=torch.load(path,map_location='cpu')['state_dict']
    new=model.state_dict()
    for name in new:
        if name in old and new[name].shape==old[name].shape:
            new[name].copy_(old[name])
    model.load_state_dict(new,strict=True)
    with torch.no_grad():
        model.xyz.fc1.weight.zero_()
        model.xyz.fc1.weight[:,:544].copy_(old['xyz.fc1.weight'][:,:544])
        model.xyz.fc1.weight[:,611:659].copy_(old['xyz.fc1.weight'][:,544:592])
    return model


def test_full_burn_fallback_preserves_whole_batch_response_and_coin_selection():
    from dataclasses import replace
    a=c0_high_response_model('full').train()
    b=copy.deepcopy(a).train()
    main=packet(8,12)
    prefixes=[packet(8,12) for _ in range(3)]
    for i,prefix in enumerate(prefixes):
        prefix.target.zero_();prefix.events[:,1]+=i
    previous=None
    b.eval()
    with torch.no_grad(),torch.autocast('cpu',dtype=torch.bfloat16,cache_enabled=False):
        for prefix in prefixes:
            previous=b.forward_packet(prefix if previous is None else replace(prefix,prev_state=previous)).detach()
    b.train()
    expected=main.prev_state.clone();selected=torch.tensor([0,2,3,4,7])
    expected.index_copy_(0,selected,previous.index_select(0,selected))
    a._trainer=type('Trainer',(),{'global_step':2500,'global_rank':0})()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        pa,truth,_,packed=a._predict_batch({'main':main,'prefix':prefixes,'global_row':torch.arange(8)})
        pb=b.forward_packet(replace(main,prev_state=expected))
        la=(pa.float()-truth).square().mean();lb=(pb.float()-truth).square().mean()
    assert torch.equal(packed.prev_state,expected)
    assert torch.equal(pa,pb)
    la.backward();lb.backward()
    assert all(torch.equal(aa.grad,bb.grad) for aa,bb in zip(a.parameters(),b.parameters()))
    assert a.xdl_self_rows==5 and a.xdl_burn_rows==24
    assert a.training


def test_full_mode_teacher_and_zero_probability_still_skip_all_prefixes():
    for self_group in (False,True):
        m=make_model(self_group=self_group,burn_mode='full').train()
        m._trainer=type('Trainer',(),{'global_step':0,'global_rank':0})()
        p=packet()
        _,_,_,packed=m._predict_batch({'main':p,'prefix':[None]*3,'global_row':torch.arange(2)})
        assert torch.equal(packed.prev_state,p.prev_state)
        assert m.xdl_burn_rows==0 and m.xdl_self_rows==0
