"""Opt-in nonempty state accumulation; EP0 default contract remains separate."""
import copy
import dataclasses
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parent))
from test_s37_empty_state_precision import (_config,_packet,_amp,_execution_contract,
                                          device,precision,MNISTModel)

@pytest.fixture
def model(device):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(892)
        cfg=_config();cfg['MODEL']['STATE_ACCUM_FP32']=True
        m=MNISTModel(cfg).to(device).eval()
    assert m.state_accum_fp32
    return m

def raw_delta(m,p):
    old=m.predict_delta
    try:
        m.predict_delta=False
        return m.forward_packet(p)
    finally:m.predict_delta=old

@pytest.mark.parametrize('kind',['empty','mixed','nonempty'])
def test_promoted_same_amp_delta_value_and_autograd_oracle(model,device,precision,kind):
    packet=_packet(model,kind)
    ref=copy.deepcopy(model)
    rp=dataclasses.replace(packet,prev_state=packet.prev_state.detach().clone().requires_grad_())
    with _amp(device,precision):
        actual=model.forward_packet(packet)
        raw=raw_delta(ref,rp)
        empty=(rp.counts<=0).unsqueeze(1)
        expected=torch.where(empty,rp.prev_state,rp.prev_state.float()+raw.float())
    assert actual.dtype==torch.float32 and torch.equal(actual,expected)
    w=torch.linspace(-.71,.63,51,device=device)
    ga=torch.autograd.grad((actual*w).sum(),[packet.prev_state,*model.parameters()],allow_unused=True)
    gb=torch.autograd.grad((expected*w).sum(),[rp.prev_state,*ref.parameters()],allow_unused=True)
    assert len(ga)==len(gb)
    for a,b in zip(ga,gb):
        assert (a is None)==(b is None)
        if a is not None:assert torch.equal(a,b)

def test_nonempty_zero_and_small_delta_preserve_state_precision(model,device,precision):
    packet=_packet(model,'nonempty')
    # Real nonempty encoder/routing still run; zero learned weights isolate the update boundary.
    with torch.no_grad():
        for p in model.parameters():p.zero_()
        with _amp(device,precision):actual=model.forward_packet(packet)
        assert torch.equal(actual,packet.prev_state)
        model.prev_mlp[2].bias.fill_(.0003)
        with _amp(device,precision):
            raw=raw_delta(model,packet)
            actual=model.forward_packet(packet)
        assert torch.equal(actual,packet.prev_state+raw.float())
        assert torch.all(actual[:,2]!=packet.prev_state[:,2])
        if precision=='bf16':
            old=(raw+packet.prev_state.to(raw.dtype)).float()
            assert not torch.equal(old,actual)
            # BF16 z updates below its ULP are swallowed by the legacy boundary.
            assert torch.equal(old[:,2],packet.prev_state[:,2].bfloat16().float())
        state=packet.prev_state.detach().clone()
        expected=state.clone()
        for _ in range(5):
            with _amp(device,precision):state=model.forward_packet(dataclasses.replace(packet,prev_state=state))
            expected=expected+raw.float()
        assert torch.equal(state,expected)

def test_fp32_flag_preserves_output_and_full_parameter_gradients(model,device):
    packet=_packet(model,'mixed');old=copy.deepcopy(model);old.state_accum_fp32=False
    rp=dataclasses.replace(packet,prev_state=packet.prev_state.detach().clone().requires_grad_())
    a=model.forward_packet(packet);b=old.forward_packet(rp)
    assert torch.equal(a,b) and model.state_dict().keys()==old.state_dict().keys()
    # Nonlinear scalar exposes gradient dependence on the actual output.
    ga=torch.autograd.grad(a.square().sum(),[packet.prev_state,*model.parameters()],allow_unused=True)
    gb=torch.autograd.grad(b.square().sum(),[rp.prev_state,*old.parameters()],allow_unused=True)
    for x,y in zip(ga,gb):
        assert (x is None)==(y is None)
        if x is not None:assert torch.equal(x,y)

def test_config_is_explicit_and_rejects_unsupported_absolute():
    assert 'STATE_ACCUM_FP32' in MNISTModel.MODEL_KEYS
    cfg=_config();cfg['MODEL'].update(STATE_ACCUM_FP32=True,PREDICT_DELTA=False)
    with pytest.raises(ValueError,match='STATE_ACCUM_FP32'):MNISTModel(cfg)

@pytest.mark.parametrize('state_dtype',[torch.float16,torch.bfloat16,torch.float64])
def test_state_dtype_promotion_is_at_least_fp32(model,device,precision,state_dtype):
    p=_packet(model,'mixed',state_dtype=state_dtype)
    with _amp(device,precision):
        raw=raw_delta(model,p)
        out=model.forward_packet(p)
    dtype=torch.promote_types(state_dtype,torch.float32)
    expected=torch.where((p.counts<=0).unsqueeze(1),p.prev_state,
                         p.prev_state.to(dtype)+raw.to(dtype))
    assert out.dtype==dtype and torch.equal(out,expected)

def test_real_loss_backward_and_optimizer_step(model,device,precision):
    p=_packet(model,'mixed')
    before={n:v.detach().clone() for n,v in model.named_parameters()}
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    with _amp(device,precision):
        out=model.forward_packet(p)
        loss,_=model._compute_loss(out,p.target,p.betas)
        objective=loss.float().log10()
    assert torch.isfinite(objective)
    objective.backward()
    gradients=[v.grad for v in model.parameters() if v.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum()>0 for g in gradients)
    optimizer.step()
    assert any(not torch.equal(before[n],v) for n,v in model.named_parameters())
    assert all(torch.isfinite(v).all() for v in model.parameters())
