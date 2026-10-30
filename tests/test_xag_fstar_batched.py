import pytest
import torch
from tools.xag.diagnose import optimize_h3
from tools.xag.fstar_batched import optimize


def decoder(x):
    # Independent same-width scalar observations; no batch reduction.
    return x.flatten(1)[:, :1].expand(-1, 51) * .1


def test_batched_sum_preserves_per_example_gradients_and_adam():
    torch.manual_seed(2);h=torch.randn(4,3,2);support=torch.tensor([[1,1,1],[1,0,1],[0,0,0],[1,1,0]],dtype=torch.float32)
    std=torch.tensor([.3,.1]);target=torch.randn(4,51)*.1
    hbatch,pbatch,audit,snapshot=optimize(decoder,h,support,std,target,{},steps=12,snapshot=True)
    for i in range(4):
        hi,pi,ai,si=optimize(decoder,h[i:i+1],support[i:i+1],std,target[i:i+1],{},steps=12,snapshot=True)
        assert torch.equal(snapshot['gradient'][i:i+1],si['gradient'])
        assert torch.equal(snapshot['exp_avg'][i:i+1],si['exp_avg'])
        assert torch.equal(snapshot['exp_avg_sq'][i:i+1],si['exp_avg_sq'])
        assert torch.equal(hbatch[i:i+1],hi)
        assert torch.equal(pbatch[i:i+1],pi)
    assert audit['max_bound_violation'].max()==0


def test_optimized_b1_matches_original_full200_bitwise():
    torch.manual_seed(4);h=torch.randn(1,3,2);support=torch.ones(1,3);std=torch.tensor([.2,.3]);target=torch.randn(1,51)*.1
    old_h,old_prediction,old=optimize_h3(decoder,h,support,std,target,{})
    new_h,new_prediction,new,_=optimize(decoder,h,support,std,target,{})
    assert torch.equal(old_h,new_h)
    assert torch.equal(old_prediction,new_prediction)
    assert torch.equal(old['best_loss'],new['best_loss'])


def test_zero_radius_and_unsupported_features_are_exact():
    h=torch.randn(3,4,2);support=torch.zeros(3,4);std=torch.ones(2);target=torch.zeros(3,51)
    best,pred,audit,_=optimize(decoder,h,support,std,target,{},radius=0)
    assert torch.equal(best,h) and torch.equal(pred,decoder(h)) and audit['steps']==0
    best,pred,audit,_=optimize(decoder,h,support,std,target,{},steps=3)
    assert torch.equal(best,h) and torch.equal(pred,decoder(h))


def test_nonfinite_batch_never_becomes_valid_artifact():
    h=torch.zeros(2,3,2);target=torch.ones(2,51);target[1,0]=float('nan')
    with pytest.raises(FloatingPointError):
        optimize(decoder,h,torch.ones(2,3),torch.ones(2),target,{},steps=3)
