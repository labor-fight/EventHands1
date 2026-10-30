import pytest
import torch
from tools.xag import fstar_failure_safe as F


def test_numerical_failure_retains_all_rows_and_isolates_failed_sample():
    h=torch.zeros(3,2,1);h[:,0,0]=torch.tensor([1.,2.,3.]);target=torch.ones(3,51)
    factory=lambda a,b:lambda features:features[:,0,:1].expand(-1,51)*.1
    def optimizer(call,features,support,std,target,cfg,steps):
        if (features[:,0,0]==2).any():raise FloatingPointError('simulated arithmetic failure')
        prediction=call(features)
        return features,prediction,{'initial_loss':torch.ones(len(features)),'best_loss':torch.zeros(len(features)),
            'max_bound_violation':torch.zeros(len(features))},{}
    pred,audit=F.safe_optimize(factory,h,torch.ones(3,2),torch.ones(1),target,{},optimizer=optimizer)
    assert pred.shape==(3,51)
    assert [r['optimization_failed'] for r in audit]==[False,True,False]
    assert all(r['numerical_batch_retry'] for r in audit)
    assert audit[1]['fallback_original'] and audit[1]['completed_steps']==200
    assert torch.equal(pred[1],factory(1,2)(h[1:2])[0])


def test_programming_error_is_never_recorded_as_model_failure():
    def broken(*args,**kwargs):raise IndexError('bad implementation')
    with pytest.raises(IndexError):
        F.safe_optimize(lambda a,b:None,torch.zeros(1,2,1),torch.ones(1,2),torch.ones(1),torch.ones(1,51),{},optimizer=broken)


def test_nonfinite_original_loss_is_null_not_success():
    def failed(*args,**kwargs):raise FloatingPointError('bad optimizer')
    factory=lambda a,b:lambda h:torch.full((len(h),51),float('nan'))
    prediction,rows=F.safe_optimize(factory,torch.zeros(1,2,1),torch.ones(1,2),torch.ones(1),torch.ones(1,51),{},optimizer=failed)
    assert not torch.isfinite(prediction).any()
    assert rows[0]['initial_loss']==[None] and rows[0]['best_loss']==[None]
    assert rows[0]['optimization_failed'] and rows[0]['fallback_original']


def test_successful_wrapper_leaves_optimizer_prediction_bitwise_unchanged():
    torch.manual_seed(4);h=torch.randn(1,3,2);support=torch.ones(1,3);std=torch.tensor([.2,.3]);target=torch.randn(1,51)*.1
    factory=lambda a,b:lambda features:features.flatten(1)[:,:1].expand(-1,51)*.1
    _,reference,audit,_=F.B.optimize(factory(0,1),h,support,std,target,{},steps=200)
    prediction,rows=F.safe_optimize(factory,h,support,std,target,{},steps=200)
    assert torch.equal(prediction,reference)
    assert not rows[0]['optimization_failed'] and rows[0]['completed_steps']==200
