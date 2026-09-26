"""Equivalent row gathering for deterministic X1 training, including repeated neighbours."""
import copy

import pytest
import torch

from semkine import event_hier as EH
from semkine.event_gnn import EdgeConv, gather_node_features
from test_x1_event_hier import _batch, _small


def expanded(h,idx):
    b,n,c=h.shape
    return h.gather(1,idx.reshape(b,-1,1).expand(b,idx[0].numel(),c)).reshape(*idx.shape,c)


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('noncontiguous',[False,True])
def test_selected_values_and_repeated_index_gradients(dtype,noncontiguous):
    torch.manual_seed(81)
    h=torch.randn(3,17,7,dtype=dtype)
    if noncontiguous:h=h.transpose(1,2).contiguous().transpose(1,2)
    a=h.detach().requires_grad_();b=h.detach().clone().requires_grad_()
    idx=torch.randint(0,17,(3,11,9));idx[:,:,0:4]=0
    old,new=expanded(a,idx),gather_node_features(b,idx)
    assert torch.equal(old,new)
    weight=torch.randn_like(old)
    (old*weight).sum().backward();(new*weight).sum().backward()
    tol=(1e-12,1e-10) if dtype==torch.float64 else (1e-5,1e-4)
    torch.testing.assert_close(a.grad,b.grad,atol=tol[0],rtol=tol[1])


def test_x1_full_encoder_and_gradients_preserved(monkeypatch):
    new=_small(max_nodes=64,hidden=8,feat_dim=16)
    old=copy.deepcopy(new)
    for module in old.modules():
        if isinstance(module,EdgeConv):module.compact_gather=False
    packet=_batch([0,7,111],seed=819)
    pred_new=new(*packet);pred_new.square().sum().backward()
    monkeypatch.setattr(EH,'gather_node_features',expanded)
    pred_old=old(*packet);pred_old.square().sum().backward()
    assert torch.equal(pred_new,pred_old)
    assert new.state_dict().keys()==old.state_dict().keys()
    for (name,p),(old_name,q) in zip(new.named_parameters(),old.named_parameters()):
        assert name==old_name
        torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-4)


def test_other_event_gnns_keep_original_gather_by_default():
    assert EdgeConv(8).compact_gather is False
