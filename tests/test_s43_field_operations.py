"""Executed dense MAC count versus separately labelled occupancy theory."""
from types import SimpleNamespace
import torch
from torch import nn
from tools.s43_revisit.s43_field_operations import NNOperations


class MaskedBN2d(nn.Module):
    def forward(self,x,m):return x*m


class RN(nn.Module):
    def __init__(self):
        super().__init__();self.conv1=nn.Conv2d(3,4,1,bias=False);self.bn1=MaskedBN2d();self.fc=nn.Linear(4,51)
    def pooled(self,x,m):return self.bn1(self.conv1(x),m).mean((2,3)),None


class Model(nn.Module):
    def __init__(self):
        super().__init__();self.conv1=nn.Conv2d(2,3,3,padding=1,bias=False);self.rn=RN()
    def forward(self,x,m):return self.rn.fc(self.rn.pooled(self.conv1(x)*m,m)[0])


def test_masked_dense_execution_is_not_charged_as_sparse_and_hooks_restore():
    m=Model();x=torch.ones(1,2,4,4);mask=torch.zeros(1,1,4,4);mask[:,:,1,1]=1
    before=m(x,mask).detach();runner=SimpleNamespace(model=m,phase_category='steady')
    with NNOperations(runner) as op:after=m(x,mask).detach()
    torch.testing.assert_close(before,after,rtol=0,atol=0)
    r=op.report(dict(steady=1))['phases']['steady']
    assert r['actual_dense_operator_equivalent_MACs_total']==864+192+204
    assert r['occupancy_output_site_theoretical_MACs_total']==54+12+204
    assert not any(mod._forward_hooks for mod in m.modules())
    torch.testing.assert_close(before,m(x,mask),rtol=0,atol=0)


def test_local720_patch_encoder_and_head_has_67531776_dense_FLOPs():
    m=Model();m.local_encoder=nn.Sequential(nn.Conv2d(5,8,3,padding=1),nn.SiLU(),nn.Conv2d(8,8,3,padding=1),nn.SiLU(),nn.AdaptiveAvgPool2d(1))
    m.local_head=nn.Sequential(nn.Linear(5760,128),nn.SiLU(),nn.Linear(128,51))
    with torch.no_grad(),NNOperations(SimpleNamespace(model=m,phase_category='steady')) as op:
        m.local_head(m.local_encoder(torch.zeros(720,5,7,7)).reshape(1,5760))
    r=op.report(dict(steady=1))['phases']['steady']
    assert r['local_patch_encoder_and_head_FLOPs_per_output']==67531776
    assert r['occupancy_output_site_theoretical_NN_FLOPs_per_output']==67531776
