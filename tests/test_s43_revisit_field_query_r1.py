"""R1 statistical controls; no new task scores."""
import numpy as np
import torch
from tools.s43_revisit import field_query_r1 as p


def test_context_reader_ignores_both_measurement_banks_without_bias_leak():
    torch.set_num_threads(2);torch.manual_seed(3407);m=p.make_reader('context_only')
    c=torch.randn(2,70);g=torch.randn(2,40,180,240);l=torch.randn(2,8,90,5,7,7)
    a=m(g,l,c);b=m(torch.zeros_like(g),torch.zeros_like(l),c)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.sum().backward();assert all(v.grad is None for v in m.global_encoder.parameters())
    assert all(v.grad is None for v in m.local_encoder.parameters())


def test_all_arms_have_same_allocated_architecture_and_paired_initialization():
    torch.manual_seed(3407);a=p.make_reader('anatomical').state_dict()
    for arm in p.ARMS:
        torch.manual_seed(3407);b=p.make_reader(arm).state_dict()
        assert sum(v.numel() for v in b.values())==874319
        for name in a:torch.testing.assert_close(a[name],b[name],rtol=0,atol=0)


def test_fixed_kernel_statistics_are_not_refitted_on_new_library():
    rng=np.random.default_rng(1);x=rng.random((12,5520)).astype(np.float32);c=rng.random((12,70)).astype(np.float32);y=rng.random((12,63))
    full=p.ref.LowCapacityRBF().fit(x,c,y);stats={k:getattr(full,k) for k in ('scale','cm','cs','context_scale','bandwidth')}
    nominal=p.fit_kernel(x[:4],c[:4],y[:4],stats)
    for key in stats:np.testing.assert_array_equal(getattr(nominal,key),stats[key])
    assert nominal.library.shape[0]==4 and nominal.bandwidth==full.bandwidth


def test_same_fixed_transform_reproduces_original_solve():
    rng=np.random.default_rng(2);x=rng.random((8,5520)).astype(np.float32);c=rng.random((8,70)).astype(np.float32);y=rng.random((8,63))
    head=p.ref.LowCapacityRBF().fit(x,c,y);stats={k:getattr(head,k) for k in ('scale','cm','cs','context_scale','bandwidth')}
    same=p.fit_kernel(x,c,y,stats);np.testing.assert_array_equal(head.predict(x,c),same.predict(x,c))
