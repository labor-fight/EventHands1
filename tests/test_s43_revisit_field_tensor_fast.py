"""Engineering parity only: exact frozen integer/Torch readout, no labels."""
from dataclasses import replace
import hashlib
import numpy as np
import pytest
import torch

from tools.s43_revisit.field_core import TimeFieldConfig,TimeFieldMemory
from tools.s43_revisit.field_tensor import tensorize as reference
from tools.s43_revisit.field_tensor_fast import (aggregate_memories,aggregate_snapshots,
    tensorize,tensorize_memories)


def make(res=2,T=73,boundary=3,seed=3407):
    m=TimeFieldMemory(TimeFieldConfig(resolution=res),seq_id='engineering',boundary_ms=boundary)
    rng=np.random.default_rng(seed)
    e=np.column_stack((rng.integers(0,240,317),rng.integers(0,180,317),
                       rng.integers(boundary,T,317),rng.integers(0,2,317)))
    e=e[np.argsort(e[:,2],kind='stable')]
    m.ingest(e,T,0,source_range=(900,900+len(e)))
    return m


def state_hash(m):
    s=m.state_dict();h=hashlib.sha256(repr(s['scalars']).encode())
    for a in s['arrays'].values():h.update(a.tobytes())
    return h.hexdigest()


def same(a,b):
    assert torch.equal(a.global40,b.global40)
    assert torch.equal(a.cell3,b.cell3)
    assert torch.equal(a.observed_support,b.observed_support)
    assert a.count_payload_bytes==b.count_payload_bytes
    assert a.reference_grid_bytes==b.reference_grid_bytes


@pytest.mark.parametrize('res',[1,2])
@pytest.mark.parametrize('T',[7,50,73])
def test_snapshot_and_direct_native_ring_are_exact(res,T):
    m=make(res,T);s=m.readout();before=state_hash(m);rng=torch.get_rng_state().clone()
    integer,exp,widths,_=aggregate_memories([m]);r=s.rasterize()
    np.testing.assert_array_equal(integer[0],r['fine_counts'])
    np.testing.assert_array_equal(exp[0],r['bin_exposure_ms'])
    np.testing.assert_array_equal(widths[0],r['bin_width_ms'])
    same(reference([s]),tensorize([s]));same(reference([s]),tensorize_memories([m]))
    assert state_hash(m)==before and torch.equal(rng,torch.get_rng_state())


def test_pending_groups_duplicate_multiplicity_empty_gap_and_no_alias():
    m=TimeFieldMemory(boundary_ms=3)
    m.ingest(np.array([[1,1,3,1],[1,1,3,1]],np.int64),3,0,source_range=(50,52))
    same(reference([m.readout()]),tensorize_memories([m]))
    assert not tensorize_memories([m]).observed_support.any()
    m.ingest(np.array([[1,1,3,1]],np.int64),4,1,source_range=(52,53))
    first=tensorize_memories([m]);saved=first.global40.clone()
    assert aggregate_memories([m])[0].sum()==3
    m.ingest(np.empty((0,4),np.int64),1003,2,source_range=(53,53))
    same(reference([m.readout()]),tensorize_memories([m]))
    assert torch.equal(saved,first.global40)
    assert not tensorize_memories([m]).observed_support.any()
    assert torch.all(tensorize_memories([m]).global40[:,16:24]==1)


def test_max_uint32_counts_and_exposure_holes_match_without_signed_overflow():
    m=make();m._counts.fill(np.iinfo(np.uint32).max)
    m._times[25]=-1  # Exact handling even for a restored/test ring with a hole.
    s=m.readout();a=aggregate_memories([m]);b=aggregate_snapshots([s])
    for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
    assert a[0].max()>np.iinfo(np.uint32).max
    same(reference([s]),tensorize_memories([m]))


def test_batch64_distinct_prefixes_integer_exact_and_owned_outputs():
    # Repeated references are permitted read-only lanes; no storage escapes.
    ms=[make(T=50,seed=i) for i in range(2)]
    ms=(ms*32);ss=[m.readout() for m in ms]
    a,exp,widths,_=aggregate_memories(ms)
    assert a.shape==(64,8,90,120,2)
    for i,s in enumerate(ss):
        np.testing.assert_array_equal(a[i],s.rasterize()['fine_counts'])
    a.fill(0);assert aggregate_memories(ms)[0].sum()>0
    assert exp.shape==widths.shape==(64,8)


def test_cutoff_lifecycle_configuration_and_fabricated_overflow_guards():
    m=make();s=m.readout()
    for cut in (m.watermark_ms-1,float(m.watermark_ms),[m.watermark_ms,m.watermark_ms]):
        with pytest.raises(ValueError):tensorize_memories([m],cut)
    same(tensorize_memories([m]),tensorize_memories([m],m.watermark_ms))
    with pytest.raises(ValueError):tensorize_memories([])
    huge=s.counts.astype(np.uint64);huge[0,0,0,0]=2**63
    with pytest.raises(OverflowError):tensorize([replace(s,counts=huge)])
    with pytest.raises(OverflowError):reference([replace(s,counts=huge)])
    m.release()
    with pytest.raises(RuntimeError):tensorize_memories([m])


@pytest.mark.skipif(not torch.cuda.is_available(),reason='explicit allowed GPU only')
def test_cuda_old_snapshot_and_direct_ring_are_bitwise_equal():
    ms=[make(),make(T=11,seed=8)]
    ss=[m.readout() for m in ms]
    same(reference(ss,'cuda'),tensorize(ss,'cuda'))
    same(reference(ss,'cuda'),tensorize_memories(ms,device='cuda'))
