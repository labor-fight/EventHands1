"""Numerical phase, exposure and count-conservation checks; no task labels."""
import numpy as np
from dataclasses import replace
import pytest
import torch

from tools.s43_revisit.field_core import TimeFieldConfig,TimeFieldMemory
from tools.s43_revisit.field_tensor import tensorize


def make(resolution,cutoff=50):
    m=TimeFieldMemory(TimeFieldConfig(resolution=resolution),seq_id="fixture")
    rng=np.random.default_rng(3407)
    ev=np.column_stack((rng.integers(0,240,900),rng.integers(0,180,900),
                        rng.integers(0,cutoff,900),rng.integers(0,2,900)))
    ev=ev[np.argsort(ev[:,2],kind="stable")]
    m.ingest(ev,cutoff,0,source_range=(0,len(ev)))
    return m,m.readout(cutoff)


@pytest.mark.parametrize("resolution",[1,2])
@pytest.mark.parametrize("cutoff",[7,50])
def test_torch_grid_matches_frozen_numpy_phase_exposure_and_conservation(resolution,cutoff):
    m,s=make(resolution,cutoff);before=s.content_sha256();f=tensorize([s])
    np.testing.assert_allclose(f.global40[0].numpy(),s.to_grid40ch(),atol=3e-7,rtol=2e-7)
    assert f.global40.dtype==torch.float32 and f.cell3.shape==(1,8,3,180//resolution,240//resolution)
    assert before==s.content_sha256() and m.version==s.version
    # Coverage before a real sequence boundary is absent, not observed silence.
    exposure=s.rasterize()["bin_exposure_ms"]/s.rasterize()["bin_width_ms"]
    np.testing.assert_array_equal(f.global40[0,16:24,0,0],exposure.astype(np.float32))
    recovered=torch.expm1(f.cell3[:,:,:2]).sum().item()*resolution**2
    assert recovered==pytest.approx(900,abs=.001)
    assert f.count_payload_bytes<f.reference_grid_bytes


def test_empty_observed_field_is_not_active_just_because_exposure_is_nonzero():
    m=TimeFieldMemory();m.ingest(np.empty((0,4),np.int64),50,0,source_range=(0,0));s=m.readout(50)
    f=tensorize([s]);assert not f.observed_support.any()
    assert torch.count_nonzero(f.global40[:,:16])==0 and torch.all(f.global40[:,16:24]==1)


def test_support_retains_coarse_observations_outside_the_fine_cell():
    m=TimeFieldMemory();m.ingest(np.array([[0,0,2,1]],np.int64),50,0,source_range=(0,1))
    f=tensorize([m.readout(50)])
    assert int(f.observed_support.sum())==16  # the measured4px Qcell
    expected=f.global40[:,:16].gt(0).any(1,keepdim=True)|f.global40[:,24:].gt(0).any(1,keepdim=True)
    assert torch.equal(f.observed_support,expected)


def test_batch_lanes_keep_distinct_counts_and_support():
    _,a=make(2);_,b=make(2,7);f=tensorize([a,b])
    for i,s in enumerate((a,b)):
        torch.testing.assert_close(f.global40[i],tensorize([s]).global40[0],rtol=0,atol=0)
    assert not torch.equal(f.global40[0],f.global40[1])


def test_high_count_coarse_sums_round_only_after_integer_aggregation():
    _,s=make(2);counts=s.counts.copy();counts[25,1,1,0]=2**30+1
    high=replace(s,counts=counts)
    f=tensorize([high])
    np.testing.assert_allclose(f.global40[0].numpy(),high.to_grid40ch(),atol=3e-6,rtol=2e-7)


def test_reject_semantically_different_time_horizon():
    m=TimeFieldMemory(TimeFieldConfig(horizon_ms=100))
    with pytest.raises(ValueError):tensorize([m.readout(0)])


@pytest.mark.skipif(not torch.cuda.is_available(),reason="explicit allowed GPU verification separate")
def test_cuda_and_cpu_analytic_readout_agree():
    _,s=make(2);cpu=tensorize([s]);gpu=tensorize([s],"cuda")
    torch.testing.assert_close(gpu.global40.cpu(),cpu.global40,atol=3e-7,rtol=2e-7)
    torch.testing.assert_close(gpu.cell3.cpu(),cpu.cell3,atol=3e-7,rtol=2e-7)
    assert torch.equal(gpu.observed_support.cpu(),cpu.observed_support)
