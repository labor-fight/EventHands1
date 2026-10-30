"""Probe transport/reader engineering, using toy observations and no task GT."""
import inspect
import json

import numpy as np
import pytest

from tools.s43_revisit import witness_probe as p


def test_native_packet_transport_preserves_absolute_source_offsets_and_silence():
    raw=np.asarray([[1,2,3,0],[2,3,3,1],[3,4,6,0]],np.int64)
    packets=list(p.packet_iter(raw,3,8,source_start=123))
    assert [ms for ms,_,_ in packets]==[3,4,5,6,7]
    assert [span for _,_,span in packets]==[(123,125),(125,125),(125,125),(125,126),(126,126)]
    np.testing.assert_array_equal(np.concatenate([ev for _,ev,_ in packets]),raw)


@pytest.mark.parametrize('raw',[np.array([[1.,2.,3.5,0.]]),np.array([[1,2,4,0],[1,2,3,0]]),np.array([[1,2,3,-1]])])
def test_bad_native_input_rejected_without_field_advance(raw):
    field=p.DenseFieldMemory(2,3)
    with pytest.raises(ValueError):field.ingest(raw,3)
    assert field.events==0 and field.watermark==3 and not field.count.any()


def test_dense_fields_preserve_count_mass_and_distinguish_zero_from_unobserved():
    raw=np.asarray([[1,2,3,0],[1,2,3,0],[2,3,6,1]],np.int64)
    for r in (1,2):
        field=p.DenseFieldMemory(r,3)
        for ms,ev,_ in p.packet_iter(raw,3,53):field.ingest(ev,ms)
        grid,meta=field.grid(53)
        assert grid.shape==(40,180,240) and np.all(grid[16:24]==1)
        # Fine input is native density, Q repeats each4px cell count16 times.
        assert np.isclose(np.expm1(grid[:16].astype(np.float64)).sum(),len(raw),rtol=1e-6)
        assert np.isclose(np.expm1(grid[24:].astype(np.float64)).sum()/16,len(raw),rtol=1e-6)
        assert meta['source_count']==3 and meta['native_fine_observed']==(r==1)
        for ms in range(53,103):field.ingest(np.empty((0,4),np.int64),ms)
        tail,_=field.grid(103)
        assert not tail[:16].any() and np.all(tail[16:24]==1) and not tail[24:].any()


def test_actual_core_nonzero_boundary_and_fullfine_count_parity():
    from tools.s43_revisit.witness_core import EpisodeConfig,EpisodeMemory
    from tools.s43_revisit.witness_geometry_runtime import CacheBoundMotionPolicy as MotionPolicy
    anchor=np.asarray([0,0,.6]+[0.]*48,np.float64);beta=np.zeros(10)
    K=np.asarray([[180.,0,120],[0,180,90],[0,0,1]])
    policy=MotionPolicy(anchor,beta,K,3,mode='witness')
    memory=EpisodeMemory(EpisodeConfig(policy='witness'),policy=policy).reset('toy',boundary_ms=3,policy=policy)
    raw=np.asarray([[100,90,3,0],[101,91,12,1]],np.int64);field=p.DenseFieldMemory(1,3)
    for ordinal,(ms,ev,span) in enumerate(p.packet_iter(raw,3,13)):
        memory.ingest(ev,ms+1,ordinal,source_range=span);field.ingest(ev,ms)
    snapshot=memory.readout(13);grid,_=field.grid(13)
    assert snapshot.partial_epoch_start_ms==13 and np.all(snapshot.epoch_start_ms==3)
    assert int(snapshot.rasterize()['fine'].sum())==2
    assert np.isclose(np.expm1(grid[:16]).sum(),2,rtol=1e-6)


def test_case_builder_restores_only_injection_and_never_accepts_targets(tmp_path):
    signature=inspect.signature(p.build_case_features)
    assert not any('target' in name or name in ('finger','sign') for name in signature.parameters)
    anchor=np.asarray([0,0,.6]+[0.]*48,np.float64);original=anchor.copy()
    K=np.asarray([[480.,0,320],[0,480.,240],[0,0,1.]])
    rows=p.build_case_features(np.asarray([[100,90,3,0]],np.int64),anchor,np.zeros(10),K,3,5,tmp_path,'toy',(5,),source_start=70)
    assert len(rows)==30 and all(r['target_not_read'] for r in rows)
    assert all(r['consumed_to_ms']==5 and r['full_input_events']==1 for r in rows)
    assert {tuple(r['context']) for r in rows}=={tuple(p.condition70(anchor,np.zeros(10),K))}
    np.testing.assert_array_equal(anchor,original)
    for arm in ('uniform','recency','density','generic_temporal','field2','fullfine'):
        assert len({r['feature_file'] for r in rows if r['arm']==arm})==1


def test_reader_local_nonlinearity_precedes_spatial_pool_and_gradients_flow():
    import torch
    torch.set_num_threads(2)
    model=p.make_cnn(10)
    assert isinstance(model.encoder[0],torch.nn.Conv2d) and model.encoder[0].kernel_size==(3,3)
    assert isinstance(model.encoder[1],torch.nn.SiLU) and isinstance(model.encoder[2],torch.nn.AvgPool2d)
    pred=model(torch.ones(2,40,180,240),torch.zeros(2,70))
    pred.square().mean().backward()
    assert pred.shape==(2,10) and model.encoder[0].weight.grad.abs().sum()>0


def test_gate_rejects_unfrozen_implementation(tmp_path):
    path=tmp_path/'invalid.json';path.write_text(json.dumps({'schema':'s43_witness_actual_sourcefreeze_v1','protocol_sha256':p.PROTOCOL_SHA,'engineering_passed':False}))
    with pytest.raises(RuntimeError,match='engineering gate'):p.require_gate(path)

