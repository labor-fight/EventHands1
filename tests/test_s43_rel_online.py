"""Standalone client input, quantization, source accounting and no-label IO."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn

from tools.s43_rel.online import MMapEventStream, load_calibration
from tools.s43_rel.model import S43RelationModel


def save_events(path, timestamps):
    times=np.asarray(timestamps,np.int64)
    np.save(path,np.column_stack((np.zeros(len(times),np.int64),np.ones(len(times),np.int64),times,
                                 np.arange(len(times))%2)))


def test_mmap_us_quantizes_groups_and_never_sorts_intrabin(tmp_path):
    path=tmp_path/'events.npy'
    save_events(path,[1999,1000,1500,2999,2000,3001])
    stream=MMapEventStream(path,'us',start_ms=0,stop_ms=5,validation_chunk=2)
    assert isinstance(stream.events,np.memmap)
    assert np.shares_memory(stream.timestamps,stream.events)
    packets=[]
    for ms in range(5):
        packet,source=stream.packet(ms)
        packets.append(packet)
    combined=np.concatenate(packets)
    np.testing.assert_array_equal(combined[:,2],[1,1,1,2,2,3])
    np.testing.assert_array_equal(combined[:,3],np.arange(6)%2)
    assert stream.source_end == 6
    with pytest.raises(ValueError,match='exactly once'):
        stream.packet(4)


def test_unsorted_native_groups_rejected_without_sorting(tmp_path):
    path=tmp_path/'bad.npy';save_events(path,[0,1000,999])
    with pytest.raises(ValueError,match='out-of-order'):
        MMapEventStream(path,'us',validation_chunk=2)


def test_explicit_coverage_and_default_mean_shape(tmp_path):
    path=tmp_path/'events.npy';save_events(path,[10,11])
    with pytest.raises(ValueError,match='include every'):
        MMapEventStream(path,'ms',start_ms=11,stop_ms=12)
    calibration=tmp_path/'calibration.json'
    calibration.write_text(json.dumps({'K_cam':[[1000,0,320],[0,1000,240],[0,0,1]]}))
    K,beta,initial,metadata=load_calibration(calibration)
    assert np.count_nonzero(beta)==0
    assert initial[2] == np.float32(.6)
    assert metadata['GT_shape_required'] is False


def test_client_runs_without_dataset_reader_and_consumes_tail(monkeypatch,tmp_path):
    from tools.s43_rel import online,replay
    class Base(nn.Module):
        sparse_trunk=True
        active_head=False
        transl_head=None
        def __init__(self):
            super().__init__();self.conv1=nn.Conv2d(2,3,3,padding=1)
            self.rn=nn.Module();self.rn.fc=nn.Linear(3,51)
        def forward(self,image,prev,betas=None,camera_K=None):
            assert torch.count_nonzero(betas)==0
            return self.rn.fc(self.conv1(image.permute(0,3,1,2)).mean((2,3)))+prev
        def _fk(self,pose,betas):
            return pose[:,:3,None].transpose(1,2).expand(-1,778,-1), pose[:,:3,None].transpose(1,2).expand(-1,21,-1)
    path=tmp_path/'events.npy';save_events(path,np.arange(260))
    calibration=tmp_path/'calibration.json';calibration.write_text(json.dumps({'K_cam':[[1000,0,320],[0,1000,240],[0,0,1]]}))
    marker=tmp_path/'marker';marker.write_text('identity fixture')
    base=Base().eval();model=S43RelationModel(base,'s43').eval()
    cfg={'REL':{'BASE_CHECKPOINT':str(marker),'BASE_CONFIG':str(marker)}}
    monkeypatch.setattr(online,'load_relation_model',lambda *args,**kwargs:(model,cfg))
    monkeypatch.setattr(online,'load_legacy',lambda *args,**kwargs:(base,cfg))
    def forbidden(*args,**kwargs):
        raise AssertionError('standalone client tried to instantiate a dataset reader')
    monkeypatch.setattr(replay,'SequenceReader',forbidden)
    out=tmp_path/'output.jsonl'
    summary=online.run(path,calibration,marker,marker,out,start_ms=0,stop_ms=260,device='cpu',threads=1)
    rows=[json.loads(line) for line in out.read_text().splitlines()]
    assert summary['all_external_events_consumed_once'] and summary['source_event_count']==260
    assert summary['native_ms_packets']==260 and summary['warmup_canonical_windows']==2
    assert summary['outputs']==len(rows)==3 and summary['tail_events']==summary['tail_ms']==10
    assert [row['evidence_cutoff_ms'] for row in rows]==[150,200,250]
    assert [row['t_query_ms'] for row in rows]==[149,199,249]
    assert all(np.asarray(row['vertices_m']).shape==(778,3) and np.asarray(row['joints_m']).shape==(21,3) for row in rows)
    assert summary['GT_or_target_files_read'] is False
    assert all(summary['source_files_unchanged'].values())
    with pytest.raises(FileExistsError):
        online.run(path,calibration,marker,marker,out,device='cpu')
