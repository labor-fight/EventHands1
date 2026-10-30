"""External-source only client tests; no training, labels, aux or GPU."""
import json
import numpy as np
import pytest
import torch
from torch import nn
from tools.s43_revisit import s43_field_online as O


def inputs(tmp_path,unit='ms'):
    e=np.column_stack((np.full(700,90),np.full(700,80),np.repeat(np.arange(350),2),np.tile([0,1],350))).astype(np.int64)
    if unit=='us':e[:,2]=e[:,2]*1000+np.tile([1,999],350)
    np.save(tmp_path/'events.npy',e);K=[[600,0,320],[0,600,240],[0,0,1]]
    (tmp_path/'calibration.json').write_text(json.dumps(dict(K_source=K,boundary_ms=0,stop_ms=350)))
    return tmp_path/'events.npy',tmp_path/'calibration.json'


def test_mmap_offsets_us_quantization_and_calibration_single_scale(tmp_path):
    e,c=inputs(tmp_path,'us');r=O.NpyInputReader(e,c,time_unit='us')
    assert np.shares_memory(r.timestamps,r.events) and not isinstance(r.offsets,np.ndarray)
    p=r.packet(25,50);assert p['source_range']==(50,100)
    np.testing.assert_array_equal(p['events'][:,2],np.repeat(np.arange(25,50),2))
    d=json.loads(c.read_text());d['K_event']=(np.asarray(d.pop('K_source'))*np.array([[.375],[.375],[1]])).tolist();c.write_text(json.dumps(d))
    k,b,_=O.calibration(c);np.testing.assert_array_equal(k,r.K_cam);assert not b.any()
    r.close()


def test_invalid_order_and_run_discard_and_packet_cap_reject(tmp_path):
    e,c=inputs(tmp_path);r=O.NpyInputReader(e,c,max_logical_events=99)
    with pytest.raises(ValueError,match='cap'):r.packet(0,50)
    assert r.offsets[0]==0;r.close()
    d=json.loads(c.read_text());d['stop_ms']=349;c.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='discard'):O.NpyInputReader(e,c)
    d['stop_ms']=350;c.write_text(json.dumps(d));a=np.load(e);a[5,2]=0;np.save(e,a)
    with pytest.raises(ValueError,match='unordered'):O.NpyInputReader(e,c)


def test_provided_shape_is_explicit_and_default_ignores_calibration_shape(tmp_path):
    e,c=inputs(tmp_path);d=json.loads(c.read_text());d['beta_input']=[1.]*10;c.write_text(json.dumps(d))
    _,b,_=O.calibration(c);assert not b.any()
    _,b,_=O.calibration(c,'provided');np.testing.assert_array_equal(b,np.ones(10))
    d['K_event']=d['K_source'];c.write_text(json.dumps(d))
    with pytest.raises(ValueError,match='exactly one'):O.calibration(c)


def test_client_saves_full_mesh_and_fixedcommit_extraquery_without_dataset(tmp_path,monkeypatch):
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.p=nn.Parameter(torch.zeros(()));self.rn=nn.Module();self.rn.fc=nn.Identity()
        def forward(self,image,prev,betas,camera_K):return prev+.001+self.p
        def _fk(self,p,b):return p[:,:3,None].transpose(1,2).expand(-1,778,-1),p[:,:3,None].transpose(1,2).expand(-1,21,-1)
    e,c=inputs(tmp_path)
    payload=dict(arm='canonical',runtime_contract=dict(runtime_arm='legacy_s43',temporal_collapse=False,tensor_readout=True,state_fp32=True),base_checkpoint=str(e),base_config=str(c))
    monkeypatch.setattr(O.R,'load_model',lambda *a,**k:(Model(),payload))
    monkeypatch.setattr(O.R,'code_dependencies',lambda:{})
    def no_dataset(*a,**k):raise AssertionError('datasetinput reached')
    monkeypatch.setattr(O.R,'MeanShapeInputReader',no_dataset)
    out=tmp_path/'trace.jsonl';r=O.run(e,c,out,arm='canonical',hop=25,commit_ms=50)
    rows=[json.loads(s) for s in out.read_text().splitlines()]
    assert r['status']=='COMPLETE' and r['native_input_count']==700 and r['outputs']==10
    assert len(rows)==12 and len(rows[0]['vertices_m'])==778 and len(rows[0]['joints_m'])==21 and len(rows[0]['state51'])==51
    assert not rows[2]['committed'] and rows[3]['committed']
    assert r['GT_input'] is False and r['dataset_reader_instantiated'] is False
