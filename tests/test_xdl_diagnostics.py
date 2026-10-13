"""Physical counterfactual and failure-retention contracts for XD-V2."""
from pathlib import Path
import importlib
import importlib.util
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation


def geometry():
    assert importlib.util.find_spec('tools.xdl.geometry') is not None, 'missing physical perturbation/metric implementation'
    return importlib.import_module('tools.xdl.geometry')


def preparation():
    assert importlib.util.find_spec('tools.xdl.prepare') is not None, 'missing frozen chronological selection implementation'
    return importlib.import_module('tools.xdl.prepare')


def test_probe_counts_and_seen_axis_rotation():
    g = geometry()
    full = g.probe_specs('zgz_local:0:49')
    seen = g.probe_specs('ch_local:0:49', seen_ordinal=5)
    assert len(full) == 214 and len(seen) == 94
    fingers = [s for s in seen if s['block'] == 'finger']
    assert {s['axis'] for s in fingers if s['joint'] == 2} == {(5 + 2) % 3}
    assert len(g.recovery_specs('zgz_local:0:49')) == 18
    assert len(g.control_specs(full, 'zgz_local:0:49')) == 10


def test_physical_finger_uses_full_pose_mean_once_and_root_keeps_wrist():
    g = geometry()
    mean = np.linspace(-.2, .2, 45)
    state = np.zeros(51); state[2] = .4; state[6:] = -.1
    spec = {'block': 'finger', 'joint': 3, 'axis': 1, 'sign': -1, 'scale': 'large'}
    got = g.perturb_state(state, spec, mean)
    base = Rotation.from_rotvec((state[6:] + mean).reshape(15,3)).as_matrix()
    changed = Rotation.from_rotvec((got[6:] + mean).reshape(15,3)).as_matrix()
    expected = Rotation.from_rotvec([0,-math.radians(20),0]).as_matrix() @ base[3]
    assert np.max(abs(changed[3] - expected)) < 1e-12
    assert np.max(abs(np.delete(changed,3,0)-np.delete(base,3,0))) < 1e-12
    wrist = lambda p: np.asarray(p[:3]) + Rotation.from_rotvec(p[3:6]).apply([.03,.01,-.02])
    root = g.perturb_state(state, {'block':'root','axis':2,'sign':1,'scale':'large'}, mean, wrist_fn=wrist)
    assert np.linalg.norm(wrist(root)-wrist(state))*1000 < .01


def test_rotation_transport_handles_common_left_and_right_motion_without_correction():
    g = geometry()
    r0 = Rotation.from_rotvec([.2,-.1,.3]).as_matrix()
    rd = Rotation.from_rotvec([.15,.05,0]).as_matrix() @ r0
    move = Rotation.from_rotvec([-.2,.3,.1]).as_matrix()
    left = g.rotation_pair_metrics(r0,rd,move@r0,move@rd)
    right = g.rotation_pair_metrics(r0,rd,r0@move,rd@move)
    assert left['rho'] == pytest.approx(1)
    assert left['left']['gain'] == pytest.approx(0, abs=1e-12)
    assert right['right']['gain'] == pytest.approx(0, abs=1e-12)
    assert g.vector_pair_metrics([0,0,0],[1,0,0],[0,0,0],[.5,0,0])['gain'] == pytest.approx(.5)
    assert g.vector_pair_metrics([0,0,0],[0,0,0],[0,0,0],[1,0,0])['rho'] is None
    fixed = g.vector_pair_metrics([0,0,0],[1,0,0],[9,0,0],[9,0,0])
    assert fixed['rho'] == 0 and fixed['gain'] == 1  # endpoint/event gates still required


def test_auc_real_times_sustained_recovery_and_invalid_not_dropped():
    g = geometry()
    assert g.time_auc([10,6,2],[0,49,99]) == pytest.approx((49*8+50*4)/99)
    assert g.time_auc([10,np.nan,2],[0,49,99]) is None
    a = g.sustained_recovery([30,19,18,17],[15,9,8,7],[0,49,99,149])
    assert a['hit_step'] == 1 and a['time_ms'] == 49 and not a['right_censored']
    b = g.sustained_recovery([30,19,30,19],[15,9,15,9],[0,49,99,149])
    assert b['hit_step'] is None and b['right_censored'] and b['horizon_ms'] == 149


def test_wrong_events_preserve_time_count_polarity_and_xy_multiset():
    g = geometry()
    ev = np.zeros((40,5),np.float32); ev[:,1]=np.arange(40);ev[:,2]=np.arange(40)%12
    ev[:,3]=np.linspace(0,.049,40); ev[:,4]=np.arange(40)%2
    swap, audit = g.transform_events(ev,'swap')
    assert np.array_equal(swap[:,3:],ev[:,3:]) and len(swap)==len(ev)
    assert sorted(map(tuple,swap[:,1:3]))==sorted(map(tuple,ev[:,1:3]))
    assert audit['changed_fraction'] > .01
    mirror, _ = g.transform_events(ev,'mirror')
    assert np.array_equal(mirror[:,1],239-ev[:,1])


def test_cpu_selection_does_not_cross_runs_and_freezes_midpoint_ranks():
    p = preparation()
    rows = p.packet_rows([[10,1060],[2000,2550]])
    assert rows[0]['previous_end_ms']==10 and rows[0]['end_ms']==59
    assert rows[1]['previous_end_ms']==59 and rows[1]['end_ms']==109
    assert p.midpoint_indices(31,1)==[15]
    clips = p.clip_candidates(rows,20)
    assert len(clips)==1 and clips[0]['start_index']==0
    assert clips[0]['end_indices'][-1]==19
    with pytest.raises(ValueError): p.midpoint_indices(2,3)


def test_motion_inactive_undefined_and_finite_failure_kept():
    g = geometry()
    still = np.zeros((11,21,3))
    got = g.motion_response(still,still,np.arange(11)*50)
    assert got['gain'] is None and got['cosine'] is None and got['active_windows']==0
    moving = still.copy();moving[:,:,0]=np.arange(11)[:,None]*.005
    got = g.motion_response(moving,moving,np.arange(11)*50,root_relative=False)
    assert got['gain']==pytest.approx(1) and got['cosine']==pytest.approx(1)
    assert got['lag_ms'] is None and not got['lag_identifiable']  # constant velocity has a flat correlation
    invalid=[];clean=g.sanitize({'rows':[1,np.nan,3]}, invalid=invalid)
    assert clean['rows']==[1,None,3] and invalid==['$.rows[1]']


def diagnostics():
    assert importlib.util.find_spec('tools.xdl.diagnostics') is not None, 'missing target-free batch replay implementation'
    return importlib.import_module('tools.xdl.diagnostics')


def test_packet_clock_target_zero_and_each_row_same_raw_events():
    d=diagnostics()
    import torch
    ev=np.zeros((3,5),np.float32);ev[:,1]=[1,2,3];ev[:,3]=[0,.01,.049]
    states=np.zeros((2,51),np.float32)
    packet=d.make_packet(ev,states,torch.zeros(1,10),torch.eye(3).unsqueeze(0),torch.device('cpu'),{'event_start_ms':100,'end_ms':149})
    assert not packet.target.any() and packet.ptr.tolist()==[0,3,6]
    assert packet.events[:,0].tolist()==[0,0,0,1,1,1]
    assert packet.t_start_us.tolist()==[100000,100000]
    assert torch.equal(packet.events[:3,1:],packet.events[3:,1:])


def test_recursive_loop_uses_own_prediction_retains_nonfinite_frames():
    d=diagnostics()
    import torch
    class ConstantVelocity:
        def forward_packet(self,packet):
            assert not packet.target.any()
            result=packet.prev_state.clone();result[:,0]+=.001
            result[result[:,2]<0,0]=float('nan')
            return result
    states=np.zeros((2,51),np.float32);states[1,2]=-1
    rows=[{'event_start_ms':0,'end_ms':49,'previous_end_ms':0}, {'event_start_ms':50,'end_ms':99,'previous_end_ms':49}]
    data={'events':np.zeros((0,3),np.float32),'offsets':np.zeros(101,np.int64),'tsub':None}
    pred,audit=d.recursive_loop(ConstantVelocity(),data,rows,states,torch.zeros(1,10),torch.eye(3).unsqueeze(0),torch.device('cpu'))
    assert pred.shape==(2,3,51)
    assert pred[0,2,0]==pytest.approx(.002)
    assert np.isnan(pred[1,1:,0]).all() and len(audit['failures'])==1
    assert audit['query_start_offset_ms']==[0,1] and audit['state_interval_ms']==[49,50]


def test_rotation_motion_true_movement_and_zero_energy():
    g=geometry()
    aa=np.zeros((13,3));aa[:,0]=np.arange(13)*math.radians(2)
    mats=Rotation.from_rotvec(aa).as_matrix();times=np.arange(13)*50
    assert hasattr(g,'rotation_motion_response'), 'missing physical angular motion guard'
    matched=g.rotation_motion_response(mats,mats,times)
    assert matched['gain']==pytest.approx(1) and matched['cosine']==pytest.approx(1)
    still=g.rotation_motion_response(np.broadcast_to(np.eye(3),(13,3,3)),np.broadcast_to(np.eye(3),(13,3,3)),times)
    assert still['gain'] is None and still['active_windows']==0


def test_actual_mano_root_and_local_interventions_preserve_wrist():
    d=diagnostics()
    import torch
    mano=d.ManoLayer(d.REPO/'assets/mano_right.npz',add_mean=False).eval();mano.fast_chain=True
    state=np.zeros(51,np.float32);state[2]=.45;state[3:6]=[.2,-.1,.3]
    specs=[{'block':'clean','id':'clean'}, {'block':'root','axis':0,'scale':'large','sign':1,'id':'root'},
           {'block':'finger','joint':0,'axis':2,'scale':'large','sign':-1,'id':'finger'}]
    _,audit=d.initial_states(mano,state,state,specs,mano.hands_mean.numpy(),torch.zeros(1,10),torch.device('cpu'))
    assert audit['passed']


def test_motion_eligibility_respects_first_49ms_packet_clock():
    g=geometry()
    times=np.array([0,49,99,149,199,249,299])
    still=np.zeros((7,21,3))
    value=g.motion_response(still,still,times)
    assert value['eligible_time_windows']==2  # exact49->249 and99->299; no200ms endpoint from0


def test_reanalysis_corrects_counter_metadata_preserves_original_motion_scores():
    d=diagnostics()
    payload={'rows':[{'times_ms':np.array([0,49,99,149,199,249,299]),'motion':{
        'absolute':{'horizon_ms':200,'eligible_time_windows':3,'gain':1.25,'cosine':.7}}}]}
    assert hasattr(d,'refresh_motion_counts'), 'missing preserved-raw reanalysis path'
    revised,changes=d.refresh_motion_counts(payload)
    assert revised['rows'][0]['motion']['absolute']['eligible_time_windows']==2
    assert revised['rows'][0]['motion']['absolute']['gain']==1.25
    assert payload['rows'][0]['motion']['absolute']['eligible_time_windows']==3
    assert len(changes)==1


@pytest.mark.parametrize('section', ['recovery','motion'])
def test_selected_serial_batch_applies_to_all_closed_loops_including_short_natural(monkeypatch,section):
    d=diagnostics()
    import torch
    torch.set_num_threads(2)
    class OwnState:
        def __init__(self):self.batch_sizes=[]
        def set_hand_context(self,*args):pass
        def forward_packet(self,packet):
            assert not packet.target.any()
            self.batch_sizes.append(packet.batch_size)
            result=packet.prev_state.clone();result[:,0]+=.0001
            return result
    model=OwnState();device=torch.device('cpu')
    mano=d.ManoLayer(d.REPO/'assets/mano_right.npz',add_mean=False).eval();mano.fast_chain=True
    rows=preparation().packet_rows([[0,1050],[1100,1250]])
    for i,row in enumerate(rows):row['index']=i;row['packet_id']=f'zgz_local:{row["run_id"]}:{row["end_ms"]}'
    gt=np.zeros((1251,51),np.float32);gt[:,2]=.45
    data={'events':np.zeros((0,3),np.float32),'offsets':np.zeros(1252,np.int64),'tsub':None,'gt':gt,
          'aux':{'betas':np.zeros(10,np.float32),'camera_K':np.eye(3,dtype=np.float32)}}
    monkeypatch.setattr(d,'load_data',lambda *_:data)
    reference={'zgz_local':[{**r,'prev':gt[r['previous_end_ms']].copy()} for r in rows]}
    meta={'cohort':'zgz','rows':rows,'valid_runs_ms':[[0,1050],[1100,1250]],
          'clips':[{'clip_id':'zgz_local:0:49','run_id':0,'end_indices':list(range(20))}]}
    bank={'sequences':{'zgz_local':meta}}
    if section=='recovery':value=d.recovery_diagnostic(model,mano,{},bank,reference,device,legacy=True,batch_size=1)
    else:value=d.full_motion(model,mano,{},bank,reference,device,batch_size=1)
    assert len(model.batch_sizes)==value['queries_attempted'] and set(model.batch_sizes)=={1}
    if section=='recovery':assert value['queries_attempted']==18*20
    else:
        assert len(value['short_natural'])==1
        assert value['queries_attempted']==len(rows)*9+3
