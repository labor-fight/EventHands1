"""Independent completeness/finite-subset and scope guards for XDL results."""
import importlib
import importlib.util
import copy
import numpy as np
import pytest


def finalize():
    assert importlib.util.find_spec('tools.xdl.finalize') is not None, 'missing independent raw result auditor'
    return importlib.import_module('tools.xdl.finalize')


def test_numeric_failure_is_retained_and_never_reduced_on_finite_subset():
    f=finalize()
    good={'prediction':np.zeros(51), 'finite':True,'endpoint':{'wrist_mm':1.}}
    bad={'prediction':np.zeros(51), 'finite':True,'endpoint':{'wrist_mm':float('nan')}}
    audit=f.finite_records([good,bad],('prediction','endpoint'))
    assert not audit['passed'] and audit['n_records']==2 and audit['invalid_paths']
    assert f.describe_required([1.,float('nan'),3.])['mean'] is None
    assert f.describe_required([None,2.])['mean']==2. and f.describe_required([None,2.])['n_undefined']==1


def test_bank_missing_or_duplicated_probe_does_not_pass_completeness():
    f=finalize()
    bank={'sequences':{'s':{'bank':[{'packet_id':'s:0:49','specs':[{'id':'a'},{'id':'b'}]}],'cohort':'seen'}}}
    good=[{'seq':'s','packet_id':'s:0:49','spec':{'id':'a'}},{'seq':'s','packet_id':'s:0:49','spec':{'id':'b'}}]
    assert f.audit_bank_keys(bank,good)['passed']
    assert not f.audit_bank_keys(bank,good[:1])['passed']
    assert not f.audit_bank_keys(bank,[good[0],good[0]])['passed']


def test_mandatory_trajectory_checks_all21_positions_true_clock_and_failures():
    f=finalize()
    row={'times_ms':np.array([0,49,99]),'predictions':np.zeros((3,51)),'finite':True,'initial':np.zeros(51),'errors':{k:[0,1,2] for k in ('wrist_mm','mpjpe_abs_mm','mpjpe_ra_mm','mpvpe_abs_mm','mpvpe_ra_mm','root_deg','finger_deg')}}
    assert f.audit_trajectory(row,[0,49,99])['passed']
    wrong=copy.deepcopy(row);wrong['times_ms']=np.array([0,50,100])
    assert not f.audit_trajectory(wrong,[0,49,99])['passed']
    truncated=copy.deepcopy(row);truncated['predictions']=np.zeros((2,51))
    assert not f.audit_trajectory(truncated,[0,49,99])['passed']
    bad=copy.deepcopy(row);bad['predictions'][1,0]=np.nan
    assert not f.audit_trajectory(bad,[0,49,99])['passed']


def test_contrasts_keep_all_declared_arms_and_single_seed_scope():
    f=finalize()
    arms={a:{'status':'COMPLETE','metrics':{'bank|zgz|root|large|rho':float(v)}} for a,v in zip(('f0','q0','f1','q1','n1'),(1,.8,.9,.6,1.1))}
    c=f.build_contrasts(arms)
    assert c['query_teacher']['values']['bank|zgz|root|large|rho']['improvement']==pytest.approx(.2)
    assert c['event_content']['base']=='n1' and c['event_content']['candidate']=='q1'
    assert c['event_content']['values']['bank|zgz|root|large|rho']['improvement']==pytest.approx(.5)
    assert c['interaction']['values']['bank|zgz|root|large|rho']['interaction']==pytest.approx(.1)
    assert all(v['evidence_scope']=='single-seed exploratory; no repeat reliability gate or inferential CI/p' for v in c.values())
    arms['n1']['status']='FAIL'
    assert f.build_contrasts(arms)['event_content']['status']=='UNAVAILABLE'


def test_raw_cohort_labels_cannot_change_fixed_subject_denominators():
    f=finalize()
    bank={'sequences':{'ch_local':{'cohort':'seen'}}}
    assert hasattr(f,'audit_labels'), 'missing raw/split cohort identity check'
    assert f.audit_labels(bank,[{'seq':'ch_local','cohort':'seen','subject':'ch'}])['passed']
    assert not f.audit_labels(bank,[{'seq':'ch_local','cohort':'zgz','subject':'zgz'}])['passed']


def test_required_endpoint_maps_and_values_cannot_be_missing_or_null():
    f=finalize()
    assert not f.finite_records([{'prediction':np.zeros(51),'finite':True}],('prediction','endpoint'))['passed']
    assert not f.finite_records([{'prediction':np.zeros(51),'finite':True,'endpoint':{'wrist_mm':None}}],('prediction','endpoint'))['passed']
    row={'times_ms':np.array([0,49]),'predictions':np.zeros((2,51)),'finite':True,'errors':{}}
    assert not f.audit_trajectory(row,[0,49])['passed']


def test_actual_initial_must_match_first_stored_recursive_state():
    f=finalize()
    row={'times_ms':np.array([0,49]),'predictions':np.zeros((2,51)),'initial':np.ones(51),'finite':True,
         'errors':{k:[0,0] for k in ('wrist_mm','mpjpe_abs_mm','mpjpe_ra_mm','mpvpe_abs_mm','mpvpe_ra_mm','root_deg','finger_deg')}}
    assert not f.audit_trajectory(row,[0,49])['passed']


def test_arm_modes_and_training_contract_cannot_silently_alias_another_condition():
    f=finalize()
    assert hasattr(f,'config_mode_checks'), 'missing exact five-condition config guard'
    cfg={'SEED':3407,'MODEL':{'XDL_STUDY':True,'XDL_QUERY':'mano','XDL_SELF':True,'XDL_OBSERVATION_NULL':True,'XYZ_GEOM':False,'XDL_BURN_MODE':'full'},
         'DATA':{'WINDOW_MIN':50,'WINDOW_MAX':50},'AUG':{'SPEED_AUG':False},'TRAIN':{'DEVICES':2,'BATCH_SIZE_PER_GPU':512,'ACCUMULATE_GRAD_BATCHES':1,'MAX_STEPS':6000}}
    assert all(f.config_mode_checks('xdl_n1_s3407',cfg).values())
    assert not all(f.config_mode_checks('xdl_q1_s3407',cfg).values())
