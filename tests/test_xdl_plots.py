"""Plotting gates preserve full paired denominators and real elapsed times."""
import importlib
import importlib.util
import numpy as np
import pytest


def plots():
    assert importlib.util.find_spec('tools.xdl.plot_recovery') is not None, 'missing hash-bound diagnostic figure producer'
    return importlib.import_module('tools.xdl.plot_recovery')


def test_formal_new_figures_wait_for_all_five_complete_arms():
    p=plots()
    audit={'legacy_runs':{r:{'status':'COMPLETE'} for r in ('xa_c0_s3407','xa_c1_s3407')},
           'new_runs':{a:{'status':'COMPLETE'} for a in ('f0','q0','f1','q1','n1')},'new_stage_complete':False}
    with pytest.raises(ValueError,match='five'):p.select_runs(audit)
    audit['new_stage_complete']=True
    assert len(p.select_runs(audit))==7
    audit['new_runs']['n1']['status']='FAIL'
    with pytest.raises(ValueError,match='five'):p.select_runs(audit)
    assert p.select_runs(audit,legacy_only=True)==['xa_c0_s3407','xa_c1_s3407']


def test_large_bank_summary_never_averages_a_finite_subset():
    p=plots()
    records=[{'seq':'zgz_local','spec':{'block':'root','scale':'large'},'finite':True,'paired':{'root':{'rho':.5}},'endpoint':{'root_deg':10.}},
             {'seq':'zgz_local','spec':{'block':'root','scale':'large'},'finite':True,'paired':{'root':{'rho':.7}},'endpoint':{'root_deg':20.}}]
    point=p.bank_point(records,'zgz_local','root',expected=2)
    assert point['median_rho']==pytest.approx(.6) and point['mean_gt_error']==15
    records[1]['endpoint']['root_deg']=np.nan
    with pytest.raises(ValueError,match='nonfinite'):p.bank_point(records,'zgz_local','root',expected=2)
    with pytest.raises(ValueError,match='denominator'):p.bank_point(records[:1],'zgz_local','root',expected=2)


def test_natural_curves_use_actual_times_all24_clips_and_no_fake_ci():
    p=plots()
    records=[{'seq':'zgz_local','spec':{'block':'natural'},'times_ms':np.arange(21)*50+(1000+i*1000),'finite':True,
              'errors':{'wrist_mm':np.arange(21,dtype=float)}} for i in range(24)]
    records[0]['times_ms'][1:]-=1
    curve=p.natural_curve(records,'zgz_local','wrist_mm')
    assert curve['n_clips']==24 and curve['mean_actual_elapsed_ms'][1]==pytest.approx(50-1/24)
    assert curve['mean_error'][3]==3 and 'confidence_interval' not in curve
    records[3]['errors']['wrist_mm'][10]=np.inf
    with pytest.raises(ValueError,match='nonfinite'):p.natural_curve(records,'zgz_local','wrist_mm')
    with pytest.raises(ValueError,match='24'):p.natural_curve(records[:23],'zgz_local','wrist_mm')


def test_blocked_formal_plot_retains_failure_without_creating_partial_images(tmp_path):
    import json
    p=plots()
    audit=tmp_path/'audit.json';audit.write_text(json.dumps({'new_stage_complete':False}))
    figures=tmp_path/'figures'
    with pytest.raises(ValueError):p.plot(audit,directory=figures)
    assert (figures/'xdl_figure_failure.json').exists(), 'blocked figure must preserve explicit failure artifact'
    failure=json.loads((figures/'xdl_figure_failure.json').read_text())
    assert failure['status']=='FAIL' and not failure['finite_subset_reduction']
    assert not list(figures.glob('*.png')) and not list(figures.glob('*.svg'))


def test_json_snapshot_is_bound_to_exact_digest_and_detects_hash_mismatch(tmp_path):
    import hashlib,json
    p=plots();path=tmp_path/'a.json';raw=b'{"value":3}';path.write_bytes(raw)
    assert hasattr(p,'verified_json'), 'missing atomic hash-bound JSON snapshot reader'
    value,digest=p.verified_json(path,hashlib.sha256(raw).hexdigest())
    assert value['value']==3 and digest==hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError,match='SHA'):p.verified_json(path,'0'*64)
