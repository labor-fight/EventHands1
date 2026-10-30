"""Independent reducers: preserve paired units, frame energy, and fit geometry."""
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'model')]
import numpy as np
import pytest

from tools.xkf_statistics.assess import (GROUPS, MODES, SEEDS, balanced_absolute,
                                        coefficient_fit, flow_reduction, primary_reduction)


def primary_cells():
    rows = []
    for si, seed in enumerate(SEEDS):
        for identity in range(9):
            # Different clip counts must not change identity weight. Seed effects
            # average to zero within each identity, before computing nine-unit SE.
            for clip in range(identity+1):
                for mode in MODES:
                    improvement = identity+1+(si-1)*20
                    rows.append(dict(seed=seed, identity=str(identity), clip=clip,
                                     mode=mode, condition='small_gaussian', scope='main_onward',
                                     joint_mse_mm2=100. if mode=='baseline' else 100.-improvement))
    return rows


def test_primary_is_nine_paired_identity_units_and_preserves_failing_seed():
    result=primary_reduction(primary_cells())
    assert result['mean_improvement_mm2']==pytest.approx(5.)
    assert result['paired_SE_mm2']==pytest.approx(np.std(np.arange(1.,10.),ddof=1)/3.)
    assert result['independent_units']==9
    assert result['mean_gt_one_paired_SE']
    assert result['per_seed_noninferiority']=={'3407':False,'3408':True,'3409':True}
    assert result['per_seed_mean_improvement_mm2']['3407']==pytest.approx(-15.)


def test_missing_identity_or_mode_is_not_finite_subset_success():
    rows=primary_cells()
    with pytest.raises(ValueError,match='nine identities'):
        primary_reduction([r for r in rows if r['identity']!='8'])
    with pytest.raises(ValueError,match='missing mode'):
        primary_reduction([r for r in rows if not (r['seed']==3407 and r['identity']=='0' and r['mode']=='identity_cv')])


def test_clean_uses_clip_then_identity_weighting():
    rows=[]
    for identity in range(9):
        for clip in range(identity+1):
            rows.append(dict(seed=3407,identity=str(identity),mode='baseline',condition='clean',
                             scope='all_observed',abs_mpjpe_mm=float(identity),clip=clip))
    assert balanced_absolute(rows,3407,'baseline','all_observed')==4.
    with pytest.raises(ValueError,match='whole identity'):
        balanced_absolute([r for r in rows if r['identity']!='0'],3407,'baseline','all_observed')


def test_high_motion_projection_is_energy_weighted_and_rmse_counts_joints():
    cells=[dict(n=1,error_energy=21.,dot=2.,true_energy=1.),
           dict(n=3,error_energy=63.,dot=50.,true_energy=100.)]
    result=flow_reduction(cells)
    assert result['frames']==4
    assert result['joint_motion_rmse_mm']==1.
    assert result['projection']==pytest.approx(52./101.)
    assert result['projection']!=pytest.approx((2.+.5)/2.)
    with pytest.raises(ValueError,match='empty'):
        flow_reduction([])


def test_independent_quadratic_fit_retains_bias_correlation_and_count_gate():
    rng=np.random.default_rng(20261015)
    shared=rng.normal(size=(119,51))
    ei=shared+rng.normal(size=shared.shape)*.3+.7
    ed=shared*.4+rng.normal(size=shared.shape)*.5-.2
    g=rng.uniform(size=119)
    answer=coefficient_fit(ei,ed,g,np.full(19,.2),1e-18)
    for k,(a,b) in zip(answer,GROUPS):
        objective=lambda x:np.mean(np.sum((ei[:,a:b]-g[:,None]*x*(ei[:,a:b]-ed[:,a:b]))**2,axis=1))
        assert objective(k)<=min(objective(x) for x in np.linspace(0.,1.,1001))+1e-12
    assert coefficient_fit(ei,ei,g,np.full(19,.2),1e-18)==[.2]*19
