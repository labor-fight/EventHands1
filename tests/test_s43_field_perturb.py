"""Fixed public-pose perturbations: no task data, GT, training or GPU."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tools.s43_revisit import s43_field_perturb as P
from tools.s43_revisit import witness_geometry as G


def public():return np.array([0.,0.,.6]+[0.]*48)


def test_none_is_independent_identity_without_input_mutation():
    p=public();before=p.copy();q,m=P.perturb_pose(p,'none')
    np.testing.assert_array_equal(p,before);np.testing.assert_array_equal(q,p)
    assert q is not p and q.dtype==np.float64 and m['GT_used'] is False


def test_root_x10mm_is_only_translation_and_has_zero_wrist_relative_change():
    p=public();p[3:6]=[.3,-.2,.1]
    q,m=P.perturb_pose(p,'root_x10mm');expected=p.copy();expected[0]+=.01
    np.testing.assert_array_equal(q,expected)
    assert m['absolute_mean21_joint_displacement_mm']==pytest.approx(10,abs=1e-10)
    assert m['wrist_relative_mean21_joint_displacement_mm']<1e-10


def test_root_yaw_is_right_so3_action_and_physical_angle_is_fixed():
    p=public();p[3:6]=[.4,-.3,.2]
    q,m=P.perturb_pose(p,'root_yaw0.1')
    before=Rotation.from_rotvec(p[3:6]).as_matrix();delta=Rotation.from_rotvec([0,.1,0]).as_matrix()
    np.testing.assert_allclose(Rotation.from_rotvec(q[3:6]).as_matrix(),before@delta,atol=1e-14,rtol=0)
    assert np.max(np.abs(before@delta-delta@before))>1e-3
    assert m['actual_rotation_geodesic_rad']==pytest.approx(.1,abs=1e-13)
    np.testing.assert_array_equal(q[:3],p[:3]);np.testing.assert_array_equal(q[6:],p[6:])


@pytest.mark.parametrize('old_angle',[np.pi-.03,2*np.pi+.3,-2*np.pi-.3])
def test_yaw_keeps_closest_original_chart_branch_near_seam_or_winding(old_angle):
    p=public();p[4]=old_angle;q,m=P.perturb_pose(p,'root_yaw0.1')
    assert np.linalg.norm(q[3:6]-p[3:6])==pytest.approx(.1,abs=1e-12)
    assert q[4]==pytest.approx(old_angle+.1,abs=1e-12)
    assert m['branch']['exported_rotation_max_abs_error']<1e-12
    if old_angle>np.pi-.1:assert np.linalg.norm(q[3:6])>np.pi


def test_closest_identity_rotation_can_keep_existing_identity_winding():
    old=np.array([0.,2*np.pi,0.]);q,m=P.closest_rotation_chart(np.eye(3),old)
    np.testing.assert_allclose(q,old,atol=1e-14,rtol=0)
    assert m['chart_step_rad']<1e-13


@pytest.mark.parametrize('root_rotation',[[0.,0.,0.],[.8,-.4,.2]])
def test_public_index_flex_is_real_five_mm_and_mean_is_added_once(root_rotation):
    p=public();p[3:6]=root_rotation;original=p.copy()
    q,m=P.perturb_pose(p,'index_flex5mm')
    row=G.public_flexion_axes()[1];joint=row['mcp_mano'];start=3+3*joint
    unchanged=np.ones(51,bool);unchanged[start:start+3]=False
    np.testing.assert_array_equal(q[unchanged],p[unchanged]);np.testing.assert_array_equal(p,original)
    assert m['index_wrist_relative_mean4_joint_displacement_mm']==pytest.approx(5,abs=2e-4)
    assert m['calibration']['target_reached'] and 0<m['angle_rad']<=.6
    assert m['calibration']['shape'].startswith('public beta0')
    mean=G._joint_constants()['mean'][3*(joint-1):3*joint]
    oldR=Rotation.from_rotvec(p[start:start+3]+mean).as_matrix()
    newR=Rotation.from_rotvec(q[start:start+3]+mean).as_matrix()
    axis=np.eye(3)[row['axis']]*row['sign']
    np.testing.assert_allclose(newR,oldR@Rotation.from_rotvec(axis*m['angle_rad']).as_matrix(),atol=1e-14,rtol=0)


def test_helper_has_no_history_or_event_side_effect_and_rejects_invalid_inputs():
    p=public();p.flags.writeable=False
    q,_=P.perturb_pose(p,'index_flex5mm');assert np.isfinite(q).all()
    with pytest.raises(ValueError):P.perturb_pose(np.full(51,np.nan),'root_x10mm')
    with pytest.raises(ValueError):P.perturb_pose(p,'arbitrary_GT_tuned_condition')
    with pytest.raises(ValueError):P.closest_rotation_chart(np.eye(3)*2,np.zeros(3))
