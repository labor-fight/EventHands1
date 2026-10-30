"""Fixed Stage5 pose perturbations from public/deployed conditions only.

No events, targets, current GT shape, history or runtime state are accepted.
The caller injects once and updates its last committed history knot. Returned
poses are float64; deployment may explicitly cast the result to FP32.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from . import witness_geometry as G

CONDITIONS=('none','root_x10mm','root_yaw0.1','index_flex5mm')


def _pose(value):
    if hasattr(value,'detach'):value=value.detach().cpu().numpy()
    p=np.asarray(value,np.float64)
    if p.shape!=(51,) or not np.isfinite(p).all():raise ValueError('finite deployed/public pose51 required')
    return p.copy()


def closest_rotation_chart(matrix,original_actual_aa):
    """Axis-angle representative of matrix closest to the old actual chart.

    Every nonidentity representative is (theta+2*pi*k)*principal_axis.
    Compare integer windings around the analytic nearest projection. Identity
    admits every axis; the nearest identity representative uses the old axis.
    This controls chart jumps, not the physical SO(3) perturbation.
    """
    R=np.asarray(matrix,np.float64);old=np.asarray(original_actual_aa,np.float64)
    if R.shape!=(3,3) or old.shape!=(3,) or not np.isfinite(np.r_[R.ravel(),old]).all():raise ValueError('finite rotation and old actual AA required')
    if not np.allclose(R.T@R,np.eye(3),atol=1e-10,rtol=0) or np.linalg.det(R)<0:
        raise ValueError('proper SO3 matrix required')
    principal=Rotation.from_matrix(R).as_rotvec();theta=float(np.linalg.norm(principal))
    if theta<1e-13:
        norm=float(np.linalg.norm(old))
        axis=old/norm if norm>1e-13 else np.array([1.,0.,0.])
        center=int(np.rint(norm/(2*np.pi)))
        ks=np.arange(center-2,center+3,dtype=np.int64)
        candidates=2*np.pi*ks[:,None]*axis
    else:
        axis=principal/theta
        center=int(np.rint((float(old@axis)-theta)/(2*np.pi)))
        ks=np.arange(center-2,center+3,dtype=np.int64)
        candidates=(theta+2*np.pi*ks[:,None])*axis
    distances=np.linalg.norm(candidates-old,axis=1)
    order=sorted(range(len(ks)),key=lambda i:(float(distances[i]),abs(int(ks[i])),int(ks[i])))
    chosen=order[0];value=candidates[chosen].copy()
    physical_error=float(np.max(np.abs(Rotation.from_rotvec(value).as_matrix()-R)))
    if physical_error>1e-10:raise ArithmeticError('nearest chart changed physical rotation')
    return value,dict(principal_norm_rad=theta,selected_winding=int(ks[chosen]),
        chart_step_rad=float(distances[chosen]),principal_chart_step_rad=float(np.linalg.norm(principal-old)),
        exported_rotation_max_abs_error=physical_error,
        remaining_chart_seam=bool(distances[chosen]>np.pi),
        branch_rule='nearest Euclidean actual-axisangle representative; deterministic winding tie')


def _right_action(p,joint,axis,angle):
    start=3+3*joint
    mean=np.zeros(3) if joint==0 else G._joint_constants()['mean'][3*(joint-1):3*joint]
    original=p[start:start+3]+mean
    rotated=Rotation.from_rotvec(original).as_matrix()@Rotation.from_rotvec(np.asarray(axis)*angle).as_matrix()
    chart,metadata=closest_rotation_chart(rotated,original)
    result=p.copy();result[start:start+3]=chart-mean
    return result,metadata


def perturb_pose(pose51,condition):
    """Return (new_pose51, metadata); no input mutation and no feedback action.

    root_x10mm adds x+.01m. root_yaw0.1 right-multiplies root Exp(0,.1,0).
    index_flex5mm uses the frozen public beta0 MCP flexion axis and the
    frozen motion_endpoints FK calibration at this deployed pose, beta0,
    target5mm over four wrist-relative index joints, anglecap.6rad.
    """
    p=_pose(pose51)
    if condition not in CONDITIONS:raise ValueError('unknown fixed Stage5 perturbation')
    result=p.copy();branch={};angle=0.;calibration=None
    axis=None;joint=None
    if condition=='root_x10mm':result[0]+=.01
    elif condition=='root_yaw0.1':
        angle=.1;joint=0;axis=np.array([0.,1.,0.])
        result,branch=_right_action(p,joint,axis,angle)
    elif condition=='index_flex5mm':
        row=G.public_flexion_axes()[1];joint=int(row['mcp_mano'])
        axis=np.eye(3)[int(row['axis'])]*int(row['sign'])
        _,cal=G.motion_endpoints(p,np.zeros(10),target_mm=5.,max_angle_rad=.6)
        angle=float(cal['angles_rad'][1][0]);result,branch=_right_action(p,joint,axis,angle)
        calibration=dict(target_mm=5.,actual_calibration_mm=float(cal['actual_mm'][1][0]),
            cap_rad=.6,target_reached=bool(cal['target_reached'][1][0]),
            public_axis=dict(finger=row['finger'],mcp_mano=joint,axis=int(row['axis']),sign=int(row['sign'])),
            shape='public beta0; no auxiliary/GT beta input',iterations=16,
            definition='mean displacement of four wrist-relative index joints, exact frozen MANO21 FK')
    before,after=G.fk_joints(np.stack((p,result)),np.zeros(10))
    relative=(after-after[:1])-(before-before[:1]);indices=G.FINGER_JOINTS[1]
    if joint is not None:
        start=3+3*joint;mean=np.zeros(3) if joint==0 else G._joint_constants()['mean'][3*(joint-1):3*joint]
        relative_R=Rotation.from_rotvec(p[start:start+3]+mean).as_matrix().T@Rotation.from_rotvec(result[start:start+3]+mean).as_matrix()
        actual_angle=float(np.linalg.norm(Rotation.from_matrix(relative_R).as_rotvec()))
    else:actual_angle=0.
    metadata=dict(schema='s43_stage5_fixed_pose_perturbation_v1',condition=condition,
        input_sha256=hashlib.sha256(p.tobytes()).hexdigest(),output_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        geometry_source_sha256=hashlib.sha256(Path(G.__file__).read_bytes()).hexdigest(),
        mano_asset_sha256=G._asset_sha(),joint_mano=joint,
        axis_right_local=None if axis is None else axis.tolist(),angle_rad=angle,
        actual_rotation_geodesic_rad=actual_angle,translation_delta_m=(result[:3]-p[:3]).tolist(),
        absolute_mean21_joint_displacement_mm=float(np.linalg.norm(after-before,axis=-1).mean()*1000),
        wrist_relative_mean21_joint_displacement_mm=float(np.linalg.norm(relative,axis=-1).mean()*1000),
        index_wrist_relative_mean4_joint_displacement_mm=float(np.linalg.norm(relative[indices],axis=-1).mean()*1000),
        branch=branch,calibration=calibration,output_dtype='float64; caller explicitly casts committed state',
        hand_mean='once: lift residual to actual AA, right action, nearest log, remove mean once',
        GT_used=False,events_used=False,history_mutated=False,
        injection='caller once after two own50mswarmup commits; update currentpose+lastknot only; no source/core mutation')
    return result,metadata
