"""Explicit joint-pose goal admission; no relabeling of gyro-only evidence."""
from lewm.sensor_anchored_goal_development import AnchoredGoal
import math
import numpy as np
from lewm.causal_sensor_state import SensorContractError, _identity, _ns
from lewm.joint_rgbd_rigid_pose_development import proper, angle


def current_joint_pose(evidence, *, identity, now_ns):
    return _current_rgbd_pose(evidence, identity=identity, now_ns=now_ns,
        fitting_mode='joint', gyro_role='consistency_monitor_only')


def current_supported_rgbd_pose(evidence, *, identity, now_ns):
    """Explicit dispatch for the separate gyro-conditioned camera estimator."""
    if evidence.get('observer_variant') != 'gyro_consensus_dual_camera_v1':
        return current_joint_pose(evidence, identity=identity, now_ns=now_ns)
    if evidence.get('gyro_conditioned_image_consensus_estimator') is not True:
        raise SensorContractError('explicit gyro-conditioned image estimator required')
    return _current_rgbd_pose(evidence, identity=identity, now_ns=now_ns,
        fitting_mode='gyro_rgbd_refit', gyro_role='rotation_estimator')


def _current_rgbd_pose(evidence, *, identity, now_ns, fitting_mode, gyro_role):
    now = _ns(now_ns, 'anchored goal clock')
    if (evidence['schema'] != 'visual_led_motion_evidence_development.v1'
            or _identity(evidence['identity']) != _identity(identity)
            or evidence['decision_ns'] != now or evidence['status'] != 'CURRENT_VISUAL_POSE'
            or evidence['terminal_failure'] is not None):
        raise SensorContractError('same-episode current visual evidence required')
    pose = evidence['current_pose']
    if (pose is None or pose['mode'] != fitting_mode or pose['measured_ns'] != now
            or pose['available_ns'] > now or type(pose['frame']) is not int or pose['frame'] < 0):
        raise SensorContractError('current joint-RGB-D visual frame required')
    if not pose['measured_ns'] <= _ns(pose['available_ns'], 'joint pose availability') <= now:
        raise SensorContractError('causal joint pose availability required')
    p = np.asarray(pose['position_initial_body_m'], float)
    R = proper(pose['rotation_initial_body_from_current_body'])
    if p.shape != (3,) or not np.isfinite(p).all():
        raise SensorContractError('finite visual position required')
    for key in ('rgb_sha256', 'depth_sha256'):
        value = pose[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise SensorContractError('explicit RGB/depth binding required')
    check_joint_witness(evidence, p, R, now, fitting_mode=fitting_mode, gyro_role=gyro_role)
    return p.copy(), R.copy(), pose


class JointAnchoredGoal(AnchoredGoal):
    @classmethod
    def from_observation(cls, evidence, displacement_body_xy, yaw_delta_rad, *, identity, now_ns):
        p, R, pose = current_joint_pose(evidence, identity=identity, now_ns=now_ns)
        d = np.asarray(displacement_body_xy, float)
        if (d.shape != (2,) or not np.isfinite(d).all() or np.linalg.norm(d) > .4+1e-12
                or isinstance(yaw_delta_rad, bool) or not isinstance(yaw_delta_rad, (float, int))
                or not math.isfinite(yaw_delta_rad) or abs(yaw_delta_rad) > math.pi):
            raise SensorContractError('bounded .4m local displacement and wrapped yaw request required')
        target = p + R @ np.r_[d, 0.]
        heading = R @ [math.cos(yaw_delta_rad), math.sin(yaw_delta_rad), 0.]
        if np.linalg.norm(heading[:2]) < .2:
            raise SensorContractError('nonvertical final heading required')
        return cls(_identity(identity), now_ns, pose['frame'], pose['rgb_sha256'], pose['depth_sha256'],
                   tuple(p), tuple(tuple(row) for row in R), tuple(d), float(yaw_delta_rad),
                   tuple(target[:2]), math.atan2(heading[1], heading[0]))

    def snapshot(self):
        return super().snapshot() | dict(pose_mode='joint', gyro_role='consistency_monitor_only')


def check_joint_witness(evidence, p, R, now, *, fitting_mode='joint', gyro_role='consistency_monitor_only'):
    """Check current witness structure/composition, not independent image fitting.

    The uninterrupted executor owns chronology. Full raw replay must additionally
    reconstruct historical reference selection and the sensor-to-command path.
    """
    pose=evidence['current_pose']; e=evidence['continuity_evidence']
    def need(condition, reason):
        if not condition: raise SensorContractError(reason)
    need(evidence['continuity_evidence_current'] is True
        and evidence['command_integration_used'] is False
        and evidence['bridge_is_command_or_inertial_extrapolation'] is False
        and evidence['anchor_promotion_from_bridge'] is False
        and pose['gyro_role']==gyro_role
        and pose['native_pose_input'] is False and pose['global_history_reset'] is False
        and pose['position_error_bound'] is None and pose['orientation_error_bound'] is None
        and pose['uncertainty_model_validated'] is False and e['uncertainty_calibrated'] is False,
        'current measured joint pose with explicit uncalibrated semantics required')
    if fitting_mode == 'gyro_rgbd_refit':
        need(np.allclose(R, proper(pose['gyro_rotation_initial_body_from_current_body']),
            rtol=0, atol=1e-10) and evidence['gyro_bias_estimated'] is False,
            'gyro-conditioned rotation must compose current measured gyro increments')
    frame=pose['frame']
    if frame==0:
        need(e['status']=='INITIAL_REFERENCE' and pose['reference_frame']==0
            and not pose['promoted_keyframe'] and e['bridge_frames']==0
            and np.allclose(p,0.,rtol=0,atol=1e-12)
            and np.allclose(R,np.eye(3),rtol=0,atol=1e-12), 'single initial joint frame required')
        return
    need(e['status'] in ('ANCHOR_MEASUREMENT','MEASURED_INCREMENT_BRIDGE')
        and e['previous_frame']==frame-1 and e['previous_measured_ns']==now-100_000_000
        and e['rotation_fitting_mode']==fitting_mode and e['gyro_role']==gyro_role
        and e['gyro_bias_estimated'] is False, 'current qualified joint continuity required')
    witnesses=e['rotation_measurement_witnesses']
    need(type(witnesses) is list and 1<=len(witnesses)<=9, 'bounded current rotation witnesses required')
    for w in witnesses:
        need(type(w['reference_frame']) is int and 0<=w['reference_frame']<frame
            and w['current_frame']==frame
            and w['reference_measured_ns']==now-(frame-w['reference_frame'])*100_000_000
            and w['fitting_mode']==fitting_mode and w['candidate_envelope_passed'] is True
            and w['witness_alone_grants_pose'] is False, 'current measured reference identity required')
        ref=proper(w['reference_rotation_initial_body_from_reference_body'])
        fitted=proper(w['fitted_rotation_reference_body_from_current_body'])
        composed=proper(w['composed_rotation_initial_body_from_current_body'])
        gyro=proper(w['gyro_rotation_reference_body_from_current_body'])
        if fitting_mode == 'gyro_rgbd_refit':
            need(w.get('gyro_conditioned_rgbd_translation') is True
                and np.allclose(fitted, gyro, rtol=0, atol=1e-12),
                'current gyro rotation and refitted camera translation witness required')
        disagreement=angle(gyro.T@fitted)
        need(np.allclose(ref@fitted,composed,rtol=0,atol=1e-12)
            and math.isfinite(w['gyro_disagreement_rad'])
            and abs(disagreement-w['gyro_disagreement_rad'])<=1e-12
            and disagreement<=.10, 'joint rotation composition and unchanged gyro gate required')
    a=e['selected_anchor_rotation_witness']; b=e['incremental_rotation_witness']
    need(type(e['anchor_available']) is bool and type(e['incremental_available']) is bool
        and (a is not None)==e['anchor_available'] and (b is not None)==e['incremental_available']
        and e['incremental_rotation_witness_saved']==(b is not None), 'explicit selected witness availability required')
    for w in (a,b):
        if w is not None: need(w in witnesses, 'selected witness must be a current qualified fit')
    if b is not None:
        need(b['reference_frame']==frame-1, 'increment must use immediately previous frame')
    if e['status']=='MEASURED_INCREMENT_BRIDGE':
        need(a is None and b is not None and not pose['promoted_keyframe']
            and type(e['bridge_frames']) is int and 1<=e['bridge_frames']<=10
            and e['disagreement_m'] is None and e['disagreement_rad'] is None,
            'bounded measured bridge without promotion required')
    else:
        need(a is not None and e['bridge_frames']==0, 'retained anchor measurement required')
        if b is not None:
            distance=float(np.linalg.norm(np.asarray(a['position_initial_body_m'])-b['position_initial_body_m']))
            rotation=angle(proper(a['composed_rotation_initial_body_from_current_body']).T
                @proper(b['composed_rotation_initial_body_from_current_body']))
            need(distance<=.02 and rotation<=.10
                and abs(distance-e['disagreement_m'])<=1e-12
                and abs(rotation-e['disagreement_rad'])<=1e-12, 'unchanged anchor/increment agreement required')
        else:
            need(e['disagreement_m'] is None and e['disagreement_rad'] is None,
                'missing increment is not agreement')
    selected=a if a is not None else b
    need(selected['reference_frame']==pose['reference_frame']
        and np.allclose(selected['position_initial_body_m'],p,rtol=0,atol=1e-12)
        and np.allclose(selected['composed_rotation_initial_body_from_current_body'],R,rtol=0,atol=1e-12),
        'command pose must equal selected current measurement')
