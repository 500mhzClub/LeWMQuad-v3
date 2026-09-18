"""Current visual motion expressed from the most recent admitted floor pose.

This is a separate, uncalibrated pose hypothesis. It does not admit an
unobservable current plane, extrapolate a command, or change old map entries.
"""
from copy import deepcopy
import numpy as np

from lewm.causal_sensor_state import _identity, _ns
from lewm.joint_sensor_anchored_goal_development import current_supported_rgbd_pose as current_joint_pose
from lewm.joint_floor_registered_evidence_development import original_pose_validation_fields
from lewm.joint_floor_registered_evidence_development import (
    SCHEMA as FLOOR_SCHEMA, current_joint_floor_registered_pose)
from lewm.joint_measured_floor_plane_development import CAMERAS, compose_plane, fit_joint_plane
from lewm.floor_pose_registration_development import unit
from lewm.joint_rgbd_rigid_pose_development import proper, angle

SCHEMA = 'measured_visual_floor_transport_evidence_development.v1'
MISSING = ('insufficient_combined_measured_candidates', 'insufficient_combined_two_axis_extent')


def composition(anchor, raw, missing_plane, *, identity, now_ns):
    """Validate both measured poses and return their explicit SE(3) composition."""
    now = _ns(now_ns, 'measured floor transport')
    p, R, pose = current_joint_pose(raw, identity=identity, now_ns=now)
    anchor_ns = anchor['decision_ns']
    a, A, anchor_pose = current_joint_floor_registered_pose(anchor, identity=identity, now_ns=anchor_ns)
    b, B, raw_anchor = current_joint_pose(anchor['original_visual_evidence'], identity=identity, now_ns=anchor_ns)
    if (not 0 <= anchor_pose['frame'] < pose['frame'] < 4096
            or anchor_pose['frame'] != raw_anchor['frame']
            or now != 1_500_000_000+pose['frame']*100_000_000
            or anchor_ns != 1_500_000_000+anchor_pose['frame']*100_000_000):
        raise ValueError('earlier admitted floor anchor and current uninterrupted visual clock required')
    reference = anchor['floor_registration']['reference']
    up = R.T@unit(reference['initial_up_body'])
    expected = compose_plane(missing_plane['camera_statistics'], up)
    if (expected['available'] is not False or expected['reason'] not in MISSING
            or any(missing_plane[k] != v for k, v in expected.items())
            or missing_plane['camera_residuals'] is not None
            or not np.array_equal(unit(missing_plane['up_body']), up)):
        raise ValueError('only exactly reconstructed current count or extent missingness permits transport')
    correction_R = proper(A@B.T)
    transported_p = a+correction_R@(p-b)
    transported_R = proper(correction_R@R)
    displacement = float(np.linalg.norm(transported_p-p))
    rotation = angle(R.T@transported_R)
    if displacement > .05 or rotation > .10:
        raise ValueError('transport exceeds original 5 cm / 0.10 rad development correction magnitudes')
    reference_plane = reference['joint_plane']
    normal = unit(reference_plane['normal_body'])
    return dict(position_initial_body_m=transported_p.tolist(),
        rotation_initial_body_from_current_body=transported_R.tolist(),
        raw_position_initial_body_m=p.tolist(), raw_rotation_initial_body_from_current_body=R.tolist(),
        anchor_frame=anchor_pose['frame'], anchor_measured_ns=anchor_ns,
        anchor_age_frames=pose['frame']-anchor_pose['frame'],
        correction_rotation_initial_from_raw=correction_R.tolist(),
        correction_magnitude_m=displacement, correction_angle_rad=rotation,
        maximum_correction_m=.05, maximum_correction_rad=.10,
        transported_reference_normal_body=(transported_R.T@normal).tolist(),
        transported_reference_offset_body_m=float(reference_plane['offset_body_m']+normal@transported_p),
        transported_reference_is_current_measured_plane=False,
        in_plane_translation_preserved=False, in_plane_forward_azimuth_preserved=False,
        native_pose_used=False, command_integration_used=False, historical_map_rewritten=False,
        position_error_bound=None, orientation_error_bound=None, uncertainty_model_validated=False,
        floor_or_support_certified=False, navigation_qualified=False)


def check_candidate_residuals(rows, plane, correction):
    """Check complete residual accounting; independent raw replay owns maxima."""
    if type(rows) is not list or [r['camera'] for r in rows] != list(CAMERAS):
        raise ValueError('both current camera populations require transport conflict accounting')
    normal = unit(correction['transported_reference_normal_body'])
    offset = correction['transported_reference_offset_body_m']
    for row, stat in zip(rows, plane['camera_statistics'], strict=True):
        if row['count'] != stat['count']:
            raise ValueError('every measured candidate must be checked against the transported reference')
        maximum, rms = row['maximum_residual_m'], row['rms_residual_m']
        if not stat['count']:
            if maximum is not None or rms is not None:
                raise ValueError('missing points supply no residual or geometric agreement')
            continue
        if not np.isfinite([maximum, rms]).all() or not 0 <= rms <= maximum <= .003:
            raise ValueError('current measured candidate conflicts with transported floor reference')
        mean = np.asarray(stat['mean_body_m']); covariance = np.asarray(stat['covariance_body_m2'])
        square = float(normal@covariance@normal+(normal@mean+offset)**2)
        if not np.isclose(rms**2, square, rtol=1e-8, atol=1e-12):
            raise ValueError('transport residual moments do not reconstruct from all current candidates')


def current_measured_floor_pose(evidence, *, identity, now_ns):
    """Explicit dispatch; old evidence remains subject to its unchanged accessor."""
    if evidence['schema'] == FLOOR_SCHEMA:
        return current_joint_floor_registered_pose(evidence, identity=identity, now_ns=now_ns)
    if (evidence['schema'] != SCHEMA or _identity(evidence['identity']) != _identity(identity)
            or evidence['decision_ns'] != _ns(now_ns, 'transport pose decision')
            or evidence['status'] != 'CURRENT_VISUAL_FLOOR_TRANSPORT_POSE'
            or evidence['native_pose_used'] is not False or evidence['command_integration_used'] is not False
            or evidence['historical_map_rewritten'] is not False or evidence['navigation_qualified'] is not False):
        raise ValueError('current separately typed measured floor transport required')
    raw = evidence['original_visual_evidence']; witness = evidence['floor_transport']
    _, _, raw_pose = current_joint_pose(raw, identity=identity, now_ns=now_ns)
    if (witness['frame'] != raw_pose['frame'] or witness['measured_ns'] != now_ns
            or witness['rgb_sha256'] != raw_pose['rgb_sha256']
            or witness['primary_depth_sha256'] != raw_pose['depth_sha256']
            or witness['auxiliary_depth_sha256'] != raw_pose['auxiliary_depth_sha256']):
        raise ValueError('same current paired visual and depth witnesses required')
    correction = composition(witness['anchor'], raw, witness['unavailable_current_plane'],
        identity=identity, now_ns=now_ns)
    if correction != witness['correction']:
        raise ValueError('current transported pose must reconstruct from the admitted anchor and raw visual pose')
    check_candidate_residuals(witness['camera_residuals'], witness['unavailable_current_plane'], correction)
    expected = deepcopy(raw_pose) | dict(mode='measured_visual_floor_transport',
        position_initial_body_m=correction['position_initial_body_m'],
        rotation_initial_body_from_current_body=correction['rotation_initial_body_from_current_body'],
        current_floor_registration_used=False, historical_floor_anchor_used=True,
        **original_pose_validation_fields(raw_pose))
    if evidence['current_pose'] != expected:
        raise ValueError('transport pose differs from measured witness composition')
    return (np.asarray(expected['position_initial_body_m']),
        np.asarray(expected['rotation_initial_body_from_current_body']), evidence['current_pose'])


def transport_evidence(anchor, raw, plane, clouds, *, identity, now_ns, auxiliary_depth_sha256):
    correction = composition(anchor, raw, plane, identity=identity, now_ns=now_ns)
    if fit_joint_plane(*clouds, plane['up_body']) != plane:
        raise ValueError('unchanged current measured candidates must reconstruct the missing plane')
    rows = []
    n = np.asarray(correction['transported_reference_normal_body']); d = correction['transported_reference_offset_body_m']
    for camera, cloud in zip(CAMERAS, clouds, strict=True):
        errors = cloud@n+d
        rows.append(dict(camera=camera, count=len(cloud),
            maximum_residual_m=float(np.abs(errors).max()) if len(cloud) else None,
            rms_residual_m=float(np.sqrt(np.mean(errors**2))) if len(cloud) else None))
    check_candidate_residuals(rows, plane, correction)
    pose = raw['current_pose']
    result = dict(schema=SCHEMA, identity=tuple(identity), decision_ns=now_ns,
        status='CURRENT_VISUAL_FLOOR_TRANSPORT_POSE', original_visual_evidence=deepcopy(raw),
        current_pose=deepcopy(pose) | dict(mode='measured_visual_floor_transport',
            position_initial_body_m=correction['position_initial_body_m'],
            rotation_initial_body_from_current_body=correction['rotation_initial_body_from_current_body'],
            current_floor_registration_used=False, historical_floor_anchor_used=True,
            **original_pose_validation_fields(pose)),
        floor_transport=dict(frame=pose['frame'], measured_ns=now_ns,
            rgb_sha256=pose['rgb_sha256'], primary_depth_sha256=pose['depth_sha256'],
            auxiliary_depth_sha256=auxiliary_depth_sha256, anchor=deepcopy(anchor),
            unavailable_current_plane=deepcopy(plane), correction=correction, camera_residuals=rows),
        native_pose_used=False, command_integration_used=False, historical_map_rewritten=False,
        navigation_qualified=False)
    current_measured_floor_pose(result, identity=identity, now_ns=now_ns)
    return result
