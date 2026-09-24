"""Longer-budget measured transport; only the frame ceiling differs.

This is not integrated into a native controller. Original measured plane,
correction, continuity and conflict rules remain in force.
"""
import numpy as np
from lewm import measured_floor_transport_development as original
from lewm.measured_floor_transport_development import (
    _ns, current_joint_pose, current_joint_floor_registered_pose, unit,
    compose_plane, MISSING, proper, angle)
from lewm.extended_return_budget_mission_development import MAX_OBSERVATIONS
from lewm.eligible_floor_registration_development import bind
from lewm.tiled_density_floor_registration_development import TiledDensityFloorRegistration


def composition(anchor, raw, missing_plane, *, identity, now_ns):
    """Validate both measured poses and return their explicit SE(3) composition."""
    now = _ns(now_ns, 'measured floor transport')
    p, R, pose = current_joint_pose(raw, identity=identity, now_ns=now)
    anchor_ns = anchor['decision_ns']
    a, A, anchor_pose = current_joint_floor_registered_pose(anchor, identity=identity, now_ns=anchor_ns)
    b, B, raw_anchor = current_joint_pose(anchor['original_visual_evidence'], identity=identity, now_ns=anchor_ns)
    if (not 0 <= anchor_pose['frame'] < pose['frame'] < MAX_OBSERVATIONS
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

current_measured_floor_pose = bind(original.current_measured_floor_pose, composition=composition)
transport_evidence = bind(original.transport_evidence, composition=composition,
    current_measured_floor_pose=current_measured_floor_pose)


class ExtendedReturnBudgetFloorRegistration(TiledDensityFloorRegistration):
    observe = bind(TiledDensityFloorRegistration.observe, transport_evidence=transport_evidence)
