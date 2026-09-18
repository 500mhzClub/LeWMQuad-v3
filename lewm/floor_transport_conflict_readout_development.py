"""Reconstruct an unchanged floor-transport rejection from admitted witnesses."""
from copy import deepcopy
import hashlib
import numpy as np
from lewm.causal_depth_observation_development import BODY_FROM_OPTICAL
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical
from lewm.floor_pose_registration_development import measured_candidates, ROWS, COLUMNS
from lewm.joint_measured_floor_plane_development import CAMERAS, fit_joint_plane
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.joint_floor_registered_evidence_development import SCHEMA as FLOOR_SCHEMA
from lewm.floor_registered_pose_readout_development import json_identity
from lewm.measured_floor_transport_json_development import restore_evidence
from lewm.measured_floor_transport_development import current_measured_floor_pose, composition
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.dual_camera_visual_motion_development import current_dual_camera_pose

FAILURE = 'current measured candidate conflicts with transported floor reference'


def candidate_residuals(cloud, mask, normal, offset, *, camera):
    cloud = np.asarray(cloud, float); mask = np.asarray(mask)
    if (cloud.ndim != 2 or cloud.shape[1:] != (3,) or not np.isfinite(cloud).all()
            or mask.dtype != bool or mask.shape != (len(ROWS), len(COLUMNS)) or mask.sum() != len(cloud)):
        raise ValueError('complete ordered candidate cloud and sampling mask required')
    errors = cloud@normal+offset
    row = dict(camera=camera, count=len(cloud), maximum_residual_m=None, rms_residual_m=None,
        mean_signed_residual_m=None, minimum_signed_residual_m=None, maximum_signed_residual_m=None,
        over_3mm_count=0, worst_candidate=None,
        candidate_points_sha256=hashlib.sha256(cloud.tobytes()).hexdigest(),
        candidate_mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest())
    if len(cloud):
        index = int(np.argmax(np.abs(errors))); rr, cc = np.nonzero(mask)
        row.update(maximum_residual_m=float(np.abs(errors).max()),
            rms_residual_m=float(np.sqrt(np.mean(errors**2))), mean_signed_residual_m=float(errors.mean()),
            minimum_signed_residual_m=float(errors.min()), maximum_signed_residual_m=float(errors.max()),
            over_3mm_count=int((np.abs(errors) > .003).sum()),
            worst_candidate=dict(candidate_index=index, sampled_row=int(ROWS[rr[index]]),
                sampled_column=int(COLUMNS[cc[index]]), point_body_m=cloud[index].tolist(),
                signed_residual_m=float(errors[index])))
    return row


def reconstruct(prior_evidence, raw_evidence, policy, primary, auxiliary, image, *, now_ns):
    prior = restore_evidence(prior_evidence); raw = json_identity(raw_evidence)
    _, _, previous_pose = current_measured_floor_pose(prior, identity=(0, 0, 0), now_ns=now_ns-100_000_000)
    current_dual_camera_pose(raw, policy, image, auxiliary, identity=(0, 0, 0), now_ns=now_ns)
    _, R, pose = current_joint_pose(raw, identity=(0, 0, 0), now_ns=now_ns)
    if previous_pose['frame']+1 != pose['frame']:
        raise ValueError('immediately preceding admitted floor pose required')
    anchor = prior if prior['schema'] == FLOOR_SCHEMA else prior['floor_transport']['anchor']
    registration = MeasuredFloorTransportRegistration(identity=(0, 0, 0))
    registration.anchor = deepcopy(anchor)
    registration.reference = deepcopy(anchor['floor_registration']['reference'])
    registration.frame = previous_pose['frame']
    preserved = deepcopy((registration.anchor, registration.reference, registration.frame))
    try:
        registration.observe(policy, primary, auxiliary, raw, now_ns=now_ns)
    except ValueError as error:
        if str(error) != FAILURE: raise ValueError('different original registration failure: '+str(error)) from error
    else:
        raise ValueError('original registration did not reproduce the expected conflict')
    if not registration.failed or (registration.anchor, registration.reference, registration.frame) != preserved:
        raise ValueError('original failure latch and unchanged anchor/reference/frame required')
    up = R.T@np.asarray(registration.reference['initial_up_body'])
    clouds = []; masks = []
    for depth, E in ((primary, np.asarray(BODY_FROM_OPTICAL)), (auxiliary, body_from_optical())):
        cloud, mask = measured_candidates(depth['depth_m'], depth['valid'], E, up)
        clouds.append(cloud); masks.append(mask)
    joint = fit_joint_plane(*clouds, up)
    correction = composition(anchor, raw, joint, identity=(0, 0, 0), now_ns=now_ns)
    normal = np.asarray(correction['transported_reference_normal_body'])
    offset = correction['transported_reference_offset_body_m']
    rows = [candidate_residuals(cloud, mask, normal, offset, camera=camera)
        for cloud, mask, camera in zip(clouds, masks, CAMERAS, strict=True)]
    if not any(row['over_3mm_count'] for row in rows):
        raise ValueError('reconstructed complete candidate population must explain the rejection')
    return dict(frame=pose['frame'], measured_ns=now_ns, prior_frame=previous_pose['frame'],
        original_failure=FAILURE, original_registration_rejection_reproduced=True,
        original_failure_latched=True, original_anchor_reference_and_frame_preserved=True,
        joint_plane=joint, correction=correction, camera_residuals=rows,
        conflicting_cameras=[row['camera'] for row in rows if row['over_3mm_count']],
        original_visual_pose=deepcopy(pose), selected_camera=raw['camera_selection']['selected_camera'],
        direct_flow_fallback=deepcopy(raw.get('direct_corner_flow_fallback')),
        every_current_candidate_retained=True, thresholds_changed=False,
        state_restored_from_immediately_preceding_admitted_receipt=True,
        tracker_or_model_reexecuted=False, controller_decisions_changed=False,
        native_pose_used=False, physical_floor_identity_certified=False,
        navigation_qualified=False, goal_achieved=False)
