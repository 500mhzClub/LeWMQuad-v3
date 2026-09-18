"""Evaluate actual raw/registered/transported poses without admitting synthetic planes."""
import numpy as np
from lewm.measured_floor_transport_json_development import restore_evidence
from lewm.measured_floor_transport_development import current_measured_floor_pose, SCHEMA as TRANSPORT_SCHEMA
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.joint_rgbd_rigid_pose_development import angle
from lewm.physical_execution_development import rotation_xyzw


def registered_pose_accuracy(poses, rows):
    poses = np.asarray(poses, float)
    if (poses.ndim != 2 or poses.shape[1:] != (7,) or len(poses) < 750
            or not np.isfinite(poses).all()
            or not np.allclose(np.linalg.norm(poses[:, 3:], axis=1), 1., atol=1e-6, rtol=0)):
        raise ValueError('complete finite normalized native pose trace required')
    initial = rotation_xyzw(poses[749, 3:]); records = []; missing = []
    for frame, row in enumerate(rows):
        if row['tick'] != frame: raise ValueError('consecutive recorded decisions required')
        now = 1_500_000_000+frame*100_000_000; sample = 749+50*frame
        if sample >= len(poses): raise ValueError('current observation requires its actual physical endpoint')
        decision = row['decision']; evidence = decision['evidence']
        if evidence is None:
            if decision['terminal'] is None: raise ValueError('nonterminal decision missing registered pose')
            missing.append(frame); continue
        e = restore_evidence(evidence)
        p, R, pose = current_measured_floor_pose(e, identity=(0, 0, 0), now_ns=now)
        raw_p, raw_R, raw_pose = current_joint_pose(e['original_visual_evidence'], identity=(0, 0, 0), now_ns=now)
        if pose['frame'] != frame or raw_pose['frame'] != frame:
            raise ValueError('matching current registered and original visual frames required')
        if decision['original_visual_evidence'] != evidence['original_visual_evidence']:
            raise ValueError('complete unchanged raw visual record required')
        actual_p = initial.T@(poses[sample, :3]-poses[749, :3])
        actual_R = initial.T@rotation_xyzw(poses[sample, 3:])
        transported = evidence['schema'] == TRANSPORT_SCHEMA
        correction = evidence['floor_transport' if transported else 'floor_registration']['correction']
        records.append(dict(frame=frame, physical_sample_index=sample,
            raw_xyz_error_m=float(np.linalg.norm(raw_p-actual_p)),
            registered_xyz_error_m=float(np.linalg.norm(p-actual_p)),
            raw_xy_error_m=float(np.linalg.norm(raw_p[:2]-actual_p[:2])),
            registered_xy_error_m=float(np.linalg.norm(p[:2]-actual_p[:2])),
            raw_rotation_error_rad=angle(actual_R.T@raw_R),
            registered_rotation_error_rad=angle(actual_R.T@R),
            normal_translation_correction_m=None if transported else correction['normal_translation_correction_m'],
            normal_alignment_rad=None if transported else correction['normal_alignment_rad'],
            pose_admission_kind='measured_visual_floor_transport' if transported else 'current_joint_floor_registration',
            floor_anchor_age_frames=correction['anchor_age_frames'] if transported else 0,
            transport_correction_magnitude_m=correction['correction_magnitude_m'] if transported else None,
            transport_correction_angle_rad=correction['correction_angle_rad'] if transported else None))
    keys = ('raw_xyz_error_m', 'registered_xyz_error_m', 'raw_xy_error_m', 'registered_xy_error_m',
        'raw_rotation_error_rad', 'registered_rotation_error_rad')
    summaries = {k: dict(maximum=max((r[k] for r in records), default=None),
        mean=float(np.mean([r[k] for r in records])) if records else None) for k in keys}
    return dict(records=records, admitted_pose_frames=len(records), terminal_frames_without_pose=missing,
        error_summary=summaries, transported_pose_frames=sum(r['pose_admission_kind']=='measured_visual_floor_transport' for r in records),
        native_state_evaluator_only=True, native_state_used_for_commands=False,
        accuracy_on_executed_trajectory_only=True, unexecuted_trajectories_inferred=False,
        pose_uncertainty_calibrated=False, physical_floor_identity_certified=False,
        navigation_qualified=False, hardware_qualified=False)

