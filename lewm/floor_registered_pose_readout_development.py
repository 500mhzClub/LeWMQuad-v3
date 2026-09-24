"""Evaluator-only paired raw/registered pose accuracy; no counterfactual motion."""
import numpy as np
from lewm.floor_registered_evidence_development import current_registered_pose
from lewm.floor_registered_evidence_development import SCHEMA as INDEPENDENT_SCHEMA
from lewm.joint_floor_registered_evidence_development import (
    current_joint_floor_registered_pose, SCHEMA as JOINT_SCHEMA)
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.joint_rgbd_rigid_pose_development import angle
from lewm.physical_execution_development import rotation_xyzw


def json_identity(evidence):
    identity = evidence.get('identity')
    if (type(identity) is not list or len(identity) != 3
            or any(type(v) is not int or v < 0 for v in identity)):
        raise ValueError('three nonnegative JSON integer identity fields required')
    return evidence | dict(identity=tuple(identity))


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
        e = json_identity(evidence)
        e['original_visual_evidence'] = json_identity(e['original_visual_evidence'])
        accessor = {INDEPENDENT_SCHEMA: current_registered_pose,
            JOINT_SCHEMA: current_joint_floor_registered_pose}.get(e['schema'])
        if accessor is None: raise ValueError('explicit independently validated registration schema required')
        p, R, pose = accessor(e, identity=(0, 0, 0), now_ns=now)
        raw_p, raw_R, raw_pose = current_joint_pose(e['original_visual_evidence'], identity=(0, 0, 0), now_ns=now)
        if pose['frame'] != frame or raw_pose['frame'] != frame:
            raise ValueError('matching current registered and original visual frames required')
        if decision['original_visual_evidence'] != evidence['original_visual_evidence']:
            raise ValueError('complete unchanged raw visual record required')
        actual_p = initial.T@(poses[sample, :3]-poses[749, :3])
        actual_R = initial.T@rotation_xyzw(poses[sample, 3:])
        correction = evidence['floor_registration']['correction']
        records.append(dict(frame=frame, physical_sample_index=sample,
            raw_xyz_error_m=float(np.linalg.norm(raw_p-actual_p)),
            registered_xyz_error_m=float(np.linalg.norm(p-actual_p)),
            raw_xy_error_m=float(np.linalg.norm(raw_p[:2]-actual_p[:2])),
            registered_xy_error_m=float(np.linalg.norm(p[:2]-actual_p[:2])),
            raw_rotation_error_rad=angle(actual_R.T@raw_R),
            registered_rotation_error_rad=angle(actual_R.T@R),
            normal_translation_correction_m=correction['normal_translation_correction_m'],
            normal_alignment_rad=correction['normal_alignment_rad']))
    keys = ('raw_xyz_error_m', 'registered_xyz_error_m', 'raw_xy_error_m', 'registered_xy_error_m',
        'raw_rotation_error_rad', 'registered_rotation_error_rad')
    summaries = {k: dict(maximum=max((r[k] for r in records), default=None),
        mean=float(np.mean([r[k] for r in records])) if records else None) for k in keys}
    return dict(records=records, admitted_pose_frames=len(records), terminal_frames_without_pose=missing,
        error_summary=summaries, native_state_evaluator_only=True, native_state_used_for_commands=False,
        accuracy_on_executed_trajectory_only=True, unexecuted_trajectories_inferred=False,
        pose_uncertainty_calibrated=False, physical_floor_identity_certified=False,
        navigation_qualified=False, hardware_qualified=False)
