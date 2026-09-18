"""Front-first measured RGB-D continuity with downward-camera missingness fallback.

Each retained reference stores both simultaneously acquired views at one
accepted body pose. No cross-camera descriptor matching, history reset,
command extrapolation, native state or calibrated uncertainty is supplied.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import cv2
import numpy as np
from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.causal_depth_observation_development import validate_depth
from lewm.causal_auxiliary_rgb_observation_development import validate_rgb, depth_digest
from lewm.corner_support_features_development import CornerSupportFeatureFrame
from lewm.joint_temporal_anchor_continuity_development import JointTemporalAnchorRGBDPose
from lewm.multi_reference_rgbd_pose_development import Reference
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.joint_rgbd_rigid_pose_development import register, proper, angle, RIGID_RULES
from lewm.auxiliary_reference_pose_adapter_development import (
    body_from_reference, gyro_in_reference, pose_in_body)
from lewm.temporal_anchor_continuity_development import CONTINUITY_RULES
from lewm.support_aware_rgbd_pose_development import support_near_limit
from lewm.overlap_retention_joint_observer_development import overlap_retention


class DualCameraAnchorPose(JointTemporalAnchorRGBDPose):
    overlap_promotion_enabled = True

    def __init__(self):
        super().__init__()
        self.camera = 'primary'
        self.last_camera_selection = None
        self.last_overlap_retention = None

    def _candidate(self, ref, current, G):
        if self.camera == 'primary':
            candidate = super()._candidate(replace(ref, features=ref.features['primary']),
                current['primary'], G)
            candidate['reference'] = ref
            return candidate
        a, b, ua, ub = matched_points(ref.features['auxiliary'], current['auxiliary'])
        relative_gyro = ref.gyro.T @ G
        Q, v, mask, reg = register(a, b, ua, ub,
            gyro_rotation=gyro_in_reference(relative_gyro), mode='joint', frame=self.frame)
        local_R, t = pose_in_body(Q, v)
        R = ref.rotation @ local_R
        p = ref.position + ref.rotation @ t
        if (np.linalg.norm(t) > RIGID_RULES['maximum_reference_translation_m']
                or np.linalg.norm(p-self.last_p) > RIGID_RULES['maximum_increment_translation_m']
                or angle(self.last_R.T @ R) > RIGID_RULES['maximum_increment_rotation_rad']):
            raise SensorContractError('auxiliary body-frame displacement envelope rejected')
        A, offset = body_from_reference()
        reg |= dict(reference_frame=ref.frame, reference_measured_ns=ref.measured_ns,
            translation_reference_body_m=t.tolist(), relative_rotation=local_R.tolist(),
            gyro_relative_rotation=relative_gyro.tolist(),
            reference_inlier_pixels=ua[mask].tolist(), current_inlier_pixels=ub[mask].tolist(),
            reference_inlier_points_body_m=(a[mask] @ A.T+offset).tolist(),
            current_inlier_points_body_m=(b[mask] @ A.T+offset).tolist(),
            camera='auxiliary', fixed_reference_frame_adapter_used=True)
        if len(self.rotation_measurements) >= 9:
            raise SensorContractError('bounded eight anchors plus one increment required per camera')
        self.rotation_measurements.append(dict(reference_frame=ref.frame,
            reference_measured_ns=ref.measured_ns, current_frame=self.frame,
            reference_rotation_initial_body_from_reference_body=ref.rotation.tolist(),
            fitted_rotation_reference_body_from_current_body=local_R.tolist(),
            composed_rotation_initial_body_from_current_body=R.tolist(),
            gyro_rotation_reference_body_from_current_body=relative_gyro.tolist(),
            position_initial_body_m=p.tolist(), fitting_mode=reg['mode'],
            inliers=reg['inliers'], inlier_fraction=reg['inlier_fraction'],
            reference_grid_cells=reg['reference_grid_cells'], current_grid_cells=reg['current_grid_cells'],
            residual_rms_m=reg['residual_rms_m'], gyro_disagreement_rad=reg['gyro_disagreement_rad'],
            candidate_envelope_passed=True, witness_alone_grants_pose=False, camera='auxiliary'))
        return dict(reference=ref, R=R, p=p, local_R=local_R, t=t, registration=reg)

    def _measure(self, current, G, now):
        self.camera = 'primary'
        self.last_camera_selection = dict(selected_camera=None, auxiliary_attempted=False,
            primary_failure=None, primary_continuity=None, primary_reference_selection=None,
            cross_camera_disagreement_m=None, cross_camera_disagreement_rad=None,
            measurements_independent=False, thresholds_unchanged=True)
        try:
            candidate = super()._measure(current, G, now)
            self.last_camera_selection['selected_camera'] = 'primary'
            return candidate
        except SensorContractError as error:
            self.last_camera_selection.update(primary_failure=str(error),
                primary_continuity=deepcopy(self.last_continuity),
                primary_reference_selection=deepcopy(self.last_selection))
            # A qualified conflict is never reinterpreted as missing input.
            if self.last_continuity.get('status') not in (
                    'NO_CURRENT_MEASURED_TRANSLATION', 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'):
                raise
        self.camera = 'auxiliary'
        self.last_camera_selection['auxiliary_attempted'] = True
        candidate = super()._measure(current, G, now)
        primary = self.last_camera_selection['primary_continuity']
        witness = primary.get('incremental_rotation_witness')
        if primary.get('incremental_available'):
            if witness is None:
                raise SensorContractError('qualified primary increment requires its rotation witness')
            distance = float(np.linalg.norm(candidate[0]['p']-primary['incremental_position_initial_body_m']))
            rotation = angle(np.asarray(witness['composed_rotation_initial_body_from_current_body']).T @ candidate[0]['R'])
            self.last_camera_selection.update(cross_camera_disagreement_m=distance,
                cross_camera_disagreement_rad=rotation)
            if (distance > CONTINUITY_RULES['maximum_measured_disagreement_m']
                    or rotation > CONTINUITY_RULES['maximum_measured_disagreement_rad']):
                self.last_continuity['status'] = 'CROSS_CAMERA_MEASUREMENT_CONFLICT'
                raise SensorContractError('qualified primary increment and auxiliary pose conflict')
        self.last_camera_selection['selected_camera'] = 'auxiliary'
        return candidate

    def observe(self, policy, depth, fast, *, auxiliary_rgb, auxiliary_depth, now_ns):
        if self.failed:
            raise SensorContractError('dual-camera observer terminal; no recovery or reinitialization')
        self.last_overlap_retention = None
        self.last_camera_selection = None
        self.last_continuity = dict(status='VALIDATING_CURRENT_INPUT', uncertainty_calibrated=False)
        try:
            now = _ns(now_ns, 'dual-camera query')
            if depth['measured_ns'] != now:
                raise SensorContractError('current co-timed RGB-D required')
            if self.previous is not None and now-self.previous.measured_ns != CONTINUITY_RULES['sample_interval_ns']:
                raise SensorContractError('exact100ms accepted visual interval required')
            validate_depth(depth, policy, now_ns=now)
            validate_rgb(auxiliary_rgb, auxiliary_depth, policy, now_ns=now)
            attitude = (self.gyro.begin(policy, fast, now_ns=now) if self.reference is None
                        else self.gyro.step(policy, fast, now_ns=now))
            G = np.asarray(attitude['rotation_initial_body_from_current_body'])
            current = dict(primary=CornerSupportFeatureFrame(policy['image']['rgb'], depth),
                auxiliary=CornerSupportFeatureFrame(auxiliary_rgb['rgb'], auxiliary_depth))
            self.frame += 1
            reg = None; reason = None; ref_frame = 0
            if self.reference is None:
                R = np.eye(3); p = np.zeros(3)
                self._remember(current, R, G, p, now)
                self.nodes.append(dict(frame=0, measured_ns=now, parent_frame=None,
                    position_initial_body_m=p.tolist(), rotation_initial_body_from_current_body=R.tolist(),
                    pose_error_bound=None))
                self.last_selection = dict(status='INITIAL_REFERENCE', selected_reference=0, retained_references=1)
                self.last_continuity = dict(status='INITIAL_REFERENCE', bridge_frames=0,
                    bridge_path_m=0., uncertainty_calibrated=False)
                self.last_camera_selection = dict(selected_camera=None, auxiliary_attempted=False,
                    initial_paired_reference=True)
            else:
                candidate, fallback, bridge = self._measure(current, G, now)
                R, p, reg = candidate['R'], candidate['p'], candidate['registration']
                ref = candidate['reference']; ref_frame = ref.frame
                if not bridge:
                    motion = np.linalg.norm(candidate['t']) >= .4 or angle(candidate['local_R']) >= .35
                    reason = ('qualified_recent_reference' if fallback else 'motion_threshold' if motion
                              else 'accepted_support_margin' if support_near_limit(reg) else None)
                    reference_features = (sum(len(ref.features[c].keypoints) for c in ('primary', 'auxiliary'))
                        if reg.get('joint_camera_retained_anchor') else len(ref.features[self.camera].keypoints))
                    receipt = overlap_retention(inliers=reg['inliers'],
                        reference_features=reference_features, anchored=True,
                        already_promoted=reason is not None)
                    self.last_overlap_retention = receipt | dict(frame=self.frame,
                        reference_frame=ref_frame, measured_ns=now)
                    if receipt['retain'] and self.overlap_promotion_enabled:
                        reason = 'accepted_half_feature_overlap'
                    if reason is not None:
                        self.nodes.append(dict(frame=self.frame, measured_ns=now, parent_frame=ref_frame,
                            position_initial_body_m=p.tolist(), rotation_initial_body_from_current_body=R.tolist(),
                            pose_error_bound=None))
                        self._remember(current, R, G, p, now)
            proper(R)
            self.last_R = R.copy(); self.last_p = p.copy()
            self.previous = Reference(self.frame, now, current, R.copy(), G.copy(), p.copy())
            return dict(frame=self.frame, measured_ns=now, mode='joint', reference_frame=ref_frame,
                position_initial_body_m=p.tolist(), rotation_initial_body_from_current_body=R.tolist(),
                gyro_rotation_initial_body_from_current_body=G.tolist(),
                global_image_gyro_disagreement_rad=angle(G.T @ R), registration=reg,
                promoted_keyframe=reason is not None, promotion_reason=reason, keyframe_count=len(self.nodes),
                rgb_sha256=depth['rgb_sha256'], depth_sha256=hashlib.sha256(
                    depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest(),
                auxiliary_rgb_sha256=auxiliary_rgb['rgb_sha256'], auxiliary_depth_sha256=depth_digest(auxiliary_depth),
                camera_selection=deepcopy(self.last_camera_selection),
                position_error_bound=None, orientation_error_bound=None, uncertainty_model_validated=False,
                gyro_role='consistency_monitor_only', native_pose_input=False, global_history_reset=False,
                navigation_qualified=False)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, cv2.error, np.linalg.LinAlgError) as error:
            self.failed = True
            raise SensorContractError('dual-camera RGBD pose unavailable; terminal failure') from error
