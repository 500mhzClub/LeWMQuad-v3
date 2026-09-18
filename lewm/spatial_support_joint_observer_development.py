"""Distinct spatial-feature joint observer; all registration/continuity gates inherited."""
import hashlib
import cv2
import numpy as np
from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.causal_depth_observation_development import validate_depth
from lewm.multi_reference_rgbd_pose_development import Reference
from lewm.joint_rgbd_rigid_pose_development import proper, angle
from lewm.support_aware_rgbd_pose_development import support_near_limit
from lewm.temporal_anchor_continuity_development import CONTINUITY_RULES
from lewm.joint_temporal_anchor_continuity_development import (
    JointTemporalAnchorRGBDPose, JointTemporalAnchorVisualLedMotion)
from lewm.spatial_support_features_development import SpatialSupportFeatureFrame


class SpatialSupportJointRGBDPose(JointTemporalAnchorRGBDPose):
    def observe(self, policy, depth, fast, *, now_ns):
        if self.failed:
            raise SensorContractError('temporal anchor observer terminal; no recovery or reinitialization')
        self.last_continuity = dict(status='VALIDATING_CURRENT_INPUT', uncertainty_calibrated=False)
        try:
            now = _ns(now_ns, 'temporal anchor query')
            if self.previous is not None and now - self.previous.measured_ns != CONTINUITY_RULES['sample_interval_ns']:
                raise SensorContractError('exact100ms accepted visual interval required')
            validate_depth(depth, policy, now_ns=now)
            attitude = (self.gyro.begin(policy, fast, now_ns=now) if self.reference is None
                        else self.gyro.step(policy, fast, now_ns=now))
            G = np.asarray(attitude['rotation_initial_body_from_current_body'])
            current = SpatialSupportFeatureFrame(policy['image']['rgb'], depth)
            self.frame += 1
            reg = None; reason = None; ref = 0
            if self.reference is None:
                R = np.eye(3); p = np.zeros(3)
                self._remember(current, R, G, p, now)
                self.nodes.append(dict(frame=0, measured_ns=now, parent_frame=None,
                    position_initial_body_m=p.tolist(), rotation_initial_body_from_current_body=R.tolist(),
                    pose_error_bound=None))
                self.last_selection = dict(status='INITIAL_REFERENCE', selected_reference=0, retained_references=1)
                self.last_continuity = dict(status='INITIAL_REFERENCE', bridge_frames=0,
                    bridge_path_m=0., uncertainty_calibrated=False)
            else:
                c, fallback, bridge = self._measure(current, G, now)
                R, p, reg = c['R'], c['p'], c['registration']; ref = c['reference'].frame
                if not bridge:
                    motion = np.linalg.norm(c['t']) >= .4 or angle(c['local_R']) >= .35
                    reason = ('qualified_recent_reference' if fallback else 'motion_threshold' if motion
                              else 'accepted_support_margin' if support_near_limit(reg) else None)
                    if reason is not None:
                        self.nodes.append(dict(frame=self.frame, measured_ns=now, parent_frame=ref,
                            position_initial_body_m=p.tolist(), rotation_initial_body_from_current_body=R.tolist(),
                            pose_error_bound=None))
                        self._remember(current, R, G, p, now)
            proper(R)
            self.last_R = R.copy(); self.last_p = p.copy()
            self.previous = Reference(self.frame, now, current, R.copy(), G.copy(), p.copy())
            return dict(frame=self.frame, measured_ns=now, mode='joint', reference_frame=ref,
                position_initial_body_m=p.tolist(), rotation_initial_body_from_current_body=R.tolist(),
                gyro_rotation_initial_body_from_current_body=G.tolist(),
                global_image_gyro_disagreement_rad=angle(G.T @ R), registration=reg,
                promoted_keyframe=reason is not None, promotion_reason=reason, keyframe_count=len(self.nodes),
                rgb_sha256=depth['rgb_sha256'], depth_sha256=hashlib.sha256(
                    depth['depth_m'].tobytes() + depth['valid'].tobytes()).hexdigest(),
                position_error_bound=None, orientation_error_bound=None, uncertainty_model_validated=False,
                gyro_role='consistency_monitor_only', native_pose_input=False, global_history_reset=False,
                navigation_qualified=False)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, cv2.error, np.linalg.LinAlgError) as error:
            self.failed = True
            raise SensorContractError('temporal anchor RGBD pose unavailable; terminal failure') from error


class SpatialSupportVisualLedMotion(JointTemporalAnchorVisualLedMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = SpatialSupportJointRGBDPose()

    def snapshot(self, *, now_ns):
        row = super().snapshot(now_ns=now_ns)
        previous = self.model.previous
        return row | dict(feature_selection="spatial_public_depth_sift_v1",
            last_accepted_feature_witness=None if previous is None else previous.features.witness(),
            candidate_adopted=False)

