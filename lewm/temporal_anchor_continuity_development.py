"""Bounded measured RGB-D increments across unavailable retained anchors.

No commanded motion, inertial/contact fallback, pose reset or weakened pair
gate. A bridge is a chain of current measurements, not an uncertainty bound.
Retained anchors are not promoted from bridge-only poses.
"""
from copy import deepcopy
import hashlib

import cv2
import numpy as np

from lewm.causal_sensor_state import SensorContractError, _ns
from lewm.causal_depth_observation_development import validate_depth
from lewm.keyframe_rgbd_pose_development import FeatureFrame
from lewm.multi_reference_rgbd_pose_development import (
    MultiReferenceRGBDPose, MultiReferenceVisualLedMotion, Reference)
from lewm.joint_rgbd_rigid_pose_development import proper, angle
from lewm.support_aware_rgbd_pose_development import support_near_limit

CONTINUITY_RULES = dict(maximum_bridge_frames=10, sample_interval_ns=100_000_000,
    maximum_measured_disagreement_m=.02, maximum_measured_disagreement_rad=.10)


class TemporalAnchorRGBDPose(MultiReferenceRGBDPose):
    def __init__(self):
        super().__init__()
        self.previous = None
        self.bridge_frames = self.total_bridge_frames = 0
        self.bridge_path_m = 0.
        self.last_continuity = None

    def _measure(self, current, G, now):
        """Both hypotheses use the unchanged descriptor matcher and rigid gates."""
        if self.previous is None or now - self.previous.measured_ns != CONTINUITY_RULES['sample_interval_ns']:
            raise SensorContractError('immediately preceding accepted visual observation required')
        evidence = dict(status='MEASURING', incremental_available=False,
            anchor_available=False, measurements_independent=False,
            previous_frame=self.previous.frame, previous_measured_ns=self.previous.measured_ns,
            disagreement_m=None, disagreement_rad=None, anchor_failure=None,
            incremental_failure=None, error_bound_m=None, uncertainty_calibrated=False)
        self.last_continuity = evidence
        anchor = None
        try:
            anchor, alternative = super()._choose(current, G)
            evidence['anchor_available'] = True
        except SensorContractError as error:
            evidence['anchor_failure'] = str(error)
            # Contradictory qualified anchors are not mere missingness.
            if self.last_selection['status'] != 'NO_QUALIFIED_REFERENCE':
                evidence['status'] = 'ANCHOR_CONFLICT_OR_INVALID'
                raise
        incremental = None
        if anchor is not None and anchor['reference'].frame == self.previous.frame:
            incremental = anchor
            evidence['same_reference_measurement_reused'] = True
        else:
            evidence['same_reference_measurement_reused'] = False
            try:
                incremental = self._candidate(self.previous, current, G)
            except SensorContractError as error:
                evidence['incremental_failure'] = str(error)
        evidence['incremental_available'] = incremental is not None
        if incremental is not None:
            evidence['incremental_position_initial_body_m'] = incremental['p'].tolist()
        if anchor is not None:
            if incremental is not None:
                distance = float(np.linalg.norm(anchor['p'] - incremental['p']))
                rotation = angle(anchor['R'].T @ incremental['R'])
                evidence.update(disagreement_m=distance, disagreement_rad=rotation)
                if (distance > CONTINUITY_RULES['maximum_measured_disagreement_m']
                        or rotation > CONTINUITY_RULES['maximum_measured_disagreement_rad']):
                    evidence['status'] = 'ANCHOR_INCREMENT_CONFLICT'
                    self.last_selection = self.last_selection | dict(status='ANCHOR_INCREMENT_CONFLICT')
                    raise SensorContractError('qualified anchor and incremental measurements conflict')
            evidence.update(status='ANCHOR_MEASUREMENT', preceding_bridge_frames=self.bridge_frames,
                preceding_bridge_path_m=self.bridge_path_m, bridge_frames=0, bridge_path_m=0.)
            self.bridge_frames = 0
            self.bridge_path_m = 0.
            return anchor, alternative, False
        if incremental is None:
            evidence['status'] = 'NO_CURRENT_MEASURED_TRANSLATION'
            raise SensorContractError('neither retained anchor nor previous frame supports current pose')
        if self.bridge_frames >= CONTINUITY_RULES['maximum_bridge_frames']:
            evidence['status'] = 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'
            raise SensorContractError('bounded measured bridge exhausted without anchor observation')
        self.bridge_frames += 1
        self.total_bridge_frames += 1
        self.bridge_path_m += float(np.linalg.norm(incremental['p'] - self.last_p))
        evidence.update(status='MEASURED_INCREMENT_BRIDGE', bridge_frames=self.bridge_frames,
            bridge_path_m=self.bridge_path_m, total_bridge_frames=self.total_bridge_frames,
            anchored_error_accumulates=True)
        self.last_selection = self.last_selection | dict(status='MEASURED_INCREMENT_BRIDGE',
            anchor_status='NO_QUALIFIED_REFERENCE', selected_reference=self.previous.frame,
            selected_reference_retained_anchor=False)
        return incremental, False, True

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
            current = FeatureFrame(policy['image']['rgb'], depth)
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
            return dict(frame=self.frame, measured_ns=now, mode='gyro', reference_frame=ref,
                position_initial_body_m=p.tolist(), rotation_initial_body_from_current_body=R.tolist(),
                gyro_rotation_initial_body_from_current_body=G.tolist(),
                global_image_gyro_disagreement_rad=angle(G.T @ R), registration=reg,
                promoted_keyframe=reason is not None, promotion_reason=reason, keyframe_count=len(self.nodes),
                rgb_sha256=depth['rgb_sha256'], depth_sha256=hashlib.sha256(
                    depth['depth_m'].tobytes() + depth['valid'].tobytes()).hexdigest(),
                position_error_bound=None, orientation_error_bound=None, uncertainty_model_validated=False,
                gyro_role='rotation_estimator', native_pose_input=False, global_history_reset=False,
                navigation_qualified=False)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError, cv2.error, np.linalg.LinAlgError) as error:
            self.failed = True
            raise SensorContractError('temporal anchor RGBD pose unavailable; terminal failure') from error


class TemporalAnchorVisualLedMotion(MultiReferenceVisualLedMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.model = TemporalAnchorRGBDPose()

    def snapshot(self, *, now_ns):
        row = super().snapshot(now_ns=now_ns)
        return row | dict(continuity_evidence=deepcopy(self.model.last_continuity),
            continuity_evidence_current=row['current_pose'] is not None,
            bridge_is_command_or_inertial_extrapolation=False,
            bridge_is_calibrated_uncertainty=False, anchor_promotion_from_bridge=False)
