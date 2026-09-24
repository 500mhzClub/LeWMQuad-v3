"""Separate joint-RGB-D continuity candidate, not an adopted motion observer.

Uses the existing joint rigid fit for BOTH retained and previous-frame pairs.
Matching, consensus, reference retention, bridge budget, disagreement envelopes
and terminal failures are inherited unchanged. Gyro still gates consistency;
this estimates visual rotation, not gyro bias or calibrated uncertainty.
"""
from copy import deepcopy
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.keyframe_rgbd_pose_development import matched_points
from lewm.joint_rgbd_rigid_pose_development import register, angle, RIGID_RULES
from lewm.temporal_anchor_continuity_development import TemporalAnchorRGBDPose
from lewm.visual_led_motion_development import VisualLedMotion


class JointTemporalAnchorRGBDPose(TemporalAnchorRGBDPose):
    def __init__(self):
        super().__init__()
        self.mode = 'joint'
        self.rotation_measurements = []

    def _candidate(self, ref, current, G):
        a, b, ua, ub = matched_points(ref.features, current)
        relative_gyro = ref.gyro.T @ G
        local_R, t, mask, reg = register(a, b, ua, ub,
            gyro_rotation=relative_gyro, mode='joint', frame=self.frame)
        R = ref.rotation @ local_R
        p = ref.position + ref.rotation @ t
        # Same envelope as the frozen gyro candidate, applied to the actual
        # joint fit before it can count as a qualified anchor or increment.
        if (np.linalg.norm(p - self.last_p) > RIGID_RULES['maximum_increment_translation_m']
                or angle(self.last_R.T @ R) > RIGID_RULES['maximum_increment_rotation_rad']):
            raise SensorContractError('consecutive rigid-pose displacement envelope rejected')
        reg |= dict(reference_frame=ref.frame, reference_measured_ns=ref.measured_ns,
            translation_reference_body_m=t.tolist(), relative_rotation=local_R.tolist(),
            gyro_relative_rotation=relative_gyro.tolist(),
            reference_inlier_pixels=ua[mask].tolist(), current_inlier_pixels=ub[mask].tolist(),
            reference_inlier_points_body_m=a[mask].tolist(), current_inlier_points_body_m=b[mask].tolist())
        if len(self.rotation_measurements) >= 9:
            raise SensorContractError('bounded eight anchors plus one increment required')
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
            candidate_envelope_passed=True, witness_alone_grants_pose=False))
        return dict(reference=ref, R=R, p=p, local_R=local_R, t=t, registration=reg)

    def _measure(self, current, G, now):
        self.rotation_measurements = []
        try:
            return super()._measure(current, G, now)
        finally:
            # Preserve qualified but conflicting hypotheses even if the inherited
            # continuity selector rejects the current frame. Never invent a pose.
            if isinstance(self.last_continuity, dict):
                previous = next((w for w in self.rotation_measurements
                    if w['reference_frame'] == self.previous.frame), None)
                selected = (self.last_selection or {}).get('selected_reference')
                anchor = next((w for w in self.rotation_measurements
                    if w['reference_frame'] == selected), None)
                self.last_continuity.update(
                    rotation_measurement_witnesses=deepcopy(self.rotation_measurements),
                    incremental_rotation_witness=deepcopy(previous)
                        if self.last_continuity.get('incremental_available') else None,
                    selected_anchor_rotation_witness=deepcopy(anchor)
                        if self.last_continuity.get('anchor_available') else None,
                    rotation_fitting_mode='joint', gyro_role='consistency_monitor_only',
                    incremental_rotation_witness_saved=bool(previous is not None
                        and self.last_continuity.get('incremental_available')),
                    gyro_bias_estimated=False,
                    independent_sensor_acquisition=False)

    def observe(self, policy, depth, fast, *, now_ns):
        result = super().observe(policy, depth, fast, now_ns=now_ns)
        # The inherited numerical state is already joint: override its frozen
        # descriptive labels, not its calculations or acceptance decisions.
        return result | dict(mode='joint', gyro_role='consistency_monitor_only')


class JointTemporalAnchorVisualLedMotion(VisualLedMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__('joint', identity=identity)
        self.model = JointTemporalAnchorRGBDPose()

    def snapshot(self, *, now_ns):
        row = super().snapshot(now_ns=now_ns)
        return row | dict(reference_selection=deepcopy(self.model.last_selection),
            continuity_evidence=deepcopy(self.model.last_continuity),
            continuity_evidence_current=row['current_pose'] is not None,
            bridge_is_command_or_inertial_extrapolation=False,
            bridge_is_calibrated_uncertainty=False, anchor_promotion_from_bridge=False,
            gyro_bias_estimated=False, candidate_adopted=False)
