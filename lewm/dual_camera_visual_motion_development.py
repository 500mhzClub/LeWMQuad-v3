"""Explicit dual-camera motion evidence consumed through existing joint witnesses."""
from copy import deepcopy
import numpy as np
from lewm.causal_sensor_state import SensorContractError, _identity, _ns
from lewm.causal_auxiliary_rgb_observation_development import validate_rgb, depth_digest
from lewm.dual_camera_anchor_pose_development import DualCameraAnchorPose
from lewm.visual_led_motion_development import VisualLedMotion, POSE_FIELDS
from lewm.joint_sensor_anchored_goal_development import current_supported_rgbd_pose


def current_dual_camera_pose(evidence, policy, image, depth, *, identity, now_ns):
    validate_rgb(image, depth, policy, now_ns=now_ns)
    p, R, pose = current_supported_rgbd_pose(evidence, identity=identity, now_ns=now_ns)
    choice = evidence['camera_selection']
    if (evidence['observer_variant'] not in ('front_first_dual_camera_anchor_v1', 'gyro_consensus_dual_camera_v1')
            or evidence['camera_selection_current'] is not True
            or pose['auxiliary_rgb_sha256'] != image['rgb_sha256']
            or pose['auxiliary_depth_sha256'] != depth_digest(depth)
            or evidence['calibration_ids']['auxiliary_rgb'] != image['calibration_id']
            or evidence['calibration_ids']['auxiliary_depth'] != depth['calibration_id']):
        raise SensorContractError('current explicitly bound dual-camera pose required')
    if pose['frame'] == 0:
        if choice.get('selected_camera') is not None or choice.get('initial_paired_reference') is not True:
            raise SensorContractError('single initial paired-camera reference required')
    else:
        camera = choice.get('selected_camera')
        joint_enabled = evidence.get('joint_camera_retained_anchor_fallback') is True
        if (camera not in ('primary', 'auxiliary', 'joint')
                or (camera == 'joint' and (not joint_enabled or choice.get('joint_camera_measurement',
                    choice.get('joint_camera_retained_anchor')) is not True))
                or choice['auxiliary_attempted'] != (camera in ('auxiliary', 'joint'))):
            raise SensorContractError('explicit selected current measurement camera required')
        if camera == 'auxiliary':
            if choice['primary_continuity']['status'] not in (
                    'NO_CURRENT_MEASURED_TRANSLATION', 'MEASURED_BRIDGE_BUDGET_EXHAUSTED'):
                raise SensorContractError('auxiliary use requires preserved primary missingness')
        for witness in evidence['continuity_evidence']['rotation_measurement_witnesses']:
            witness_camera = witness.get('camera', 'primary')
            if joint_enabled and witness_camera == 'joint':
                counts = witness.get('camera_inliers', [])
                if (len(counts) != 2 or min(counts) < 3 or sum(counts) != witness['inliers']
                        or witness.get('camera_specific_reprojection') is not True):
                    raise SensorContractError('joint-camera witness requires measured support from both views')
            elif witness_camera != (choice.get('search_camera') if camera == 'joint' else camera):
                raise SensorContractError('rotation witness must name the selected camera')
    return p, R, pose


class DualCameraVisualMotion(VisualLedMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__('joint', identity=identity)
        self.model = DualCameraAnchorPose()

    def observe(self, policy, depth, fast, *, auxiliary_rgb, auxiliary_depth, now_ns):
        now = _ns(now_ns, 'dual-camera motion decision')
        if self.last_decision_ns is not None and now < self.last_decision_ns:
            raise SensorContractError('motion decision cannot move backwards')
        self.last_decision_ns = now
        if self.failure is not None:
            return self.snapshot(now_ns=now)
        try:
            if _identity(policy['sensor_state']['identity']) != self.identity:
                raise SensorContractError('motion episode mismatch')
            state = self.model.observe(policy, depth, fast, auxiliary_rgb=auxiliary_rgb,
                auxiliary_depth=auxiliary_depth, now_ns=now)
            metadata = dict(rgb=policy['image']['calibration_id'], depth=depth['calibration_id'],
                fast_gyro=fast['calibration_id'], body=policy['sensor_state']['sensed']['gyro']['calibration_id'],
                auxiliary_rgb=auxiliary_rgb['calibration_id'], auxiliary_depth=auxiliary_depth['calibration_id'])
            if self.calibrations is not None and metadata != self.calibrations:
                raise SensorContractError('motion calibration changed within episode')
            self.last_visual = {k: deepcopy(state[k]) for k in (*POSE_FIELDS,
                'auxiliary_rgb_sha256', 'auxiliary_depth_sha256')}
            self.last_visual.update(available_ns=now, processing_latency_accounted=False,
                acquisition_available_ns=max(policy['image']['available_ns'], depth['available_ns'],
                    int(np.max(fast['available_ns'])), auxiliary_rgb['available_ns'], auxiliary_depth['available_ns']))
            self.calibrations = metadata
            current_dual_camera_pose(self.snapshot(now_ns=now), policy, auxiliary_rgb, auxiliary_depth,
                identity=self.identity, now_ns=now)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            chain = []; cause = error
            while cause is not None:
                chain.append(str(cause)); cause = cause.__cause__
            self.failure = dict(decision_ns=now, chain=chain)
            self.contact = dict(status='VISUAL_TERMINAL_NO_COMPARISON', displacement_disagreement_m=None)
        return self.snapshot(now_ns=now)

    def snapshot(self, *, now_ns):
        row = super().snapshot(now_ns=now_ns)
        previous = self.model.previous
        return row | dict(reference_selection=deepcopy(self.model.last_selection),
            continuity_evidence=deepcopy(self.model.last_continuity),
            continuity_evidence_current=row['current_pose'] is not None,
            bridge_is_command_or_inertial_extrapolation=False,
            bridge_is_calibrated_uncertainty=False, anchor_promotion_from_bridge=False,
            gyro_bias_estimated=False, candidate_adopted=False,
            feature_selection='corner_upright_sift_support_v1',
            last_accepted_feature_witness=None if previous is None else previous.features['primary'].witness(),
            auxiliary_feature_witness=None if previous is None else previous.features['auxiliary'].witness(),
            overlap_retention=deepcopy(self.model.last_overlap_retention),
            observer_variant='front_first_dual_camera_anchor_v1',
            camera_selection=deepcopy(self.model.last_camera_selection),
            camera_selection_current=row['current_pose'] is not None,
            auxiliary_rgb_is_additional_modality=True)
