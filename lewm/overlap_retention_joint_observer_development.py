"""Retain already anchor-qualified views at half remaining feature overlap."""
from copy import deepcopy
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.corner_support_joint_observer_development import (
    CornerSupportJointRGBDPose, CornerSupportVisualLedMotion)


def overlap_retention(*, inliers, reference_features, anchored, already_promoted):
    if (type(inliers) is not int or type(reference_features) is not int
            or not 0 < inliers <= reference_features <= 600
            or type(anchored) is not bool or type(already_promoted) is not bool):
        raise SensorContractError('bounded accepted feature population required')
    return dict(inliers=inliers, reference_features=reference_features,
        remaining_fraction=inliers/reference_features, threshold_fraction=.5,
        retain=anchored and not already_promoted and 2*inliers <= reference_features,
        already_anchor_qualified=anchored, acceptance_thresholds_changed=False,
        calibrated_uncertainty=False)


class OverlapRetentionJointRGBDPose(CornerSupportJointRGBDPose):
    def __init__(self):
        super().__init__(); self.last_overlap_retention = None

    def observe(self, policy, depth, fast, *, now_ns):
        self.last_overlap_retention = None
        references = {r.frame: r for r in self.references}
        row = super().observe(policy, depth, fast, now_ns=now_ns)
        if row['registration'] is None:
            return row
        # Bridge references need not be retained anchors and cannot be promoted.
        if self.last_continuity['status'] != 'ANCHOR_MEASUREMENT':
            return row
        try:
            ref = references[row['reference_frame']]
            receipt = overlap_retention(inliers=row['registration']['inliers'],
                reference_features=len(ref.features.keypoints), anchored=True,
                already_promoted=row['promoted_keyframe'])
            self.last_overlap_retention = receipt | dict(frame=self.frame,
                reference_frame=ref.frame, measured_ns=now_ns)
            if receipt['retain']:
                R = np.asarray(row['rotation_initial_body_from_current_body'])
                G = np.asarray(row['gyro_rotation_initial_body_from_current_body'])
                p = np.asarray(row['position_initial_body_m'])
                self.nodes.append(dict(frame=self.frame, measured_ns=now_ns,
                    parent_frame=ref.frame, position_initial_body_m=p.tolist(),
                    rotation_initial_body_from_current_body=R.tolist(), pose_error_bound=None))
                self._remember(self.previous.features, R, G, p, now_ns)
                row.update(promoted_keyframe=True, promotion_reason='accepted_half_feature_overlap',
                    keyframe_count=len(self.nodes))
            return row
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            raise SensorContractError('overlap retention failed; terminal') from error


class OverlapRetentionVisualLedMotion(CornerSupportVisualLedMotion):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity); self.model = OverlapRetentionJointRGBDPose()

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            overlap_retention=deepcopy(self.model.last_overlap_retention),
            observer_variant='accepted_half_feature_overlap_v1')
