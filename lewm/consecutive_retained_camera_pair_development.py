"""Allow consecutive pooled fitting for an identical retained acquisition.

A retained Reference and `previous` can be distinct objects describing exactly
the same already anchored frame. Use the existing consecutive-camera fitting
path only after checking that equivalence, then restore the retained identity.
No new pose is promoted from a bridge and no measurement threshold changes.
"""
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.local_inverse_depth_floor_tracking_development import (
    LocalInverseDepthFloorPose, LocalInverseDepthFloorMotion)


class ConsecutiveRetainedCameraPairMixin:
    def _candidate(self, ref, current, G):
        previous = self.previous
        eligible = (previous is not None and ref is not previous
            and any(ref is r for r in self.references)
            and ref.frame == previous.frame == self.frame-1)
        if not eligible:
            return super()._candidate(ref, current, G)
        if (ref.measured_ns != previous.measured_ns or ref.features is not previous.features
                or any(not np.array_equal(getattr(ref, field), getattr(previous, field))
                    for field in ('position', 'rotation', 'gyro'))):
            raise SensorContractError('retained consecutive acquisition differs from previous measurement')
        candidate = super()._candidate(previous, current, G)
        # Every fit and subsequent refinement used the identical sensor data,
        # pose and clock. Restore actual retained ownership for the selector.
        candidate['reference'] = ref
        reg = candidate['registration']
        if reg.get('joint_camera_measurement'):
            reg.update(joint_camera_retained_anchor=True,
                identical_consecutive_retained_acquisition=True,
                retained_and_previous_features_same_object=True,
                retained_and_previous_pose_exactly_equal=True)
        return candidate

    def observe(self, *args, **kwargs):
        self.consecutive_retained_pooled_fit_selected = False
        row = super().observe(*args, **kwargs)
        self.consecutive_retained_pooled_fit_selected = (
            (row.get('registration') or {}).get('identical_consecutive_retained_acquisition') is True)
        return row


class ConsecutiveRetainedCameraPairPose(ConsecutiveRetainedCameraPairMixin,
        LocalInverseDepthFloorPose):
    pass


class ConsecutiveRetainedCameraPairMotion(LocalInverseDepthFloorMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = ConsecutiveRetainedCameraPairPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            consecutive_retained_camera_pair_enabled=True,
            consecutive_retained_pooled_fit_selected=getattr(self.model,
                'consecutive_retained_pooled_fit_selected', False),
            consecutive_retained_acquisition_requires_exact_equivalence=True,
            bridge_allowance_changed=False, bridge_promotions_enabled=False)
