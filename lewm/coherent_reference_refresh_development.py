"""Refresh accepted reference imagery while retaining coherent floor fitting."""
from lewm.gyro_coherent_floor_tracking_development import (
    GyroCoherentFloorPose, GyroCoherentFloorMotion)
from lewm.recent_anchored_reference_refresh_development import (
    RecentAnchoredReferenceRefreshMixin, MAX_RECENT_REFERENCE_AGE_NS)


class CoherentReferenceRefreshPose(RecentAnchoredReferenceRefreshMixin, GyroCoherentFloorPose):
    pass


class CoherentReferenceRefreshMotion(GyroCoherentFloorMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = CoherentReferenceRefreshPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            recent_reference_refresh_from_accepted_anchor=True,
            maximum_recent_reference_age_ns=MAX_RECENT_REFERENCE_AGE_NS,
            bridge_measurements_promoted=False,
            bridge_budget_and_measurement_acceptance_rules_unchanged=True)
