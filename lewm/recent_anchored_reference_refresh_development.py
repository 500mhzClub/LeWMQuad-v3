"""Refresh recent reference imagery only from an already accepted anchored pose."""
from lewm.stable_gyro_reference_development import (
    CompiledFloorStableGyroReferenceMotion, CompiledFloorStableGyroReferencePose)

MAX_RECENT_REFERENCE_AGE_NS = 400_000_000  # Existing navigation planning period.


class RecentAnchoredReferenceRefreshMixin:
    def _measure(self, *args, **kwargs):
        result = super()._measure(*args, **kwargs)
        self._refresh_anchored_measurement = not result[2]
        return result

    def _refresh_accepted_reference(self, row):
        if (not self._refresh_anchored_measurement or row['promoted_keyframe']
                or not self.references):
            return row
        previous = self.previous
        newest = max(self.references, key=lambda r:r.measured_ns)
        if previous.measured_ns-newest.measured_ns < MAX_RECENT_REFERENCE_AGE_NS:
            return row
        if previous.frame != row['frame'] or previous.measured_ns != row['measured_ns']:
            raise ValueError('refresh requires the just-accepted measured pose')
        # Reuse the accepted image/gyro pose and existing stable-anchor retention.
        # The original _remember also binds this frame's measured floor evidence.
        self._remember(previous.features, previous.rotation, previous.gyro,
            previous.position, previous.measured_ns)
        self.nodes.append(dict(frame=row['frame'], measured_ns=row['measured_ns'],
            parent_frame=row['reference_frame'],
            position_initial_body_m=row['position_initial_body_m'],
            rotation_initial_body_from_current_body=row['rotation_initial_body_from_current_body'],
            pose_error_bound=None))
        return row | dict(promoted_keyframe=True,
            promotion_reason='accepted_anchor_recent_reference_age',
            keyframe_count=len(self.nodes))

    def observe(self, *args, **kwargs):
        self._refresh_anchored_measurement = False
        row = super().observe(*args, **kwargs)
        return self._refresh_accepted_reference(row)


class RecentAnchoredReferenceRefreshPose(RecentAnchoredReferenceRefreshMixin,
        CompiledFloorStableGyroReferencePose):
    pass


class RecentAnchoredReferenceRefreshMotion(CompiledFloorStableGyroReferenceMotion):
    def __init__(self, *, identity=(0,0,0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = RecentAnchoredReferenceRefreshPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            recent_reference_refresh_from_accepted_anchor=True,
            maximum_recent_reference_age_ns=MAX_RECENT_REFERENCE_AGE_NS,
            recent_reference_age_limit_applies_only_to_anchored_measurements=True,
            bridge_measurements_promoted=False,
            bridge_budget_and_measurement_acceptance_rules_unchanged=True)
