"""Keep every tracking frame, but probe old local views every four frames."""
from lewm.cached_moments_deferred_copy_tracking_development import (
    CachedMomentsDeferredCopyPose, CachedMomentsDeferredCopyMotion)
from lewm.local_view_revisit_tracking_development import LocalViewRevisitPose

REVISIT_PERIOD_FRAMES = 4


class CadencedViewRevisitPose(CachedMomentsDeferredCopyPose):
    def _measure(self, current, gyro, now):
        if self.frame % REVISIT_PERIOD_FRAMES == 0:
            return super()._measure(current, gyro, now)
        # Continue the original recent/stable-reference measurement and all
        # acceptance checks. Only the optional old-view bank probe is skipped.
        return super(LocalViewRevisitPose, self)._measure(current, gyro, now)


class CadencedViewRevisitMotion(CachedMomentsDeferredCopyMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity, activation_frame=activation_frame)
        self.model = CadencedViewRevisitPose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            old_view_revisit_period_frames=REVISIT_PERIOD_FRAMES,
            every_camera_frame_tracked=True,
            original_reference_acceptance_limits_unchanged=True)
