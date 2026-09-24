"""Test low-overlap promotion separately from measurement acceptance."""
from lewm.orthonormal_gyro_visual_motion_development import (
    OrthonormalGyroPose, OrthonormalGyroVisualMotion)


class RetainedGyroReferencePose(OrthonormalGyroPose):
    def __init__(self, *, activation_frame=0):
        super().__init__()
        self.activation_frame = activation_frame

    def observe(self, *args, **kwargs):
        self.overlap_promotion_enabled = self.frame + 1 < self.activation_frame
        return super().observe(*args, **kwargs)


class RetainedGyroReferenceMotion(OrthonormalGyroVisualMotion):
    def __init__(self, *, identity=(0, 0, 0), activation_frame=0):
        super().__init__(identity=identity)
        self.model = RetainedGyroReferencePose(activation_frame=activation_frame)

    def snapshot(self, *, now_ns):
        return super().snapshot(now_ns=now_ns) | dict(
            overlap_only_reference_promotion_enabled=self.model.overlap_promotion_enabled,
            reference_promotion_treatment_activation_frame=self.model.activation_frame,
            accepted_support_and_motion_promotion_unchanged=True)
