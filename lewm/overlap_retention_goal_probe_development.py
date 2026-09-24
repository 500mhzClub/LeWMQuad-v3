"""Unchanged retained-patch mission with qualified-overlap visual retention."""
from lewm.retained_patch_contact_goal_probe_development import RetainedPatchContactGoalProbe
from lewm.overlap_retention_joint_observer_development import OverlapRetentionVisualLedMotion


class OverlapRetentionGoalProbe(RetainedPatchContactGoalProbe):
    def __init__(self, model, geometry, *, condition, variant, persistent):
        super().__init__(model, geometry, condition=condition, variant=variant, persistent=persistent)
        self.motion = OverlapRetentionVisualLedMotion(identity=(0, 0, 0))

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(controller='overlap_retention_goal_probe_v1')
