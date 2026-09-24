"""Separate hold-reorientation successor; frozen native controllers unchanged."""
import numpy as np
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationSelector, ResidualAnchoredContinuationController)
from lewm.hold_reorientation_development import HoldReorientation


class HoldReorientationSelector(ResidualAnchoredContinuationSelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hold_reorientation = HoldReorientation()

    def set_goal(self, goal_initial_body_xy_m):
        previous = self.goal.copy()
        super().set_goal(goal_initial_body_xy_m)
        if not np.array_equal(previous, self.goal):
            self.hold_reorientation.reset_goal()

    def choose(self, model, history, mapper, geometry, *, now_ns):
        original = super().choose(model, history, mapper, geometry, now_ns=now_ns)
        if (mapper.failed or mapper.surface.failed or mapper.surface.last_ns != now_ns
                or len(mapper.surface.route) - 1 != self.residual.frame):
            raise ValueError('same admitted current observed map and residual frame required')
        return self.hold_reorientation.reconsider(original, frame=self.residual.frame, now_ns=now_ns)


class HoldReorientationController(ResidualAnchoredContinuationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = HoldReorientationSelector(residual=self.residual,
            condition=self.selector.condition, variant=self.selector.variant,
            goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='hold_reorientation_controller_v1', hold_reorientation_enabled=True)
