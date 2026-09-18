"""Separate prospective controller for a bounded, observed turn recovery."""
from lewm.hold_reorientation_controller_development import HoldReorientationSelector, HoldReorientationController
from lewm.residual_anchored_continuation_controller_development import ResidualAnchoredContinuationSelector
from lewm.sustained_hold_reorientation_development import SustainedHoldReorientation


class SustainedHoldReorientationSelector(HoldReorientationSelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hold_reorientation = SustainedHoldReorientation()

    def choose(self, model, history, mapper, geometry, *, now_ns):
        original = ResidualAnchoredContinuationSelector.choose(
            self, model, history, mapper, geometry, now_ns=now_ns)
        if (mapper.failed or mapper.surface.failed or mapper.surface.last_ns != now_ns
                or len(mapper.surface.route)-1 != self.residual.frame):
            raise ValueError('same admitted current observed map and residual frame required')
        current = (mapper.map_from_initial@mapper.surface.rotation)[:2,0]
        return self.hold_reorientation.reconsider(original, frame=self.residual.frame,
            now_ns=now_ns, observed_heading_map=current)


class SustainedHoldReorientationController(HoldReorientationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = SustainedHoldReorientationSelector(residual=self.residual,
            condition=self.selector.condition, variant=self.selector.variant,
            goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='sustained_hold_reorientation_controller_v1',
            sustained_hold_reorientation_enabled=True)
