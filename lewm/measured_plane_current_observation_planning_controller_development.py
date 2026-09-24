"""Planning-cell memory comparator with the same measured-plane perception.

Only routing cells are restricted to the current paired observation. Contact,
tracking, floor anchors, model history, residuals and mission state remain.
"""
from lewm.residual_current_observation_planning_controller_development import ResidualCurrentObservationPlanningController
from lewm.measured_plane_visual_motion_development import MeasuredPlaneVisualMotion


class MeasuredPlaneCurrentObservationPlanningController(ResidualCurrentObservationPlanningController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = MeasuredPlaneVisualMotion(identity=(0, 0, 0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='measured_plane_current_pair_planning_controller_v1',
            measured_plane_constrained_estimator=True)
