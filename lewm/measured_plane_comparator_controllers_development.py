"""Comparators sharing the measured-plane estimator with the learned controller.

The forecast-source comparison retains predictive planning and observed
residual correction. The reactive comparison replaces the whole action-selection
method and does not isolate predictive ranking. Neither class establishes
matched physical execution, checkpoint admission or navigation success.
"""
from lewm.forecast_source_residual_controller_development import ForecastSourceResidualController
from lewm.reactive_floor_transport_controller_development import ReactiveFloorTransportController
from lewm.measured_plane_visual_motion_development import MeasuredPlaneVisualMotion


class MeasuredPlaneForecastSourceController(ForecastSourceResidualController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = MeasuredPlaneVisualMotion(identity=(0, 0, 0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='measured_plane_forecast_source_controller_v1',
            measured_plane_constrained_estimator=True)


class MeasuredPlaneReactiveController(ReactiveFloorTransportController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.motion = MeasuredPlaneVisualMotion(identity=(0, 0, 0))

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='measured_plane_reactive_controller_v1',
            measured_plane_constrained_estimator=True,
            fully_nonpredictive_controller=True,
            reactive_is_whole_method_comparison=True)
