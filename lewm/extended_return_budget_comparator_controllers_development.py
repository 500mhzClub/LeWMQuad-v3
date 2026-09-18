"""Source-only controls sharing the longer chained controller's perception.

Reactive selection remains a whole-method comparison. Nominal forecasts still
plan ahead and retain observed residual correction. No study or native runner
adopts these classes merely by importing them.
"""
from lewm.extended_return_budget_controller_development import (
    ExtendedReturnBudgetChainedController, ExtendedReturnBudgetSelector,
    current_measured_floor_pose, fork)
from lewm.forecast_source_residual_controller_development import (
    ForecastSourceEightStepSelector, require_source)
from lewm.reactive_floor_transport_controller_development import ReactiveFloorTransportController


class ExtendedReturnBudgetForecastSelector(ExtendedReturnBudgetSelector, ForecastSourceEightStepSelector):
    def __init__(self, *, forecast_source, **kwargs):
        self._forecast_source = require_source(forecast_source)
        super().__init__(**kwargs)

    @property
    def forecast_source(self):
        return self._forecast_source


class ExtendedReturnBudgetForecastSourceController(ExtendedReturnBudgetChainedController):
    def __init__(self, *args, forecast_source, **kwargs):
        source = require_source(forecast_source)
        super().__init__(*args, **kwargs)
        self.selector = ExtendedReturnBudgetForecastSelector(forecast_source=source,
            residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='extended_return_budget_forecast_source_controller_v1',
            assigned_forecast_source=self.selector.forecast_source,
            shared_observed_residual_correction_retained=True,
            fully_nonpredictive_controller=False)


class ExtendedReturnBudgetReactiveController(ReactiveFloorTransportController):
    def __init__(self, geometry, *, public_mission, navigation_ticks):
        # This fresh donor constructs no model and processes no observation.
        # Reuse the exact full composition so future perception comparisons
        # cannot silently substitute an older mapper, index or tracker.
        donor = ExtendedReturnBudgetChainedController(None, geometry,
            public_mission=public_mission, navigation_ticks=navigation_ticks,
            condition='direct', variant='no_rgb', persistent=True)
        super().__init__(geometry, public_mission=public_mission,
            navigation_ticks=min(navigation_ticks, 4000))
        for name in ('motion', 'registration', 'mapper', 'memory', 'mission'):
            setattr(self, name, getattr(donor, name))
        if self.memory is not self.mapper.surface or hasattr(self, 'model') or hasattr(self, 'residual'):
            raise ValueError('fresh shared perception without reactive model or residual required')

    advance = fork(ReactiveFloorTransportController.advance,
        current_measured_floor_pose=current_measured_floor_pose)

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='extended_return_budget_reactive_controller_v1',
            measured_plane_constrained_estimator=True,
            chained_anchor_reacquisition_enabled=True,
            direct_corner_flow_missingness_fallback_enabled=True,
            single_pass_measured_bound_queries_enabled=True,
            observation_local_body_projection_reuse_enabled=True,
            extended_return_budget_enabled=True,
            fully_nonpredictive_controller=True,
            reactive_is_whole_method_comparison=True)
