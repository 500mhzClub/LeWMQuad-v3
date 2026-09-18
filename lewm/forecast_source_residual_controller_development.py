"""Shared residual planner with one declared forecast-source intervention.

This compares learned forecasts with nominal requested-twist forecasts. The
nominal mode still plans ahead and retains the observed residual correction;
it must not be described as fully nonpredictive or fully memoryless.
No existing controller, source global or independent study is modified.
"""
from functools import partial
from types import FunctionType

from lewm.forecast_source_selection_development import select, require_source
from lewm.mission_target_waypoint_selection_development import MissionTargetWaypointSelector
from lewm.mission_target_eight_step_selection_development import MissionTargetEightStepSelector
from lewm.observation_horizon_waypoint_selection_development import restrict
from lewm.observation_horizon_waypoint_utility_development import score_commitment_pose
from lewm.observation_horizon_nominal_constraint_development import constrain
from lewm.eight_step_planning_development import plan
from lewm.geometry_progress_pilot_development import ACTIONS
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationController, ResidualAnchoredContinuationSelector)


class ForecastSourceWaypointSelector(MissionTargetWaypointSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        original = MissionTargetWaypointSelector.choose
        provider = partial(select, forecast_source=self.forecast_source)
        if original.__closure__ is not None:
            raise ValueError('closure-free original waypoint method required')
        function = FunctionType(original.__code__, original.__globals__ | {'select': provider},
            original.__name__, original.__defaults__)
        function.__kwdefaults__ = original.__kwdefaults__
        return function(self, model, history, mapper, geometry, now_ns=now_ns)


class ForecastSourceEightStepSelector(MissionTargetEightStepSelector, ForecastSourceWaypointSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        # Continue immediately below the original eight-step layer, entering
        # the private forecast-provider waypoint method through cooperative MRO.
        result = super(MissionTargetEightStepSelector, self).choose(
            model, history, mapper, geometry, now_ns=now_ns)
        if 'prediction' not in result:
            return result
        flags = {key: result[key] for key in ('model_prediction_corrected', 'translation_bias_training_only')}
        if result['mode'] == 'WAYPOINT':
            result = restrict(score_commitment_pose(result), ACTIONS)
            result['surface_conflict_filter_still_required'] = False
            result['original_surface_conflicts_preserved'] = True
        B = mapper.map_from_initial
        p = B@mapper.surface.position
        R = B@mapper.surface.rotation
        result = constrain(result, p, R, mapper.occupied)
        provenance = result['forecast_provenance']
        if provenance['forecast_source'] != self.forecast_source:
            raise ValueError('assigned forecast source must survive the planning layers')
        # Original scoring/plan functions reset this historical flag. Restore the actual
        # provider's values rather than declaring every prediction corrected.
        return plan(result, p, R, mapper.occupied) | flags


class ForecastSourceResidualSelector(ResidualAnchoredContinuationSelector, ForecastSourceEightStepSelector):
    def __init__(self, *, forecast_source, **kwargs):
        self._forecast_source = require_source(forecast_source)
        super().__init__(**kwargs)

    @property
    def forecast_source(self):
        return self._forecast_source


class ForecastSourceResidualController(ResidualAnchoredContinuationController):
    def __init__(self, *args, forecast_source, **kwargs):
        source = require_source(forecast_source)
        super().__init__(*args, **kwargs)
        self.selector = ForecastSourceResidualSelector(forecast_source=source,
            residual=self.residual, condition=self.selector.condition, variant=self.selector.variant,
            goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='forecast_source_residual_anchored_controller_v1',
            assigned_forecast_source=self.selector.forecast_source,
            shared_observed_residual_correction_retained=True,
            fully_nonpredictive_controller=False)
