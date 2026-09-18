"""Apply the same observed stopping rule to every longer-budget comparator."""
from functools import partial

from lewm.stop_conditioned_settling_development import StopConditionedSettlingMission
from lewm.extended_return_budget_comparator_controllers_development import (
    ExtendedReturnBudgetForecastSourceController, ExtendedReturnBudgetReactiveController)
from lewm.extended_return_budget_current_planning_development import ExtendedReturnBudgetCurrentPlanningController


class StopConditionedMissionMixin:
    def __init__(self, *args, public_mission, navigation_ticks, **kwargs):
        super().__init__(*args, public_mission=public_mission,
            navigation_ticks=navigation_ticks, **kwargs)
        self.mission = StopConditionedSettlingMission(public_mission,
            navigation_ticks=navigation_ticks)

    def _result(self, *args, **kwargs):
        result = super()._result(*args, **kwargs)
        return result | dict(controller=result['controller']+'_stop_conditioned',
            zero_request_boundary_required_before_dwell=True)


class StopConditionedForecastController(StopConditionedMissionMixin, ExtendedReturnBudgetForecastSourceController):
    pass


class StopConditionedReactiveController(StopConditionedMissionMixin, ExtendedReturnBudgetReactiveController):
    pass


class StopConditionedCurrentPlanningController(StopConditionedMissionMixin, ExtendedReturnBudgetCurrentPlanningController):
    pass


CONTROLLERS = dict(
    frozen_reference=partial(StopConditionedForecastController, forecast_source='frozen_world_model'),
    nominal=partial(StopConditionedForecastController, forecast_source='nominal_requested_twist'),
    reactive=StopConditionedReactiveController,
    current_planning=StopConditionedCurrentPlanningController)
