"""Current-observation routing with the longer controller's retained sensing.

This removes accumulated routing cells from selection, not all memory.
Tracking, contact evidence, forecast history, residuals and mission remain.
"""
from copy import deepcopy

from lewm.extended_return_budget_controller_development import (
    ExtendedReturnBudgetChainedController, ExtendedReturnBudgetFloorMap,
    ExtendedReturnBudgetSelector)
from lewm.current_observation_planning_map_development import CurrentObservationPlanningMap
from lewm.residual_current_observation_planning_controller_development import METADATA


class ExtendedReturnBudgetCurrentPlanningMap(ExtendedReturnBudgetFloorMap, CurrentObservationPlanningMap):
    """Retain extended geometry capture and the original checked current view."""


class ExtendedReturnBudgetCurrentPlanningSelector(ExtendedReturnBudgetSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        view = mapper.planning_view(now_ns=now_ns)
        result = super().choose(model, history, view, geometry, now_ns=now_ns)
        return result | dict(planning_map_receipt=deepcopy(view.receipt))


class ExtendedReturnBudgetCurrentPlanningController(ExtendedReturnBudgetChainedController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        original = self.mapper
        if (type(original) is not ExtendedReturnBudgetFloorMap or self.memory is not original.surface
                or self.tick != -1 or self.memory.route or original.failed or original.frame_geometry is not None):
            raise ValueError('fresh exact extended mapper and memory alias required')
        revised = object.__new__(ExtendedReturnBudgetCurrentPlanningMap)
        revised.__dict__ = vars(original).copy()
        revised.current_planning_view = None
        self.mapper = revised
        self.selector = ExtendedReturnBudgetCurrentPlanningSelector(residual=self.residual,
            condition=self.selector.condition, variant=self.selector.variant,
            goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='extended_return_budget_current_planning_controller_v1', **METADATA)
