"""Current paired planning cells through the complete residual selector chain.

Reuse the checked observation-map implementation. Contact, localization,
prediction history and mission state retain their original persistent scope.
"""
from copy import deepcopy
from lewm.current_observation_planning_map_development import CurrentObservationPlanningMap
from lewm.residual_anchored_continuation_controller_development import (
    ResidualAnchoredContinuationController, ResidualAnchoredContinuationSelector)

METADATA = dict(planning_map_variant='current_paired_observation',
    accumulated_planning_cells_queried=False, selector_scan_state_retained=True,
    persistent_contact_history_retained=True, tracking_and_floor_anchor_history_retained=True,
    learned_temporal_history_and_residual_retained=True,
    mission_and_settling_state_retained=True, memoryless_controller=False)


class ResidualCurrentObservationPlanningSelector(ResidualAnchoredContinuationSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        view = mapper.planning_view(now_ns=now_ns)
        selection = super().choose(model, history, view, geometry, now_ns=now_ns)
        return selection | dict(planning_map_receipt=deepcopy(view.receipt))


class ResidualCurrentObservationPlanningController(ResidualAnchoredContinuationController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = CurrentObservationPlanningMap(identity=(0, 0, 0))
        self.memory = self.mapper.surface
        self.selector = ResidualCurrentObservationPlanningSelector(residual=self.residual,
            condition=self.selector.condition, variant=self.selector.variant,
            goal_initial_body_xy_m=self.mission.target())

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='residual_current_observation_planning_controller_v1', **METADATA)
