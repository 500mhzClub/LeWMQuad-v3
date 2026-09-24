"""Ablate spatial planning persistence while preserving other temporal evidence."""
from copy import deepcopy
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportController
from lewm.current_observation_planning_map_development import CurrentObservationPlanningMap
from lewm.view_reentry_round_trip_controller_development import ViewReentrySelector


class CurrentObservationPlanningSelector(ViewReentrySelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        view = mapper.planning_view(now_ns=now_ns)
        selection = super().choose(model, history, view, geometry, now_ns=now_ns)
        return selection | dict(planning_map_receipt=deepcopy(view.receipt))


class CurrentObservationPlanningController(MeasuredFloorTransportController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mapper = CurrentObservationPlanningMap(identity=(0,0,0))
        self.memory = self.mapper.surface
        self.selector = CurrentObservationPlanningSelector(residual=self.residual,
            condition=self.selector.condition, variant=self.selector.variant,
            goal_initial_body_xy_m=self.mission.target())

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='current_observation_planning_round_trip_controller_v1',
            planning_map_variant='current_paired_observation',
            accumulated_planning_cells_queried=False, selector_scan_state_retained=True,
            persistent_contact_history_retained=True,
            tracking_and_floor_anchor_history_retained=True,
            learned_temporal_history_and_residual_retained=True,
            mission_and_settling_state_retained=True, memoryless_controller=False)
