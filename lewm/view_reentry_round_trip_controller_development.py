"""Execution-time waypoint navigation with explicit translating view recovery."""
from lewm.observed_round_trip_controller_development import RoundTripMissionSelector
from lewm.executed_waypoint_round_trip_controller_development import ExecutedWaypointRoundTripController
from lewm.executed_waypoint_score_development import score_waypoint_execution
from lewm.view_reentry_selection_development import reenter_with_translation


class ViewReentrySelector(RoundTripMissionSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        selection = super().choose(model, history, mapper, geometry, now_ns=now_ns)
        B = mapper.map_from_initial
        selection = reenter_with_translation(selection, B@mapper.surface.position,
            B@mapper.surface.rotation, mapper.occupied)
        receipt = self.residual.snapshot()
        if receipt['measured_ns'] != now_ns:
            raise ValueError('current observed residual state required')
        return score_waypoint_execution(selection, receipt)


class ViewReentryRoundTripController(ExecutedWaypointRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = ViewReentrySelector(residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='view_reentry_round_trip_controller_v1', view_reentry_translation_policy_enabled=True)
