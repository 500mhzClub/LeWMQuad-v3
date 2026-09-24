"""Round-trip navigation using execution-time intermediate waypoint utility."""
from lewm.nominal_reentry_round_trip_controller_development import (
    NominalReentryRoundTripController, NominalReentrySelector)
from lewm.executed_waypoint_score_development import score_waypoint_execution


class ExecutedWaypointSelector(NominalReentrySelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        selection = super().choose(model, history, mapper, geometry, now_ns=now_ns)
        receipt = self.residual.snapshot()
        if receipt['measured_ns'] != now_ns:
            raise ValueError('current observed residual state required')
        return score_waypoint_execution(selection, receipt)


class ExecutedWaypointRoundTripController(NominalReentryRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = ExecutedWaypointSelector(residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='executed_waypoint_round_trip_controller_v1', executed_waypoint_policy_enabled=True)
