"""The observed round-trip controller with an explicit nominal reentry selector."""
from lewm.observed_round_trip_controller_development import ObservedRoundTripController, RoundTripMissionSelector
from lewm.nominal_clearance_reentry_development import reenter


class NominalReentrySelector(RoundTripMissionSelector):
    def choose(self, model, history, mapper, geometry, *, now_ns):
        selection = super().choose(model, history, mapper, geometry, now_ns=now_ns)
        B = mapper.map_from_initial
        return reenter(selection, B@mapper.surface.position, B@mapper.surface.rotation, mapper.occupied)


class NominalReentryRoundTripController(ObservedRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = NominalReentrySelector(residual=self.residual, condition=self.selector.condition,
            variant=self.selector.variant, goal_initial_body_xy_m=self.mission.target())

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='nominal_reentry_round_trip_controller_v1', nominal_reentry_policy_enabled=True)
