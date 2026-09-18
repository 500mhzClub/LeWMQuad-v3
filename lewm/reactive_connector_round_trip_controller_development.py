"""Reactive round-trip control with measured-connector waypoint fallback."""
from lewm.reactive_nominal_route_selection_development import ReactiveNominalRouteSelector
from lewm.reactive_nominal_round_trip_controller_development import ReactiveNominalRoundTripController
from lewm.reactive_route_connector_development import nearer_route_target


class ReactiveConnectorRouteSelector(ReactiveNominalRouteSelector):
    def choose(self, mapper, geometry, *, now_ns):
        selection = super().choose(mapper, geometry, now_ns=now_ns)
        B = mapper.map_from_initial
        return nearer_route_target(selection, B@mapper.surface.position,
            B@mapper.surface.rotation, mapper.floor, mapper.occupied)


class ReactiveConnectorRoundTripController(ReactiveNominalRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = ReactiveConnectorRouteSelector(goal_initial_body_xy_m=self.mission.target())

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='reactive_connector_round_trip_controller_v1', nearer_observed_route_target_policy_enabled=True)
