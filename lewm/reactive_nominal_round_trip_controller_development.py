"""Reactive mission allowing explicit unknown start connectors, without prediction."""
from lewm.reactive_observed_round_trip_controller_development import ReactiveObservedRoundTripController
from lewm.reactive_nominal_route_selection_development import ReactiveNominalRouteSelector


class ReactiveNominalRoundTripController(ReactiveObservedRoundTripController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selector = ReactiveNominalRouteSelector(goal_initial_body_xy_m=self.mission.target())

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='reactive_nominal_round_trip_controller_v1',
            unknown_start_connector_policy='explicitly_recorded_nominal_connector_without_floor_coverage_veto',
            unobserved_space_certified=False)
