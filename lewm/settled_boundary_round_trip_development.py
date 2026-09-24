"""Start observed dwell only after a measured low-motion boundary exists."""
from copy import deepcopy
from lewm.measured_settling_round_trip_mission_development import MeasuredSettlingRoundTripMission
from lewm.measured_settling_round_trip_controller_development import MeasuredSettlingRoundTripController


class SettledBoundaryRoundTripMission(MeasuredSettlingRoundTripMission):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_motion_quiet = False

    def advance(self, position_initial_body_m, *, frame, now_ns, previous_requested_command):
        if self.terminal is not None:
            return deepcopy(self.last)
        boundary_quiet = self.previous_motion_quiet
        if not boundary_quiet:
            self.quiet = 0
            self.was_within = False
        result = super().advance(position_initial_body_m, frame=frame, now_ns=now_ns,
            previous_requested_command=previous_requested_command)
        receipt = result.get('observed_settling')
        self.previous_motion_quiet = bool(receipt and result['failure'] is None
            and receipt['measured_motion_quiet'])
        if receipt is not None:
            receipt |= dict(previous_boundary_measured_quiet=boundary_quiet,
                both_boundaries_measured_quiet=boundary_quiet and self.previous_motion_quiet,
                first_quiet_observation_starts_dwell=True)
        self.last = result
        return deepcopy(self.last)


class SettledBoundaryRoundTripController(MeasuredSettlingRoundTripController):
    def __init__(self, model, geometry, *, public_mission, navigation_ticks, **kwargs):
        super().__init__(model, geometry, public_mission=public_mission, navigation_ticks=navigation_ticks, **kwargs)
        self.mission = SettledBoundaryRoundTripMission(public_mission, navigation_ticks=navigation_ticks)

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='settled_boundary_round_trip_controller_v1',
            measured_quiet_boundary_required_before_dwell=True)
