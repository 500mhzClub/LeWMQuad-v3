"""Start arrival dwell after quiet visual motion under an actual zero request.

An interval following a movement request can look quiet while decelerating.
Require a preceding measured, zero-request interval before counting the dwell.
This remains an observation-based candidate, not an instantaneous speed bound.
"""
from copy import deepcopy
import numpy as np

from lewm.extended_return_budget_mission_development import ExtendedReturnBudgetMeasuredMission
from lewm.extended_return_budget_controller_development import ExtendedReturnBudgetChainedController


class StopConditionedSettlingMission(ExtendedReturnBudgetMeasuredMission):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.previous_stopped_boundary = False

    def advance(self, position_initial_body_m, *, frame, now_ns, previous_requested_command):
        if self.terminal is not None:
            return super().advance(position_initial_body_m, frame=frame, now_ns=now_ns,
                previous_requested_command=previous_requested_command)
        boundary = self.previous_stopped_boundary
        if not boundary:
            self.previous_motion_quiet = False
        result = super().advance(position_initial_body_m, frame=frame, now_ns=now_ns,
            previous_requested_command=previous_requested_command)
        receipt = result.get('observed_settling')
        zero = bool(result['failure'] is None
            and np.array_equal(previous_requested_command, np.zeros(3)))
        self.previous_stopped_boundary = bool(receipt and zero
            and receipt['measured_motion_quiet'] and result.get('phase_transition') is None)
        if receipt is not None:
            receipt.update(previous_boundary_quiet_under_zero_request=boundary,
                preceding_interval_zero_requested=zero,
                zero_request_boundary_required_before_dwell=True)
        self.last = deepcopy(result)
        return result


class StopConditionedSettlingController(ExtendedReturnBudgetChainedController):
    def __init__(self, model, geometry, *, public_mission, navigation_ticks, **kwargs):
        super().__init__(model, geometry, public_mission=public_mission,
            navigation_ticks=navigation_ticks, **kwargs)
        self.mission = StopConditionedSettlingMission(public_mission,
            navigation_ticks=navigation_ticks)

    def _result(self, *args, **kwargs):
        return super()._result(*args, **kwargs) | dict(
            controller='stop_conditioned_settling_controller_v1',
            zero_request_boundary_required_before_dwell=True)
