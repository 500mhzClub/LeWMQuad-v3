"""Prospective 8,000-step mission; native and memory integration is separate.

Only constructor admission changes. All mission transitions, measured quiet
boundaries, failure latching and global-budget accounting are inherited.
"""
from lewm.observed_round_trip_mission_development import ObservedRoundTripMission
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMission
from lewm.eligible_floor_registration_development import bind
from lewm.novel_maze_round_trip_contract_development import WARMUP_TICKS, DRAIN_TICKS

NAVIGATION_TICKS = 8000
MAX_COMMAND_TICKS = WARMUP_TICKS+NAVIGATION_TICKS+DRAIN_TICKS
MAX_OBSERVATIONS = MAX_COMMAND_TICKS+1


class ExtendedReturnBudgetObservedMission(ObservedRoundTripMission):
    __init__ = bind(ObservedRoundTripMission.__init__, MAX_NAVIGATION_TICKS=NAVIGATION_TICKS)


class ExtendedReturnBudgetMeasuredMission(
        MeasuredFloorTransportMission, ExtendedReturnBudgetObservedMission):
    """Preserve cooperative measured-settling initialization and advance rules."""
