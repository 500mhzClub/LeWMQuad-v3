"""Synthetic mission clocks/positions; no physical arrival qualification."""
from copy import deepcopy

import numpy as np
import pytest

from lewm import extended_return_budget_mission_development as new
from lewm.observed_round_trip_mission_development import MAX_NAVIGATION_TICKS
from lewm.measured_floor_transport_controller_development import MeasuredFloorTransportMission

PUBLIC = dict(goal_initial_body_xy_m=[3., 0.], return_initial_body_xy_m=[0., 0.],
    require_return_after_goal=True)


def advance(mission, frame, position, command=(0., 0., 0.)):
    return mission.advance(position, frame=frame, now_ns=1_500_000_000+frame*100_000_000,
        previous_requested_command=command)


@pytest.mark.parametrize('budget', [0, -1, 8001, True, 1.5, None])
def test_rejects_unbounded_or_noninteger_mission_allowance(budget):
    with pytest.raises(ValueError):
        new.ExtendedReturnBudgetMeasuredMission(PUBLIC, navigation_ticks=budget)


def test_constructor_keeps_original_limit_and_complete_settling_state():
    assert MAX_NAVIGATION_TICKS == 4000
    with pytest.raises(ValueError):
        MeasuredFloorTransportMission(PUBLIC, navigation_ticks=8000)
    original = MeasuredFloorTransportMission(PUBLIC, navigation_ticks=4000)
    revised = new.ExtendedReturnBudgetMeasuredMission(PUBLIC, navigation_ticks=8000)
    assert vars(original).keys() == vars(revised).keys()
    for name, value in vars(original).items():
        if name == 'navigation_ticks': assert revised.navigation_ticks == 8000
        elif isinstance(value, np.ndarray): np.testing.assert_array_equal(getattr(revised, name), value)
        else: assert getattr(revised, name) == value
    assert new.MAX_COMMAND_TICKS == 8013 and new.MAX_OBSERVATIONS == 8014


def test_retains_complete_original_prefix_then_stops_at_new_global_deadline():
    old = MeasuredFloorTransportMission(PUBLIC, navigation_ticks=4000)
    revised = new.ExtendedReturnBudgetMeasuredMission(PUBLIC, navigation_ticks=8000)
    prefix = None
    for frame in range(8004):
        # A synthetic outbound dwell, then a return position away from home.
        position = [3., 0., .3] if 3040 <= frame < 3060 else [1., 0., .3]
        current = advance(revised, frame, position)
        if frame <= 4003:
            prior = advance(old, frame, position)
            if frame < 4003:
                normalized = deepcopy(current)
                normalized['global_navigation_ticks'] = 4000
                assert normalized == prior
            else:
                assert prior['terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
                assert current['terminal'] is None and current['phase'] == 'RETURN'
                assert current['arrivals'] == prior['arrivals']
                prefix = deepcopy(current['arrivals'])
        if 4003 <= frame < 8003:
            assert current['terminal'] is None and current['phase'] == 'RETURN'
            assert current['arrivals'] == prefix
            assert not current['observed_map_reset_required']
    assert revised.frame == 8003 and current['global_navigation_ticks'] == 8000
    assert current['terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
    assert current['hold_required'] and not current['verified_round_trip']
    assert len(current['arrivals']) == 1
    assert advance(revised, 8004, [0., 0., .3]) == current


def test_late_return_still_requires_both_quiet_boundaries_and_full_dwell():
    mission = new.ExtendedReturnBudgetMeasuredMission(PUBLIC, navigation_ticks=8000)
    arrival = None
    for frame in range(7013):
        position = [3., 0., .3] if 20 <= frame < 40 else [1., 0., .3]
        if frame >= 7000: position = [0., 0., .3]
        result = advance(mission, frame, position)
        if frame in (7000, 7001):
            assert result['quiet_intervals'] == 0
            assert not result['arrival_confirmed_this_frame']
            assert result['terminal'] is None
        if result['terminal'] is not None:
            arrival = result
            break
    assert arrival is not None and arrival['frame'] == 7011
    assert arrival['terminal'] == 'OBSERVED_ROUND_TRIP_CANDIDATE'
    assert [row['phase'] for row in arrival['arrivals']] == ['OUTBOUND', 'RETURN']
    assert arrival['quiet_intervals'] == 10
    assert arrival['observed_settling']['both_boundaries_measured_quiet']
    assert not arrival['observed_settling']['continuous_speed_bound']
    assert not arrival['verified_round_trip']
    assert not any(row['native_verified'] for row in arrival['arrivals'])


def test_late_invalid_position_latches_instead_of_using_remaining_budget():
    mission = new.ExtendedReturnBudgetMeasuredMission(PUBLIC, navigation_ticks=8000)
    for frame in range(4100): advance(mission, frame, [1., 0., .3])
    result = advance(mission, 4100, [np.nan, 0., .3])
    assert result['terminal'] == 'SENSOR_OR_MISSION_FAILURE' and result['hold_required']
    assert not result['verified_round_trip']
    assert advance(mission, 4101, [0., 0., .3]) == result
