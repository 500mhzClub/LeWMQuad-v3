from lewm.stop_conditioned_settling_development import (
    StopConditionedSettlingMission, StopConditionedSettlingController)


MISSION = dict(goal_initial_body_xy_m=[.2, 0.],
    return_initial_body_xy_m=[0., 0.], require_return_after_goal=True)


def step(m, i, command=(0., 0., 0.), position=(.2, 0., 0.)):
    return m.advance(position, frame=i, now_ns=1_500_000_000+i*100_000_000,
        previous_requested_command=command)


def test_visually_quiet_movement_cannot_start_zero_command_dwell():
    m = StopConditionedSettlingMission(MISSION, navigation_ticks=8000)
    for i in range(4):
        step(m, i, command=(.16, 0., .45))
    first_stop = step(m, 4)
    assert first_stop['quiet_intervals'] == 0
    assert first_stop['observed_settling']['measured_motion_quiet']
    assert not first_stop['observed_settling']['previous_boundary_quiet_under_zero_request']
    for i in range(5, 14):
        assert not step(m, i)['arrivals']
    arrival = step(m, 14)
    assert arrival['arrivals'][0]['quiet_intervals'] == 10
    assert arrival['phase'] == 'RETURN'
    assert not arrival['verified_round_trip']
    assert not arrival['observed_settling']['continuous_speed_bound']


def test_command_during_dwell_requires_new_stopped_boundary():
    m = StopConditionedSettlingMission(MISSION, navigation_ticks=8000)
    for i in range(8):
        step(m, i)
    assert step(m, 8, command=(0., 0., .45))['quiet_intervals'] == 0
    assert step(m, 9)['quiet_intervals'] == 0
    assert step(m, 10)['quiet_intervals'] == 1
    assert step(m, 11, position=(.2, 0., .01))['quiet_intervals'] == 0
    assert step(m, 12, position=(.2, 0., .01))['quiet_intervals'] == 0


def test_budget_and_invalid_observation_remain_terminal():
    m = StopConditionedSettlingMission(MISSION, navigation_ticks=4)
    for i in range(8):
        result = step(m, i)
    assert result['terminal'] == 'MISSION_TICK_BUDGET_EXHAUSTED'
    assert step(m, 8) == result
    m = StopConditionedSettlingMission(MISSION, navigation_ticks=8000)
    step(m, 0)
    failed = step(m, 2)
    assert failed['terminal'] == 'SENSOR_OR_MISSION_FAILURE'
    assert not m.previous_stopped_boundary
    assert step(m, 3) == failed


def test_controller_installs_mission_with_shared_memory_and_residuals():
    c = StopConditionedSettlingController(None, None, public_mission=MISSION,
        navigation_ticks=8000, condition='direct', variant='no_rgb', persistent=True)
    assert isinstance(c.mission, StopConditionedSettlingMission)
    assert c.mission.navigation_ticks == 8000
    assert c.memory is c.mapper.surface
    assert c.residual is c.selector.residual
    assert c.tick == -1 and not c.history
