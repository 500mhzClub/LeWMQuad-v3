from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from lewm.causal_sensor_state import SensorContractError
from lewm.continuous_startup_handoff_development import ContinuousStartupHandoff
from lewm.tests.test_startup_observation_turn_development import components
from lewm.tests.test_correlated_moment_sensitivity_development import stream, room_depth


def frames(count):
    # Synthetic static room with acknowledged startup commands; not dynamics.
    for tick, policy, fast, _ in stream(count):
        control = policy['sensor_state']['control']['applied_command']
        control['values'][control['measured_ns'] == 1_700_000_000] = [0., 0., .35]
        depth = room_depth(policy)
        yield policy, depth, fast, policy['sensor_state']['decision_ns']


def model():
    geometry, kwargs = components()
    return ContinuousStartupHandoff(geometry, **kwargs)


def test_same_observer_memory_survive_exact_tail_then_continue_without_startup_restart():
    owner = model(); memory = owner._memory; observer = owner._relative
    rows = []
    for p, d, f, now in frames(9): rows.append(owner.observe(p, d, f, now_ns=now))
    assert rows[4]['status'] == 'MEASURED_ZERO_TAIL'
    assert [r['stopping_tail_observations'] for r in rows[5:8]] == [1, 2, 3]
    assert not any(r['handoff_ready'] for r in rows[:7])
    assert rows[7]['handoff_ready'] and rows[8]['handoff_ready']
    assert rows[8]['requested_command'] is None  # A downstream planner is required.
    assert memory is owner._memory is owner._startup.memory and observer is owner._relative
    assert owner._startup.last_ns == 2_000_000_000  # Frozen terminal never observed again.
    snapshot = owner.navigation_snapshot(now_ns=now)
    assert snapshot['consumed_observations'] == 9 and snapshot['gyro_intervals'] == 400
    assert snapshot['fusion']['initial_velocity_source'] == 'SUPPLIED_SETUP_PRIOR_NOT_SENSOR'
    assert snapshot['initial_epoch_ns'] == 1_600_000_000
    assert snapshot['latest_view_ns'] == now and snapshot['retained_view_ns'][0] == 1_600_000_000
    assert not snapshot['navigation_action_permitted'] and not snapshot['ground_support_permission']
    assert not snapshot['current_posture_is_future_sweep']
    snapshot['fusion']['position_initial_body_m'][0] = 999.
    assert owner.navigation_snapshot(now_ns=now)['fusion']['position_initial_body_m'][0] != 999.
    with pytest.raises(SensorContractError): owner.navigation_snapshot(now_ns=now-100_000_000)


def test_incomplete_tail_has_no_handoff_even_with_quiet_data():
    owner = model()
    for p, d, f, now in frames(7): row = owner.observe(p, d, f, now_ns=now)
    assert row['status'] == 'MEASURED_ZERO_TAIL' and row['stopping_tail_observations'] == 2
    with pytest.raises(SensorContractError): owner.navigation_snapshot(now_ns=now)


@pytest.mark.parametrize('fault', ['duplicate', 'gap', 'epoch', 'identity', 'privileged', 'depth_clock',
    'gyro_missing', 'acknowledgement', 'terminal_quiet_spike'])
def test_faults_latch_zero_and_prevent_state_export(fault):
    owner = model(); packets = list(frames(8)); index = 7 if fault == 'terminal_quiet_spike' else 1
    if fault == 'epoch': index = 0
    for p, d, f, now in packets[:index]: assert not owner.observe(p, d, f, now_ns=now)['terminal']
    p, d, f, now = deepcopy(packets[index])
    if fault == 'duplicate': p, d, f, now = packets[index-1]
    if fault in ('gap', 'epoch'): now += 100_000_000
    if fault == 'identity': p['sensor_state']['identity'] = (0, 0, 9)
    if fault == 'privileged': p['world_pose'] = [0., 0., 0.]
    if fault == 'depth_clock': d['measured_ns'] -= 1
    if fault == 'gyro_missing': f['valid'][25] = False
    if fault == 'acknowledgement': p['sensor_state']['control']['applied_command']['values'][-1] = [0., 0., 0.]
    if fault == 'terminal_quiet_spike': f['values'][25, 2] = .11
    row = owner.observe(p, d, f, now_ns=now)
    assert row['terminal'] and row['requested_command'] == [0., 0., 0.]
    assert not row['handoff_ready']
    with pytest.raises(SensorContractError): owner.navigation_snapshot(now_ns=now)
    with pytest.raises(SensorContractError): owner.relative_observation(now_ns=now)
    with pytest.raises(SensorContractError): owner.observe(*packets[index][:3], now_ns=packets[index][3])


def test_tail_cannot_silently_extend_starting_region():
    geometry, kwargs = components()
    kwargs['region_prior'] = replace(kwargs['region_prior'], valid_until_ns=2_650_000_000)
    owner = ContinuousStartupHandoff(geometry, **kwargs)
    for p, d, f, now in frames(8): row = owner.observe(p, d, f, now_ns=now)
    assert row['terminal'] and any('expiry' in x for x in row['failure_chain'])


def test_startup_failure_does_not_become_handoff_by_waiting(monkeypatch):
    owner = model()
    monkeypatch.setattr(owner._startup, 'observe', lambda *a, **k: dict(status='FAILED_OBSERVED_CONFLICT'))
    p, d, f, now = next(frames(1)); row = owner.observe(p, d, f, now_ns=now)
    assert row['terminal'] and not row['handoff_ready']
    assert owner._count == 0


def test_export_rejects_faulted_underlying_memory():
    owner = model()
    for p, d, f, now in frames(8): owner.observe(p, d, f, now_ns=now)
    owner._memory.failed = True
    with pytest.raises(SensorContractError): owner.navigation_snapshot(now_ns=now)
