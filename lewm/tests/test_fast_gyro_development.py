import copy

import numpy as np
import pytest

from lewm.fast_gyro_development import FastGyroBuffer, FastRelativeOrientation, validate_fast_packet
from lewm.fast_gyro_scan_development import FastGyroScan
from lewm.simulated_fast_gyro_development import IdealFastGyro
from lewm.relative_gyro_turn_development import rotation_increment
from lewm.tests.test_relative_gyro_turn_development import initialized, append, packet


def buffers(rate=(0., 0., 0.)):
    slow = initialized(rate)
    fast = FastGyroBuffer((0, 0, 0))
    for step in range(750, 801):
        fast.append(rate, np.ones(3, bool), measured_ns=step * 2_000_000, available_ns=step * 2_000_000)
    return slow, fast


def advance(slow, fast, tick, rate):
    for step in range(800 + (tick - 1) * 50 + 1, 801 + tick * 50):
        fast.append(rate, np.ones(3, bool), measured_ns=step * 2_000_000, available_ns=step * 2_000_000)
        if step % 10 == 0:
            append(slow, step // 10, rate)
    now = 1_600_000_000 + tick * 100_000_000
    return packet(slow, 80 + tick * 5), fast.packet(now_ns=now), now


def test_all_fifty_intervals_integrated_once_and_model_packet_unchanged():
    rate = (.1, -.2, .3)
    slow, fast = buffers(rate)
    p = packet(slow, 80)
    before = copy.deepcopy(p)
    model = FastRelativeOrientation()
    model.begin(p, fast.packet(now_ns=1_600_000_000), now_ns=1_600_000_000)
    assert np.array_equal(p['image']['rgb'], before['image']['rgb'])
    assert set(p) == {'image', 'sensor_state'}
    for tick in range(1, 11):
        p, f, now = advance(slow, fast, tick, rate)
        result = model.step(p, f, now_ns=now)
    assert result['samples_integrated'] == 500
    assert np.allclose(result['rotation_initial_body_from_current_body'], rotation_increment(rate), rtol=0, atol=1e-12)
    result['rotation_initial_body_from_current_body'][0][0] = 999
    assert model.rotation[0, 0] != 999


@pytest.mark.parametrize('fault', ['gap', 'future', 'stale', 'invalid', 'identity', 'privilege',
    'calibration', 'units', 'rewrite', 'slow_disagreement', 'packet_gap'])
def test_bad_fast_stream_latches_without_partial_state(fault):
    slow, fast = buffers()
    model = FastRelativeOrientation()
    model.begin(packet(slow, 80), fast.packet(now_ns=1_600_000_000), now_ns=1_600_000_000)
    p, f, now = advance(slow, fast, 1, (0., 0., .2))
    if fault == 'gap': f['measured_ns'][-3] += 1
    elif fault == 'future': f['available_ns'][-1] += 1
    elif fault == 'stale': f['decision_ns'] -= 1
    elif fault == 'invalid': f['valid'][-3] = False
    elif fault == 'identity': f['identity'] = (0, 0, 1)
    elif fault == 'privilege': f['world_yaw'] = 0.
    elif fault == 'calibration': f['calibration_id'] = 'hardware-unknown'
    elif fault == 'units': f['units'] = ('deg/s',) * 3
    elif fault == 'rewrite':
        f['available_ns'][0] += 1  # Values still match slow sensing; overlap availability changed.
    elif fault == 'slow_disagreement': f['values'][-1, 2] += .1
    elif fault == 'packet_gap': p, f, now = advance(slow, fast, 2, (0., 0., .2))
    with pytest.raises(ValueError): model.step(p, f, now_ns=now)
    assert model.status == 'FAILED_SENSOR' and np.array_equal(model.rotation, np.eye(3))
    with pytest.raises(ValueError): model.step(p, f, now_ns=now)


def test_short_history_and_wrong_slow_packet_rejected():
    slow, _ = buffers()
    fast = FastGyroBuffer((0, 0, 0))
    fast.append([0, 0, 0], np.ones(3, bool), measured_ns=1_600_000_000, available_ns=1_600_000_000)
    with pytest.raises(ValueError):
        validate_fast_packet(fast.packet(now_ns=1_600_000_000), packet(slow, 80), now_ns=1_600_000_000)


def test_ideal_scan_completes_under_same_rule_and_budget():
    slow, fast = buffers()
    scan = FastGyroScan()
    result = scan.begin(packet(slow, 80), fast.packet(now_ns=1_600_000_000), now_ns=1_600_000_000)
    for tick in range(1, 301):
        p, f, now = advance(slow, fast, tick, result['requested_command'])
        result = scan.step(p, f, now_ns=now)
        if result['status'] == 'COMPLETE': break
    assert result['status'] == 'COMPLETE' and result['completed_target_views'] == 4
    assert result['requested_command'] == [0, 0, 0]
    assert not result['metric_clearance_qualified']


def test_simulator_adapter_rotates_current_measurement_and_rejects_gap():
    sensor = IdealFastGyro()
    value, valid = sensor.sample(measured_ns=2_000_000, quaternion_xyzw=[0, 0, 2**-.5, 2**-.5],
                                 angular_velocity_world=[1, 0, 0])
    assert value == pytest.approx([0, -1, 0], abs=1e-12) and valid.all()
    with pytest.raises(ValueError):
        sensor.sample(measured_ns=6_000_000, quaternion_xyzw=[0, 0, 0, 1], angular_velocity_world=[0, 0, 0])
