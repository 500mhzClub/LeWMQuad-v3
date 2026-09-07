import numpy as np
import pytest

from lewm.causal_sensor_state import (CausalSensorBuffer, SensorContractError, SensorSchema,
                                 reorder_named_channels, build_decision_packet)

EPISODE = (0, 1, 0)


def buffer():
    result = CausalSensorBuffer((
        SensorSchema('gyro', ('wx', 'wy', 'wz'), ('rad/s',) * 3, 3, 100, 'calibration-test'),
        SensorSchema('command', ('vx', 'yaw'), ('m/s', 'rad/s'), 2, 100, 'command-v1', role='control'),
    ), capacity_per_sensor=4)
    result.begin_episode(EPISODE)
    return result


def append(b, measured=10, available=10, values=(1, 2, 3), valid=(True,) * 3, **overrides):
    arguments = dict(measured_ns=measured, available_ns=available, identity=EPISODE,
                     calibration_id='calibration-test')
    arguments.update(overrides)
    b.append('gyro', values, valid, **arguments)


def snapshot(b, image=30, decision=30):
    return b.snapshot(image_ns=image, decision_ns=decision, identity=EPISODE)


def test_history_is_ordered_left_padded_and_roles_are_separate():
    b = buffer()
    append(b, 10, 11)
    append(b, 20, 21, values=(4, 5, 6))
    b.append('command', [0.2, 0.4], [True, True], measured_ns=20, available_ns=20,
             identity=EPISODE, calibration_id='command-v1')
    result = snapshot(b)
    row = result['sensed']['gyro']
    assert row['measured_ns'].tolist() == [-1, 10, 20]
    assert row['values'].tolist() == [[0, 0, 0], [1, 2, 3], [4, 5, 6]]
    assert row['valid'].tolist() == [[False] * 3, [True] * 3, [True] * 3]
    assert set(result['sensed']) == {'gyro'} and set(result['control']) == {'command'}


def test_future_measurement_and_late_arrival_cannot_enter_earlier_decision():
    b = buffer()
    append(b, 10, 10)
    append(b, 20, 40)
    append(b, 50, 50)
    assert snapshot(b, 30, 30)['sensed']['gyro']['measured_ns'].tolist() == [-1, -1, 10]
    assert snapshot(b, 30, 45)['sensed']['gyro']['measured_ns'].tolist() == [-1, 10, 20]


def test_staleness_uses_decision_time_and_missing_is_not_confident_zero():
    b = buffer()
    append(b)
    row = snapshot(b, 10, 111)['sensed']['gyro']
    assert not row['valid'].any() and np.all(row['values'] == 0)
    assert np.all(row['measured_ns'] == -1)


def test_invalid_nan_is_inert_and_snapshot_does_not_alias_buffer():
    b = buffer()
    append(b, values=(np.nan, 0, 3), valid=(False, True, True))
    row = snapshot(b)['sensed']['gyro']
    assert row['values'][-1].tolist() == [0, 0, 3]
    assert row['valid'][-1].tolist() == [False, True, True]
    row['values'][:] = 999
    assert snapshot(b)['sensed']['gyro']['values'][-1].tolist() == [0, 0, 3]


def test_explicit_reset_clears_all_modalities_and_rejects_old_episode():
    b = buffer()
    append(b)
    b.begin_episode((0, 1, 1))
    with pytest.raises(SensorContractError):
        append(b)
    result = b.snapshot(image_ns=30, decision_ns=30, identity=(0, 1, 1))
    assert not result['sensed']['gyro']['valid'].any()


@pytest.mark.parametrize('changes', [
    {'measured_ns': -1}, {'measured_ns': 1.0}, {'measured_ns': True},
    {'available_ns': 9}, {'identity': (1, 1, 0)}, {'calibration_id': 'wrong'},
])
def test_bad_clock_identity_or_calibration_is_rejected(changes):
    with pytest.raises(SensorContractError):
        append(buffer(), **changes)


@pytest.mark.parametrize('values,valid', [
    ((1, 2), (True, True)), ((1, 2, 3), (1, 1, 1)),
    ((np.nan, 2, 3), (True, True, True)),
])
def test_invalid_channel_contract_is_rejected(values, valid):
    with pytest.raises(SensorContractError):
        append(buffer(), values=values, valid=valid)


def test_duplicate_and_out_of_order_samples_are_rejected():
    b = buffer()
    append(b, 20, 30)
    for mt, at in ((20, 30), (19, 30), (21, 29)):
        with pytest.raises(SensorContractError):
            append(b, mt, at)


def test_bounded_capacity_and_no_future_decision():
    b = buffer()
    for time in range(10, 70, 10):
        append(b, time, time)
    assert snapshot(b, 60, 60)['sensed']['gyro']['measured_ns'].tolist() == [40, 50, 60]
    with pytest.raises(SensorContractError):
        snapshot(b, 60, 59)


def test_explicit_online_anchor_uses_newer_sensors_but_not_late_arrivals():
    b = buffer()
    append(b, 20, 20)
    append(b, 40, 40)
    append(b, 45, 60)
    online = b.snapshot(image_ns=30, decision_ns=50, identity=EPISODE, sensor_anchor='decision')
    offline = b.snapshot(image_ns=30, decision_ns=50, identity=EPISODE)
    assert online['sensor_anchor_ns'] == 50
    assert online['sensed']['gyro']['measured_ns'].tolist() == [-1, 20, 40]
    assert offline['sensor_anchor_ns'] == 30
    assert offline['sensed']['gyro']['measured_ns'].tolist() == [-1, -1, 20]
    with pytest.raises(SensorContractError):
        b.snapshot(image_ns=30, decision_ns=50, identity=EPISODE, sensor_anchor='future')


def test_timestamp_cannot_overflow_packet_int64():
    with pytest.raises(SensorContractError):
        append(buffer(), measured=2**63, available=2**63)


def test_unitree_to_model_joint_order_uses_names_not_assumed_indices():
    source = tuple(f'{leg}_{joint}_joint' for leg in ('FR', 'FL', 'RR', 'RL')
                   for joint in ('hip', 'thigh', 'calf'))
    target = tuple(f'{leg}_{joint}_joint' for joint in ('hip', 'thigh', 'calf')
                   for leg in ('FL', 'FR', 'RL', 'RR'))
    result = reorder_named_channels(np.arange(12), [True] * 12, source, target)
    assert result['values'].tolist() == [3, 0, 9, 6, 4, 1, 10, 7, 5, 2, 11, 8]
    assert result['valid'].all() and not result['missing_channels']


def test_missing_and_invalid_channels_remain_distinct_from_valid_zero():
    result = reorder_named_channels([0, np.nan, 999], [True, False, True],
                                    ('q1', 'q2', 'unrequested'), ('q1', 'q2', 'q3'))
    assert result['values'].tolist() == [0, 0, 0]
    assert result['valid'].tolist() == [True, False, False]
    assert result['missing_channels'] == ('q3',)


@pytest.mark.parametrize('source,target', [(('q', 'q'), ('q',)), (('q',), ('q', 'q'))])
def test_ambiguous_channel_names_fail_closed(source, target):
    with pytest.raises(SensorContractError):
        reorder_named_channels([0] * len(source), [True] * len(source), source, target)


def packet(b, **overrides):
    args = dict(image_ns=20, image_available_ns=30, decision_ns=40, identity=EPISODE,
                camera_calibration_id='camera-test', expected_calibration_id='camera-test',
                expected_rgb_shape=(3, 4, 3), max_image_age_ns=30)
    args.update(overrides)
    return build_decision_packet(b, np.zeros((3, 4, 3), dtype=np.uint8), **args)


def test_online_packet_keeps_image_age_and_newer_received_sensor_history():
    b = buffer()
    append(b, 35, 36)
    result = packet(b)
    assert result['image']['available_ns'] == 30
    assert result['sensor_state']['sensor_anchor_ns'] == 40
    assert result['sensor_state']['sensed']['gyro']['measured_ns'][-1] == 35
    assert result['image']['rgb'].flags.c_contiguous


@pytest.mark.parametrize('overrides', [
    {'image_available_ns': 41}, {'image_available_ns': 19}, {'decision_ns': 60},
    {'camera_calibration_id': 'wrong'}, {'identity': (0, 1, 1)},
    {'expected_rgb_shape': (4, 4, 3)}, {'expected_rgb_shape': (3, 4, 4)},
    {'max_image_age_ns': -1},
])
def test_online_packet_rejects_unavailable_stale_or_misbound_images(overrides):
    with pytest.raises(SensorContractError):
        packet(buffer(), **overrides)
